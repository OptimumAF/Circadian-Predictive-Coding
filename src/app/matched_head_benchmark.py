"""Fixed-feature benchmark routes for matched vision heads.

Each route takes a vision benchmark config, builds one frozen ResNet feature
bank, and returns descriptive head reports with state hashes. The two-head
gate and three-head circadian route share batches and initial tensors. This
module does not select winners or make an end-to-end throughput comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from json import dumps
from math import isfinite
from os import getpid
from time import perf_counter
from typing import Any, Callable


from src.app.circadian_checkpoint import (
    CircadianResumePosition,
    capture_circadian_checkpoint,
    restore_circadian_checkpoint,
)
from src.app.fixed_feature_checkpoint import (
    CudaAllocatorSegment,
    FixedFeatureCheckpointStore,
    FixedFeatureCircadianCheckpoint,
    FixedFeatureCircadianProgress,
    fixed_feature_config_digest,
    fixed_feature_data_digest,
    validate_fixed_feature_checkpoint,
)
from src.app.resnet50_benchmark import (
    ResNet50BenchmarkConfig,
    VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL,
    _build_benchmark_loaders,
    _build_circadian_head_config,
    _compute_rollback_delta,
    _require_finite_guard_scores,
    _resolve_device,
    _set_seed,
    _validate_benchmark_config,
)
from src.app.sleep_schedule import (
    SleepAttemptDecision,
    SleepRollbackCooldown,
    decide_sleep_attempt,
    resolve_rollback_cooldown_epochs,
)
from src.app.torch_sleep_decisions import (
    describe_failed_torch_sleep_decision,
    describe_guarded_torch_sleep_decision,
    describe_skipped_torch_sleep_decision,
    describe_unguarded_torch_sleep_decision,
)
from src.app.torch_sleep_transaction import (
    capture_torch_sleep_process_random,
    restore_torch_sleep_process_random,
)
from src.core.resnet50_variants import (
    BackpropMLPHead,
    CircadianPredictiveCodingHead,
    PredictiveCodingHead,
    _build_resnet50_backbone,
)
from src.core.sleep_clocks import SleepEpochProgress
from src.core.sleep_telemetry import SleepEventTelemetry
from src.shared.process_memory import ProcessRssSampler, ProcessRssSegment
from src.shared.torch_runtime import require_torch, sync_device

TWO_HEAD_FIXED_FEATURE_PROTOCOL = "vision_two_head_fixed_feature_v1"
THREE_HEAD_FIXED_FEATURE_PROTOCOL = "vision_three_head_fixed_feature_v1"
THREE_HEAD_FIXED_FEATURE_WALL_TIME_PROTOCOL = "vision_three_head_fixed_feature_wall_time_v1"
THREE_HEAD_FIXED_FEATURE_MEMORY_PROTOCOL = "vision_three_head_fixed_feature_memory_v1"
THREE_HEAD_FIXED_FEATURE_WALL_TIME_MEMORY_PROTOCOL = (
    "vision_three_head_fixed_feature_wall_time_memory_v2"
)
THREE_HEAD_FIXED_WIDTH_CAPACITY_MEMORY_PROTOCOL = "vision_three_head_fixed_width_capacity_memory_v1"
THREE_HEAD_FIXED_WIDTH_CAPACITY_CHECKPOINT_PROTOCOL = (
    "vision_three_head_fixed_width_capacity_checkpoint_v1"
)
THREE_HEAD_FIXED_FEATURE_CHECKPOINT_MEMORY_PROTOCOL = (
    "vision_three_head_fixed_feature_checkpoint_memory_v1"
)
THREE_HEAD_FIXED_FEATURE_WALL_TIME_CHECKPOINT_MEMORY_PROTOCOL = (
    "vision_three_head_fixed_feature_wall_time_checkpoint_memory_v1"
)
THREE_HEAD_FIXED_WIDTH_CAPACITY_CHECKPOINT_MEMORY_PROTOCOL = (
    "vision_three_head_fixed_width_capacity_checkpoint_memory_v1"
)
THREE_HEAD_FIXED_FEATURE_CUDA_CHECKPOINT_MEMORY_PROTOCOL = (
    "vision_three_head_fixed_feature_cuda_checkpoint_memory_v2"
)
THREE_HEAD_FIXED_FEATURE_WALL_TIME_CUDA_CHECKPOINT_MEMORY_PROTOCOL = (
    "vision_three_head_fixed_feature_wall_time_cuda_checkpoint_memory_v2"
)
THREE_HEAD_FIXED_WIDTH_CAPACITY_CUDA_CHECKPOINT_MEMORY_PROTOCOL = (
    "vision_three_head_fixed_width_capacity_cuda_checkpoint_memory_v2"
)
CHECKPOINT_MEMORY_SCOPE = "committed_head_training_segments_absolute_process_rss"
CHECKPOINT_CUDA_MEMORY_SCOPE = "committed_head_training_segments_absolute_rss_cuda_allocator"
FROZEN_SHARED_REPRESENTATION_TRACK = "frozen_shared_representation"
PROCESS_RSS_SAMPLE_INTERVAL_SECONDS = 0.005
_HEAD_TENSOR_NAMES = (
    "weight_feature_hidden",
    "bias_hidden",
    "weight_hidden_output",
    "bias_output",
)
FeatureBatches = tuple[tuple[Any, Any], ...]


@dataclass(frozen=True)
class FixedFeatureHeadReport:
    head_name: str
    head_type: str
    benchmark_track: str
    backbone_trainable: bool
    backbone_pretraining: str
    total_parameters: int
    epochs_ran: int
    validation_accuracy: float
    test_accuracy: float
    test_cross_entropy: float
    train_seconds: float
    seen_samples: int
    trainable_parameters: int
    wake_batches: int = 0
    latent_relaxation_steps: int = 0
    guard_examples_scored: int = 0
    sleep_attempts: int = 0
    sleep_cooldown_suppressions: int = 0
    sleep_retry_cooldown_epochs: int = 0
    replay_examples: int = 0
    stop_reason: str = "epoch_cap"
    deadline_overshoot_seconds: float = 0.0
    hidden_dim_start: int | None = None
    hidden_dim_end: int | None = None
    total_splits: int = 0
    total_prunes: int = 0
    total_rollbacks: int = 0
    process_rss_start_bytes: int | None = None
    process_rss_peak_observed_bytes: int | None = None
    process_rss_samples: int = 0
    process_rss_segments: tuple[ProcessRssSegment, ...] = ()
    cuda_allocated_start_bytes: int | None = None
    cuda_allocated_peak_bytes: int | None = None
    cuda_reserved_peak_bytes: int | None = None
    cuda_allocator_segments: tuple[CudaAllocatorSegment, ...] = ()
    sleep_events: tuple[SleepEventTelemetry, ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class TwoHeadFixedFeatureResult:
    protocol_id: str
    source_protocol_id: str
    benchmark_track: str
    backbone_trainable: bool
    backbone_pretraining: str
    backbone_parameters: int
    training_order: tuple[str, ...]
    backbone_weights: str
    backbone_hash: str
    initial_head_hashes: dict[str, str]
    trained_head_hashes: dict[str, str]
    split_hashes: dict[str, str]
    feature_hashes: dict[str, str]
    feature_bytes: int
    backprop: FixedFeatureHeadReport
    predictive_coding: FixedFeatureHeadReport
    wall_time_budget_seconds: float | None = field(default=None, kw_only=True)
    memory_telemetry_enabled: bool = field(default=False, kw_only=True)
    memory_observation_scope: str | None = field(default=None, kw_only=True)
    process_rss_sample_interval_seconds: float | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class FixedWidthCapacityControl:
    mode: str
    hidden_dim: int
    head_parameters: int
    initial_head_parameters: dict[str, int]
    final_head_parameters: dict[str, int]


@dataclass(frozen=True)
class ThreeHeadFixedFeatureResult(TwoHeadFixedFeatureResult):
    circadian: FixedFeatureHeadReport
    capacity_control: FixedWidthCapacityControl | None = None


@dataclass(frozen=True)
class _TrainedHead:
    head: Any
    name: str
    epochs_ran: int
    validation_accuracy: float
    train_seconds: float
    seen_samples: int
    parameter_count: int
    wake_batches: int
    latent_relaxation_steps: int
    guard_examples_scored: int
    sleep_attempts: int
    stop_reason: str
    sleep_cooldown_suppressions: int = 0
    sleep_retry_cooldown_epochs: int = 0
    hidden_dim_start: int | None = None
    hidden_dim_end: int | None = None
    total_splits: int = 0
    total_prunes: int = 0
    total_rollbacks: int = 0
    process_rss_start_bytes: int | None = None
    process_rss_peak_observed_bytes: int | None = None
    process_rss_samples: int = 0
    process_rss_segments: tuple[ProcessRssSegment, ...] = ()
    cuda_allocated_start_bytes: int | None = None
    cuda_allocated_peak_bytes: int | None = None
    cuda_reserved_peak_bytes: int | None = None
    cuda_allocator_segments: tuple[CudaAllocatorSegment, ...] = ()
    sleep_events: tuple[SleepEventTelemetry, ...] = field(default=(), compare=False)


def run_two_head_fixed_feature_benchmark(
    config: ResNet50BenchmarkConfig,
    *,
    model_order: tuple[str, ...] | None = None,
) -> TwoHeadFixedFeatureResult:
    """Train matched backprop/PC heads, then open final test for reporting."""
    return _run_fixed_feature_benchmark(
        config,
        include_circadian=False,
        model_order=model_order,
    )


def run_three_head_fixed_feature_benchmark(
    config: ResNet50BenchmarkConfig,
    *,
    model_order: tuple[str, ...] | None = None,
    measure_memory: bool = False,
    checkpoint_store: FixedFeatureCheckpointStore | None = None,
    resume_from_checkpoint: bool = False,
) -> ThreeHeadFixedFeatureResult:
    """Train all three matched heads on one frozen feature bank."""
    if config.circadian_head_hidden_dim != config.predictive_head_hidden_dim:
        raise ValueError("Matched heads require the same initial hidden width.")
    result = _run_fixed_feature_benchmark(
        config,
        include_circadian=True,
        model_order=model_order,
        measure_memory=measure_memory,
        checkpoint_store=checkpoint_store,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    assert isinstance(result, ThreeHeadFixedFeatureResult)
    return result


def run_three_head_fixed_width_capacity_benchmark(
    config: ResNet50BenchmarkConfig,
    *,
    checkpoint_store: FixedFeatureCheckpointStore | None = None,
    resume_from_checkpoint: bool = False,
    checkpoint_memory: bool = False,
) -> ThreeHeadFixedFeatureResult:
    """Compare equal-width heads; checkpoint memory requires explicit opt-in."""
    _validate_fixed_width_capacity_config(config)
    if checkpoint_memory and checkpoint_store is None:
        raise ValueError("checkpoint_memory requires a fixed-feature checkpoint store")
    result = _run_fixed_feature_benchmark(
        config,
        include_circadian=True,
        model_order=None,
        measure_memory=checkpoint_store is None or checkpoint_memory,
        fixed_width_capacity_control=True,
        checkpoint_store=checkpoint_store,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    assert isinstance(result, ThreeHeadFixedFeatureResult)
    return result


def run_three_head_fixed_feature_wall_time_benchmark(
    config: ResNet50BenchmarkConfig,
    *,
    wall_time_budget_seconds: float,
    model_order: tuple[str, ...] | None = None,
    measure_memory: bool = False,
    checkpoint_store: FixedFeatureCheckpointStore | None = None,
    resume_from_checkpoint: bool = False,
) -> ThreeHeadFixedFeatureResult:
    """Train matched heads under one per-head deadline after shared feature setup."""
    if not isfinite(wall_time_budget_seconds) or wall_time_budget_seconds <= 0.0:
        raise ValueError("wall_time_budget_seconds must be positive finite seconds.")
    if config.target_accuracy is not None:
        raise ValueError("Wall-time comparison requires target_accuracy=None.")
    if config.circadian_head_hidden_dim != config.predictive_head_hidden_dim:
        raise ValueError("Matched heads require the same initial hidden width.")
    result = _run_fixed_feature_benchmark(
        config,
        include_circadian=True,
        model_order=model_order,
        time_budget_seconds=wall_time_budget_seconds,
        measure_memory=measure_memory,
        checkpoint_store=checkpoint_store,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    assert isinstance(result, ThreeHeadFixedFeatureResult)
    return result


def _run_fixed_feature_benchmark(
    config: ResNet50BenchmarkConfig,
    *,
    include_circadian: bool,
    model_order: tuple[str, ...] | None,
    time_budget_seconds: float | None = None,
    measure_memory: bool = False,
    fixed_width_capacity_control: bool = False,
    checkpoint_store: FixedFeatureCheckpointStore | None = None,
    resume_from_checkpoint: bool = False,
) -> TwoHeadFixedFeatureResult:
    if resume_from_checkpoint and checkpoint_store is None:
        raise ValueError("resume_from_checkpoint requires a fixed-feature checkpoint store")
    if checkpoint_store is not None and not include_circadian:
        raise ValueError("checkpoint store currently requires the three-head route")
    training_order = _resolve_training_order(model_order, include_circadian)
    _validate_benchmark_config(config)
    if config.protocol_id != VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL:
        raise ValueError("Fixed-feature route requires the guard-separated vision protocol.")
    if not config.backprop_freeze_backbone:
        raise ValueError("Fixed-feature route requires a frozen backprop backbone setting.")

    torch = require_torch()
    _set_seed(torch, config.seed)
    device = _resolve_device(torch, config.device)
    loaders = _build_benchmark_loaders(config)
    if "guard" not in loaders.split_hashes or loaders.guard_loader is loaders.validation_loader:
        raise ValueError("Fixed-feature route requires a distinct guard loader.")
    backbone, feature_dim = _build_resnet50_backbone(
        device=device,
        freeze_backbone=True,
        backbone_weights=config.backbone_weights,
    )
    backbone.eval()
    backbone_hash = _hash_named_tensors(tuple(backbone.state_dict().items()))
    backbone_parameters = int(sum(parameter.numel() for parameter in backbone.parameters()))

    # Cache one fixed view per role before training. Fork the CPU RNG so
    # augmentation and loader iteration cannot depend on backbone draws or
    # head execution order.
    train_batches = _materialize_role_features(
        torch,
        backbone,
        loaders.train_loader,
        device,
        config.seed + 101,
    )
    guard_batches = _materialize_role_features(
        torch,
        backbone,
        loaders.guard_loader,
        device,
        config.seed + 102,
    )
    validation_batches = _materialize_role_features(
        torch,
        backbone,
        loaders.validation_loader,
        device,
        config.seed + 103,
    )
    head_seed = config.seed + 11
    backprop_head = BackpropMLPHead(
        feature_dim,
        config.predictive_head_hidden_dim,
        loaders.num_classes,
        device,
        head_seed,
    )
    predictive_head = PredictiveCodingHead(
        feature_dim,
        config.predictive_head_hidden_dim,
        loaders.num_classes,
        device,
        head_seed,
    )
    circadian_head = None
    if include_circadian:
        circadian_head = CircadianPredictiveCodingHead(
            feature_dim=feature_dim,
            hidden_dim=config.circadian_head_hidden_dim,
            num_classes=loaders.num_classes,
            device=device,
            seed=head_seed,
            config=_build_circadian_head_config(config),
            min_hidden_dim=config.circadian_min_hidden_dim,
            max_hidden_dim=config.circadian_max_hidden_dim,
        )
    initial_head_hashes = {
        "backprop_mlp": _hash_head(backprop_head),
        "predictive_coding": _hash_head(predictive_head),
    }
    if circadian_head is not None:
        initial_head_hashes["circadian_predictive_coding"] = _hash_head(circadian_head)
    if len(set(initial_head_hashes.values())) != 1:
        raise AssertionError("Matched heads must start from identical parameter tensors.")
    initial_head_parameters = {
        "backprop_mlp": backprop_head.parameter_count(),
        "predictive_coding": predictive_head.parameter_count(),
    }
    if circadian_head is not None:
        initial_head_parameters["circadian_predictive_coding"] = circadian_head.parameter_count()

    trainers: dict[str, Callable[[], _TrainedHead]] = {
        "backprop_mlp": lambda: _train_backprop_head(
            torch,
            device,
            backprop_head,
            train_batches,
            guard_batches,
            validation_batches,
            config,
            time_budget_seconds,
        ),
        "predictive_coding": lambda: _train_predictive_head(
            torch,
            device,
            predictive_head,
            train_batches,
            guard_batches,
            validation_batches,
            config,
            time_budget_seconds,
        ),
    }
    checkpoint_memory_trainer: (
        Callable[[ProcessRssSampler, tuple[int, int] | None], _TrainedHead] | None
    ) = None
    if circadian_head is not None:

        def train_circadian(
            memory_sampler: ProcessRssSampler | None = None,
            cuda_segment_start: tuple[int, int] | None = None,
        ) -> _TrainedHead:
            assert circadian_head is not None
            if checkpoint_store is not None:
                return _train_circadian_head(
                    torch,
                    device,
                    circadian_head,
                    train_batches,
                    guard_batches,
                    validation_batches,
                    config,
                    time_budget_seconds,
                    checkpoint_store=checkpoint_store,
                    resume_from_checkpoint=resume_from_checkpoint,
                    split_hashes=dict(loaders.split_hashes),
                    checkpoint_protocol_id=(
                        _resolve_checkpoint_memory_protocol(
                            time_budget_seconds,
                            fixed_width_capacity_control,
                            cuda=str(device).startswith("cuda"),
                        )
                        if measure_memory
                        else (
                            THREE_HEAD_FIXED_WIDTH_CAPACITY_CHECKPOINT_PROTOCOL
                            if fixed_width_capacity_control
                            else None
                        )
                    ),
                    memory_sampler=memory_sampler,
                    cuda_segment_start=cuda_segment_start,
                )
            return _train_circadian_head(
                torch,
                device,
                circadian_head,
                train_batches,
                guard_batches,
                validation_batches,
                config,
                time_budget_seconds,
            )

        trainers["circadian_predictive_coding"] = train_circadian
        checkpoint_memory_trainer = train_circadian
    trained: dict[str, _TrainedHead] = {}
    for name in training_order:
        if (
            measure_memory
            and checkpoint_store is not None
            and name == "circadian_predictive_coding"
        ):
            assert checkpoint_memory_trainer is not None
            trained[name] = _train_with_checkpoint_memory_telemetry(
                torch, device, checkpoint_memory_trainer
            )
        elif measure_memory:
            trained[name] = _train_with_memory_telemetry(
                torch, device, trainers[name], report_segment=checkpoint_store is not None
            )
        else:
            trained[name] = trainers[name]()
    if time_budget_seconds is not None and any(
        outcome.stop_reason != "deadline" for outcome in trained.values()
    ):
        raise ValueError("Wall-time deadline was not reached by every head; increase epochs.")
    backprop = trained["backprop_mlp"]
    predictive = trained["predictive_coding"]
    circadian = trained.get("circadian_predictive_coding")
    trained_head_hashes = {
        "backprop_mlp": _hash_trained_head(backprop_head),
        "predictive_coding": _hash_trained_head(predictive_head),
    }
    if circadian_head is not None:
        trained_head_hashes["circadian_predictive_coding"] = _hash_trained_head(circadian_head)
    capacity_control = (
        _verify_fixed_width_capacity(config, initial_head_parameters, trained)
        if fixed_width_capacity_control
        else None
    )
    # The final loader is first accessed after all included heads have stopped training.
    test_batches = _materialize_role_features(
        torch,
        backbone,
        loaders.test_loader,
        device,
        config.seed + 104,
    )
    backprop_report = _finalize_head(
        torch,
        device,
        backprop,
        test_batches,
        backbone_parameters,
        config.backbone_weights,
        time_budget_seconds,
    )
    predictive_report = _finalize_head(
        torch,
        device,
        predictive,
        test_batches,
        backbone_parameters,
        config.backbone_weights,
        time_budget_seconds,
    )
    circadian_report = (
        _finalize_head(
            torch,
            device,
            circadian,
            test_batches,
            backbone_parameters,
            config.backbone_weights,
            time_budget_seconds,
        )
        if circadian is not None
        else None
    )
    role_batches = {
        "train": train_batches,
        "guard": guard_batches,
        "validation": validation_batches,
        "test": test_batches,
    }
    if checkpoint_store is not None and measure_memory:
        result_protocol_id = _resolve_checkpoint_memory_protocol(
            time_budget_seconds,
            fixed_width_capacity_control,
            cuda=str(device).startswith("cuda"),
        )
    elif fixed_width_capacity_control:
        result_protocol_id = (
            THREE_HEAD_FIXED_WIDTH_CAPACITY_MEMORY_PROTOCOL
            if measure_memory
            else THREE_HEAD_FIXED_WIDTH_CAPACITY_CHECKPOINT_PROTOCOL
        )
    else:
        result_protocol_id = _resolve_fixed_feature_protocol(
            include_circadian, time_budget_seconds, measure_memory
        )
    common = dict(
        protocol_id=result_protocol_id,
        source_protocol_id=config.protocol_id,
        benchmark_track=FROZEN_SHARED_REPRESENTATION_TRACK,
        backbone_trainable=False,
        backbone_pretraining=config.backbone_weights,
        backbone_parameters=backbone_parameters,
        training_order=training_order,
        backbone_weights=config.backbone_weights,
        backbone_hash=backbone_hash,
        initial_head_hashes=initial_head_hashes,
        trained_head_hashes=trained_head_hashes,
        split_hashes=dict(loaders.split_hashes),
        feature_hashes={role: _hash_batches(batches) for role, batches in role_batches.items()},
        feature_bytes=sum(
            tensor.numel() * tensor.element_size()
            for batches in role_batches.values()
            for pair in batches
            for tensor in pair
        ),
        wall_time_budget_seconds=time_budget_seconds,
        memory_telemetry_enabled=measure_memory,
        memory_observation_scope=(
            (
                CHECKPOINT_CUDA_MEMORY_SCOPE
                if str(device).startswith("cuda")
                else CHECKPOINT_MEMORY_SCOPE
            )
            if checkpoint_store is not None and measure_memory
            else None
        ),
        process_rss_sample_interval_seconds=(
            PROCESS_RSS_SAMPLE_INTERVAL_SECONDS if measure_memory else None
        ),
        backprop=backprop_report,
        predictive_coding=predictive_report,
    )
    if circadian_report is not None:
        return ThreeHeadFixedFeatureResult(
            **common,
            circadian=circadian_report,
            capacity_control=capacity_control,
        )
    return TwoHeadFixedFeatureResult(**common)


def _validate_fixed_width_capacity_config(config: ResNet50BenchmarkConfig) -> None:
    _validate_benchmark_config(config)
    if config.protocol_id != VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL:
        raise ValueError("Fixed-width capacity control requires the guard-separated protocol.")
    if not config.backprop_freeze_backbone:
        raise ValueError("Fixed-width capacity control requires a frozen backbone.")
    if config.target_accuracy is not None:
        raise ValueError("Fixed-width capacity control requires target_accuracy=None.")
    if not (
        config.predictive_head_hidden_dim
        == config.circadian_min_hidden_dim
        == config.circadian_head_hidden_dim
        == config.circadian_max_hidden_dim
    ):
        raise ValueError("Fixed-width capacity control requires one equal hidden width.")
    if not config.circadian_force_sleep or config.circadian_sleep_interval <= 0:
        raise ValueError("Fixed-width capacity control requires scheduled forced sleep.")
    if config.circadian_sleep_mode == "disabled":
        raise ValueError("Fixed-width capacity control requires enabled sleep.")
    if config.circadian_sleep_mode == "components" and not (
        config.circadian_sleep_enable_split and config.circadian_sleep_enable_prune
    ):
        raise ValueError("Fixed-width capacity control requires enabled split and prune.")
    if (
        config.epochs < config.circadian_sleep_interval
        or config.circadian_sleep_warmup_steps >= config.circadian_sleep_interval
    ):
        raise ValueError("Fixed-width capacity control requires sleep within the epoch cap.")
    if not config.circadian_enable_sleep_rollback:
        raise ValueError("Fixed-width capacity control requires guard-based sleep rollback.")
    if (
        config.circadian_max_split_per_sleep <= 0
        or config.circadian_max_prune_per_sleep <= 0
        or config.circadian_sleep_max_change_fraction <= 0.0
        or config.circadian_sleep_min_change_count <= 0
    ):
        raise ValueError("Fixed-width capacity control requires an executable sleep budget.")


def _verify_fixed_width_capacity(
    config: ResNet50BenchmarkConfig,
    initial: dict[str, int],
    trained: dict[str, _TrainedHead],
) -> FixedWidthCapacityControl:
    if len(initial) != 3 or len(set(initial.values())) != 1:
        raise AssertionError("Fixed-width capacity requires equal initial head parameters.")
    final = {name: outcome.head.parameter_count() for name, outcome in trained.items()}
    reported = {name: outcome.parameter_count for name, outcome in trained.items()}
    if final != reported or final != initial:
        raise AssertionError("Fixed-width capacity changed head parameter counts.")
    circadian = trained["circadian_predictive_coding"]
    if (
        circadian.hidden_dim_start != config.circadian_head_hidden_dim
        or circadian.hidden_dim_end != config.circadian_head_hidden_dim
        or circadian.total_splits != 0
        or circadian.total_prunes != 0
        or circadian.sleep_attempts < 1
        or circadian.guard_examples_scored <= trained["predictive_coding"].guard_examples_scored
    ):
        raise AssertionError("Fixed-width capacity or guarded sleep invariant failed.")
    return FixedWidthCapacityControl(
        mode="fixed_width_parameter_matched",
        hidden_dim=config.circadian_head_hidden_dim,
        head_parameters=next(iter(initial.values())),
        initial_head_parameters=dict(initial),
        final_head_parameters=final,
    )


def _resolve_training_order(
    model_order: tuple[str, ...] | None,
    include_circadian: bool,
) -> tuple[str, ...]:
    expected: tuple[str, ...] = ("backprop_mlp", "predictive_coding")
    if include_circadian:
        expected += ("circadian_predictive_coding",)
    if model_order is None:
        return expected
    if len(model_order) != len(expected) or set(model_order) != set(expected):
        raise ValueError("model_order must be a permutation of the included heads.")
    return model_order


def _resolve_fixed_feature_protocol(
    include_circadian: bool,
    time_budget_seconds: float | None,
    measure_memory: bool,
) -> str:
    if time_budget_seconds is not None:
        return (
            THREE_HEAD_FIXED_FEATURE_WALL_TIME_MEMORY_PROTOCOL
            if measure_memory
            else THREE_HEAD_FIXED_FEATURE_WALL_TIME_PROTOCOL
        )
    if include_circadian:
        return (
            THREE_HEAD_FIXED_FEATURE_MEMORY_PROTOCOL
            if measure_memory
            else THREE_HEAD_FIXED_FEATURE_PROTOCOL
        )
    return TWO_HEAD_FIXED_FEATURE_PROTOCOL


def _resolve_checkpoint_memory_protocol(
    time_budget_seconds: float | None, fixed_width_capacity_control: bool, *, cuda: bool = False
) -> str:
    if fixed_width_capacity_control:
        return (
            THREE_HEAD_FIXED_WIDTH_CAPACITY_CUDA_CHECKPOINT_MEMORY_PROTOCOL
            if cuda
            else THREE_HEAD_FIXED_WIDTH_CAPACITY_CHECKPOINT_MEMORY_PROTOCOL
        )
    if time_budget_seconds is not None:
        return (
            THREE_HEAD_FIXED_FEATURE_WALL_TIME_CUDA_CHECKPOINT_MEMORY_PROTOCOL
            if cuda
            else THREE_HEAD_FIXED_FEATURE_WALL_TIME_CHECKPOINT_MEMORY_PROTOCOL
        )
    return (
        THREE_HEAD_FIXED_FEATURE_CUDA_CHECKPOINT_MEMORY_PROTOCOL
        if cuda
        else THREE_HEAD_FIXED_FEATURE_CHECKPOINT_MEMORY_PROTOCOL
    )


def _train_with_checkpoint_memory_telemetry(
    torch: Any,
    device: Any,
    trainer: Callable[[ProcessRssSampler, tuple[int, int] | None], _TrainedHead],
) -> _TrainedHead:
    with ProcessRssSampler(interval_seconds=PROCESS_RSS_SAMPLE_INTERVAL_SECONDS) as sampler:
        sampler.snapshot()  # Fail before training when this host has no RSS reader.
        cuda_start = _begin_cuda_allocator_segment(torch, device)
        outcome = trainer(sampler, cuda_start)
        sampler.sample()
        cuda_final = _snapshot_cuda_allocator_segment(torch, device, cuda_start)
    segments = (*outcome.process_rss_segments, sampler.snapshot())
    cuda_segments = (*outcome.cuda_allocator_segments, cuda_final) if cuda_final is not None else ()
    return replace(
        outcome,
        process_rss_start_bytes=None,
        process_rss_peak_observed_bytes=max(segment.peak_bytes for segment in segments),
        process_rss_samples=sum(segment.sample_count for segment in segments),
        process_rss_segments=segments,
        cuda_allocated_start_bytes=None,
        cuda_allocated_peak_bytes=(
            max(segment.allocated_peak_bytes for segment in cuda_segments)
            if cuda_segments
            else None
        ),
        cuda_reserved_peak_bytes=(
            max(segment.reserved_peak_bytes for segment in cuda_segments) if cuda_segments else None
        ),
        cuda_allocator_segments=cuda_segments,
    )


def _train_with_memory_telemetry(
    torch: Any,
    device: Any,
    trainer: Callable[[], _TrainedHead],
    *,
    report_segment: bool = False,
) -> _TrainedHead:
    cuda_segment_start = (
        _begin_cuda_allocator_segment(torch, device)
        if report_segment and str(device).startswith("cuda")
        else None
    )
    cuda_start = (
        cuda_segment_start[0]
        if cuda_segment_start is not None
        else _reset_cuda_memory_peak(torch, device)
    )
    with ProcessRssSampler(interval_seconds=PROCESS_RSS_SAMPLE_INTERVAL_SECONDS) as sampler:
        if report_segment:
            sampler.snapshot()
        outcome = trainer()
        sync_device(torch, device)
        cuda_peak, cuda_reserved_peak = _read_cuda_memory_peak(torch, device)
        sampler.sample()
    segments = (sampler.snapshot(),) if report_segment else ()
    cuda_segment = (
        CudaAllocatorSegment(
            pid=getpid(),
            device=str(torch.device(device)),
            allocated_start_bytes=cuda_segment_start[0],
            reserved_start_bytes=cuda_segment_start[1],
            allocated_peak_bytes=cuda_peak,
            reserved_peak_bytes=cuda_reserved_peak,
        )
        if cuda_segment_start is not None
        and cuda_peak is not None
        and cuda_reserved_peak is not None
        else None
    )
    return replace(
        outcome,
        process_rss_start_bytes=sampler.start_bytes,
        process_rss_peak_observed_bytes=sampler.peak_bytes,
        process_rss_samples=sampler.sample_count,
        process_rss_segments=segments,
        cuda_allocated_start_bytes=cuda_start,
        cuda_allocated_peak_bytes=cuda_peak,
        cuda_reserved_peak_bytes=cuda_reserved_peak,
        cuda_allocator_segments=(cuda_segment,) if cuda_segment is not None else (),
    )


def _begin_cuda_allocator_segment(torch: Any, device: Any) -> tuple[int, int] | None:
    if not str(device).startswith("cuda"):
        return None
    sync_device(torch, device)
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)
    return allocated, reserved


def _snapshot_cuda_allocator_segment(
    torch: Any, device: Any, start: tuple[int, int] | None
) -> CudaAllocatorSegment | None:
    if start is None:
        return None
    sync_device(torch, device)
    return CudaAllocatorSegment(
        pid=getpid(),
        device=str(torch.device(device)),
        allocated_start_bytes=start[0],
        reserved_start_bytes=start[1],
        allocated_peak_bytes=int(torch.cuda.max_memory_allocated(device)),
        reserved_peak_bytes=int(torch.cuda.max_memory_reserved(device)),
    )


def _reset_cuda_memory_peak(torch: Any, device: Any) -> int | None:
    if not str(device).startswith("cuda"):
        return None
    sync_device(torch, device)
    start = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    return start


def _read_cuda_memory_peak(torch: Any, device: Any) -> tuple[int | None, int | None]:
    if not str(device).startswith("cuda"):
        return None, None
    sync_device(torch, device)
    return int(torch.cuda.max_memory_allocated(device)), int(torch.cuda.max_memory_reserved(device))


def _materialize_features(
    torch: Any,
    backbone: Any,
    loader: Any,
    device: Any,
) -> FeatureBatches:
    batches: list[tuple[Any, Any]] = []
    with torch.no_grad():
        for images, labels in loader:
            features = backbone(images.to(device)).detach().cpu().contiguous()
            batches.append((features, labels.detach().cpu().contiguous()))
    if not batches:
        raise ValueError("A fixed-feature data role has no batches.")
    return tuple(batches)


def _materialize_role_features(
    torch: Any,
    backbone: Any,
    loader: Any,
    device: Any,
    seed: int,
) -> FeatureBatches:
    # torch.manual_seed also changes CUDA generators; only the CPU stream is
    # needed for torchvision transforms and DataLoader worker base seeds.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return _materialize_features(torch, backbone, loader, device)


def _deadline_reached(
    torch: Any,
    device: Any,
    start: float,
    seconds: float | None,
    clock: Callable[[], float],
) -> bool:
    if seconds is None:
        return False
    # CUDA updates are asynchronous; include completed device work before deciding.
    sync_device(torch, device)
    return clock() - start >= seconds


def _train_backprop_head(
    torch: Any,
    device: Any,
    head: BackpropMLPHead,
    train: FeatureBatches,
    guard: FeatureBatches,
    validation: FeatureBatches,
    config: ResNet50BenchmarkConfig,
    time_budget_seconds: float | None = None,
    clock: Callable[[], float] | None = None,
) -> _TrainedHead:
    seen_samples = 0
    wake_batches = 0
    completed_epochs = 0
    stop_reason = "epoch_cap"
    now = clock or perf_counter
    if time_budget_seconds is not None:
        sync_device(torch, device)
    start = now()
    optimizer = torch.optim.SGD(
        head.trainable_parameters(),
        lr=config.backprop_learning_rate,
        momentum=config.backprop_momentum,
    )
    criterion = torch.nn.CrossEntropyLoss()
    if time_budget_seconds is None:
        # Keep the historical epoch-route timing scope unchanged.
        start = now()
    for epoch in range(1, config.epochs + 1):
        for cpu_features, cpu_labels in train:
            if _deadline_reached(torch, device, start, time_budget_seconds, now):
                stop_reason = "deadline"
                break
            features, labels = cpu_features.to(device), cpu_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head.forward_logits(features), labels)
            loss.backward()
            optimizer.step()
            seen_samples += int(labels.shape[0])
            wake_batches += 1
        if stop_reason == "deadline" or _deadline_reached(
            torch,
            device,
            start,
            time_budget_seconds,
            now,
        ):
            stop_reason = "deadline"
            break
        guard_accuracy, _ = _evaluate_head(
            torch,
            device,
            head.forward_logits,
            guard,
            config.evaluation_batches,
        )
        completed_epochs = epoch
        if config.target_accuracy is not None and guard_accuracy >= config.target_accuracy:
            stop_reason = "target_accuracy"
            break
        if _deadline_reached(torch, device, start, time_budget_seconds, now):
            stop_reason = "deadline"
            break
    sync_device(torch, device)
    elapsed = now() - start
    validation_accuracy, _ = _evaluate_head(
        torch,
        device,
        head.forward_logits,
        validation,
        None,
    )
    return _TrainedHead(
        head=head,
        name="BackpropMLPHead",
        epochs_ran=completed_epochs,
        validation_accuracy=validation_accuracy,
        train_seconds=elapsed,
        seen_samples=seen_samples,
        parameter_count=head.parameter_count(),
        wake_batches=wake_batches,
        latent_relaxation_steps=0,
        guard_examples_scored=completed_epochs
        * _count_evaluated_examples(
            guard,
            config.evaluation_batches,
        ),
        sleep_attempts=0,
        stop_reason=stop_reason,
    )


def _train_predictive_head(
    torch: Any,
    device: Any,
    head: PredictiveCodingHead,
    train: FeatureBatches,
    guard: FeatureBatches,
    validation: FeatureBatches,
    config: ResNet50BenchmarkConfig,
    time_budget_seconds: float | None = None,
    clock: Callable[[], float] | None = None,
) -> _TrainedHead:
    seen_samples = 0
    wake_batches = 0
    completed_epochs = 0
    stop_reason = "epoch_cap"
    now = clock or perf_counter
    if time_budget_seconds is not None:
        sync_device(torch, device)
    start = now()
    for epoch in range(1, config.epochs + 1):
        for cpu_features, cpu_labels in train:
            if _deadline_reached(torch, device, start, time_budget_seconds, now):
                stop_reason = "deadline"
                break
            features, labels = cpu_features.to(device), cpu_labels.to(device)
            head.train_step(
                features=features,
                targets=labels,
                learning_rate=config.predictive_learning_rate,
                inference_steps=config.predictive_inference_steps,
                inference_learning_rate=config.predictive_inference_learning_rate,
            )
            seen_samples += int(labels.shape[0])
            wake_batches += 1
        if stop_reason == "deadline" or _deadline_reached(
            torch,
            device,
            start,
            time_budget_seconds,
            now,
        ):
            stop_reason = "deadline"
            break
        guard_accuracy, _ = _evaluate_head(
            torch,
            device,
            head.predict_logits,
            guard,
            config.evaluation_batches,
        )
        completed_epochs = epoch
        if config.target_accuracy is not None and guard_accuracy >= config.target_accuracy:
            stop_reason = "target_accuracy"
            break
        if _deadline_reached(torch, device, start, time_budget_seconds, now):
            stop_reason = "deadline"
            break
    sync_device(torch, device)
    elapsed = now() - start
    validation_accuracy, _ = _evaluate_head(
        torch,
        device,
        head.predict_logits,
        validation,
        None,
    )
    return _TrainedHead(
        head=head,
        name="PredictiveCodingHead",
        epochs_ran=completed_epochs,
        validation_accuracy=validation_accuracy,
        train_seconds=elapsed,
        seen_samples=seen_samples,
        parameter_count=head.parameter_count(),
        wake_batches=wake_batches,
        latent_relaxation_steps=wake_batches * config.predictive_inference_steps,
        guard_examples_scored=completed_epochs
        * _count_evaluated_examples(
            guard,
            config.evaluation_batches,
        ),
        sleep_attempts=0,
        stop_reason=stop_reason,
    )


def _train_circadian_head(
    torch: Any,
    device: Any,
    head: CircadianPredictiveCodingHead,
    train: FeatureBatches,
    guard: FeatureBatches,
    validation: FeatureBatches,
    config: ResNet50BenchmarkConfig,
    time_budget_seconds: float | None = None,
    clock: Callable[[], float] | None = None,
    *,
    checkpoint_store: FixedFeatureCheckpointStore | None = None,
    resume_from_checkpoint: bool = False,
    split_hashes: dict[str, str] | None = None,
    checkpoint_protocol_id: str | None = None,
    memory_sampler: ProcessRssSampler | None = None,
    cuda_segment_start: tuple[int, int] | None = None,
) -> _TrainedHead:
    if resume_from_checkpoint and checkpoint_store is None:
        raise ValueError("resume_from_checkpoint requires a fixed-feature checkpoint store")
    if checkpoint_store is not None:
        requested_device = torch.device(device)
        head_device = head.weight_feature_hidden.device
        requested_index = requested_device.index
        if requested_device.type == "cuda" and requested_index is None:
            requested_index = torch.cuda.current_device()
        if requested_device.type != head_device.type or (
            requested_device.type == "cuda" and requested_index != head_device.index
        ):
            raise ValueError("checkpoint head/device mismatch")
        if memory_sampler is not None and requested_device.type == "cuda":
            if cuda_segment_start is None:
                raise ValueError("CUDA checkpoint memory requires an allocator start")
        elif cuda_segment_start is not None:
            raise ValueError("CUDA allocator start requires checkpoint memory on CUDA")
    retry = SleepRollbackCooldown(
        resolve_rollback_cooldown_epochs(
            config.circadian_sleep_mode, config.circadian_sleep_rollback_cooldown_epochs
        )
    )
    initial_width = head.hidden_dim
    initial_head_hash = _hash_head(head) if checkpoint_store is not None else ""
    progress = FixedFeatureCircadianProgress(initial_width=initial_width)
    sleep_events: list[SleepEventTelemetry] = []
    guard_role_hash = _hash_batches(guard)
    protocol_id = checkpoint_protocol_id or _resolve_fixed_feature_protocol(
        True, time_budget_seconds, False
    )
    previous_memory_segments: tuple[ProcessRssSegment, ...] = ()
    previous_cuda_segments: tuple[CudaAllocatorSegment, ...] = ()
    feature_hashes = (
        (
            ("train", _hash_batches(train)),
            ("guard", _hash_batches(guard)),
            ("validation", _hash_batches(validation)),
        )
        if checkpoint_store is not None
        else ()
    )
    split_hash_pairs = tuple(sorted((split_hashes or {}).items()))
    runner_digest = (
        fixed_feature_config_digest(
            config, protocol_id, wall_time_budget_seconds=time_budget_seconds
        )
        if checkpoint_store
        else ""
    )
    data_digest = (
        fixed_feature_data_digest(feature_hashes, split_hash_pairs) if checkpoint_store else ""
    )
    resume_position: CircadianResumePosition | None = None
    if resume_from_checkpoint:
        assert checkpoint_store is not None
        saved = checkpoint_store.load()
        progress = validate_fixed_feature_checkpoint(
            saved,
            protocol_id=protocol_id,
            runner_config_digest=runner_digest,
            initial_head_hash=initial_head_hash,
            feature_hashes=feature_hashes,
            split_hashes=split_hash_pairs,
            batch_sizes=tuple(int(labels.shape[0]) for _, labels in train),
            guard_batch_sizes=tuple(int(labels.shape[0]) for _, labels in guard),
            epochs=config.epochs,
            initial_width=initial_width,
            config=config,
            memory_sample_interval_seconds=(
                PROCESS_RSS_SAMPLE_INTERVAL_SECONDS if memory_sampler is not None else None
            ),
            cuda_memory_device=(
                str(head.weight_feature_hidden.device) if cuda_segment_start is not None else None
            ),
        )
        sleep_events = list(saved.sleep_events)
        if memory_sampler is not None:
            previous_memory_segments = saved.memory_segments
        if cuda_segment_start is not None:
            previous_cuda_segments = saved.cuda_allocator_segments
        resume_position = restore_circadian_checkpoint(
            head,
            saved.combined,
            retry=retry,
            protocol_id=protocol_id,
            config=head.config,
            data_digest=data_digest,
            expected_stage=saved.combined.position.stage,
        )
    stop_reason = "epoch_cap"
    now = clock or perf_counter
    if time_budget_seconds is not None:
        sync_device(torch, device)
    start = now()
    elapsed_before_segment = progress.elapsed_seconds
    remaining_budget_seconds = (
        None if time_budget_seconds is None else time_budget_seconds - elapsed_before_segment
    )
    checkpoint_pause_seconds = 0.0

    def active_now() -> float:
        return now() - checkpoint_pause_seconds if time_budget_seconds is not None else now()

    def save_checkpoint(position: CircadianResumePosition) -> None:
        nonlocal checkpoint_pause_seconds
        if checkpoint_store is None:
            return
        memory_segments = previous_memory_segments
        cuda_segments = previous_cuda_segments
        if memory_sampler is not None:
            memory_sampler.sample()
            memory_segments = (*memory_segments, memory_sampler.snapshot())
        if cuda_segment_start is not None:
            cuda_snapshot = _snapshot_cuda_allocator_segment(torch, device, cuda_segment_start)
            assert cuda_snapshot is not None
            cuda_segments = (*cuda_segments, cuda_snapshot)
        before_save = now()
        checkpoint_store.save(
            FixedFeatureCircadianCheckpoint(
                format_version=2,
                protocol_id=protocol_id,
                runner_config_digest=runner_digest,
                initial_head_hash=initial_head_hash,
                feature_hashes=feature_hashes,
                split_hashes=split_hash_pairs,
                progress=replace(
                    progress, elapsed_seconds=elapsed_before_segment + active_now() - start
                ),
                combined=capture_circadian_checkpoint(
                    head,
                    retry=retry,
                    position=position,
                    protocol_id=protocol_id,
                    config=head.config,
                    data_digest=data_digest,
                ),
                sleep_events=tuple(sleep_events),
                memory_segments=memory_segments,
                cuda_allocator_segments=cuda_segments,
            )
        )
        if time_budget_seconds is not None:
            checkpoint_pause_seconds += max(0.0, now() - before_save)

    first_epoch = (
        resume_position.completed_epoch + (1 if resume_position.stage == "wake" else 0)
        if resume_position is not None
        else 1
    )
    for epoch in range(first_epoch, config.epochs + 1):
        stage = resume_position.stage if resume_position is not None else None
        if stage not in {"before_sleep", "after_sleep"}:
            first_batch = (
                resume_position.next_batch_index
                if resume_position is not None and stage == "wake"
                else 0
            )
            for batch_index in range(first_batch, len(train)):
                if _deadline_reached(torch, device, start, remaining_budget_seconds, active_now):
                    stop_reason = "deadline"
                    break
                cpu_features, cpu_labels = train[batch_index]
                features, labels = cpu_features.to(device), cpu_labels.to(device)
                head.train_step(
                    features=features,
                    targets=labels,
                    learning_rate=config.circadian_learning_rate,
                    inference_steps=config.circadian_inference_steps,
                    inference_learning_rate=config.circadian_inference_learning_rate,
                )
                progress = replace(
                    progress,
                    seen_samples=progress.seen_samples + int(labels.shape[0]),
                    wake_batches=progress.wake_batches + 1,
                )
                save_checkpoint(
                    CircadianResumePosition(
                        epoch - 1, "wake", progress.wake_batches, batch_index + 1
                    )
                )
        if stop_reason == "deadline" or _deadline_reached(
            torch,
            device,
            start,
            remaining_budget_seconds,
            active_now,
        ):
            stop_reason = "deadline"
            break
        if stage not in {"before_sleep", "after_sleep"}:
            save_checkpoint(CircadianResumePosition(epoch, "before_sleep", progress.wake_batches))
        if stage != "after_sleep":
            adaptive_triggered = (
                config.circadian_use_adaptive_sleep_trigger and head.should_trigger_sleep()
            )
            sleep_decision = decide_sleep_attempt(
                sleep_mode=config.circadian_sleep_mode,
                completed_epochs=epoch,
                interval_epochs=config.circadian_sleep_interval,
                adaptive_due=adaptive_triggered,
                force_periodic=config.circadian_force_sleep,
            )
            attempted = retry.allow_due_attempt(
                sleep_decision, completed_epochs=epoch, wake_batches=progress.wake_batches
            )
            if attempted:
                progress = replace(progress, sleep_attempts=progress.sleep_attempts + 1)

                def record_failed_sleep(failed: SleepEventTelemetry) -> None:
                    # Why this: a failed attempt is durable while the restored
                    # before-sleep cursor remains explicitly retryable.
                    sleep_events.append(failed)
                    save_checkpoint(
                        CircadianResumePosition(epoch, "before_sleep", progress.wake_batches)
                    )

                event, rolled_back = _guarded_sleep_event(
                    torch,
                    device,
                    head,
                    guard,
                    config,
                    epoch,
                    sleep_decision.force_sleep,
                    decision=sleep_decision,
                    guard_role_hash=guard_role_hash,
                    on_error=record_failed_sleep,
                )
                assert event.telemetry is not None
                sleep_events.append(event.telemetry)
                rollback_examples = (
                    2
                    * _count_evaluated_examples(guard, config.circadian_sleep_rollback_eval_batches)
                    if config.circadian_enable_sleep_rollback
                    else 0
                )
                progress = replace(
                    progress,
                    rollback_guard_examples=progress.rollback_guard_examples + rollback_examples,
                    total_splits=progress.total_splits + len(event.split_indices),
                    total_prunes=progress.total_prunes + len(event.pruned_indices),
                    total_rollbacks=progress.total_rollbacks + int(rolled_back),
                )
                if rolled_back:
                    retry.record_rejection(
                        completed_epochs=epoch, wake_batches=progress.wake_batches
                    )
            else:
                sleep_events.append(
                    describe_skipped_torch_sleep_decision(
                        head,
                        sleep_decision,
                        completed_epoch=epoch,
                        cooldown_suppressed=sleep_decision.attempted,
                    )
                )
            save_checkpoint(CircadianResumePosition(epoch, "after_sleep", progress.wake_batches))
            if attempted and _deadline_reached(
                torch, device, start, remaining_budget_seconds, active_now
            ):
                stop_reason = "deadline"
                break
        guard_accuracy, _ = _evaluate_head(
            torch,
            device,
            head.predict_logits,
            guard,
            config.evaluation_batches,
        )
        progress = replace(progress, completed_epochs=epoch)
        if config.target_accuracy is not None and guard_accuracy >= config.target_accuracy:
            stop_reason = "target_accuracy"
            break
        if _deadline_reached(torch, device, start, remaining_budget_seconds, active_now):
            stop_reason = "deadline"
            break
        resume_position = None
    sync_device(torch, device)
    elapsed = elapsed_before_segment + active_now() - start
    validation_accuracy, _ = _evaluate_head(
        torch,
        device,
        head.predict_logits,
        validation,
        None,
    )
    return _TrainedHead(
        head=head,
        name="CircadianPredictiveCodingHead",
        epochs_ran=progress.completed_epochs,
        validation_accuracy=validation_accuracy,
        train_seconds=elapsed,
        seen_samples=progress.seen_samples,
        parameter_count=head.parameter_count(),
        wake_batches=progress.wake_batches,
        latent_relaxation_steps=progress.wake_batches * config.circadian_inference_steps,
        guard_examples_scored=(
            progress.completed_epochs * _count_evaluated_examples(guard, config.evaluation_batches)
            + progress.rollback_guard_examples
        ),
        sleep_attempts=progress.sleep_attempts,
        sleep_cooldown_suppressions=retry.snapshot_state().suppressed_due_attempts,
        sleep_retry_cooldown_epochs=retry.cooldown_epochs,
        stop_reason=stop_reason,
        hidden_dim_start=progress.initial_width,
        hidden_dim_end=head.hidden_dim,
        total_splits=progress.total_splits,
        total_prunes=progress.total_prunes,
        total_rollbacks=progress.total_rollbacks,
        process_rss_segments=previous_memory_segments,
        cuda_allocator_segments=previous_cuda_segments,
        sleep_events=tuple(sleep_events),
    )


def _guarded_sleep_event(
    torch: Any,
    device: Any,
    head: CircadianPredictiveCodingHead,
    guard: FeatureBatches,
    config: ResNet50BenchmarkConfig,
    epoch: int,
    force_sleep: bool,
    *,
    decision: SleepAttemptDecision | None = None,
    guard_role_hash: str | None = None,
    on_error: Callable[[SleepEventTelemetry], None] | None = None,
) -> tuple[Any, bool]:
    attempt_started = perf_counter()
    snapshot = head.snapshot_state()
    process_random = capture_torch_sleep_process_random(torch, device)
    guarded = config.circadian_enable_sleep_rollback
    guard_examples = _count_evaluated_examples(guard, config.circadian_sleep_rollback_eval_batches)
    pre_accuracy: float | None = None
    pre_cross_entropy: float | None = None
    post_accuracy: float | None = None
    post_cross_entropy: float | None = None
    core_result: Any = None
    examples_scored = 0
    stage = "inner_guard_pre" if guarded else "sleep_core"

    def record_scored_batch(count: int) -> None:
        nonlocal examples_scored
        examples_scored += count

    try:
        if guarded:
            pass_start = examples_scored
            accuracy, cross_entropy = _evaluate_head(
                torch,
                device,
                head.predict_logits,
                guard,
                config.circadian_sleep_rollback_eval_batches,
                on_examples_scored=record_scored_batch,
            )
            examples_scored = pass_start + guard_examples
            _require_finite_guard_scores(accuracy, cross_entropy)
            pre_accuracy, pre_cross_entropy = accuracy, cross_entropy
        stage = "sleep_core"
        core_result = head.sleep_event(
            force_sleep=force_sleep,
            epoch_progress=SleepEpochProgress(epoch, config.epochs),
        )
        if guarded:
            stage = "inner_guard_post"
            pass_start = examples_scored
            accuracy, cross_entropy = _evaluate_head(
                torch,
                device,
                head.predict_logits,
                guard,
                config.circadian_sleep_rollback_eval_batches,
                on_examples_scored=record_scored_batch,
            )
            examples_scored = pass_start + guard_examples
            _require_finite_guard_scores(accuracy, cross_entropy)
            post_accuracy, post_cross_entropy = accuracy, cross_entropy
            stage = "inner_guard_delta"
            assert pre_accuracy is not None and pre_cross_entropy is not None
            rollback_delta = _compute_rollback_delta(
                metric_name=config.circadian_sleep_rollback_metric,
                pre_accuracy=pre_accuracy,
                post_accuracy=post_accuracy,
                pre_cross_entropy=pre_cross_entropy,
                post_cross_entropy=post_cross_entropy,
            )
            if not isfinite(rollback_delta):
                raise FloatingPointError("nonfinite guard rollback delta")
    except Exception as error:
        head.restore_state(snapshot)
        restore_torch_sleep_process_random(torch, device, process_random)
        if decision is not None and on_error is not None:
            reason = (
                f"{stage}_nonfinite"
                if stage in {"inner_guard_pre", "inner_guard_post"}
                and isinstance(error, FloatingPointError)
                else f"{stage}_exception"
            )
            if reason == "sleep_core_exception" or guarded:
                on_error(
                    describe_failed_torch_sleep_decision(
                        head,
                        decision,
                        completed_epoch=epoch,
                        guard_role_hash=guard_role_hash if guarded else None,
                        metric_name=config.circadian_sleep_rollback_metric,
                        tolerance=config.circadian_sleep_rollback_tolerance,
                        pre_accuracy=pre_accuracy,
                        pre_cross_entropy=pre_cross_entropy,
                        post_accuracy=post_accuracy,
                        post_cross_entropy=post_cross_entropy,
                        examples_scored=examples_scored,
                        reason=reason,
                        attempt_seconds=perf_counter() - attempt_started,
                        result=core_result,
                    )
                )
        raise

    event = core_result
    if guarded and rollback_delta > config.circadian_sleep_rollback_tolerance:
        head.restore_state(snapshot)
        rejected = type(event)(
            old_hidden_dim=head.hidden_dim,
            new_hidden_dim=head.hidden_dim,
            split_indices=(),
            pruned_indices=(),
        )
        if decision is not None:
            assert guard_role_hash is not None
            assert pre_accuracy is not None and pre_cross_entropy is not None
            assert post_accuracy is not None and post_cross_entropy is not None
            rejected = replace(
                rejected,
                telemetry=describe_guarded_torch_sleep_decision(
                    decision,
                    event,
                    completed_epoch=epoch,
                    guard_role_hash=guard_role_hash,
                    pre_accuracy=pre_accuracy,
                    post_accuracy=post_accuracy,
                    pre_cross_entropy=pre_cross_entropy,
                    post_cross_entropy=post_cross_entropy,
                    metric_name=config.circadian_sleep_rollback_metric,
                    tolerance=config.circadian_sleep_rollback_tolerance,
                    guard_examples=_count_evaluated_examples(
                        guard, config.circadian_sleep_rollback_eval_batches
                    ),
                    accepted=False,
                    attempt_seconds=perf_counter() - attempt_started,
                ),
            )
        return rejected, True
    if decision is not None:
        if guarded:
            assert pre_accuracy is not None and pre_cross_entropy is not None
            assert post_accuracy is not None and post_cross_entropy is not None
            telemetry = describe_guarded_torch_sleep_decision(
                decision,
                event,
                completed_epoch=epoch,
                guard_role_hash=guard_role_hash or _hash_batches(guard),
                pre_accuracy=pre_accuracy,
                post_accuracy=post_accuracy,
                pre_cross_entropy=pre_cross_entropy,
                post_cross_entropy=post_cross_entropy,
                metric_name=config.circadian_sleep_rollback_metric,
                tolerance=config.circadian_sleep_rollback_tolerance,
                guard_examples=_count_evaluated_examples(
                    guard, config.circadian_sleep_rollback_eval_batches
                ),
                accepted=True,
                attempt_seconds=perf_counter() - attempt_started,
            )
        else:
            telemetry = describe_unguarded_torch_sleep_decision(
                decision,
                event,
                completed_epoch=epoch,
                attempt_seconds=perf_counter() - attempt_started,
            )
        event = replace(event, telemetry=telemetry)
    return event, False


def _evaluate_head(
    torch: Any,
    device: Any,
    forward_logits: Callable[[Any], Any],
    batches: FeatureBatches,
    max_batches: int | None,
    *,
    on_examples_scored: Callable[[int], None] | None = None,
) -> tuple[float, float]:
    correct = 0
    total = 0
    loss_total = 0.0
    with torch.no_grad():
        for index, (cpu_features, cpu_labels) in enumerate(batches):
            features, labels = cpu_features.to(device), cpu_labels.to(device)
            logits = forward_logits(features)
            loss_total += float(
                torch.nn.functional.cross_entropy(
                    logits,
                    labels,
                    reduction="sum",
                ).item()
            )
            correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
            total += int(labels.shape[0])
            if on_examples_scored is not None:
                on_examples_scored(int(labels.shape[0]))
            if max_batches is not None and max_batches > 0 and index + 1 >= max_batches:
                break
    return correct / total, loss_total / total


def _count_evaluated_examples(batches: FeatureBatches, max_batches: int | None) -> int:
    selected = batches if max_batches is None or max_batches <= 0 else batches[:max_batches]
    return sum(int(labels.shape[0]) for _, labels in selected)


def _finalize_head(
    torch: Any,
    device: Any,
    trained: _TrainedHead,
    test: FeatureBatches,
    backbone_parameters: int,
    backbone_pretraining: str,
    time_budget_seconds: float | None,
) -> FixedFeatureHeadReport:
    forward = (
        trained.head.forward_logits
        if trained.name == "BackpropMLPHead"
        else trained.head.predict_logits
    )
    test_accuracy, test_cross_entropy = _evaluate_head(torch, device, forward, test, None)
    head_types = {
        "BackpropMLPHead": "backprop_mlp",
        "PredictiveCodingHead": "predictive_coding",
        "CircadianPredictiveCodingHead": "circadian_predictive_coding",
    }
    return FixedFeatureHeadReport(
        head_name=trained.name,
        head_type=head_types[trained.name],
        benchmark_track=FROZEN_SHARED_REPRESENTATION_TRACK,
        backbone_trainable=False,
        backbone_pretraining=backbone_pretraining,
        total_parameters=backbone_parameters + trained.parameter_count,
        epochs_ran=trained.epochs_ran,
        validation_accuracy=trained.validation_accuracy,
        test_accuracy=test_accuracy,
        test_cross_entropy=test_cross_entropy,
        train_seconds=trained.train_seconds,
        seen_samples=trained.seen_samples,
        trainable_parameters=trained.parameter_count,
        wake_batches=trained.wake_batches,
        latent_relaxation_steps=trained.latent_relaxation_steps,
        guard_examples_scored=trained.guard_examples_scored,
        sleep_attempts=trained.sleep_attempts,
        sleep_cooldown_suppressions=trained.sleep_cooldown_suppressions,
        sleep_retry_cooldown_epochs=trained.sleep_retry_cooldown_epochs,
        replay_examples=0,
        stop_reason=trained.stop_reason,
        deadline_overshoot_seconds=(
            max(0.0, trained.train_seconds - time_budget_seconds)
            if time_budget_seconds is not None
            else 0.0
        ),
        hidden_dim_start=trained.hidden_dim_start,
        hidden_dim_end=trained.hidden_dim_end,
        total_splits=trained.total_splits,
        total_prunes=trained.total_prunes,
        total_rollbacks=trained.total_rollbacks,
        process_rss_start_bytes=trained.process_rss_start_bytes,
        process_rss_peak_observed_bytes=trained.process_rss_peak_observed_bytes,
        process_rss_samples=trained.process_rss_samples,
        process_rss_segments=trained.process_rss_segments,
        cuda_allocated_start_bytes=trained.cuda_allocated_start_bytes,
        cuda_allocated_peak_bytes=trained.cuda_allocated_peak_bytes,
        cuda_reserved_peak_bytes=trained.cuda_reserved_peak_bytes,
        cuda_allocator_segments=trained.cuda_allocator_segments,
        sleep_events=trained.sleep_events,
    )


def _hash_head(head: Any) -> str:
    return _hash_named_tensors(tuple((name, getattr(head, name)) for name in _HEAD_TENSOR_NAMES))


def _hash_trained_head(head: Any) -> str:
    if isinstance(head, BackpropMLPHead):
        return _hash_head(head)

    torch = require_torch()
    named_tensors: list[tuple[str, Any]] = []
    named_scalars: list[tuple[str, Any]] = []
    if isinstance(head, CircadianPredictiveCodingHead):
        for name, value in head.snapshot_state().items():
            target = named_tensors if torch.is_tensor(value) else named_scalars
            target.append((name, value))
        named_tensors.append(("split_generator", head._split_generator.get_state()))
    else:
        named_tensors.extend((name, getattr(head, name)) for name in _HEAD_TENSOR_NAMES)
        named_tensors.append(("traffic_sum", head._traffic_sum))
        named_scalars.append(("traffic_steps", head._traffic_steps))

    digest = sha256(_hash_named_tensors(tuple(named_tensors)).encode("utf-8"))
    for name, value in sorted(named_scalars):
        digest.update(name.encode("utf-8"))
        digest.update(dumps(value, sort_keys=True, allow_nan=False).encode("utf-8"))
    return digest.hexdigest()


def _hash_batches(batches: FeatureBatches) -> str:
    return _hash_named_tensors(
        tuple(
            (f"batch{index}.{role}", tensor)
            for index, pair in enumerate(batches)
            for role, tensor in zip(("features", "labels"), pair)
        )
    )


def _hash_named_tensors(tensors: tuple[tuple[str, Any], ...]) -> str:
    digest = sha256()
    for name, tensor in sorted(tensors):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(value.shape).encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()
