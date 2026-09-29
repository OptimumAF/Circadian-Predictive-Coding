"""Application workflow for ResNet-50 speed and accuracy benchmarking."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from json import dumps
from math import isfinite
import random
from time import perf_counter
from typing import Any, Callable

import numpy as np

from src.app.seeded_vision_loader import (
    SeededEpochLoaderState,
    SeededEpochTrainLoader as _SeededEpochTrainLoader,
)
from src.app.shared_vision_loader import (
    SharedEpochLoaderState,
    SharedEpochTrainLoader as _SharedEpochTrainLoader,
)
from src.app.vision_checkpoint import (
    VisionCheckpointStore,
    VisionCircadianCheckpointContext,
    VisionCircadianProgress,
    capture_vision_checkpoint,
    restore_completed_vision_outcome,
    restore_vision_process_rng,
    snapshot_completed_vision_outcome,
    validate_vision_checkpoint,
    vision_config_digest,
    vision_development_data_digest,
)
from src.app.sleep_schedule import (
    SleepRollbackCooldown,
    SleepRollbackCooldownState,
    decide_sleep_attempt,
    resolve_rollback_cooldown_epochs,
)
from src.core.resnet50_variants import (
    BackpropResNet50Classifier,
    CircadianHeadConfig,
    CircadianPredictiveCodingResNet50Classifier,
    PredictiveCodingResNet50Classifier,
    TORCH_PC_ENERGY_ID,
)
from src.core.sleep_clocks import SleepEpochProgress
from src.infra.vision_datasets import (
    SyntheticVisionDatasetConfig,
    TorchVisionDatasetConfig,
    build_synthetic_vision_dataloaders,
    build_torchvision_vision_dataloaders,
)
from src.shared.torch_runtime import require_torch, sync_device

VISION_VALIDATION_UNMATCHED_PROTOCOL = "vision_validation_unmatched_v1"
VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL = "vision_guard_separated_unmatched_v2"
VISION_SEEDED_UNMATCHED_PROTOCOL = "vision_guard_separated_seeded_unmatched_v3"
GUARD_SEPARATED_VISION_PROTOCOLS = {
    VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL,
    VISION_SEEDED_UNMATCHED_PROTOCOL,
}
VISION_UNMATCHED_REFERENCE_TRACK = "unmatched_reference"
VISION_END_TO_END_BACKPROP_TRACK = "end_to_end_backprop"


@dataclass(frozen=True)
class ResNet50BenchmarkConfig:
    """Configurable benchmark settings for all three ResNet-50 variants."""

    train_samples: int = 2000
    protocol_id: str = VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL
    validation_samples: int = 64
    guard_samples: int = 64
    test_samples: int = 500
    num_classes: int = 10
    image_size: int = 96
    batch_size: int = 32
    dataset_name: str = "synthetic"
    dataset_data_root: str = "data"
    dataset_download: bool = True
    dataset_train_subset_size: int = 0
    dataset_validation_subset_size: int = 1000
    dataset_guard_subset_size: int = 1000
    dataset_test_subset_size: int = 0
    dataset_num_workers: int = 0
    dataset_use_augmentation: bool = True
    dataset_difficulty: str = "medium"
    dataset_noise_std: float = 0.06
    epochs: int = 8
    seed: int = 7
    device: str = "auto"
    target_accuracy: float | None = 0.99
    evaluation_batches: int = 2
    inference_batches: int = 50
    warmup_batches: int = 10

    backprop_learning_rate: float = 0.02
    backprop_momentum: float = 0.9
    backprop_freeze_backbone: bool = False
    backbone_weights: str = "none"

    predictive_head_hidden_dim: int = 384
    predictive_learning_rate: float = 0.03
    predictive_inference_steps: int = 10
    predictive_inference_learning_rate: float = 0.15

    circadian_head_hidden_dim: int = 384
    circadian_learning_rate: float = 0.025
    circadian_inference_steps: int = 14
    circadian_inference_learning_rate: float = 0.12
    circadian_sleep_interval: int = 2
    circadian_force_sleep: bool = False
    circadian_use_adaptive_sleep_trigger: bool = True
    circadian_min_sleep_steps: int = 60
    circadian_sleep_energy_window: int = 48
    circadian_sleep_plateau_delta: float = 5e-5
    circadian_sleep_chemical_variance_threshold: float = 0.02
    circadian_use_adaptive_sleep_budget: bool = True
    circadian_adaptive_sleep_budget_min_scale: float = 0.25
    circadian_adaptive_sleep_budget_max_scale: float = 1.0
    circadian_adaptive_sleep_budget_plateau_weight: float = 0.6
    circadian_adaptive_sleep_budget_variance_weight: float = 0.4
    circadian_enable_sleep_rollback: bool = True
    circadian_sleep_rollback_tolerance: float = 0.002
    circadian_sleep_rollback_metric: str = "cross_entropy"
    circadian_sleep_rollback_eval_batches: int = 2
    circadian_sleep_rollback_cooldown_epochs: int | None = None
    circadian_min_hidden_dim: int = 384
    circadian_max_hidden_dim: int = 640
    circadian_chemical_decay: float = 0.995
    circadian_chemical_buildup_rate: float = 0.02
    circadian_use_saturating_chemical: bool = True
    circadian_chemical_max_value: float = 2.5
    circadian_chemical_saturation_gain: float = 1.0
    circadian_use_dual_chemical: bool = True
    circadian_dual_fast_mix: float = 0.70
    circadian_slow_chemical_decay: float = 0.999
    circadian_slow_buildup_scale: float = 0.25
    circadian_plasticity_sensitivity: float = 0.45
    circadian_use_adaptive_plasticity_sensitivity: bool = True
    circadian_plasticity_sensitivity_min: float = 0.25
    circadian_plasticity_sensitivity_max: float = 0.55
    circadian_plasticity_importance_mix: float = 0.50
    circadian_min_plasticity: float = 0.5
    circadian_use_reward_modulated_learning: bool = False
    circadian_reward_baseline_decay: float = 0.95
    circadian_reward_difficulty_exponent: float = 1.0
    circadian_reward_scale_min: float = 0.75
    circadian_reward_scale_max: float = 1.5
    circadian_use_adaptive_thresholds: bool = True
    circadian_adaptive_split_percentile: float = 92.0
    circadian_adaptive_prune_percentile: float = 8.0
    circadian_sleep_warmup_steps: int = 6
    circadian_sleep_split_only_until_fraction: float = 0.75
    circadian_sleep_prune_only_after_fraction: float = 0.95
    circadian_sleep_max_change_fraction: float = 0.01
    circadian_sleep_min_change_count: int = 1
    circadian_prune_min_age_steps: int = 120
    circadian_split_threshold: float = 0.80
    circadian_prune_threshold: float = 0.08
    circadian_split_hysteresis_margin: float = 0.02
    circadian_prune_hysteresis_margin: float = 0.02
    circadian_split_cooldown_steps: int = 3
    circadian_prune_cooldown_steps: int = 3
    circadian_split_weight_norm_mix: float = 0.30
    circadian_prune_weight_norm_mix: float = 0.30
    circadian_split_importance_mix: float = 0.20
    circadian_prune_importance_mix: float = 0.35
    circadian_importance_ema_decay: float = 0.95
    circadian_max_split_per_sleep: int = 1
    circadian_max_prune_per_sleep: int = 1
    circadian_split_noise_scale: float = 0.01
    circadian_sleep_reset_factor: float = 0.45
    circadian_sleep_mode: str = "legacy"
    circadian_sleep_enable_chemical_reset: bool = True
    circadian_sleep_enable_homeostasis: bool = True
    circadian_sleep_enable_split: bool = True
    circadian_sleep_enable_prune: bool = True
    circadian_homeostatic_downscale_factor: float = 1.0
    circadian_homeostasis_target_input_norm: float = 0.0
    circadian_homeostasis_target_output_norm: float = 0.0
    circadian_homeostasis_strength: float = 0.50


@dataclass(frozen=True)
class ModelSpeedReport:
    """Training and inference speed metrics for a model."""

    model_name: str
    epochs_ran: int
    final_metric_name: str
    final_metric_value: float
    validation_accuracy: float
    test_accuracy: float
    train_seconds: float
    train_samples_per_second: float
    mean_train_step_ms: float
    inference_latency_mean_ms: float
    inference_latency_p95_ms: float
    inference_samples_per_second: float
    total_parameters: int
    trainable_parameters: int
    benchmark_track: str = VISION_UNMATCHED_REFERENCE_TRACK
    backbone_trainable: bool = False
    backbone_pretraining: str = "none"
    head_type: str = "unknown"
    final_cross_entropy: float | None = None
    final_energy: float | None = None
    training_energy_id: str | None = None
    circadian_hidden_dim_start: int | None = None
    circadian_hidden_dim_end: int | None = None
    circadian_total_splits: int = 0
    circadian_total_prunes: int = 0
    circadian_total_rollbacks: int = 0
    circadian_sleep_attempts: int = 0
    circadian_sleep_cooldown_suppressions: int = 0
    circadian_sleep_retry_cooldown_epochs: int = 0


@dataclass(frozen=True)
class ResNet50BenchmarkResult:
    """Benchmark result across all model variants."""

    device: str
    config: ResNet50BenchmarkConfig
    reports: list[ModelSpeedReport]
    split_hashes: dict[str, str]
    training_order: tuple[str, ...] = ()
    trained_model_hashes: dict[str, str] | None = None


@dataclass(frozen=True)
class _TrainingLoaders:
    """Only the roles that may affect model state or stopping."""

    train_loader: Any
    validation_loader: Any
    guard_loader: Any
    num_classes: int


@dataclass(frozen=True)
class _TrainingOutcome:
    model_name: str
    model: Any
    epochs_ran: int
    validation_accuracy: float
    train_seconds: float
    seen_samples: int
    step_times_ms: tuple[float, ...]
    uses_backprop_metric: bool = False
    final_energy: float | None = None
    circadian_hidden_dim_start: int | None = None
    circadian_total_splits: int = 0
    circadian_total_prunes: int = 0
    circadian_total_rollbacks: int = 0
    circadian_sleep_attempts: int = 0
    circadian_sleep_cooldown_suppressions: int = 0
    circadian_sleep_retry_cooldown_epochs: int = 0


@dataclass(frozen=True)
class ModelDevelopmentReport:
    """Validation-only metrics for tuning; no final-test result is available."""

    model_name: str
    epochs_ran: int
    validation_accuracy: float
    validation_cross_entropy: float
    train_seconds: float
    train_samples_per_second: float
    mean_train_step_ms: float
    inference_latency_mean_ms: float
    inference_latency_p95_ms: float
    inference_samples_per_second: float
    total_parameters: int
    trainable_parameters: int
    benchmark_track: str = VISION_UNMATCHED_REFERENCE_TRACK
    backbone_trainable: bool = False
    backbone_pretraining: str = "none"
    head_type: str = "unknown"
    final_energy: float | None = None
    training_energy_id: str | None = None
    circadian_hidden_dim_start: int | None = None
    circadian_hidden_dim_end: int | None = None
    circadian_total_splits: int = 0
    circadian_total_prunes: int = 0
    circadian_total_rollbacks: int = 0
    circadian_sleep_attempts: int = 0
    circadian_sleep_cooldown_suppressions: int = 0
    circadian_sleep_retry_cooldown_epochs: int = 0


def _training_loaders(loaders: Any) -> _TrainingLoaders:
    return _TrainingLoaders(
        train_loader=loaders.train_loader,
        validation_loader=loaders.validation_loader,
        guard_loader=loaders.guard_loader,
        num_classes=loaders.num_classes,
    )


def benchmark_validation_candidate(
    variant: str,
    torch: Any,
    device: Any,
    loaders: _TrainingLoaders,
    config: ResNet50BenchmarkConfig,
) -> ModelDevelopmentReport:
    """Train one candidate without accepting or reading a final-test loader."""
    if (
        config.protocol_id in GUARD_SEPARATED_VISION_PROTOCOLS
        and loaders.guard_loader is loaders.validation_loader
    ):
        raise ValueError("Guard-separated protocol requires a distinct guard loader.")
    trainers = {
        "backprop": _train_backprop,
        "predictive": _train_predictive,
        "circadian": _train_circadian,
    }
    if variant not in trainers:
        raise ValueError(f"Unknown benchmark variant: {variant}")
    if config.protocol_id == VISION_SEEDED_UNMATCHED_PROTOCOL:
        outcome = _train_seeded_variant(variant, torch, device, loaders, config)
    else:
        outcome = trainers[variant](torch, device, loaders, config)
    return _finalize_validation_report(torch, device, outcome, loaders.validation_loader, config)


def run_resnet50_benchmark(
    config: ResNet50BenchmarkConfig, *, model_order: tuple[str, ...] | None = None,
    checkpoint_store: VisionCheckpointStore | None = None,
    resume_from_checkpoint: bool = False,
) -> ResNet50BenchmarkResult:
    """Benchmark all three model families on the same vision task."""
    default_order = ("backprop", "predictive", "circadian")
    if model_order is not None:
        if config.protocol_id != VISION_SEEDED_UNMATCHED_PROTOCOL:
            raise ValueError("A custom model_order requires the seeded vision protocol.")
        if len(model_order) != 3 or set(model_order) != set(default_order):
            raise ValueError("model_order must be a permutation of the three variants.")
    training_order = model_order or default_order
    _validate_benchmark_config(config)
    if resume_from_checkpoint and checkpoint_store is None:
        raise ValueError("resume_from_checkpoint requires a vision checkpoint store")
    if checkpoint_store is not None:
        return _run_checkpointed_resnet50_benchmark(
            config, training_order, checkpoint_store, resume_from_checkpoint
        )
    torch = require_torch()
    _set_seed(torch, config.seed)
    device = _resolve_device(torch, config.device)
    loaders = _build_benchmark_loaders(config)
    if config.protocol_id in GUARD_SEPARATED_VISION_PROTOCOLS:
        if "guard" not in loaders.split_hashes or loaders.guard_loader is loaders.validation_loader:
            raise ValueError("Guard-separated protocol requires a distinct guard split.")

    training_loaders = _training_loaders(loaders)
    if config.protocol_id == VISION_SEEDED_UNMATCHED_PROTOCOL:
        trained = [
            _train_seeded_variant(name, torch, device, training_loaders, config)
            for name in training_order
        ]
        trained_model_hashes = {
            outcome.model_name: _hash_trained_model(outcome.model) for outcome in trained
        }
    else:
        trained = [
            _train_backprop(torch=torch, device=device, loaders=training_loaders, config=config),
            _train_predictive(torch=torch, device=device, loaders=training_loaders, config=config),
            _train_circadian(torch=torch, device=device, loaders=training_loaders, config=config),
        ]
        trained_model_hashes = None
    # Final labels are first available after every model has stopped learning.
    reports = [
        _finalize_test_report(
            torch=torch, device=device, outcome=outcome,
            test_loader=loaders.test_loader, config=config,
        )
        for outcome in trained
    ]
    _validate_report_models(reports)
    return ResNet50BenchmarkResult(
        device=str(device), config=config, reports=reports,
        split_hashes=dict(loaders.split_hashes), training_order=training_order,
        trained_model_hashes=trained_model_hashes,
    )


def _run_checkpointed_resnet50_benchmark(
    config: ResNet50BenchmarkConfig,
    training_order: tuple[str, ...],
    checkpoint_store: VisionCheckpointStore,
    resume_from_checkpoint: bool,
) -> ResNet50BenchmarkResult:
    """Continue unmatched variants before final-test scoring."""
    seeded_protocol = config.protocol_id == VISION_SEEDED_UNMATCHED_PROTOCOL
    torch = require_torch()
    device = _resolve_device(torch, config.device)
    if str(device) != "cpu":
        raise ValueError("vision checkpoints currently require a CPU device")
    saved = checkpoint_store.load() if resume_from_checkpoint else None
    if saved is not None and (
        saved.protocol_id != config.protocol_id
        or saved.training_order != training_order
        or saved.config_digest != vision_config_digest(config, training_order)
    ):
        raise ValueError("incompatible vision checkpoint config or order")
    if saved is None:
        _set_seed(torch, config.seed)
    previous_random = (
        random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    ) if saved is not None else None
    restored_outcomes: list[_TrainingOutcome] = []
    try:
        loaders = _build_benchmark_loaders(config)
        if config.protocol_id in GUARD_SEPARATED_VISION_PROTOCOLS and (
            "guard" not in loaders.split_hashes
            or loaders.guard_loader is loaders.validation_loader
        ):
            raise ValueError("guard-separated vision checkpoint requires a distinct guard split")
        if seeded_protocol:
            _SeededEpochTrainLoader(
                torch, loaders.train_loader, config.seed + 3_001
            ).snapshot_state()
        else:
            _SharedEpochTrainLoader(torch, loaders.train_loader).snapshot_state()
        data_digest = vision_development_data_digest(loaders)
        if saved is not None:
            validate_vision_checkpoint(
                saved, config=config, data_digest=data_digest,
                training_order=training_order, torch=torch,
                shared_train_loader=not seeded_protocol,
            )
            restored_outcomes = _restore_completed_vision_outcomes(
                saved, training_order, config, device, loaders
            )
            if saved.active_circadian is not None:
                _preflight_active_circadian(saved, config, device, loaders)
    except Exception:
        if previous_random is not None:
            python_state, numpy_state, torch_state = previous_random
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
        raise
    outcomes = restored_outcomes
    if saved is not None:
        restore_vision_process_rng(saved, torch)
        if not seeded_protocol:
            loaders.train_loader.generator.set_state(saved.shared_train_generator_state)
    training_loaders = _training_loaders(loaders)
    legacy_trainers = {
        "backprop": _train_backprop,
        "predictive": _train_predictive,
    }
    for name in training_order[len(outcomes):]:
        if name == "circadian":
            preceding = tuple(
                snapshot_completed_vision_outcome(item, training_order[index])
                for index, item in enumerate(outcomes)
            )
            active = saved if saved is not None and saved.active_circadian is not None else None
            context = VisionCircadianCheckpointContext(
                store=checkpoint_store, config=config, data_digest=data_digest,
                training_order=training_order, completed_outcomes=preceding,
                completed_hashes=tuple(_hash_trained_model(item.model) for item in outcomes),
                outer_entry_torch_state=(
                    active.active_circadian.outer_entry_torch_state
                    if active is not None and active.active_circadian is not None
                    else torch.get_rng_state().clone()
                ),
                resume_checkpoint=active,
            )
            if seeded_protocol:
                outcome = _train_seeded_variant(
                    name, torch, device, training_loaders, config,
                    checkpoint_context=context,
                )
            else:
                shared_loaders = replace(
                    training_loaders,
                    train_loader=_SharedEpochTrainLoader(
                        torch, training_loaders.train_loader
                    ),
                )
                outcome = _train_circadian(
                    torch, device, shared_loaders, config, checkpoint_context=context
                )
        elif seeded_protocol:
            outcome = _train_seeded_variant(name, torch, device, training_loaders, config)
        else:
            outcome = legacy_trainers[name](torch, device, training_loaders, config)
        outcomes.append(outcome)
        checkpoint_store.save(capture_vision_checkpoint(
            config=config, data_digest=data_digest, training_order=training_order,
            completed_outcomes=tuple(
                snapshot_completed_vision_outcome(item, training_order[index])
                for index, item in enumerate(outcomes)
            ),
            completed_hashes=tuple(_hash_trained_model(item.model) for item in outcomes),
            torch=torch,
            shared_train_generator_state=(
                loaders.train_loader.generator.get_state() if not seeded_protocol else None
            ),
        ))
    trained_hashes = {item.model_name: _hash_trained_model(item.model) for item in outcomes}
    reports = [
        _finalize_test_report(torch, device, item, loaders.test_loader, config)
        for item in outcomes
    ]
    _validate_report_models(reports)
    return ResNet50BenchmarkResult(
        device=str(device), config=config, reports=reports,
        split_hashes=dict(loaders.split_hashes), training_order=training_order,
        trained_model_hashes=trained_hashes,
    )


def _restore_completed_vision_outcomes(
    checkpoint: Any, training_order: tuple[str, ...], config: ResNet50BenchmarkConfig,
    device: Any, loaders: Any,
) -> list[_TrainingOutcome]:
    expected = {
        "backprop": ("BackpropResNet50", BackpropResNet50Classifier),
        "predictive": ("PredictiveCodingResNet50", PredictiveCodingResNet50Classifier),
        "circadian": (
            "CircadianPredictiveCodingResNet50", CircadianPredictiveCodingResNet50Classifier
        ),
    }
    restored: list[_TrainingOutcome] = []
    train_loader = loaders.train_loader
    train_batch_count = len(train_loader)
    batch_size = train_loader.batch_size
    if batch_size is None:
        raise ValueError("vision checkpoint requires fixed training batches")
    train_sample_count = (
        train_batch_count * batch_size
        if train_loader.drop_last else len(train_loader.dataset)
    )
    for variant, stored, saved_hash in zip(
        training_order, checkpoint.completed_outcomes, checkpoint.completed_hashes
    ):
        name, model_type = expected[variant]
        if (
            not isinstance(stored, _TrainingOutcome)
            or stored.model_name != name
            or type(stored.epochs_ran) is not int
            or not 1 <= stored.epochs_ran <= config.epochs
            or type(stored.seen_samples) is not int
            or stored.seen_samples != stored.epochs_ran * train_sample_count
        ):
            raise ValueError("incompatible vision checkpoint completed model")
        _preflight_completed_vision_report(
            stored, variant, train_batch_count * stored.epochs_ran, config
        )
        try:
            outcome = restore_completed_vision_outcome(
                stored, variant=variant, config=config, device=device,
                num_classes=loaders.num_classes,
                circadian_config=_build_circadian_head_config(config),
            )
            if not isinstance(outcome.model, model_type):
                raise ValueError("incompatible vision checkpoint model type")
            actual_hash = _hash_trained_model(outcome.model)
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("incompatible vision checkpoint completed model") from exc
        if actual_hash != saved_hash:
            raise ValueError("incompatible vision checkpoint completed model hash")
        restored.append(outcome)
    return restored


def _preflight_completed_vision_report(
    stored: _TrainingOutcome, variant: str, expected_steps: int,
    config: ResNet50BenchmarkConfig,
) -> None:
    """A completed model's learning report must match its saved train cursor."""
    if (
        not isinstance(stored.step_times_ms, tuple)
        or len(stored.step_times_ms) != expected_steps
        or any(not _finite_nonnegative(value) for value in stored.step_times_ms)
        or not _finite_nonnegative(stored.train_seconds)
        or not isinstance(stored.validation_accuracy, (int, float))
        or not isfinite(stored.validation_accuracy)
        or not 0 <= stored.validation_accuracy <= 1
        or stored.uses_backprop_metric != (variant == "backprop")
    ):
        raise ValueError("incompatible vision checkpoint completed report")
    if variant == "backprop":
        if stored.final_energy is not None:
            raise ValueError("incompatible vision checkpoint backprop report")
    elif not isinstance(stored.final_energy, (int, float)) or not isfinite(stored.final_energy):
        raise ValueError("incompatible vision checkpoint model energy")
    sleep_counts = (
        stored.circadian_total_splits, stored.circadian_total_prunes,
        stored.circadian_total_rollbacks, stored.circadian_sleep_attempts,
        stored.circadian_sleep_cooldown_suppressions,
        stored.circadian_sleep_retry_cooldown_epochs,
    )
    if any(type(value) is not int or value < 0 for value in sleep_counts):
        raise ValueError("incompatible vision checkpoint sleep report")
    if variant == "circadian":
        if (
            stored.circadian_hidden_dim_start != config.circadian_head_hidden_dim
            or stored.circadian_total_rollbacks > stored.circadian_sleep_attempts
            or stored.circadian_sleep_attempts > stored.epochs_ran
        ):
            raise ValueError("incompatible vision checkpoint circadian report")
    elif stored.circadian_hidden_dim_start is not None or any(sleep_counts):
        raise ValueError("incompatible vision checkpoint baseline sleep report")


def _finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and isfinite(value) and value >= 0


def _train_seeded_variant(
    variant: str, torch: Any, device: Any, loaders: _TrainingLoaders,
    config: ResNet50BenchmarkConfig,
    *, checkpoint_context: VisionCircadianCheckpointContext | None = None,
) -> _TrainingOutcome:
    trainers = {
        "backprop": _train_backprop,
        "predictive": _train_predictive,
        "circadian": _train_circadian,
    }
    seed_offsets = {"backprop": 2_101, "predictive": 2_102, "circadian": 2_103}
    variant_loaders = replace(
        loaders,
        train_loader=_SeededEpochTrainLoader(torch, loaders.train_loader, config.seed + 3_001),
    )
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    if checkpoint_context is not None and checkpoint_context.resume_checkpoint is not None:
        torch.set_rng_state(checkpoint_context.outer_entry_torch_state.detach().clone())
    with torch.random.fork_rng(devices=cuda_devices):
        _set_seed(torch, config.seed + seed_offsets[variant])
        if checkpoint_context is not None:
            if variant != "circadian":
                raise ValueError("vision training checkpoint context requires circadian variant")
            return _train_circadian(
                torch, device, variant_loaders, config, checkpoint_context=checkpoint_context
            )
        return trainers[variant](torch, device, variant_loaders, config)


def _hash_trained_model(model: Any) -> str:
    torch = require_torch()
    named_tensors = [
        (f"backbone.{name}", tensor)
        for name, tensor in model.backbone.state_dict().items()
    ]
    named_scalars: list[tuple[str, Any]] = []
    if hasattr(model, "classifier"):
        named_tensors.extend(
            (f"classifier.{name}", tensor)
            for name, tensor in model.classifier.state_dict().items()
        )
    else:
        head = model.head
        if hasattr(head, "snapshot_state"):
            # Chemical, traffic, and cooldown state affects later learning even when weights match.
            for name, value in head.snapshot_state().items():
                target = named_tensors if torch.is_tensor(value) else named_scalars
                target.append((f"head.{name}", value))
            named_tensors.append(("head.split_generator", head._split_generator.get_state()))
        else:
            named_tensors.extend(
                (f"head.{name}", getattr(head, name))
                for name in (
                    "weight_feature_hidden", "bias_hidden", "weight_hidden_output", "bias_output",
                    "_traffic_sum",
                )
            )
            named_scalars.append(("head.traffic_steps", head._traffic_steps))
    digest = sha256()
    for name, tensor in sorted(named_tensors):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(value.shape).encode("utf-8"))
        digest.update(value.tobytes())
    for name, value in sorted(named_scalars):
        digest.update(name.encode("utf-8"))
        digest.update(dumps(value, sort_keys=True, allow_nan=False).encode("utf-8"))
    return digest.hexdigest()


def format_resnet50_benchmark_result(result: ResNet50BenchmarkResult) -> str:
    """Create a readable benchmark report."""
    lines = [
        "ResNet-50 Reference Benchmark (Backprop, Predictive, Circadian)",
        "------------------------------------------------------------",
        f"Protocol: {result.config.protocol_id}",
        f"Circadian sleep mode: {result.config.circadian_sleep_mode}",
        "Repeated stopping and rollback decisions use guard examples; "
        "outer validation is measured after training."
        if result.config.protocol_id in GUARD_SEPARATED_VISION_PROTOCOLS
        else "Legacy validation also supplied repeated stopping and rollback decisions.",
        "Comparison status: legacy unmatched heads and backbone state; deltas are descriptive.",
        "Legacy linear-head reference: BackpropResNet50.",
        f"Device: {result.device}",
        _format_dataset_summary(result.config),
        f"Split hashes: {result.split_hashes}",
        f"Training order: {result.training_order}",
        "",
    ]
    if result.trained_model_hashes is not None:
        lines.insert(-1, f"Trained model hashes: {result.trained_model_hashes}")
    backprop_report = _find_report_by_name(result.reports, "BackpropResNet50")
    for report in result.reports:
        lines.extend(
            [
                f"{report.model_name}",
                (
                    f"  track={report.benchmark_track}, "
                    f"backbone_trainable={report.backbone_trainable}, "
                    f"pretraining={report.backbone_pretraining}, "
                    f"head={report.head_type}"
                ),
                (
                    f"  epochs={report.epochs_ran}, "
                    f"{report.final_metric_name}={report.final_metric_value:.4f}, "
                    f"validation_acc={report.validation_accuracy:.3f}, "
                    f"test_acc={report.test_accuracy:.3f}"
                ),
                (
                    f"  training: {report.train_seconds:.2f}s total, "
                    f"{report.train_samples_per_second:.1f} samples/s, "
                    f"{report.mean_train_step_ms:.2f} ms/step"
                ),
                (
                    f"  inference: mean={report.inference_latency_mean_ms:.2f} ms, "
                    f"p95={report.inference_latency_p95_ms:.2f} ms, "
                    f"{report.inference_samples_per_second:.1f} samples/s"
                ),
                (
                    f"  params: total={report.total_parameters:,}, "
                    f"trainable={report.trainable_parameters:,}"
                ),
            ]
        )
        if report.model_name != "BackpropResNet50":
            accuracy_delta = report.test_accuracy - backprop_report.test_accuracy
            speed_delta = report.train_samples_per_second - backprop_report.train_samples_per_second
            lines.append(
                (
                    "  vs legacy linear backprop (descriptive): "
                    f"acc_delta={accuracy_delta:+.3f}, "
                    f"train_samples_per_second_delta={speed_delta:+.1f}"
                )
            )
        if report.final_cross_entropy is not None and report.final_metric_name != "cross_entropy":
            lines.append(f"  cross_entropy={report.final_cross_entropy:.4f}")
        if report.final_energy is not None:
            energy_id = report.training_energy_id or "legacy_unlabeled"
            lines.append(
                f"  last training diagnostic [{energy_id}]={report.final_energy:.4f}"
            )
        if report.circadian_hidden_dim_start is not None and report.circadian_hidden_dim_end is not None:
            lines.append(
                (
                    "  circadian sleep: "
                    f"hidden={report.circadian_hidden_dim_start}->{report.circadian_hidden_dim_end}, "
                    f"splits={report.circadian_total_splits}, "
                    f"prunes={report.circadian_total_prunes}, "
                    f"rollbacks={report.circadian_total_rollbacks}, "
                    f"attempts={report.circadian_sleep_attempts}, "
                    f"cooldown_skips={report.circadian_sleep_cooldown_suppressions}, "
                    f"cooldown_epochs={report.circadian_sleep_retry_cooldown_epochs}"
                )
            )
        lines.append("")
    return "\n".join(lines).strip()


def _build_benchmark_loaders(config: ResNet50BenchmarkConfig) -> Any:
    guard_separated = config.protocol_id in GUARD_SEPARATED_VISION_PROTOCOLS
    if config.dataset_name == "synthetic":
        return build_synthetic_vision_dataloaders(
            SyntheticVisionDatasetConfig(
                train_samples=config.train_samples,
                validation_samples=config.validation_samples,
                guard_samples=config.guard_samples if guard_separated else 0,
                test_samples=config.test_samples,
                num_classes=config.num_classes,
                image_size=config.image_size,
                batch_size=config.batch_size,
                noise_std=config.dataset_noise_std,
                difficulty=config.dataset_difficulty,
                seed=config.seed,
                num_workers=config.dataset_num_workers,
            )
        )
    return build_torchvision_vision_dataloaders(
        TorchVisionDatasetConfig(
            dataset_name=config.dataset_name,
            data_root=config.dataset_data_root,
            batch_size=config.batch_size,
            image_size=config.image_size,
            seed=config.seed,
            num_workers=config.dataset_num_workers,
            download=config.dataset_download,
            train_subset_size=config.dataset_train_subset_size,
            validation_subset_size=config.dataset_validation_subset_size,
            guard_subset_size=config.dataset_guard_subset_size if guard_separated else 0,
            test_subset_size=config.dataset_test_subset_size,
            use_augmentation=config.dataset_use_augmentation,
        )
    )


def _format_dataset_summary(config: ResNet50BenchmarkConfig) -> str:
    guard_separated = config.protocol_id in GUARD_SEPARATED_VISION_PROTOCOLS
    if config.dataset_name == "synthetic":
        guard_summary = f"{config.guard_samples} guard / " if guard_separated else ""
        return (
            "Dataset: synthetic"
            f" ({config.train_samples} train / {config.validation_samples} validation / "
            f"{guard_summary}"
            f"{config.test_samples} test, "
            f"classes={config.num_classes}, size={config.image_size}, "
            f"difficulty={config.dataset_difficulty}, noise={config.dataset_noise_std:.3f})"
        )
    subset_suffix = ""
    if (
        config.dataset_train_subset_size > 0
        or config.dataset_test_subset_size > 0
    ):
        subset_suffix = (
            f", subset_train={config.dataset_train_subset_size or 'full'}"
            f", subset_test={config.dataset_test_subset_size or 'full'}"
        )
    return (
        f"Dataset: {config.dataset_name} "
        f"(root={config.dataset_data_root}, size={config.image_size}, "
        f"batch={config.batch_size}, validation={config.dataset_validation_subset_size}, "
        f"guard={config.dataset_guard_subset_size if guard_separated else 0}, "
        f"augmentation={config.dataset_use_augmentation}"
        f"{subset_suffix})"
    )


def _benchmark_backprop(
    torch: Any,
    device: Any,
    loaders: Any,
    config: ResNet50BenchmarkConfig,
) -> ModelSpeedReport:
    """Compatibility entry point for historical single-variant scripts."""
    training_loaders = _training_loaders(loaders)
    outcome = (
        _train_seeded_variant("backprop", torch, device, training_loaders, config)
        if config.protocol_id == VISION_SEEDED_UNMATCHED_PROTOCOL
        else _train_backprop(torch, device, training_loaders, config)
    )
    return _finalize_test_report(
        torch=torch, device=device,
        outcome=outcome,
        test_loader=loaders.test_loader, config=config,
    )


def _train_backprop(
    torch: Any,
    device: Any,
    loaders: _TrainingLoaders,
    config: ResNet50BenchmarkConfig,
) -> _TrainingOutcome:
    model = BackpropResNet50Classifier(
        num_classes=loaders.num_classes,
        device=device,
        freeze_backbone=config.backprop_freeze_backbone,
        backbone_weights=config.backbone_weights,
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.trainable_parameters(),
        lr=config.backprop_learning_rate,
        momentum=config.backprop_momentum,
    )

    step_times_ms: list[float] = []
    seen_samples = 0
    epochs_ran = 0
    eval_batches = _resolve_eval_batches(config.evaluation_batches)

    if config.backprop_freeze_backbone:
        model.backbone.eval()
    else:
        model.backbone.train()
    model.classifier.train()

    train_timer_start = perf_counter()
    for epoch in range(1, config.epochs + 1):
        for images, labels in loaders.train_loader:
            images = images.to(device)
            labels = labels.to(device)

            sync_device(torch, device)
            step_start = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_logits(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            sync_device(torch, device)

            step_times_ms.append((perf_counter() - step_start) * 1000.0)
            seen_samples += int(labels.shape[0])

        epochs_ran = epoch
        eval_accuracy, _ = _compute_backprop_metrics(
            torch, model, loaders.guard_loader, device, max_batches=eval_batches
        )
        if _should_stop_early(
            target_accuracy=config.target_accuracy,
            accuracy=eval_accuracy,
        ):
            break
    train_seconds = perf_counter() - train_timer_start

    model.backbone.eval()
    model.classifier.eval()
    validation_accuracy, _ = _compute_backprop_metrics(
        torch, model, loaders.validation_loader, device, max_batches=None
    )
    return _TrainingOutcome(
        model_name="BackpropResNet50",
        model=model,
        epochs_ran=epochs_ran,
        validation_accuracy=validation_accuracy,
        train_seconds=train_seconds,
        seen_samples=seen_samples,
        step_times_ms=tuple(step_times_ms),
        uses_backprop_metric=True,
    )


def _benchmark_predictive(
    torch: Any,
    device: Any,
    loaders: Any,
    config: ResNet50BenchmarkConfig,
) -> ModelSpeedReport:
    """Compatibility entry point for historical single-variant scripts."""
    training_loaders = _training_loaders(loaders)
    outcome = (
        _train_seeded_variant("predictive", torch, device, training_loaders, config)
        if config.protocol_id == VISION_SEEDED_UNMATCHED_PROTOCOL
        else _train_predictive(torch, device, training_loaders, config)
    )
    return _finalize_test_report(
        torch=torch, device=device,
        outcome=outcome,
        test_loader=loaders.test_loader, config=config,
    )


def _train_predictive(
    torch: Any,
    device: Any,
    loaders: _TrainingLoaders,
    config: ResNet50BenchmarkConfig,
) -> _TrainingOutcome:
    model = PredictiveCodingResNet50Classifier(
        num_classes=loaders.num_classes,
        device=device,
        head_hidden_dim=config.predictive_head_hidden_dim,
        seed=config.seed + 11,
        freeze_backbone=True,
        backbone_weights=config.backbone_weights,
    )

    step_times_ms: list[float] = []
    seen_samples = 0
    epochs_ran = 0
    final_energy = 0.0
    eval_batches = _resolve_eval_batches(config.evaluation_batches)

    train_timer_start = perf_counter()
    for epoch in range(1, config.epochs + 1):
        for images, labels in loaders.train_loader:
            images = images.to(device)
            labels = labels.to(device)

            sync_device(torch, device)
            step_start = perf_counter()
            energy = model.train_step(
                images=images,
                targets=labels,
                learning_rate=config.predictive_learning_rate,
                inference_steps=config.predictive_inference_steps,
                inference_learning_rate=config.predictive_inference_learning_rate,
            )
            sync_device(torch, device)

            step_times_ms.append((perf_counter() - step_start) * 1000.0)
            final_energy = float(energy)
            seen_samples += int(labels.shape[0])

        epochs_ran = epoch
        eval_accuracy, _ = _compute_pc_metrics(
            torch, model, loaders.guard_loader, device, max_batches=eval_batches
        )
        if _should_stop_early(
            target_accuracy=config.target_accuracy,
            accuracy=eval_accuracy,
        ):
            break
    train_seconds = perf_counter() - train_timer_start

    validation_accuracy, _ = _compute_pc_metrics(
        torch, model, loaders.validation_loader, device, max_batches=None
    )
    return _TrainingOutcome(
        model_name="PredictiveCodingResNet50",
        model=model,
        epochs_ran=epochs_ran,
        final_energy=final_energy,
        validation_accuracy=validation_accuracy,
        train_seconds=train_seconds,
        seen_samples=seen_samples,
        step_times_ms=tuple(step_times_ms),
    )


def _benchmark_circadian(
    torch: Any,
    device: Any,
    loaders: Any,
    config: ResNet50BenchmarkConfig,
) -> ModelSpeedReport:
    """Compatibility entry point for historical single-variant scripts."""
    training_loaders = _training_loaders(loaders)
    outcome = (
        _train_seeded_variant("circadian", torch, device, training_loaders, config)
        if config.protocol_id == VISION_SEEDED_UNMATCHED_PROTOCOL
        else _train_circadian(torch, device, training_loaders, config)
    )
    return _finalize_test_report(
        torch=torch, device=device,
        outcome=outcome,
        test_loader=loaders.test_loader, config=config,
    )


def _build_circadian_head_config(config: ResNet50BenchmarkConfig) -> CircadianHeadConfig:
    """Keep circadian dynamics identical across image and fixed-feature tracks."""
    return CircadianHeadConfig(
        chemical_decay=config.circadian_chemical_decay,
        chemical_buildup_rate=config.circadian_chemical_buildup_rate,
        use_saturating_chemical=config.circadian_use_saturating_chemical,
        chemical_max_value=config.circadian_chemical_max_value,
        chemical_saturation_gain=config.circadian_chemical_saturation_gain,
        use_dual_chemical=config.circadian_use_dual_chemical,
        dual_fast_mix=config.circadian_dual_fast_mix,
        slow_chemical_decay=config.circadian_slow_chemical_decay,
        slow_buildup_scale=config.circadian_slow_buildup_scale,
        plasticity_sensitivity=config.circadian_plasticity_sensitivity,
        use_adaptive_plasticity_sensitivity=config.circadian_use_adaptive_plasticity_sensitivity,
        plasticity_sensitivity_min=config.circadian_plasticity_sensitivity_min,
        plasticity_sensitivity_max=config.circadian_plasticity_sensitivity_max,
        plasticity_importance_mix=config.circadian_plasticity_importance_mix,
        min_plasticity=config.circadian_min_plasticity,
        use_reward_modulated_learning=config.circadian_use_reward_modulated_learning,
        reward_baseline_decay=config.circadian_reward_baseline_decay,
        reward_difficulty_exponent=config.circadian_reward_difficulty_exponent,
        reward_scale_min=config.circadian_reward_scale_min,
        reward_scale_max=config.circadian_reward_scale_max,
        use_adaptive_thresholds=config.circadian_use_adaptive_thresholds,
        adaptive_split_percentile=config.circadian_adaptive_split_percentile,
        adaptive_prune_percentile=config.circadian_adaptive_prune_percentile,
        sleep_warmup_steps=config.circadian_sleep_warmup_steps,
        sleep_split_only_until_fraction=config.circadian_sleep_split_only_until_fraction,
        sleep_prune_only_after_fraction=config.circadian_sleep_prune_only_after_fraction,
        sleep_max_change_fraction=config.circadian_sleep_max_change_fraction,
        sleep_min_change_count=config.circadian_sleep_min_change_count,
        prune_min_age_steps=config.circadian_prune_min_age_steps,
        split_threshold=config.circadian_split_threshold,
        prune_threshold=config.circadian_prune_threshold,
        split_hysteresis_margin=config.circadian_split_hysteresis_margin,
        prune_hysteresis_margin=config.circadian_prune_hysteresis_margin,
        split_cooldown_steps=config.circadian_split_cooldown_steps,
        prune_cooldown_steps=config.circadian_prune_cooldown_steps,
        split_weight_norm_mix=config.circadian_split_weight_norm_mix,
        prune_weight_norm_mix=config.circadian_prune_weight_norm_mix,
        split_importance_mix=config.circadian_split_importance_mix,
        prune_importance_mix=config.circadian_prune_importance_mix,
        importance_ema_decay=config.circadian_importance_ema_decay,
        max_split_per_sleep=config.circadian_max_split_per_sleep,
        max_prune_per_sleep=config.circadian_max_prune_per_sleep,
        split_noise_scale=config.circadian_split_noise_scale,
        sleep_reset_factor=config.circadian_sleep_reset_factor,
        sleep_mode=config.circadian_sleep_mode,
        sleep_enable_chemical_reset=config.circadian_sleep_enable_chemical_reset,
        sleep_enable_homeostasis=config.circadian_sleep_enable_homeostasis,
        sleep_enable_split=config.circadian_sleep_enable_split,
        sleep_enable_prune=config.circadian_sleep_enable_prune,
        homeostatic_downscale_factor=config.circadian_homeostatic_downscale_factor,
        homeostasis_target_input_norm=config.circadian_homeostasis_target_input_norm,
        homeostasis_target_output_norm=config.circadian_homeostasis_target_output_norm,
        homeostasis_strength=config.circadian_homeostasis_strength,
        use_adaptive_sleep_trigger=config.circadian_use_adaptive_sleep_trigger,
        min_sleep_steps=config.circadian_min_sleep_steps,
        sleep_energy_window=config.circadian_sleep_energy_window,
        sleep_plateau_delta=config.circadian_sleep_plateau_delta,
        sleep_chemical_variance_threshold=config.circadian_sleep_chemical_variance_threshold,
        use_adaptive_sleep_budget=config.circadian_use_adaptive_sleep_budget,
        adaptive_sleep_budget_min_scale=config.circadian_adaptive_sleep_budget_min_scale,
        adaptive_sleep_budget_max_scale=config.circadian_adaptive_sleep_budget_max_scale,
        adaptive_sleep_budget_plateau_weight=config.circadian_adaptive_sleep_budget_plateau_weight,
        adaptive_sleep_budget_variance_weight=config.circadian_adaptive_sleep_budget_variance_weight,
    )


def _new_circadian_classifier(
    device: Any, num_classes: int, config: ResNet50BenchmarkConfig,
) -> CircadianPredictiveCodingResNet50Classifier:
    return CircadianPredictiveCodingResNet50Classifier(
        num_classes=num_classes,
        device=device,
        head_hidden_dim=config.circadian_head_hidden_dim,
        seed=config.seed + 23,
        freeze_backbone=True,
        backbone_weights=config.backbone_weights,
        circadian_config=_build_circadian_head_config(config),
        min_hidden_dim=config.circadian_min_hidden_dim,
        max_hidden_dim=config.circadian_max_hidden_dim,
    )


def _preflight_active_circadian(
    checkpoint: Any, config: ResNet50BenchmarkConfig, device: Any, loaders: Any,
) -> None:
    """Reject malformed progress and model state before restoring process RNG."""
    torch = require_torch()
    progress = checkpoint.active_circadian
    if not isinstance(progress, VisionCircadianProgress):
        raise ValueError("incompatible vision checkpoint circadian progress")
    train_loader = loaders.train_loader
    batch_count = len(train_loader)
    batch_size = train_loader.batch_size
    if batch_size is None:
        raise ValueError("vision checkpoint requires fixed training batches")
    sample_count = len(train_loader.dataset)
    batch_sizes = tuple(
        min(batch_size, sample_count - index * batch_size)
        for index in range(batch_count)
    )
    if train_loader.drop_last:
        batch_sizes = tuple(batch_size for _ in range(batch_count))
    counters = (
        progress.completed_epoch, progress.next_batch_index, progress.initial_hidden_dim,
        progress.epochs_ran, progress.wake_batches, progress.seen_samples,
        progress.sleep_attempts, progress.sleep_rollbacks, progress.sleep_splits,
        progress.sleep_prunes,
    )
    if any(type(value) is not int or value < 0 for value in counters):
        raise ValueError("incompatible vision checkpoint circadian counters")
    if progress.stage == "wake":
        if not (
            progress.completed_epoch < config.epochs
            and 1 <= progress.next_batch_index <= batch_count
            and progress.epochs_ran == progress.completed_epoch
        ):
            raise ValueError("incompatible vision checkpoint wake cursor")
        prefix = progress.next_batch_index
    elif progress.stage in {"before_sleep", "after_sleep"}:
        if not (
            1 <= progress.completed_epoch <= config.epochs
            and progress.next_batch_index == 0
            and progress.epochs_ran == progress.completed_epoch - 1
        ):
            raise ValueError("incompatible vision checkpoint sleep cursor")
        prefix = 0
    else:
        raise ValueError("incompatible vision checkpoint sleep stage")
    expected_batches = progress.completed_epoch * batch_count + prefix
    expected_samples = progress.completed_epoch * sum(batch_sizes) + sum(batch_sizes[:prefix])
    if (
        progress.initial_hidden_dim != config.circadian_head_hidden_dim
        or progress.wake_batches != expected_batches
        or progress.seen_samples != expected_samples
        or progress.sleep_rollbacks > progress.sleep_attempts
        or not isinstance(progress.step_times_ms, tuple)
        or len(progress.step_times_ms) != progress.wake_batches
        or any(not _finite_nonnegative(value) for value in progress.step_times_ms)
        or not isinstance(progress.final_energy, (int, float))
        or not isfinite(progress.final_energy)
        or not _finite_nonnegative(progress.elapsed_seconds)
    ):
        raise ValueError("incompatible vision checkpoint circadian report")
    expected_loader_epoch = progress.completed_epoch
    loader_state_type = (
        SeededEpochLoaderState
        if config.protocol_id == VISION_SEEDED_UNMATCHED_PROTOCOL
        else SharedEpochLoaderState
    )
    if (
        not isinstance(progress.loader_state, loader_state_type)
        or not isinstance(progress.retry_state, SleepRollbackCooldownState)
        or progress.loader_state.epoch != expected_loader_epoch
        or progress.loader_state.next_batch_index != prefix
    ):
        raise ValueError("incompatible vision checkpoint loader cursor")
    try:
        if config.protocol_id == VISION_SEEDED_UNMATCHED_PROTOCOL:
            if not isinstance(progress.loader_state, SeededEpochLoaderState):
                raise ValueError("incompatible seeded vision checkpoint loader state")
            _SeededEpochTrainLoader(
                torch, train_loader, config.seed + 3_001
            ).restore_state(progress.loader_state)
        else:
            if not isinstance(progress.loader_state, SharedEpochLoaderState):
                raise ValueError("incompatible shared vision checkpoint loader state")
            _SharedEpochTrainLoader(torch, train_loader).restore_state(
                progress.loader_state
            )
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("incompatible vision checkpoint loader state") from exc
    if config.protocol_id != VISION_SEEDED_UNMATCHED_PROTOCOL and not torch.equal(
        progress.loader_state.generator_state, checkpoint.shared_train_generator_state
    ):
        raise ValueError("incompatible vision checkpoint shared loader state")
    retry = SleepRollbackCooldown(resolve_rollback_cooldown_epochs(
        config.circadian_sleep_mode, config.circadian_sleep_rollback_cooldown_epochs
    ))
    try:
        retry.restore_state(progress.retry_state)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("incompatible vision checkpoint rollback state") from exc
    if retry.snapshot_state().rejected_attempts != progress.sleep_rollbacks:
        raise ValueError("incompatible vision checkpoint rollback counters")
    entry = progress.outer_entry_torch_state
    if not torch.is_tensor(entry) or entry.dtype != torch.uint8:
        raise ValueError("incompatible vision checkpoint outer Torch RNG")
    try:
        torch.Generator(device="cpu").set_state(entry.detach().clone())
    except RuntimeError as exc:
        raise ValueError("incompatible vision checkpoint outer Torch RNG") from exc
    candidate = _new_circadian_classifier(device, loaders.num_classes, config)
    try:
        candidate.restore_full_state(progress.classifier_state)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("incompatible vision checkpoint classifier state") from exc
    if candidate.get_sleep_clocks().wake_batches != progress.wake_batches:
        raise ValueError("incompatible vision checkpoint classifier wake clock")


def _train_circadian(
    torch: Any,
    device: Any,
    loaders: _TrainingLoaders,
    config: ResNet50BenchmarkConfig,
    *, checkpoint_context: VisionCircadianCheckpointContext | None = None,
) -> _TrainingOutcome:
    model = _new_circadian_classifier(device, loaders.num_classes, config)
    hidden_dim_start = model.head.hidden_dim
    sleep_splits = 0
    sleep_prunes = 0
    sleep_rollbacks = 0
    sleep_attempts = 0
    wake_batches = 0
    retry = SleepRollbackCooldown(
        resolve_rollback_cooldown_epochs(
            config.circadian_sleep_mode, config.circadian_sleep_rollback_cooldown_epochs
        )
    )

    step_times_ms: list[float] = []
    seen_samples = 0
    epochs_ran = 0
    final_energy = 0.0
    resume_checkpoint = (
        checkpoint_context.resume_checkpoint if checkpoint_context is not None else None
    )
    resume_progress = (
        resume_checkpoint.active_circadian if resume_checkpoint is not None else None
    )
    if resume_progress is not None:
        model.restore_full_state(resume_progress.classifier_state)
        loaders.train_loader.restore_state(resume_progress.loader_state)
        retry.restore_state(resume_progress.retry_state)
        hidden_dim_start = resume_progress.initial_hidden_dim
        sleep_splits = resume_progress.sleep_splits
        sleep_prunes = resume_progress.sleep_prunes
        sleep_rollbacks = resume_progress.sleep_rollbacks
        sleep_attempts = resume_progress.sleep_attempts
        wake_batches = resume_progress.wake_batches
        step_times_ms = list(resume_progress.step_times_ms)
        seen_samples = resume_progress.seen_samples
        epochs_ran = resume_progress.epochs_ran
        final_energy = resume_progress.final_energy
        assert resume_checkpoint is not None
        restore_vision_process_rng(resume_checkpoint, torch)
    eval_batches = _resolve_eval_batches(config.evaluation_batches)
    rollback_eval_batches = _resolve_eval_batches(
        config.circadian_sleep_rollback_eval_batches
    )
    if rollback_eval_batches is None:
        rollback_eval_batches = eval_batches

    train_timer_start = perf_counter()
    elapsed_before = resume_progress.elapsed_seconds if resume_progress is not None else 0.0
    checkpoint_pause = 0.0

    def active_elapsed() -> float:
        replay = getattr(loaders.train_loader, "replay_seconds", 0.0)
        return elapsed_before + perf_counter() - train_timer_start - checkpoint_pause - replay

    def save_checkpoint(stage: str, completed_epoch: int, next_batch_index: int) -> None:
        nonlocal checkpoint_pause
        if checkpoint_context is None:
            return
        pause_start = perf_counter()
        elapsed = active_elapsed()
        progress = VisionCircadianProgress(
            stage=stage,
            completed_epoch=completed_epoch,
            next_batch_index=next_batch_index,
            classifier_state=model.snapshot_full_state(),
            loader_state=loaders.train_loader.snapshot_state(),
            retry_state=retry.snapshot_state(),
            outer_entry_torch_state=checkpoint_context.outer_entry_torch_state.detach().clone(),
            initial_hidden_dim=hidden_dim_start,
            epochs_ran=epochs_ran,
            wake_batches=wake_batches,
            seen_samples=seen_samples,
            sleep_attempts=sleep_attempts,
            sleep_rollbacks=sleep_rollbacks,
            sleep_splits=sleep_splits,
            sleep_prunes=sleep_prunes,
            final_energy=final_energy,
            step_times_ms=tuple(step_times_ms),
            elapsed_seconds=elapsed,
        )
        checkpoint_context.store.save(capture_vision_checkpoint(
            config=config,
            data_digest=checkpoint_context.data_digest,
            training_order=checkpoint_context.training_order,
            completed_outcomes=checkpoint_context.completed_outcomes,
            completed_hashes=checkpoint_context.completed_hashes,
            torch=torch,
            active_circadian=progress,
            shared_train_generator_state=(
                loaders.train_loader.loader.generator.get_state()
                if config.protocol_id != VISION_SEEDED_UNMATCHED_PROTOCOL else None
            ),
        ))
        checkpoint_pause += perf_counter() - pause_start

    first_epoch = (
        resume_progress.completed_epoch + (1 if resume_progress.stage == "wake" else 0)
        if resume_progress is not None else 1
    )
    for epoch in range(first_epoch, config.epochs + 1):
        stage = resume_progress.stage if resume_progress is not None else None
        if stage not in {"before_sleep", "after_sleep"}:
            for images, labels in loaders.train_loader:
                images = images.to(device)
                labels = labels.to(device)

                sync_device(torch, device)
                step_start = perf_counter()
                energy = model.train_step(
                    images=images,
                    targets=labels,
                    learning_rate=config.circadian_learning_rate,
                    inference_steps=config.circadian_inference_steps,
                    inference_learning_rate=config.circadian_inference_learning_rate,
                )
                sync_device(torch, device)

                step_times_ms.append((perf_counter() - step_start) * 1000.0)
                final_energy = float(energy)
                seen_samples += int(labels.shape[0])
                wake_batches += 1
                if checkpoint_context is not None:
                    cursor = loaders.train_loader.snapshot_state()
                    save_checkpoint("wake", epoch - 1, cursor.next_batch_index)
            if checkpoint_context is not None:
                save_checkpoint("before_sleep", epoch, 0)

        if stage != "after_sleep":
            adaptive_triggered = (
                config.circadian_use_adaptive_sleep_trigger and model.should_trigger_sleep()
            )
            sleep_decision = decide_sleep_attempt(
                sleep_mode=config.circadian_sleep_mode,
                completed_epochs=epoch,
                interval_epochs=config.circadian_sleep_interval,
                adaptive_due=adaptive_triggered,
                force_periodic=config.circadian_force_sleep,
            )
            if retry.allow_due_attempt(
                sleep_decision, completed_epochs=epoch, wake_batches=wake_batches
            ):
                sleep_attempts += 1
                sleep_result, rolled_back = _guarded_circadian_sleep_event(
                    torch,
                    device,
                    model,
                    loaders.guard_loader,
                    config,
                    epoch,
                    sleep_decision.force_sleep,
                    rollback_eval_batches,
                )
                sleep_rollbacks += int(rolled_back)
                if rolled_back:
                    retry.record_rejection(completed_epochs=epoch, wake_batches=wake_batches)
                sleep_splits += len(sleep_result.split_indices)
                sleep_prunes += len(sleep_result.pruned_indices)
            if checkpoint_context is not None:
                save_checkpoint("after_sleep", epoch, 0)

        epochs_ran = epoch
        eval_accuracy, _ = _compute_pc_metrics(
            torch, model, loaders.guard_loader, device, max_batches=eval_batches
        )
        if _should_stop_early(
            target_accuracy=config.target_accuracy,
            accuracy=eval_accuracy,
        ):
            break
        resume_progress = None
    train_seconds = active_elapsed()

    validation_accuracy, _ = _compute_pc_metrics(
        torch, model, loaders.validation_loader, device, max_batches=None
    )
    return _TrainingOutcome(
        model_name="CircadianPredictiveCodingResNet50",
        model=model,
        epochs_ran=epochs_ran,
        final_energy=final_energy,
        validation_accuracy=validation_accuracy,
        train_seconds=train_seconds,
        seen_samples=seen_samples,
        step_times_ms=tuple(step_times_ms),
        circadian_hidden_dim_start=hidden_dim_start,
        circadian_total_splits=sleep_splits,
        circadian_total_prunes=sleep_prunes,
        circadian_total_rollbacks=sleep_rollbacks,
        circadian_sleep_attempts=sleep_attempts,
        circadian_sleep_cooldown_suppressions=retry.snapshot_state().suppressed_due_attempts,
        circadian_sleep_retry_cooldown_epochs=retry.cooldown_epochs,
    )


def _guarded_circadian_sleep_event(
    torch: Any,
    device: Any,
    model: CircadianPredictiveCodingResNet50Classifier,
    guard_loader: Any,
    config: ResNet50BenchmarkConfig,
    epoch: int,
    force_sleep: bool,
    rollback_eval_batches: int | None,
) -> tuple[Any, bool]:
    snapshot = None
    pre_accuracy = pre_cross_entropy = 0.0
    if config.circadian_enable_sleep_rollback:
        snapshot = model.snapshot_state()
    try:
        if snapshot is not None:
            pre_accuracy, pre_cross_entropy = _compute_pc_metrics(
                torch, model, guard_loader, device, max_batches=rollback_eval_batches
            )
            _require_finite_guard_scores(pre_accuracy, pre_cross_entropy)
        event = model.sleep_event(
            force_sleep=force_sleep,
            epoch_progress=SleepEpochProgress(epoch, config.epochs),
        )
        if snapshot is not None:
            post_accuracy, post_cross_entropy = _compute_pc_metrics(
                torch, model, guard_loader, device, max_batches=rollback_eval_batches
            )
            _require_finite_guard_scores(post_accuracy, post_cross_entropy)
            rollback_delta = _compute_rollback_delta(
                metric_name=config.circadian_sleep_rollback_metric,
                pre_accuracy=pre_accuracy,
                post_accuracy=post_accuracy,
                pre_cross_entropy=pre_cross_entropy,
                post_cross_entropy=post_cross_entropy,
            )
            if not isfinite(rollback_delta):
                raise FloatingPointError("nonfinite guard rollback delta")
    except Exception:
        if snapshot is not None:
            model.restore_state(snapshot)
        raise

    if snapshot is not None and rollback_delta > config.circadian_sleep_rollback_tolerance:
        model.restore_state(snapshot)
        return type(event)(
            old_hidden_dim=model.head.hidden_dim,
            new_hidden_dim=model.head.hidden_dim,
            split_indices=(),
            pruned_indices=(),
        ), True
    return event, False


def _require_finite_guard_scores(accuracy: float, cross_entropy: float) -> None:
    if not isfinite(accuracy) or not isfinite(cross_entropy):
        raise FloatingPointError("nonfinite guard score")


def _finalize_test_report(
    torch: Any,
    device: Any,
    outcome: _TrainingOutcome,
    test_loader: Any,
    config: ResNet50BenchmarkConfig,
    benchmark_track: str = VISION_UNMATCHED_REFERENCE_TRACK,
) -> ModelSpeedReport:
    """Evaluate a trained model after the learning and selection boundary."""
    model = outcome.model
    if outcome.uses_backprop_metric:
        metric_fn = _compute_backprop_metrics
        forward_logits = model.forward_logits
    else:
        metric_fn = _compute_pc_metrics
        forward_logits = model.predict_logits
    test_accuracy, final_cross_entropy = metric_fn(
        torch, model, test_loader, device, max_batches=None
    )
    inference_metrics = _benchmark_inference(
        torch=torch,
        device=device,
        loader=test_loader,
        forward_logits=forward_logits,
        warmup_batches=config.warmup_batches,
        benchmark_batches=config.inference_batches,
    )
    return ModelSpeedReport(
        model_name=outcome.model_name,
        epochs_ran=outcome.epochs_ran,
        final_metric_name="cross_entropy",
        final_metric_value=final_cross_entropy,
        validation_accuracy=outcome.validation_accuracy,
        test_accuracy=test_accuracy,
        train_seconds=outcome.train_seconds,
        train_samples_per_second=_safe_div(outcome.seen_samples, outcome.train_seconds),
        mean_train_step_ms=(
            float(np.mean(outcome.step_times_ms)) if outcome.step_times_ms else 0.0
        ),
        inference_latency_mean_ms=inference_metrics["latency_mean_ms"],
        inference_latency_p95_ms=inference_metrics["latency_p95_ms"],
        inference_samples_per_second=inference_metrics["samples_per_second"],
        total_parameters=model.parameter_count(),
        trainable_parameters=model.trainable_parameter_count(),
        benchmark_track=benchmark_track,
        backbone_trainable=outcome.uses_backprop_metric and not config.backprop_freeze_backbone,
        backbone_pretraining=config.backbone_weights,
        head_type=_head_type(outcome),
        final_cross_entropy=final_cross_entropy,
        final_energy=outcome.final_energy,
        training_energy_id=(TORCH_PC_ENERGY_ID if outcome.final_energy is not None else None),
        circadian_hidden_dim_start=outcome.circadian_hidden_dim_start,
        circadian_hidden_dim_end=(
            model.head.hidden_dim if outcome.circadian_hidden_dim_start is not None else None
        ),
        circadian_total_splits=outcome.circadian_total_splits,
        circadian_total_prunes=outcome.circadian_total_prunes,
        circadian_total_rollbacks=outcome.circadian_total_rollbacks,
        circadian_sleep_attempts=outcome.circadian_sleep_attempts,
        circadian_sleep_cooldown_suppressions=outcome.circadian_sleep_cooldown_suppressions,
        circadian_sleep_retry_cooldown_epochs=outcome.circadian_sleep_retry_cooldown_epochs,
    )


def _head_type(outcome: _TrainingOutcome) -> str:
    if outcome.uses_backprop_metric:
        return "linear"
    if outcome.circadian_hidden_dim_start is not None:
        return "circadian_predictive_coding"
    return "predictive_coding"


def _finalize_validation_report(
    torch: Any,
    device: Any,
    outcome: _TrainingOutcome,
    validation_loader: Any,
    config: ResNet50BenchmarkConfig,
) -> ModelDevelopmentReport:
    """Measure a tuning candidate using only the validation split."""
    model = outcome.model
    if outcome.uses_backprop_metric:
        metric_fn = _compute_backprop_metrics
        forward_logits = model.forward_logits
    else:
        metric_fn = _compute_pc_metrics
        forward_logits = model.predict_logits
    validation_accuracy, validation_cross_entropy = metric_fn(
        torch, model, validation_loader, device, max_batches=None
    )
    inference_metrics = _benchmark_inference(
        torch=torch,
        device=device,
        loader=validation_loader,
        forward_logits=forward_logits,
        warmup_batches=config.warmup_batches,
        benchmark_batches=config.inference_batches,
    )
    return ModelDevelopmentReport(
        model_name=outcome.model_name,
        epochs_ran=outcome.epochs_ran,
        validation_accuracy=validation_accuracy,
        validation_cross_entropy=validation_cross_entropy,
        train_seconds=outcome.train_seconds,
        train_samples_per_second=_safe_div(outcome.seen_samples, outcome.train_seconds),
        mean_train_step_ms=(
            float(np.mean(outcome.step_times_ms)) if outcome.step_times_ms else 0.0
        ),
        inference_latency_mean_ms=inference_metrics["latency_mean_ms"],
        inference_latency_p95_ms=inference_metrics["latency_p95_ms"],
        inference_samples_per_second=inference_metrics["samples_per_second"],
        total_parameters=model.parameter_count(),
        trainable_parameters=model.trainable_parameter_count(),
        benchmark_track=VISION_UNMATCHED_REFERENCE_TRACK,
        backbone_trainable=outcome.uses_backprop_metric and not config.backprop_freeze_backbone,
        backbone_pretraining=config.backbone_weights,
        head_type=_head_type(outcome),
        final_energy=outcome.final_energy,
        training_energy_id=(TORCH_PC_ENERGY_ID if outcome.final_energy is not None else None),
        circadian_hidden_dim_start=outcome.circadian_hidden_dim_start,
        circadian_hidden_dim_end=(
            model.head.hidden_dim if outcome.circadian_hidden_dim_start is not None else None
        ),
        circadian_total_splits=outcome.circadian_total_splits,
        circadian_total_prunes=outcome.circadian_total_prunes,
        circadian_total_rollbacks=outcome.circadian_total_rollbacks,
        circadian_sleep_attempts=outcome.circadian_sleep_attempts,
        circadian_sleep_cooldown_suppressions=outcome.circadian_sleep_cooldown_suppressions,
        circadian_sleep_retry_cooldown_epochs=outcome.circadian_sleep_retry_cooldown_epochs,
    )


def _benchmark_inference(
    torch: Any,
    device: Any,
    loader: Any,
    forward_logits: Callable[[Any], Any],
    warmup_batches: int,
    benchmark_batches: int,
) -> dict[str, float]:
    if benchmark_batches <= 0:
        raise ValueError("benchmark_batches must be positive.")

    cached_batches = list(loader)
    if not cached_batches:
        raise ValueError("Inference loader is empty.")
    batch_index = 0

    def next_cached_batch() -> tuple[Any, Any]:
        nonlocal batch_index
        batch = cached_batches[batch_index % len(cached_batches)]
        batch_index += 1
        return batch

    latencies_ms: list[float] = []
    seen_samples = 0
    total_time_seconds = 0.0

    with torch.no_grad():
        for _ in range(max(warmup_batches, 0)):
            images, _ = next_cached_batch()
            images = images.to(device)
            _ = forward_logits(images)
            sync_device(torch, device)

        for _ in range(benchmark_batches):
            images, _ = next_cached_batch()
            images = images.to(device)

            sync_device(torch, device)
            start = perf_counter()
            _ = forward_logits(images)
            sync_device(torch, device)
            elapsed = perf_counter() - start

            latencies_ms.append(elapsed * 1000.0)
            total_time_seconds += elapsed
            seen_samples += int(images.shape[0])

    return {
        "latency_mean_ms": float(np.mean(latencies_ms)),
        "latency_p95_ms": float(np.percentile(latencies_ms, 95)),
        "samples_per_second": _safe_div(seen_samples, total_time_seconds),
    }


def _compute_backprop_accuracy(torch: Any, model: Any, loader: Any, device: Any) -> float:
    accuracy, _ = _compute_backprop_metrics(
        torch, model, loader, device=device, max_batches=None
    )
    return accuracy


def _compute_backprop_metrics(
    torch: Any,
    model: Any,
    loader: Any,
    device: Any,
    max_batches: int | None,
) -> tuple[float, float]:
    correct = 0
    total = 0
    total_loss = 0.0
    backbone_training = model.backbone.training
    classifier_training = model.classifier.training
    model.backbone.eval()
    model.classifier.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    try:
        with torch.no_grad():
            for batch_index, (images, labels) in enumerate(loader):
                images = images.to(device)
                labels = labels.to(device)
                logits = model.forward_logits(images)
                batch_loss = criterion(logits, labels)
                predictions = torch.argmax(logits, dim=1)
                correct += int((predictions == labels).sum().item())
                total += int(labels.shape[0])
                total_loss += float(batch_loss.item())
                if max_batches is not None and batch_index + 1 >= max_batches:
                    break
    finally:
        model.backbone.train(backbone_training)
        model.classifier.train(classifier_training)
    return _safe_div(correct, total), _safe_div(total_loss, total)


def _compute_pc_accuracy(torch: Any, model: Any, loader: Any, device: Any) -> float:
    accuracy, _ = _compute_pc_metrics(torch, model, loader, device=device, max_batches=None)
    return accuracy


def _compute_pc_metrics(
    torch: Any,
    model: Any,
    loader: Any,
    device: Any,
    max_batches: int | None,
) -> tuple[float, float]:
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            logits = model.predict_logits(images)
            batch_loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.shape[0])
            total_loss += float(batch_loss.item())
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    return _safe_div(correct, total), _safe_div(total_loss, total)


def _resolve_eval_batches(configured_batches: int) -> int | None:
    if configured_batches <= 0:
        return None
    return configured_batches


def _compute_rollback_delta(
    metric_name: str,
    pre_accuracy: float,
    post_accuracy: float,
    pre_cross_entropy: float,
    post_cross_entropy: float,
) -> float:
    if metric_name == "accuracy":
        return pre_accuracy - post_accuracy
    if metric_name == "cross_entropy":
        return post_cross_entropy - pre_cross_entropy
    raise ValueError(
        "circadian_sleep_rollback_metric must be one of: accuracy, cross_entropy."
    )


def _resolve_device(torch: Any, requested_device: str) -> Any:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def _find_report_by_name(reports: list[ModelSpeedReport], model_name: str) -> ModelSpeedReport:
    for report in reports:
        if report.model_name == model_name:
            return report
    raise ValueError(f"Missing report for model {model_name}.")


def _validate_report_models(reports: list[ModelSpeedReport]) -> None:
    expected = {
        "BackpropResNet50",
        "PredictiveCodingResNet50",
        "CircadianPredictiveCodingResNet50",
    }
    observed = {report.model_name for report in reports}
    if observed != expected:
        raise RuntimeError(
            "Benchmark must compare traditional backprop, traditional predictive coding, "
            f"and circadian predictive coding. Observed={sorted(observed)}"
        )


def _validate_benchmark_config(config: ResNet50BenchmarkConfig) -> None:
    if config.protocol_id not in {
        VISION_VALIDATION_UNMATCHED_PROTOCOL,
        VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL,
        VISION_SEEDED_UNMATCHED_PROTOCOL,
    }:
        raise ValueError(f"Unknown vision benchmark protocol: {config.protocol_id}")
    if config.protocol_id in GUARD_SEPARATED_VISION_PROTOCOLS:
        if config.guard_samples <= 0 or config.dataset_guard_subset_size <= 0:
            raise ValueError("Guard-separated protocol requires positive guard sample counts.")
    if config.dataset_name not in {"synthetic", "cifar10", "cifar100"}:
        raise ValueError("dataset_name must be one of: synthetic, cifar10, cifar100.")
    if config.dataset_num_workers < 0:
        raise ValueError("dataset_num_workers must be non-negative.")
    if config.dataset_train_subset_size < 0:
        raise ValueError("dataset_train_subset_size must be non-negative.")
    if config.dataset_validation_subset_size <= 0:
        raise ValueError("dataset_validation_subset_size must be positive.")
    if config.dataset_test_subset_size < 0:
        raise ValueError("dataset_test_subset_size must be non-negative.")
    if config.validation_samples <= 0:
        raise ValueError("validation_samples must be positive.")
    if config.dataset_name != "synthetic":
        expected_classes = 10 if config.dataset_name == "cifar10" else 100
        if config.num_classes != expected_classes:
            raise ValueError(
                f"num_classes must be {expected_classes} when dataset_name={config.dataset_name}."
            )
    if config.dataset_difficulty not in {"easy", "medium", "hard"}:
        raise ValueError("dataset_difficulty must be one of: easy, medium, hard.")
    if config.dataset_noise_std < 0.0:
        raise ValueError("dataset_noise_std must be non-negative.")
    if config.target_accuracy is not None and not (0.0 <= config.target_accuracy <= 1.0):
        raise ValueError("target_accuracy must be in [0, 1] when provided.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if config.evaluation_batches < 0:
        raise ValueError("evaluation_batches must be non-negative.")
    if config.circadian_sleep_interval < 0:
        raise ValueError("circadian_sleep_interval must be non-negative.")
    if config.circadian_sleep_mode not in {"legacy", "components", "disabled"}:
        raise ValueError("circadian_sleep_mode must be one of: legacy, components, disabled.")
    sleep_switches = (
        "circadian_sleep_enable_chemical_reset",
        "circadian_sleep_enable_homeostasis",
        "circadian_sleep_enable_split",
        "circadian_sleep_enable_prune",
    )
    for field_name in sleep_switches:
        if type(getattr(config, field_name)) is not bool:
            raise ValueError(f"{field_name} must be a bool.")
    if config.circadian_sleep_mode == "legacy" and any(
        not getattr(config, field_name) for field_name in sleep_switches
    ):
        raise ValueError("circadian_sleep_mode='components' is required for component switches.")
    if config.backbone_weights not in {"none", "imagenet"}:
        raise ValueError("backbone_weights must be one of: none, imagenet.")
    if config.circadian_chemical_max_value <= 0.0:
        raise ValueError("circadian_chemical_max_value must be positive.")
    if config.circadian_chemical_saturation_gain <= 0.0:
        raise ValueError("circadian_chemical_saturation_gain must be positive.")
    if config.circadian_plasticity_sensitivity_min <= 0.0:
        raise ValueError("circadian_plasticity_sensitivity_min must be positive.")
    if config.circadian_plasticity_sensitivity_max < config.circadian_plasticity_sensitivity_min:
        raise ValueError(
            "circadian_plasticity_sensitivity_max must be >= circadian_plasticity_sensitivity_min."
        )
    if not (0.0 <= config.circadian_plasticity_importance_mix <= 1.0):
        raise ValueError("circadian_plasticity_importance_mix must be between 0 and 1.")
    if not (0.0 <= config.circadian_reward_baseline_decay < 1.0):
        raise ValueError("circadian_reward_baseline_decay must be in [0, 1).")
    if config.circadian_reward_difficulty_exponent <= 0.0:
        raise ValueError("circadian_reward_difficulty_exponent must be positive.")
    if config.circadian_reward_scale_min <= 0.0:
        raise ValueError("circadian_reward_scale_min must be positive.")
    if config.circadian_reward_scale_max < config.circadian_reward_scale_min:
        raise ValueError("circadian_reward_scale_max must be >= circadian_reward_scale_min.")
    if config.circadian_sleep_warmup_steps < 0:
        raise ValueError("circadian_sleep_warmup_steps must be non-negative.")
    if not (0.0 <= config.circadian_sleep_split_only_until_fraction <= 1.0):
        raise ValueError("circadian_sleep_split_only_until_fraction must be between 0 and 1.")
    if not (0.0 <= config.circadian_sleep_prune_only_after_fraction <= 1.0):
        raise ValueError("circadian_sleep_prune_only_after_fraction must be between 0 and 1.")
    if (
        config.circadian_sleep_split_only_until_fraction
        > config.circadian_sleep_prune_only_after_fraction
    ):
        raise ValueError(
            "circadian_sleep_split_only_until_fraction must be <= "
            "circadian_sleep_prune_only_after_fraction."
        )
    if not (0.0 <= config.circadian_sleep_max_change_fraction <= 1.0):
        raise ValueError("circadian_sleep_max_change_fraction must be between 0 and 1.")
    if config.circadian_sleep_min_change_count < 0:
        raise ValueError("circadian_sleep_min_change_count must be non-negative.")
    if config.circadian_prune_min_age_steps < 0:
        raise ValueError("circadian_prune_min_age_steps must be non-negative.")
    if config.circadian_min_sleep_steps < 0:
        raise ValueError("circadian_min_sleep_steps must be non-negative.")
    if config.circadian_sleep_energy_window < 2:
        raise ValueError("circadian_sleep_energy_window must be at least 2.")
    if config.circadian_sleep_plateau_delta < 0.0:
        raise ValueError("circadian_sleep_plateau_delta must be non-negative.")
    if config.circadian_sleep_chemical_variance_threshold < 0.0:
        raise ValueError("circadian_sleep_chemical_variance_threshold must be non-negative.")
    if config.circadian_adaptive_sleep_budget_min_scale <= 0.0:
        raise ValueError("circadian_adaptive_sleep_budget_min_scale must be positive.")
    if (
        config.circadian_adaptive_sleep_budget_max_scale
        < config.circadian_adaptive_sleep_budget_min_scale
    ):
        raise ValueError(
            "circadian_adaptive_sleep_budget_max_scale must be >= "
            "circadian_adaptive_sleep_budget_min_scale."
        )
    if config.circadian_adaptive_sleep_budget_max_scale > 1.0:
        raise ValueError("circadian_adaptive_sleep_budget_max_scale must be <= 1.0.")
    if config.circadian_adaptive_sleep_budget_plateau_weight < 0.0:
        raise ValueError("circadian_adaptive_sleep_budget_plateau_weight must be non-negative.")
    if config.circadian_adaptive_sleep_budget_variance_weight < 0.0:
        raise ValueError("circadian_adaptive_sleep_budget_variance_weight must be non-negative.")
    if (
        not isfinite(config.circadian_sleep_rollback_tolerance)
        or config.circadian_sleep_rollback_tolerance < 0.0
    ):
        raise ValueError("circadian_sleep_rollback_tolerance must be finite and non-negative.")
    if config.circadian_sleep_rollback_eval_batches < 0:
        raise ValueError("circadian_sleep_rollback_eval_batches must be non-negative.")
    resolve_rollback_cooldown_epochs(
        config.circadian_sleep_mode, config.circadian_sleep_rollback_cooldown_epochs
    )
    if config.circadian_sleep_rollback_metric not in {"accuracy", "cross_entropy"}:
        raise ValueError(
            "circadian_sleep_rollback_metric must be one of: accuracy, cross_entropy."
        )
    if config.circadian_min_hidden_dim > config.circadian_head_hidden_dim:
        raise ValueError(
            "circadian_min_hidden_dim cannot exceed circadian_head_hidden_dim."
        )
    if config.circadian_max_hidden_dim < config.circadian_head_hidden_dim:
        raise ValueError(
            "circadian_max_hidden_dim cannot be lower than circadian_head_hidden_dim."
        )


def _set_seed(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _should_stop_early(target_accuracy: float | None, accuracy: float) -> bool:
    if target_accuracy is None:
        return False
    return accuracy >= target_accuracy


def _safe_div(numerator: float | int, denominator: float | int) -> float:
    if float(denominator) == 0.0:
        return 0.0
    return float(numerator) / float(denominator)
