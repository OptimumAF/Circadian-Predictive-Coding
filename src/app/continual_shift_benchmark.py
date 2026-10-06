"""Continual-learning benchmark with controlled distribution shift.

Why this: circadian sleep is designed to trade off retention and adaptation,
so evaluating phase-A retention after phase-B drift is a direct strength test.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from collections import deque
from time import perf_counter
from typing import Callable

import numpy as np

from src.app.comparison_scope import NumpyComparisonScope, scope_for_hidden_dims
from src.app.circadian_checkpoint import (
    CircadianResumePosition,
    capture_circadian_checkpoint,
    restore_circadian_checkpoint,
)
from src.app.continual_checkpoint import (
    ContinualCheckpointStore,
    ContinualRunnerCheckpoint,
    ContinualRunnerState,
    ContinualUnscoredSeed,
    continual_config_digest,
    continual_data_digest,
    continual_phase_a_data_digest,
    continual_test_digest,
    validate_continual_checkpoint,
    validate_continual_sleep_events,
)
from src.app.numpy_checkpoint_validation import LabeledRole, validate_numpy_baseline_model
from src.app.numpy_sleep_decisions import (
    describe_failed_guarded_numpy_sleep_decision,
    describe_guarded_numpy_sleep_decision,
    describe_unguarded_numpy_sleep_decision,
)
from src.app.sleep_schedule import decide_sleep_attempt
from src.core.backprop_mlp import BackpropMLP, BackpropTrainResult
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianNetworkSnapshot,
    CircadianPredictiveCodingNetwork,
    CircadianTrainResult,
    ReplayRetentionBudget,
    ReplayRetentionSnapshot,
    ReplaySnapshot,
    SleepEventResult,
    replay_sample_id,
)
from src.core.predictive_coding import PredictiveCodingNetwork, PredictiveCodingTrainResult
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.sleep_clocks import SleepEpochProgress
from src.core.sleep_telemetry import SleepEventTelemetry
from src.infra.datasets import (
    DatasetSplit,
    LabeledData,
    RoleSeparatedDataset,
    generate_two_cluster_dataset_with_transform,
    make_role_separated_dataset,
    split_training_validation,
)

CONTINUAL_VALIDATION_PROTOCOL = "continual_validation_v1"
CONTINUAL_LEGACY_PROTOCOL = "continual_legacy_train_test_v0"
CONTINUAL_PHASE_ARRIVAL_PROTOCOL = "continual_phase_arrival_v2"
CONTINUAL_PHASE_LOCAL_SCHEDULE_PROTOCOL = "continual_phase_local_schedule_v3"
CONTINUAL_BOUNDED_REPLAY_PROTOCOL = "continual_bounded_replay_v4"
CONTINUAL_GLOBAL_SEAL_PROTOCOL = "continual_global_test_seal_v5"
CONTINUAL_MODEL_ORDER = ("backprop", "predictive_coding", "circadian_predictive_coding")
_BOUNDED_REPLAY_PROTOCOLS = frozenset(
    {CONTINUAL_BOUNDED_REPLAY_PROTOCOL, CONTINUAL_GLOBAL_SEAL_PROTOCOL}
)
_PHASE_ARRIVAL_CHECKPOINT_FORMATS = {
    CONTINUAL_PHASE_ARRIVAL_PROTOCOL: 2,
    CONTINUAL_PHASE_LOCAL_SCHEDULE_PROTOCOL: 3,
    CONTINUAL_BOUNDED_REPLAY_PROTOCOL: 4,
    CONTINUAL_GLOBAL_SEAL_PROTOCOL: 5,
}
_PHASE_LOCAL_SCHEDULE_PROTOCOLS = frozenset(
    {CONTINUAL_PHASE_LOCAL_SCHEDULE_PROTOCOL, *_BOUNDED_REPLAY_PROTOCOLS}
)


@dataclass(frozen=True)
class ContinualShiftConfig:
    """Configuration for the continual shift benchmark."""

    sample_count_phase_a: int = 500
    sample_count_phase_b: int = 500
    protocol_id: str = CONTINUAL_VALIDATION_PROTOCOL
    validation_fraction: float = 0.20
    test_ratio: float = 0.25
    phase_b_train_fraction: float = 0.14

    phase_a_noise_scale: float = 0.8
    phase_b_noise_scale: float = 1.0
    phase_b_rotation_degrees: float = 40.0
    phase_b_translation_x: float = 0.9
    phase_b_translation_y: float = -0.7

    hidden_dim: int = 12
    hidden_dims: tuple[int, ...] | None = None
    phase_a_epochs: int = 110
    phase_b_epochs: int = 80

    backprop_learning_rate: float = 0.12
    pc_learning_rate: float = 0.05
    pc_inference_steps: int = 25
    pc_inference_learning_rate: float = 0.2

    circadian_learning_rate: float = 0.05
    circadian_inference_steps: int = 25
    circadian_inference_learning_rate: float = 0.2
    circadian_sleep_interval_phase_a: int = 40
    circadian_sleep_interval_phase_b: int = 8
    circadian_force_sleep: bool = True
    circadian_config: CircadianConfig = field(default_factory=CircadianConfig)
    model_order: tuple[str, ...] = CONTINUAL_MODEL_ORDER


@dataclass(frozen=True, kw_only=True)
class ContinualBoundedReplayConfig(ContinualShiftConfig):
    """Opt-in observed-example replay limits without changing v1/v2 config identity."""

    protocol_id: str = CONTINUAL_BOUNDED_REPLAY_PROTOCOL
    replay_max_examples: int
    replay_max_bytes: int


@dataclass(frozen=True, kw_only=True)
class ContinualGlobalSealConfig(ContinualBoundedReplayConfig):
    """Opt-in all-seed final-test release; guard/selection roles remain open."""

    protocol_id: str = CONTINUAL_GLOBAL_SEAL_PROTOCOL


@dataclass(frozen=True)
class ModelShiftReport:
    """Per-model metrics for one seed run."""

    phase_a_pre_accuracy: float
    phase_a_post_accuracy: float
    phase_b_post_accuracy: float
    retention_ratio: float
    balanced_score: float


@dataclass(frozen=True)
class CircadianShiftReport:
    """Circadian metrics plus sleep telemetry for one seed run."""

    phase_a_pre_accuracy: float
    phase_a_post_accuracy: float
    phase_b_post_accuracy: float
    retention_ratio: float
    balanced_score: float
    sleep_event_count: int
    total_splits: int
    total_prunes: int
    hidden_dim_start: int
    hidden_dim_end: int
    sleep_events: tuple[SleepEventTelemetry, ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class ContinualShiftSeedResult:
    """Single-seed output for all three models."""

    seed: int
    backprop: ModelShiftReport
    predictive_coding: ModelShiftReport
    circadian_predictive_coding: CircadianShiftReport
    split_hashes: dict[str, str]
    training_order: tuple[str, ...] = CONTINUAL_MODEL_ORDER


@dataclass(frozen=True)
class ContinualReplayRetentionReport:
    """Declared cap and actual retained content IDs at both phase boundaries."""

    budget_examples: int
    budget_bytes: int
    phase_a: ReplayRetentionSnapshot
    phase_b: ReplayRetentionSnapshot


@dataclass(frozen=True, kw_only=True)
class ContinualBoundedReplaySeedResult(ContinualShiftSeedResult):
    """Versioned replay report attached only to bounded-replay results."""

    replay_retention: ContinualReplayRetentionReport


@dataclass(frozen=True)
class AggregateModelShiftStats:
    """Aggregate metrics for one non-circadian model."""

    mean_phase_a_pre_accuracy: float
    std_phase_a_pre_accuracy: float
    mean_phase_a_post_accuracy: float
    std_phase_a_post_accuracy: float
    mean_phase_b_post_accuracy: float
    std_phase_b_post_accuracy: float
    mean_retention_ratio: float
    std_retention_ratio: float
    mean_balanced_score: float
    std_balanced_score: float


@dataclass(frozen=True)
class AggregateCircadianShiftStats:
    """Aggregate metrics for circadian model including sleep telemetry."""

    mean_phase_a_pre_accuracy: float
    std_phase_a_pre_accuracy: float
    mean_phase_a_post_accuracy: float
    std_phase_a_post_accuracy: float
    mean_phase_b_post_accuracy: float
    std_phase_b_post_accuracy: float
    mean_retention_ratio: float
    std_retention_ratio: float
    mean_balanced_score: float
    std_balanced_score: float
    mean_sleep_event_count: float
    mean_total_splits: float
    mean_total_prunes: float
    mean_hidden_dim_end: float


@dataclass(frozen=True)
class ContinualShiftAggregate:
    """Aggregate output across seeds."""

    run_count: int
    backprop: AggregateModelShiftStats
    predictive_coding: AggregateModelShiftStats
    circadian_predictive_coding: AggregateCircadianShiftStats


@dataclass(frozen=True)
class ContinualShiftBenchmarkResult:
    """Full continual shift benchmark result."""

    config: ContinualShiftConfig
    seeds: list[int]
    seed_results: list[ContinualShiftSeedResult]
    aggregate: ContinualShiftAggregate
    comparison_scope: NumpyComparisonScope


@dataclass
class _ContinualTrainingState:
    """Model states and sleep telemetry passed between train-only phase helpers."""

    backprop_model: BackpropMLP
    predictive_model: PredictiveCodingNetwork
    circadian_model: CircadianPredictiveCodingNetwork
    backprop_after_a: BackpropMLP
    predictive_after_a: PredictiveCodingNetwork
    circadian_after_a: CircadianPredictiveCodingNetwork
    sleep_event_count: int
    total_splits: int
    total_prunes: int
    hidden_dim_start: int


@dataclass(frozen=True)
class _PendingOrdinarySeed:
    """Trained seed held without final-test access until the run is frozen."""

    seed: int
    state: _ContinualTrainingState
    phase_a_test: LabeledRole
    phase_b_test: LabeledRole
    split_hashes: dict[str, str]
    phase_a_roles: RoleSeparatedDataset | None
    phase_b_roles: RoleSeparatedDataset | None
    sleep_events: tuple[SleepEventTelemetry, ...]


@dataclass(frozen=True)
class _CheckpointPhaseAData:
    """Only arrived Phase A roles, with final test sealed from training."""

    phase_a_train: LabeledData
    phase_a_validation: LabeledData | None
    phase_a_test: LabeledRole
    split_hashes: dict[str, str]
    data_digest: str


@dataclass(frozen=True)
class _CheckpointSeedData(_CheckpointPhaseAData):
    """Both development phases, available at or after Phase B arrival."""

    phase_b_train: LabeledData
    phase_b_validation: LabeledData | None
    phase_b_test: LabeledRole


@dataclass(frozen=True)
class _CheckpointTrainingData:
    """Only roles and identity needed by checkpointed training helpers."""

    phase_a_train: LabeledData
    phase_b_train: LabeledData | None
    split_hashes: dict[str, str]
    data_digest: str


def _checkpoint_training_data(data: _CheckpointPhaseAData) -> _CheckpointTrainingData:
    """Exclude final tests from the context passed into training."""
    return _CheckpointTrainingData(
        phase_a_train=data.phase_a_train,
        phase_b_train=data.phase_b_train if isinstance(data, _CheckpointSeedData) else None,
        split_hashes=data.split_hashes,
        data_digest=data.data_digest,
    )


@dataclass
class _CheckpointContext:
    """Immutable run identity and previously committed seed outputs."""

    config: ContinualShiftConfig
    store: ContinualCheckpointStore
    seeds: tuple[int, ...]
    config_digest: str
    seed_index: int
    data: _CheckpointTrainingData
    completed_results: list[ContinualShiftSeedResult]
    completed_data_digests: list[str]
    completed_test_digests: list[str]
    unscored_seeds: list[ContinualUnscoredSeed] = field(default_factory=list)
    sleep_events: list[SleepEventTelemetry] = field(default_factory=list)


def run_continual_shift_benchmark(
    config: ContinualShiftConfig,
    seeds: list[int],
    *,
    checkpoint_store: ContinualCheckpointStore | None = None,
    resume_from_checkpoint: bool = False,
) -> ContinualShiftBenchmarkResult:
    """Run phase-A then phase-B shift training for all three models."""
    _validate_config(config)
    if not seeds:
        raise ValueError("seeds cannot be empty")
    if resume_from_checkpoint and checkpoint_store is None:
        raise ValueError("resume_from_checkpoint requires a checkpoint store")

    if config.protocol_id == CONTINUAL_GLOBAL_SEAL_PROTOCOL:
        if checkpoint_store is None:
            # Why this: no seed may expose final labels while a later seed
            # can still train or make a sleep/replay decision.
            pending = [_train_single_seed(config=config, seed=seed) for seed in seeds]
            seed_results = [_score_pending_seed(config, item) for item in pending]
        else:
            seed_results = _run_checkpointed_global_seeds(
                config, seeds, checkpoint_store, resume_from_checkpoint
            )
    elif checkpoint_store is None:
        seed_results = [_run_single_seed(config=config, seed=seed) for seed in seeds]
    else:
        seed_results = _run_checkpointed_seeds(
            config, seeds, checkpoint_store, resume_from_checkpoint
        )
    return ContinualShiftBenchmarkResult(
        config=config,
        seeds=list(seeds),
        seed_results=seed_results,
        comparison_scope=scope_for_hidden_dims(
            config.hidden_dims if config.hidden_dims is not None else (config.hidden_dim,)
        ),
        aggregate=ContinualShiftAggregate(
            run_count=len(seed_results),
            backprop=_aggregate_model_stats([result.backprop for result in seed_results]),
            predictive_coding=_aggregate_model_stats(
                [result.predictive_coding for result in seed_results]
            ),
            circadian_predictive_coding=_aggregate_circadian_stats(
                [result.circadian_predictive_coding for result in seed_results]
            ),
        ),
    )


def format_continual_shift_benchmark(result: ContinualShiftBenchmarkResult) -> str:
    """Format continual shift output as human-readable text."""
    config = result.config
    split_hashes_by_seed = {run.seed: run.split_hashes for run in result.seed_results}
    lines = [
        "Continual Shift Benchmark",
        "-------------------------",
        f"Protocol: {config.protocol_id}",
        f"Circadian sleep mode: {config.circadian_config.sleep_mode}",
        f"Comparison scope: {result.comparison_scope.scope_id}. {result.comparison_scope.description}",
        (
            "Learning rules: "
            f"Backprop={result.comparison_scope.backprop_algorithm_id}, "
            f"PC={result.comparison_scope.predictive_algorithm_id}, "
            f"Circadian={result.comparison_scope.circadian_algorithm_id}"
        ),
        f"Training order: {config.model_order}",
        f"Split hashes by seed: {split_hashes_by_seed}",
        # Why this: identity transforms are a supported control, so the
        # report must describe the configured source rather than imply drift.
        "Phase A uses the base source; Phase B uses configured noise and transform.",
        f"Seeds: {result.seeds}",
        (
            "Setup: "
            f"hidden_dim={config.hidden_dim}, "
            f"hidden_dims={list(config.hidden_dims) if config.hidden_dims is not None else [config.hidden_dim]}, "
            f"phaseA_epochs={config.phase_a_epochs}, phaseB_epochs={config.phase_b_epochs}, "
            f"phaseA_noise={config.phase_a_noise_scale:.2f}, phaseB_noise={config.phase_b_noise_scale:.2f}"
        ),
        (
            "Phase B transform: "
            f"rotation={config.phase_b_rotation_degrees:.1f} deg, "
            f"translation=({config.phase_b_translation_x:.2f}, {config.phase_b_translation_y:.2f})"
        ),
        f"Phase B train fraction: {config.phase_b_train_fraction:.2f}",
        "",
        (
            "Backprop: "
            + _format_model_stats(
                result.aggregate.backprop.mean_phase_a_pre_accuracy,
                result.aggregate.backprop.std_phase_a_pre_accuracy,
                result.aggregate.backprop.mean_phase_a_post_accuracy,
                result.aggregate.backprop.std_phase_a_post_accuracy,
                result.aggregate.backprop.mean_phase_b_post_accuracy,
                result.aggregate.backprop.std_phase_b_post_accuracy,
                result.aggregate.backprop.mean_retention_ratio,
                result.aggregate.backprop.std_retention_ratio,
                result.aggregate.backprop.mean_balanced_score,
                result.aggregate.backprop.std_balanced_score,
            )
        ),
        (
            "Predictive coding: "
            + _format_model_stats(
                result.aggregate.predictive_coding.mean_phase_a_pre_accuracy,
                result.aggregate.predictive_coding.std_phase_a_pre_accuracy,
                result.aggregate.predictive_coding.mean_phase_a_post_accuracy,
                result.aggregate.predictive_coding.std_phase_a_post_accuracy,
                result.aggregate.predictive_coding.mean_phase_b_post_accuracy,
                result.aggregate.predictive_coding.std_phase_b_post_accuracy,
                result.aggregate.predictive_coding.mean_retention_ratio,
                result.aggregate.predictive_coding.std_retention_ratio,
                result.aggregate.predictive_coding.mean_balanced_score,
                result.aggregate.predictive_coding.std_balanced_score,
            )
        ),
        (
            "Circadian predictive coding: "
            + _format_model_stats(
                result.aggregate.circadian_predictive_coding.mean_phase_a_pre_accuracy,
                result.aggregate.circadian_predictive_coding.std_phase_a_pre_accuracy,
                result.aggregate.circadian_predictive_coding.mean_phase_a_post_accuracy,
                result.aggregate.circadian_predictive_coding.std_phase_a_post_accuracy,
                result.aggregate.circadian_predictive_coding.mean_phase_b_post_accuracy,
                result.aggregate.circadian_predictive_coding.std_phase_b_post_accuracy,
                result.aggregate.circadian_predictive_coding.mean_retention_ratio,
                result.aggregate.circadian_predictive_coding.std_retention_ratio,
                result.aggregate.circadian_predictive_coding.mean_balanced_score,
                result.aggregate.circadian_predictive_coding.std_balanced_score,
            )
            + ", "
            + (
                "sleep_events="
                f"{result.aggregate.circadian_predictive_coding.mean_sleep_event_count:.2f}, "
                f"splits={result.aggregate.circadian_predictive_coding.mean_total_splits:.2f}, "
                f"prunes={result.aggregate.circadian_predictive_coding.mean_total_prunes:.2f}, "
                f"hidden_end={result.aggregate.circadian_predictive_coding.mean_hidden_dim_end:.2f}"
            )
        ),
    ]
    if config.protocol_id in _BOUNDED_REPLAY_PROTOCOLS:
        for seed_result in result.seed_results:
            assert isinstance(seed_result, ContinualBoundedReplaySeedResult)
            retained = seed_result.replay_retention
            lines.append(
                f"Replay retained seed={seed_result.seed}: "
                f"budget={retained.budget_examples} examples/{retained.budget_bytes} bytes; "
                f"A={retained.phase_a.example_count} examples/"
                f"{retained.phase_a.retained_bytes} bytes IDs={retained.phase_a.sample_ids}; "
                f"B={retained.phase_b.example_count} examples/"
                f"{retained.phase_b.retained_bytes} bytes IDs={retained.phase_b.sample_ids}"
            )
    return "\n".join(lines)


def _run_single_seed(config: ContinualShiftConfig, seed: int) -> ContinualShiftSeedResult:
    if config.protocol_id == CONTINUAL_GLOBAL_SEAL_PROTOCOL:
        raise ValueError("global-test-seal v5 must train all seeds before scoring")
    return _score_pending_seed(config, _train_single_seed(config, seed))


def _train_single_seed(config: ContinualShiftConfig, seed: int) -> _PendingOrdinarySeed:
    """Train both phases without opening a deferred final-test source."""
    sleep_events: list[SleepEventTelemetry] = []
    phase_a_source = generate_two_cluster_dataset_with_transform(
        sample_count=config.sample_count_phase_a,
        noise_scale=config.phase_a_noise_scale,
        seed=seed,
        test_ratio=config.test_ratio,
    )
    if config.protocol_id != CONTINUAL_LEGACY_PROTOCOL:
        phase_a_roles = split_training_validation(
            phase_a_source,
            validation_fraction=config.validation_fraction,
            seed=seed + 17,
            hash_test=config.protocol_id not in _BOUNDED_REPLAY_PROTOCOLS,
            defer_test_access=config.protocol_id in _BOUNDED_REPLAY_PROTOCOLS,
        )
        phase_a_train = phase_a_roles.train
        split_hashes = {
            f"phase_a_{role}": digest for role, digest in phase_a_roles.split_hashes.items()
        }
    else:
        phase_a_train = LabeledData(phase_a_source.train_input, phase_a_source.train_target)
        split_hashes = {}

    state = _train_phase_a_models(
        config=config,
        seed=seed,
        phase_a_train=phase_a_train,
        on_sleep_event=sleep_events.append,
    )
    if config.protocol_id != CONTINUAL_LEGACY_PROTOCOL:
        phase_b_roles = _build_phase_b_roles(
            config=config,
            seed=seed + 101,
            hash_test=config.protocol_id not in _BOUNDED_REPLAY_PROTOCOLS,
            defer_test_access=config.protocol_id in _BOUNDED_REPLAY_PROTOCOLS,
        )
        phase_b_train = phase_b_roles.train
        split_hashes.update(
            {f"phase_b_{role}": digest for role, digest in phase_b_roles.split_hashes.items()}
        )
    else:
        phase_b_source = _build_phase_b_dataset(config=config, seed=seed + 101)
        phase_b_train = LabeledData(phase_b_source.train_input, phase_b_source.train_target)

    state = _train_phase_b_models(
        config=config,
        phase_b_train=phase_b_train,
        state=state,
        on_sleep_event=sleep_events.append,
    )
    phase_a_test = (
        phase_a_roles.test
        if config.protocol_id != CONTINUAL_LEGACY_PROTOCOL
        else LabeledData(phase_a_source.test_input, phase_a_source.test_target)
    )
    phase_b_test = (
        phase_b_roles.test
        if config.protocol_id != CONTINUAL_LEGACY_PROTOCOL
        else LabeledData(phase_b_source.test_input, phase_b_source.test_target)
    )
    return _PendingOrdinarySeed(
        seed=seed,
        state=state,
        phase_a_test=phase_a_test,
        phase_b_test=phase_b_test,
        split_hashes=split_hashes,
        phase_a_roles=phase_a_roles if config.protocol_id != CONTINUAL_LEGACY_PROTOCOL else None,
        phase_b_roles=phase_b_roles if config.protocol_id != CONTINUAL_LEGACY_PROTOCOL else None,
        sleep_events=tuple(sleep_events),
    )


def _score_pending_seed(
    config: ContinualShiftConfig, pending: _PendingOrdinarySeed
) -> ContinualShiftSeedResult:
    split_hashes = dict(pending.split_hashes)
    if config.protocol_id in _BOUNDED_REPLAY_PROTOCOLS:
        assert pending.phase_a_roles is not None and pending.phase_b_roles is not None
        split_hashes.update(_final_test_hashes(pending.phase_a_roles, pending.phase_b_roles))
    return _score_seed_models(
        config,
        pending.seed,
        pending.state,
        pending.phase_a_test,
        pending.phase_b_test,
        split_hashes,
        sleep_events=pending.sleep_events,
    )


def _final_test_hashes(
    phase_a_roles: RoleSeparatedDataset,
    phase_b_roles: RoleSeparatedDataset,
) -> dict[str, str]:
    """Bind held-out role content only after both phases finish training."""
    hashes: dict[str, str] = {}
    for name, roles in (("phase_a", phase_a_roles), ("phase_b", phase_b_roles)):
        bound = make_role_separated_dataset(
            train=roles.train,
            validation=roles.validation,
            test=roles.test,
        )
        hashes[f"{name}_test"] = bound.split_hashes["test"]
    return hashes


def _score_seed_models(
    config: ContinualShiftConfig,
    seed: int,
    state: _ContinualTrainingState,
    phase_a_test: LabeledRole,
    phase_b_test: LabeledRole,
    split_hashes: dict[str, str],
    *,
    sleep_events: tuple[SleepEventTelemetry, ...] = (),
) -> ContinualShiftSeedResult:
    """Open both held-out roles only after both training phases complete."""
    backprop_model = state.backprop_model
    predictive_coding_model = state.predictive_model
    circadian_model = state.circadian_model
    backprop_after_a = state.backprop_after_a
    predictive_after_a = state.predictive_after_a
    circadian_after_a = state.circadian_after_a

    # All test scoring happens after every model has finished both phases.
    backprop_pre_a = backprop_after_a.compute_accuracy(phase_a_test.input, phase_a_test.target)
    predictive_pre_a = predictive_after_a.compute_accuracy(phase_a_test.input, phase_a_test.target)
    circadian_pre_a = circadian_after_a.compute_accuracy(phase_a_test.input, phase_a_test.target)

    backprop_post_a = backprop_model.compute_accuracy(phase_a_test.input, phase_a_test.target)
    predictive_post_a = predictive_coding_model.compute_accuracy(
        phase_a_test.input, phase_a_test.target
    )
    circadian_post_a = circadian_model.compute_accuracy(phase_a_test.input, phase_a_test.target)

    backprop_post_b = backprop_model.compute_accuracy(phase_b_test.input, phase_b_test.target)
    predictive_post_b = predictive_coding_model.compute_accuracy(
        phase_b_test.input, phase_b_test.target
    )
    circadian_post_b = circadian_model.compute_accuracy(phase_b_test.input, phase_b_test.target)

    scored = ContinualShiftSeedResult(
        seed=seed,
        split_hashes=split_hashes,
        training_order=config.model_order,
        backprop=_build_model_report(backprop_pre_a, backprop_post_a, backprop_post_b),
        predictive_coding=_build_model_report(
            predictive_pre_a, predictive_post_a, predictive_post_b
        ),
        circadian_predictive_coding=CircadianShiftReport(
            phase_a_pre_accuracy=circadian_pre_a,
            phase_a_post_accuracy=circadian_post_a,
            phase_b_post_accuracy=circadian_post_b,
            retention_ratio=_safe_ratio(circadian_post_a, circadian_pre_a),
            balanced_score=0.5 * (circadian_post_a + circadian_post_b),
            sleep_event_count=state.sleep_event_count,
            total_splits=state.total_splits,
            total_prunes=state.total_prunes,
            hidden_dim_start=state.hidden_dim_start,
            hidden_dim_end=circadian_model.hidden_dim,
            sleep_events=sleep_events,
        ),
    )
    if config.protocol_id not in _BOUNDED_REPLAY_PROTOCOLS:
        return scored
    assert isinstance(config, ContinualBoundedReplayConfig)
    return ContinualBoundedReplaySeedResult(
        seed=scored.seed,
        backprop=scored.backprop,
        predictive_coding=scored.predictive_coding,
        circadian_predictive_coding=scored.circadian_predictive_coding,
        split_hashes=scored.split_hashes,
        training_order=scored.training_order,
        replay_retention=ContinualReplayRetentionReport(
            budget_examples=config.replay_max_examples,
            budget_bytes=config.replay_max_bytes,
            phase_a=circadian_after_a.get_replay_retention(),
            phase_b=circadian_model.get_replay_retention(),
        ),
    )


def _build_checkpoint_phase_a_data(
    config: ContinualShiftConfig, seed: int
) -> _CheckpointPhaseAData:
    """Build only arrived data so Phase A resume cannot inspect Phase B."""
    source = generate_two_cluster_dataset_with_transform(
        sample_count=config.sample_count_phase_a,
        noise_scale=config.phase_a_noise_scale,
        seed=seed,
        test_ratio=config.test_ratio,
    )
    roles = split_training_validation(
        source,
        validation_fraction=config.validation_fraction,
        seed=seed + 17,
        hash_test=False,
        defer_test_access=config.protocol_id in _BOUNDED_REPLAY_PROTOCOLS,
    )
    return _CheckpointPhaseAData(
        phase_a_train=roles.train,
        phase_a_validation=roles.validation,
        phase_a_test=roles.test,
        split_hashes={
            f"phase_a_{name}": roles.split_hashes[name] for name in ("train", "validation")
        },
        data_digest=continual_phase_a_data_digest(roles.train, roles.validation),
    )


def _complete_checkpoint_seed_data(
    config: ContinualShiftConfig, seed: int, phase_a: _CheckpointPhaseAData
) -> _CheckpointSeedData:
    """Bind Phase B only after arrival, then use the existing full digest."""
    phase_b_roles = _build_phase_b_roles(
        config=config,
        seed=seed + 101,
        hash_test=False,
        defer_test_access=config.protocol_id in _BOUNDED_REPLAY_PROTOCOLS,
    )
    return _CheckpointSeedData(
        phase_a_train=phase_a.phase_a_train,
        phase_a_validation=phase_a.phase_a_validation,
        phase_a_test=phase_a.phase_a_test,
        phase_b_train=phase_b_roles.train,
        phase_b_validation=phase_b_roles.validation,
        phase_b_test=phase_b_roles.test,
        split_hashes={
            **phase_a.split_hashes,
            **{f"phase_b_{name}": digest for name, digest in phase_b_roles.split_hashes.items()},
        },
        data_digest=continual_data_digest(
            phase_a.phase_a_train,
            phase_a.phase_a_validation,
            phase_b_roles.train,
            phase_b_roles.validation,
        ),
    )


def _bind_checkpoint_final_test_hashes(data: _CheckpointSeedData) -> _CheckpointSeedData:
    """Bind held-out roles only after both phases finish or for completed seeds."""
    assert data.phase_a_validation is not None and data.phase_b_validation is not None
    phase_a_roles = make_role_separated_dataset(
        train=data.phase_a_train,
        validation=data.phase_a_validation,
        test=LabeledData(data.phase_a_test.input, data.phase_a_test.target),
    )
    phase_b_roles = make_role_separated_dataset(
        train=data.phase_b_train,
        validation=data.phase_b_validation,
        test=LabeledData(data.phase_b_test.input, data.phase_b_test.target),
    )
    return replace(
        data,
        split_hashes={
            **data.split_hashes,
            "phase_a_test": phase_a_roles.split_hashes["test"],
            "phase_b_test": phase_b_roles.split_hashes["test"],
        },
    )


def _build_checkpoint_seed_data(config: ContinualShiftConfig, seed: int) -> _CheckpointSeedData:
    """Regenerate seeded roles without passing final tests to training code."""
    phase_a_source = generate_two_cluster_dataset_with_transform(
        sample_count=config.sample_count_phase_a,
        noise_scale=config.phase_a_noise_scale,
        seed=seed,
        test_ratio=config.test_ratio,
    )
    if config.protocol_id == CONTINUAL_VALIDATION_PROTOCOL:
        phase_a_roles = split_training_validation(
            phase_a_source, validation_fraction=config.validation_fraction, seed=seed + 17
        )
        phase_b_roles = _build_phase_b_roles(config=config, seed=seed + 101)
        phase_a_train, phase_a_validation, phase_a_test = (
            phase_a_roles.train,
            phase_a_roles.validation,
            phase_a_roles.test,
        )
        phase_b_train, phase_b_validation, phase_b_test = (
            phase_b_roles.train,
            phase_b_roles.validation,
            phase_b_roles.test,
        )
        split_hashes = {
            **{f"phase_a_{name}": digest for name, digest in phase_a_roles.split_hashes.items()},
            **{f"phase_b_{name}": digest for name, digest in phase_b_roles.split_hashes.items()},
        }
    else:
        phase_b_source = _build_phase_b_dataset(config=config, seed=seed + 101)
        phase_a_train = LabeledData(phase_a_source.train_input, phase_a_source.train_target)
        phase_a_validation = None
        phase_a_test = LabeledData(phase_a_source.test_input, phase_a_source.test_target)
        phase_b_train = LabeledData(phase_b_source.train_input, phase_b_source.train_target)
        phase_b_validation = None
        phase_b_test = LabeledData(phase_b_source.test_input, phase_b_source.test_target)
        split_hashes = {}
    return _CheckpointSeedData(
        phase_a_train=phase_a_train,
        phase_a_validation=phase_a_validation,
        phase_a_test=phase_a_test,
        phase_b_train=phase_b_train,
        phase_b_validation=phase_b_validation,
        phase_b_test=phase_b_test,
        split_hashes=split_hashes,
        data_digest=continual_data_digest(
            phase_a_train, phase_a_validation, phase_b_train, phase_b_validation
        ),
    )


def _new_checkpoint_models(
    config: ContinualShiftConfig,
    seed: int,
    retention_policy: ReplayRetentionPolicy | None = None,
) -> tuple[ContinualRunnerState, CircadianPredictiveCodingNetwork]:
    hidden_dims = list(config.hidden_dims) if config.hidden_dims is not None else None
    backprop = BackpropMLP(
        input_dim=2, hidden_dim=config.hidden_dim, seed=seed, hidden_dims=hidden_dims
    )
    predictive = PredictiveCodingNetwork(
        input_dim=2, hidden_dim=config.hidden_dim, seed=seed + 1, hidden_dims=hidden_dims
    )
    circadian = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        seed=seed + 2,
        circadian_config=config.circadian_config,
        hidden_dims=hidden_dims,
    )
    _configure_replay_retention(config, circadian, retention_policy)
    return (
        ContinualRunnerState(
            backprop_model=backprop,
            predictive_model=predictive,
            hidden_dim_start=circadian.hidden_dim,
        ),
        circadian,
    )


def _configure_replay_retention(
    config: ContinualShiftConfig,
    model: CircadianPredictiveCodingNetwork,
    retention_policy: ReplayRetentionPolicy | None = None,
) -> None:
    if config.protocol_id in _BOUNDED_REPLAY_PROTOCOLS:
        assert isinstance(config, ContinualBoundedReplayConfig)
        model.configure_replay_retention(
            ReplayRetentionBudget(config.replay_max_examples, config.replay_max_bytes),
            policy=retention_policy,
        )


def _observed_replay_ids(train: LabeledData) -> set[str]:
    return {
        replay_sample_id(train.input[index : index + 1], train.target[index : index + 1])
        for index in range(train.input.shape[0])
    }


def _validate_replay_snapshot_provenance(
    snapshot: CircadianNetworkSnapshot,
    *,
    budget: ReplayRetentionBudget,
    observed_ids: set[str],
) -> None:
    """Reject a saved future example before restore or another wake update."""
    if not isinstance(snapshot, CircadianNetworkSnapshot):
        raise ValueError("incompatible replay observed-example snapshot")
    memory = snapshot.state.get("_replay_memory")
    if (
        snapshot.state.get("_replay_retention_budget") != budget
        or not isinstance(memory, deque)
        or memory.maxlen is not None
    ):
        raise ValueError("incompatible replay observed-example budget")
    retained_ids: set[str] = set()
    retained_bytes = 0
    for item in memory:
        if not isinstance(item, ReplaySnapshot):
            raise ValueError("incompatible replay observed-example snapshot")
        try:
            sample_id = replay_sample_id(item.input_batch, item.target_batch)
        except (AttributeError, ValueError) as error:
            raise ValueError("incompatible replay observed-example row") from error
        if (
            item.input_batch.shape[1] != 2
            or item.input_batch.dtype != np.float64
            or item.target_batch.dtype != np.float64
            or not np.all(np.isfinite(item.input_batch))
            or not np.isin(item.target_batch, (0.0, 1.0)).all()
            or sample_id not in observed_ids
            or sample_id in retained_ids
        ):
            raise ValueError("incompatible replay observed training examples")
        retained_ids.add(sample_id)
        retained_bytes += item.input_batch.nbytes + item.target_batch.nbytes
    if len(retained_ids) > budget.max_examples or retained_bytes > budget.max_bytes:
        raise ValueError("incompatible replay observed-example budget")


def _validate_checkpoint_replay_provenance(
    checkpoint: ContinualRunnerCheckpoint,
    data: _CheckpointPhaseAData,
    config: ContinualBoundedReplayConfig,
) -> None:
    budget = ReplayRetentionBudget(config.replay_max_examples, config.replay_max_bytes)
    phase_a_ids = _observed_replay_ids(data.phase_a_train)
    observed_ids = set(phase_a_ids)
    if isinstance(data, _CheckpointSeedData):
        observed_ids.update(_observed_replay_ids(data.phase_b_train))
    active_snapshot = checkpoint.combined.model_state
    if not isinstance(active_snapshot, CircadianNetworkSnapshot):
        raise ValueError("incompatible replay observed-example snapshot")
    _validate_replay_snapshot_provenance(active_snapshot, budget=budget, observed_ids=observed_ids)
    if checkpoint.phase != "a":
        frozen = checkpoint.state.circadian_after_a
        if frozen is None:
            raise ValueError("incompatible replay observed Phase A state")
        _validate_replay_snapshot_provenance(frozen, budget=budget, observed_ids=phase_a_ids)


def _validate_committed_seed(
    checkpoint: ContinualRunnerCheckpoint,
    index: int,
    data: _CheckpointSeedData,
    config: ContinualShiftConfig,
) -> None:
    """A resumed aggregate cannot reuse a result from changed data."""
    if (
        checkpoint.completed_data_digests[index] != data.data_digest
        or checkpoint.completed_results[index].split_hashes != data.split_hashes
        or checkpoint.completed_test_digests[index]
        != continual_test_digest(data.phase_a_test, data.phase_b_test)
    ):
        raise ValueError("incompatible continual checkpoint completed seed data")
    validate_continual_sleep_events(
        checkpoint.completed_results[index].circadian_predictive_coding.sleep_events,
        config.phase_a_epochs + config.phase_b_epochs,
    )
    if config.protocol_id == CONTINUAL_BOUNDED_REPLAY_PROTOCOL:
        assert isinstance(config, ContinualBoundedReplayConfig)
        result = checkpoint.completed_results[index]
        if not isinstance(result, ContinualBoundedReplaySeedResult):
            raise ValueError("incompatible replay observed completed seed report")
        retained = result.replay_retention
        if not isinstance(retained, ContinualReplayRetentionReport):
            raise ValueError("incompatible replay observed completed seed report")
        if (
            retained.budget_examples != config.replay_max_examples
            or retained.budget_bytes != config.replay_max_bytes
        ):
            raise ValueError("incompatible replay observed completed seed budget")
        phase_a_ids = _observed_replay_ids(data.phase_a_train)
        phase_b_ids = phase_a_ids | _observed_replay_ids(data.phase_b_train)
        for snapshot, observed_ids, role in (
            (retained.phase_a, phase_a_ids, data.phase_a_train),
            (retained.phase_b, phase_b_ids, data.phase_b_train),
        ):
            bytes_per_row = role.input[:1].nbytes + role.target[:1].nbytes
            if (
                not isinstance(snapshot, ReplayRetentionSnapshot)
                or snapshot.example_count != len(snapshot.sample_ids)
                or snapshot.example_count > config.replay_max_examples
                or snapshot.retained_bytes != snapshot.example_count * bytes_per_row
                or snapshot.retained_bytes > config.replay_max_bytes
                or len(set(snapshot.sample_ids)) != snapshot.example_count
                or not set(snapshot.sample_ids).issubset(observed_ids)
            ):
                raise ValueError("incompatible replay observed completed seed report")


def _run_checkpointed_seeds(
    config: ContinualShiftConfig,
    seeds: list[int],
    store: ContinualCheckpointStore,
    resume: bool,
) -> list[ContinualShiftSeedResult]:
    """Continue seed/phase transactions while preserving prior reports."""
    if any(type(seed) is not int for seed in seeds):
        raise ValueError("checkpoint seed list must contain Python integers")
    ordered_seeds = tuple(seeds)
    config_digest = continual_config_digest(config, ordered_seeds)
    checkpoint = store.load() if resume else None
    if checkpoint is not None and (
        not isinstance(checkpoint, ContinualRunnerCheckpoint)
        or checkpoint.runner_config_digest != config_digest
        or checkpoint.seeds != ordered_seeds
        or type(checkpoint.seed_index) is not int
        or not 0 <= checkpoint.seed_index < len(seeds)
        or checkpoint.phase not in {"a", "b", "seed_complete"}
        or not isinstance(checkpoint.completed_results, list)
        or not isinstance(checkpoint.completed_data_digests, tuple)
        or not isinstance(checkpoint.completed_test_digests, tuple)
        or len(checkpoint.completed_results)
        != checkpoint.seed_index + int(checkpoint.phase == "seed_complete")
        or len(checkpoint.completed_data_digests) != len(checkpoint.completed_results)
        or len(checkpoint.completed_test_digests) != len(checkpoint.completed_results)
        or not all(
            isinstance(result, ContinualShiftSeedResult)
            and result.seed == seeds[index]
            and result.training_order == config.model_order
            for index, result in enumerate(checkpoint.completed_results)
        )
    ):
        raise ValueError("incompatible continual checkpoint config or seed list")
    results: list[ContinualShiftSeedResult] = (
        deepcopy(checkpoint.completed_results) if checkpoint is not None else []
    )
    data_digests = list(checkpoint.completed_data_digests) if checkpoint is not None else []
    test_digests = list(checkpoint.completed_test_digests) if checkpoint is not None else []
    for index, seed in enumerate(seeds):
        if config.protocol_id in _PHASE_ARRIVAL_CHECKPOINT_FORMATS:
            phase_a_data = _build_checkpoint_phase_a_data(config, seed)
            data: _CheckpointPhaseAData = phase_a_data
            if checkpoint is not None and (
                index < checkpoint.seed_index
                or (index == checkpoint.seed_index and checkpoint.phase != "a")
            ):
                data = _complete_checkpoint_seed_data(config, seed, phase_a_data)
                if index < checkpoint.seed_index or checkpoint.phase == "seed_complete":
                    data = _bind_checkpoint_final_test_hashes(data)
        else:
            data = _build_checkpoint_seed_data(config, seed)
        if checkpoint is not None and index < checkpoint.seed_index:
            assert isinstance(data, _CheckpointSeedData)
            _validate_committed_seed(checkpoint, index, data, config)
            continue
        if checkpoint is not None and index == checkpoint.seed_index:
            position = validate_continual_checkpoint(
                checkpoint,
                config_digest=config_digest,
                seeds=ordered_seeds,
                data_digest=data.data_digest,
                split_hashes=tuple(sorted(data.split_hashes.items())),
                protocol_id=config.protocol_id,
                model_order=config.model_order,
                phase_a_epochs=config.phase_a_epochs,
                phase_b_epochs=config.phase_b_epochs,
                hidden_dims=config.hidden_dims or (config.hidden_dim,),
                expected_format_version=_PHASE_ARRIVAL_CHECKPOINT_FORMATS.get(
                    config.protocol_id, 1
                ),
            )
            if config.protocol_id == CONTINUAL_BOUNDED_REPLAY_PROTOCOL:
                assert isinstance(config, ContinualBoundedReplayConfig)
                _validate_checkpoint_replay_provenance(checkpoint, data, config)
            if checkpoint.phase == "seed_complete":
                assert isinstance(data, _CheckpointSeedData)
                _validate_committed_seed(checkpoint, index, data, config)
            state, circadian = _new_checkpoint_models(config, seed)
            if checkpoint.phase in {"b", "seed_complete"}:
                frozen = checkpoint.state.circadian_after_a
                assert frozen is not None
                # Why this: validate the saved A-only topology on a detached
                # model before the combined restore changes process RNG.
                candidate = _new_checkpoint_models(config, seed)[1]
                candidate.restore_state(frozen)
                if candidate.get_sleep_clocks().wake_batches != config.phase_a_epochs:
                    raise ValueError("incompatible continual checkpoint phase-A wake state")
            state = deepcopy(checkpoint.state)
            restore_circadian_checkpoint(
                circadian,
                checkpoint.combined,
                retry=None,
                protocol_id=config.protocol_id,
                config=circadian.config,
                data_digest=data.data_digest,
                expected_stage=position.stage,
            )
            if checkpoint.phase == "seed_complete":
                continue
            resume_phase = checkpoint.phase
        else:
            state, circadian = _new_checkpoint_models(config, seed)
            position = None
            resume_phase = None
        context = _CheckpointContext(
            config=config,
            store=store,
            seeds=ordered_seeds,
            config_digest=config_digest,
            seed_index=index,
            data=_checkpoint_training_data(data),
            completed_results=results,
            completed_data_digests=data_digests,
            completed_test_digests=test_digests,
            sleep_events=(
                list(checkpoint.sleep_events)
                if checkpoint is not None and index == checkpoint.seed_index
                else []
            ),
        )
        if resume_phase != "b":
            _train_checkpoint_phase(
                context,
                state,
                circadian,
                phase="a",
                resume_position=position if resume_phase == "a" else None,
            )
            state.backprop_after_a = deepcopy(state.backprop_model)
            state.predictive_after_a = deepcopy(state.predictive_model)
            state.circadian_after_a = circadian.snapshot_state()
            if config.protocol_id in _PHASE_ARRIVAL_CHECKPOINT_FORMATS:
                # Why this: the first B checkpoint must bind the newly arrived
                # B roles, while every A checkpoint remains A-only.
                data = _complete_checkpoint_seed_data(config, seed, data)
                context.data = _checkpoint_training_data(data)
            _save_continual_checkpoint(
                context,
                state,
                circadian,
                phase="b",
                phase_epoch_completed=0,
                stage="after_sleep",
                next_model_index=0,
            )
        _train_checkpoint_phase(
            context,
            state,
            circadian,
            phase="b",
            resume_position=position if resume_phase == "b" else None,
        )
        assert isinstance(data, _CheckpointSeedData)
        if config.protocol_id in _PHASE_ARRIVAL_CHECKPOINT_FORMATS:
            data = _bind_checkpoint_final_test_hashes(data)
            context.data = _checkpoint_training_data(data)
        result = _score_checkpoint_seed(
            config, seed, state, circadian, data, sleep_events=tuple(context.sleep_events)
        )
        results.append(result)
        data_digests.append(data.data_digest)
        test_digests.append(continual_test_digest(data.phase_a_test, data.phase_b_test))
        _save_continual_checkpoint(
            context,
            state,
            circadian,
            phase="seed_complete",
            phase_epoch_completed=config.phase_b_epochs,
            stage="after_sleep",
            next_model_index=0,
        )
    return results


def _capture_unscored_seed(
    seed: int,
    data: _CheckpointSeedData,
    state: ContinualRunnerState,
    circadian: CircadianPredictiveCodingNetwork,
    sleep_events: tuple[SleepEventTelemetry, ...],
) -> ContinualUnscoredSeed:
    """Commit arrived development identity and trained state without test data."""
    if any("test" in role for role in data.split_hashes):
        raise ValueError("unscored seed cannot contain final-test hashes")
    return ContinualUnscoredSeed(
        seed=seed,
        data_digest=data.data_digest,
        split_hashes=tuple(sorted(data.split_hashes.items())),
        state=deepcopy(state),
        circadian_final=circadian.snapshot_state(),
        sleep_events=sleep_events,
    )


def _validate_unscored_seed(
    record: ContinualUnscoredSeed,
    data: _CheckpointSeedData,
    config: ContinualGlobalSealConfig,
    seed: int,
) -> None:
    """Reject changed development roles or saved models before another update."""
    if (
        not isinstance(record, ContinualUnscoredSeed)
        or record.seed != seed
        or record.data_digest != data.data_digest
        or record.split_hashes != tuple(sorted(data.split_hashes.items()))
        or any("test" in role for role, _ in record.split_hashes)
        or not isinstance(record.state, ContinualRunnerState)
    ):
        raise ValueError("incompatible global unscored development data")
    state = record.state
    hidden_dims = config.hidden_dims or (config.hidden_dim,)
    total_epochs = config.phase_a_epochs + config.phase_b_epochs
    if (
        not isinstance(state.backprop_after_a, BackpropMLP)
        or not isinstance(state.predictive_after_a, PredictiveCodingNetwork)
        or not isinstance(state.circadian_after_a, CircadianNetworkSnapshot)
        or not isinstance(record.circadian_final, CircadianNetworkSnapshot)
        or any(
            type(value) is not int or value < 0
            for value in (
                state.sleep_event_count,
                state.total_splits,
                state.total_prunes,
                state.hidden_dim_start,
            )
        )
        or state.sleep_event_count > total_epochs
        or state.hidden_dim_start != hidden_dims[-1]
    ):
        raise ValueError("incompatible global unscored trained state")
    validate_continual_sleep_events(record.sleep_events, total_epochs)
    validate_numpy_baseline_model(state.backprop_model, hidden_dims, total_epochs)
    validate_numpy_baseline_model(state.predictive_model, hidden_dims, total_epochs)
    validate_numpy_baseline_model(state.backprop_after_a, hidden_dims, config.phase_a_epochs)
    validate_numpy_baseline_model(state.predictive_after_a, hidden_dims, config.phase_a_epochs)
    budget = ReplayRetentionBudget(config.replay_max_examples, config.replay_max_bytes)
    phase_a_ids = _observed_replay_ids(data.phase_a_train)
    _validate_replay_snapshot_provenance(
        state.circadian_after_a, budget=budget, observed_ids=phase_a_ids
    )
    _validate_replay_snapshot_provenance(
        record.circadian_final,
        budget=budget,
        observed_ids=phase_a_ids | _observed_replay_ids(data.phase_b_train),
    )
    candidate_a = _new_checkpoint_models(config, seed)[1]
    try:
        candidate_a.restore_state(state.circadian_after_a)
    except (AssertionError, KeyError, TypeError, ValueError) as error:
        raise ValueError("incompatible global unscored Phase A state") from error
    if candidate_a.get_sleep_clocks().wake_batches != config.phase_a_epochs:
        raise ValueError("incompatible global unscored Phase A wake progress")
    candidate = _new_checkpoint_models(config, seed)[1]
    try:
        candidate.restore_state(record.circadian_final)
    except (AssertionError, KeyError, TypeError, ValueError) as error:
        raise ValueError("incompatible global unscored circadian state") from error
    if candidate.get_sleep_clocks().wake_batches != total_epochs:
        raise ValueError("incompatible global unscored wake progress")


def _score_global_unscored_seeds(
    config: ContinualGlobalSealConfig,
    seeds: list[int],
    records: list[ContinualUnscoredSeed],
) -> list[ContinualShiftSeedResult]:
    """Open final tests only after every declared seed has a trained state."""
    if len(records) != len(seeds):
        raise ValueError("global final-test scoring requires every trained seed")
    scored: list[ContinualShiftSeedResult] = []
    for seed, record in zip(seeds, records, strict=True):
        phase_a = _build_checkpoint_phase_a_data(config, seed)
        data = _complete_checkpoint_seed_data(config, seed, phase_a)
        _validate_unscored_seed(record, data, config, seed)
        bound = _bind_checkpoint_final_test_hashes(data)
        circadian = _new_checkpoint_models(config, seed)[1]
        circadian.restore_state(record.circadian_final)
        scored.append(
            _score_checkpoint_seed(
                config,
                seed,
                record.state,
                circadian,
                bound,
                sleep_events=record.sleep_events,
            )
        )
    return scored


def _run_checkpointed_global_seeds(
    config: ContinualShiftConfig,
    seeds: list[int],
    store: ContinualCheckpointStore,
    resume: bool,
) -> list[ContinualShiftSeedResult]:
    """Persist unscored v5 seeds, then release final tests after the last seed."""
    assert isinstance(config, ContinualGlobalSealConfig)
    if any(type(seed) is not int for seed in seeds):
        raise ValueError("checkpoint seed list must contain Python integers")
    ordered_seeds = tuple(seeds)
    config_digest = continual_config_digest(config, ordered_seeds)
    checkpoint = store.load() if resume else None
    if checkpoint is not None and (
        not isinstance(checkpoint, ContinualRunnerCheckpoint)
        or checkpoint.format_version != 5
        or vars(checkpoint).get("sleep_event_history_version") != 1
        or checkpoint.runner_config_digest != config_digest
        or checkpoint.seeds != ordered_seeds
        or type(checkpoint.seed_index) is not int
        or not 0 <= checkpoint.seed_index < len(seeds)
        or checkpoint.phase not in {"a", "b", "seed_complete"}
        or not isinstance(checkpoint.unscored_seeds, tuple)
        or len(checkpoint.unscored_seeds)
        != checkpoint.seed_index + int(checkpoint.phase == "seed_complete")
    ):
        raise ValueError("incompatible global checkpoint config or seed list")
    records = list(deepcopy(checkpoint.unscored_seeds)) if checkpoint is not None else []
    for index, seed in enumerate(seeds):
        phase_a = _build_checkpoint_phase_a_data(config, seed)
        data: _CheckpointPhaseAData = phase_a
        if checkpoint is not None and (
            index < checkpoint.seed_index
            or (index == checkpoint.seed_index and checkpoint.phase != "a")
        ):
            data = _complete_checkpoint_seed_data(config, seed, phase_a)
        if checkpoint is not None and index < checkpoint.seed_index:
            assert isinstance(data, _CheckpointSeedData)
            _validate_unscored_seed(records[index], data, config, seed)
            continue
        if checkpoint is not None and index == checkpoint.seed_index:
            position = validate_continual_checkpoint(
                checkpoint,
                config_digest=config_digest,
                seeds=ordered_seeds,
                data_digest=data.data_digest,
                split_hashes=tuple(sorted(data.split_hashes.items())),
                protocol_id=config.protocol_id,
                model_order=config.model_order,
                phase_a_epochs=config.phase_a_epochs,
                phase_b_epochs=config.phase_b_epochs,
                hidden_dims=config.hidden_dims or (config.hidden_dim,),
                expected_format_version=5,
            )
            _validate_checkpoint_replay_provenance(checkpoint, data, config)
            if checkpoint.phase == "seed_complete":
                assert isinstance(data, _CheckpointSeedData)
                _validate_unscored_seed(records[index], data, config, seed)
                continue
            state, circadian = _new_checkpoint_models(config, seed)
            if checkpoint.phase == "b":
                frozen = checkpoint.state.circadian_after_a
                assert frozen is not None
                candidate = _new_checkpoint_models(config, seed)[1]
                candidate.restore_state(frozen)
                if candidate.get_sleep_clocks().wake_batches != config.phase_a_epochs:
                    raise ValueError("incompatible global checkpoint Phase A wake state")
            state = deepcopy(checkpoint.state)
            restore_circadian_checkpoint(
                circadian,
                checkpoint.combined,
                retry=None,
                protocol_id=config.protocol_id,
                config=circadian.config,
                data_digest=data.data_digest,
                expected_stage=position.stage,
            )
            resume_phase = checkpoint.phase
        else:
            state, circadian = _new_checkpoint_models(config, seed)
            position = None
            resume_phase = None
        context = _CheckpointContext(
            config=config,
            store=store,
            seeds=ordered_seeds,
            config_digest=config_digest,
            seed_index=index,
            data=_checkpoint_training_data(data),
            completed_results=[],
            completed_data_digests=[],
            completed_test_digests=[],
            unscored_seeds=records,
            sleep_events=(
                list(checkpoint.sleep_events)
                if checkpoint is not None and index == checkpoint.seed_index
                else []
            ),
        )
        if resume_phase != "b":
            _train_checkpoint_phase(
                context,
                state,
                circadian,
                phase="a",
                resume_position=position if resume_phase == "a" else None,
            )
            state.backprop_after_a = deepcopy(state.backprop_model)
            state.predictive_after_a = deepcopy(state.predictive_model)
            state.circadian_after_a = circadian.snapshot_state()
            data = _complete_checkpoint_seed_data(config, seed, data)
            context.data = _checkpoint_training_data(data)
            _save_continual_checkpoint(
                context,
                state,
                circadian,
                phase="b",
                phase_epoch_completed=0,
                stage="after_sleep",
                next_model_index=0,
            )
        _train_checkpoint_phase(
            context,
            state,
            circadian,
            phase="b",
            resume_position=position if resume_phase == "b" else None,
        )
        assert isinstance(data, _CheckpointSeedData)
        records.append(
            _capture_unscored_seed(seed, data, state, circadian, tuple(context.sleep_events))
        )
        _save_continual_checkpoint(
            context,
            state,
            circadian,
            phase="seed_complete",
            phase_epoch_completed=config.phase_b_epochs,
            stage="after_sleep",
            next_model_index=0,
        )
    return _score_global_unscored_seeds(config, seeds, records)


def _score_checkpoint_seed(
    config: ContinualShiftConfig,
    seed: int,
    state: ContinualRunnerState,
    circadian: CircadianPredictiveCodingNetwork,
    data: _CheckpointSeedData,
    *,
    sleep_events: tuple[SleepEventTelemetry, ...],
) -> ContinualShiftSeedResult:
    """Rehydrate A-only state solely for post-training held-out scoring."""
    assert state.backprop_after_a is not None
    assert state.predictive_after_a is not None
    assert state.circadian_after_a is not None
    circadian_after_a = _new_checkpoint_models(config, seed)[1]
    circadian_after_a.restore_state(state.circadian_after_a)
    scored = _ContinualTrainingState(
        backprop_model=state.backprop_model,
        predictive_model=state.predictive_model,
        circadian_model=circadian,
        backprop_after_a=state.backprop_after_a,
        predictive_after_a=state.predictive_after_a,
        circadian_after_a=circadian_after_a,
        sleep_event_count=state.sleep_event_count,
        total_splits=state.total_splits,
        total_prunes=state.total_prunes,
        hidden_dim_start=state.hidden_dim_start,
    )
    return _score_seed_models(
        config,
        seed,
        scored,
        data.phase_a_test,
        data.phase_b_test,
        data.split_hashes,
        sleep_events=sleep_events,
    )


def _train_checkpoint_phase(
    context: _CheckpointContext,
    state: ContinualRunnerState,
    circadian: CircadianPredictiveCodingNetwork,
    *,
    phase: str,
    resume_position: CircadianResumePosition | None,
) -> None:
    """Train one role with model-step and sleep transaction boundaries."""
    config = context.config
    if phase == "a":
        epoch_count = config.phase_a_epochs
        offset = 0
        interval = config.circadian_sleep_interval_phase_a
        train = context.data.phase_a_train
    else:
        if context.data.phase_b_train is None:
            raise ValueError("Phase B training requires arrived Phase B roles")
        epoch_count = config.phase_b_epochs
        offset = config.phase_a_epochs
        interval = config.circadian_sleep_interval_phase_b
        train = context.data.phase_b_train
    start_epoch, start_model = 1, 0
    if resume_position is not None:
        local = resume_position.completed_epoch - offset
        if resume_position.stage == "wake":
            start_epoch, start_model = local + 1, resume_position.next_batch_index
        elif resume_position.stage == "before_sleep":
            start_epoch, start_model = local, len(config.model_order)
        else:
            start_epoch = local + 1
    for epoch in range(start_epoch, epoch_count + 1):
        model_start = start_model if epoch == start_epoch else 0
        for model_index in range(model_start, len(config.model_order)):
            _train_named_model_epoch(
                config,
                train,
                state.backprop_model,
                state.predictive_model,
                circadian,
                config.model_order[model_index],
            )
            if model_index + 1 < len(config.model_order):
                _save_continual_checkpoint(
                    context,
                    state,
                    circadian,
                    phase=phase,
                    phase_epoch_completed=epoch - 1,
                    stage="wake",
                    next_model_index=model_index + 1,
                )
        if model_start < len(config.model_order):
            _save_continual_checkpoint(
                context,
                state,
                circadian,
                phase=phase,
                phase_epoch_completed=epoch,
                stage="before_sleep",
                next_model_index=0,
            )
        (
            state.sleep_event_count,
            state.total_splits,
            state.total_prunes,
        ) = _apply_scheduled_sleep(
            model=circadian,
            sleep_interval=interval,
            epoch_index=epoch,
            global_epoch=offset + epoch,
            total_epochs=(
                config.phase_a_epochs
                if phase == "a" and config.protocol_id in _PHASE_LOCAL_SCHEDULE_PROTOCOLS
                else config.phase_a_epochs + config.phase_b_epochs
            ),
            force_sleep=config.circadian_force_sleep,
            sleep_event_count=state.sleep_event_count,
            total_splits=state.total_splits,
            total_prunes=state.total_prunes,
            on_sleep_event=context.sleep_events.append,
        )
        _save_continual_checkpoint(
            context,
            state,
            circadian,
            phase=phase,
            phase_epoch_completed=epoch,
            stage="after_sleep",
            next_model_index=0,
        )


def _save_continual_checkpoint(
    context: _CheckpointContext,
    state: ContinualRunnerState,
    circadian: CircadianPredictiveCodingNetwork,
    *,
    phase: str,
    phase_epoch_completed: int,
    stage: str,
    next_model_index: int,
) -> None:
    global_seal = context.config.protocol_id == CONTINUAL_GLOBAL_SEAL_PROTOCOL
    offset = 0 if phase == "a" else context.config.phase_a_epochs
    position = CircadianResumePosition(
        completed_epoch=offset + phase_epoch_completed,
        stage=stage,
        wake_batches=circadian.get_sleep_clocks().wake_batches,
        next_batch_index=next_model_index,
    )
    context.store.save(
        ContinualRunnerCheckpoint(
            format_version=_PHASE_ARRIVAL_CHECKPOINT_FORMATS.get(context.config.protocol_id, 1),
            runner_config_digest=context.config_digest,
            seeds=context.seeds,
            seed_index=context.seed_index,
            phase=phase,
            phase_epoch_completed=phase_epoch_completed,
            data_digest=context.data.data_digest,
            split_hashes=tuple(sorted(context.data.split_hashes.items())),
            completed_results=[] if global_seal else deepcopy(context.completed_results),
            completed_data_digests=() if global_seal else tuple(context.completed_data_digests),
            completed_test_digests=() if global_seal else tuple(context.completed_test_digests),
            state=deepcopy(state),
            combined=capture_circadian_checkpoint(
                circadian,
                retry=None,
                position=position,
                protocol_id=context.config.protocol_id,
                config=circadian.config,
                data_digest=context.data.data_digest,
            ),
            unscored_seeds=deepcopy(tuple(context.unscored_seeds)) if global_seal else (),
            sleep_event_history_version=1,
            sleep_events=tuple(context.sleep_events),
        )
    )


def _train_phase_a_models(
    *,
    config: ContinualShiftConfig,
    seed: int,
    phase_a_train: LabeledData,
    phase_a_guard: LabeledData | None = None,
    guard_drop_tolerance: float = 0.0,
    on_model_update: Callable[[str, int], None] | None = None,
    on_guard_decision: Callable[[int, float, float, bool, bool], None] | None = None,
    on_sleep_event: Callable[[SleepEventTelemetry], None] | None = None,
    guard_role_hash: str | None = None,
    sleep_error_retries: int = 0,
    retention_policy: ReplayRetentionPolicy | None = None,
) -> _ContinualTrainingState:
    """Train phase A without receiving its final-test role."""
    resolved_hidden_dims = list(config.hidden_dims) if config.hidden_dims is not None else None
    backprop_model = BackpropMLP(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        seed=seed,
        hidden_dims=resolved_hidden_dims,
    )
    predictive_coding_model = PredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        seed=seed + 1,
        hidden_dims=resolved_hidden_dims,
    )
    circadian_model = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        seed=seed + 2,
        circadian_config=config.circadian_config,
        hidden_dims=resolved_hidden_dims,
    )
    _configure_replay_retention(config, circadian_model, retention_policy)

    sleep_event_count = 0
    total_splits = 0
    total_prunes = 0
    hidden_dim_start = circadian_model.hidden_dim
    # Why this: an unarrived phase cannot set Phase A's progress-dependent
    # split/prune budget. Reviewed v1/v2 routes retain their full-run horizon.
    total_epochs = (
        config.phase_a_epochs
        if config.protocol_id in _PHASE_LOCAL_SCHEDULE_PROTOCOLS
        else config.phase_a_epochs + config.phase_b_epochs
    )

    for epoch_index in range(1, config.phase_a_epochs + 1):
        _train_models_one_epoch(
            config,
            phase_a_train,
            backprop_model,
            predictive_coding_model,
            circadian_model,
            epoch_index=epoch_index,
            on_model_update=on_model_update,
        )
        sleep_event_count, total_splits, total_prunes = _run_sleep_with_error_retries(
            lambda event_sink: _apply_scheduled_sleep(
                model=circadian_model,
                sleep_interval=config.circadian_sleep_interval_phase_a,
                epoch_index=epoch_index,
                global_epoch=epoch_index,
                total_epochs=total_epochs,
                force_sleep=config.circadian_force_sleep,
                sleep_event_count=sleep_event_count,
                total_splits=total_splits,
                total_prunes=total_prunes,
                guard=phase_a_guard,
                guard_drop_tolerance=guard_drop_tolerance,
                on_guard_decision=on_guard_decision,
                on_sleep_event=event_sink,
                guard_role_hash=guard_role_hash,
            ),
            on_sleep_event,
            sleep_error_retries,
        )

    # Preserve the A-only state without opening test labels while B can still train.
    backprop_after_a = deepcopy(backprop_model)
    predictive_after_a = deepcopy(predictive_coding_model)
    circadian_after_a = deepcopy(circadian_model)
    return _ContinualTrainingState(
        backprop_model=backprop_model,
        predictive_model=predictive_coding_model,
        circadian_model=circadian_model,
        backprop_after_a=backprop_after_a,
        predictive_after_a=predictive_after_a,
        circadian_after_a=circadian_after_a,
        sleep_event_count=sleep_event_count,
        total_splits=total_splits,
        total_prunes=total_prunes,
        hidden_dim_start=hidden_dim_start,
    )


def _train_phase_b_models(
    *,
    config: ContinualShiftConfig,
    phase_b_train: LabeledData,
    state: _ContinualTrainingState,
    phase_b_guard: LabeledData | None = None,
    guard_drop_tolerance: float = 0.0,
    on_model_update: Callable[[str, int], None] | None = None,
    on_guard_decision: Callable[[int, float, float, bool, bool], None] | None = None,
    on_sleep_event: Callable[[SleepEventTelemetry], None] | None = None,
    guard_role_hash: str | None = None,
    sleep_error_retries: int = 0,
) -> _ContinualTrainingState:
    """Train phase B without receiving either final-test role."""
    backprop_model = state.backprop_model
    predictive_coding_model = state.predictive_model
    circadian_model = state.circadian_model
    sleep_event_count = state.sleep_event_count
    total_splits = state.total_splits
    total_prunes = state.total_prunes
    total_epochs = config.phase_a_epochs + config.phase_b_epochs
    for epoch_index in range(1, config.phase_b_epochs + 1):
        _train_models_one_epoch(
            config,
            phase_b_train,
            backprop_model,
            predictive_coding_model,
            circadian_model,
            epoch_index=epoch_index,
            on_model_update=on_model_update,
        )
        sleep_event_count, total_splits, total_prunes = _run_sleep_with_error_retries(
            lambda event_sink: _apply_scheduled_sleep(
                model=circadian_model,
                sleep_interval=config.circadian_sleep_interval_phase_b,
                epoch_index=epoch_index,
                global_epoch=config.phase_a_epochs + epoch_index,
                total_epochs=total_epochs,
                force_sleep=config.circadian_force_sleep,
                sleep_event_count=sleep_event_count,
                total_splits=total_splits,
                total_prunes=total_prunes,
                guard=phase_b_guard,
                guard_drop_tolerance=guard_drop_tolerance,
                on_guard_decision=on_guard_decision,
                on_sleep_event=event_sink,
                guard_role_hash=guard_role_hash,
            ),
            on_sleep_event,
            sleep_error_retries,
        )

    state.sleep_event_count = sleep_event_count
    state.total_splits = total_splits
    state.total_prunes = total_prunes
    return state


def _train_models_one_epoch(
    config: ContinualShiftConfig,
    train: LabeledData,
    backprop: BackpropMLP,
    predictive: PredictiveCodingNetwork,
    circadian: CircadianPredictiveCodingNetwork,
    *,
    epoch_index: int = 0,
    on_model_update: Callable[[str, int], None] | None = None,
) -> None:
    """Apply one shared phase role in the requested model order."""
    for model_name in config.model_order:
        _train_named_model_epoch(config, train, backprop, predictive, circadian, model_name)
        if on_model_update is not None:
            on_model_update(model_name, epoch_index)


def _train_named_model_epoch(
    config: ContinualShiftConfig,
    train: LabeledData,
    backprop: BackpropMLP,
    predictive: PredictiveCodingNetwork,
    circadian: CircadianPredictiveCodingNetwork,
    model_name: str,
) -> BackpropTrainResult | PredictiveCodingTrainResult | CircadianTrainResult:
    """Apply one model update and return its already-computed diagnostic."""
    if model_name == "backprop":
        return backprop.train_epoch(
            input_batch=train.input,
            target_batch=train.target,
            learning_rate=config.backprop_learning_rate,
        )
    elif model_name == "predictive_coding":
        return predictive.train_epoch(
            input_batch=train.input,
            target_batch=train.target,
            learning_rate=config.pc_learning_rate,
            inference_steps=config.pc_inference_steps,
            inference_learning_rate=config.pc_inference_learning_rate,
        )
    else:
        return circadian.train_epoch(
            input_batch=train.input,
            target_batch=train.target,
            learning_rate=config.circadian_learning_rate,
            inference_steps=config.circadian_inference_steps,
            inference_learning_rate=config.circadian_inference_learning_rate,
        )


def _build_phase_b_dataset(config: ContinualShiftConfig, seed: int) -> DatasetSplit:
    full_phase_b = _generate_phase_b_source(config, seed)
    train_count = full_phase_b.train_input.shape[0]
    subset_count = max(8, int(train_count * config.phase_b_train_fraction))
    rng = np.random.default_rng(seed + 17)
    subset_input, subset_target = _sample_balanced_binary_subset(
        input_batch=full_phase_b.train_input,
        target_batch=full_phase_b.train_target,
        subset_count=subset_count,
        rng=rng,
    )
    return DatasetSplit(
        train_input=subset_input,
        train_target=subset_target,
        test_input=full_phase_b.test_input,
        test_target=full_phase_b.test_target,
    )


def _build_phase_b_roles(
    config: ContinualShiftConfig,
    seed: int,
    *,
    hash_test: bool = True,
    defer_test_access: bool = False,
) -> RoleSeparatedDataset:
    full_phase_b = split_training_validation(
        _generate_phase_b_source(config, seed),
        validation_fraction=config.validation_fraction,
        seed=seed + 37,
        hash_test=hash_test,
        defer_test_access=defer_test_access,
    )
    train_count = full_phase_b.train.input.shape[0]
    subset_count = max(8, int(train_count * config.phase_b_train_fraction))
    subset_input, subset_target = _sample_balanced_binary_subset(
        input_batch=full_phase_b.train.input,
        target_batch=full_phase_b.train.target,
        subset_count=subset_count,
        rng=np.random.default_rng(seed + 17),
    )
    return make_role_separated_dataset(
        train=LabeledData(subset_input, subset_target),
        validation=full_phase_b.validation,
        test=full_phase_b.test,
        hash_test=hash_test,
    )


def _generate_phase_b_source(config: ContinualShiftConfig, seed: int) -> DatasetSplit:
    return generate_two_cluster_dataset_with_transform(
        sample_count=config.sample_count_phase_b,
        noise_scale=config.phase_b_noise_scale,
        seed=seed,
        test_ratio=config.test_ratio,
        rotation_degrees=config.phase_b_rotation_degrees,
        translation=(config.phase_b_translation_x, config.phase_b_translation_y),
    )


def _sample_balanced_binary_subset(
    input_batch: np.ndarray,
    target_batch: np.ndarray,
    subset_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if subset_count <= 0:
        raise ValueError("subset_count must be positive")
    if subset_count >= input_batch.shape[0]:
        return input_batch.copy(), target_batch.copy()

    targets = target_batch.reshape(-1)
    positive_indices = np.where(targets >= 0.5)[0]
    negative_indices = np.where(targets < 0.5)[0]
    if positive_indices.size == 0 or negative_indices.size == 0:
        selected_indices = rng.choice(input_batch.shape[0], size=subset_count, replace=False)
    else:
        half_count = subset_count // 2
        pos_count = min(positive_indices.size, half_count)
        neg_count = min(negative_indices.size, subset_count - pos_count)
        if pos_count + neg_count < subset_count:
            remaining = subset_count - (pos_count + neg_count)
            if positive_indices.size - pos_count >= remaining:
                pos_count += remaining
            else:
                neg_count += remaining

        selected_positive = rng.choice(positive_indices, size=pos_count, replace=False)
        selected_negative = rng.choice(negative_indices, size=neg_count, replace=False)
        selected_indices = np.concatenate([selected_positive, selected_negative])

    rng.shuffle(selected_indices)
    return input_batch[selected_indices], target_batch[selected_indices]


def _run_sleep_with_error_retries(
    attempt_sleep: Callable[[Callable[[SleepEventTelemetry], None] | None], tuple[int, int, int]],
    on_sleep_event: Callable[[SleepEventTelemetry], None] | None,
    retry_limit: int,
) -> tuple[int, int, int]:
    """Retry only a typed, restored sleep error; default behavior still raises."""
    if type(retry_limit) is not int or retry_limit < 0:
        raise ValueError("sleep error retries must be a nonnegative integer")
    if retry_limit == 0:
        return attempt_sleep(on_sleep_event)
    if on_sleep_event is None:
        raise ValueError("sleep error retries require typed event capture")
    for attempt_index in range(retry_limit + 1):
        error_recorded = False

        def record_event(event: SleepEventTelemetry) -> None:
            nonlocal error_recorded
            on_sleep_event(event)
            error_recorded = event.outcome == "error"

        try:
            return attempt_sleep(record_event)
        except Exception:
            # Why this: a guard rejection is a valid decision, while only a
            # recorded atomic error can be retried without repeating wake work.
            if not error_recorded or attempt_index == retry_limit:
                raise
    raise AssertionError("sleep retry loop did not return or raise")


def _apply_scheduled_sleep(
    model: CircadianPredictiveCodingNetwork,
    sleep_interval: int,
    epoch_index: int,
    global_epoch: int,
    total_epochs: int,
    force_sleep: bool,
    sleep_event_count: int,
    total_splits: int,
    total_prunes: int,
    guard: LabeledData | None = None,
    guard_drop_tolerance: float = 0.0,
    on_guard_decision: Callable[[int, float, float, bool, bool], None] | None = None,
    on_sleep_event: Callable[[SleepEventTelemetry], None] | None = None,
    guard_role_hash: str | None = None,
) -> tuple[int, int, int]:
    # Why this: corrected component runs can attempt adaptive sleep on any
    # epoch; legacy runs retain their phase-local interval-only schedule.
    adaptive_due = model.config.sleep_mode == "components" and model.should_trigger_sleep()
    decision = decide_sleep_attempt(
        sleep_mode=model.config.sleep_mode,
        completed_epochs=epoch_index,
        interval_epochs=sleep_interval,
        adaptive_due=adaptive_due,
        force_periodic=force_sleep,
    )
    if guard is not None and on_sleep_event is not None and guard_role_hash is None:
        raise ValueError("guarded sleep event capture requires its role hash")
    if not decision.attempted:
        if on_sleep_event is not None:
            on_sleep_event(
                describe_unguarded_numpy_sleep_decision(
                    model, decision, completed_epoch=global_epoch
                )
            )
        return sleep_event_count, total_splits, total_prunes

    attempt_started_at = perf_counter()
    saved = model.snapshot_state() if guard is not None else None

    def record_error(
        reason: str,
        *,
        before: float | None,
        examples_scored: int,
        result: SleepEventResult | None = None,
    ) -> None:
        if on_sleep_event is None or guard is None or guard_role_hash is None:
            return
        on_sleep_event(
            describe_failed_guarded_numpy_sleep_decision(
                model,
                decision,
                completed_epoch=global_epoch,
                role_hash=guard_role_hash,
                accuracy_before=before,
                examples_scored=examples_scored,
                tolerance=guard_drop_tolerance,
                reason=reason,
                attempt_seconds=perf_counter() - attempt_started_at,
                result=result,
            )
        )

    try:
        before_guard = (
            model.compute_accuracy(guard.input, guard.target) if guard is not None else None
        )
    except Exception:
        if saved is not None:
            model.restore_state(saved)
        record_error("inner_guard_pre_exception", before=None, examples_scored=0)
        raise
    if before_guard is not None and not np.isfinite(before_guard):
        assert saved is not None
        model.restore_state(saved)
        record_error("inner_guard_pre_nonfinite", before=None, examples_scored=0)
        raise ValueError("inner guard accuracy must be finite before sleep")
    try:
        sleep_result = model.sleep_event(
            adaptation_policy=None,
            force_sleep=decision.force_sleep,
            epoch_progress=SleepEpochProgress(global_epoch, total_epochs),
        )
    except Exception:
        if saved is not None:
            model.restore_state(saved)
        record_error(
            "sleep_core_exception",
            before=before_guard,
            examples_scored=len(guard.input) if guard is not None else 0,
        )
        raise
    try:
        after_guard = (
            model.compute_accuracy(guard.input, guard.target) if guard is not None else None
        )
    except Exception:
        if saved is not None:
            model.restore_state(saved)
        record_error(
            "inner_guard_post_exception",
            before=before_guard,
            examples_scored=len(guard.input) if guard is not None else 0,
            result=sleep_result,
        )
        raise
    if after_guard is not None and not np.isfinite(after_guard):
        assert saved is not None
        model.restore_state(saved)
        record_error(
            "inner_guard_post_nonfinite",
            before=before_guard,
            examples_scored=len(guard.input) if guard is not None else 0,
            result=sleep_result,
        )
        raise ValueError("inner guard accuracy must be finite after sleep")
    if on_sleep_event is not None and guard is None:
        on_sleep_event(
            describe_unguarded_numpy_sleep_decision(
                model,
                decision,
                completed_epoch=global_epoch,
                result=sleep_result,
                attempt_seconds=perf_counter() - attempt_started_at,
            )
        )
    # Why this: retain historical legacy counts while recognizing corrected
    # consolidation-only events without conflating them with topology changes.
    if model.config.sleep_mode == "legacy":
        performed_sleep = (
            len(sleep_result.split_indices) > 0
            or len(sleep_result.pruned_indices) > 0
            or sleep_result.new_hidden_dim != sleep_result.old_hidden_dim
        )
    else:
        performed_sleep = sleep_result.performed
    if before_guard is not None and after_guard is not None:
        accepted = after_guard + guard_drop_tolerance >= before_guard
        if not accepted:
            assert saved is not None
            model.restore_state(saved)
        if on_guard_decision is not None:
            on_guard_decision(epoch_index, before_guard, after_guard, performed_sleep, accepted)
        if on_sleep_event is not None:
            assert guard is not None and guard_role_hash is not None
            on_sleep_event(
                describe_guarded_numpy_sleep_decision(
                    decision,
                    sleep_result,
                    completed_epoch=global_epoch,
                    guard_role_hash=guard_role_hash,
                    accuracy_before=before_guard,
                    accuracy_after=after_guard,
                    tolerance=guard_drop_tolerance,
                    guard_examples=len(guard.input),
                    accepted=accepted,
                    attempt_seconds=perf_counter() - attempt_started_at,
                    model=model,
                )
            )
        if not accepted:
            return sleep_event_count, total_splits, total_prunes
    if not performed_sleep:
        return sleep_event_count, total_splits, total_prunes
    return (
        sleep_event_count + 1,
        total_splits + len(sleep_result.split_indices),
        total_prunes + len(sleep_result.pruned_indices),
    )


def _build_model_report(
    phase_a_pre_accuracy: float,
    phase_a_post_accuracy: float,
    phase_b_post_accuracy: float,
) -> ModelShiftReport:
    return ModelShiftReport(
        phase_a_pre_accuracy=phase_a_pre_accuracy,
        phase_a_post_accuracy=phase_a_post_accuracy,
        phase_b_post_accuracy=phase_b_post_accuracy,
        retention_ratio=_safe_ratio(phase_a_post_accuracy, phase_a_pre_accuracy),
        balanced_score=0.5 * (phase_a_post_accuracy + phase_b_post_accuracy),
    )


def _aggregate_model_stats(reports: list[ModelShiftReport]) -> AggregateModelShiftStats:
    return AggregateModelShiftStats(
        mean_phase_a_pre_accuracy=float(
            np.mean([report.phase_a_pre_accuracy for report in reports])
        ),
        std_phase_a_pre_accuracy=float(np.std([report.phase_a_pre_accuracy for report in reports])),
        mean_phase_a_post_accuracy=float(
            np.mean([report.phase_a_post_accuracy for report in reports])
        ),
        std_phase_a_post_accuracy=float(
            np.std([report.phase_a_post_accuracy for report in reports])
        ),
        mean_phase_b_post_accuracy=float(
            np.mean([report.phase_b_post_accuracy for report in reports])
        ),
        std_phase_b_post_accuracy=float(
            np.std([report.phase_b_post_accuracy for report in reports])
        ),
        mean_retention_ratio=float(np.mean([report.retention_ratio for report in reports])),
        std_retention_ratio=float(np.std([report.retention_ratio for report in reports])),
        mean_balanced_score=float(np.mean([report.balanced_score for report in reports])),
        std_balanced_score=float(np.std([report.balanced_score for report in reports])),
    )


def _aggregate_circadian_stats(reports: list[CircadianShiftReport]) -> AggregateCircadianShiftStats:
    return AggregateCircadianShiftStats(
        mean_phase_a_pre_accuracy=float(
            np.mean([report.phase_a_pre_accuracy for report in reports])
        ),
        std_phase_a_pre_accuracy=float(np.std([report.phase_a_pre_accuracy for report in reports])),
        mean_phase_a_post_accuracy=float(
            np.mean([report.phase_a_post_accuracy for report in reports])
        ),
        std_phase_a_post_accuracy=float(
            np.std([report.phase_a_post_accuracy for report in reports])
        ),
        mean_phase_b_post_accuracy=float(
            np.mean([report.phase_b_post_accuracy for report in reports])
        ),
        std_phase_b_post_accuracy=float(
            np.std([report.phase_b_post_accuracy for report in reports])
        ),
        mean_retention_ratio=float(np.mean([report.retention_ratio for report in reports])),
        std_retention_ratio=float(np.std([report.retention_ratio for report in reports])),
        mean_balanced_score=float(np.mean([report.balanced_score for report in reports])),
        std_balanced_score=float(np.std([report.balanced_score for report in reports])),
        mean_sleep_event_count=float(np.mean([report.sleep_event_count for report in reports])),
        mean_total_splits=float(np.mean([report.total_splits for report in reports])),
        mean_total_prunes=float(np.mean([report.total_prunes for report in reports])),
        mean_hidden_dim_end=float(np.mean([report.hidden_dim_end for report in reports])),
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1e-8:
        return 0.0
    return numerator / denominator


def _format_model_stats(
    mean_pre_a: float,
    std_pre_a: float,
    mean_post_a: float,
    std_post_a: float,
    mean_post_b: float,
    std_post_b: float,
    mean_retention: float,
    std_retention: float,
    mean_balanced: float,
    std_balanced: float,
) -> str:
    return (
        f"A_pre={mean_pre_a:.3f}+/-{std_pre_a:.3f}, "
        f"A_post={mean_post_a:.3f}+/-{std_post_a:.3f}, "
        f"B_post={mean_post_b:.3f}+/-{std_post_b:.3f}, "
        f"retention={mean_retention:.3f}+/-{std_retention:.3f}, "
        f"balanced={mean_balanced:.3f}+/-{std_balanced:.3f}"
    )


def _validate_config(config: ContinualShiftConfig) -> None:
    if (
        type(config.model_order) is not tuple
        or len(config.model_order) != len(CONTINUAL_MODEL_ORDER)
        or any(type(name) is not str for name in config.model_order)
        or set(config.model_order) != set(CONTINUAL_MODEL_ORDER)
    ):
        raise ValueError("model_order must be a permutation of the three continual models")
    if config.protocol_id not in (
        CONTINUAL_VALIDATION_PROTOCOL,
        CONTINUAL_LEGACY_PROTOCOL,
        CONTINUAL_PHASE_ARRIVAL_PROTOCOL,
        CONTINUAL_PHASE_LOCAL_SCHEDULE_PROTOCOL,
        CONTINUAL_BOUNDED_REPLAY_PROTOCOL,
        CONTINUAL_GLOBAL_SEAL_PROTOCOL,
    ):
        raise ValueError(f"Unknown continual benchmark protocol: {config.protocol_id}")
    if config.protocol_id in _BOUNDED_REPLAY_PROTOCOLS:
        expected = (
            ContinualGlobalSealConfig
            if config.protocol_id == CONTINUAL_GLOBAL_SEAL_PROTOCOL
            else ContinualBoundedReplayConfig
        )
        if type(config) is not expected:
            if config.protocol_id == CONTINUAL_BOUNDED_REPLAY_PROTOCOL:
                raise ValueError("bounded replay config is required for the v4 protocol")
            raise ValueError("global-test-seal config is required for the v5 protocol")
        assert isinstance(config, ContinualBoundedReplayConfig)
        if (
            type(config.replay_max_examples) is not int
            or config.replay_max_examples <= 0
            or type(config.replay_max_bytes) is not int
            or config.replay_max_bytes < 24
        ):
            raise ValueError("replay example/byte budget must be positive and fit one row")
        if config.circadian_config.replay_steps <= 0:
            raise ValueError("bounded replay requires replay_steps > 0")
        if config.circadian_config.sleep_mode != "components":
            raise ValueError("bounded replay requires components sleep mode")
    elif isinstance(config, ContinualBoundedReplayConfig):
        raise ValueError("bounded replay config requires its matching protocol")
    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if config.sample_count_phase_a < 20:
        raise ValueError("sample_count_phase_a must be at least 20")
    if config.sample_count_phase_b < 20:
        raise ValueError("sample_count_phase_b must be at least 20")
    if config.test_ratio <= 0.0 or config.test_ratio >= 0.5:
        raise ValueError("test_ratio must be between 0 and 0.5")
    if config.phase_b_train_fraction <= 0.0 or config.phase_b_train_fraction > 1.0:
        raise ValueError("phase_b_train_fraction must be in (0, 1]")
    if config.phase_a_noise_scale <= 0.0 or config.phase_b_noise_scale <= 0.0:
        raise ValueError("phase noise scales must be positive")
    if config.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if config.hidden_dims is not None:
        if len(config.hidden_dims) == 0:
            raise ValueError("hidden_dims cannot be empty")
        if any(hidden <= 0 for hidden in config.hidden_dims):
            raise ValueError("all hidden_dims values must be positive")
        if config.hidden_dim != config.hidden_dims[-1]:
            raise ValueError("hidden_dim must match the last value in hidden_dims")
    if config.phase_a_epochs <= 0 or config.phase_b_epochs <= 0:
        raise ValueError("phase epochs must be positive")
    if config.backprop_learning_rate <= 0.0:
        raise ValueError("backprop_learning_rate must be positive")
    if config.pc_learning_rate <= 0.0:
        raise ValueError("pc_learning_rate must be positive")
    if config.pc_inference_steps <= 0:
        raise ValueError("pc_inference_steps must be positive")
    if config.pc_inference_learning_rate <= 0.0:
        raise ValueError("pc_inference_learning_rate must be positive")
    if config.circadian_learning_rate <= 0.0:
        raise ValueError("circadian_learning_rate must be positive")
    if config.circadian_inference_steps <= 0:
        raise ValueError("circadian_inference_steps must be positive")
    if config.circadian_inference_learning_rate <= 0.0:
        raise ValueError("circadian_inference_learning_rate must be positive")
    if config.circadian_sleep_interval_phase_a < 0:
        raise ValueError("circadian_sleep_interval_phase_a must be non-negative")
    if config.circadian_sleep_interval_phase_b < 0:
        raise ValueError("circadian_sleep_interval_phase_b must be non-negative")
