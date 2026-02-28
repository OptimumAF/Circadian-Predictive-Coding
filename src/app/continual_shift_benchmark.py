"""Continual-learning benchmark with controlled distribution shift.

Why this: circadian sleep is designed to trade off retention and adaptation,
so evaluating phase-A retention after phase-B drift is a direct strength test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from src.infra.datasets import DatasetSplit, generate_two_cluster_dataset_with_transform


@dataclass(frozen=True)
class ContinualShiftConfig:
    """Configuration for the continual shift benchmark."""

    sample_count_phase_a: int = 500
    sample_count_phase_b: int = 500
    test_ratio: float = 0.25
    phase_b_train_fraction: float = 0.14

    phase_a_noise_scale: float = 0.8
    phase_b_noise_scale: float = 1.0
    phase_b_rotation_degrees: float = 40.0
    phase_b_translation_x: float = 0.9
    phase_b_translation_y: float = -0.7

    hidden_dim: int = 12
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


@dataclass(frozen=True)
class ContinualShiftSeedResult:
    """Single-seed output for all three models."""

    seed: int
    backprop: ModelShiftReport
    predictive_coding: ModelShiftReport
    circadian_predictive_coding: CircadianShiftReport


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


def run_continual_shift_benchmark(
    config: ContinualShiftConfig,
    seeds: list[int],
) -> ContinualShiftBenchmarkResult:
    """Run phase-A then phase-B shift training for all three models."""
    _validate_config(config)
    if not seeds:
        raise ValueError("seeds cannot be empty")

    seed_results = [_run_single_seed(config=config, seed=seed) for seed in seeds]
    return ContinualShiftBenchmarkResult(
        config=config,
        seeds=list(seeds),
        seed_results=seed_results,
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
    lines = [
        "Continual Shift Benchmark",
        "-------------------------",
        "Phase A trains on base distribution; phase B trains on shifted/rotated distribution.",
        f"Seeds: {result.seeds}",
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
    return "\n".join(lines)


def _run_single_seed(config: ContinualShiftConfig, seed: int) -> ContinualShiftSeedResult:
    phase_a = generate_two_cluster_dataset_with_transform(
        sample_count=config.sample_count_phase_a,
        noise_scale=config.phase_a_noise_scale,
        seed=seed,
        test_ratio=config.test_ratio,
    )
    phase_b = _build_phase_b_dataset(config=config, seed=seed + 101)

    backprop_model = BackpropMLP(input_dim=2, hidden_dim=config.hidden_dim, seed=seed)
    predictive_coding_model = PredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        seed=seed + 1,
    )
    circadian_model = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        seed=seed + 2,
        circadian_config=config.circadian_config,
    )

    sleep_event_count = 0
    total_splits = 0
    total_prunes = 0
    hidden_dim_start = circadian_model.hidden_dim
    total_epochs = config.phase_a_epochs + config.phase_b_epochs

    for epoch_index in range(1, config.phase_a_epochs + 1):
        backprop_model.train_epoch(
            input_batch=phase_a.train_input,
            target_batch=phase_a.train_target,
            learning_rate=config.backprop_learning_rate,
        )
        predictive_coding_model.train_epoch(
            input_batch=phase_a.train_input,
            target_batch=phase_a.train_target,
            learning_rate=config.pc_learning_rate,
            inference_steps=config.pc_inference_steps,
            inference_learning_rate=config.pc_inference_learning_rate,
        )
        circadian_model.train_epoch(
            input_batch=phase_a.train_input,
            target_batch=phase_a.train_target,
            learning_rate=config.circadian_learning_rate,
            inference_steps=config.circadian_inference_steps,
            inference_learning_rate=config.circadian_inference_learning_rate,
        )
        sleep_event_count, total_splits, total_prunes = _apply_scheduled_sleep(
            model=circadian_model,
            sleep_interval=config.circadian_sleep_interval_phase_a,
            epoch_index=epoch_index,
            global_epoch=epoch_index,
            total_epochs=total_epochs,
            force_sleep=config.circadian_force_sleep,
            sleep_event_count=sleep_event_count,
            total_splits=total_splits,
            total_prunes=total_prunes,
        )

    backprop_pre_a = backprop_model.compute_accuracy(phase_a.test_input, phase_a.test_target)
    predictive_pre_a = predictive_coding_model.compute_accuracy(phase_a.test_input, phase_a.test_target)
    circadian_pre_a = circadian_model.compute_accuracy(phase_a.test_input, phase_a.test_target)

    for epoch_index in range(1, config.phase_b_epochs + 1):
        backprop_model.train_epoch(
            input_batch=phase_b.train_input,
            target_batch=phase_b.train_target,
            learning_rate=config.backprop_learning_rate,
        )
        predictive_coding_model.train_epoch(
            input_batch=phase_b.train_input,
            target_batch=phase_b.train_target,
            learning_rate=config.pc_learning_rate,
            inference_steps=config.pc_inference_steps,
            inference_learning_rate=config.pc_inference_learning_rate,
        )
        circadian_model.train_epoch(
            input_batch=phase_b.train_input,
            target_batch=phase_b.train_target,
            learning_rate=config.circadian_learning_rate,
            inference_steps=config.circadian_inference_steps,
            inference_learning_rate=config.circadian_inference_learning_rate,
        )
        sleep_event_count, total_splits, total_prunes = _apply_scheduled_sleep(
            model=circadian_model,
            sleep_interval=config.circadian_sleep_interval_phase_b,
            epoch_index=epoch_index,
            global_epoch=config.phase_a_epochs + epoch_index,
            total_epochs=total_epochs,
            force_sleep=config.circadian_force_sleep,
            sleep_event_count=sleep_event_count,
            total_splits=total_splits,
            total_prunes=total_prunes,
        )

    backprop_post_a = backprop_model.compute_accuracy(phase_a.test_input, phase_a.test_target)
    predictive_post_a = predictive_coding_model.compute_accuracy(phase_a.test_input, phase_a.test_target)
    circadian_post_a = circadian_model.compute_accuracy(phase_a.test_input, phase_a.test_target)

    backprop_post_b = backprop_model.compute_accuracy(phase_b.test_input, phase_b.test_target)
    predictive_post_b = predictive_coding_model.compute_accuracy(phase_b.test_input, phase_b.test_target)
    circadian_post_b = circadian_model.compute_accuracy(phase_b.test_input, phase_b.test_target)

    return ContinualShiftSeedResult(
        seed=seed,
        backprop=_build_model_report(backprop_pre_a, backprop_post_a, backprop_post_b),
        predictive_coding=_build_model_report(predictive_pre_a, predictive_post_a, predictive_post_b),
        circadian_predictive_coding=CircadianShiftReport(
            phase_a_pre_accuracy=circadian_pre_a,
            phase_a_post_accuracy=circadian_post_a,
            phase_b_post_accuracy=circadian_post_b,
            retention_ratio=_safe_ratio(circadian_post_a, circadian_pre_a),
            balanced_score=0.5 * (circadian_post_a + circadian_post_b),
            sleep_event_count=sleep_event_count,
            total_splits=total_splits,
            total_prunes=total_prunes,
            hidden_dim_start=hidden_dim_start,
            hidden_dim_end=circadian_model.hidden_dim,
        ),
    )


def _build_phase_b_dataset(config: ContinualShiftConfig, seed: int) -> DatasetSplit:
    full_phase_b = generate_two_cluster_dataset_with_transform(
        sample_count=config.sample_count_phase_b,
        noise_scale=config.phase_b_noise_scale,
        seed=seed,
        test_ratio=config.test_ratio,
        rotation_degrees=config.phase_b_rotation_degrees,
        translation=(config.phase_b_translation_x, config.phase_b_translation_y),
    )
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
) -> tuple[int, int, int]:
    if sleep_interval <= 0 or epoch_index % sleep_interval != 0:
        return sleep_event_count, total_splits, total_prunes

    sleep_result = model.sleep_event(
        adaptation_policy=None,
        force_sleep=force_sleep,
        current_step=global_epoch,
        total_steps=total_epochs,
    )
    performed_sleep = (
        len(sleep_result.split_indices) > 0
        or len(sleep_result.pruned_indices) > 0
        or sleep_result.new_hidden_dim != sleep_result.old_hidden_dim
    )
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
        mean_phase_a_pre_accuracy=float(np.mean([report.phase_a_pre_accuracy for report in reports])),
        std_phase_a_pre_accuracy=float(np.std([report.phase_a_pre_accuracy for report in reports])),
        mean_phase_a_post_accuracy=float(np.mean([report.phase_a_post_accuracy for report in reports])),
        std_phase_a_post_accuracy=float(np.std([report.phase_a_post_accuracy for report in reports])),
        mean_phase_b_post_accuracy=float(np.mean([report.phase_b_post_accuracy for report in reports])),
        std_phase_b_post_accuracy=float(np.std([report.phase_b_post_accuracy for report in reports])),
        mean_retention_ratio=float(np.mean([report.retention_ratio for report in reports])),
        std_retention_ratio=float(np.std([report.retention_ratio for report in reports])),
        mean_balanced_score=float(np.mean([report.balanced_score for report in reports])),
        std_balanced_score=float(np.std([report.balanced_score for report in reports])),
    )


def _aggregate_circadian_stats(reports: list[CircadianShiftReport]) -> AggregateCircadianShiftStats:
    return AggregateCircadianShiftStats(
        mean_phase_a_pre_accuracy=float(np.mean([report.phase_a_pre_accuracy for report in reports])),
        std_phase_a_pre_accuracy=float(np.std([report.phase_a_pre_accuracy for report in reports])),
        mean_phase_a_post_accuracy=float(np.mean([report.phase_a_post_accuracy for report in reports])),
        std_phase_a_post_accuracy=float(np.std([report.phase_a_post_accuracy for report in reports])),
        mean_phase_b_post_accuracy=float(np.mean([report.phase_b_post_accuracy for report in reports])),
        std_phase_b_post_accuracy=float(np.std([report.phase_b_post_accuracy for report in reports])),
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
