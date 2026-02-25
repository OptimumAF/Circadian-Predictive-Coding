"""In-depth comparison use case across seeds and data difficulty settings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.app.experiment_runner import ExperimentConfig, ExperimentResult, run_experiment


@dataclass(frozen=True)
class AggregateModelStats:
    """Aggregate statistics for one model in a scenario."""

    mean_test_accuracy: float
    std_test_accuracy: float
    mean_final_metric: float
    std_final_metric: float
    mean_epoch_80pct_progress: float
    std_epoch_80pct_progress: float


@dataclass(frozen=True)
class AggregateCircadianStats:
    """Aggregate statistics for circadian model plus sleep dynamics."""

    mean_test_accuracy: float
    std_test_accuracy: float
    mean_final_metric: float
    std_final_metric: float
    mean_epoch_80pct_progress: float
    std_epoch_80pct_progress: float
    mean_sleep_splits: float
    mean_sleep_prunes: float
    mean_hidden_dim_end: float


@dataclass(frozen=True)
class ScenarioComparison:
    """Aggregated comparison report at a specific noise level."""

    noise_scale: float
    run_count: int
    backprop: AggregateModelStats
    predictive_coding: AggregateModelStats
    circadian_predictive_coding: AggregateCircadianStats


@dataclass(frozen=True)
class InDepthComparisonResult:
    """In-depth comparison output across scenarios."""

    seeds: list[int]
    scenario_reports: list[ScenarioComparison]


def run_indepth_comparison(
    base_config: ExperimentConfig,
    seeds: list[int],
    noise_levels: list[float],
) -> InDepthComparisonResult:
    """Run repeated experiments and aggregate behavior of all three models."""
    if not seeds:
        raise ValueError("seeds cannot be empty")
    if not noise_levels:
        raise ValueError("noise_levels cannot be empty")
    for noise in noise_levels:
        if noise <= 0.0:
            raise ValueError("noise levels must be positive")

    scenario_reports: list[ScenarioComparison] = []
    for noise in noise_levels:
        run_results = _run_scenario(base_config=base_config, seeds=seeds, noise_scale=noise)
        scenario_reports.append(_aggregate_scenario(noise_scale=noise, run_results=run_results))

    return InDepthComparisonResult(seeds=list(seeds), scenario_reports=scenario_reports)


def format_indepth_comparison_result(result: InDepthComparisonResult) -> str:
    """Format in-depth aggregate results as human-readable text."""
    lines = [
        "In-Depth Model Comparison",
        "-------------------------",
        "Models: Backprop, Predictive Coding, Circadian Predictive Coding",
        f"Seeds: {result.seeds}",
        "",
    ]
    for scenario in result.scenario_reports:
        lines.extend(
            [
                f"Scenario noise={scenario.noise_scale:.2f}, runs={scenario.run_count}",
                (
                    "Backprop: "
                    + _format_aggregate_line(
                        scenario.backprop.mean_test_accuracy,
                        scenario.backprop.std_test_accuracy,
                        scenario.backprop.mean_final_metric,
                        scenario.backprop.std_final_metric,
                        scenario.backprop.mean_epoch_80pct_progress,
                        scenario.backprop.std_epoch_80pct_progress,
                    )
                ),
                (
                    "Predictive coding: "
                    + _format_aggregate_line(
                        scenario.predictive_coding.mean_test_accuracy,
                        scenario.predictive_coding.std_test_accuracy,
                        scenario.predictive_coding.mean_final_metric,
                        scenario.predictive_coding.std_final_metric,
                        scenario.predictive_coding.mean_epoch_80pct_progress,
                        scenario.predictive_coding.std_epoch_80pct_progress,
                    )
                ),
                (
                    "Circadian predictive coding: "
                    + _format_aggregate_line(
                        scenario.circadian_predictive_coding.mean_test_accuracy,
                        scenario.circadian_predictive_coding.std_test_accuracy,
                        scenario.circadian_predictive_coding.mean_final_metric,
                        scenario.circadian_predictive_coding.std_final_metric,
                        scenario.circadian_predictive_coding.mean_epoch_80pct_progress,
                        scenario.circadian_predictive_coding.std_epoch_80pct_progress,
                    )
                    + ", "
                    + (
                        "sleep splits="
                        f"{scenario.circadian_predictive_coding.mean_sleep_splits:.2f}, "
                        f"sleep prunes={scenario.circadian_predictive_coding.mean_sleep_prunes:.2f}, "
                        f"hidden_end={scenario.circadian_predictive_coding.mean_hidden_dim_end:.2f}"
                    )
                ),
                "",
            ]
        )
    return "\n".join(lines).strip()


def _run_scenario(
    base_config: ExperimentConfig,
    seeds: list[int],
    noise_scale: float,
) -> list[ExperimentResult]:
    scenario_results: list[ExperimentResult] = []
    for seed in seeds:
        scenario_config = ExperimentConfig(
            sample_count=base_config.sample_count,
            noise_scale=noise_scale,
            hidden_dim=base_config.hidden_dim,
            epoch_count=base_config.epoch_count,
            backprop_learning_rate=base_config.backprop_learning_rate,
            pc_learning_rate=base_config.pc_learning_rate,
            pc_inference_steps=base_config.pc_inference_steps,
            pc_inference_learning_rate=base_config.pc_inference_learning_rate,
            circadian_learning_rate=base_config.circadian_learning_rate,
            circadian_inference_steps=base_config.circadian_inference_steps,
            circadian_inference_learning_rate=base_config.circadian_inference_learning_rate,
            circadian_sleep_interval=base_config.circadian_sleep_interval,
            random_seed=seed,
        )
        scenario_results.append(run_experiment(config=scenario_config))
    return scenario_results


def _aggregate_scenario(noise_scale: float, run_results: list[ExperimentResult]) -> ScenarioComparison:
    backprop_accuracies = [run.backprop.test_accuracy for run in run_results]
    predictive_accuracies = [run.predictive_coding.test_accuracy for run in run_results]
    circadian_accuracies = [run.circadian_predictive_coding.test_accuracy for run in run_results]

    backprop_finals = [run.backprop.loss_history[-1] for run in run_results]
    predictive_finals = [run.predictive_coding.loss_history[-1] for run in run_results]
    circadian_finals = [run.circadian_predictive_coding.loss_history[-1] for run in run_results]

    backprop_progress_epochs = [
        _epoch_for_80pct_progress(run.backprop.loss_history) for run in run_results
    ]
    predictive_progress_epochs = [
        _epoch_for_80pct_progress(run.predictive_coding.loss_history) for run in run_results
    ]
    circadian_progress_epochs = [
        _epoch_for_80pct_progress(run.circadian_predictive_coding.loss_history) for run in run_results
    ]

    circadian_splits = [float(run.circadian_sleep.total_splits) for run in run_results]
    circadian_prunes = [float(run.circadian_sleep.total_prunes) for run in run_results]
    circadian_hidden_ends = [float(run.circadian_sleep.hidden_dim_end) for run in run_results]

    return ScenarioComparison(
        noise_scale=noise_scale,
        run_count=len(run_results),
        backprop=_aggregate_model(
            accuracies=backprop_accuracies,
            final_metrics=backprop_finals,
            progress_epochs=backprop_progress_epochs,
        ),
        predictive_coding=_aggregate_model(
            accuracies=predictive_accuracies,
            final_metrics=predictive_finals,
            progress_epochs=predictive_progress_epochs,
        ),
        circadian_predictive_coding=AggregateCircadianStats(
            mean_test_accuracy=float(np.mean(circadian_accuracies)),
            std_test_accuracy=float(np.std(circadian_accuracies)),
            mean_final_metric=float(np.mean(circadian_finals)),
            std_final_metric=float(np.std(circadian_finals)),
            mean_epoch_80pct_progress=float(np.mean(circadian_progress_epochs)),
            std_epoch_80pct_progress=float(np.std(circadian_progress_epochs)),
            mean_sleep_splits=float(np.mean(circadian_splits)),
            mean_sleep_prunes=float(np.mean(circadian_prunes)),
            mean_hidden_dim_end=float(np.mean(circadian_hidden_ends)),
        ),
    )


def _aggregate_model(
    accuracies: list[float],
    final_metrics: list[float],
    progress_epochs: list[int],
) -> AggregateModelStats:
    return AggregateModelStats(
        mean_test_accuracy=float(np.mean(accuracies)),
        std_test_accuracy=float(np.std(accuracies)),
        mean_final_metric=float(np.mean(final_metrics)),
        std_final_metric=float(np.std(final_metrics)),
        mean_epoch_80pct_progress=float(np.mean(progress_epochs)),
        std_epoch_80pct_progress=float(np.std(progress_epochs)),
    )


def _epoch_for_80pct_progress(metric_history: list[float]) -> int:
    if not metric_history:
        raise ValueError("metric_history cannot be empty")

    start_metric = metric_history[0]
    end_metric = metric_history[-1]
    total_change = start_metric - end_metric
    if abs(total_change) < 1e-12:
        return len(metric_history)

    target_metric = start_metric - 0.8 * total_change
    for index, value in enumerate(metric_history, start=1):
        if total_change > 0.0 and value <= target_metric:
            return index
        if total_change < 0.0 and value >= target_metric:
            return index
    return len(metric_history)


def _format_aggregate_line(
    mean_accuracy: float,
    std_accuracy: float,
    mean_metric: float,
    std_metric: float,
    mean_epoch: float,
    std_epoch: float,
) -> str:
    return (
        f"acc={mean_accuracy:.3f}+/-{std_accuracy:.3f}, "
        f"final_metric={mean_metric:.4f}+/-{std_metric:.4f}, "
        f"epoch_80pct={mean_epoch:.1f}+/-{std_epoch:.1f}"
    )

