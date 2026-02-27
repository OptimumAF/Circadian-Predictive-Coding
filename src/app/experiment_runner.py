"""Use-case orchestration for model comparison experiments."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.neuron_adaptation import (
    LayerTraffic,
    NeuronAdaptationPolicy,
    NoOpNeuronAdaptationPolicy,
)
from src.core.predictive_coding import PredictiveCodingNetwork
from src.infra.datasets import generate_two_cluster_dataset


@dataclass(frozen=True)
class ExperimentConfig:
    """Configurable parameters for the baseline comparison."""

    sample_count: int = 400
    noise_scale: float = 0.8
    hidden_dim: int = 12
    epoch_count: int = 160
    backprop_learning_rate: float = 0.12
    pc_learning_rate: float = 0.05
    pc_inference_steps: int = 25
    pc_inference_learning_rate: float = 0.2
    circadian_learning_rate: float = 0.05
    circadian_inference_steps: int = 25
    circadian_inference_learning_rate: float = 0.2
    circadian_sleep_interval: int = 40
    circadian_force_sleep: bool = True
    circadian_use_policy_for_sleep: bool = False
    circadian_config: CircadianConfig | None = None
    random_seed: int = 7


@dataclass(frozen=True)
class ModelReport:
    """Result metrics for one model."""

    loss_history: list[float]
    test_accuracy: float
    traffic_by_layer: list[LayerTraffic]


@dataclass(frozen=True)
class CircadianSleepSummary:
    """Sleep-event summary for the circadian model."""

    event_count: int
    total_splits: int
    total_prunes: int
    hidden_dim_start: int
    hidden_dim_end: int


@dataclass(frozen=True)
class ExperimentResult:
    """End-to-end comparison output."""

    backprop: ModelReport
    predictive_coding: ModelReport
    circadian_predictive_coding: ModelReport
    circadian_sleep: CircadianSleepSummary


def run_experiment(
    config: ExperimentConfig,
    adaptation_policy: NeuronAdaptationPolicy | None = None,
) -> ExperimentResult:
    """Train all three models on the same data and return comparable reports."""
    policy = adaptation_policy or NoOpNeuronAdaptationPolicy()
    dataset = generate_two_cluster_dataset(
        sample_count=config.sample_count,
        noise_scale=config.noise_scale,
        seed=config.random_seed,
    )

    backprop_model = BackpropMLP(input_dim=2, hidden_dim=config.hidden_dim, seed=config.random_seed)
    predictive_coding_model = PredictiveCodingNetwork(
        input_dim=2, hidden_dim=config.hidden_dim, seed=config.random_seed + 1
    )
    circadian_model = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        seed=config.random_seed + 2,
        circadian_config=config.circadian_config,
    )

    backprop_losses: list[float] = []
    predictive_coding_energies: list[float] = []
    circadian_energies: list[float] = []

    sleep_event_count = 0
    total_splits = 0
    total_prunes = 0
    hidden_dim_start = circadian_model.hidden_dim

    for epoch_index in range(1, config.epoch_count + 1):
        backprop_step = backprop_model.train_epoch(
            input_batch=dataset.train_input,
            target_batch=dataset.train_target,
            learning_rate=config.backprop_learning_rate,
        )
        backprop_losses.append(backprop_step.loss)

        predictive_coding_step = predictive_coding_model.train_epoch(
            input_batch=dataset.train_input,
            target_batch=dataset.train_target,
            learning_rate=config.pc_learning_rate,
            inference_steps=config.pc_inference_steps,
            inference_learning_rate=config.pc_inference_learning_rate,
        )
        predictive_coding_energies.append(predictive_coding_step.energy)

        circadian_step = circadian_model.train_epoch(
            input_batch=dataset.train_input,
            target_batch=dataset.train_target,
            learning_rate=config.circadian_learning_rate,
            inference_steps=config.circadian_inference_steps,
            inference_learning_rate=config.circadian_inference_learning_rate,
        )
        circadian_energies.append(circadian_step.energy)

        if config.circadian_sleep_interval > 0 and epoch_index % config.circadian_sleep_interval == 0:
            sleep_policy = policy if config.circadian_use_policy_for_sleep else None
            sleep_result = circadian_model.sleep_event(
                adaptation_policy=sleep_policy,
                force_sleep=config.circadian_force_sleep,
                current_step=epoch_index,
                total_steps=config.epoch_count,
            )
            performed_sleep = (
                len(sleep_result.split_indices) > 0
                or len(sleep_result.pruned_indices) > 0
                or sleep_result.new_hidden_dim != sleep_result.old_hidden_dim
            )
            if performed_sleep:
                sleep_event_count += 1
                total_splits += len(sleep_result.split_indices)
                total_prunes += len(sleep_result.pruned_indices)

    backprop_traffic = backprop_model.get_layer_traffic()
    predictive_coding_traffic = predictive_coding_model.get_layer_traffic()
    circadian_traffic = circadian_model.get_layer_traffic()

    backprop_model.apply_neuron_proposals(policy.propose(backprop_traffic))
    predictive_coding_model.apply_neuron_proposals(policy.propose(predictive_coding_traffic))
    circadian_model.apply_neuron_proposals(policy.propose(circadian_traffic))

    backprop_accuracy = backprop_model.compute_accuracy(
        dataset.test_input, dataset.test_target
    )
    predictive_coding_accuracy = predictive_coding_model.compute_accuracy(
        dataset.test_input, dataset.test_target
    )
    circadian_accuracy = circadian_model.compute_accuracy(dataset.test_input, dataset.test_target)

    return ExperimentResult(
        backprop=ModelReport(
            loss_history=backprop_losses,
            test_accuracy=backprop_accuracy,
            traffic_by_layer=backprop_traffic,
        ),
        predictive_coding=ModelReport(
            loss_history=predictive_coding_energies,
            test_accuracy=predictive_coding_accuracy,
            traffic_by_layer=predictive_coding_traffic,
        ),
        circadian_predictive_coding=ModelReport(
            loss_history=circadian_energies,
            test_accuracy=circadian_accuracy,
            traffic_by_layer=circadian_traffic,
        ),
        circadian_sleep=CircadianSleepSummary(
            event_count=sleep_event_count,
            total_splits=total_splits,
            total_prunes=total_prunes,
            hidden_dim_start=hidden_dim_start,
            hidden_dim_end=circadian_model.hidden_dim,
        ),
    )


def format_experiment_result(result: ExperimentResult) -> str:
    """Build a human-readable experiment summary."""
    bp_start = result.backprop.loss_history[0]
    bp_end = result.backprop.loss_history[-1]
    pc_start = result.predictive_coding.loss_history[0]
    pc_end = result.predictive_coding.loss_history[-1]
    cpc_start = result.circadian_predictive_coding.loss_history[0]
    cpc_end = result.circadian_predictive_coding.loss_history[-1]

    return "\n".join(
        [
            "Backprop vs Predictive Coding vs Circadian Predictive Coding",
            "------------------------------------------------------------",
            f"Backprop loss: {bp_start:.4f} -> {bp_end:.4f}",
            f"Backprop test accuracy: {result.backprop.test_accuracy:.3f}",
            f"Predictive coding energy: {pc_start:.4f} -> {pc_end:.4f}",
            f"Predictive coding test accuracy: {result.predictive_coding.test_accuracy:.3f}",
            f"Circadian predictive coding energy: {cpc_start:.4f} -> {cpc_end:.4f}",
            f"Circadian predictive coding test accuracy: {result.circadian_predictive_coding.test_accuracy:.3f}",
            (
                "Circadian sleep: "
                f"events={result.circadian_sleep.event_count}, "
                f"splits={result.circadian_sleep.total_splits}, "
                f"prunes={result.circadian_sleep.total_prunes}, "
                f"hidden_dim={result.circadian_sleep.hidden_dim_start}"
                f"->{result.circadian_sleep.hidden_dim_end}"
            ),
            "",
            "Traffic snapshot (mean absolute activation):",
            "Backprop: "
            + _format_traffic_vector(result.backprop.traffic_by_layer[0].mean_abs_activation),
            "Predictive coding: "
            + _format_traffic_vector(
                result.predictive_coding.traffic_by_layer[0].mean_abs_activation
            ),
            "Circadian hidden: "
            + _format_traffic_vector(
                result.circadian_predictive_coding.traffic_by_layer[0].mean_abs_activation
            ),
            "Circadian chemical: "
            + _format_traffic_vector(
                result.circadian_predictive_coding.traffic_by_layer[1].mean_abs_activation
            ),
        ]
    )


def _format_traffic_vector(values: object) -> str:
    if hasattr(values, "tolist"):
        rounded = [f"{float(v):.3f}" for v in values.tolist()]
    else:
        rounded = [str(values)]
    return "[" + ", ".join(rounded) + "]"
