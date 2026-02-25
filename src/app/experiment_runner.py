"""Use-case orchestration for model comparison experiments."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.backprop_mlp import BackpropMLP
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
    random_seed: int = 7


@dataclass(frozen=True)
class ModelReport:
    """Result metrics for one model."""

    loss_history: list[float]
    test_accuracy: float
    traffic_by_layer: list[LayerTraffic]


@dataclass(frozen=True)
class ExperimentResult:
    """End-to-end comparison output."""

    backprop: ModelReport
    predictive_coding: ModelReport


def run_experiment(
    config: ExperimentConfig,
    adaptation_policy: NeuronAdaptationPolicy | None = None,
) -> ExperimentResult:
    """Train both models on the same data and return comparable reports."""
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

    backprop_losses: list[float] = []
    predictive_coding_energies: list[float] = []
    for _ in range(config.epoch_count):
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

    backprop_traffic = backprop_model.get_layer_traffic()
    predictive_coding_traffic = predictive_coding_model.get_layer_traffic()

    backprop_model.apply_neuron_proposals(policy.propose(backprop_traffic))
    predictive_coding_model.apply_neuron_proposals(policy.propose(predictive_coding_traffic))

    backprop_accuracy = backprop_model.compute_accuracy(
        dataset.test_input, dataset.test_target
    )
    predictive_coding_accuracy = predictive_coding_model.compute_accuracy(
        dataset.test_input, dataset.test_target
    )

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
    )


def format_experiment_result(result: ExperimentResult) -> str:
    """Build a human-readable experiment summary."""
    bp_start = result.backprop.loss_history[0]
    bp_end = result.backprop.loss_history[-1]
    pc_start = result.predictive_coding.loss_history[0]
    pc_end = result.predictive_coding.loss_history[-1]

    return "\n".join(
        [
            "Predictive Coding vs Backprop Baseline",
            "--------------------------------------",
            f"Backprop loss: {bp_start:.4f} -> {bp_end:.4f}",
            f"Backprop test accuracy: {result.backprop.test_accuracy:.3f}",
            f"Predictive coding energy: {pc_start:.4f} -> {pc_end:.4f}",
            f"Predictive coding test accuracy: {result.predictive_coding.test_accuracy:.3f}",
            "",
            "Traffic snapshot (mean absolute activation, hidden layer):",
            "Backprop: "
            + _format_traffic_vector(result.backprop.traffic_by_layer[0].mean_abs_activation),
            "Predictive coding: "
            + _format_traffic_vector(
                result.predictive_coding.traffic_by_layer[0].mean_abs_activation
            ),
        ]
    )


def _format_traffic_vector(values: object) -> str:
    if hasattr(values, "tolist"):
        rounded = [f"{float(v):.3f}" for v in values.tolist()]
    else:
        rounded = [str(values)]
    return "[" + ", ".join(rounded) + "]"

