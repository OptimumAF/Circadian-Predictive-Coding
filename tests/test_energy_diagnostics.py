"""Numeric contracts for the existing training-energy diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.activations import sigmoid
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork


def binary_cross_entropy(probability: np.ndarray, target: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-8, 1.0 - 1e-8)
    return float(np.mean(-(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped))))


@pytest.mark.parametrize("hidden_dims", [(2, 3), (2, 5)])
def test_numpy_pc_energy_is_preupdate_bce_plus_unit_mean_hidden_diagnostic(
    hidden_dims: tuple[int, int],
) -> None:
    inputs = np.array([[0.2, -0.4], [0.1, 0.3]], dtype=np.float64)
    targets = np.array([[1.0], [0.0]], dtype=np.float64)
    model = PredictiveCodingNetwork(
        input_dim=2, hidden_dim=hidden_dims[0], hidden_dims=hidden_dims, seed=17
    )
    priors: list[np.ndarray] = []
    prior_input = inputs
    for weight, bias in zip(model._hidden_weights, model._hidden_biases):
        prior_input = np.tanh(prior_input @ weight + bias)
        priors.append(prior_input)
    states = [prior.copy() for prior in priors]
    output_weight = model.weight_hidden_output.copy()
    output_bias = model.bias_output.copy()
    for _ in range(2):
        output = sigmoid(states[-1] @ output_weight + output_bias)
        output_error = output - targets
        errors = [state - prior for state, prior in zip(states, priors)]
        gradients = [np.zeros_like(state) for state in states]
        gradients[-1] = errors[-1] + output_error @ output_weight.T
        for layer in range(len(states) - 2, -1, -1):
            gradients[layer] = (
                errors[layer] + errors[layer + 1] @ model._hidden_weights[layer + 1].T
            )
        states = [state - 0.2 * gradient for state, gradient in zip(states, gradients)]
    final_output = sigmoid(states[-1] @ output_weight + output_bias)
    final_errors = [state - prior for state, prior in zip(states, priors)]
    bce = binary_cross_entropy(final_output, targets)
    squared_sum = sum(float(np.sum(error * error)) for error in final_errors)
    expected = bce + 0.5 * squared_sum / (len(inputs) * sum(hidden_dims))

    result = model.train_epoch(
        inputs, targets, learning_rate=0.1, inference_steps=2, inference_learning_rate=0.2
    )

    assert squared_sum > 0.0
    assert result.energy == pytest.approx(expected, abs=1e-12)
    assert (result.energy - bce) / (0.5 * squared_sum / len(inputs)) == pytest.approx(
        1.0 / sum(hidden_dims)
    )
    assert result.energy_definition == "numpy_pc_bce_plus_half_mean_all_hidden_error_sq_v1"


def test_numpy_circadian_energy_averages_over_adaptive_hidden_width() -> None:
    inputs = np.array([[0.2, -0.4], [0.1, 0.3]], dtype=np.float64)
    targets = np.array([[1.0], [0.0]], dtype=np.float64)
    model = CircadianPredictiveCodingNetwork(input_dim=2, hidden_dim=3, min_hidden_dim=2, seed=19)
    prior = np.tanh(inputs @ model.weight_input_hidden + model.bias_hidden)
    state = prior.copy()
    output_weight = model.weight_hidden_output.copy()
    output_bias = model.bias_output.copy()
    for _ in range(2):
        output = sigmoid(state @ output_weight + output_bias)
        state = state - 0.2 * ((state - prior) + (output - targets) @ output_weight.T)
    final_output = sigmoid(state @ output_weight + output_bias)
    error = state - prior
    bce = binary_cross_entropy(final_output, targets)
    squared_sum = float(np.sum(error * error))
    expected = bce + 0.5 * squared_sum / error.size

    result = model.train_epoch(
        inputs, targets, learning_rate=0.1, inference_steps=2, inference_learning_rate=0.2
    )

    assert squared_sum > 0.0
    assert result.energy == pytest.approx(expected, abs=1e-12)
    assert (result.energy - bce) / (0.5 * squared_sum / len(inputs)) == pytest.approx(1.0 / 3.0)
    assert result.energy_definition == "numpy_circadian_bce_plus_half_mean_final_hidden_error_sq_v1"


@pytest.mark.parametrize("circadian", [False, True])
def test_torch_pc_energy_is_separately_averaged_squared_residual_diagnostic(
    circadian: bool,
) -> None:
    torch = pytest.importorskip("torch")
    from src.core.resnet50_variants import CircadianPredictiveCodingHead, PredictiveCodingHead

    features = torch.tensor([[0.2, -0.4], [0.1, 0.3]], dtype=torch.float32)
    targets = torch.tensor([1, 0], dtype=torch.long)
    head: PredictiveCodingHead
    if circadian:
        head = CircadianPredictiveCodingHead(
            feature_dim=2,
            hidden_dim=3,
            num_classes=4,
            device=torch.device("cpu"),
            seed=23,
            min_hidden_dim=2,
            max_hidden_dim=6,
        )
    else:
        head = PredictiveCodingHead(
            feature_dim=2, hidden_dim=3, num_classes=4, device=torch.device("cpu"), seed=23
        )
    prior = torch.tanh(features @ head.weight_feature_hidden + head.bias_hidden)
    state = prior.clone()
    output_weight = head.weight_hidden_output.clone()
    output_bias = head.bias_output.clone()
    one_hot = torch.nn.functional.one_hot(targets, num_classes=4).to(features.dtype)
    for _ in range(2):
        probabilities = torch.softmax(state @ output_weight + output_bias, dim=1)
        state = state - 0.2 * ((state - prior) + (probabilities - one_hot) @ output_weight.T)
    probabilities = torch.softmax(state @ output_weight + output_bias, dim=1)
    output_error = probabilities - one_hot
    hidden_error = state - prior
    output_term = torch.sum(output_error.square()) / (2 * len(features) * 4)
    hidden_term = torch.sum(hidden_error.square()) / (2 * len(features) * 3)
    expected = output_term + hidden_term

    energy = head.train_step(
        features, targets, learning_rate=0.1, inference_steps=2, inference_learning_rate=0.2
    )

    assert energy == pytest.approx(float(expected), abs=1e-7)
    assert (
        head.energy_definition
        == "torch_pc_half_mean_output_error_sq_plus_half_mean_hidden_error_sq_v1"
    )
    assert float(torch.sum(hidden_error.square())) > 0.0


def test_torch_squared_output_diagnostic_does_not_have_cross_entropy_logit_gradient() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[1.0, 0.2, -0.7]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    probability = torch.softmax(logits, dim=1)
    output_error = probability - target
    squared_diagnostic = 0.5 * torch.mean(output_error.square())
    squared_gradient = torch.autograd.grad(squared_diagnostic, logits)[0]

    assert not torch.allclose(squared_gradient, output_error)
