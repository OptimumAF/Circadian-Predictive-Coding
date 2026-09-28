"""Independent float64 checks for the one-hidden, ungated local objective."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
import pytest

from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.backprop_mlp import BackpropMLP
from src.core.predictive_coding import PredictiveCodingNetwork

Array = NDArray[np.float64]


def finite_difference(array: Array, objective: Callable[[], float], epsilon: float = 1e-6) -> Array:
    derivative = np.empty_like(array)
    for index in np.ndindex(array.shape):
        original = float(array[index])
        array[index] = original + epsilon
        positive = objective()
        array[index] = original - epsilon
        negative = objective()
        array[index] = original
        derivative[index] = (positive - negative) / (2 * epsilon)
    return derivative


def binary_fixture() -> tuple[Array, Array, dict[str, Array]]:
    inputs = np.array([[0.3, -0.2], [-0.5, 0.4]], dtype=np.float64)
    targets = np.array([[1.0], [0.0]], dtype=np.float64)
    values = {
        "input_weight": np.array([[0.13, -0.21, 0.08], [-0.17, 0.05, 0.19]]),
        "hidden_bias": np.array([[0.02, -0.04, 0.03]]),
        "output_weight": np.array([[0.2], [-0.3], [0.4]]),
        "output_bias": np.array([[0.07]]),
    }
    prior = np.tanh(inputs @ values["input_weight"] + values["hidden_bias"])
    values["state"] = prior + np.array([[0.04, -0.03, 0.02], [-0.01, 0.05, -0.02]])
    return inputs, targets, values


def binary_objective(inputs: Array, targets: Array, values: dict[str, Array]) -> float:
    prior = np.tanh(inputs @ values["input_weight"] + values["hidden_bias"])
    error = values["state"] - prior
    logits = values["state"] @ values["output_weight"] + values["output_bias"]
    cross_entropy = np.mean(np.logaddexp(0.0, logits) - targets * logits)
    return float(cross_entropy + np.sum(error * error) / (2 * len(inputs)))


def binary_analytic(inputs: Array, targets: Array, values: dict[str, Array]) -> dict[str, Array]:
    prior = np.tanh(inputs @ values["input_weight"] + values["hidden_bias"])
    error = values["state"] - prior
    logits = values["state"] @ values["output_weight"] + values["output_bias"]
    probability = 1.0 / (1.0 + np.exp(-logits))
    output_error = probability - targets
    hidden_prior_delta = -error * (1.0 - prior * prior)
    batch_size = len(inputs)
    return {
        "state": (error + output_error @ values["output_weight"].T) / batch_size,
        "input_weight": inputs.T @ hidden_prior_delta / batch_size,
        "hidden_bias": np.sum(hidden_prior_delta, axis=0, keepdims=True) / batch_size,
        "output_weight": values["state"].T @ output_error / batch_size,
        "output_bias": np.sum(output_error, axis=0, keepdims=True) / batch_size,
    }


def relaxed_binary_state(inputs: Array, targets: Array, values: dict[str, Array]) -> Array:
    prior = np.tanh(inputs @ values["input_weight"] + values["hidden_bias"])
    state = prior.copy()
    for _ in range(2):
        logits = state @ values["output_weight"] + values["output_bias"]
        probability = 1.0 / (1.0 + np.exp(-logits))
        state -= 0.2 * ((state - prior) + (probability - targets) @ values["output_weight"].T)
    return state


def set_binary_model_parameters(model: object, values: dict[str, Array]) -> None:
    for field, value in (
        ("weight_input_hidden", values["input_weight"]),
        ("bias_hidden", values["hidden_bias"]),
        ("weight_hidden_output", values["output_weight"]),
        ("bias_output", values["output_bias"]),
    ):
        np.copyto(getattr(model, field), value)


def test_binary_local_partials_match_central_differences_and_autograd() -> None:
    torch = pytest.importorskip("torch")
    inputs, targets, values = binary_fixture()
    analytic = binary_analytic(inputs, targets, values)
    for name, value in values.items():
        numeric = finite_difference(value, lambda: binary_objective(inputs, targets, values))
        np.testing.assert_allclose(analytic[name], numeric, atol=2e-9, rtol=2e-7)

    tensors = {
        name: torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for name, value in values.items()
    }
    torch_inputs = torch.tensor(inputs, dtype=torch.float64)
    torch_targets = torch.tensor(targets, dtype=torch.float64)
    prior = torch.tanh(torch_inputs @ tensors["input_weight"] + tensors["hidden_bias"])
    logits = tensors["state"] @ tensors["output_weight"] + tensors["output_bias"]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch_targets)
    loss = loss + torch.sum((tensors["state"] - prior).square()) / (2 * len(inputs))
    autograd = torch.autograd.grad(loss, tuple(tensors.values()))
    for name, derivative in zip(tensors, autograd):
        np.testing.assert_allclose(analytic[name], derivative.detach().numpy(), atol=2e-12)


@pytest.mark.parametrize("circadian", [False, True])
def test_binary_model_step_uses_batch_mean_held_state_partials(circadian: bool) -> None:
    inputs, targets, values = binary_fixture()
    if circadian:
        model: PredictiveCodingNetwork | CircadianPredictiveCodingNetwork = (
            CircadianPredictiveCodingNetwork(
                input_dim=2,
                hidden_dim=3,
                min_hidden_dim=2,
                seed=31,
                circadian_config=CircadianConfig(min_plasticity=1.0),
            )
        )
    else:
        model = PredictiveCodingNetwork(input_dim=2, hidden_dim=3, seed=31)
    set_binary_model_parameters(model, values)
    at_relaxed_state = {**values, "state": relaxed_binary_state(inputs, targets, values)}
    expected = binary_analytic(inputs, targets, at_relaxed_state)

    model.train_epoch(
        inputs, targets, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.2
    )

    for field, name in (
        ("weight_input_hidden", "input_weight"),
        ("bias_hidden", "hidden_bias"),
        ("weight_hidden_output", "output_weight"),
        ("bias_output", "output_bias"),
    ):
        actual = getattr(model, field)
        np.testing.assert_allclose((values[name] - actual) / 0.03, expected[name], atol=2e-12)


def test_binary_weight_step_is_invariant_to_exact_batch_duplication() -> None:
    inputs, targets, values = binary_fixture()
    inputs = inputs[:1]
    targets = targets[:1]
    updated: list[Array] = []
    for copies in (1, 3):
        model = PredictiveCodingNetwork(input_dim=2, hidden_dim=3, seed=31)
        set_binary_model_parameters(model, values)
        model.train_epoch(
            np.repeat(inputs, copies, axis=0),
            np.repeat(targets, copies, axis=0),
            learning_rate=0.03,
            inference_steps=2,
            inference_learning_rate=0.2,
        )
        updated.append(model.weight_input_hidden.copy())
    np.testing.assert_allclose(updated[0], updated[1], atol=2e-16)


def multiclass_fixture() -> tuple[Array, NDArray[np.int64], dict[str, Array]]:
    inputs = np.array([[0.3, -0.2], [-0.5, 0.4]], dtype=np.float64)
    targets = np.array([0, 2], dtype=np.int64)
    values = {
        "input_weight": np.array([[0.13, -0.21, 0.08], [-0.17, 0.05, 0.19]]),
        "hidden_bias": np.array([[0.02, -0.04, 0.03]]),
        "output_weight": np.array([[0.2, -0.1, 0.07], [-0.3, 0.15, -0.09], [0.4, 0.06, -0.2]]),
        "output_bias": np.array([[0.07, -0.03, 0.01]]),
    }
    prior = np.tanh(inputs @ values["input_weight"] + values["hidden_bias"])
    values["state"] = prior + np.array([[0.04, -0.03, 0.02], [-0.01, 0.05, -0.02]])
    return inputs, targets, values


def softmax(logits: Array) -> Array:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def multiclass_objective(
    inputs: Array, targets: NDArray[np.int64], values: dict[str, Array]
) -> float:
    prior = np.tanh(inputs @ values["input_weight"] + values["hidden_bias"])
    error = values["state"] - prior
    logits = values["state"] @ values["output_weight"] + values["output_bias"]
    stabilized = logits - np.max(logits, axis=1, keepdims=True)
    log_partition = np.log(np.sum(np.exp(stabilized), axis=1))
    cross_entropy = np.mean(log_partition - stabilized[np.arange(len(inputs)), targets])
    return float(cross_entropy + np.sum(error * error) / (2 * len(inputs)))


def multiclass_analytic(
    inputs: Array, targets: NDArray[np.int64], values: dict[str, Array]
) -> dict[str, Array]:
    prior = np.tanh(inputs @ values["input_weight"] + values["hidden_bias"])
    error = values["state"] - prior
    logits = values["state"] @ values["output_weight"] + values["output_bias"]
    output_error = softmax(logits)
    output_error[np.arange(len(inputs)), targets] -= 1.0
    hidden_prior_delta = -error * (1.0 - prior * prior)
    batch_size = len(inputs)
    return {
        "state": (error + output_error @ values["output_weight"].T) / batch_size,
        "input_weight": inputs.T @ hidden_prior_delta / batch_size,
        "hidden_bias": np.sum(hidden_prior_delta, axis=0, keepdims=True) / batch_size,
        "output_weight": values["state"].T @ output_error / batch_size,
        "output_bias": np.sum(output_error, axis=0, keepdims=True) / batch_size,
    }


def relaxed_multiclass_state(
    inputs: Array, targets: NDArray[np.int64], values: dict[str, Array]
) -> Array:
    prior = np.tanh(inputs @ values["input_weight"] + values["hidden_bias"])
    state = prior.copy()
    for _ in range(2):
        logits = state @ values["output_weight"] + values["output_bias"]
        output_error = softmax(logits)
        output_error[np.arange(len(inputs)), targets] -= 1.0
        state -= 0.2 * ((state - prior) + output_error @ values["output_weight"].T)
    return state


def test_multiclass_local_partials_match_central_differences_and_autograd() -> None:
    torch = pytest.importorskip("torch")
    inputs, targets, values = multiclass_fixture()
    analytic = multiclass_analytic(inputs, targets, values)
    for name, value in values.items():
        numeric = finite_difference(value, lambda: multiclass_objective(inputs, targets, values))
        np.testing.assert_allclose(analytic[name], numeric, atol=2e-9, rtol=2e-7)

    tensors = {
        name: torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for name, value in values.items()
    }
    torch_inputs = torch.tensor(inputs, dtype=torch.float64)
    torch_targets = torch.tensor(targets, dtype=torch.long)
    prior = torch.tanh(torch_inputs @ tensors["input_weight"] + tensors["hidden_bias"])
    logits = tensors["state"] @ tensors["output_weight"] + tensors["output_bias"]
    loss = torch.nn.functional.cross_entropy(logits, torch_targets)
    loss = loss + torch.sum((tensors["state"] - prior).square()) / (2 * len(inputs))
    autograd = torch.autograd.grad(loss, tuple(tensors.values()))
    for name, derivative in zip(tensors, autograd):
        np.testing.assert_allclose(analytic[name], derivative.detach().numpy(), atol=2e-12)


@pytest.mark.parametrize("circadian", [False, True])
def test_multiclass_head_step_uses_batch_mean_held_state_partials(circadian: bool) -> None:
    torch = pytest.importorskip("torch")
    from src.core.resnet50_variants import (
        CircadianHeadConfig,
        CircadianPredictiveCodingHead,
        PredictiveCodingHead,
    )

    inputs, targets, values = multiclass_fixture()
    if circadian:
        head: PredictiveCodingHead = CircadianPredictiveCodingHead(
            feature_dim=2,
            hidden_dim=3,
            num_classes=3,
            device=torch.device("cpu"),
            seed=37,
            config=CircadianHeadConfig(min_plasticity=1.0),
            min_hidden_dim=2,
            max_hidden_dim=6,
        )
    else:
        head = PredictiveCodingHead(
            feature_dim=2,
            hidden_dim=3,
            num_classes=3,
            device=torch.device("cpu"),
            seed=37,
        )
    for field, name in (
        ("weight_feature_hidden", "input_weight"),
        ("bias_hidden", "hidden_bias"),
        ("weight_hidden_output", "output_weight"),
        ("bias_output", "output_bias"),
    ):
        setattr(head, field, torch.tensor(values[name], dtype=torch.float64))
    at_relaxed_state = {**values, "state": relaxed_multiclass_state(inputs, targets, values)}
    expected = multiclass_analytic(inputs, targets, at_relaxed_state)

    head.train_step(
        torch.tensor(inputs, dtype=torch.float64),
        torch.tensor(targets, dtype=torch.long),
        learning_rate=0.03,
        inference_steps=2,
        inference_learning_rate=0.2,
    )

    for field, name in (
        ("weight_feature_hidden", "input_weight"),
        ("bias_hidden", "hidden_bias"),
        ("weight_hidden_output", "output_weight"),
        ("bias_output", "output_bias"),
    ):
        actual = getattr(head, field).detach().numpy()
        np.testing.assert_allclose((values[name] - actual) / 0.03, expected[name], atol=2e-12)


def test_multiclass_weight_step_is_invariant_to_exact_batch_duplication() -> None:
    torch = pytest.importorskip("torch")
    from src.core.resnet50_variants import PredictiveCodingHead

    inputs, targets, values = multiclass_fixture()
    updated: list[Array] = []
    for copies in (1, 3):
        head = PredictiveCodingHead(2, 3, 3, torch.device("cpu"), seed=37)
        head.weight_feature_hidden = torch.tensor(values["input_weight"], dtype=torch.float64)
        head.bias_hidden = torch.tensor(values["hidden_bias"], dtype=torch.float64)
        head.weight_hidden_output = torch.tensor(values["output_weight"], dtype=torch.float64)
        head.bias_output = torch.tensor(values["output_bias"], dtype=torch.float64)
        head.train_step(
            torch.tensor(np.repeat(inputs[:1], copies, axis=0), dtype=torch.float64),
            torch.tensor(np.repeat(targets[:1], copies, axis=0), dtype=torch.long),
            learning_rate=0.03,
            inference_steps=2,
            inference_learning_rate=0.2,
        )
        updated.append(head.weight_feature_hidden.detach().numpy())
    np.testing.assert_allclose(updated[0], updated[1], atol=2e-16)


def backprop_fixture() -> tuple[Array, Array, dict[str, Array]]:
    inputs = np.array([[0.3, -0.2], [-0.5, 0.4]], dtype=np.float64)
    targets = np.array([[1.0], [0.0]], dtype=np.float64)
    values = {
        "first_weight": np.array([[0.13, -0.21, 0.08], [-0.17, 0.05, 0.19]]),
        "first_bias": np.array([[0.02, -0.04, 0.03]]),
        "second_weight": np.array([[0.2, -0.1], [-0.3, 0.15], [0.4, 0.06]]),
        "second_bias": np.array([[0.07, -0.03]]),
        "output_weight": np.array([[0.2], [-0.3]]),
        "output_bias": np.array([[0.04]]),
    }
    return inputs, targets, values


def backprop_forward(inputs: Array, values: dict[str, Array]) -> tuple[Array, Array, Array]:
    first = np.tanh(inputs @ values["first_weight"] + values["first_bias"])
    second = np.tanh(first @ values["second_weight"] + values["second_bias"])
    logits = second @ values["output_weight"] + values["output_bias"]
    return first, second, logits


def backprop_objective(inputs: Array, targets: Array, values: dict[str, Array]) -> float:
    _, _, logits = backprop_forward(inputs, values)
    return float(np.mean(np.logaddexp(0.0, logits) - targets * logits))


def backprop_analytic(inputs: Array, targets: Array, values: dict[str, Array]) -> dict[str, Array]:
    first, second, logits = backprop_forward(inputs, values)
    output_delta = (1.0 / (1.0 + np.exp(-logits)) - targets) / len(inputs)
    second_delta = (output_delta @ values["output_weight"].T) * (1.0 - second * second)
    first_delta = (second_delta @ values["second_weight"].T) * (1.0 - first * first)
    return {
        "first_weight": inputs.T @ first_delta,
        "first_bias": np.sum(first_delta, axis=0, keepdims=True),
        "second_weight": first.T @ second_delta,
        "second_bias": np.sum(second_delta, axis=0, keepdims=True),
        "output_weight": second.T @ output_delta,
        "output_bias": np.sum(output_delta, axis=0, keepdims=True),
    }


def set_backprop_parameters(model: BackpropMLP, values: dict[str, Array]) -> None:
    for actual, name in (
        (model._hidden_weights[0], "first_weight"),
        (model._hidden_biases[0], "first_bias"),
        (model._hidden_weights[1], "second_weight"),
        (model._hidden_biases[1], "second_bias"),
        (model.weight_hidden_output, "output_weight"),
        (model.bias_output, "output_bias"),
    ):
        np.copyto(actual, values[name])


def test_multilayer_numpy_backprop_derivatives_match_finite_differences_and_autograd() -> None:
    torch = pytest.importorskip("torch")
    inputs, targets, values = backprop_fixture()
    analytic = backprop_analytic(inputs, targets, values)
    for name, value in values.items():
        numeric = finite_difference(value, lambda: backprop_objective(inputs, targets, values))
        np.testing.assert_allclose(analytic[name], numeric, atol=2e-9, rtol=2e-7)

    tensors = {
        name: torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for name, value in values.items()
    }
    torch_inputs = torch.tensor(inputs, dtype=torch.float64)
    torch_targets = torch.tensor(targets, dtype=torch.float64)
    first = torch.tanh(torch_inputs @ tensors["first_weight"] + tensors["first_bias"])
    second = torch.tanh(first @ tensors["second_weight"] + tensors["second_bias"])
    logits = second @ tensors["output_weight"] + tensors["output_bias"]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch_targets)
    autograd = torch.autograd.grad(loss, tuple(tensors.values()))
    for name, derivative in zip(tensors, autograd):
        np.testing.assert_allclose(analytic[name], derivative.detach().numpy(), atol=2e-12)

    model = BackpropMLP(input_dim=2, hidden_dim=2, hidden_dims=(3, 2), seed=41)
    set_backprop_parameters(model, values)
    model.train_epoch(inputs, targets, learning_rate=0.03)
    for actual, name in (
        (model._hidden_weights[0], "first_weight"),
        (model._hidden_biases[0], "first_bias"),
        (model._hidden_weights[1], "second_weight"),
        (model._hidden_biases[1], "second_bias"),
        (model.weight_hidden_output, "output_weight"),
        (model.bias_output, "output_bias"),
    ):
        np.testing.assert_allclose((values[name] - actual) / 0.03, analytic[name], atol=2e-12)


def test_multilayer_backprop_weight_step_averages_duplicate_examples() -> None:
    inputs, targets, values = backprop_fixture()
    updated: list[Array] = []
    for copies in (1, 3):
        model = BackpropMLP(input_dim=2, hidden_dim=2, hidden_dims=(3, 2), seed=41)
        set_backprop_parameters(model, values)
        model.train_epoch(
            np.repeat(inputs[:1], copies, axis=0),
            np.repeat(targets[:1], copies, axis=0),
            learning_rate=0.03,
        )
        updated.append(model._hidden_weights[0].copy())
    np.testing.assert_allclose(updated[0], updated[1], atol=2e-16)


def test_circadian_gate_scales_local_partials_after_chemical_update() -> None:
    inputs, targets, values = binary_fixture()
    model = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=3,
        min_hidden_dim=2,
        seed=43,
        circadian_config=CircadianConfig(min_plasticity=0.1, replay_steps=0),
    )
    set_binary_model_parameters(model, values)
    initial_chemical = np.array([0.2, 0.6, 0.9], dtype=np.float64)
    model.set_chemical_state(initial_chemical)
    state = relaxed_binary_state(inputs, targets, values)
    gradients = binary_analytic(inputs, targets, {**values, "state": state})
    expected_chemical = 0.995 * initial_chemical + 0.02 * np.mean(np.abs(state), axis=0)
    gate = np.clip(np.exp(-0.7 * expected_chemical), 0.1, 1.0)

    model.train_epoch(
        inputs, targets, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.2
    )

    np.testing.assert_allclose(model.get_chemical_state(), expected_chemical, atol=2e-15)
    np.testing.assert_allclose(
        (values["input_weight"] - model.weight_input_hidden) / 0.03,
        gradients["input_weight"] * gate[None, :],
        atol=2e-12,
    )
    np.testing.assert_allclose(
        (values["hidden_bias"] - model.bias_hidden) / 0.03,
        gradients["hidden_bias"] * gate[None, :],
        atol=2e-12,
    )
    np.testing.assert_allclose(
        (values["output_weight"] - model.weight_hidden_output) / 0.03,
        gradients["output_weight"] * gate[:, None],
        atol=2e-12,
    )
    np.testing.assert_allclose(
        (values["output_bias"] - model.bias_output) / 0.03,
        gradients["output_bias"],
        atol=2e-12,
    )


def deeper_circadian_fixture() -> tuple[Array, Array, dict[str, Array]]:
    inputs = np.array([[0.3, -0.2], [-0.5, 0.4]], dtype=np.float64)
    targets = np.array([[1.0], [0.0]], dtype=np.float64)
    values = {
        "pre_weight": np.array([[0.13, -0.21, 0.08], [-0.17, 0.05, 0.19]]),
        "pre_bias": np.array([[0.02, -0.04, 0.03]]),
        "adaptive_weight": np.array([[0.2, -0.1], [-0.3, 0.15], [0.4, 0.06]]),
        "adaptive_bias": np.array([[0.07, -0.03]]),
        "output_weight": np.array([[0.2], [-0.3]]),
        "output_bias": np.array([[0.04]]),
    }
    pre_state = np.tanh(inputs @ values["pre_weight"] + values["pre_bias"])
    prior = np.tanh(pre_state @ values["adaptive_weight"] + values["adaptive_bias"])
    values["state"] = prior + np.array([[0.04, -0.03], [-0.01, 0.05]])
    return inputs, targets, values


def deeper_circadian_prior(inputs: Array, values: dict[str, Array]) -> tuple[Array, Array]:
    pre_state = np.tanh(inputs @ values["pre_weight"] + values["pre_bias"])
    prior = np.tanh(pre_state @ values["adaptive_weight"] + values["adaptive_bias"])
    return pre_state, prior


def deeper_circadian_objective(inputs: Array, targets: Array, values: dict[str, Array]) -> float:
    _, prior = deeper_circadian_prior(inputs, values)
    logits = values["state"] @ values["output_weight"] + values["output_bias"]
    error = values["state"] - prior
    return float(
        np.mean(np.logaddexp(0.0, logits) - targets * logits)
        + np.sum(error * error) / (2 * len(inputs))
    )


def deeper_circadian_analytic(
    inputs: Array, targets: Array, values: dict[str, Array]
) -> dict[str, Array]:
    pre_state, prior = deeper_circadian_prior(inputs, values)
    error = values["state"] - prior
    logits = values["state"] @ values["output_weight"] + values["output_bias"]
    output_error = 1.0 / (1.0 + np.exp(-logits)) - targets
    adaptive_delta = -error * (1.0 - prior * prior)
    pre_delta = (adaptive_delta @ values["adaptive_weight"].T) * (1.0 - pre_state * pre_state)
    batch_size = len(inputs)
    return {
        "state": (error + output_error @ values["output_weight"].T) / batch_size,
        "pre_weight": inputs.T @ pre_delta / batch_size,
        "pre_bias": np.sum(pre_delta, axis=0, keepdims=True) / batch_size,
        "adaptive_weight": pre_state.T @ adaptive_delta / batch_size,
        "adaptive_bias": np.sum(adaptive_delta, axis=0, keepdims=True) / batch_size,
        "output_weight": values["state"].T @ output_error / batch_size,
        "output_bias": np.sum(output_error, axis=0, keepdims=True) / batch_size,
    }


def test_deeper_circadian_prior_chain_matches_finite_difference_autograd_and_step() -> None:
    torch = pytest.importorskip("torch")
    inputs, targets, values = deeper_circadian_fixture()
    analytic = deeper_circadian_analytic(inputs, targets, values)
    for name, value in values.items():
        numeric = finite_difference(
            value, lambda: deeper_circadian_objective(inputs, targets, values)
        )
        np.testing.assert_allclose(analytic[name], numeric, atol=2e-9, rtol=2e-7)

    tensors = {
        name: torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for name, value in values.items()
    }
    torch_inputs = torch.tensor(inputs, dtype=torch.float64)
    torch_targets = torch.tensor(targets, dtype=torch.float64)
    pre_state = torch.tanh(torch_inputs @ tensors["pre_weight"] + tensors["pre_bias"])
    prior = torch.tanh(pre_state @ tensors["adaptive_weight"] + tensors["adaptive_bias"])
    logits = tensors["state"] @ tensors["output_weight"] + tensors["output_bias"]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch_targets)
    loss = loss + torch.sum((tensors["state"] - prior).square()) / (2 * len(inputs))
    autograd = torch.autograd.grad(loss, tuple(tensors.values()))
    for name, derivative in zip(tensors, autograd):
        np.testing.assert_allclose(analytic[name], derivative.detach().numpy(), atol=2e-12)

    model = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=2,
        hidden_dims=(3, 2),
        min_hidden_dim=2,
        seed=47,
        circadian_config=CircadianConfig(min_plasticity=1.0, replay_steps=0),
    )
    for actual, name in (
        (model._pre_hidden_weights[0], "pre_weight"),
        (model._pre_hidden_biases[0], "pre_bias"),
        (model.weight_input_hidden, "adaptive_weight"),
        (model.bias_hidden, "adaptive_bias"),
        (model.weight_hidden_output, "output_weight"),
        (model.bias_output, "output_bias"),
    ):
        np.copyto(actual, values[name])
    pre_state, prior = deeper_circadian_prior(inputs, values)
    assert pre_state.shape == (2, 3)
    state = prior.copy()
    for _ in range(2):
        logits = state @ values["output_weight"] + values["output_bias"]
        probability = 1.0 / (1.0 + np.exp(-logits))
        state -= 0.2 * ((state - prior) + (probability - targets) @ values["output_weight"].T)
    expected = deeper_circadian_analytic(inputs, targets, {**values, "state": state})

    model.train_epoch(
        inputs, targets, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.2
    )

    for actual, name in (
        (model._pre_hidden_weights[0], "pre_weight"),
        (model._pre_hidden_biases[0], "pre_bias"),
        (model.weight_input_hidden, "adaptive_weight"),
        (model.bias_hidden, "adaptive_bias"),
        (model.weight_hidden_output, "output_weight"),
        (model.bias_output, "output_bias"),
    ):
        np.testing.assert_allclose((values[name] - actual) / 0.03, expected[name], atol=2e-12)


def test_torch_circadian_chemical_and_reward_gates_scale_local_partials() -> None:
    torch = pytest.importorskip("torch")
    from src.core.resnet50_variants import CircadianHeadConfig, CircadianPredictiveCodingHead

    inputs, targets, values = multiclass_fixture()
    state = relaxed_multiclass_state(inputs, targets, values)
    gradients = multiclass_analytic(inputs, targets, {**values, "state": state})
    initial_chemical = np.array([0.2, 0.6, 0.9], dtype=np.float64)
    expected_chemical = 0.995 * initial_chemical + 0.02 * np.mean(np.abs(state), axis=0)
    gate = np.clip(np.exp(-0.7 * expected_chemical), 0.1, 1.0)
    logits = state @ values["output_weight"] + values["output_bias"]
    output_error = softmax(logits)
    output_error[np.arange(len(inputs)), targets] -= 1.0
    difficulty = float(np.mean(np.abs(output_error)))
    reward = float(np.clip(difficulty / 0.15, 0.75, 1.5))
    assert reward != 1.0

    head = CircadianPredictiveCodingHead(
        feature_dim=2,
        hidden_dim=3,
        num_classes=3,
        device=torch.device("cpu"),
        seed=53,
        config=CircadianHeadConfig(
            min_plasticity=0.1,
            use_reward_modulated_learning=True,
        ),
        min_hidden_dim=2,
        max_hidden_dim=6,
    )
    for field, name in (
        ("weight_feature_hidden", "input_weight"),
        ("bias_hidden", "hidden_bias"),
        ("weight_hidden_output", "output_weight"),
        ("bias_output", "output_bias"),
    ):
        setattr(head, field, torch.tensor(values[name], dtype=torch.float64))
    for field in ("_chemical", "_chemical_fast", "_chemical_slow"):
        setattr(head, field, torch.tensor(initial_chemical, dtype=torch.float64))
    head._reward_error_ema = 0.15

    head.train_step(
        torch.tensor(inputs, dtype=torch.float64),
        torch.tensor(targets, dtype=torch.long),
        learning_rate=0.03,
        inference_steps=2,
        inference_learning_rate=0.2,
    )

    assert head.last_reward_scale() == pytest.approx(reward)
    np.testing.assert_allclose(head._chemical.detach().numpy(), expected_chemical, atol=2e-15)
    for field, name, scale in (
        ("weight_feature_hidden", "input_weight", gate[None, :]),
        ("bias_hidden", "hidden_bias", gate[None, :]),
        ("weight_hidden_output", "output_weight", gate[:, None]),
        ("bias_output", "output_bias", 1.0),
    ):
        actual = getattr(head, field).detach().numpy()
        np.testing.assert_allclose(
            (values[name] - actual) / 0.03, reward * gradients[name] * scale, atol=2e-12
        )


def test_multilayer_pc_lower_latent_rule_differs_from_fixed_prior_objective_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = np.array([[0.3, -0.2], [-0.5, 0.4]], dtype=np.float64)
    targets = np.array([[1.0], [0.0]], dtype=np.float64)
    model = PredictiveCodingNetwork(input_dim=2, hidden_dim=2, hidden_dims=(2, 3), seed=59)
    lower_prior = np.tanh(inputs @ model._hidden_weights[0] + model._hidden_biases[0])
    upper_prior = np.tanh(lower_prior @ model._hidden_weights[1] + model._hidden_biases[1])
    output_weight = model.weight_hidden_output.copy()
    output_bias = model.bias_output.copy()
    initial_output = 1.0 / (1.0 + np.exp(-(upper_prior @ output_weight + output_bias)))
    upper_after_first = upper_prior - 0.2 * ((initial_output - targets) @ output_weight.T)
    upper_error = upper_after_first - upper_prior
    lower_topdown = upper_error @ model._hidden_weights[1].T
    assert float(np.linalg.norm(lower_topdown)) > 1e-6

    lower_at_second = lower_prior.copy()

    def fixed_prior_diagnostic() -> float:
        logits = upper_after_first @ output_weight + output_bias
        bce = float(np.mean(np.logaddexp(0.0, logits) - targets * logits))
        hidden_penalty = (np.sum((lower_at_second - lower_prior) ** 2) + np.sum(upper_error**2)) / (
            2 * len(inputs) * 5
        )
        return float(bce + hidden_penalty)

    diagnostic_gradient = finite_difference(lower_at_second, fixed_prior_diagnostic)
    np.testing.assert_allclose(diagnostic_gradient, 0.0, atol=2e-10)
    observed_states: list[Array] = []
    original_record = model._record_hidden_traffic

    def capture(states: list[Array]) -> None:
        observed_states.extend(state.copy() for state in states)
        original_record(states)

    monkeypatch.setattr(model, "_record_hidden_traffic", capture)
    model.train_epoch(
        inputs, targets, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.2
    )

    np.testing.assert_allclose(observed_states[0], lower_prior - 0.2 * lower_topdown, atol=2e-15)
    assert not np.allclose(lower_topdown, diagnostic_gradient)
