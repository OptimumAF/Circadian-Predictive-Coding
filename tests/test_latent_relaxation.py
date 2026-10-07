"""Small-step relaxation contracts, independent of reported diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pytest

from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork

Array = NDArray[np.float64]


def _binary_fixture() -> tuple[Array, Array, dict[str, Array]]:
    features = np.array([[0.3, -0.2], [-0.5, 0.4]], dtype=np.float64)
    targets = np.array([[1.0], [0.0]], dtype=np.float64)
    values = {
        "input_weight": np.array([[0.13, -0.21], [-0.17, 0.05]], dtype=np.float64),
        "hidden_bias": np.array([[0.12, 0.14]], dtype=np.float64),
        "output_weight": np.array([[0.2], [-0.3]], dtype=np.float64),
        "output_bias": np.array([[0.07]], dtype=np.float64),
    }
    return features, targets, values


def _numpy_model(kind: str) -> PredictiveCodingNetwork | CircadianPredictiveCodingNetwork:
    if kind == "ordinary":
        return PredictiveCodingNetwork(input_dim=2, hidden_dim=2, seed=73)
    return CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=2,
        seed=73,
        min_hidden_dim=2,
        circadian_config=CircadianConfig(min_plasticity=1.0, replay_steps=0),
    )


def _set_numpy_parameters(
    model: PredictiveCodingNetwork | CircadianPredictiveCodingNetwork,
    values: dict[str, Array],
) -> None:
    for field, name in (
        ("weight_input_hidden", "input_weight"),
        ("bias_hidden", "hidden_bias"),
        ("weight_hidden_output", "output_weight"),
        ("bias_output", "output_bias"),
    ):
        np.copyto(getattr(model, field), values[name])


def _binary_autograd_trace(
    features: Array,
    targets: Array,
    values: dict[str, Array],
    *,
    steps: int,
    step_size: float,
) -> tuple[Array, list[float], list[float]]:
    torch = pytest.importorskip("torch")
    prior = np.tanh(features @ values["input_weight"] + values["hidden_bias"])
    state = torch.tensor(prior, dtype=torch.float64)
    prior_tensor = state.clone()
    output_weight = torch.tensor(values["output_weight"], dtype=torch.float64)
    output_bias = torch.tensor(values["output_bias"], dtype=torch.float64)
    target_tensor = torch.tensor(targets, dtype=torch.float64)
    losses: list[float] = []
    gradient_norms: list[float] = []
    for iteration in range(steps + 1):
        state = state.detach().requires_grad_(True)
        logits = state @ output_weight + output_bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target_tensor)
        loss = loss + torch.sum((state - prior_tensor).square()) / (2 * len(features))
        gradient = torch.autograd.grad(loss, state)[0]
        losses.append(float(loss.item()))
        gradient_norms.append(float(torch.linalg.vector_norm(gradient).item()))
        if iteration < steps:
            state = state - step_size * len(features) * gradient
    return state.detach().numpy(), losses, gradient_norms


@pytest.mark.parametrize("kind", ["ordinary", "circadian"])
@pytest.mark.parametrize("steps", [1, 60])
def test_numpy_one_hidden_relaxation_matches_autograd_and_decreases_local_objective(
    kind: str, steps: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    features, targets, values = _binary_fixture()
    assert 0.2 <= 1.0 / (1.0 + np.linalg.norm(values["output_weight"], ord=2) ** 2)
    expected, losses, gradient_norms = _binary_autograd_trace(
        features, targets, values, steps=steps, step_size=0.2
    )
    assert all(right < left for left, right in zip(losses, losses[1:]))
    if steps == 60:
        assert gradient_norms[-1] < 1e-4 * gradient_norms[0]

    model = _numpy_model(kind)
    _set_numpy_parameters(model, values)
    observed: list[Array] = []
    original = model._record_hidden_traffic

    def record(state: Array | list[Array]) -> None:
        observed.append((state[-1] if isinstance(state, list) else state).copy())
        original(state)  # type: ignore[arg-type]

    monkeypatch.setattr(model, "_record_hidden_traffic", record)
    result = model.train_epoch(
        features, targets, learning_rate=0.03, inference_steps=steps, inference_learning_rate=0.2
    )
    assert len(observed) == 1
    np.testing.assert_allclose(observed[0], expected, atol=2e-12)
    assert np.isfinite(result.energy)


def _torch_head(kind: str):
    torch = pytest.importorskip("torch")
    from src.core.resnet50_variants import (
        CircadianHeadConfig,
        CircadianPredictiveCodingHead,
        PredictiveCodingHead,
    )

    if kind == "ordinary":
        head = PredictiveCodingHead(2, 2, 2, torch.device("cpu"), seed=79)
    else:
        head = CircadianPredictiveCodingHead(
            feature_dim=2,
            hidden_dim=2,
            num_classes=2,
            device=torch.device("cpu"),
            seed=79,
            config=CircadianHeadConfig(min_plasticity=1.0),
            min_hidden_dim=2,
            max_hidden_dim=4,
        )
    values = {
        "weight_feature_hidden": [[0.25, 0.2], [0.3, 0.1]],
        "bias_hidden": [[0.1, 0.2]],
        "weight_hidden_output": [[0.25, -0.1], [-0.2, 0.3]],
        "bias_output": [[0.05, -0.02]],
    }
    for name, value in values.items():
        setattr(head, name, torch.tensor(value, dtype=torch.float64))
    return head


def _multiclass_autograd_trace(head, features, targets, *, steps: int, step_size: float):
    torch = pytest.importorskip("torch")
    prior = torch.tanh(features @ head.weight_feature_hidden + head.bias_hidden)
    state = prior.clone()
    losses: list[float] = []
    gradient_norms: list[float] = []
    for iteration in range(steps + 1):
        state = state.detach().requires_grad_(True)
        logits = state @ head.weight_hidden_output + head.bias_output
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss = loss + torch.sum((state - prior).square()) / (2 * len(features))
        gradient = torch.autograd.grad(loss, state)[0]
        losses.append(float(loss.item()))
        gradient_norms.append(float(torch.linalg.vector_norm(gradient).item()))
        if iteration < steps:
            state = state - step_size * len(features) * gradient
    return state.detach(), losses, gradient_norms


@pytest.mark.parametrize("kind", ["ordinary", "circadian"])
@pytest.mark.parametrize("steps", [1, 60])
def test_torch_one_hidden_relaxation_matches_autograd_and_decreases_local_objective(
    kind: str, steps: int
) -> None:
    torch = pytest.importorskip("torch")
    head = _torch_head(kind)
    features = torch.tensor([[0.2, 0.4]], dtype=torch.float64)
    targets = torch.tensor([1], dtype=torch.long)
    spectral_norm = float(torch.linalg.matrix_norm(head.weight_hidden_output, ord=2).item())
    assert 0.2 <= 1.0 / (1.0 + spectral_norm**2)
    expected, losses, gradient_norms = _multiclass_autograd_trace(
        head, features, targets, steps=steps, step_size=0.2
    )
    assert torch.all(expected > 0)
    assert all(right < left for left, right in zip(losses, losses[1:]))
    if steps == 60:
        assert gradient_norms[-1] < 1e-4 * gradient_norms[0]

    diagnostic = head.train_step(
        features, targets, learning_rate=0.03, inference_steps=steps, inference_learning_rate=0.2
    )
    # One positive-state row makes the public traffic summary the inferred state.
    torch.testing.assert_close(head.mean_hidden_traffic(), expected.squeeze(0), atol=2e-12, rtol=0)
    assert np.isfinite(diagnostic)


@pytest.mark.parametrize("kind", ["ordinary", "circadian"])
@pytest.mark.parametrize("output_scale", [0.0, 1e-8])
def test_numpy_zero_or_small_output_drive_keeps_latent_near_prior(
    kind: str, output_scale: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    features, _, values = _binary_fixture()
    targets = np.ones((len(features), 1), dtype=np.float64)
    values["output_weight"][:] = output_scale
    values["output_bias"][:] = 0.0
    model = _numpy_model(kind)
    _set_numpy_parameters(model, values)
    prior = np.tanh(features @ values["input_weight"] + values["hidden_bias"])
    observed: list[Array] = []

    def record(state: Array | list[Array]) -> None:
        observed.append((state[-1] if isinstance(state, list) else state).copy())

    monkeypatch.setattr(model, "_record_hidden_traffic", record)
    model.train_epoch(features, targets, 0.03, 60, 0.2)
    displacement = float(np.max(np.abs(observed[0] - prior)))
    if output_scale == 0.0:
        np.testing.assert_array_equal(observed[0], prior)
    else:
        assert 0.0 < displacement < output_scale


@pytest.mark.parametrize("kind", ["ordinary", "circadian"])
@pytest.mark.parametrize("output_scale", [0.0, 1e-8])
def test_torch_zero_or_small_output_drive_keeps_latent_near_prior(
    kind: str, output_scale: float
) -> None:
    torch = pytest.importorskip("torch")
    head = _torch_head(kind)
    head.weight_hidden_output.zero_()
    head.weight_hidden_output[:, 0] = output_scale
    features = torch.tensor([[0.2, 0.4]], dtype=torch.float64)
    prior = torch.tanh(features @ head.weight_feature_hidden + head.bias_hidden)
    head.train_step(features, torch.tensor([1]), 0.03, 60, 0.2)
    displacement = float(torch.max(torch.abs(head.mean_hidden_traffic() - prior.squeeze(0))).item())
    if output_scale == 0.0:
        torch.testing.assert_close(head.mean_hidden_traffic(), prior.squeeze(0), atol=0, rtol=0)
    else:
        assert 0.0 < displacement < output_scale


@pytest.mark.parametrize("steps", [1, 60])
def test_multilayer_numpy_pc_follows_documented_fixed_prior_update_and_converges_locally(
    steps: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    features, targets, _ = _binary_fixture()
    model = PredictiveCodingNetwork(input_dim=2, hidden_dim=2, hidden_dims=(2, 2), seed=83)
    lower_weight = np.array([[0.1, 0.2], [-0.2, 0.1]], dtype=np.float64)
    upper_weight = np.array([[0.15, -0.08], [0.05, 0.12]], dtype=np.float64)
    output_weight = np.array([[0.2], [-0.25]], dtype=np.float64)
    np.copyto(model._hidden_weights[0], lower_weight)
    np.copyto(model._hidden_weights[1], upper_weight)
    np.copyto(model._hidden_biases[0], [[0.1, 0.2]])
    np.copyto(model._hidden_biases[1], [[0.03, 0.06]])
    np.copyto(model.weight_hidden_output, output_weight)
    np.copyto(model.bias_output, [[0.04]])
    lower_prior = np.tanh(features @ lower_weight + model._hidden_biases[0])
    upper_prior = np.tanh(lower_prior @ upper_weight + model._hidden_biases[1])
    lower_state = lower_prior.copy()
    upper_state = upper_prior.copy()

    def gradients() -> tuple[Array, Array]:
        logits = upper_state @ output_weight + model.bias_output
        prediction = 1.0 / (1.0 + np.exp(-logits))
        upper_error = upper_state - upper_prior
        return (
            lower_state - lower_prior + upper_error @ upper_weight.T,
            upper_error + (prediction - targets) @ output_weight.T,
        )

    initial_norm = float(np.linalg.norm(np.concatenate([part.ravel() for part in gradients()])))
    for _ in range(steps):
        lower_gradient, upper_gradient = gradients()
        lower_state -= 0.2 * lower_gradient
        upper_state -= 0.2 * upper_gradient
    final_norm = float(np.linalg.norm(np.concatenate([part.ravel() for part in gradients()])))
    if steps == 60:
        assert final_norm < 1e-4 * initial_norm

    observed: list[Array] = []
    original = model._record_hidden_traffic

    def record(states: list[Array]) -> None:
        observed.extend(state.copy() for state in states)
        original(states)

    monkeypatch.setattr(model, "_record_hidden_traffic", record)
    model.train_epoch(features, targets, 0.03, steps, 0.2)
    assert len(observed) == 2
    np.testing.assert_allclose(observed[0], lower_state, atol=2e-12)
    np.testing.assert_allclose(observed[1], upper_state, atol=2e-12)


@pytest.mark.parametrize("kind", ["ordinary", "circadian"])
def test_numpy_invalid_or_nonfinite_relaxation_inputs_fail_before_update(kind: str) -> None:
    features, targets, values = _binary_fixture()
    for changed_features, changed_targets, rate, steps in (
        (np.full_like(features, np.nan), targets, 0.2, 2),
        (features, np.full_like(targets, np.inf), 0.2, 2),
        (features, targets, float("nan"), 2),
        (features, targets, 0.2, 0),
    ):
        model = _numpy_model(kind)
        _set_numpy_parameters(model, values)
        before = model.weight_hidden_output.copy()
        with pytest.raises(ValueError):
            model.train_epoch(changed_features, changed_targets, 0.03, steps, rate)
        np.testing.assert_array_equal(model.weight_hidden_output, before)


@pytest.mark.parametrize("kind", ["ordinary", "circadian"])
def test_torch_invalid_or_nonfinite_relaxation_inputs_fail_before_update(kind: str) -> None:
    torch = pytest.importorskip("torch")
    for features, rate, steps in (
        (torch.tensor([[float("nan"), 0.4]], dtype=torch.float64), 0.2, 2),
        (torch.tensor([[0.2, 0.4]], dtype=torch.float64), float("nan"), 2),
        (torch.tensor([[0.2, 0.4]], dtype=torch.float64), 0.2, 0),
    ):
        head = _torch_head(kind)
        before = head.weight_hidden_output.clone()
        with pytest.raises(ValueError):
            head.train_step(features, torch.tensor([1]), 0.03, steps, rate)
        torch.testing.assert_close(head.weight_hidden_output, before, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["ordinary", "circadian"])
def test_numpy_nonfinite_latent_is_detected_before_weight_update(kind: str) -> None:
    model = _numpy_model(kind)
    model.weight_input_hidden.fill(0.0)
    model.bias_hidden.fill(0.0)
    model.weight_hidden_output.fill(1e308)
    features = np.zeros((1, 2), dtype=np.float64)
    targets = np.ones((1, 1), dtype=np.float64)
    before = model.weight_hidden_output.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(FloatingPointError, match="latent"):
            model.train_epoch(features, targets, 0.03, 1, 1e308)
    np.testing.assert_array_equal(model.weight_hidden_output, before)


@pytest.mark.parametrize("kind", ["ordinary", "circadian"])
def test_torch_nonfinite_latent_is_detected_before_weight_update(kind: str) -> None:
    torch = pytest.importorskip("torch")
    head = _torch_head(kind)
    head.weight_feature_hidden.zero_()
    head.bias_hidden.zero_()
    head.weight_hidden_output = torch.tensor([[1e308, 0.0], [1e308, 0.0]], dtype=torch.float64)
    features = torch.zeros((1, 2), dtype=torch.float64)
    before = head.weight_hidden_output.clone()
    with pytest.raises(FloatingPointError, match="latent"):
        head.train_step(features, torch.tensor([1]), 0.03, 1, 1e308)
    torch.testing.assert_close(head.weight_hidden_output, before, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["ordinary", "circadian"])
def test_numpy_overflowed_prior_is_detected_before_tanh_masks_it(kind: str) -> None:
    model = _numpy_model(kind)
    model.weight_input_hidden.fill(1e308)
    features = np.ones((1, 2), dtype=np.float64)
    targets = np.ones((1, 1), dtype=np.float64)
    before = model.weight_hidden_output.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(FloatingPointError, match="latent prior"):
            model.train_epoch(features, targets, 0.03, 1, 0.2)
    np.testing.assert_array_equal(model.weight_hidden_output, before)


def test_numpy_circadian_overflowed_earlier_prior_is_detected() -> None:
    model = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=2,
        hidden_dims=(2, 2),
        seed=89,
        min_hidden_dim=2,
        circadian_config=CircadianConfig(min_plasticity=1.0, replay_steps=0),
    )
    model._pre_hidden_weights[0].fill(1e308)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(FloatingPointError, match="latent prior"):
            model.train_epoch(np.ones((1, 2)), np.ones((1, 1)), 0.03, 1, 0.2)


@pytest.mark.parametrize("kind", ["ordinary", "circadian"])
def test_torch_overflowed_prior_is_detected_before_tanh_masks_it(kind: str) -> None:
    torch = pytest.importorskip("torch")
    head = _torch_head(kind)
    head.weight_feature_hidden.fill_(1e308)
    features = torch.ones((1, 2), dtype=torch.float64)
    before = head.weight_hidden_output.clone()
    with pytest.raises(FloatingPointError, match="latent"):
        head.train_step(features, torch.tensor([1]), 0.03, 1, 0.2)
    torch.testing.assert_close(head.weight_hidden_output, before, atol=0, rtol=0)
