"""Independent residual scaling equations, raw SGD, and owned-state controls.

The Torch oracle differentiates the complete effective residual network's
MSE/s quotient. Tiny fixed fixtures verify formulas, not residual learning benefit.
Exactly 24 public residual update calls occur in a full run of this file.
"""

from dataclasses import FrozenInstanceError
from math import sqrt
from typing import Any

import numpy as np
import pytest
import torch

from src.core.linear_pc_reference import evaluate_linear_equilibrium
from src.core.linear_pc_scaling import (
    LinearScalingConfig,
    effective_linear_weights,
    initialize_linear_scaling,
)
from src.core.residual_linear_pc import (
    LinearResidualConfig,
    LinearResidualState,
    effective_residual_weights,
    initialize_residual_linear,
    step_residual_linear_pc,
)

SCHEMES = (
    ("legacy_gaussian", 1.0),
    ("fan_in", 1.0),
    ("mean_field", 0.5),
    ("mean_field", 1.0),
    ("mean_field", 2.0),
)


def expected_scales(scheme, gamma, hidden):
    if scheme != "mean_field":
        return (1.0,) * (hidden + 1)
    return (1 / sqrt(2), *((1 / sqrt(3 * (hidden + 1)),) * (hidden - 1)), 1 / (gamma * 3))


def torch_raw_quotient(raw, scales, x, y):
    tensors = tuple(torch.tensor(w, dtype=torch.float64, requires_grad=True) for w in raw)
    effective = []
    for index, (weight, scale) in enumerate(zip(tensors, scales, strict=True)):
        branch = weight * scale
        effective.append(
            branch + torch.eye(weight.shape[0], dtype=torch.float64)
            if 0 < index < len(raw) - 1
            else branch
        )
    prediction = torch.tensor(x, dtype=torch.float64)
    for weight in effective:
        prediction = prediction @ weight
    mse = ((prediction - torch.tensor(y, dtype=torch.float64)) ** 2).sum() / (2 * len(x))
    product = torch.ones((1, 1), dtype=torch.float64)
    rescaling = torch.ones((), dtype=torch.float64)
    for weight in reversed(effective[1:]):
        product = weight @ product
        rescaling = rescaling + (product**2).sum()
    energy = mse / rescaling
    return energy, mse, rescaling, torch.autograd.grad(energy, tensors)


@pytest.mark.parametrize("hidden", (1, 2, 4))
@pytest.mark.parametrize("scheme,gamma", SCHEMES)
def test_should_expose_residual_factors_and_shared_raw_standard_normal_draws(hidden, scheme, gamma):
    config = LinearResidualConfig(scheme, 2, 3, hidden, gamma)
    assert config.depth == hidden + 1
    scales = expected_scales(scheme, gamma, hidden)
    np.testing.assert_allclose(config.forward_scales, scales, rtol=1e-15)
    stddevs = (
        (1.0,) * (hidden + 1)
        if scheme == "mean_field"
        else (
            (0.5,) * (hidden + 1)
            if scheme == "legacy_gaussian"
            else (1 / sqrt(2), *((1 / sqrt(3),) * hidden))
        )
    )
    np.testing.assert_allclose(config.initial_stddevs, stddevs, rtol=1e-15)
    factor = gamma * gamma * 3 if scheme == "mean_field" else 1.0
    assert config.raw_learning_rate_factor == pytest.approx(factor)
    np.testing.assert_allclose(
        config.effective_learning_rate_factors,
        [factor * scale * scale for scale in scales],
        rtol=2e-15,
    )
    if scheme == "mean_field":
        np.testing.assert_allclose(
            config.effective_learning_rate_factors,
            [gamma**2 * 3 / 2, *((gamma**2 / (hidden + 1),) * (hidden - 1)), 1 / 3],
            rtol=2e-15,
        )
    state = initialize_residual_linear(config, 41)
    mlp = initialize_linear_scaling(LinearScalingConfig(scheme, 2, 3, hidden, gamma), 41)
    for a, b in zip(state.raw_weights, mlp.raw_weights, strict=True):
        np.testing.assert_array_equal(a, b)
    rng = np.random.default_rng(41)
    shapes = ((2, 3), *(((3, 3),) * (hidden - 1)), (3, 1))
    for raw, shape, stddev in zip(state.raw_weights, shapes, stddevs, strict=True):
        np.testing.assert_array_equal(raw, rng.standard_normal(shape) * stddev)
    with pytest.raises(FrozenInstanceError):
        setattr(config, "width", 5)


@pytest.mark.parametrize("hidden", (1, 2, 4))
@pytest.mark.parametrize("scheme,gamma", SCHEMES)
def test_should_follow_complete_residual_torch_quotient_and_raw_chain_rule(hidden, scheme, gamma):
    config = LinearResidualConfig(scheme, 2, 3, hidden, gamma)
    state = initialize_residual_linear(config, 41)
    original = tuple(w.copy() for w in state.raw_weights)
    scales = expected_scales(scheme, gamma, hidden)
    x, y = np.array([[0.4, -0.7], [-0.3, 0.2], [1.1, 0.6]]), np.array([[-1.0], [0.3], [1.4]])
    energy, mse, rescaling, raw_gradients = torch_raw_quotient(original, scales, x, y)
    result = step_residual_linear_pc(state, x, y, 0.01)
    assert result.evaluation.energy == pytest.approx(energy.item(), abs=2e-13, rel=2e-13)
    assert result.evaluation.mse == pytest.approx(mse.item(), abs=2e-13, rel=2e-13)
    assert result.evaluation.rescaling == pytest.approx(rescaling.item(), abs=2e-13, rel=2e-13)
    assert result.state is not state and result.state.config == config
    new_effective = effective_residual_weights(result.state)
    old_effective = effective_residual_weights(state)
    for index, (old, new, raw_gradient, effective_gradient, scale, effective_rate) in enumerate(
        zip(
            original,
            result.state.raw_weights,
            raw_gradients,
            result.evaluation.weight_gradients,
            scales,
            config.effective_learning_rate_factors,
            strict=True,
        )
    ):
        gradient = raw_gradient.detach().numpy()
        np.testing.assert_allclose(gradient, scale * effective_gradient, atol=2e-13, rtol=2e-12)
        np.testing.assert_allclose(
            new, old - 0.01 * config.raw_learning_rate_factor * gradient, atol=2e-13, rtol=2e-12
        )
        np.testing.assert_allclose(
            new_effective[index],
            old_effective[index] - 0.01 * effective_rate * effective_gradient,
            atol=2e-13,
            rtol=2e-12,
        )
        # The fixed middle identity remains after the raw branch update.
        expected = scale * new + (np.eye(3) if 0 < index < len(original) - 1 else 0)
        np.testing.assert_array_equal(new_effective[index], expected)
        for source in state.raw_weights:
            assert not np.shares_memory(source, new)
        new.fill(999)
    for observed, expected in zip(state.raw_weights, original, strict=True):
        np.testing.assert_array_equal(observed, expected)
    with pytest.raises(FrozenInstanceError):
        setattr(result, "state", state)


@pytest.mark.parametrize("scheme,gamma", SCHEMES)
def test_should_have_identity_only_in_middle_and_match_mlp_exactly_without_middle(scheme, gamma):
    config = LinearResidualConfig(scheme, 2, 2, 3, gamma)
    raw = (np.full((2, 2), 0.25), np.zeros((2, 2)), np.full((2, 2), -0.2), np.full((2, 1), 0.4))
    state = LinearResidualState(config, raw)
    effective = effective_residual_weights(state)
    for index, (actual, source, scale) in enumerate(
        zip(effective, raw, config.forward_scales, strict=True)
    ):
        expected = source * scale + (np.eye(2) if index in (1, 2) else 0)
        np.testing.assert_array_equal(actual, expected)
        assert not np.shares_memory(actual, state.raw_weights[index])
    for weight in raw:
        weight.fill(123)
    np.testing.assert_array_equal(state.raw_weights[1], np.zeros((2, 2)))
    effective[1].fill(999)
    np.testing.assert_array_equal(effective_residual_weights(state)[1], np.eye(2))
    with pytest.raises(FrozenInstanceError):
        setattr(state, "raw_weights", raw)
    residual_one = initialize_residual_linear(LinearResidualConfig(scheme, 2, 3, 1, gamma), 41)
    mlp_one = initialize_linear_scaling(LinearScalingConfig(scheme, 2, 3, 1, gamma), 41)
    for a, b in zip(
        effective_residual_weights(residual_one), effective_linear_weights(mlp_one), strict=True
    ):
        np.testing.assert_array_equal(a, b)
    x, y = np.ones((2, 2)), np.array([[-1.0], [2.0]])
    residual_evaluation = evaluate_linear_equilibrium(
        effective_residual_weights(residual_one), x, y
    )
    mlp_evaluation = evaluate_linear_equilibrium(effective_linear_weights(mlp_one), x, y)
    assert residual_evaluation.energy == mlp_evaluation.energy
    assert residual_evaluation.mse == mlp_evaluation.mse
    assert residual_evaluation.rescaling == mlp_evaluation.rescaling
    for ga, gb in zip(
        residual_evaluation.weight_gradients, mlp_evaluation.weight_gradients, strict=True
    ):
        np.testing.assert_array_equal(ga, gb)


@pytest.mark.parametrize(
    "changes",
    (
        {"scheme": "unknown"},
        {"scheme": 1},
        {"input_dim": True},
        {"input_dim": 0},
        {"input_dim": 2.0},
        {"width": np.int64(3)},
        {"width": -1},
        {"hidden_layers": 0},
        {"hidden_layers": 2.0},
        {"gamma0": True},
        {"gamma0": 0},
        {"gamma0": -1},
        {"gamma0": np.float64(1)},
        {"gamma0": float("nan")},
        {"gamma0": float("inf")},
        {"gamma0": 10**1000},
        {"gamma0": 1e200},
        {"gamma0": 1e-200},
        {"width": 1001, "hidden_layers": 2},
        {"hidden_layers": 10**1000},
        {"input_dim": 1_000_001, "width": 1},
    ),
)
def test_should_refuse_invalid_config_and_allocation_limit_before_initialization(changes: Any):
    kwargs: dict[str, Any] = dict(scheme="mean_field", input_dim=2, width=3, hidden_layers=2)
    kwargs.update(changes)
    with pytest.raises(ValueError):
        LinearResidualConfig(**kwargs)


@pytest.mark.parametrize("scheme", ("legacy_gaussian", "fan_in"))
def test_should_require_unit_gamma_in_controls(scheme):
    with pytest.raises(ValueError):
        LinearResidualConfig(scheme, 2, 3, 2, 2)


@pytest.mark.parametrize("seed", (True, -1, 41.0, np.int64(41), "41", None))
def test_should_require_native_nonnegative_integer_seed(seed: Any):
    with pytest.raises(ValueError):
        initialize_residual_linear(LinearResidualConfig("mean_field", 2, 3, 1), seed)


def malformed_states():
    config = LinearResidualConfig("fan_in", 2, 3, 1)
    weights = (np.ones((2, 3)), np.ones((3, 1)))
    return [
        (None, weights),
        (LinearScalingConfig("fan_in", 2, 3, 1), weights),
        (config, list(weights)),
        (config, ()),
        (config, weights[:1]),
        (config, (weights[0].tolist(), weights[1])),
        (config, (weights[0].astype(np.float32), weights[1])),
        (config, (weights[0].astype(np.int64), weights[1])),
        (config, (np.ones((3, 3)), weights[1])),
        (config, (np.ones((2, 3, 1)), weights[1])),
        (config, (np.zeros((2, 0)), weights[1])),
        (config, (weights[0], np.zeros((3, 2)))),
        (config, (np.full((2, 3), np.nan), weights[1])),
        (config, (weights[0], np.full((3, 1), np.inf))),
    ]


@pytest.mark.parametrize("config,weights", malformed_states())
def test_should_refuse_malformed_state_families_arrays_and_shapes(config: Any, weights: Any):
    with pytest.raises(ValueError):
        LinearResidualState(config, weights)


@pytest.mark.parametrize("rate", (True, 0, float("nan"), np.float64(0.1)))
def test_should_refuse_invalid_rate_without_mutation(rate: Any):
    state = initialize_residual_linear(LinearResidualConfig("fan_in", 2, 3, 1), 41)
    original = tuple(w.copy() for w in state.raw_weights)
    with pytest.raises(ValueError):
        step_residual_linear_pc(state, np.ones((1, 2)), np.ones((1, 1)), rate)
    for observed, expected in zip(state.raw_weights, original, strict=True):
        np.testing.assert_array_equal(observed, expected)


def test_should_revalidate_mutated_state_before_effective_conversion_and_update():
    state = initialize_residual_linear(LinearResidualConfig("mean_field", 2, 3, 1), 41)
    state.raw_weights[0][0, 0] = np.nan
    with pytest.raises(ValueError):
        effective_residual_weights(state)
    with pytest.raises(ValueError):
        step_residual_linear_pc(state, np.ones((1, 2)), np.ones((1, 1)), 0.01)


def test_should_refuse_foreign_state_family_before_native_update():
    invalid: Any = initialize_linear_scaling(LinearScalingConfig("fan_in", 2, 3, 1), 41)
    with pytest.raises(ValueError):
        effective_residual_weights(invalid)
    with pytest.raises(ValueError):
        step_residual_linear_pc(invalid, np.ones((1, 2)), np.ones((1, 1)), 0.01)


def test_should_refuse_malformed_batch_atomically():
    state = initialize_residual_linear(LinearResidualConfig("fan_in", 2, 3, 1), 41)
    original = tuple(w.copy() for w in state.raw_weights)
    with pytest.raises(ValueError):
        step_residual_linear_pc(state, np.ones((1, 2), dtype=np.float32), np.ones((1, 1)), 0.01)
    for observed, expected in zip(state.raw_weights, original, strict=True):
        np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("feature", (1.0, 10.0))
def test_should_refuse_nonfinite_candidate_or_equilibrium_atomically(feature):
    state = LinearResidualState(
        LinearResidualConfig("fan_in", 1, 1, 1), (np.ones((1, 1)), np.ones((1, 1)))
    )
    original = tuple(w.copy() for w in state.raw_weights)
    with pytest.raises((ValueError, FloatingPointError)):
        step_residual_linear_pc(state, np.array([[feature]]), np.zeros((1, 1)), 1e308)
    for observed, expected in zip(state.raw_weights, original, strict=True):
        np.testing.assert_array_equal(observed, expected)
