"""Independent raw-parameter scaling and equilibrium SGD controls.

Every reference graph differentiates MSE/s through both numerator and denominator.
Fixed seed 41 and tiny widths exercise formulas, not a scaling experiment.
"""

from dataclasses import FrozenInstanceError, replace
from math import sqrt
from typing import Any

import numpy as np
import pytest
import torch

from src.core.linear_pc_scaling import (
    LinearScalingConfig,
    LinearScalingState,
    effective_linear_weights,
    initialize_linear_scaling,
    step_linear_pc,
)

SCHEMES = (
    ("legacy_gaussian", 1.0),
    ("fan_in", 1.0),
    ("mean_field", 0.5),
    ("mean_field", 1.0),
    ("mean_field", 2.0),
)


def shapes(config):
    dims = (config.input_dim, *((config.width,) * config.hidden_layers), 1)
    return tuple(zip(dims, dims[1:]))


def torch_raw_graph(raw, scales, x, y):
    tensors = tuple(torch.tensor(w, dtype=torch.float64, requires_grad=True) for w in raw)
    weights = tuple(w * a for w, a in zip(tensors, scales, strict=True))
    prediction = torch.tensor(x, dtype=torch.float64)
    for weight in weights:
        prediction = prediction @ weight
    mse = ((prediction - torch.tensor(y, dtype=torch.float64)) ** 2).sum() / (2 * len(x))
    product = torch.ones((1, 1), dtype=torch.float64)
    rescaling = torch.ones((), dtype=torch.float64)
    for weight in reversed(weights[1:]):
        product = weight @ product
        rescaling = rescaling + (product**2).sum()
    energy = mse / rescaling
    return energy, torch.autograd.grad(energy, tensors)


@pytest.mark.parametrize("hidden_layers", (1, 2, 4))
@pytest.mark.parametrize("scheme,gamma", SCHEMES)
def test_should_expose_declared_raw_initialization_forward_and_update_scalings(
    hidden_layers, scheme, gamma
):
    config = LinearScalingConfig(scheme, 2, 3, hidden_layers, gamma)
    if scheme == "mean_field":
        scales = (1 / sqrt(2), *((1 / sqrt(3),) * (hidden_layers - 1)), 1 / (gamma * 3))
        stddevs = (1.0,) * (hidden_layers + 1)
        raw_rate = gamma**2 * 3
    else:
        scales = (1.0,) * (hidden_layers + 1)
        stddevs = (
            (0.5,) * (hidden_layers + 1)
            if scheme == "legacy_gaussian"
            else (1 / sqrt(2), *((1 / sqrt(3),) * hidden_layers))
        )
        raw_rate = 1.0
    np.testing.assert_allclose(config.forward_scales, scales, rtol=1e-15)
    np.testing.assert_allclose(config.initial_stddevs, stddevs, rtol=1e-15)
    assert config.raw_learning_rate_factor == pytest.approx(raw_rate, rel=1e-15)
    np.testing.assert_allclose(
        config.effective_learning_rate_factors, [raw_rate * a * a for a in scales], rtol=2e-15
    )
    with pytest.raises(FrozenInstanceError):
        setattr(config, "width", 7)


@pytest.mark.parametrize("hidden_layers", (1, 2, 4))
def test_should_use_same_per_layer_standard_normal_draws_for_every_scheme(hidden_layers):
    reference_shapes = ((2, 3), *(((3, 3),) * (hidden_layers - 1)), (3, 1))
    rng = np.random.default_rng(41)
    draws = tuple(rng.standard_normal(shape) for shape in reference_shapes)
    for scheme, gamma in SCHEMES:
        config = LinearScalingConfig(scheme, 2, 3, hidden_layers, gamma)
        state = initialize_linear_scaling(config, 41)
        effective = effective_linear_weights(state)
        for raw, weight, draw, stddev, scale in zip(
            state.raw_weights,
            effective,
            draws,
            config.initial_stddevs,
            config.forward_scales,
            strict=True,
        ):
            np.testing.assert_array_equal(raw, draw * stddev)
            np.testing.assert_array_equal(weight, draw * stddev * scale)
            assert not np.shares_memory(raw, weight)


@pytest.mark.parametrize("hidden_layers", (1, 2, 4))
@pytest.mark.parametrize("scheme,gamma", SCHEMES)
def test_should_follow_full_raw_torch_gradient_and_equivalent_effective_step(
    hidden_layers, scheme, gamma
):
    config = LinearScalingConfig(scheme, 2, 3, hidden_layers, gamma)
    state = initialize_linear_scaling(config, 41)
    original = tuple(w.copy() for w in state.raw_weights)
    x = np.array([[0.4, -0.7], [-0.3, 0.2], [1.1, 0.6]])
    y = np.array([[-1.0], [0.3], [1.4]])
    energy, raw_gradients = torch_raw_graph(state.raw_weights, config.forward_scales, x, y)
    rate = 0.01
    result = step_linear_pc(state, x, y, rate)
    assert result.evaluation.energy == pytest.approx(energy.item(), abs=2e-14)
    assert result.state is not state
    assert result.state.config == config
    for old, new, raw_gradient, effective_gradient, scale, effective_rate in zip(
        original,
        result.state.raw_weights,
        raw_gradients,
        result.evaluation.weight_gradients,
        config.forward_scales,
        config.effective_learning_rate_factors,
        strict=True,
    ):
        raw_gradient_array = raw_gradient.detach().numpy()
        np.testing.assert_allclose(
            raw_gradient_array, scale * effective_gradient, atol=2e-14, rtol=2e-12
        )
        expected_raw = old - rate * config.raw_learning_rate_factor * raw_gradient_array
        np.testing.assert_allclose(new, expected_raw, atol=2e-14, rtol=2e-12)
        np.testing.assert_allclose(
            new * scale,
            old * scale - rate * effective_rate * effective_gradient,
            atol=2e-14,
            rtol=2e-12,
        )
        for source in state.raw_weights:
            assert not np.shares_memory(source, new)
    for actual, expected in zip(state.raw_weights, original, strict=True):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(FrozenInstanceError):
        setattr(result, "state", state)


def test_should_copy_explicit_raw_state_and_detach_returned_effective_arrays():
    config = LinearScalingConfig("mean_field", 2, 3, 2, 2)
    raw = tuple(np.full(shape, 0.25 + layer) for layer, shape in enumerate(shapes(config)))
    state = LinearScalingState(config, raw)
    expected = tuple(w.copy() for w in raw)
    for supplied in raw:
        supplied.fill(123)
    effective = effective_linear_weights(state)
    for index, (observed, original, scale) in enumerate(
        zip(state.raw_weights, expected, config.forward_scales, strict=True)
    ):
        np.testing.assert_array_equal(observed, original)
        np.testing.assert_array_equal(effective[index], original * scale)
        assert not np.shares_memory(observed, effective[index])
    effective[0].fill(999)
    for observed, original in zip(state.raw_weights, expected, strict=True):
        np.testing.assert_array_equal(observed, original)
    with pytest.raises(FrozenInstanceError):
        setattr(state, "raw_weights", raw)


@pytest.mark.parametrize(
    "changes",
    [
        {"scheme": "unknown"},
        {"scheme": 1},
        {"input_dim": True},
        {"input_dim": 0},
        {"input_dim": 2.0},
        {"input_dim": np.int64(2)},
        {"width": False},
        {"width": -1},
        {"hidden_layers": 0},
        {"hidden_layers": 1.0},
        {"gamma0": True},
        {"gamma0": 0},
        {"gamma0": -1.0},
        {"gamma0": float("nan")},
        {"gamma0": float("inf")},
        {"gamma0": np.float64(1.0)},
        {"gamma0": 10**1000},
        {"gamma0": 1e-200},
        {"gamma0": 1e200},
        {"width": 1001, "hidden_layers": 2},
        {"input_dim": 1_000_001, "width": 1},
        {"hidden_layers": 10**1000},
    ],
)
def test_should_refuse_invalid_or_unrepresentable_config_before_initialization(changes: Any):
    kwargs: dict[str, Any] = dict(scheme="mean_field", input_dim=2, width=3, hidden_layers=2)
    kwargs.update(changes)
    with pytest.raises(ValueError):
        LinearScalingConfig(**kwargs)


@pytest.mark.parametrize("scheme", ("legacy_gaussian", "fan_in"))
def test_should_refuse_nonunit_gamma_for_control_schemes(scheme):
    with pytest.raises(ValueError):
        LinearScalingConfig(scheme, 2, 3, 1, 2.0)


@pytest.mark.parametrize("seed", (True, -1, 41.0, np.int64(41), "41", None))
def test_should_refuse_invalid_initialization_seed(seed: Any):
    with pytest.raises(ValueError):
        initialize_linear_scaling(LinearScalingConfig("mean_field", 2, 3, 1), seed)


def malformed_states():
    config = LinearScalingConfig("fan_in", 2, 3, 1)
    weights = (np.ones((2, 3)), np.ones((3, 1)))
    return [
        (None, weights),
        (replace(config, width=4), weights),
        (config, list(weights)),
        (config, ()),
        (config, weights[:1]),
        (config, (weights[0].tolist(), weights[1])),
        (config, (weights[0].astype(np.float32), weights[1])),
        (config, (weights[0].astype(np.int64), weights[1])),
        (config, (np.ones((2, 3, 1)), weights[1])),
        (config, (np.ones((2, 0)), weights[1])),
        (config, (weights[0], np.ones((3, 2)))),
        (config, (np.full((2, 3), np.nan), weights[1])),
        (config, (weights[0], np.full((3, 1), np.inf))),
    ]


@pytest.mark.parametrize("config,raw_weights", malformed_states())
def test_should_refuse_malformed_scaling_state(config: Any, raw_weights: Any):
    with pytest.raises(ValueError):
        LinearScalingState(config, raw_weights)


@pytest.mark.parametrize(
    "rate", (True, 0, -1, float("nan"), float("inf"), np.float64(0.1), "0.1", 10**1000)
)
def test_should_refuse_invalid_rate_without_mutation(rate: Any):
    state = initialize_linear_scaling(LinearScalingConfig("fan_in", 2, 3, 1), 41)
    original = tuple(w.copy() for w in state.raw_weights)
    with pytest.raises(ValueError):
        step_linear_pc(state, np.ones((1, 2)), np.ones((1, 1)), rate)
    for actual, expected in zip(state.raw_weights, original, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_should_revalidate_arrays_mutated_after_state_construction():
    state = initialize_linear_scaling(LinearScalingConfig("mean_field", 2, 3, 1), 41)
    state.raw_weights[-1][0, 0] = np.nan
    with pytest.raises(ValueError):
        effective_linear_weights(state)
    with pytest.raises(ValueError):
        step_linear_pc(state, np.ones((1, 2)), np.ones((1, 1)), 0.01)


@pytest.mark.parametrize(
    "features,targets",
    (
        (np.ones((1, 2), dtype=np.float32), np.ones((1, 1))),
        (np.ones((1, 2)), np.array([[np.inf]])),
    ),
)
def test_should_refuse_invalid_batch_without_mutating_state(features, targets):
    state = initialize_linear_scaling(LinearScalingConfig("mean_field", 2, 3, 1), 41)
    original = tuple(w.copy() for w in state.raw_weights)
    with pytest.raises(ValueError):
        step_linear_pc(state, features, targets, 0.01)
    for actual, expected in zip(state.raw_weights, original, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("feature", (1.0, 10.0))
def test_should_refuse_overflowing_candidate_arrays_or_equilibrium_atomically(feature):
    config = LinearScalingConfig("fan_in", 1, 1, 1)
    state = LinearScalingState(config, (np.ones((1, 1)), np.ones((1, 1))))
    original = tuple(w.copy() for w in state.raw_weights)
    # feature=1 gives finite huge candidate arrays but overflowing products;
    # feature=10 also overflows the proposed raw parameter step itself.
    with pytest.raises((ValueError, FloatingPointError)):
        step_linear_pc(state, np.array([[feature]]), np.zeros((1, 1)), 1e308)
    for actual, expected in zip(state.raw_weights, original, strict=True):
        np.testing.assert_array_equal(actual, expected)
