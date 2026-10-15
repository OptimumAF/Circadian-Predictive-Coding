"""Independent full-forward BCE derivatives and raw-coordinate BP updates.

Latents are recomputed inside the finite-difference oracle: this is full BP,
whereas the PC diagnostic partial holds its latent arguments fixed.
"""

from dataclasses import asdict, replace
from decimal import Decimal, localcontext
from math import sqrt
from typing import Any

import numpy as np
import pytest

import src.core.scaled_matched_backprop as scaled
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_energy import (
    CoupledParameters,
    evaluate_coupled,
    feedforward_latents,
    predict_coupled,
)
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    CoupledScalingState,
    copy_coupled_scaling_state,
    effective_coupled_parameters,
    initialize_coupled_scaling,
    step_coupled_scaling,
)
from src.core.matched_backprop import (
    MatchedBackpropEvaluation,
    MatchedBackpropNetwork,
    evaluate_matched_backprop,
)
from src.core.scaled_matched_backprop import (
    evaluate_scaled_matched_backprop,
    step_scaled_matched_backprop,
)

SCHEMES = (
    ("legacy_gaussian", 1.0),
    ("fan_in", 1.0),
    ("mean_field", 0.5),
    ("mean_field", 1.0),
    ("mean_field", 2.0),
)


def arrays(p):
    return (*p.hidden_weights, *p.hidden_biases, p.output_weight, p.output_bias)


def packed(values, depth):
    return CoupledParameters(
        tuple(values[:depth]), tuple(values[depth : 2 * depth]), values[-2], values[-1]
    )


def batch():
    return np.array([[0.3, -0.4], [0.7, 0.2]]), np.array([[0.2], [0.8]])


def scales(config):
    if config.scheme == "mean_field":
        a = (
            1 / sqrt(config.input_dim),
            *((1 / sqrt(config.width),) * (config.hidden_layers - 1)),
            1 / (config.gamma0 * config.width),
        )
        rate = config.gamma0 * config.gamma0 * config.width
        bias = 1 / sqrt(rate)
    else:
        a, rate, bias = (1.0,) * (config.hidden_layers + 1), 1.0, 1.0
    return (*a[:-1], *((bias,) * config.hidden_layers), a[-1], bias), rate


def scalar_objective(raw, config, x, y):
    factors, _ = scales(config)
    p = packed([a * s for a, s in zip(arrays(raw), factors, strict=True)], config.hidden_layers)
    z = x
    for w, b in zip(p.hidden_weights, p.hidden_biases, strict=True):
        z = np.tanh(z @ w + b)
    logits = z @ p.output_weight + p.output_bias
    return float(np.mean(np.logaddexp(0, logits) - y * logits))


@pytest.mark.parametrize("hidden", (1, 2, 4))
@pytest.mark.parametrize("gamma", (0.5, 2.0))
def test_should_differentiate_every_raw_coordinate_through_all_hidden_states(hidden, gamma):
    config = CoupledScalingConfig(
        "mean_field", 2, 2, hidden, gamma, precisions=tuple(0.7 + 0.3 * i for i in range(hidden))
    )
    state = initialize_coupled_scaling(config, 41)
    raw = replace(
        state.raw_parameters,
        hidden_biases=tuple(
            np.full_like(b, 0.13 * (i + 1))
            for i, b in enumerate(state.raw_parameters.hidden_biases)
        ),
        output_bias=np.array([[-0.21]]),
    )
    state = CoupledScalingState(config, raw)
    x, y = batch()
    actual = evaluate_scaled_matched_backprop(state, x, y)
    assert actual.evaluation.loss_definition == "numpy_matched_binary_bce_preupdate_v1"
    assert actual.evaluation.loss == pytest.approx(scalar_objective(raw, config, x, y), abs=2e-14)
    for index, (value, gradient) in enumerate(
        zip(arrays(raw), arrays(actual.raw_gradients), strict=True)
    ):
        for coordinate in np.ndindex(value.shape):
            plus, minus = [a.copy() for a in arrays(raw)], [a.copy() for a in arrays(raw)]
            plus[index][coordinate] += 1e-6
            minus[index][coordinate] -= 1e-6
            expected = (
                scalar_objective(packed(plus, hidden), config, x, y)
                - scalar_objective(packed(minus, hidden), config, x, y)
            ) / 2e-6
            assert gradient[coordinate] == pytest.approx(expected, abs=2e-8, rel=2e-6)


@pytest.mark.parametrize("scheme,gamma", SCHEMES)
def test_should_use_actual_raw_and_effective_learning_factors(scheme, gamma):
    state = initialize_coupled_scaling(
        CoupledScalingConfig(scheme, 2, 3, 2, gamma, precisions=(0.7, 1.3)), 41
    )
    before = asdict(state)
    effective = effective_coupled_parameters(state)
    x, y = batch()
    expected = evaluate_matched_backprop(effective, x, y)
    factors, rate = scales(state.config)
    result = step_scaled_matched_backprop(state, x, y, 0.03)
    assert result.evaluation.loss == expected.loss
    for old, new, g, scale in zip(
        arrays(state.raw_parameters),
        arrays(result.state.raw_parameters),
        arrays(expected.parameter_gradients),
        factors,
        strict=True,
    ):
        np.testing.assert_allclose(new, old - 0.03 * rate * scale * g, rtol=1e-13, atol=1e-15)
    for old, new, g, scale in zip(
        arrays(effective),
        arrays(effective_coupled_parameters(result.state)),
        arrays(expected.parameter_gradients),
        factors,
        strict=True,
    ):
        np.testing.assert_allclose(
            new, old - 0.03 * rate * scale * scale * g, rtol=1e-13, atol=1e-15
        )
    np.testing.assert_equal(asdict(state), before)
    result.state.raw_parameters.output_bias[:] = 99
    result.evaluation.parameter_gradients.output_bias[:] = 99
    np.testing.assert_equal(asdict(state), before)


@pytest.mark.parametrize("hidden", (1, 2, 4))
def test_should_match_two_native_fan_in_bp_updates_with_complete_restore(hidden):
    state = initialize_coupled_scaling(CoupledScalingConfig("fan_in", 2, 3, hidden), 41)
    native = MatchedBackpropNetwork(effective_coupled_parameters(state))
    x, y = batch()
    for _ in range(2):
        result = step_scaled_matched_backprop(state, x, y, 0.03)
        expected = native.train_epoch(x, y, 0.03)
        assert result.evaluation.loss == pytest.approx(expected.loss, rel=1e-13, abs=1e-15)
        for actual, reference in zip(
            arrays(effective_coupled_parameters(result.state)),
            arrays(native.snapshot_parameters()),
            strict=True,
        ):
            np.testing.assert_allclose(actual, reference, rtol=1e-13, atol=1e-15)
        state = result.state


def test_should_share_initial_predictions_but_differentiate_through_hidden_states():
    state = initialize_coupled_scaling(
        CoupledScalingConfig("mean_field", 2, 3, 2, 2.0, precisions=(0.7, 1.3)), 41
    )
    x, y = batch()
    p = effective_coupled_parameters(state)
    before = copy_coupled_scaling_state(state)
    bp = evaluate_scaled_matched_backprop(state, x, y)
    assert state.config.precisions is not None
    fixed = evaluate_coupled(p, x, y, feedforward_latents(p, x), state.config.precisions)
    assert bp.evaluation.loss == fixed.task_loss
    np.testing.assert_array_equal(fixed.parameter_gradients.hidden_weights[0], 0)
    assert np.max(np.abs(bp.evaluation.parameter_gradients.hidden_weights[0])) > 1e-4
    native = MatchedBackpropNetwork(p)
    np.testing.assert_array_equal(native.predict_proba(x), predict_coupled(p, x))
    bp_step = step_scaled_matched_backprop(state, x, y, 0.03)
    pc_step = step_coupled_scaling(state, x, y, 0.03, 2, 0.1)
    assert not np.allclose(
        bp_step.state.raw_parameters.hidden_weights[0],
        pc_step.state.raw_parameters.hidden_weights[0],
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_equal(asdict(state), asdict(before))


def test_should_ignore_preserved_precision_metadata_for_bp_loss_and_gradients():
    a = initialize_coupled_scaling(
        CoupledScalingConfig("mean_field", 2, 3, 2, 2.0, precisions=(0.7, 1.3)), 41
    )
    b = CoupledScalingState(replace(a.config, precisions=(1.1, 2.2)), a.raw_parameters)
    x, y = batch()
    np.testing.assert_equal(
        asdict(evaluate_scaled_matched_backprop(a, x, y)),
        asdict(evaluate_scaled_matched_backprop(b, x, y)),
    )


def test_should_apply_finite_full_product_when_raw_diagnostic_bias_partial_overflows():
    config = CoupledScalingConfig("mean_field", 1, 1, 1, float.fromhex("0x1p-530"))
    raw = CoupledParameters(
        (np.zeros((1, 1)),),
        (np.zeros((1, 1)),),
        np.array([[float.fromhex("0x1p-19")]]),
        np.zeros((1, 1)),
    )
    state = CoupledScalingState(config, raw)
    x, y = np.zeros((1, 1)), np.ones((1, 1))
    result = step_scaled_matched_backprop(state, x, y, 1.0)
    assert result.state.raw_parameters.hidden_biases[0][0, 0] == float.fromhex("0x1p-20")
    assert np.isfinite(result.evaluation.loss)
    with pytest.raises((ValueError, FloatingPointError)):
        evaluate_scaled_matched_backprop(state, x, y)
    np.testing.assert_array_equal(state.raw_parameters.hidden_biases[0], 0)


def test_should_recover_subnormal_gradient_before_large_rate_scale_product(monkeypatch):
    config = CoupledScalingConfig("mean_field", 1, 1, 1, float.fromhex("0x1p500"))
    zeros = CoupledParameters(
        (np.zeros((1, 1)),), (np.zeros((1, 1)),), np.zeros((1, 1)), np.zeros((1, 1))
    )
    state = CoupledScalingState(config, zeros)
    tiny = float.fromhex("0x1p-1070")
    original = evaluate_matched_backprop
    calls = []

    def controlled(p, x, y):
        calls.append(1)
        if len(calls) == 1:
            return MatchedBackpropEvaluation(0.7, replace(zeros, output_bias=np.array([[tiny]])))
        return original(p, x, y)

    monkeypatch.setattr(scaled, "evaluate_matched_backprop", controlled)
    result = step_scaled_matched_backprop(state, np.zeros((1, 1)), np.full((1, 1), 0.5), 1.0)
    assert tiny * config.bias_scale == 0
    with localcontext() as context:
        context.prec = 100
        expected = float(
            Decimal.from_float(tiny)
            * Decimal.from_float(config.raw_learning_rate_factor)
            * Decimal.from_float(config.bias_scale)
        )
    assert expected > 0
    assert result.state.raw_parameters.output_bias[0, 0] == -expected


@pytest.mark.parametrize("rate", (True, 0, -1, float("nan"), np.float64(0.03), 10**1000))
def test_should_refuse_invalid_rates_without_changing_source(rate: Any):
    state = initialize_coupled_scaling(CoupledScalingConfig("fan_in", 2, 3, 1), 41)
    before = asdict(state)
    x, y = batch()
    with pytest.raises(ValueError):
        step_scaled_matched_backprop(state, x, y, rate)
    np.testing.assert_equal(asdict(state), before)


@pytest.mark.parametrize(
    "invalid", ("dtype", "width", "target_range", "target_dtype", "target_shape", "nonfinite")
)
def test_should_refuse_invalid_batches_before_any_update(invalid):
    state = initialize_coupled_scaling(CoupledScalingConfig("fan_in", 2, 3, 1), 41)
    x, y = batch()
    if invalid == "dtype":
        x = x.astype(np.float32)
    elif invalid == "width":
        x = np.ones((2, 3))
    elif invalid == "target_range":
        y[0, 0] = 2
    elif invalid == "target_dtype":
        y = y.astype(np.float32)
    elif invalid == "target_shape":
        y = y.ravel()
    else:
        x[0, 0] = np.inf
    before = asdict(state)
    with pytest.raises(ValueError):
        step_scaled_matched_backprop(state, x, y, 0.03)
    np.testing.assert_equal(asdict(state), before)


@pytest.mark.parametrize("family", ("neutral", "active"))
def test_should_refuse_circadian_state_even_when_gates_are_neutral(family):
    policy = (
        CoupledCircadianConfig.matched_pc_control()
        if family == "neutral"
        else CoupledCircadianConfig()
    )
    state = initialize_coupled_scaling(
        CoupledScalingConfig("fan_in", 2, 3, 1, circadian=policy), 41
    )
    before = asdict(state)
    with pytest.raises(ValueError):
        evaluate_scaled_matched_backprop(state, *batch())
    x, y = batch()
    with pytest.raises(ValueError):
        step_scaled_matched_backprop(state, x, y, 0.03)
    np.testing.assert_equal(asdict(state), before)


@pytest.mark.parametrize("feature", (3.0, 10.0))
def test_should_refuse_candidate_nonfinite_arrays_or_loss_without_publication(feature):
    state = CoupledScalingState(
        CoupledScalingConfig("fan_in", 1, 1, 1),
        CoupledParameters(
            (np.zeros((1, 1)),), (np.zeros((1, 1)),), np.ones((1, 1)), np.zeros((1, 1))
        ),
    )
    before = asdict(state)
    with pytest.raises((ValueError, FloatingPointError)):
        step_scaled_matched_backprop(state, np.array([[feature]]), np.ones((1, 1)), 1e308)
    np.testing.assert_equal(asdict(state), before)
