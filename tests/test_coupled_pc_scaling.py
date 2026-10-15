"""Independent raw-coordinate objective derivatives and direct-product updates.

Tiny fixed controls establish local arithmetic and ownership, not width stability.
Finite differences hold latents fixed and never differentiate through inference.
"""

from dataclasses import asdict, replace
from decimal import Decimal, localcontext
from math import sqrt
from typing import Any

import numpy as np
import pytest

from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_energy import (
    CoupledEvaluation,
    CoupledParameters,
    evaluate_coupled,
    feedforward_latents,
)
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    CoupledScalingState,
    copy_coupled_scaling_state,
    effective_coupled_parameters,
    evaluate_coupled_scaling,
    initialize_coupled_scaling,
    step_coupled_scaling,
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


def factors(config):
    if config.scheme == "mean_field":
        rate = config.gamma0 * config.gamma0 * config.width
        scales = (
            1 / sqrt(config.input_dim),
            *((1 / sqrt(config.width),) * (config.hidden_layers - 1)),
            1 / (config.gamma0 * config.width),
        )
        return scales, rate, 1 / sqrt(rate)
    return (1.0,) * (config.hidden_layers + 1), 1.0, 1.0


def effective_reference(raw, config):
    scales, _, bias = factors(config)
    return CoupledParameters(
        tuple(w * a for w, a in zip(raw.hidden_weights, scales[:-1], strict=True)),
        tuple(b * bias for b in raw.hidden_biases),
        raw.output_weight * scales[-1],
        raw.output_bias * bias,
    )


def scalar_objective(raw, config, x, y, latents):
    p = effective_reference(raw, config)
    previous, energy = x, 0.0
    precisions = (1.0,) * config.hidden_layers if config.precisions is None else config.precisions
    for w, b, z, beta in zip(p.hidden_weights, p.hidden_biases, latents, precisions, strict=True):
        residual = z - np.tanh(previous @ w + b)
        energy += float(beta * np.sum(residual**2) / (2 * len(x)))
        previous = z
    logits = latents[-1] @ p.output_weight + p.output_bias
    return energy + float(np.mean(np.logaddexp(0, logits) - y * logits))


def batch():
    return np.array([[0.3, -0.4], [0.7, 0.2]]), np.array([[0.2], [0.8]])


def manual_inference(state, x, y, steps=2, rate=0.1):
    p = effective_reference(state.raw_parameters, state.config)
    z = feedforward_latents(p, x)
    beta = (
        (1.0,) * state.config.hidden_layers
        if state.config.precisions is None
        else state.config.precisions
    )
    for _ in range(steps):
        evaluation = evaluate_coupled(p, x, y, z, beta)
        z = tuple(v - rate * g for v, g in zip(z, evaluation.latent_gradients, strict=True))
    return z, evaluate_coupled(p, x, y, z, beta)


@pytest.mark.parametrize("scheme,gamma", SCHEMES)
def test_should_expose_declared_forward_bias_initialization_and_sgd_factors(scheme, gamma):
    config = CoupledScalingConfig(scheme, 2, 3, 2, gamma)
    scales, rate, bias = factors(config)
    np.testing.assert_allclose(config.forward_scales, scales, rtol=1e-15)
    assert config.raw_learning_rate_factor == pytest.approx(rate)
    assert config.bias_scale == pytest.approx(bias)
    assert config.effective_bias_learning_rate_factor == pytest.approx(
        rate * bias * bias, rel=2e-15
    )
    np.testing.assert_allclose(
        config.effective_learning_rate_factors, [rate * a * a for a in scales], rtol=2e-15
    )
    std = (
        (1.0, 1.0, 1.0)
        if scheme == "mean_field"
        else (
            (0.5,) * 3 if scheme == "legacy_gaussian" else (1 / sqrt(2), 1 / sqrt(3), 1 / sqrt(3))
        )
    )
    np.testing.assert_allclose(config.initial_stddevs, std, rtol=1e-15)
    state = initialize_coupled_scaling(config, 41)
    rng = np.random.default_rng(41)
    for raw, shape, sigma in zip(
        (*state.raw_parameters.hidden_weights, state.raw_parameters.output_weight),
        ((2, 3), (3, 3), (3, 1)),
        std,
        strict=True,
    ):
        np.testing.assert_array_equal(raw, rng.standard_normal(shape) * sigma)
    for raw in (*state.raw_parameters.hidden_biases, state.raw_parameters.output_bias):
        np.testing.assert_array_equal(raw, 0)


@pytest.mark.parametrize("hidden", (1, 2, 4))
@pytest.mark.parametrize("gamma", (0.5, 2.0))
def test_should_match_every_raw_parameter_and_latent_finite_difference(hidden, gamma):
    config = CoupledScalingConfig(
        "mean_field", 2, 2, hidden, gamma, precisions=tuple(0.6 + 0.4 * i for i in range(hidden))
    )
    state = initialize_coupled_scaling(config, 41)
    p = state.raw_parameters
    raw = replace(
        p,
        hidden_biases=tuple(np.full_like(b, 0.13 * (i + 1)) for i, b in enumerate(p.hidden_biases)),
        output_bias=np.array([[-0.21]]),
    )
    state = CoupledScalingState(config, raw)
    x, y = batch()
    latents = tuple(np.array([[0.2, -0.3], [0.1, 0.4]]) + 0.07 * i for i in range(hidden))
    actual = evaluate_coupled_scaling(state, x, y, latents)
    assert actual.evaluation.energy_definition == "numpy_coupled_binary_energy_v1"
    assert actual.evaluation.energy == pytest.approx(
        scalar_objective(raw, config, x, y, latents), abs=2e-14
    )
    for index, (value, gradient) in enumerate(
        zip(arrays(raw), arrays(actual.raw_gradients), strict=True)
    ):
        for coordinate in np.ndindex(value.shape):
            plus, minus = [a.copy() for a in arrays(raw)], [a.copy() for a in arrays(raw)]
            plus[index][coordinate] += 1e-6
            minus[index][coordinate] -= 1e-6
            expected = (
                scalar_objective(packed(plus, hidden), config, x, y, latents)
                - scalar_objective(packed(minus, hidden), config, x, y, latents)
            ) / 2e-6
            assert gradient[coordinate] == pytest.approx(expected, abs=2e-8, rel=2e-6)
    for layer, (value, gradient) in enumerate(
        zip(latents, actual.evaluation.latent_gradients, strict=True)
    ):
        for coordinate in np.ndindex(value.shape):
            plus, minus = [a.copy() for a in latents], [a.copy() for a in latents]
            plus[layer][coordinate] += 1e-6
            minus[layer][coordinate] -= 1e-6
            expected = (
                scalar_objective(raw, config, x, y, tuple(plus))
                - scalar_objective(raw, config, x, y, tuple(minus))
            ) / 2e-6
            assert gradient[coordinate] == pytest.approx(expected, abs=2e-8, rel=2e-6)


@pytest.mark.parametrize("scheme,gamma", SCHEMES)
def test_should_follow_raw_chain_rule_and_actual_effective_learning_factors(scheme, gamma):
    state = initialize_coupled_scaling(
        CoupledScalingConfig(scheme, 2, 3, 2, gamma, precisions=(0.6, 1.3)), 41
    )
    before = copy_coupled_scaling_state(state)
    x, y = batch()
    latents, evaluation = manual_inference(state, x, y)
    result = step_coupled_scaling(state, x, y, 0.03, 2, 0.1)
    assert result.inference is None
    assert result.evaluation.energy == pytest.approx(evaluation.energy, abs=2e-14)
    scales, rate, bias = factors(state.config)
    raw_factors = (*scales[:-1], *((bias,) * state.config.hidden_layers), scales[-1], bias)
    for original, new, gradient, scale in zip(
        arrays(state.raw_parameters),
        arrays(result.state.raw_parameters),
        arrays(evaluation.parameter_gradients),
        raw_factors,
        strict=True,
    ):
        np.testing.assert_allclose(
            new, original - 0.03 * rate * scale * gradient, atol=2e-14, rtol=2e-13
        )
    old_effective, new_effective = (
        effective_reference(state.raw_parameters, state.config),
        effective_coupled_parameters(result.state),
    )
    for old, new, gradient, scale in zip(
        arrays(old_effective),
        arrays(new_effective),
        arrays(evaluation.parameter_gradients),
        raw_factors,
        strict=True,
    ):
        np.testing.assert_allclose(
            new, old - 0.03 * rate * scale * scale * gradient, atol=2e-14, rtol=2e-13
        )
    for a, b in zip(result.latents, latents, strict=True):
        np.testing.assert_allclose(a, b, atol=2e-14)
    np.testing.assert_equal(asdict(state), asdict(before))


@pytest.mark.parametrize("scheme", ("legacy_gaussian", "fan_in", "mean_field"))
def test_should_continue_real_neutral_chemistry_with_exact_raw_parameter_parity(scheme):
    pc = initialize_coupled_scaling(CoupledScalingConfig(scheme, 2, 3, 2), 41)
    cpc = initialize_coupled_scaling(
        CoupledScalingConfig(
            scheme, 2, 3, 2, circadian=CoupledCircadianConfig.matched_pc_control()
        ),
        41,
    )
    x, y = batch()
    for _ in range(2):
        a = step_coupled_scaling(pc, x, y, 0.03, 2, 0.1)
        b = step_coupled_scaling(cpc, x, y, 0.03, 2, 0.1)
        assert a.evaluation.energy == b.evaluation.energy
        np.testing.assert_equal(asdict(a.state.raw_parameters), asdict(b.state.raw_parameters))
        pc, cpc = a.state, b.state
    assert cpc.wake_state is not None and cpc.wake_state.wake_updates == 2
    assert np.all(cpc.wake_state.chemical_state > 0)
    np.testing.assert_array_equal(cpc.wake_state.plasticity_state, np.ones(3))


@pytest.mark.parametrize("circadian", (False, True))
def test_should_match_existing_native_fan_in_updates_and_complete_wake_state(circadian):
    policy = CoupledCircadianConfig(0.9, 0.2, 0.7, 0.2) if circadian else None
    state = initialize_coupled_scaling(
        CoupledScalingConfig("fan_in", 2, 3, 2, circadian=policy), 41
    )
    native: CoupledPredictiveCodingNetwork = (
        CoupledPredictiveCodingNetwork(2, (3, 3), seed=41)
        if policy is None
        else CoupledCircadianPredictiveCodingNetwork(2, (3, 3), seed=41, config=policy)
    )
    np.testing.assert_equal(
        asdict(effective_coupled_parameters(state)), asdict(native.snapshot_parameters())
    )
    x, y = batch()
    for _ in range(2):
        result = step_coupled_scaling(state, x, y, 0.03, 2, 0.1)
        expected = native.train_epoch(x, y, 0.03, 2, 0.1)
        assert result.evaluation.energy == pytest.approx(expected.energy, abs=2e-14)
        for a, b in zip(
            arrays(effective_coupled_parameters(result.state)),
            arrays(native.snapshot_parameters()),
            strict=True,
        ):
            np.testing.assert_allclose(a, b, atol=2e-14, rtol=2e-13)
        if policy is not None:
            assert type(native) is CoupledCircadianPredictiveCodingNetwork
            assert result.state.wake_state is not None
            np.testing.assert_allclose(
                result.state.wake_state.chemical_state, native.get_chemical_state(), atol=2e-14
            )
            np.testing.assert_allclose(
                result.state.wake_state.plasticity_state, native.get_plasticity_state(), atol=2e-14
            )
        state = result.state


def test_should_apply_prospective_cpc_gates_only_to_last_incoming_bias_and_outgoing_rows():
    policy = CoupledCircadianConfig(0.9, 0.2, 0.7, 0.2)
    state = initialize_coupled_scaling(
        CoupledScalingConfig("mean_field", 2, 3, 2, 2.0, circadian=policy), 41
    )
    assert state.wake_state is not None
    chemical = np.array([0.3, 0.7, 1.1])
    state = replace(
        state,
        wake_state=replace(
            state.wake_state,
            chemical_state=chemical,
            plasticity_state=np.exp(-0.7 * chemical),
            wake_updates=4,
        ),
    )
    x, y = batch()
    latents, evaluation = manual_inference(state, x, y)
    prospective = 0.9 * chemical + 0.2 * np.mean(np.abs(latents[-1]), axis=0)
    gates = np.clip(np.exp(-0.7 * prospective), 0.2, 1.0)
    result = step_coupled_scaling(state, x, y, 0.03, 2, 0.1)
    scales, rate, bias = factors(state.config)
    factors_and_gates = (
        scales[0],
        scales[1] * gates,
        bias,
        bias * gates,
        scales[-1] * gates[:, None],
        bias,
    )
    for old, new, gradient, factor in zip(
        arrays(state.raw_parameters),
        arrays(result.state.raw_parameters),
        arrays(evaluation.parameter_gradients),
        factors_and_gates,
        strict=True,
    ):
        np.testing.assert_allclose(
            new, old - 0.03 * rate * gradient * factor, atol=2e-14, rtol=2e-13
        )
    assert result.state.wake_state is not None and result.state.wake_state.wake_updates == 5
    np.testing.assert_allclose(result.state.wake_state.chemical_state, prospective, atol=2e-14)
    np.testing.assert_allclose(result.state.wake_state.plasticity_state, gates, atol=2e-14)


def test_should_support_guarded_inference_without_changing_the_energy_definition():
    state = initialize_coupled_scaling(CoupledScalingConfig("mean_field", 2, 2, 1), 41)
    x, y = batch()
    result = step_coupled_scaling(
        state, x, y, 0.03, 2, 0.01, inference_policy=CoupledInferencePolicy(gradient_tolerance=1e6)
    )
    assert result.inference is not None
    assert result.inference.reason == "gradient_tolerance" and result.inference.accepted_steps == 1
    assert result.evaluation.energy_definition == "numpy_coupled_binary_energy_v1"


def test_should_update_when_raw_diagnostic_partial_overflows_but_full_product_is_finite():
    config = CoupledScalingConfig("mean_field", 1, 1, 1, float.fromhex("0x1p-530"))
    raw = CoupledParameters(
        (np.zeros((1, 1)),),
        (np.zeros((1, 1)),),
        np.array([[float.fromhex("0x1p-19")]]),
        np.zeros((1, 1)),
    )
    state = CoupledScalingState(config, raw)
    x, y = np.zeros((1, 1)), np.ones((1, 1))
    result = step_coupled_scaling(state, x, y, 1.0, 1, 1.0)
    assert result.state.raw_parameters.hidden_biases[0][0, 0] == float.fromhex("0x1p-20")
    assert np.isfinite(result.evaluation.energy)
    with pytest.raises((ValueError, FloatingPointError)):
        evaluate_coupled_scaling(state, x, y, result.latents)
    np.testing.assert_array_equal(state.raw_parameters.hidden_biases[0], 0)


def test_should_recover_underflowing_gradient_gate_product_with_large_raw_scale(monkeypatch):
    policy = CoupledCircadianConfig(1.0, 0.0, 1.0, 1e-6)
    config = CoupledScalingConfig("mean_field", 1, 1, 1, float.fromhex("0x1p500"), circadian=policy)
    initialized = initialize_coupled_scaling(config, 41)
    assert initialized.wake_state is not None
    zero = CoupledParameters(
        (np.zeros((1, 1)),), (np.zeros((1, 1)),), np.zeros((1, 1)), np.zeros((1, 1))
    )
    state = CoupledScalingState(
        config,
        zero,
        replace(
            initialized.wake_state,
            chemical_state=np.array([20.0]),
            plasticity_state=np.array([1e-6]),
            wake_updates=1,
        ),
    )
    tiny = float.fromhex("0x1p-1070")

    def fixed_gradient(self, x, y, rate, steps, inference_rate, inference_policy):
        gradients = replace(zero, output_weight=np.array([[tiny]]))
        evaluation = CoupledEvaluation(0.7, 0.7, (0.0,), (np.zeros((1, 1)),), gradients)
        return rate, (np.zeros((1, 1)),), evaluation, None

    monkeypatch.setattr(CoupledPredictiveCodingNetwork, "_training_with_policy", fixed_gradient)
    result = step_coupled_scaling(state, np.zeros((1, 1)), np.full((1, 1), 0.5), 1.0, 1, 0.1)
    assert tiny * 1e-6 == 0.0
    with localcontext() as context:
        context.prec = 100
        expected = float(
            Decimal.from_float(tiny)
            * Decimal.from_float(config.raw_learning_rate_factor)
            * Decimal.from_float(config.forward_scales[-1])
            * Decimal.from_float(1e-6)
        )
    assert expected > 0
    assert result.state.raw_parameters.output_weight[0, 0] == pytest.approx(
        -expected, rel=2e-15, abs=0
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"input_dim": True},
        {"width": 0},
        {"hidden_layers": 17},
        {"width": 1001, "hidden_layers": 2},
        {"gamma0": True},
        {"gamma0": float("inf")},
        {"precisions": [1.0, 1.0]},
        {"precisions": (True, 1.0)},
        {"precisions": (0.0, 1.0)},
        {"circadian": {}},
        {"scheme": "unknown"},
    ),
)
def test_should_refuse_invalid_configuration_before_initialization(changes: Any):
    values: dict[str, Any] = dict(scheme="mean_field", input_dim=2, width=3, hidden_layers=2)
    values.update(changes)
    with pytest.raises(ValueError):
        CoupledScalingConfig(**values)


@pytest.mark.parametrize("rate", (True, 0, -1, float("nan"), np.float64(0.03), 10**1000))
def test_should_refuse_invalid_rates_without_mutating_source(rate: Any):
    state = initialize_coupled_scaling(CoupledScalingConfig("fan_in", 2, 2, 1), 41)
    before = copy_coupled_scaling_state(state)
    x, y = batch()
    with pytest.raises(ValueError):
        step_coupled_scaling(state, x, y, rate, 2, 0.1)
    np.testing.assert_equal(asdict(state), asdict(before))


@pytest.mark.parametrize("features", (np.ones((1, 2), dtype=np.float32), np.ones((1, 3))))
def test_should_refuse_malformed_features_before_update(features):
    state = initialize_coupled_scaling(CoupledScalingConfig("fan_in", 2, 2, 1), 41)
    before = copy_coupled_scaling_state(state)
    with pytest.raises(ValueError):
        step_coupled_scaling(state, features, np.ones((1, 1)), 0.03, 2, 0.1)
    np.testing.assert_equal(asdict(state), asdict(before))


@pytest.mark.parametrize("feature", (10.0, 100.0))
def test_should_refuse_nonfinite_candidate_arrays_or_held_latent_energy_atomically(feature):
    config = CoupledScalingConfig("fan_in", 1, 1, 1)
    state = CoupledScalingState(
        config,
        CoupledParameters(
            (np.zeros((1, 1)),), (np.zeros((1, 1)),), np.ones((1, 1)), np.zeros((1, 1))
        ),
    )
    before = copy_coupled_scaling_state(state)
    with pytest.raises((ValueError, FloatingPointError)):
        step_coupled_scaling(state, np.array([[feature]]), np.ones((1, 1)), 1e308, 1, 0.1)
    np.testing.assert_equal(asdict(state), asdict(before))


def test_should_own_raw_effective_copied_policy_and_complete_wake_state():
    policy = CoupledCircadianConfig()
    config = CoupledScalingConfig("mean_field", 2, 3, 2, 2.0, circadian=policy)
    initial = initialize_coupled_scaling(config, 41)
    assert initial.wake_state is not None
    raw = packed([a.copy() for a in arrays(initial.raw_parameters)], 2)
    wake = replace(
        initial.wake_state,
        chemical_state=np.full(3, 0.4),
        plasticity_state=np.exp(np.full(3, -0.28)),
        wake_updates=2,
    )
    state = CoupledScalingState(config, raw, wake)
    before = asdict(state)
    raw.hidden_weights[0][:] = 99
    wake.chemical_state[:] = 99
    wake.plasticity_state[:] = 99
    object.__setattr__(policy, "chemical_decay", 0.1)
    object.__setattr__(config, "gamma0", 3.0)
    np.testing.assert_equal(asdict(state), before)
    effective = effective_coupled_parameters(state)
    copied = copy_coupled_scaling_state(state)
    for output in (*arrays(effective), *arrays(copied.raw_parameters)):
        output[:] = 123
    assert copied.wake_state is not None
    copied.wake_state.chemical_state[:] = 456
    np.testing.assert_equal(asdict(state), before)


@pytest.mark.parametrize("invalid", ("dtype", "shape", "tuple", "nan", "family", "wake"))
def test_should_refuse_malformed_raw_state_before_mapping(invalid):
    config = CoupledScalingConfig("fan_in", 2, 3, 2)
    state = initialize_coupled_scaling(config, 41)
    raw: Any = state.raw_parameters
    if invalid == "dtype":
        raw = replace(raw, output_weight=raw.output_weight.astype(np.float32))
    elif invalid == "shape":
        raw = replace(raw, output_bias=np.zeros((1, 2)))
    elif invalid == "tuple":
        raw = replace(raw, hidden_weights=list(raw.hidden_weights))
    elif invalid == "nan":
        raw.output_weight[0, 0] = np.nan
    elif invalid == "family":
        raw = asdict(raw)
    wake: Any = {} if invalid == "wake" else None
    with pytest.raises(ValueError):
        CoupledScalingState(config, raw, wake)


@pytest.mark.parametrize("invalid", ("count", "gate", "policy", "chemistry"))
def test_should_refuse_inconsistent_full_cpc_wake_state(invalid):
    state = initialize_coupled_scaling(
        CoupledScalingConfig("fan_in", 2, 3, 1, circadian=CoupledCircadianConfig()), 41
    )
    assert state.wake_state is not None
    changes: dict[str, dict[str, Any]] = {
        "count": {"wake_updates": True},
        "gate": {"plasticity_state": np.full(3, 0.5)},
        "policy": {"chemical_decay": 0.1},
        "chemistry": {"chemical_state": np.full(3, -1.0)},
    }
    with pytest.raises(ValueError):
        CoupledScalingState(
            state.config, state.raw_parameters, replace(state.wake_state, **changes[invalid])
        )


@pytest.mark.parametrize("seed", (True, -1, np.int64(41), 0.5))
def test_should_refuse_non_native_or_negative_seed(seed):
    with pytest.raises(ValueError):
        initialize_coupled_scaling(CoupledScalingConfig("fan_in", 2, 3, 1), seed)


@pytest.mark.parametrize("scheme", ("legacy_gaussian", "fan_in"))
def test_should_refuse_nonunit_gamma_for_declared_controls(scheme):
    with pytest.raises(ValueError):
        CoupledScalingConfig(scheme, 2, 3, 1, 2.0)
