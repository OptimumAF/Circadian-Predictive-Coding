"""Wake chemistry, selective plasticity and atomicity for coupled CPC."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_energy import CoupledParameters, evaluate_coupled, feedforward_latents


def _batch():
    return np.array([[0.4, -0.2], [-0.3, 0.1], [0.2, 0.6]]), np.array([[1.0], [0.0], [1.0]])


def _arrays(parameters):
    return (
        *parameters.hidden_weights,
        *parameters.hidden_biases,
        parameters.output_weight,
        parameters.output_bias,
    )


def _assert_parameters(actual, expected):
    for left, right in zip(_arrays(actual), _arrays(expected)):
        np.testing.assert_array_equal(left, right)


def _relax(model, features, targets, steps=3, inference_rate=0.03):
    parameters = model.snapshot_parameters()
    latents = feedforward_latents(parameters, features)
    for _ in range(steps):
        evaluation = evaluate_coupled(parameters, features, targets, latents, model.precisions)
        latents = tuple(
            z - inference_rate * gradient
            for z, gradient in zip(latents, evaluation.latent_gradients)
        )
    return latents, evaluate_coupled(parameters, features, targets, latents, model.precisions)


def test_active_gates_scale_only_last_incoming_columns_bias_and_output_rows():
    config = CoupledCircadianConfig(
        chemical_decay=0.9,
        chemical_buildup_rate=2.0,
        plasticity_sensitivity=0.5,
        min_plasticity=0.1,
    )
    model = CoupledCircadianPredictiveCodingNetwork(2, (2, 3), 4, (0.8, 1.2), config=config)
    features, targets = _batch()
    before = model.snapshot_parameters()
    latents, relaxed = _relax(model, features, targets)
    chemicals = 2.0 * np.mean(np.abs(latents[-1]), axis=0)
    gates = np.clip(np.exp(-0.5 * chemicals), 0.1, 1.0)
    assert np.all(gates < 1)
    gradients = relaxed.parameter_gradients
    assert np.any(gradients.hidden_weights[0] != 0)
    assert np.any(gradients.output_bias != 0)
    expected = CoupledParameters(
        (
            before.hidden_weights[0] - 0.02 * gradients.hidden_weights[0],
            before.hidden_weights[1] - 0.02 * (gradients.hidden_weights[1] * gates),
        ),
        (
            before.hidden_biases[0] - 0.02 * gradients.hidden_biases[0],
            before.hidden_biases[1] - 0.02 * (gradients.hidden_biases[1] * gates),
        ),
        before.output_weight - 0.02 * (gradients.output_weight * gates[:, None]),
        before.output_bias - 0.02 * gradients.output_bias,
    )
    result = model.train_epoch(features, targets, 0.02, 3, 0.03)
    _assert_parameters(model.snapshot_parameters(), expected)
    np.testing.assert_array_equal(model.get_chemical_state(), chemicals)
    np.testing.assert_array_equal(model.get_plasticity_state(), gates)
    assert result.energy == relaxed.energy
    assert result.energy_definition == relaxed.energy_definition
    assert model.wake_updates == 1


def test_chemistry_uses_final_inferred_activity_and_previous_wake_decay():
    config = CoupledCircadianConfig(chemical_decay=0.4, chemical_buildup_rate=0.3)
    model = CoupledCircadianPredictiveCodingNetwork(2, (3,), 2, config=config)
    features, targets = _batch()
    chemicals = np.zeros(3)
    for count in (1, 2):
        latents, _ = _relax(model, features, targets)
        chemicals = 0.4 * chemicals + 0.3 * np.mean(np.abs(latents[-1]), axis=0)
        model.train_epoch(features, targets, 0.02, 3, 0.03)
        np.testing.assert_array_equal(model.get_chemical_state(), chemicals)
        np.testing.assert_array_equal(
            model.get_plasticity_state(), np.clip(np.exp(-0.7 * chemicals), 0.2, 1.0)
        )
        assert model.wake_updates == count


def test_matched_control_retains_real_nonzero_chemistry_with_exact_unit_gates():
    config = CoupledCircadianConfig.matched_pc_control()
    assert config.min_plasticity == 1.0
    model = CoupledCircadianPredictiveCodingNetwork(2, (2,), 3, config=config)
    features, targets = _batch()
    model.train_epoch(features, targets, 0.02, 2, 0.03)
    assert np.all(model.get_chemical_state() > 0)
    np.testing.assert_array_equal(model.get_plasticity_state(), np.ones(2))
    assert model.wake_updates == 1


def test_zero_chemistry_and_exponential_saturation_have_finite_valid_gates():
    features, targets = _batch()
    zero = CoupledCircadianPredictiveCodingNetwork(
        2, (2,), 1, config=CoupledCircadianConfig(chemical_decay=0, chemical_buildup_rate=0)
    )
    zero.train_epoch(features, targets, 0.02, 2, 0.03)
    np.testing.assert_array_equal(zero.get_chemical_state(), np.zeros(2))
    np.testing.assert_array_equal(zero.get_plasticity_state(), np.ones(2))
    saturated = CoupledCircadianPredictiveCodingNetwork(
        2,
        (2,),
        1,
        config=CoupledCircadianConfig(chemical_buildup_rate=1e6, plasticity_sensitivity=1e6),
    )
    with np.errstate(over="raise", invalid="raise", under="raise"):
        saturated.train_epoch(features, targets, 0.02, 2, 0.03)
    assert np.isfinite(saturated.get_chemical_state()).all()
    np.testing.assert_array_equal(saturated.get_plasticity_state(), np.full(2, 0.2))


def test_state_getters_are_detached_and_parameter_restore_preserves_wake_state():
    model = CoupledCircadianPredictiveCodingNetwork(2, (2,), 1)
    original = model.snapshot_parameters()
    features, targets = _batch()
    model.train_epoch(features, targets, 0.02, 2, 0.03)
    chemicals, gates = model.get_chemical_state(), model.get_plasticity_state()
    assert chemicals.shape == gates.shape == (2,)
    chemicals_copy, gates_copy = chemicals.copy(), gates.copy()
    chemicals.fill(99)
    gates.fill(99)
    model.restore_parameters(original)
    _assert_parameters(model.snapshot_parameters(), original)
    np.testing.assert_array_equal(model.get_chemical_state(), chemicals_copy)
    np.testing.assert_array_equal(model.get_plasticity_state(), gates_copy)
    assert model.wake_updates == 1
    with pytest.raises(AttributeError):
        setattr(model, "config", CoupledCircadianConfig())
    with pytest.raises(FrozenInstanceError):
        setattr(model.config, "chemical_decay", 0.5)


@pytest.mark.parametrize(
    "field,value",
    [
        ("chemical_decay", True),
        ("chemical_decay", -0.1),
        ("chemical_decay", 1.1),
        ("chemical_buildup_rate", -1),
        ("chemical_buildup_rate", 1e6 + 1),
        ("plasticity_sensitivity", -1),
        ("plasticity_sensitivity", np.inf),
        ("plasticity_sensitivity", 10**1000),
        ("min_plasticity", 0),
        ("min_plasticity", 1e-7),
        ("min_plasticity", 1.1),
        ("chemical_decay", np.nan),
        ("chemical_decay", np.float64(0.5)),
    ],
)
def test_invalid_configuration_values_are_refused(field, value):
    with pytest.raises(ValueError):
        CoupledCircadianConfig(**{field: value})


@pytest.mark.parametrize("config", [object(), {}, False])
def test_foreign_configuration_types_are_refused(config):
    with pytest.raises(ValueError):
        CoupledCircadianPredictiveCodingNetwork(2, (2,), 0, config=config)


@pytest.mark.parametrize("fault", ["rate", "steps", "features", "candidate_energy"])
def test_failed_wake_preserves_parameters_chemistry_gates_and_count(fault):
    model = CoupledCircadianPredictiveCodingNetwork(2, (1,), 3)
    features, targets = _batch()
    model.train_epoch(features, targets, 0.02, 2, 0.03)
    if fault == "candidate_energy":
        model.restore_parameters(
            CoupledParameters(
                (np.zeros((2, 1)),),
                (np.zeros((1, 1)),),
                np.full((1, 1), 1e200),
                np.zeros((1, 1)),
            )
        )
        features, targets = np.array([[1e200, 0.0]]), np.ones((1, 1))
    before = model.snapshot_parameters()
    chemicals, gates, count = (
        model.get_chemical_state(),
        model.get_plasticity_state(),
        model.wake_updates,
    )
    with pytest.raises((ValueError, FloatingPointError)):
        if fault == "rate":
            model.train_epoch(features, targets, np.nan, 2, 0.03)
        elif fault == "steps":
            model.train_epoch(features, targets, 0.02, 0, 0.03)
        elif fault == "features":
            model.train_epoch(np.ones((2, 3)), np.ones((2, 1)), 0.02, 2, 0.03)
        else:
            model.train_epoch(features, targets, 1.0, 1, 1e-200)
    _assert_parameters(model.snapshot_parameters(), before)
    np.testing.assert_array_equal(model.get_chemical_state(), chemicals)
    np.testing.assert_array_equal(model.get_plasticity_state(), gates)
    assert model.wake_updates == count
