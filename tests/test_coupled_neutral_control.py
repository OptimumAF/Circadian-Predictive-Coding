"""Matched coupled PC/CPC wake controls with active, independently checked traces.

All comparisons use tiny fixed training batches and identical initial parameters,
precisions and inference settings. Exact neutral parity is a local integration
property; the nonneutral case checks gating mechanics, not learning advantage.
No replay, sleep, structure change or scientific performance claim is exercised.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pytest

from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork

Array = NDArray[np.float64]
WIDTHS = ((2,), (2, 3), (2, 3, 2, 1))
LEARNING_RATE = 0.03
INFERENCE_STEPS = 4
INFERENCE_RATE = 0.08


def batch() -> tuple[Array, Array]:
    return np.array(
        [
            [-0.8, 0.2, 0.4],
            [0.3, -0.6, 0.1],
            [0.7, 0.4, -0.2],
            [-0.1, 0.8, -0.5],
        ]
    ), np.array([[0.0], [1.0], [0.0], [1.0]])


def parameter_arrays(parameters: CoupledParameters) -> tuple[Array, ...]:
    return (
        *parameters.hidden_weights,
        *parameters.hidden_biases,
        parameters.output_weight,
        parameters.output_bias,
    )


def assert_parameters_identical(actual: CoupledParameters, expected: CoupledParameters) -> None:
    for left, right in zip(parameter_arrays(actual), parameter_arrays(expected), strict=True):
        np.testing.assert_array_equal(left, right)


def independently_inferred_latents(
    parameters: CoupledParameters, features: Array, targets: Array, precisions: tuple[float, ...]
) -> tuple[Array, ...]:
    """Reconstruct the fixed simultaneous latent steps without a production helper."""
    latents = []
    previous = features
    for weight, bias in zip(parameters.hidden_weights, parameters.hidden_biases, strict=True):
        previous = np.tanh(previous @ weight + bias)
        latents.append(previous)
    states = tuple(latents)
    for _ in range(INFERENCE_STEPS):
        predictions = tuple(
            np.tanh(lower @ weight + bias)
            for lower, weight, bias in zip(
                (features, *states[:-1]),
                parameters.hidden_weights,
                parameters.hidden_biases,
                strict=True,
            )
        )
        residuals = tuple(z - p for z, p in zip(states, predictions, strict=True))
        gradients = []
        for index, (residual, precision) in enumerate(zip(residuals, precisions, strict=True)):
            gradient = precision * residual / len(features)
            if index + 1 < len(states):
                downstream = residuals[index + 1] * (1 - predictions[index + 1] ** 2)
                gradient -= (
                    precisions[index + 1] * downstream @ parameters.hidden_weights[index + 1].T
                ) / len(features)
            else:
                logits = states[-1] @ parameters.output_weight + parameters.output_bias
                gradient += (
                    (1 / (1 + np.exp(-logits)) - targets) @ parameters.output_weight.T
                ) / len(features)
            gradients.append(gradient)
        states = tuple(z - INFERENCE_RATE * g for z, g in zip(states, gradients, strict=True))
    return states


def precision_profile(widths: tuple[int, ...], profile: str) -> tuple[float, ...]:
    return (1.0,) * len(widths) if profile == "unit" else (0.7, 1.4, 0.6, 2.1)[: len(widths)]


@pytest.mark.parametrize("widths", WIDTHS, ids=("one-layer", "two-layers", "four-layers"))
@pytest.mark.parametrize("profile", ("unit", "nonuniform"))
def test_neutral_control_matches_every_parameter_energy_and_prediction_through_four_wakes(
    widths: tuple[int, ...],
    profile: str,
) -> None:
    precisions = precision_profile(widths, profile)
    pc = CoupledPredictiveCodingNetwork(3, widths, seed=41, precisions=precisions)
    config = CoupledCircadianConfig.matched_pc_control()
    circadian = CoupledCircadianPredictiveCodingNetwork(
        3, widths, seed=41, precisions=precisions, config=config
    )
    assert config == CoupledCircadianConfig(min_plasticity=1.0)
    assert pc.hidden_dims == circadian.hidden_dims == widths
    assert pc.precisions == circadian.precisions == precisions
    assert_parameters_identical(circadian.snapshot_parameters(), pc.snapshot_parameters())
    features, targets = batch()
    original_features, original_targets = features.copy(), targets.copy()
    expected_chemical = np.zeros(widths[-1])
    np.testing.assert_array_equal(circadian.get_chemical_state(), expected_chemical)
    np.testing.assert_array_equal(circadian.get_plasticity_state(), np.ones(widths[-1]))
    assert type(circadian.wake_updates) is int and circadian.wake_updates == 0
    for wake, indices in enumerate(([0, 1, 2], [3, 1], [2], [0, 1, 2, 3]), start=1):
        X, y = features[indices], targets[indices]
        before_X, before_y = X.copy(), y.copy()
        latents = independently_inferred_latents(pc.snapshot_parameters(), X, y, precisions)
        expected_chemical = (
            config.chemical_decay * expected_chemical
            + config.chemical_buildup_rate * np.mean(np.abs(latents[-1]), axis=0)
        )
        pc_result = pc.train_epoch(X, y, LEARNING_RATE, INFERENCE_STEPS, INFERENCE_RATE)
        cpc_result = circadian.train_epoch(X, y, LEARNING_RATE, INFERENCE_STEPS, INFERENCE_RATE)
        assert pc_result.energy == cpc_result.energy
        assert (
            pc_result.energy_definition
            == cpc_result.energy_definition
            == "numpy_coupled_binary_energy_v1"
        )
        assert_parameters_identical(circadian.snapshot_parameters(), pc.snapshot_parameters())
        np.testing.assert_array_equal(circadian.predict_proba(features), pc.predict_proba(features))
        np.testing.assert_allclose(
            circadian.get_chemical_state(), expected_chemical, atol=1e-16, rtol=1e-13
        )
        assert np.any(circadian.get_chemical_state() > 0)
        np.testing.assert_array_equal(circadian.get_plasticity_state(), np.ones(widths[-1]))
        assert circadian.wake_updates == wake
        assert circadian.precisions == pc.precisions == precisions
        np.testing.assert_array_equal(X, before_X)
        np.testing.assert_array_equal(y, before_y)
    np.testing.assert_array_equal(features, original_features)
    np.testing.assert_array_equal(targets, original_targets)


@pytest.mark.parametrize("widths", WIDTHS, ids=("one-layer", "two-layers", "four-layers"))
def test_nonneutral_trace_gates_only_declared_parameter_groups(widths: tuple[int, ...]) -> None:
    precisions = precision_profile(widths, "nonuniform")
    pc = CoupledPredictiveCodingNetwork(3, widths, seed=41, precisions=precisions)
    config = CoupledCircadianConfig()
    assert config.min_plasticity == 0.2
    circadian = CoupledCircadianPredictiveCodingNetwork(
        3, widths, seed=41, precisions=precisions, config=config
    )
    features, targets = batch()
    initial = pc.snapshot_parameters()
    inferred = independently_inferred_latents(initial, features, targets, precisions)
    expected_chemical = config.chemical_buildup_rate * np.mean(np.abs(inferred[-1]), axis=0)
    expected_gate = np.clip(np.exp(-config.plasticity_sensitivity * expected_chemical), 0.2, 1.0)
    pc_result = pc.train_epoch(features, targets, LEARNING_RATE, INFERENCE_STEPS, INFERENCE_RATE)
    cpc_result = circadian.train_epoch(
        features, targets, LEARNING_RATE, INFERENCE_STEPS, INFERENCE_RATE
    )
    assert pc_result.energy == cpc_result.energy
    np.testing.assert_allclose(
        circadian.get_chemical_state(), expected_chemical, atol=1e-16, rtol=1e-13
    )
    np.testing.assert_allclose(
        circadian.get_plasticity_state(), expected_gate, atol=1e-16, rtol=1e-13
    )
    assert np.all(expected_gate >= 0.2) and np.all(expected_gate < 1.0)
    ordinary, gated = pc.snapshot_parameters(), circadian.snapshot_parameters()
    for layer in range(len(widths) - 1):
        np.testing.assert_array_equal(gated.hidden_weights[layer], ordinary.hidden_weights[layer])
        np.testing.assert_array_equal(gated.hidden_biases[layer], ordinary.hidden_biases[layer])
    expected_weight = (
        initial.hidden_weights[-1]
        + (ordinary.hidden_weights[-1] - initial.hidden_weights[-1]) * expected_gate[None, :]
    )
    expected_bias = (
        initial.hidden_biases[-1]
        + (ordinary.hidden_biases[-1] - initial.hidden_biases[-1]) * expected_gate[None, :]
    )
    expected_output = (
        initial.output_weight
        + (ordinary.output_weight - initial.output_weight) * expected_gate[:, None]
    )
    np.testing.assert_allclose(gated.hidden_weights[-1], expected_weight, atol=1e-14, rtol=1e-12)
    np.testing.assert_allclose(gated.hidden_biases[-1], expected_bias, atol=1e-14, rtol=1e-12)
    np.testing.assert_allclose(gated.output_weight, expected_output, atol=1e-14, rtol=1e-12)
    np.testing.assert_array_equal(gated.output_bias, ordinary.output_bias)
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(parameter_arrays(gated), parameter_arrays(ordinary))
    )
    assert circadian.wake_updates == 1


def test_model_and_restore_snapshots_are_independent_across_matched_learners() -> None:
    pc = CoupledPredictiveCodingNetwork(3, (2, 3), seed=41)
    circadian = CoupledCircadianPredictiveCodingNetwork(
        3, (2, 3), seed=41, config=CoupledCircadianConfig.matched_pc_control()
    )
    expected = pc.snapshot_parameters()
    supplied = pc.snapshot_parameters()
    circadian.restore_parameters(supplied)
    for array in parameter_arrays(supplied):
        array.fill(99)
    assert_parameters_identical(pc.snapshot_parameters(), expected)
    assert_parameters_identical(circadian.snapshot_parameters(), expected)
    features, targets = batch()
    before = pc.predict_proba(features)
    circadian.train_epoch(features, targets, LEARNING_RATE, INFERENCE_STEPS, INFERENCE_RATE)
    assert_parameters_identical(pc.snapshot_parameters(), expected)
    np.testing.assert_array_equal(pc.predict_proba(features), before)
    trained = circadian.snapshot_parameters()
    returned = circadian.snapshot_parameters()
    for array in parameter_arrays(returned):
        array.fill(-99)
    assert_parameters_identical(circadian.snapshot_parameters(), trained)
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(parameter_arrays(trained), parameter_arrays(expected))
    )


def test_trace_and_gate_getters_return_detached_last_width_vectors() -> None:
    model = CoupledCircadianPredictiveCodingNetwork(3, (2, 3), seed=41)
    features, targets = batch()
    model.train_epoch(features, targets, LEARNING_RATE, INFERENCE_STEPS, INFERENCE_RATE)
    chemical, plasticity = model.get_chemical_state(), model.get_plasticity_state()
    for state in (chemical, plasticity):
        assert type(state) is np.ndarray and state.dtype == np.float64 and state.shape == (3,)
        assert np.isfinite(state).all()
    expected_chemical, expected_plasticity = chemical.copy(), plasticity.copy()
    chemical.fill(123)
    plasticity.fill(0)
    np.testing.assert_array_equal(model.get_chemical_state(), expected_chemical)
    np.testing.assert_array_equal(model.get_plasticity_state(), expected_plasticity)
