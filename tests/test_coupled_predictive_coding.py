"""Behavioral checks for the opt-in coupled learner's state transitions."""

import numpy as np
import pytest

from src.core.coupled_energy import (
    COUPLED_ENERGY_ID,
    CoupledParameters,
    evaluate_coupled,
    feedforward_latents,
    predict_coupled,
)
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork


def _arrays(parameters):
    return (
        *parameters.hidden_weights,
        *parameters.hidden_biases,
        parameters.output_weight,
        parameters.output_bias,
    )


def _assert_same(actual, expected):
    for left, right in zip(_arrays(actual), _arrays(expected)):
        np.testing.assert_array_equal(left, right)


def _batch():
    return np.array([[0.4, -0.2], [-0.3, 0.1], [0.2, 0.6]]), np.array([[1.0], [0.0], [1.0]])


def test_initialization_uses_deterministic_fan_in_scaling_and_owned_arrays():
    model = CoupledPredictiveCodingNetwork(2, (3, 2), seed=8)
    parameters = model.snapshot_parameters()
    rng = np.random.default_rng(8)
    expected_weights = (
        rng.normal(0.0, 1 / np.sqrt(2), size=(2, 3)),
        rng.normal(0.0, 1 / np.sqrt(3), size=(3, 2)),
    )
    for actual, expected in zip(parameters.hidden_weights, expected_weights):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(parameters.output_weight, rng.normal(0, 1 / np.sqrt(2), (2, 1)))
    for bias in (*parameters.hidden_biases, parameters.output_bias):
        assert np.count_nonzero(bias) == 0
    twin = CoupledPredictiveCodingNetwork(2, (3, 2), seed=8)
    _assert_same(twin.snapshot_parameters(), parameters)
    for array in _arrays(parameters):
        array.fill(99)
    _assert_same(model.snapshot_parameters(), twin.snapshot_parameters())
    assert model.precisions == (1.0, 1.0)


def test_restore_copies_all_parameters_and_prediction_uses_the_restored_state():
    model = CoupledPredictiveCodingNetwork(2, (2,), seed=1)
    supplied = CoupledPredictiveCodingNetwork(2, (2,), seed=9).snapshot_parameters()
    features, _ = _batch()
    expected = predict_coupled(supplied, features)
    model.restore_parameters(supplied)
    for array in _arrays(supplied):
        array.fill(-10)
    np.testing.assert_array_equal(model.predict_proba(features), expected)
    prediction = model.predict_proba(features)
    prediction.fill(0)
    np.testing.assert_array_equal(model.predict_proba(features), expected)


def test_training_matches_manually_reconstructed_simultaneous_kernel_updates():
    model = CoupledPredictiveCodingNetwork(2, (2, 3), seed=4, precisions=(0.7, 1.3))
    features, targets = _batch()
    original_features, original_targets = features.copy(), targets.copy()
    parameters = model.snapshot_parameters()
    latents = feedforward_latents(parameters, features)
    for _ in range(4):
        evaluation = evaluate_coupled(parameters, features, targets, latents, model.precisions)
        latents = tuple(
            z - 0.03 * gradient for z, gradient in zip(latents, evaluation.latent_gradients)
        )
    relaxed = evaluate_coupled(parameters, features, targets, latents, model.precisions)
    gradients = relaxed.parameter_gradients
    expected = CoupledParameters(
        tuple(w - 0.015 * g for w, g in zip(parameters.hidden_weights, gradients.hidden_weights)),
        tuple(b - 0.015 * g for b, g in zip(parameters.hidden_biases, gradients.hidden_biases)),
        parameters.output_weight - 0.015 * gradients.output_weight,
        parameters.output_bias - 0.015 * gradients.output_bias,
    )
    result = model.train_epoch(features, targets, 0.015, 4, 0.03)
    _assert_same(model.snapshot_parameters(), expected)
    assert result.energy == relaxed.energy
    assert result.energy_definition == COUPLED_ENERGY_ID
    np.testing.assert_array_equal(features, original_features)
    np.testing.assert_array_equal(targets, original_targets)


def test_small_fixed_inference_steps_reduce_energy_on_a_tiny_fixture():
    model = CoupledPredictiveCodingNetwork(2, (2, 2), seed=5)
    features, targets = _batch()
    parameters = model.snapshot_parameters()
    latents = feedforward_latents(parameters, features)
    initial = evaluate_coupled(parameters, features, targets, latents, model.precisions)
    result = model.train_epoch(features, targets, 0.01, 5, 0.01)
    assert result.energy < initial.energy


@pytest.mark.parametrize("input_dim", [True, 0, -1, 2.0, np.int64(2)])
def test_invalid_input_dimensions_are_refused(input_dim):
    with pytest.raises(ValueError):
        CoupledPredictiveCodingNetwork(input_dim, (2,), seed=0)


@pytest.mark.parametrize("hidden_dims", [(), [2], (True,), (0,), (-1,), (2.0,)])
def test_invalid_hidden_dimensions_are_refused(hidden_dims):
    with pytest.raises(ValueError):
        CoupledPredictiveCodingNetwork(2, hidden_dims, seed=0)


@pytest.mark.parametrize("seed", [True, -1, 1.0, np.int64(2)])
def test_invalid_seeds_are_refused(seed):
    with pytest.raises(ValueError):
        CoupledPredictiveCodingNetwork(2, (2,), seed=seed)


@pytest.mark.parametrize("precisions", [(), [1.0], (True,), (0.0,), (1e-7,), (1e7,), (np.nan,)])
def test_invalid_precisions_are_refused(precisions):
    with pytest.raises(ValueError):
        CoupledPredictiveCodingNetwork(2, (2,), seed=0, precisions=precisions)


@pytest.mark.parametrize("value", [True, 0, -0.1, np.nan, np.inf, "0.1", np.float64(0.1)])
@pytest.mark.parametrize("rate_name", ["learning_rate", "inference_learning_rate"])
def test_invalid_rates_leave_all_parameters_unchanged(value, rate_name):
    model = CoupledPredictiveCodingNetwork(2, (2,), seed=0)
    features, targets = _batch()
    before = model.snapshot_parameters()
    with pytest.raises(ValueError):
        if rate_name == "learning_rate":
            model.train_epoch(features, targets, value, 1, 0.01)
        else:
            model.train_epoch(features, targets, 0.01, 1, value)
    _assert_same(model.snapshot_parameters(), before)


@pytest.mark.parametrize("steps", [True, 0, -1, 1.0, 10001, np.int64(1)])
def test_invalid_inference_steps_leave_parameters_unchanged(steps):
    model = CoupledPredictiveCodingNetwork(2, (2,), seed=0)
    features, targets = _batch()
    before = model.snapshot_parameters()
    with pytest.raises(ValueError):
        model.train_epoch(features, targets, 0.01, steps, 0.01)
    _assert_same(model.snapshot_parameters(), before)


@pytest.mark.parametrize("fault", ["shape", "dtype", "nonfinite", "depth"])
def test_invalid_restore_leaves_all_parameters_unchanged(fault):
    model = CoupledPredictiveCodingNetwork(2, (2,), seed=0)
    before = model.snapshot_parameters()
    hidden = before.hidden_weights
    if fault == "shape":
        hidden = (np.ones((3, 2)),)
    elif fault == "dtype":
        hidden = (np.ones((2, 2), dtype=np.float32),)
    elif fault == "nonfinite":
        hidden = (np.full((2, 2), np.inf),)
    else:
        hidden = ()
    invalid = CoupledParameters(
        hidden, before.hidden_biases, before.output_weight, before.output_bias
    )
    with pytest.raises(ValueError):
        model.restore_parameters(invalid)
    _assert_same(model.snapshot_parameters(), before)


@pytest.mark.parametrize(
    "features,targets",
    [
        (np.ones((2, 3)), np.ones((2, 1))),
        (np.ones((2, 2)), np.ones((2,))),
        (np.ones((2, 2)), np.array([[2.0], [0.0]])),
        (np.full((2, 2), np.nan), np.ones((2, 1))),
        (np.ones((0, 2)), np.ones((0, 1))),
    ],
)
def test_invalid_training_batches_leave_parameters_unchanged(features, targets):
    model = CoupledPredictiveCodingNetwork(2, (2,), seed=0)
    before = model.snapshot_parameters()
    with pytest.raises(ValueError):
        model.train_epoch(features, targets, 0.01, 1, 0.01)
    _assert_same(model.snapshot_parameters(), before)


@pytest.mark.parametrize("failure", ["latents", "parameters", "energy"])
def test_nonfinite_candidate_updates_are_refused_without_partial_publication(failure):
    model = CoupledPredictiveCodingNetwork(2, (1,), seed=0)
    parameters = CoupledParameters(
        (np.zeros((2, 1)),), (np.zeros((1, 1)),), np.full((1, 1), 1e200), np.zeros((1, 1))
    )
    model.restore_parameters(parameters)
    before = model.snapshot_parameters()
    if failure == "latents":
        features, rate, inference_rate = np.zeros((1, 2)), 1.0, 1e200
    elif failure == "parameters":
        features, rate, inference_rate = np.array([[1e200, 0.0]]), 1e200, 1e-200
    else:
        # The candidate arrays stay finite, but X @ candidate_weight overflows.
        # Publishing before checking the candidate energy would corrupt the model.
        features, rate, inference_rate = np.array([[1e200, 0.0]]), 1.0, 1e-200
    with pytest.raises((ValueError, FloatingPointError)):
        model.train_epoch(features, np.ones((1, 1)), rate, 1, inference_rate)
    _assert_same(model.snapshot_parameters(), before)
