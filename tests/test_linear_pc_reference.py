"""Independent small linear equilibrium oracles; no scaling or efficacy study.

The least-squares oracle optimizes explicit hidden coordinates. Torch separately
differentiates the paper's scalar quotient, including its weight-dependent
denominator. Fixed arithmetic fixtures avoid any seed or fixture selection.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest
import torch

from src.core.linear_pc_reference import (
    LINEAR_ENERGY_ID,
    evaluate_linear_equilibrium,
)

Array = NDArray[np.float64]
WIDTHS = ((2, 3, 1), (2, 3, 2, 1), (2, 3, 2, 3, 2, 1))


def fixture(widths: tuple[int, ...]) -> tuple[tuple[Array, ...], Array, Array]:
    weights = tuple(
        (np.arange(a * b, dtype=np.float64).reshape(a, b) - (a * b - 1) / 2) / (a * b + layer + 2)
        + 0.11
        for layer, (a, b) in enumerate(zip(widths, widths[1:]))
    )
    features = np.array([[0.3, -0.7], [1.2, 0.2], [-0.4, 0.9]])
    # Real-valued regression targets deliberately extend outside binary labels.
    targets = np.array([[-1.3], [0.4], [2.2]])
    return weights, features, targets


def least_squares_latents(weights: tuple[Array, ...], x: Array, y: Array) -> tuple[Array, ...]:
    """Build a rectangular residual map, independently of closed-form Eq. 5."""
    widths = [weight.shape[1] for weight in weights[:-1]]
    offsets = np.cumsum([0, *widths])
    row_offsets = np.cumsum([0, *widths, 1])
    design = np.zeros((row_offsets[-1], offsets[-1]), dtype=np.float64)
    for layer, width in enumerate(widths):
        rows = slice(row_offsets[layer], row_offsets[layer + 1])
        design[rows, offsets[layer] : offsets[layer + 1]] = np.eye(width)
        if layer:
            design[rows, offsets[layer - 1] : offsets[layer]] = -weights[layer].T
    design[-1:, offsets[-2] : offsets[-1]] = -weights[-1].T
    right = np.zeros((row_offsets[-1], len(x)), dtype=np.float64)
    right[: widths[0]] = (x @ weights[0]).T
    right[-1:] = -y.T
    optimum = np.linalg.lstsq(design, right, rcond=None)[0].T
    return tuple(optimum[:, offsets[i] : offsets[i + 1]] for i in range(len(widths)))


def torch_quotient(weights: tuple[Array, ...], x: Array, y: Array):
    tensors = tuple(
        torch.tensor(weight, dtype=torch.float64, requires_grad=True) for weight in weights
    )
    prediction = torch.tensor(x, dtype=torch.float64)
    for weight in tensors:
        prediction = prediction @ weight
    mse = ((prediction - torch.tensor(y, dtype=torch.float64)) ** 2).sum() / (2 * len(x))
    product = torch.ones((1, 1), dtype=torch.float64)
    rescaling = torch.ones((), dtype=torch.float64)
    for weight in reversed(tensors[1:]):
        product = weight @ product
        rescaling = rescaling + (product**2).sum()
    return tensors, mse, rescaling


@pytest.mark.parametrize("widths", WIDTHS)
def test_should_match_block_least_squares_and_latent_stationarity(widths):
    weights, x, y = fixture(widths)
    actual = evaluate_linear_equilibrium(weights, x, y)
    optimum = least_squares_latents(weights, x, y)
    assert len(actual.latents) == len(weights) - 1
    assert len(actual.residuals) == len(weights)
    assert len(actual.weight_gradients) == len(weights)
    for expected, observed in zip(optimum, actual.latents, strict=True):
        np.testing.assert_allclose(observed, expected, atol=2e-14, rtol=2e-13)
    states = (x, *optimum, y)
    residuals = tuple(states[i + 1] - states[i] @ w for i, w in enumerate(weights))
    for expected, observed in zip(residuals, actual.residuals, strict=True):
        np.testing.assert_allclose(observed, expected, atol=2e-14, rtol=2e-13)
    energy = sum(float(np.sum(r**2)) for r in residuals) / (2 * len(x))
    assert actual.energy == pytest.approx(energy, abs=2e-14, rel=2e-13)
    assert actual.energy == pytest.approx(actual.mse / actual.rescaling, abs=2e-14)
    for layer in range(len(optimum)):
        np.testing.assert_allclose(
            actual.residuals[layer] - actual.residuals[layer + 1] @ weights[layer + 1].T,
            0,
            atol=2e-14,
        )


@pytest.mark.parametrize("widths", WIDTHS)
def test_should_match_actual_torch_quotient_gradient_for_every_weight(widths):
    weights, x, y = fixture(widths)
    actual = evaluate_linear_equilibrium(weights, x, y)
    tensors, mse, rescaling = torch_quotient(weights, x, y)
    energy = mse / rescaling
    gradients = torch.autograd.grad(energy, tensors)
    assert actual.energy == pytest.approx(energy.item(), abs=2e-14)
    assert actual.mse == pytest.approx(mse.item(), abs=2e-14)
    assert actual.rescaling == pytest.approx(rescaling.item(), abs=2e-14)
    for expected, observed in zip(gradients, actual.weight_gradients, strict=True):
        np.testing.assert_allclose(observed, expected.detach().numpy(), atol=2e-14, rtol=2e-12)


def test_should_include_denominator_derivative_and_distinguish_mse_backprop():
    # Scalar hidden unit gives a transparent finite-width counterexample.
    weights = (np.array([[0.5]]), np.array([[2.0]]))
    x, y = np.array([[1.0]]), np.array([[0.0]])
    actual = evaluate_linear_equilibrium(weights, x, y)
    # MSE=1/2, s=5; d(MSE/s)/dw2 = .5/5 - .5*4/25 = .02.
    assert actual.mse == 0.5
    assert actual.rescaling == 5.0
    assert actual.energy == pytest.approx(0.1)
    np.testing.assert_allclose(actual.weight_gradients[0], [[0.4]], atol=1e-14)
    np.testing.assert_allclose(actual.weight_gradients[1], [[0.02]], atol=1e-14)
    assert actual.weight_gradients[1][0, 0] != pytest.approx(0.5)  # ordinary MSE BP
    assert actual.weight_gradients[1][0, 0] != pytest.approx(0.1)  # detached denominator


@pytest.mark.parametrize("widths", WIDTHS)
def test_should_decrease_reequilibrated_energy_after_tiny_gradient_step(widths):
    weights, x, y = fixture(widths)
    before = evaluate_linear_equilibrium(weights, x, y)
    stepped = tuple(w - 1e-4 * g for w, g in zip(weights, before.weight_gradients, strict=True))
    after = evaluate_linear_equilibrium(stepped, x, y)
    assert after.energy < before.energy


@pytest.mark.parametrize("widths", WIDTHS)
def test_should_preserve_batch_mean_under_duplicate_rows(widths):
    weights, x, y = fixture(widths)
    single = evaluate_linear_equilibrium(weights, x, y)
    doubled = evaluate_linear_equilibrium(weights, np.tile(x, (2, 1)), np.tile(y, (2, 1)))
    assert doubled.energy == pytest.approx(single.energy, abs=2e-14)
    assert doubled.mse == pytest.approx(single.mse, abs=2e-14)
    assert doubled.rescaling == single.rescaling
    for observed, expected in zip(doubled.weight_gradients, single.weight_gradients, strict=True):
        np.testing.assert_allclose(observed, expected, atol=2e-14, rtol=2e-13)


def test_should_return_feedforward_target_free_predictions_and_detached_arrays():
    weights, x, y = fixture(WIDTHS[1])
    originals = tuple(a.copy() for a in (*weights, x, y))
    result = evaluate_linear_equilibrium(weights, x, y)
    other = evaluate_linear_equilibrium(weights, x, -y)
    prediction = x.copy()
    for weight in weights:
        prediction = prediction @ weight
    np.testing.assert_array_equal(result.predictions, prediction)
    np.testing.assert_array_equal(other.predictions, prediction)
    arrays = (result.predictions, *result.latents, *result.residuals, *result.weight_gradients)
    for index, array in enumerate(arrays):
        assert type(array) is np.ndarray
        assert array.dtype == np.float64
        for source in (*weights, x, y):
            assert not np.shares_memory(array, source)
        for other_array in arrays[index + 1 :]:
            assert not np.shares_memory(array, other_array)
    result.predictions.fill(999)
    for observed, expected in zip((*weights, x, y), originals, strict=True):
        np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(other.predictions, prediction)
    with pytest.raises(FrozenInstanceError):
        setattr(result, "energy", 123.0)
    assert LINEAR_ENERGY_ID == "linear_clamped_pc_equilibrium_v1"


@pytest.mark.parametrize(
    "bad_weights",
    [
        [],
        (),
        (np.ones((2, 1)),),
        [np.ones((2, 2)), np.ones((2, 1))],
        (np.ones((2, 2), dtype=np.float32), np.ones((2, 1))),
        (np.ones((2, 2), dtype=np.int64), np.ones((2, 1))),
        ([[1.0, 0.0], [0.0, 1.0]], np.ones((2, 1))),
        (np.ones((2,)), np.ones((2, 1))),
        (np.ones((2, 0)), np.ones((0, 1))),
        (np.ones((2, 2)), np.ones((3, 1))),
        (np.ones((2, 2)), np.ones((2, 2))),
        (np.full((2, 2), np.nan), np.ones((2, 1))),
        (np.ones((2, 2)), np.full((2, 1), np.inf)),
    ],
)
def test_should_refuse_malformed_weight_families_before_computation(bad_weights: Any):
    with pytest.raises(ValueError):
        evaluate_linear_equilibrium(bad_weights, np.ones((3, 2)), np.ones((3, 1)))


@pytest.mark.parametrize(
    "bad_x,bad_y",
    [
        ([[1.0, 2.0]], np.ones((1, 1))),
        (np.ones((3, 2)), [[1.0], [2.0], [3.0]]),
        (np.ones((3, 2), dtype=np.float32), np.ones((3, 1))),
        (np.ones((3, 2)), np.ones((3, 1), dtype=np.int64)),
        (np.ones((3,)), np.ones((3, 1))),
        (np.ones((3, 2)), np.ones((3,))),
        (np.ones((0, 2)), np.ones((0, 1))),
        (np.ones((3, 3)), np.ones((3, 1))),
        (np.ones((3, 2)), np.ones((2, 1))),
        (np.ones((3, 2)), np.ones((3, 2))),
        (np.full((3, 2), np.nan), np.ones((3, 1))),
        (np.ones((3, 2)), np.full((3, 1), -np.inf)),
    ],
)
def test_should_refuse_malformed_batches(bad_x: Any, bad_y: Any):
    weights = (np.ones((2, 2)), np.ones((2, 1)))
    with pytest.raises(ValueError):
        evaluate_linear_equilibrium(weights, bad_x, bad_y)


def test_should_refuse_nonfinite_arithmetic_without_mutating_inputs():
    weights = (np.full((2, 2), 1e308), np.full((2, 1), 1e308))
    x, y = np.full((1, 2), 1e308), np.ones((1, 1))
    originals = tuple(a.copy() for a in (*weights, x, y))
    with pytest.raises(ValueError):
        evaluate_linear_equilibrium(weights, x, y)
    for observed, expected in zip((*weights, x, y), originals, strict=True):
        np.testing.assert_array_equal(observed, expected)
