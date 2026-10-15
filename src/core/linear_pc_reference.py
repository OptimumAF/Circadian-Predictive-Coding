"""Scalar-output, identity-activation predictive coding at activity equilibrium.

Borrowed inputs are quiescent, finite float64 matrices with row-oriented weights.
Outputs own their arrays. This pure reference has no biases, skips, parameter
scalings, optimizer, random initialization, or mutable learner/runtime state.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
LINEAR_ENERGY_ID = "linear_clamped_pc_equilibrium_v1"


@dataclass(frozen=True)
class LinearEquilibrium:
    energy: float
    mse: float
    rescaling: float
    predictions: Array
    latents: tuple[Array, ...]
    residuals: tuple[Array, ...]
    weight_gradients: tuple[Array, ...]


def _matrix(value: Array, name: str) -> None:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.float64
        or value.ndim != 2
        or min(value.shape) < 1
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be a nonempty finite float64 matrix")


def _validate_inputs(weights: tuple[Array, ...], features: Array, targets: Array) -> None:
    if type(weights) is not tuple or len(weights) < 2:
        raise ValueError("weights must be a tuple containing at least two layer matrices")
    previous_width = None
    for index, weight in enumerate(weights):
        _matrix(weight, f"weight {index}")
        if previous_width is not None and weight.shape[0] != previous_width:
            raise ValueError("weights have incompatible adjacent widths")
        previous_width = weight.shape[1]
    if previous_width != 1:
        raise ValueError("the final weight must have output width 1")
    _matrix(features, "features")
    _matrix(targets, "targets")
    if features.shape[1] != weights[0].shape[0]:
        raise ValueError("features must match the first weight's input width")
    if targets.shape != (len(features), 1):
        raise ValueError("targets must have shape (batch size, 1)")


def _finite_array(value: Array) -> Array:
    if not np.isfinite(value).all():
        raise ValueError("linear equilibrium computation produced a nonfinite array")
    return value


def _downstream_products(weights: tuple[Array, ...]) -> tuple[Array, ...]:
    # Eq. 6 in row orientation: each hidden layer's product to the scalar output.
    product = np.ones((1, 1), dtype=np.float64)
    products = []
    for weight in reversed(weights[1:]):
        product = _finite_array(weight @ product)
        products.append(product)
    return tuple(reversed(products))


def _equilibrium_activities(
    weights: tuple[Array, ...],
    features: Array,
    targets: Array,
    output_residual: Array,
    products: tuple[Array, ...],
) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
    latents = []
    residuals = []
    previous = features
    for weight, product in zip(weights[:-1], products):
        prediction = _finite_array(previous @ weight)
        residual = _finite_array(output_residual @ product.T)
        previous = _finite_array(prediction + residual)
        latents.append(previous)
        # Report the actual represented energy, including floating point rounding.
        residuals.append(_finite_array(previous - prediction))
    residuals.append(_finite_array(targets - previous @ weights[-1]))
    return tuple(latents), tuple(residuals)


def _evaluate_validated(
    weights: tuple[Array, ...], features: Array, targets: Array
) -> LinearEquilibrium:
    predictions = features
    for weight in weights:
        predictions = _finite_array(predictions @ weight)
    products = _downstream_products(weights)
    rescaling = 1.0 + sum(float(np.sum(product * product)) for product in products)
    if not isfinite(rescaling):
        raise ValueError("linear equilibrium rescaling is nonfinite")
    error = _finite_array(targets - predictions)
    latents, residuals = _equilibrium_activities(
        weights, features, targets, error / rescaling, products
    )
    batch_size = len(features)
    energy = sum(float(np.sum(residual * residual)) for residual in residuals) / (2 * batch_size)
    mse = float(np.sum(error * error)) / (2 * batch_size)
    # At equilibrium the envelope theorem removes derivatives through inference.
    gradients = tuple(
        _finite_array(-(previous.T @ residual) / batch_size)
        for previous, residual in zip((features, *latents), residuals)
    )
    if not isfinite(energy) or not isfinite(mse):
        raise ValueError("linear equilibrium energy or MSE is nonfinite")
    return LinearEquilibrium(energy, mse, rescaling, predictions, latents, residuals, gradients)


def evaluate_linear_equilibrium(
    weights: tuple[Array, ...], features: Array, targets: Array
) -> LinearEquilibrium:
    """Return the exact clamped equilibrium and local weight gradients.

    With z0=X, zL=y and e_l=z_l-z_(l-1) W_l, E=sum_l ||e_l||²/(2B).
    The scalar-output equilibrium satisfies E=MSE/s (paper Eqs. 5–6), where
    MSE=||y-feedforward(X)||²/(2B). Gradients include 1/B and hold the inferred
    activities fixed. Targets may be any real values; predictions are target free.
    Numerical overflow is refused as ValueError without changing borrowed inputs.
    """
    _validate_inputs(weights, features, targets)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _evaluate_validated(weights, features, targets)
    except FloatingPointError as error:
        raise ValueError("linear equilibrium computation overflowed or became nonfinite") from error
