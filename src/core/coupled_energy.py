"""Versioned binary predictive-coding energy and its exact partial derivatives.

Inputs are borrowed, quiescent float64 parameters, fixed inputs/targets, current
hidden latents and bounded positive precisions. Outputs own their gradient arrays.
No inference loop, optimizer, data-role authorization, IO or legacy model changes.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
COUPLED_ENERGY_ID = "numpy_coupled_binary_energy_v1"
MIN_PRECISION = 1e-6
MAX_PRECISION = 1e6


@dataclass(frozen=True)
class CoupledParameters:
    hidden_weights: tuple[Array, ...]
    hidden_biases: tuple[Array, ...]
    output_weight: Array
    output_bias: Array


@dataclass(frozen=True)
class CoupledEvaluation:
    energy: float
    task_loss: float
    hidden_penalties: tuple[float, ...]
    latent_gradients: tuple[Array, ...]
    parameter_gradients: CoupledParameters
    energy_definition: str = COUPLED_ENERGY_ID


def _matrix(value: Array, name: str) -> None:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.float64
        or value.ndim != 2
        or min(value.shape) < 1
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be a nonempty finite float64 matrix")


def _validate_parameters(parameters: CoupledParameters) -> None:
    if type(parameters) is not CoupledParameters:
        raise ValueError("parameters must be CoupledParameters")
    weights, biases = parameters.hidden_weights, parameters.hidden_biases
    if type(weights) is not tuple or not weights:
        raise ValueError("hidden_weights must be a nonempty tuple")
    if type(biases) is not tuple or len(biases) != len(weights):
        raise ValueError("hidden_biases must match hidden_weights")
    previous_width = None
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        _matrix(weight, f"hidden weight {index}")
        _matrix(bias, f"hidden bias {index}")
        if previous_width is not None and weight.shape[0] != previous_width:
            raise ValueError("hidden weights have incompatible adjacent widths")
        if bias.shape != (1, weight.shape[1]):
            raise ValueError("hidden bias must have shape (1, layer width)")
        previous_width = weight.shape[1]
    _matrix(parameters.output_weight, "output weight")
    _matrix(parameters.output_bias, "output bias")
    if parameters.output_weight.shape != (previous_width, 1):
        raise ValueError("output weight must have shape (last hidden width, 1)")
    if parameters.output_bias.shape != (1, 1):
        raise ValueError("output bias must have shape (1, 1)")


def _validate_features(parameters: CoupledParameters, features: Array) -> None:
    _validate_parameters(parameters)
    _matrix(features, "features")
    if features.shape[1] != parameters.hidden_weights[0].shape[0]:
        raise ValueError("feature width differs from input weight width")


def _validate_problem(
    parameters: CoupledParameters,
    features: Array,
    targets: Array,
    latents: tuple[Array, ...],
    precisions: tuple[float, ...],
) -> None:
    _validate_features(parameters, features)
    _matrix(targets, "targets")
    if targets.shape != (len(features), 1) or np.any((targets < 0) | (targets > 1)):
        raise ValueError("targets must have shape (batch, 1) and values in [0, 1]")
    depth = len(parameters.hidden_weights)
    if type(latents) is not tuple or len(latents) != depth:
        raise ValueError("latents must be a tuple matching hidden depth")
    for latent, weight in zip(latents, parameters.hidden_weights):
        _matrix(latent, "latent")
        if latent.shape != (len(features), weight.shape[1]):
            raise ValueError("latent shape differs from batch and hidden width")
    if type(precisions) is not tuple or len(precisions) != depth:
        raise ValueError("precisions must be a tuple matching hidden depth")
    if any(
        type(beta) not in (int, float)
        or not MIN_PRECISION <= beta <= MAX_PRECISION
        or not isfinite(beta)
        for beta in precisions
    ):
        raise ValueError("precisions must be finite numbers in [1e-6, 1e6]")


def _finite(values: tuple[Array, ...]) -> None:
    if any(not np.isfinite(value).all() for value in values):
        raise FloatingPointError("nonfinite coupled energy computation")


def _sigmoid(logits: Array) -> Array:
    # Avoid clipping: the likelihood and its derivative use the same logits.
    exponential = np.exp(-np.abs(logits))
    return np.where(logits >= 0, 1 / (1 + exponential), exponential / (1 + exponential))


def feedforward_latents(parameters: CoupledParameters, features: Array) -> tuple[Array, ...]:
    """Initialize free hidden latents from a target-free feedforward pass."""
    _validate_features(parameters, features)
    latents = []
    previous = features
    with np.errstate(over="raise", invalid="raise"):
        for weight, bias in zip(parameters.hidden_weights, parameters.hidden_biases):
            linear = previous @ weight + bias
            _finite((linear,))
            previous = np.tanh(linear)
            latents.append(previous)
    return tuple(latents)


def predict_coupled(parameters: CoupledParameters, features: Array) -> Array:
    """Target-free probabilities, with no latent relaxation at prediction time."""
    latents = feedforward_latents(parameters, features)
    with np.errstate(over="raise", invalid="raise"):
        logits = latents[-1] @ parameters.output_weight + parameters.output_bias
        _finite((logits,))
        return _sigmoid(logits)


def evaluate_coupled(
    parameters: CoupledParameters,
    features: Array,
    targets: Array,
    latents: tuple[Array, ...],
    precisions: tuple[float, ...],
) -> CoupledEvaluation:
    """Evaluate E and partials holding the other arguments fixed.

    E = mean(binary BCE logits) + sum_l beta_l ||z_l-tanh(z_prev W_l+b_l)||²/(2B).
    All gradients include 1/B. Predictions are rebuilt from CURRENT lower latents.
    This is not a derivative through the inference/optimization procedure.
    """
    _validate_problem(parameters, features, targets, latents, precisions)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        return _evaluate_validated(parameters, features, targets, latents, precisions)


def _evaluate_validated(
    parameters: CoupledParameters,
    features: Array,
    targets: Array,
    latents: tuple[Array, ...],
    precisions: tuple[float, ...],
) -> CoupledEvaluation:
    previous = (features, *latents[:-1])
    predictions = []
    for lower, weight, bias in zip(previous, parameters.hidden_weights, parameters.hidden_biases):
        linear = lower @ weight + bias
        _finite((linear,))
        predictions.append(np.tanh(linear))
    residuals = tuple(z - prediction for z, prediction in zip(latents, predictions))
    batch_size = len(features)
    penalties = tuple(
        float(beta * np.sum(residual * residual) / (2 * batch_size))
        for beta, residual in zip(precisions, residuals)
    )
    logits = latents[-1] @ parameters.output_weight + parameters.output_bias
    _finite((logits,))
    task_loss = float(np.mean(np.logaddexp(0, logits) - targets * logits))
    output_delta = (_sigmoid(logits) - targets) / batch_size
    # Negative residual derivative of tanh prediction, including its precision.
    deltas = tuple(
        -beta * residual * (1 - prediction**2) / batch_size
        for beta, residual, prediction in zip(precisions, residuals, predictions)
    )
    latent_gradients = [
        beta * residual / batch_size for beta, residual in zip(precisions, residuals)
    ]
    for index in range(len(latents) - 1):
        latent_gradients[index] += deltas[index + 1] @ parameters.hidden_weights[index + 1].T
    latent_gradients[-1] += output_delta @ parameters.output_weight.T
    gradients = CoupledParameters(
        tuple(lower.T @ delta for lower, delta in zip(previous, deltas)),
        tuple(np.sum(delta, axis=0, keepdims=True) for delta in deltas),
        latents[-1].T @ output_delta,
        np.sum(output_delta, axis=0, keepdims=True),
    )
    _finite(
        (
            *latent_gradients,
            *gradients.hidden_weights,
            *gradients.hidden_biases,
            gradients.output_weight,
            gradients.output_bias,
        )
    )
    energy = task_loss + sum(penalties)
    if not isfinite(energy):
        raise FloatingPointError("nonfinite coupled energy")
    return CoupledEvaluation(energy, task_loss, penalties, tuple(latent_gradients), gradients)
