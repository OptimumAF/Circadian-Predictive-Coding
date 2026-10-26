"""Owned continuous predictive-coding energy and exact partial derivatives.

Inputs: quiescent float64 coupled parameters, fixed features/actions, free hidden
latents, declared bounds and precisions. Outputs: owned predictions/gradients.
Task loss matches default continuous BC; hidden residuals use mean batch/summed
neurons. No inference loop, optimizer, role authorization, RNG, simulator or IO.
"""

from dataclasses import dataclass

import numpy as np

from src.core import continuous_bc as bc
from src.core.continuous_bc import Array
from src.core.continuous_demonstration import ContinuousActionSpec
from src.core.coupled_energy import MAX_PRECISION, MIN_PRECISION, CoupledParameters

CONTINUOUS_PC_ENERGY_ID = "numpy_coupled_continuous_energy_v1"


@dataclass(frozen=True)
class ContinuousPCEvaluation:
    energy: float
    task_loss: float
    hidden_penalties: tuple[float, ...]
    predictions: Array
    latent_gradients: tuple[Array, ...]
    parameter_gradients: CoupledParameters
    precisions: tuple[float, ...]
    energy_definition: str = CONTINUOUS_PC_ENERGY_ID
    task_definition: str = bc.OBJECTIVE


def _own_parameters(value: CoupledParameters, spec: ContinuousActionSpec) -> CoupledParameters:
    if (
        type(value) is not CoupledParameters
        or type(value.hidden_weights) is not tuple
        or not 1 <= len(value.hidden_weights) <= 16
        or type(value.hidden_biases) is not tuple
        or len(value.hidden_biases) != len(value.hidden_weights)
    ):
        raise ValueError("continuous PC requires exact coupled parameters with1..16 hidden layers")
    weights, biases = [], []
    previous_width = spec.observation_dim
    for weight, bias in zip(value.hidden_weights, value.hidden_biases, strict=True):
        if type(weight) is not np.ndarray or weight.ndim != 2 or not 1 <= weight.shape[1] <= 256:
            raise ValueError("continuous PC hidden widths must be in1..256")
        width = weight.shape[1]
        weights.append(bc._array(weight, (previous_width, width), "hidden weight"))
        biases.append(bc._array(bias, (1, width), "hidden bias"))
        previous_width = width
    actions = len(spec.action_units)
    return CoupledParameters(
        tuple(weights),
        tuple(biases),
        bc._array(value.output_weight, (previous_width, actions), "output weight"),
        bc._array(value.output_bias, (1, actions), "output bias"),
    )


def _inputs(parameters, features, spec):
    # Why: share strict BC geometry/array contracts without modifying either
    # frozen BC or binary-PC equations to introduce the continuous likelihood.
    owned_spec = bc._spec(spec)
    bc._geometry(owned_spec)
    return _own_parameters(parameters, owned_spec), bc._features(features, owned_spec), owned_spec


def _own_latents(parameters, latents, rows: int) -> tuple[Array, ...]:
    if type(latents) is not tuple or len(latents) != len(parameters.hidden_weights):
        raise ValueError("continuous PC latents must be an exact tuple matching hidden depth")
    return tuple(
        bc._array(latent, (rows, weight.shape[1]), "latent")
        for latent, weight in zip(latents, parameters.hidden_weights, strict=True)
    )


def _own_precisions(precisions, depth: int) -> tuple[float, ...]:
    if (
        type(precisions) is not tuple
        or len(precisions) != depth
        or any(
            type(beta) not in (int, float) or not MIN_PRECISION <= beta <= MAX_PRECISION
            for beta in precisions
        )
    ):
        raise ValueError(
            "continuous PC precisions must match depth and be native numbers in[1e-6,1e6]"
        )
    return tuple(float(beta) for beta in precisions)


def _finite(values: tuple[Array, ...]) -> None:
    if any(not np.isfinite(value).all() for value in values):
        raise FloatingPointError("continuous PC computation is nonfinite")


def _tanh_linear(lower: Array, weight: Array, bias: Array) -> Array:
    linear = lower @ weight + bias
    # BLAS may return inf without surfacing a NumPy floating-point exception.
    # Check before tanh can conceal an unrepresentable preactivation as +/-1.
    _finite((linear,))
    return np.tanh(linear)


def _feedforward(parameters: CoupledParameters, features: Array) -> tuple[Array, ...]:
    previous = features
    latents = []
    for weight, bias in zip(parameters.hidden_weights, parameters.hidden_biases, strict=True):
        previous = _tanh_linear(previous, weight, bias)
        latents.append(previous)
    result = tuple(latents)
    _finite(result)
    return result


def feedforward_continuous_latents(
    parameters: CoupledParameters, features: Array, spec: ContinuousActionSpec
) -> tuple[Array, ...]:
    """Target-free initialization; returned hidden arrays own their storage."""
    owned, values, _ = _inputs(parameters, features, spec)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        return _feedforward(owned, values)


def _output(parameters: CoupledParameters, latent: Array, spec: ContinuousActionSpec):
    normalized = _tanh_linear(latent, parameters.output_weight, parameters.output_bias)
    lower, upper, center, half, active = bc._geometry(spec)
    # Same convex blend and constant-axis behavior as the matched BC head.
    predictions = lower * ((1 - normalized) * 0.5) + upper * ((1 + normalized) * 0.5)
    predictions[:, ~active] = lower[~active]
    _finite((normalized, predictions))
    if np.any(predictions < lower) or np.any(predictions > upper):
        raise FloatingPointError("continuous PC prediction is outside declared bounds")
    return normalized, predictions, center, half, active


def predict_continuous_pc(
    parameters: CoupledParameters, features: Array, spec: ContinuousActionSpec
) -> Array:
    """Target-free bounded feedforward prediction, not latent inference."""
    owned, values, owned_spec = _inputs(parameters, features, spec)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        latents = _feedforward(owned, values)
        return _output(owned, latents[-1], owned_spec)[1]


def _task(parameters, latent, targets, spec):
    normalized, predictions, center, half, active = _output(parameters, latent, spec)
    error = np.zeros_like(normalized)
    error[:, active] = normalized[:, active] - (targets[:, active] - center[active]) / half[active]
    denominator = len(targets) * int(np.count_nonzero(active))
    loss = 0.0 if denominator == 0 else float(0.5 * np.sum(error * error) / denominator)
    delta = np.zeros_like(error) if denominator == 0 else error * (1 - normalized**2) / denominator
    return loss, delta, predictions


def _evaluate(parameters, features, targets, latents, precisions, spec) -> ContinuousPCEvaluation:
    rows = len(features)
    previous = (features, *latents[:-1])
    hidden_predictions = tuple(
        _tanh_linear(lower, weight, bias)
        for lower, weight, bias in zip(
            previous, parameters.hidden_weights, parameters.hidden_biases, strict=True
        )
    )
    residuals = tuple(
        z - prediction for z, prediction in zip(latents, hidden_predictions, strict=True)
    )
    penalties = tuple(
        float(beta * np.sum(residual * residual) / (2 * rows))
        for beta, residual in zip(precisions, residuals, strict=True)
    )
    loss, output_delta, predictions = _task(parameters, latents[-1], targets, spec)
    deltas = tuple(
        -beta * residual * (1 - prediction**2) / rows
        for beta, residual, prediction in zip(
            precisions, residuals, hidden_predictions, strict=True
        )
    )
    latent_gradients = [
        beta * residual / rows for beta, residual in zip(precisions, residuals, strict=True)
    ]
    for index in range(len(latents) - 1):
        latent_gradients[index] += deltas[index + 1] @ parameters.hidden_weights[index + 1].T
    latent_gradients[-1] += output_delta @ parameters.output_weight.T
    gradients = CoupledParameters(
        tuple(lower.T @ delta for lower, delta in zip(previous, deltas, strict=True)),
        tuple(delta.sum(axis=0, keepdims=True) for delta in deltas),
        latents[-1].T @ output_delta,
        output_delta.sum(axis=0, keepdims=True),
    )
    energy = loss + sum(penalties)
    _finite(
        (
            *hidden_predictions,
            *latent_gradients,
            *gradients.hidden_weights,
            *gradients.hidden_biases,
            gradients.output_weight,
            gradients.output_bias,
        )
    )
    if not all(np.isfinite(value) for value in (energy, loss, *penalties)):
        raise FloatingPointError("continuous PC energy is nonfinite")
    return ContinuousPCEvaluation(
        energy, loss, penalties, predictions, tuple(latent_gradients), gradients, precisions
    )


def evaluate_continuous_pc(
    parameters: CoupledParameters,
    features: Array,
    targets: Array,
    latents: tuple[Array, ...],
    precisions: tuple[float, ...],
    spec: ContinuousActionSpec,
) -> ContinuousPCEvaluation:
    """E=normalized BC task loss + sum_l beta_l||Z_l-P_l||²/(2B).

    Gradients are partial derivatives with the other free latents held fixed.
    In particular, feedforward initialization does not make hidden-weight partials
    equal total BP gradients. No inferred state or parameter update is performed.
    """
    owned, values, owned_spec = _inputs(parameters, features, spec)
    actions = bc._targets(targets, len(values), owned_spec)
    hidden = _own_latents(owned, latents, len(values))
    beta = _own_precisions(precisions, len(hidden))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        return _evaluate(owned, values, actions, hidden, beta, owned_spec)
