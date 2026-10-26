"""Detached, explicitly weighted categorical PC parameter proposal for TRAIN.

Inputs are CPU tensors and explicit validated class weights; outputs own all
storage. This primitive performs finite relaxation and one parameter proposal.
It has no learner publication, adaptive state, optimizer, IO, weight selection,
or evaluation authority. A runtime adapter must admit complete TRAIN sources and
atomically update its own counters before this can become a deployed update.
"""

from dataclasses import dataclass
import math
from typing import Any, cast

from src.core.demonstration_class_weights import validate_class_weights
from src.shared.torch_runtime import require_torch

OBJECTIVE = "categorical_target_weighted_ce_plus_half_row_hidden_squared_sum_v1"


@dataclass(frozen=True)
class CategoricalPCUpdate:
    parameters: tuple[Any, Any, Any, Any]
    gradients: tuple[Any, Any, Any, Any]
    hidden_state: Any
    energy: float
    legacy_diagnostic: float
    nominal_latent_row_iterations: int


def _tensor(value: Any, shape: tuple[int, ...], dtype: Any, name: str) -> None:
    torch = require_torch()
    if (
        type(value) is not torch.Tensor
        or tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device != torch.device("cpu")
        or value.layout != torch.strided
    ):
        raise ValueError(f"{name} requires matching dense CPU tensor geometry and dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be finite")


def _inputs(features, targets, parameters, weights, learning_rate, steps, inference_rate, role):
    torch = require_torch()
    if type(role) is not str or role != "train":
        raise ValueError("weighted PC proposals require explicit train role")
    if type(parameters) is not tuple or len(parameters) != 4:
        raise ValueError("parameters require an exact four-tensor tuple")
    if type(features) is not torch.Tensor or features.ndim != 2:
        raise ValueError("features require a two-dimensional CPU tensor")
    rows, width = features.shape
    if not 1 <= rows <= 54 or not 1 <= width <= 4096:
        raise ValueError("features require1..54 rows and1..4096 coordinates")
    if features.dtype not in (torch.float32, torch.float64):
        raise ValueError("features require float32 or float64")
    if type(parameters[0]) is not torch.Tensor or parameters[0].ndim != 2:
        raise ValueError("feature-hidden weight requires a matrix")
    hidden = parameters[0].shape[1]
    if type(parameters[2]) is not torch.Tensor or parameters[2].ndim != 2:
        raise ValueError("hidden-output weight requires a matrix")
    classes = parameters[2].shape[1]
    if not 1 <= hidden <= 256 or not 2 <= classes <= 16:
        raise ValueError("weighted PC requires1..256 hidden units and2..16 classes")
    _tensor(features, (rows, width), features.dtype, "features")
    _tensor(targets, (rows,), torch.int64, "targets")
    if bool(((targets < 0) | (targets >= classes)).any().item()):
        raise ValueError("targets must be class indices")
    shapes = ((width, hidden), (1, hidden), (hidden, classes), (1, classes))
    for parameter, shape in zip(parameters, shapes, strict=True):
        _tensor(parameter, shape, features.dtype, "parameter")
    owned_weights = validate_class_weights(weights, classes)
    if type(steps) is not int or not 1 <= steps <= 64:
        raise ValueError("inference_steps must be a native integer in1..64")
    for rate in (learning_rate, inference_rate):
        try:
            valid = type(rate) in (int, float) and math.isfinite(rate) and rate > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("learning rates must be native finite positive numbers")
    return owned_weights


def _proposal(torch, features, targets, parameters, weights, rate, steps, inference_rate):
    functional = torch.nn.functional
    feature_hidden, hidden_bias, hidden_output, output_bias = parameters
    rows = float(features.shape[0])
    linear_prior = features @ feature_hidden + hidden_bias
    if not bool(torch.isfinite(linear_prior).all().item()):
        raise FloatingPointError("nonfinite weighted PC linear prior")
    prior = torch.tanh(linear_prior)
    state = prior.clone()
    one_hot = functional.one_hot(targets, num_classes=hidden_output.shape[1]).to(features.dtype)
    target_weights = torch.tensor(weights, dtype=features.dtype)[targets]
    factors = rows * (target_weights / target_weights.sum())[:, None]
    # Why: native PC relaxes with B*dE/dz, then uses dE/dtheta. Keep its
    # exact unweighted operation order when observed target weights are uniform.
    # Absent classes cannot affect the objective or the finite relaxation path.
    unweighted = bool((target_weights == target_weights[0]).all().item())
    for _ in range(steps):
        error = functional.softmax(state @ hidden_output + output_bias, dim=1) - one_hot
        direction = state - prior + error @ hidden_output.T
        if not unweighted:
            direction = direction * factors
        state = state - inference_rate * direction
    return _finish(
        torch,
        features,
        targets,
        parameters,
        prior,
        state,
        one_hot,
        target_weights,
        factors,
        unweighted,
        rate,
        steps,
    )


def _charge(work, name):
    if work is not None:
        work[name] += 1


def _finish(
    torch,
    features,
    targets,
    parameters,
    prior,
    state,
    one_hot,
    target_weights,
    factors,
    unweighted,
    rate,
    steps,
    *,
    work=None,
):
    """Shared final partial gradients; optional work counts arithmetic returns."""
    functional = torch.nn.functional
    _, _, hidden_output, output_bias = parameters
    rows = float(features.shape[0])
    _charge(work, "parameter_gradient_evaluations_attempted")
    logits = state @ hidden_output + output_bias
    error = functional.softmax(logits, dim=1) - one_hot
    residual = state - prior
    prior_gradient = (-residual) * (1.0 - prior * prior)
    weighted_error = error if unweighted else error * factors
    weighted_prior = prior_gradient if unweighted else prior_gradient * factors
    gradients = (
        (features.T @ weighted_prior) / rows,
        torch.mean(weighted_prior, dim=0, keepdim=True),
        (state.T @ weighted_error) / rows,
        torch.mean(weighted_error, dim=0, keepdim=True),
    )
    _charge(work, "parameter_gradient_evaluations_completed")
    _charge(work, "parameter_proposals_attempted")
    candidates = tuple(
        value - rate * gradient for value, gradient in zip(parameters, gradients, strict=True)
    )
    _charge(work, "parameter_proposals_completed")
    _charge(work, "final_energy_evaluations_attempted")
    row_energy = functional.cross_entropy(logits, targets, reduction="none") + 0.5 * (
        residual.square().sum(dim=1)
    )
    energy = (target_weights * row_energy).sum() / target_weights.sum()
    _charge(work, "final_energy_evaluations_completed")
    _charge(work, "legacy_diagnostic_evaluations_attempted")
    legacy = 0.5 * (torch.mean(error * error) + torch.mean(residual * residual))
    _charge(work, "legacy_diagnostic_evaluations_completed")
    for value in (logits, state, energy, legacy, *gradients, *candidates):
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError("nonfinite weighted PC proposal or diagnostic")
    return CategoricalPCUpdate(
        parameters=cast(tuple[Any, Any, Any, Any], candidates),
        gradients=gradients,
        hidden_state=state,
        energy=float(energy.item()),
        legacy_diagnostic=float(legacy.item()),
        nominal_latent_row_iterations=int(rows) * steps,
    )


def class_weighted_pc_update(
    features: Any,
    targets: Any,
    parameters: tuple[Any, Any, Any, Any],
    *,
    class_weights: tuple[float, ...],
    learning_rate: float,
    inference_steps: int,
    inference_learning_rate: float,
    role: str,
) -> CategoricalPCUpdate:
    """Propose one detached update, never differentiating through relaxation.

    E = sum_i w[y_i]*(CE_i + .5*||z_i-prior_i||^2) / sum_i w[y_i].
    Each latent step uses B*dE/dz to preserve the native finite-step convention.
    Parameter gradients are partial dE/dtheta holding the final z fixed. Energy
    is this actual objective; legacy_diagnostic is the original unweighted MSE
    statistic for comparison only. Neither is a final-evaluation metric.
    """
    weights = _inputs(
        features,
        targets,
        parameters,
        class_weights,
        learning_rate,
        inference_steps,
        inference_learning_rate,
        role,
    )
    torch = require_torch()
    # Detachment prevents donor autograd state and output aliases from escaping.
    with torch.no_grad():
        return _proposal(
            torch,
            features.detach().clone(),
            targets.detach().clone(),
            tuple(parameter.detach().clone() for parameter in parameters),
            weights,
            learning_rate,
            inference_steps,
            inference_learning_rate,
        )
