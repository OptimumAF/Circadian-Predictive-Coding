"""Pure bounded MLP prediction and dimensionless continuous BC derivatives.

Default loss is half mean squared error in declared half-width coordinates,
averaged over rows and nonconstant axes. An explicit binary-gripper axis uses
logit2z cross entropy on that axis with the same denominator and bounded tanh
prediction. Optional positive row weights preserve all rows, with exact uniform
weights using the original numerical branch. No optimizer, roles, RNG, simulator
or IO here.
"""

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.core.continuous_demonstration import ContinuousActionSpec

Array = NDArray[np.float64]
Parameters = tuple[Array, Array, Array, Array]
OBJECTIVE = "continuous_bc_half_mean_active_axis_normalized_squared_error_v1"
MIXED_OBJECTIVE = "continuous_bc_active_axis_half_mse_plus_binary_gripper_logit2z_ce_v1"
WEIGHTED_OBJECTIVE = "continuous_bc_row_weighted_half_mean_active_axis_normalized_squared_error_v1"
WEIGHTED_MIXED_OBJECTIVE = (
    "continuous_bc_row_weighted_active_axis_half_mse_plus_binary_gripper_logit2z_ce_v1"
)


def _spec(value: ContinuousActionSpec) -> ContinuousActionSpec:
    if type(value) is not ContinuousActionSpec:
        raise ValueError("continuous BC requires an exact ContinuousActionSpec")
    return ContinuousActionSpec(
        value.observation_dim, value.action_units, value.lower_bounds, value.upper_bounds
    )


def _array(value: Array, shape: tuple[int, ...], name: str) -> Array:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.float64
        or value.shape != shape
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be a finite float64 ndarray with shape{shape}")
    return value.copy()


def _features(value: Array, spec: ContinuousActionSpec) -> Array:
    if type(value) is not np.ndarray or value.ndim != 2 or not 1 <= len(value) <= 256:
        raise ValueError("continuous BC features require1..256 rows in a2D ndarray")
    return _array(value, (len(value), spec.observation_dim), "features")


def _row_weights(value: Array | None, rows: int) -> Array | None:
    """Own raw positive weights; never replace exposure counts with weight mass."""
    if type(rows) is not int or not 1 <= rows <= 256:
        raise ValueError("continuous BC row weights require1..256 rows")
    if value is None:
        return None
    owned = _array(value, (rows,), "row_weights")
    if np.any(owned <= 0):
        raise ValueError("row_weights must all be strictly positive")
    return owned


def _row_masses(owned: Array | None) -> Array | None:
    """Normalize robustly; None selects the exact original uniform calculation."""
    if owned is None:
        return None
    if type(owned) is not np.ndarray or owned.ndim != 1:
        raise ValueError("row_weights must be a positive finite float64 vector")
    weights = _row_weights(owned, len(owned))
    assert weights is not None
    if np.all(weights == weights[0]):
        # Why: multiplying by rounded1/N changes the legacy division order.
        return None
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        scaled = weights / np.max(weights)
        masses = scaled / np.sum(scaled)
    if not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("positive normalized row mass is not representable")
    return masses


def _parameters(values: Parameters, spec: ContinuousActionSpec) -> Parameters:
    if type(values) is not tuple or len(values) != 4:
        raise ValueError("continuous BC requires exactly four parameter arrays")
    first = values[0]
    if type(first) is not np.ndarray or first.ndim != 2 or not 1 <= first.shape[1] <= 256:
        raise ValueError("continuous BC hidden dimension must be in1..256")
    hidden, actions = first.shape[1], len(spec.action_units)
    shapes = ((spec.observation_dim, hidden), (hidden,), (hidden, actions), (actions,))
    return cast(
        Parameters,
        tuple(
            _array(value, shape, "parameter") for value, shape in zip(values, shapes, strict=True)
        ),
    )


def _geometry(spec: ContinuousActionSpec) -> tuple[Array, Array, Array, Array, NDArray[np.bool_]]:
    lower, upper = np.asarray(spec.lower_bounds), np.asarray(spec.upper_bounds)
    # Why: (upper-lower)/2 can overflow for valid opposite-sign finite bounds.
    center, half = lower * 0.5 + upper * 0.5, upper * 0.5 - lower * 0.5
    active = upper > lower
    if not np.isfinite(center).all() or not np.isfinite(half).all() or np.any(half[active] <= 0):
        raise ValueError("continuous BC bounds require representable finite centers/half-widths")
    return lower, upper, center, half, active


def _targets(value: Array, rows: int, spec: ContinuousActionSpec) -> Array:
    targets = _array(value, (rows, len(spec.action_units)), "targets")
    lower, upper, _, _, _ = _geometry(spec)
    if np.any(targets < lower) or np.any(targets > upper):
        raise ValueError("continuous BC targets exceed declared physical action bounds")
    return targets


def _binary_gripper_axis(value: int | None, spec: ContinuousActionSpec) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not 0 <= value < len(spec.action_units):
        raise ValueError("binary gripper axis must be a native integer action index or None")
    if (
        spec.action_units[value] != "normalized_gripper"
        or spec.lower_bounds[value] != -1.0
        or spec.upper_bounds[value] != 1.0
    ):
        raise ValueError("binary gripper requires normalized_gripper units and bounds(-1,+1)")
    return value


def _binary_gripper_targets(actions: Array, spec: ContinuousActionSpec, axis: int | None) -> None:
    index = _binary_gripper_axis(axis, spec)
    if index is not None and not np.all((actions[:, index] == -1.0) | (actions[:, index] == 1.0)):
        raise ValueError("binary gripper targets must be exact endpoint labels(-1,+1)")


def _forward(features: Array, parameters: Parameters, spec: ContinuousActionSpec):
    w1, b1, w2, b2 = parameters
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        hidden = np.tanh(features @ w1 + b1)
        logits = hidden @ w2 + b2
        normalized = np.tanh(logits)
        lower, upper, _, _, active = _geometry(spec)
        # A convex blend returns exact endpoints at tanh saturation; no clipping.
        predictions = lower * ((1 - normalized) * 0.5) + upper * ((1 + normalized) * 0.5)
        predictions[:, ~active] = lower[~active]
    if (
        not np.isfinite(predictions).all()
        or np.any(predictions < lower)
        or np.any(predictions > upper)
    ):
        raise FloatingPointError(
            "continuous BC prediction is nonfinite or outside controller bounds"
        )
    return hidden, normalized, predictions, logits


def continuous_bc_predict(
    features: Array, parameters: Parameters, spec: ContinuousActionSpec
) -> Array:
    owned_spec = _spec(spec)
    values = _features(features, owned_spec)
    owned = _parameters(parameters, owned_spec)
    return _forward(values, owned, owned_spec)[2]


@dataclass(frozen=True)
class ContinuousBCResult:
    loss: float
    predictions: Array
    gradients: Parameters


def continuous_bc_loss_and_gradients(
    features: Array,
    targets: Array,
    parameters: Parameters,
    spec: ContinuousActionSpec,
    *,
    binary_gripper_axis: int | None = None,
    row_weights: Array | None = None,
) -> ContinuousBCResult:
    owned_spec = _spec(spec)
    values = _features(features, owned_spec)
    actions = _targets(targets, len(values), owned_spec)
    _binary_gripper_targets(actions, owned_spec, binary_gripper_axis)
    owned = _parameters(parameters, owned_spec)
    masses = _row_masses(_row_weights(row_weights, len(values)))
    hidden, normalized, predictions, logits = _forward(values, owned, owned_spec)
    _, _, center, half, active = _geometry(owned_spec)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        error = np.zeros_like(normalized)
        error[:, active] = (
            normalized[:, active] - (actions[:, active] - center[active]) / half[active]
        )
        denominator = len(values) * int(np.count_nonzero(active))
        if binary_gripper_axis is not None:
            error[:, binary_gripper_axis] = 0.0
        if masses is None:
            loss = 0.0 if denominator == 0 else float(0.5 * np.sum(error * error) / denominator)
            delta = (
                np.zeros_like(error)
                if denominator == 0
                else error * (1 - normalized**2) / denominator
            )
        else:
            axes = int(np.count_nonzero(active))
            loss = 0.0 if axes == 0 else float(0.5 * np.sum(masses[:, None] * error * error) / axes)
            delta = (
                np.zeros_like(error)
                if axes == 0
                else error * (1 - normalized**2) * (masses[:, None] / axes)
            )
        if binary_gripper_axis is not None:
            axis = binary_gripper_axis
            signs = actions[:, axis]
            # tanh(z) maps to binary probability sigmoid(2z). The chain factor
            # two stays explicit; a wrong saturated sign must retain a gradient.
            signed_logits = 2 * signs * logits[:, axis]
            if masses is None:
                loss += float(np.sum(np.logaddexp(0.0, -signed_logits)) / denominator)
                delta[:, axis] = (
                    -2 * signs * np.exp(-np.logaddexp(0.0, signed_logits)) / denominator
                )
            else:
                axes = int(np.count_nonzero(active))
                loss += float(np.sum(masses * np.logaddexp(0.0, -signed_logits)) / axes)
                delta[:, axis] = (
                    -2 * signs * np.exp(-np.logaddexp(0.0, signed_logits)) * masses / axes
                )
        hidden_delta = (delta @ owned[2].T) * (1 - hidden**2)
        gradients = (
            values.T @ hidden_delta,
            hidden_delta.sum(axis=0),
            hidden.T @ delta,
            delta.sum(axis=0),
        )
    if not np.isfinite(loss) or any(not np.isfinite(value).all() for value in gradients):
        raise FloatingPointError("continuous BC loss or gradient is nonfinite")
    return ContinuousBCResult(loss, predictions, gradients)
