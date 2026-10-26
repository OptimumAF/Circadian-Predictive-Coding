"""Bounded least-squares readout diagnostic with the hidden representation frozen.

Fit clipped inverse-tanh physical targets in one fixed-rank linear solve, then
score the original physical targets. This transformed objective is not physical
MSE optimization. No full model, learner update, statistics fit, role conversion,
RNG, deployment, simulator or IO is supplied; solver arithmetic is not counted
as known dense forward work.
"""

from dataclasses import dataclass
import math

import numpy as np

from src.core import continuous_bc as bc
from src.core.continuous_demonstration import ContinuousActionSpec


TARGET_CLIP = 1e-6
RCOND = 1e-12
DEFINITION = "frozen_hidden_clipped_inverse_tanh_minimum_norm_readout_v1"


@dataclass(frozen=True)
class ContinuousReadoutFit:
    coefficients: bc.Array
    singular_values: bc.Array
    predictions_before: bc.Array
    predictions_after: bc.Array
    rows: int
    rank: int
    clipped_targets_per_axis: tuple[int, ...]
    condition_number: float | None
    effective_condition_number: float | None
    coefficient_norm: float
    physical_mse_before: float
    physical_mse_after: float
    work: dict[str, int | bool | str]
    objective_definition: str = DEFINITION
    target_clip: float = TARGET_CLIP
    rcond: float = RCOND


def _inputs(features, targets, parameters, spec):
    owned_spec = bc._spec(spec)
    if owned_spec.observation_dim > 64:
        raise ValueError("readout fitting permits at most64 input coordinates")
    values = bc._features(features, owned_spec)
    if len(values) > 128:
        raise ValueError("readout fitting requires1..128 rows")
    actions = bc._targets(targets, len(values), owned_spec)
    owned_parameters = bc._parameters(parameters, owned_spec)
    if owned_parameters[0].shape[1] > 64:
        raise ValueError("readout fitting permits at most64 hidden units")
    return values, actions, owned_parameters, owned_spec


def _logit_targets(actions, spec):
    _, _, center, half, active = bc._geometry(spec)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        normalized = np.zeros_like(actions)
        normalized[:, active] = (actions[:, active] - center[active]) / half[active]
        clipped = np.clip(normalized, -(1.0 - TARGET_CLIP), 1.0 - TARGET_CLIP)
        logits = np.arctanh(clipped)
    if not np.isfinite(logits).all():
        raise FloatingPointError("readout target logits are not representable and finite")
    counts = tuple(
        int(np.count_nonzero(clipped[:, i] != normalized[:, i])) for i in range(actions.shape[1])
    )
    return logits, counts


def _head(hidden, weights, bias, spec):
    lower, upper, _, _, active = bc._geometry(spec)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        logits = hidden @ weights + bias
        if not np.isfinite(logits).all():
            raise FloatingPointError("readout head logits are not representable and finite")
        normalized = np.tanh(logits)
        predictions = lower * ((1.0 - normalized) * 0.5) + upper * ((1.0 + normalized) * 0.5)
        predictions[:, ~active] = lower[~active]
    if np.any(predictions < lower) or np.any(predictions > upper):
        raise FloatingPointError("readout predictions exceed physical action bounds")
    return bc._array(predictions, (len(hidden), len(spec.action_units)), "readout predictions")


def _solve(design, logits):
    # Why: a single fixed cutoff gives a reproducible numerical rank/minimum-norm
    # convention without selecting a regularizer from these target errors.
    coefficients, _, rank, singular = np.linalg.lstsq(design, logits, rcond=RCOND)
    coefficients = bc._array(
        coefficients, (design.shape[1], logits.shape[1]), "readout coefficients"
    )
    count = min(design.shape)
    singular = bc._array(singular, (count,), "readout singular values")
    if type(rank) is not int and not isinstance(rank, np.integer):
        raise ValueError("readout solver rank must be an integer")
    rank = int(rank)
    if (
        not 0 <= rank <= count
        or np.any(singular < 0.0)
        or np.any(singular[1:] > singular[:-1])
        or rank != int(np.count_nonzero(singular > RCOND * singular[0]))
    ):
        raise ValueError("readout rank/singular values disagree with the fixed cutoff")
    condition = None
    effective = None
    if rank:
        effective = float(singular[0] / singular[rank - 1])
        if rank == count:
            condition = float(singular[0] / singular[-1])
    if any(value is not None and not math.isfinite(value) for value in (condition, effective)):
        raise FloatingPointError("readout condition number is not representable and finite")
    return coefficients, singular, rank, condition, effective


def _coefficient_norm(coefficients):
    scale = float(np.max(np.abs(coefficients)))
    if scale == 0.0:
        return 0.0
    norm = scale * math.sqrt(float(np.sum((coefficients / scale) ** 2)))
    if not math.isfinite(norm):
        raise FloatingPointError("readout coefficient norm is not representable and finite")
    return norm


def _physical_mse(predictions, targets):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        value = float(np.mean((predictions - targets) ** 2))
    if not math.isfinite(value):
        raise FloatingPointError("physical readout MSE is not representable and finite")
    return value


def _work(rows, dimension, hidden, actions, rank):
    rank_values = min(rows, hidden + 1)
    return dict(
        solve_attempts=1,
        solve_completed=1,
        rows=rows,
        input_dimension=dimension,
        hidden_dimension=hidden,
        action_dimension=actions,
        design_columns=hidden + 1,
        column_nullity=hidden + 1 - rank,
        hidden_forward_passes=1,
        head_forward_passes=2,
        feature_values=rows * dimension,
        feature_numeric_bytes=8 * rows * dimension,
        target_values=rows * actions,
        target_numeric_bytes=8 * rows * actions,
        parameter_numeric_bytes=8 * (dimension * hidden + hidden + hidden * actions + actions),
        hidden_values=rows * hidden,
        hidden_numeric_bytes=8 * rows * hidden,
        design_values=rows * (hidden + 1),
        design_numeric_bytes=8 * rows * (hidden + 1),
        logit_target_values=rows * actions,
        logit_target_numeric_bytes=8 * rows * actions,
        coefficient_values=(hidden + 1) * actions,
        coefficient_numeric_bytes=8 * (hidden + 1) * actions,
        singular_values=rank_values,
        singular_numeric_bytes=8 * rank_values,
        prediction_values=2 * rows * actions,
        prediction_numeric_bytes=16 * rows * actions,
        dense_multiply_accumulates=rows * dimension * hidden + 2 * rows * hidden * actions,
        solver_work_known=False,
        scope="successful solve; logical numeric payloads/known forwards exclude solver arithmetic, "
        "workspace, inverse-tanh/elementwise/scoring, validation/copies, Python and RSS",
    )


def fit_continuous_readout(
    features: bc.Array,
    targets: bc.Array,
    parameters: bc.Parameters,
    spec: ContinuousActionSpec,
) -> ContinuousReadoutFit:
    """Fit an owned diagnostic head without changing any hidden/source array."""
    values, actions, owned, owned_spec = _inputs(features, targets, parameters, spec)
    w1, b1, w2, b2 = owned
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        preactivation = values @ w1 + b1
        if not np.isfinite(preactivation).all():
            raise FloatingPointError("frozen hidden preactivation is not representable and finite")
        hidden = np.tanh(preactivation)
    hidden = bc._array(hidden, (len(values), w1.shape[1]), "frozen readout hidden features")
    design = np.concatenate((hidden, np.ones((len(values), 1), dtype=np.float64)), axis=1)
    logits, clipped = _logit_targets(actions, owned_spec)
    before = _head(hidden, w2, b2, owned_spec)
    coefficients, singular, rank, condition, effective = _solve(design, logits)
    after = _head(hidden, coefficients[:-1], coefficients[-1], owned_spec)
    return ContinuousReadoutFit(
        coefficients,
        singular,
        before,
        after,
        len(values),
        rank,
        clipped,
        condition,
        effective,
        _coefficient_norm(coefficients),
        _physical_mse(before, actions),
        _physical_mse(after, actions),
        _work(
            len(values), owned_spec.observation_dim, w1.shape[1], len(owned_spec.action_units), rank
        ),
    )
