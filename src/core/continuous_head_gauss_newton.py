"""One bounded damped Gauss-Newton proposal for the original continuous BC loss.

Freeze the hidden representation and linearize normalized active-axis residuals.
Unlike inverse-tanh fitting, evaluate the original objective and accept only a
strict finite decrease. No target clipping, optimizer state, learner, RNG, role
conversion, deployment, line search, simulator or IO. Solver arithmetic/workspace,
elementwise operations, validation/copies and RSS are additional unknown costs.
"""

from dataclasses import dataclass
import math
from typing import NoReturn

import numpy as np

from src.core import continuous_bc as bc
from src.core.continuous_demonstration import ContinuousActionSpec


DAMPING = 1e-3
MAX_STEP_NORM = 1.0
RCOND = 1e-12
# Why: the explicit joint rehearsal case retains all115 old rows plus15 TRAIN20
# rows. Keep a finite bound rather than silently dropping two original examples.
MAX_ROWS = 130


@dataclass(frozen=True)
class HeadLinearization:
    residuals: bc.Array
    jacobians: tuple[bc.Array, ...]
    design: bc.Array
    predictions: bc.Array
    loss: float
    work: dict[str, int | bool]


@dataclass(frozen=True)
class HeadGaussNewtonProposal:
    head_weights: bc.Array
    head_bias: bc.Array
    predictions_before: bc.Array
    predictions_after: bc.Array
    accepted: bool
    reason: str
    loss_before: float
    loss_after: float
    trial_loss: float | None
    step_norm: float
    work: dict[str, int | bool]


class HeadGaussNewtonFailed(RuntimeError):
    """Failed numerical proposal with detached, permanently observed work."""

    def __init__(self, message: str, work: dict[str, int | bool]):
        super().__init__(message)
        self._work_items = tuple(work.items())

    @property
    def work(self) -> dict[str, int | bool]:
        return dict(self._work_items)


def _work() -> dict[str, int | bool]:
    return dict(
        hidden_forward_attempts=0,
        hidden_forward_completed=0,
        head_forward_attempts=0,
        head_forward_completed=0,
        linearizations_completed=0,
        jacobian_values=0,
        solve_attempts=0,
        solve_completed=0,
        solve_rows=0,
        solve_columns=0,
        solve_input_values=0,
        parameter_proposals_attempted=0,
        parameter_proposals_completed=0,
        accepted_proposals=0,
        known_dense_multiply_accumulates=0,
        failure_work_may_be_unobserved=False,
        solver_work_known=False,
        design_numeric_bytes=0,
        jacobian_numeric_bytes=0,
        residual_numeric_bytes=0,
        proposal_numeric_bytes=0,
    )


def _add(work, key, amount=1):
    work[key] += amount


def _inputs(features, targets, parameters, spec):
    owned_spec = bc._spec(spec)
    if owned_spec.observation_dim > 64:
        raise ValueError("head Gauss-Newton permits at most64 input coordinates")
    values = bc._features(features, owned_spec)
    if len(values) > MAX_ROWS:
        raise ValueError("head Gauss-Newton requires1..130 rows")
    actions = bc._targets(targets, len(values), owned_spec)
    owned = bc._parameters(parameters, owned_spec)
    if owned[0].shape[1] > 64:
        raise ValueError("head Gauss-Newton permits at most64 hidden units")
    return values, actions, owned, owned_spec


def _hidden(values, w1, b1, work):
    _add(work, "hidden_forward_attempts")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        logits = values @ w1 + b1
        if not np.isfinite(logits).all():
            raise FloatingPointError("hidden preactivation is not finite")
        hidden = np.tanh(logits)
    hidden = bc._array(hidden, (len(values), w1.shape[1]), "head Gauss-Newton hidden")
    _add(work, "hidden_forward_completed")
    _add(work, "known_dense_multiply_accumulates", len(values) * w1.shape[0] * w1.shape[1])
    return hidden


def _head(hidden, weights, bias, spec, work):
    _add(work, "head_forward_attempts")
    lower, upper, _, _, active = bc._geometry(spec)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        logits = hidden @ weights + bias
        if not np.isfinite(logits).all():
            raise FloatingPointError("head preactivation is not finite")
        normalized = np.tanh(logits)
        predictions = lower * ((1 - normalized) * 0.5) + upper * ((1 + normalized) * 0.5)
        predictions[:, ~active] = lower[~active]
    predictions = bc._array(predictions, (len(hidden), len(spec.action_units)), "head predictions")
    if np.any(predictions < lower) or np.any(predictions > upper):
        raise FloatingPointError("head prediction exceeds physical action bounds")
    _add(work, "head_forward_completed")
    _add(work, "known_dense_multiply_accumulates", len(hidden) * weights.size)
    return predictions, normalized


def _residuals(normalized, targets, spec):
    _, _, center, half, active = bc._geometry(spec)
    residuals = np.zeros_like(targets)
    if np.any(active):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            residuals[:, active] = (
                normalized[:, active] - (targets[:, active] - center[active]) / half[active]
            ) / math.sqrt(len(targets) * int(np.count_nonzero(active)))
    loss = float(0.5 * np.sum(residuals * residuals))
    if not np.isfinite(residuals).all() or not math.isfinite(loss):
        raise FloatingPointError("original head objective is not finite")
    return residuals, loss, active


def _linearize(values, targets, parameters, spec, work):
    w1, b1, w2, b2 = parameters
    hidden = _hidden(values, w1, b1, work)
    predictions, normalized = _head(hidden, w2, b2, spec, work)
    residuals, loss, active = _residuals(normalized, targets, spec)
    design = np.concatenate((hidden, np.ones((len(values), 1), dtype=np.float64)), axis=1)
    scale = math.sqrt(len(values) * int(np.count_nonzero(active))) if np.any(active) else 1.0
    jacobians = tuple(
        design * ((1 - normalized[:, axis] ** 2) / scale)[:, None]
        if active[axis]
        else np.zeros_like(design)
        for axis in range(targets.shape[1])
    )
    if any(not np.isfinite(jacobian).all() for jacobian in jacobians):
        raise FloatingPointError("head Jacobian is not finite")
    _add(work, "linearizations_completed")
    _add(work, "jacobian_values", sum(jacobian.size for jacobian in jacobians))
    work["design_numeric_bytes"] = design.nbytes
    work["jacobian_numeric_bytes"] = sum(jacobian.nbytes for jacobian in jacobians)
    work["residual_numeric_bytes"] = residuals.nbytes
    return residuals, jacobians, design, predictions, loss, active


def _raise_failed(failure, work) -> NoReturn:
    work["failure_work_may_be_unobserved"] = True
    raise HeadGaussNewtonFailed(
        f"head Gauss-Newton failed: {type(failure).__name__}: {failure}", work
    ) from failure


def linearize_continuous_head(
    features: bc.Array, targets: bc.Array, parameters: bc.Parameters, spec: ContinuousActionSpec
) -> HeadLinearization:
    values, actions, owned, owned_spec = _inputs(features, targets, parameters, spec)
    work = _work()
    try:
        residuals, jacobians, design, predictions, loss, _ = _linearize(
            values, actions, owned, owned_spec, work
        )
        return HeadLinearization(residuals, jacobians, design, predictions, loss, dict(work))
    except Exception as failure:
        _raise_failed(failure, work)


def _solve(jacobian, residuals, work):
    columns = jacobian.shape[1]
    matrix = np.concatenate((jacobian, math.sqrt(DAMPING) * np.eye(columns)), axis=0)
    rhs = np.concatenate((-residuals, np.zeros(columns)))
    _add(work, "solve_attempts")
    _add(work, "solve_rows", matrix.shape[0])
    _add(work, "solve_columns", columns)
    _add(work, "solve_input_values", matrix.size + rhs.size)
    step, errors, rank, singular = np.linalg.lstsq(matrix, rhs, rcond=RCOND)
    # A returned but invalid solver output still consumed its single solve.
    _add(work, "solve_completed")
    step = bc._array(step, (columns,), "head step")
    bc._array(errors, (1,), "head solve residual")
    singular = bc._array(singular, (columns,), "head solve singular values")
    if (
        (type(rank) is not int and not isinstance(rank, np.integer))
        or int(rank) != columns
        or np.any(singular <= 0)
        or np.any(singular[1:] > singular[:-1])
    ):
        raise ValueError("damped head solve rank/singular values are invalid")
    return step


def _cap_step(step):
    scale = float(np.max(np.abs(step)))
    if scale == 0:
        return step.copy(), 0.0
    unit = step / scale
    length = math.sqrt(float(np.sum(unit * unit)))
    if scale > MAX_STEP_NORM / length:
        return unit * (MAX_STEP_NORM / length), MAX_STEP_NORM
    return step.copy(), scale * length


def _proposal(weights, bias, before, after, accepted, reason, loss, trial_loss, norm, work):
    return HeadGaussNewtonProposal(
        weights.copy(),
        bias.copy(),
        before.copy(),
        after.copy(),
        accepted,
        reason,
        loss,
        trial_loss if accepted else loss,
        trial_loss,
        norm,
        dict(work),
    )


def propose_head_gauss_newton(
    features: bc.Array, targets: bc.Array, parameters: bc.Parameters, spec: ContinuousActionSpec
) -> HeadGaussNewtonProposal:
    values, actions, owned, owned_spec = _inputs(features, targets, parameters, spec)
    work = _work()
    try:
        residuals, jacobians, design, before, loss, active = _linearize(
            values, actions, owned, owned_spec, work
        )
        w2, b2 = owned[2:]
        if not np.any(active):
            return _proposal(w2, b2, before, before, False, "no_active_axes", loss, None, 0.0, work)
        _add(work, "parameter_proposals_attempted")
        steps = np.zeros((design.shape[1], actions.shape[1]), dtype=np.float64)
        for axis in np.flatnonzero(active):
            steps[:, axis] = _solve(jacobians[axis], residuals[:, axis], work)
        steps, norm = _cap_step(steps)
        trial_weights, trial_bias = w2.copy(), b2.copy()
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                trial_weights[:, active] += steps[:-1, active]
                trial_bias[active] += steps[-1, active]
            if not np.isfinite(trial_weights).all() or not np.isfinite(trial_bias).all():
                raise FloatingPointError("trial head parameters are not finite")
            _add(work, "parameter_proposals_completed")
            work["proposal_numeric_bytes"] = trial_weights.nbytes + trial_bias.nbytes
            after, normalized = _head(design[:, :-1], trial_weights, trial_bias, owned_spec, work)
            _, trial_loss, _ = _residuals(normalized, actions, owned_spec)
        except FloatingPointError:
            work["failure_work_may_be_unobserved"] = True
            return _proposal(
                w2, b2, before, before, False, "nonfinite_trial", loss, None, norm, work
            )
        if trial_loss < loss:
            _add(work, "accepted_proposals")
            return _proposal(
                trial_weights,
                trial_bias,
                before,
                after,
                True,
                "strict_improvement",
                loss,
                trial_loss,
                norm,
                work,
            )
        return _proposal(
            w2, b2, before, before, False, "no_strict_improvement", loss, trial_loss, norm, work
        )
    except Exception as failure:
        _raise_failed(failure, work)
