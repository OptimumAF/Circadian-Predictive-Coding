"""Bounded full-MLP generalized Gauss-Newton proposal for the existing BC loss.

All parameters may change; unchanged kernel loss decides strict acceptance.
Inputs are already released arrays. No learner, roles, RNG, runtime or IO here.
"""

from dataclasses import dataclass
import math
from typing import NoReturn, cast

import numpy as np

from src.core import continuous_bc as bc
from src.core import continuous_head_gauss_newton as head

MAX_ROWS = 220
MAX_PARAMETERS = 1092  # D12/H64/A4; bound dense curvature storage explicitly.
DAMPING = head.DAMPING
RCOND = head.RCOND


@dataclass(frozen=True)
class FullLinearization:
    logit_jacobian: bc.Array
    curvature_weights: bc.Array
    gradient: bc.Array
    predictions: bc.Array
    loss: float
    active_axes: int
    work: dict[str, int | bool]


@dataclass(frozen=True)
class FullGaussNewtonProposal:
    parameters: bc.Parameters
    predictions_before: bc.Array
    predictions_after: bc.Array
    accepted: bool
    reason: str
    loss_before: float
    loss_after: float
    trial_loss: float | None
    step_norm: float
    work: dict[str, int | bool]


class FullGaussNewtonFailed(RuntimeError):
    def __init__(self, reason, work):
        super().__init__(reason)
        self._work = tuple(work.items())

    @property
    def work(self):
        return dict(self._work)


def _work():
    # Keep scalar work compatible with the existing <=32-key failure envelope.
    return dict(
        gradient_requests=0,
        gradient_returns=0,
        gradient_rows=0,
        linearization_forward_attempts=0,
        linearization_forward_returns=0,
        linearization_rows=0,
        jacobian_values=0,
        curvature_values=0,
        gram_attempts=0,
        gram_returns=0,
        gram_values=0,
        solve_attempts=0,
        solve_returns=0,
        solve_rows=0,
        solve_columns=0,
        solve_input_values=0,
        proposal_attempts=0,
        proposal_returns=0,
        accepted_proposals=0,
        known_dense_macs=0,
        known_numeric_bytes=0,
        unknown_partial_work=False,
        solver_work_known=False,
    )


def _fail(error, work) -> NoReturn:
    work["unknown_partial_work"] = True
    raise FullGaussNewtonFailed(
        f"full-network fit failed: {type(error).__name__}: {error}", work
    ) from error


def _inputs(features, targets, parameters, spec, axis, weights):
    declared = bc._spec(spec)
    values = bc._features(features, declared)
    owned = bc._parameters(parameters, declared)
    if (
        declared.observation_dim > 64
        or owned[0].shape[1] > 64
        or len(values) > MAX_ROWS
        or sum(a.size for a in owned) > MAX_PARAMETERS
    ):
        raise ValueError("full-network fit requires<=64inputs/hidden,<=220rows,<=1092parameters")
    actions = bc._targets(targets, len(values), declared)
    bc._binary_gripper_targets(actions, declared, axis)
    owned_weights = bc._row_weights(weights, len(values))
    bc._row_masses(owned_weights)
    return values, actions, owned, declared, owned_weights


def _objective(values, targets, parameters, spec, axis, weights, work):
    work["gradient_requests"] += 1
    work["gradient_rows"] += len(values)
    options = dict(binary_gripper_axis=axis)
    if weights is not None:
        options["row_weights"] = weights
    result = bc.continuous_bc_loss_and_gradients(values, targets, parameters, spec, **options)
    work["gradient_returns"] += 1  # Record the paid ACK before validating fields.
    n, d = values.shape
    h, a = parameters[0].shape[1], targets.shape[1]
    work["known_dense_macs"] += n * (2 * d * h + 3 * h * a)
    if type(result) is not bc.ContinuousBCResult or not math.isfinite(result.loss):
        raise ValueError("original loss kernel returned invalid diagnostics")
    gradients = bc._parameters(result.gradients, spec)
    if any(x.shape != y.shape for x, y in zip(gradients, parameters, strict=True)):
        raise ValueError("original gradient shapes differ from parameter blocks")
    predictions = bc._targets(result.predictions, n, spec)
    gradient = np.concatenate([value.ravel() for value in gradients])
    work["known_numeric_bytes"] += gradient.nbytes + predictions.nbytes
    return result.loss, predictions, gradient


def _jacobian(values, hidden, weights):
    n, d = values.shape
    h, a = weights.shape
    p = d * h + h + h * a + a
    jacobian = np.zeros((n, a, p), dtype=np.float64)
    chain = (1 - hidden**2)[:, :, None] * weights[None, :, :]
    jacobian[:, :, : d * h] = (
        values[:, None, :, None] * chain.transpose(0, 2, 1)[:, :, None, :]
    ).reshape(n, a, d * h)
    jacobian[:, :, d * h : d * h + h] = chain.transpose(0, 2, 1)
    offset = d * h + h
    for axis in range(a):
        jacobian[:, axis, offset + np.arange(h) * a + axis] = hidden
        jacobian[:, axis, offset + h * a + axis] = 1.0
    if not np.isfinite(jacobian).all():
        raise FloatingPointError("full logit Jacobian is not finite")
    return jacobian


def _linearize(values, actions, parameters, spec, axis, weights, work):
    loss, predictions, gradient = _objective(values, actions, parameters, spec, axis, weights, work)
    work["linearization_forward_attempts"] += 1
    work["linearization_rows"] += len(values)
    hidden, normalized, _, _ = bc._forward(values, parameters, spec)
    work["linearization_forward_returns"] += 1
    work["known_dense_macs"] += len(values) * (parameters[0].size + parameters[2].size)
    jacobian = _jacobian(values, hidden, parameters[2])
    _, _, _, _, active = bc._geometry(spec)
    axes = int(np.count_nonzero(active))
    curvature = np.zeros_like(actions)
    if axes:
        masses = bc._row_masses(weights)
        factors = (
            np.full(len(values), 1 / (len(values) * axes)) if masses is None else masses / axes
        )
        curvature[:, active] = factors[:, None] * (1 - normalized[:, active] ** 2) ** 2
        if axis is not None:
            # CE(logit2z) curvature =4p(1-p)=1-tanh(z)^2. The original
            # gradient retains wrong-saturated errors even when curvature is0.
            curvature[:, axis] = factors * (1 - normalized[:, axis] ** 2)
    if not np.isfinite(curvature).all() or np.any(curvature < 0):
        raise FloatingPointError("logit curvature weights are invalid")
    work["jacobian_values"] += jacobian.size
    work["curvature_values"] += curvature.size
    work["known_numeric_bytes"] += jacobian.nbytes + curvature.nbytes
    return FullLinearization(jacobian, curvature, gradient, predictions, loss, axes, dict(work))


def linearize_full_network(
    features, targets, parameters, spec, *, binary_gripper_axis=None, row_weights=None
) -> FullLinearization:
    values, actions, owned, declared, weights = _inputs(
        features, targets, parameters, spec, binary_gripper_axis, row_weights
    )
    work = _work()
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _linearize(values, actions, owned, declared, binary_gripper_axis, weights, work)
    except Exception as error:
        _fail(error, work)


def _step(linear, work):
    jacobian = linear.logit_jacobian.reshape(-1, len(linear.gradient))
    weights = linear.curvature_weights.reshape(-1)
    work["gram_attempts"] += 1
    system = jacobian.T @ (weights[:, None] * jacobian)
    work["gram_returns"] += 1
    work["gram_values"] += system.size
    work["known_dense_macs"] += jacobian.shape[0] * system.size
    system.flat[:: len(system) + 1] += DAMPING
    if not np.isfinite(system).all():
        raise FloatingPointError("damped curvature matrix is not finite")
    rhs = -linear.gradient
    work["solve_attempts"] += 1
    work["solve_rows"] += len(system)
    work["solve_columns"] += len(system)
    work["solve_input_values"] += system.size + rhs.size
    work["known_numeric_bytes"] += system.nbytes + rhs.nbytes
    step, errors, rank, singular = np.linalg.lstsq(system, rhs, rcond=RCOND)
    work["solve_returns"] += 1
    columns = len(rhs)
    step = bc._array(step, (columns,), "full-network step")
    bc._array(errors, (0,), "square damped solve residuals")
    singular = bc._array(singular, (columns,), "full-network singular values")
    if (
        (type(rank) is not int and not isinstance(rank, np.integer))
        or int(rank) != columns
        or np.any(singular <= 0)
        or np.any(singular[1:] > singular[:-1])
    ):
        raise ValueError("full damped solve rank or singular values are invalid")
    return head._cap_step(step)


def _proposal(parameters, before, after, accepted, reason, loss, trial, norm, work):
    return FullGaussNewtonProposal(
        cast(bc.Parameters, tuple(a.copy() for a in parameters)),
        before.copy(),
        after.copy(),
        accepted,
        reason,
        loss,
        trial if accepted else loss,
        trial,
        norm,
        dict(work),
    )


def propose_full_gauss_newton(
    features, targets, parameters, spec, *, binary_gripper_axis=None, row_weights=None
) -> FullGaussNewtonProposal:
    values, actions, owned, declared, weights = _inputs(
        features, targets, parameters, spec, binary_gripper_axis, row_weights
    )
    work = _work()
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            linear = _linearize(
                values, actions, owned, declared, binary_gripper_axis, weights, work
            )
            if not linear.active_axes or not np.any(linear.gradient):
                return _proposal(
                    owned,
                    linear.predictions,
                    linear.predictions,
                    False,
                    "no_active_axes" if not linear.active_axes else "zero_gradient",
                    linear.loss,
                    None,
                    0.0,
                    work,
                )
            step, norm = _step(linear, work)
            work["proposal_attempts"] += 1
            blocks, offset = [], 0
            for block in owned:
                blocks.append(block + step[offset : offset + block.size].reshape(block.shape))
                offset += block.size
            trial_parameters = bc._parameters(cast(bc.Parameters, tuple(blocks)), declared)
            work["proposal_returns"] += 1
            work["known_numeric_bytes"] += sum(a.nbytes for a in trial_parameters)
            trial_loss, after, _ = _objective(
                values, actions, trial_parameters, declared, binary_gripper_axis, weights, work
            )
            accepted = trial_loss < linear.loss
            if accepted:
                work["accepted_proposals"] += 1
            return _proposal(
                trial_parameters if accepted else owned,
                linear.predictions,
                after if accepted else linear.predictions,
                accepted,
                "strict_improvement" if accepted else "no_strict_improvement",
                linear.loss,
                trial_loss,
                norm,
                work,
            )
    except Exception as error:
        _fail(error, work)
