"""Read-only physical task gradients against a realized continuous parameter step.

Inputs are released arrays and frozen normalization, outputs scalar diagnostics.
The common BP total derivative is not a PC inferred/gated energy partial. No
learner, optimizer, roles, fitting, RNG or IO. Dot products use ordinary float64
products/reductions (including underflow); unrepresentable outputs fail loudly.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from src.core import continuous_bc as task
from src.core.continuous_demonstration import ContinuousActionSpec
from src.core.observation_standardization import (
    ObservationStandardization,
    standardize_observations,
)


@dataclass
class _StepDirectionWork:
    normalization_attempts: int = 0
    normalization_returns: int = 0
    task_gradient_attempts: int = 0
    task_gradient_returns: int = 0
    summary_attempts: int = 0
    summary_returns: int = 0


@dataclass(frozen=True)
class DirectionSummary:
    gradient_l2: float
    step_l2: float
    momentum_step_l2: float
    gradient_dot_step: float
    gradient_dot_momentum_step: float
    gradient_dot_sgd_step: float
    gradient_step_cosine: float | None
    gradient_momentum_cosine: float | None


@dataclass(frozen=True)
class GroupDirection:
    name: str
    summary: DirectionSummary


@dataclass(frozen=True)
class ContinuousStepDirection:
    definition: str
    objective_definition: str
    task_loss_before: float
    overall: DirectionSummary
    groups: tuple[GroupDirection, ...]
    work: dict[str, int]


def _norm(values: task.Array) -> float:
    result = math.hypot(*(float(value) for value in values.flat))
    if not math.isfinite(result):
        raise FloatingPointError("step diagnostic norm is unrepresentable")
    return result


def _cosine(first: task.Array, second: task.Array, a: float, b: float) -> float | None:
    if a == 0 or b == 0:
        return None
    value = float(np.sum((first / a) * (second / b)))
    # Normalized-vector roundoff can exceed one by a few ulps.
    return min(1.0, max(-1.0, value))


def _summary(gradient, step, inherited, reference) -> DirectionSummary:
    g, d, m = (_norm(value) for value in (gradient, step, inherited))
    dots = tuple(float(np.sum(gradient * value)) for value in (step, inherited, reference))
    if not all(math.isfinite(value) for value in dots):
        raise FloatingPointError("step diagnostic dot is unrepresentable")
    return DirectionSummary(
        g,
        d,
        m,
        dots[0],
        dots[1],
        dots[2],
        _cosine(gradient, step, g, d),
        _cosine(gradient, inherited, g, m),
    )


def diagnose_step_direction(
    features: task.Array,
    targets: task.Array,
    before_parameters: task.Parameters,
    before_velocity: task.Parameters,
    after_parameters: task.Parameters,
    spec: ContinuousActionSpec,
    *,
    standardization: ObservationStandardization | None = None,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    _work: _StepDirectionWork | None = None,
) -> ContinuousStepDirection:
    """Negative gradient_dot_step predicts only first-order physical loss decrease.

    Nominal inherited momentum is -lr*mu*velocity_before, regardless of the
    native objective. Physical half-MSE is explicit even for mixed gripper BP.
    A failed diagnostic never grants permission to discard a paid native update.
    """
    for name, value, lower, upper in (
        ("learning_rate", learning_rate, 0, 1),
        ("momentum", momentum, -1, 1),
    ):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite native number")
        if not lower < value <= upper or (name == "momentum" and not 0 <= value < 1):
            raise ValueError(f"{name} outside permitted range")
    work = _StepDirectionWork() if _work is None else _work
    if type(work) is not _StepDirectionWork or any(
        type(value) is not int or value != 0 for value in asdict(work).values()
    ):
        raise ValueError("step diagnostic work must be a fresh exact record")
    owned_spec = task._spec(spec)
    before, velocity, after = (
        task._parameters(value, owned_spec)
        for value in (before_parameters, before_velocity, after_parameters)
    )
    if any(a.shape != b.shape or a.shape != c.shape for a, b, c in zip(before, velocity, after)):
        raise ValueError("step diagnostic parameter shapes differ")
    work.normalization_attempts += 1
    values = standardize_observations(features, standardization, owned_spec)
    work.normalization_returns += 1
    work.task_gradient_attempts += 1
    result = task.continuous_bc_loss_and_gradients(
        values, targets, before, owned_spec, binary_gripper_axis=None
    )
    work.task_gradient_returns += 1
    work.summary_attempts += 1
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        steps = tuple(b - a for a, b in zip(before, after))
        inherited = tuple((-learning_rate * momentum) * value for value in velocity)
        reference = tuple(-learning_rate * value for value in result.gradients)
        groups = tuple(
            GroupDirection(name, _summary(g, d, m, s))
            for name, g, d, m, s in zip(
                ("w1", "b1", "w2", "b2"), result.gradients, steps, inherited, reference
            )
        )
        overall = _summary(
            *(
                np.concatenate([value.ravel() for value in arrays])
                for arrays in (result.gradients, steps, inherited, reference)
            )
        )
    work.summary_returns += 1
    counts = asdict(work)
    size = sum(value.size for value in result.gradients)
    counts.update(
        gradient_values_returned=size,
        gradient_numeric_bytes_returned=size * 8,
        gradient_rows_returned=len(values),
        derived_task_dense_multiply_accumulates=len(values)
        * (
            2 * owned_spec.observation_dim * before[0].shape[1]
            + 3 * before[0].shape[1] * len(owned_spec.action_units)
        ),
    )
    return ContinuousStepDirection(
        "continuous_task_bp_realized_step_direction_v1",
        task.OBJECTIVE,
        result.loss,
        overall,
        groups,
        counts,
    )
