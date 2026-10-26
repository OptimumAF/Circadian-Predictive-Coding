"""Fixed, bounded closed-fist PushCube geometry supervision.

Inputs are observed world positions/grasp and actual executed command history.
Output is one stage and normalized action. This module does not query a simulator,
advance history, fit a learner, infer contact, inspect reward or certify success.
"""

from dataclasses import dataclass
import math

from src.core.continuous_demonstration import _vector
from src.core.cube_taskspace_state import CubeTaskSpaceState, copy_cube_taskspace_state
from src.core.pick_cube_action_history import ActionHistoryState, validate_action_history_state

DEFINITION = "push_cube_closed_fist_xy_v1"
STAGES = ("align", "close", "descend", "push", "hold")


@dataclass(frozen=True)
class PushCubeDecision:
    stage: str
    action: tuple[float, ...]

    def __post_init__(self) -> None:
        action = _vector(self.action, "PushCube teacher command", 4)
        if (
            type(self.stage) is not str
            or self.stage not in STAGES
            or len(action) != 4
            or any(not -0.5 <= value <= 0.5 for value in action[:3])
            or action[3] != -1.0
            or (self.stage in ("close", "hold") and action[:3] != (0.0, 0.0, 0.0))
        ):
            raise ValueError(
                "PushCube decision requires a known stage and bounded closed-fist command"
            )
        object.__setattr__(self, "action", action)


def _finite(values: tuple[float, ...]) -> tuple[float, ...]:
    if not all(math.isfinite(value) for value in values):
        raise ValueError("PushCube teacher geometry arithmetic is not finite")
    return values


def _difference(first: tuple[float, ...], second: tuple[float, ...]) -> tuple[float, ...]:
    return _finite(tuple(a - b for a, b in zip(first, second, strict=True)))


def _near(first: tuple[float, ...], second: tuple[float, ...], tolerance: float) -> bool:
    return max(abs(value) for value in _difference(first, second)) <= tolerance


def _translation(target: tuple[float, ...], tcp: tuple[float, ...]) -> tuple[float, ...]:
    scaled = _finite(tuple(value / 0.1 for value in _difference(target, tcp)))
    # Why: reject overflow before clipping; a bounded command must not hide an
    # unrepresentable geometry calculation.
    return tuple(max(-0.5, min(0.5, value)) for value in scaled)


def _approach_phase(
    state: CubeTaskSpaceState, history: ActionHistoryState, hover: tuple[float, ...]
) -> str:
    if not _near(state.tcp_position, hover, 0.01):
        return "align"
    return "close" if history.closing_count < 6 else "descend"


def _recover_push(
    state: CubeTaskSpaceState, direction: tuple[float, ...], contact: tuple[float, ...]
) -> bool:
    offset = _difference(state.tcp_position, state.object_position)
    projection, perpendicular, height = _finite(
        (
            offset[0] * direction[0] + offset[1] * direction[1],
            offset[0] * direction[1] - offset[1] * direction[0],
            state.tcp_position[2] - contact[2],
        )
    )
    return not -0.10 <= projection <= 0.025 or abs(perpendicular) > 0.03 or abs(height) > 0.025


def _next_phase(
    state: CubeTaskSpaceState,
    history: ActionHistoryState,
    previous: str,
    direction: tuple[float, ...],
    hover: tuple[float, ...],
    contact: tuple[float, ...],
) -> str:
    if previous == "hold":
        return "align"
    if previous in ("align", "close") or history.closing_count < 6:
        # Closed-fist readiness is evidence about executed commands, never a
        # counter of labels that this teacher happened to return.
        return _approach_phase(state, history, hover)
    if previous == "descend":
        if not _near(state.tcp_position[:2], contact[:2], 0.015):
            return "align"
        return "push" if _near(state.tcp_position, contact, 0.01) else "descend"
    return "align" if _recover_push(state, direction, contact) else "push"


class PushCubeTeacher:
    """At most 50 validated decisions, with only a step count and current phase."""

    def __init__(self) -> None:
        self._steps = 0
        self._phase = "align"

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def phase(self) -> str:
        return self._phase

    def decide(self, state: CubeTaskSpaceState, *, history: ActionHistoryState) -> PushCubeDecision:
        if (
            type(self._steps) is not int
            or not 0 <= self._steps < 50
            or type(self._phase) is not str
            or self._phase not in STAGES
        ):
            raise ValueError(
                "PushCube teacher state is invalid or its 50-decision horizon is complete"
            )
        owned = copy_cube_taskspace_state(state)
        prior = validate_action_history_state(history)
        dx, dy = _difference(owned.goal_position[:2], owned.object_position[:2])
        rho = _finite((math.hypot(dx, dy),))[0]
        action: tuple[float, ...]
        if rho <= 0.08:
            # A horizontal hold margin is not the SDK success predicate: an
            # elevated cube can still receive hold while native success is false.
            stage, action = "hold", (0.0, 0.0, 0.0, -1.0)
        else:
            direction = _finite((dx / rho, dy / rho))
            behind = _finite(
                (
                    owned.object_position[0] - 0.06 * direction[0],
                    owned.object_position[1] - 0.06 * direction[1],
                )
            )
            hover = _finite((*behind, owned.object_position[2] + 0.10))
            contact = _finite((*behind, owned.object_position[2] + 0.015))
            stage = _next_phase(owned, prior, self._phase, direction, hover, contact)
            if stage == "close":
                action = (0.0, 0.0, 0.0, -1.0)
            else:
                target = (
                    hover
                    if stage == "align"
                    else contact
                    if stage == "descend"
                    else (owned.goal_position[0], owned.goal_position[1], contact[2])
                )
                action = (*_translation(target, owned.tcp_position), -1.0)
        decision = PushCubeDecision(stage, action)
        # Compute/revalidate everything before committing: invalid state or
        # arithmetic consumes neither a decision nor a phase transition.
        self._phase = stage
        self._steps += 1
        return decision
