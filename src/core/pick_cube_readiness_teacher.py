"""Geometry readiness for one bounded privileged TRAIN teaching trajectory.

Input: exact root-frame TCP/cube/goal positions. Output: one stage and normalized
four-axis command. Scalar state remembers six closing commands and at most fifty
decisions. No simulator, rewards, learner, data-role conversion, IO or tuning.
Proximity permits a grasp attempt; it does not prove contact or a secure grasp.
"""

from dataclasses import dataclass
import math

from src.core.pick_cube_teacher import PickCubeGeometry, _delta

DEFINITION = "pick_cube_geometry_readiness_v1"
OBSERVED_GRASP_DEFINITION = "pick_cube_geometry_observed_grasp_readiness_v1"
GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION = "pick_cube_geometry_observed_grasp_goal_consistent_v2"


@dataclass(frozen=True)
class ReadinessDecision:
    stage: str
    action: tuple[float, ...]


def _error(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    return max(abs(a - b) for a, b in zip(first, second, strict=True))


def _hover(geometry: PickCubeGeometry) -> tuple[float, ...]:
    return (geometry.cube[0], geometry.cube[1], geometry.cube[2] + 0.08)


def _phase(geometry: PickCubeGeometry, previous: str, closing_steps: int) -> str:
    near_cube = _error(geometry.tcp, geometry.cube)
    if previous in ("lift", "settle"):
        if near_cube > 0.06:
            return "align"
        goal_error = _error(geometry.cube, geometry.goal)
        if goal_error <= (0.03 if previous == "settle" else 0.02):
            return "settle"
        return "lift"
    if previous == "grasp":
        if near_cube > 0.02:
            return "align"
        if closing_steps >= 6:
            return "settle" if _error(geometry.cube, geometry.goal) <= 0.02 else "lift"
        return "grasp"
    if previous == "descend":
        if _error(geometry.tcp[:2], geometry.cube[:2]) > 0.02:
            return "align"
        return "grasp" if near_cube <= 0.012 else "descend"
    return "descend" if _error(geometry.tcp, _hover(geometry)) <= 0.01 else "align"


def _command(geometry: PickCubeGeometry, phase: str) -> tuple[float, ...]:
    if phase == "align":
        translation = _delta(_hover(geometry), geometry.tcp)
    elif phase == "descend":
        translation = _delta(geometry.cube, geometry.tcp)
    elif phase == "lift":
        translation = _delta(geometry.goal, geometry.cube)
    else:
        translation = (0.0, 0.0, 0.0)
    return (*translation, 1.0 if phase in ("align", "descend") else -1.0)


class ReadinessPickCubeTeacher:
    """Fixed geometry thresholds, bounded scalar memory, no elapsed-phase switch."""

    def __init__(self) -> None:
        self._steps = 0
        self._phase = "align"
        self._closing_steps = 0

    @property
    def steps(self) -> int:
        return self._steps

    def decide(
        self, geometry: PickCubeGeometry, *, executed_gripper: float | None = None
    ) -> ReadinessDecision:
        if self._steps >= 50:
            raise ValueError("readiness teacher fifty-decision horizon is complete")
        if type(geometry) is not PickCubeGeometry:
            raise ValueError("expected exact PickCubeGeometry")
        owned = PickCubeGeometry(geometry.tcp, geometry.cube, geometry.goal)
        closing_steps = self._closing_steps
        if executed_gripper is not None:
            if type(executed_gripper) not in (float, int):
                raise ValueError("executed gripper must be a finite native normalized command")
            try:
                command = float(executed_gripper)
            except OverflowError as error:
                raise ValueError("executed gripper is not representable") from error
            if not math.isfinite(command) or not -1.0 <= command <= 1.0:
                raise ValueError("executed gripper must be finite and normalized")
            closing_steps = (
                min(6, closing_steps + 1) if self._phase == "grasp" and command < 0 else 0
            )
        phase = _phase(owned, self._phase, closing_steps)
        action = _command(owned, phase)
        decision = ReadinessDecision(phase, action)
        # Why: centimetre readiness tolerances prevent the old elapsed schedule
        # from closing at a distant cube. Six closes retain the original attempt
        # duration; neither proximity nor closing commands establish a grasp.
        # Validate/compute before committing, so a failed decision consumes none.
        # Query labels are not executed commands. When feedback is provided,
        # count only the preceding student's closing intention; the ordinary
        # self-executing teacher retains its original six-command behavior.
        if phase == "grasp" and executed_gripper is None:
            closing_steps = min(6, closing_steps + 1)
        self._closing_steps = closing_steps if phase in ("grasp", "lift", "settle") else 0
        self._phase = phase
        self._steps += 1
        return decision


class ObservedGraspPickCubeTeacher:
    """Opt-in decisions using current official grasp feedback, never close-count contact.

    True feedback selects the existing closed transport/settle commands regardless
    of the hover path. False feedback retains approach tolerances and grasp attempts
    without promoting a closing intention to an observed grasp.
    """

    def __init__(self) -> None:
        # Mandatory feedback makes this a separate interface from the v1 teacher.
        self._steps = 0
        self._phase = "align"
        self._closing_steps = 0

    @property
    def steps(self) -> int:
        return self._steps

    def _transport_phase(self, geometry: PickCubeGeometry) -> str:
        tolerance = 0.03 if self._phase == "settle" else 0.02
        return "settle" if _error(geometry.cube, geometry.goal) <= tolerance else "lift"

    def decide(
        self,
        geometry: PickCubeGeometry,
        *,
        grasped: bool,
        executed_gripper: float | None = None,
    ) -> ReadinessDecision:
        if (
            type(self._steps) is not int
            or not 0 <= self._steps < 50
            or type(self._closing_steps) is not int
            or not 0 <= self._closing_steps <= 6
            or type(self._phase) is not str
            or self._phase not in ("align", "descend", "grasp", "lift", "settle")
        ):
            raise ValueError("observed-grasp teacher state is invalid or horizon complete")
        if type(grasped) is not bool:
            raise ValueError("observed grasp feedback must be an exact bool")
        if type(geometry) is not PickCubeGeometry:
            raise ValueError("expected exact PickCubeGeometry")
        owned = PickCubeGeometry(geometry.tcp, geometry.cube, geometry.goal)
        closing_steps = self._closing_steps
        if executed_gripper is not None:
            if type(executed_gripper) not in (float, int):
                raise ValueError("executed gripper must be a finite native normalized command")
            try:
                command = float(executed_gripper)
            except OverflowError as error:
                raise ValueError("executed gripper is not representable") from error
            if not math.isfinite(command) or not -1.0 <= command <= 1.0:
                raise ValueError("executed gripper must be finite and normalized")
            closing_steps = (
                min(6, closing_steps + 1) if self._phase == "grasp" and command < 0 else 0
            )
        if grasped:
            phase = self._transport_phase(owned)
        else:
            previous = "align" if self._phase in ("lift", "settle") else self._phase
            # Why: keep the v1 geometry thresholds, but a count of commands is
            # not permission to claim contact or issue transport labels.
            phase = _phase(owned, previous, min(closing_steps, 5))
        decision = ReadinessDecision(phase, _command(owned, phase))
        if phase == "grasp" and executed_gripper is None:
            closing_steps = min(6, closing_steps + 1)
        self._closing_steps = closing_steps if phase in ("grasp", "lift", "settle") else 0
        self._phase = phase
        self._steps += 1
        return decision


class GoalConsistentObservedGraspPickCubeTeacher(ObservedGraspPickCubeTeacher):
    """Versioned Euclidean goal settling; a hold command does not certify success."""

    def _transport_phase(self, geometry: PickCubeGeometry) -> str:
        # Why: Panda PickCube placement uses a 25 mm Euclidean radius. Keep the
        # 20 mm entry margin, but never retain a hold beyond that placement bound.
        tolerance = 0.025 if self._phase == "settle" else 0.02
        return "settle" if math.dist(geometry.cube, geometry.goal) <= tolerance else "lift"
