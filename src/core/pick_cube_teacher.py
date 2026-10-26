"""Fixed bounded PickCube TRAIN teacher in controller-root metre coordinates.

Inputs are privileged TCP/cube/goal positions; output is one normalized4-axis
action. No simulator, reward, learner, IO, role conversion or tuning is provided.
The frozen50-step engineering schedule is not a learned policy or skill claim.
"""

from dataclasses import dataclass
import math


def _point(value: tuple[float, ...]) -> tuple[float, ...]:
    if type(value) is not tuple or len(value) != 3:
        raise ValueError("root-frame position must be an exact three-coordinate tuple")
    converted = []
    for coordinate in value:
        if type(coordinate) not in (float, int):
            raise ValueError("root-frame coordinates must be finite native numbers")
        try:
            item = float(coordinate)
        except OverflowError as error:
            raise ValueError("root-frame coordinates must be finite") from error
        if not math.isfinite(item) or (type(coordinate) is int and int(item) != coordinate):
            raise ValueError("root-frame coordinates must be finite")
        converted.append(item)
    return tuple(converted)


@dataclass(frozen=True)
class PickCubeGeometry:
    tcp: tuple[float, ...]
    cube: tuple[float, ...]
    goal: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("tcp", "cube", "goal"):
            object.__setattr__(self, name, _point(getattr(self, name)))


def _delta(target: tuple[float, ...], current: tuple[float, ...]) -> tuple[float, ...]:
    values = tuple((a - b) / 0.1 for a, b in zip(target, current, strict=True))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("position error is not representable")
    # Why: these bounds define the teacher's policy; no adapter clipping or
    # relabelling occurs. The accepted Panda controller maps1 to.1m root delta.
    return tuple(max(-1.0, min(1.0, value)) for value in values)


class PickCubeTeacher:
    """One50-call schedule, with scalar state only and no configurable policy."""

    def __init__(self) -> None:
        self._steps = 0

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def stage(self) -> str:
        if self._steps < 8:
            return "align"
        if self._steps < 16:
            return "descend"
        if self._steps < 22:
            return "grasp"
        if self._steps < 40:
            return "lift"
        if self._steps < 50:
            return "settle"
        return "complete"

    def action(self, geometry: PickCubeGeometry) -> tuple[float, ...]:
        if self._steps >= 50:
            raise ValueError("teacher50-step schedule is complete")
        if type(geometry) is not PickCubeGeometry:
            raise ValueError("expected exact PickCubeGeometry")
        owned = PickCubeGeometry(geometry.tcp, geometry.cube, geometry.goal)
        phase = self.stage
        if phase == "align":
            target = (owned.cube[0], owned.cube[1], owned.cube[2] + 0.08)
            translation = _delta(target, owned.tcp)
        elif phase == "descend":
            translation = _delta(owned.cube, owned.tcp)
        elif phase == "lift":
            # The held cube may have an offset from TCP; move its actual
            # position toward the goal rather than assuming a perfect grasp.
            translation = _delta(owned.goal, owned.cube)
        else:
            translation = (0.0, 0.0, 0.0)
        command = (*translation, 1.0 if phase in ("align", "descend") else -1.0)
        self._steps += 1
        return command
