"""Declared MiniGrid layouts and a public-observation-only demonstration teacher.

Inputs are explicit bounded layouts or full normalized148-coordinate observations.
Outputs are validated configurations or one native action. No environment object,
private target, learner, RNG, role release, orchestration or IO belongs here.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np

MISSION = "remember the cue, take its key, unlock the neutral door and reach the goal"


@dataclass(frozen=True)
class MiniGridKeyDoorLayout:
    context: str
    rotation: int = 0
    branch_row: int = 3
    swap_keys: bool = False
    wall_color: str = "grey"

    def __post_init__(self) -> None:
        if type(self.context) is not str or self.context not in (
            "A",
            "B",
            "C_train",
            "G_train",
            "D",
            "E-blue",
            "E-red",
            "F-blue",
            "F-red",
        ):
            raise ValueError(
                "context must be A, B, C_train, G_train, D, E-blue/E-red or F-blue/F-red"
            )
        if type(self.rotation) is not int or not 0 <= self.rotation <= 3:
            raise ValueError("rotation must be a native integer in 0..3")
        branch_rows = (
            (5,)
            if self.context == "G_train"
            else (2,)
            if self.context in ("F-blue", "F-red")
            else (3, 4)
        )
        if type(self.branch_row) is not int or self.branch_row not in branch_rows:
            raise ValueError("branch_row must be 5 for G_train, 2 for F probes, otherwise 3 or 4")
        if type(self.swap_keys) is not bool:
            raise ValueError("swap_keys must be a native bool")
        if type(self.wall_color) is not str or self.wall_color not in (
            "grey",
            "blue",
            "red",
            "yellow",
        ):
            raise ValueError("wall_color must be grey, blue, red or yellow")
        if self.context in ("F-blue", "F-red") and (
            self.rotation != 0 or self.swap_keys is not False or self.wall_color != "grey"
        ):
            raise ValueError("F probes require the fixed unrotated, unswapped grey layout")
        if self.context == "G_train" and (
            self.rotation != 0 or self.swap_keys is not False or self.wall_color != "grey"
        ):
            raise ValueError("G_train requires the fixed unrotated, unswapped grey layout")


def predefined_layouts() -> tuple[MiniGridKeyDoorLayout, ...]:
    return (
        MiniGridKeyDoorLayout("A"),
        MiniGridKeyDoorLayout("B", 1, 4, True, "blue"),
        MiniGridKeyDoorLayout("C_train", 2, 3, True, "red"),
        MiniGridKeyDoorLayout("D", 3, 4, False, "yellow"),
    )


def predefined_recurring_train_layout() -> MiniGridKeyDoorLayout:
    """Original ordinary TRAIN geometry, separate from reserved probe/final layouts."""
    # Why: a new key position survives canonical encoding. Native G alone moves
    # its visible cue aside so that the branch-five left key is never overwritten.
    return MiniGridKeyDoorLayout("G_train", branch_row=5)


def predefined_probe_layouts() -> tuple[MiniGridKeyDoorLayout, MiniGridKeyDoorLayout]:
    """Declare cue-specific disposable probe contexts, separate from final D."""
    # Why: distinct cue contexts keep recurrent episodes separate in the existing
    # probe port. This geometry is fixed independently of final D outcomes; the
    # context is report metadata, not an added observation or generalization claim.
    return (
        MiniGridKeyDoorLayout("E-blue", 1, 3, False, "yellow"),
        MiniGridKeyDoorLayout("E-red", 1, 3, False, "yellow"),
    )


def predefined_distinct_probe_layouts() -> tuple[MiniGridKeyDoorLayout, MiniGridKeyDoorLayout]:
    """Move the key branch toward the door for a distinct public task.

    Why: wall color and compass are removed by canonical encoding. A changed
    key position survives that projection while retaining the original cue rule,
    native actions and nine-step path. Observed support is checked by the pilot.
    """
    return (
        MiniGridKeyDoorLayout("F-blue", branch_row=2),
        MiniGridKeyDoorLayout("F-red", branch_row=2),
    )


def _public_image(observation: tuple[float, ...]) -> np.ndarray:
    if (
        type(observation) is not tuple
        or len(observation) != 148
        or any(
            type(value) is not float or not isfinite(value) or not -1 <= value <= 1
            for value in observation
        )
    ):
        raise ValueError("teacher requires a tuple of148 finite native float coordinates")
    scales = np.tile(np.array([10.0, 5.0, 2.0]), 49)
    raw = (np.asarray(observation[:147], dtype=np.float64) + 1.0) * scales / 2.0
    direction = (observation[-1] + 1.0) * 1.5
    if np.any(np.abs(raw - np.rint(raw)) > 1e-8) or abs(direction - round(direction)) > 1e-8:
        raise ValueError("teacher observation must encode exact public MiniGrid integers")
    return np.rint(raw).astype(np.int64).reshape(7, 7, 3)


class CueKeyDoorTeacher:
    """One fresh teacher per episode, with cue and navigation phase from public history.

    Left/right/front references are egocentric public pixels, independent of map
    rotation, key swap or layout metadata. No target is reconstructed after hiding.
    """

    def __init__(self) -> None:
        self._cue: int | None = None
        self._phase = "approach_keys"
        self._turn_back = 0

    def choose_action(self, observation: tuple[float, ...]) -> int:
        image = _public_image(observation)
        if self._cue is None:
            cue_cells = image[image[:, :, 0] == 6]
            if len(cue_cells) != 1 or int(cue_cells[0, 1]) not in (0, 2):
                raise ValueError("teacher requires exactly one initial visible red or blue cue")
            self._cue = int(cue_cells[0, 1])
        if self._phase == "pickup":
            if tuple(image[3, 5, :2]) != (5, self._cue):
                raise ValueError("matching key must be visible in front before pickup")
            self._phase = "face_corridor"
            return 3
        if self._phase == "face_corridor":
            if tuple(image[3, 6, :2]) != (5, self._cue):
                raise ValueError("matching held key must be publicly visible after pickup")
            self._phase = "approach_door"
            return self._turn_back
        if self._phase == "approach_keys":
            candidates = [
                turn for turn, x in ((0, 2), (1, 4)) if tuple(image[x, 6, :2]) == (5, self._cue)
            ]
            if len(candidates) > 1:
                raise ValueError("matching adjacent key must be unique")
            if candidates:
                turn = candidates[0]
                self._turn_back = 1 - turn
                self._phase = "pickup"
                return turn
        front_object, _, state = map(int, image[3, 5])
        if front_object == 4:
            if self._phase != "approach_door":
                raise ValueError("door reached before a matching key was collected")
            return 2 if state == 0 else 5
        if front_object not in (1, 3, 8):
            raise ValueError("teacher path must expose traversable floor, door or goal")
        return 2
