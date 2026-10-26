"""Deterministic local key/door dynamics and complete scripted demonstrations.

Inputs are an owned layout, a transient initial cue, and native actions. Outputs
are local eight-feature observations and actual transition outcomes. There is no
learner, IO, RNG, global-map observation, or recurrent cue reconstruction here.
"""

from dataclasses import dataclass
import math
from typing import Any

_DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


@dataclass(frozen=True)
class KeyDoorLayout:
    layout_id: str
    rotation: int = 0
    branch_row: int = 2
    swap_keys: bool = False
    appearance: float = 0.0
    movement_action_map: tuple[int, int, int, int] = (0, 1, 2, 3)

    def __post_init__(self) -> None:
        if (
            type(self.layout_id) is not str
            or not self.layout_id
            or self.layout_id != self.layout_id.strip()
            or len(self.layout_id.encode("utf-8")) > 32
        ):
            raise ValueError("layout_id must be trimmed nonempty text of at most 32 UTF-8 bytes")
        if type(self.rotation) is not int or not 0 <= self.rotation <= 3:
            raise ValueError("rotation must be a native integer in 0..3")
        if type(self.branch_row) is not int or not 1 <= self.branch_row <= 3:
            raise ValueError("branch_row must be a native integer in 1..3")
        if type(self.swap_keys) is not bool:
            raise ValueError("swap_keys must be a native bool")
        if (
            type(self.movement_action_map) is not tuple
            or len(self.movement_action_map) != 4
            or any(
                type(action) is not int or not 0 <= action <= 3
                for action in self.movement_action_map
            )
            or len(set(self.movement_action_map)) != 4
        ):
            raise ValueError(
                "movement_action_map must be an exact tuple of four unique native integers in 0..3"
            )
        if (
            type(self.appearance) not in (int, float)
            or not -1 <= self.appearance <= 1
            or not math.isfinite(self.appearance)
        ):
            raise ValueError("appearance must be a finite native number in [-1, 1]")
        object.__setattr__(self, "appearance", float(self.appearance))


@dataclass(frozen=True)
class KeyDoorStep:
    observation: tuple[float, ...]
    reward: float
    terminal: bool
    success: bool


def _copy_layout(layout: KeyDoorLayout) -> KeyDoorLayout:
    if type(layout) is not KeyDoorLayout:
        raise ValueError("layout must be an exact KeyDoorLayout")
    return KeyDoorLayout(
        layout.layout_id,
        layout.rotation,
        layout.branch_row,
        layout.swap_keys,
        layout.appearance,
        layout.movement_action_map,
    )


def key_door_layout_record(layout: KeyDoorLayout) -> dict[str, Any]:
    """Return validated layout metadata, preserving the legacy identity wire.

    Why: adding an optional control permutation must not silently change saved
    descriptors for the original task. Nonidentity records declare the fixed
    spatial observation basis so a control change cannot masquerade as rotation.
    """
    owned = _copy_layout(layout)
    result: dict[str, Any] = dict(
        layout_id=owned.layout_id,
        rotation=owned.rotation,
        branch_row=owned.branch_row,
        swap_keys=owned.swap_keys,
        appearance=owned.appearance,
    )
    if owned.movement_action_map != (0, 1, 2, 3):
        result.update(
            movement_action_map=owned.movement_action_map,
            observation_basis="spatial_directions_v1",
        )
    return result


def _cue(cue: int) -> int:
    if type(cue) is not int or cue not in (-1, 1):
        raise ValueError("cue must be native integer -1 or 1")
    return cue


class KeyDoorNavigation:
    """One fresh episode; legality depends on visible dynamics, never the target.

    Why: coordinates stay in the unrotated private frame; converting actions and
    local neighbors gives exactly the declared clockwise-rotated environment.
    """

    def __init__(self, layout: KeyDoorLayout, cue: int) -> None:
        self._layout = _copy_layout(layout)
        self._required_key = _cue(cue)
        self._visible_cue = cue
        self._position = (2, 4)
        left_color = -1 if self._layout.swap_keys else 1
        self._keys = {
            (1, self._layout.branch_row): left_color,
            (3, self._layout.branch_row): -left_color,
        }
        self._held_color = 0
        self._door_open = False
        self._terminal = False
        self._success = False

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def success(self) -> bool:
        return self._success

    def _neighbor(self, action: int) -> tuple[int, int]:
        dx, dy = _DIRECTIONS[(action - self._layout.rotation) % 4]
        return self._position[0] + dx, self._position[1] + dy

    def _action_neighbor(self, action: int) -> tuple[int, int]:
        # Why: controls may change while the first four observation channels
        # continue to describe the same spatial directions at the same position.
        return self._neighbor(self._layout.movement_action_map[action])

    def _tile(self, position: tuple[int, int]) -> float:
        if position == (2, 0):
            return 1.0 if self._door_open else 0.75
        if position in self._keys:
            return self._keys[position] * 0.5
        if position[0] == 2 and 1 <= position[1] <= 4:
            return 0.0
        if position in ((1, self._layout.branch_row), (3, self._layout.branch_row)):
            return 0.0
        return -1.0

    @property
    def observation(self) -> tuple[float, ...]:
        return (
            *(self._tile(self._neighbor(action)) for action in range(4)),
            self._tile(self._position),
            float(self._held_color),
            float(self._visible_cue),
            self._layout.appearance,
        )

    @property
    def legal_actions(self) -> tuple[int, ...]:
        if self._terminal:
            return ()
        actions = [
            action
            for action in range(4)
            if self._tile(self._action_neighbor(action)) not in (-1.0, 0.75)
        ]
        if (self._position in self._keys and not self._held_color) or (
            self._position == (2, 1) and not self._door_open and self._held_color
        ):
            actions.append(4)
        return tuple(actions)

    def step(self, action: int) -> KeyDoorStep:
        """Apply exactly one legal decision; refusals leave every field unchanged."""
        if type(action) is not int or not 0 <= action <= 4:
            raise ValueError("action must be native integer in 0..4")
        if self._terminal:
            raise ValueError("key/door episode is terminal")
        if action not in self.legal_actions:
            raise ValueError("key/door action is illegal in the current state")
        self._visible_cue = 0
        reward = 0.0
        if action != 4:
            self._position = self._action_neighbor(action)
            if self._position == (2, 0):
                self._terminal = self._success = True
                reward = 1.0
        elif self._position in self._keys:
            self._held_color = self._keys.pop(self._position)
        elif self._held_color == self._required_key:
            self._door_open = True
        else:
            self._terminal = True
            reward = -1.0
        return KeyDoorStep(self.observation, reward, self._terminal, self._success)


def scripted_actions(layout: KeyDoorLayout, cue: int) -> tuple[int, ...]:
    """Return eight legal optimal teacher actions using the declared hidden cue.

    The teacher is deliberately separate from learner observations and policy
    decisions. Its cue access does not imply a feedforward learner can remember.
    """
    owned = _copy_layout(layout)
    required = _cue(cue)
    left_color = -1 if owned.swap_keys else 1
    toward_key = 3 if required == left_color else 1
    raw = (
        (0,) * (4 - owned.branch_row)
        + (toward_key, 4, (toward_key + 2) % 4)
        + (0,) * (owned.branch_row - 1)
        + (4, 0)
    )
    return tuple(
        action if action == 4 else owned.movement_action_map.index((action + owned.rotation) % 4)
        for action in raw
    )
