"""Public cue-relative key colors with the original wall/direction normalization.

Inputs are raw normalized public148 frames or initial/current296 histories.
Outputs retain the visible initial cue and all non-key task information, while
encoding matching keys as blue2 and other red/blue keys as red0. No action,
reward, role, context, future frame, teacher, environment, RNG or IO is consulted.
"""

from dataclasses import replace

from src.core.demonstration_episode import DemonstrationEpisode, copy_episode
from src.core.minigrid_invariant_encoding import (
    _decode_coordinate,
    canonicalize_public_history,
)


CUE_RELATIVE_HISTORY_ENCODING_ID = "minigrid_public_cue_relative_key_color_history_296_v1"


def cue_relative_public_history(history: tuple[float, ...]) -> tuple[float, ...]:
    """Apply once to raw public initial/current coordinates, never encoded output.

    Why: keeping the initial red/blue cue makes the key remapping invertible
    relative to the old canonical representation. A red cue swaps key colors,
    so this transform is deliberately not idempotent. Both raw frames, including
    every cell and direction, are validated before cue lookup or any remapping.
    The current frame never chooses or replaces the initial cue.
    """
    canonical = canonicalize_public_history(history)
    cues = [
        _decode_coordinate(canonical[offset + 1], 5)
        for offset in range(0, 147, 3)
        if _decode_coordinate(canonical[offset], 10) == 6
    ]
    if len(cues) != 1 or cues[0] not in (0, 2):
        raise ValueError(
            "cue-relative history requires exactly one initial visible red or blue cue"
        )
    keys = []
    for frame_start in (0, 148):
        for cell in range(0, 147, 3):
            offset = frame_start + cell
            if _decode_coordinate(canonical[offset], 10) == 5:
                color = _decode_coordinate(canonical[offset + 1], 5)
                if color not in (0, 2):
                    raise ValueError(
                        "cue-relative history requires every public key to be red or blue"
                    )
                keys.append((offset + 1, color))
    encoded = list(canonical)
    for offset, color in keys:
        encoded[offset] = 2.0 * (2 if color == cues[0] else 0) / 5.0 - 1.0
    return tuple(encoded)


class CueRelativeInitialFrameHistory:
    """Retain one pre-relative canonical initial frame, reset independently.

    Reusing the pre-relative initial coordinates prevents red-cue double
    application when multiple current frames are encoded during one episode.
    """

    def __init__(self) -> None:
        self._initial: tuple[float, ...] | None = None

    @property
    def encoding_id(self) -> str:
        return CUE_RELATIVE_HISTORY_ENCODING_ID

    def reset(self, initial: tuple[float, ...]) -> tuple[float, ...]:
        # A rejected new episode must not retain the preceding episode's cue.
        self.clear()
        if type(initial) is not tuple or len(initial) != 148:
            raise ValueError("cue-relative reset requires an exact raw148-coordinate frame")
        raw = canonicalize_public_history(initial + initial)
        encoded = cue_relative_public_history(raw)
        self._initial = raw[:148]
        return encoded

    def encode(self, current: tuple[float, ...]) -> tuple[float, ...]:
        if self._initial is None:
            raise ValueError("cue-relative history requires reset for the current episode")
        if type(current) is not tuple or len(current) != 148:
            raise ValueError("cue-relative encode requires an exact raw148-coordinate frame")
        return cue_relative_public_history(self._initial + current)

    def clear(self) -> None:
        self._initial = None


def encode_cue_relative_history_episode(episode: DemonstrationEpisode) -> DemonstrationEpisode:
    """Detach a complete original148-frame episode and preserve its metadata.

    Original continuity is checked before nuisance normalization. Next frames
    are transformed independently and never supply current-frame information.
    This pure codec preserves the caller's role; adapters own role admission.
    """
    owned = copy_episode(episode)
    if len(owned.transitions[0].observation) != 148:
        raise ValueError("cue-relative episode encoding requires original148-coordinate frames")
    history = CueRelativeInitialFrameHistory()
    history.reset(owned.transitions[0].observation)
    transitions = tuple(
        replace(
            transition,
            observation=history.encode(transition.observation),
            next_observation=history.encode(transition.next_observation),
        )
        for transition in owned.transitions
    )
    return replace(owned, transitions=transitions)
