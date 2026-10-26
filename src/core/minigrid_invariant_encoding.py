"""Canonical public nuisance coordinates for the declared MiniGrid task.

Inputs are exact normalized integer-code public frames or complete episodes.
Outputs preserve spatial order and all non-wall object/color/state codes while
fixing wall color and compass direction. This explicit task representation uses
no layout/context metadata, labels, teacher, future input, hidden state, IO or
environment. It is not a claim of invariance for arbitrary games.
"""

from dataclasses import replace

from src.core.demonstration_episode import DemonstrationEpisode, copy_episode
from src.core.minigrid_public_history import _public_frame

INVARIANT_HISTORY_ENCODING_ID = "minigrid_public_wall_direction_invariant_history_296_v1"
_INTEGER_TOLERANCE = 1e-8
_SCALES = (10, 5, 2)


def _decode_coordinate(value: float, scale: int) -> int:
    decoded = (value + 1.0) * scale / 2.0
    integer = round(decoded)
    if abs(decoded - integer) > _INTEGER_TOLERANCE or not 0 <= integer <= scale:
        raise ValueError("public frame must encode discrete MiniGrid integers within 1e-8 roundoff")
    return integer


def canonicalize_public_frame(frame: tuple[float, ...]) -> tuple[float, ...]:
    """Decode/reencode one public frame; remove only declared nuisance fields."""
    owned = _public_frame(frame)
    encoded: list[float] = []
    for offset in range(0, 147, 3):
        pixel = [
            _decode_coordinate(owned[offset + index], scale) for index, scale in enumerate(_SCALES)
        ]
        if pixel[0] == 2:
            pixel[1] = 5
        encoded.extend(
            2.0 * value / scale - 1.0 for value, scale in zip(pixel, _SCALES, strict=True)
        )
    # Validate the original direction even though this declared representation
    # removes absolute compass information from an egocentric action task.
    _decode_coordinate(owned[147], 3)
    encoded.append(-1.0)
    return tuple(encoded)


def canonicalize_public_history(history: tuple[float, ...]) -> tuple[float, ...]:
    """Canonicalize only the provided initial/current frames, preserving order."""
    if type(history) is not tuple or len(history) != 296:
        raise ValueError("invariant public history requires an exact tuple with 296 coordinates")
    return canonicalize_public_frame(history[:148]) + canonicalize_public_frame(history[148:])


class InvariantInitialFrameHistory:
    """Keep one owned canonical initial frame, independently reset each episode."""

    def __init__(self) -> None:
        self._initial: tuple[float, ...] | None = None

    @property
    def encoding_id(self) -> str:
        return INVARIANT_HISTORY_ENCODING_ID

    def reset(self, initial: tuple[float, ...]) -> tuple[float, ...]:
        # A rejected new episode cannot leave an earlier episode's cue available.
        self.clear()
        owned = canonicalize_public_frame(initial)
        self._initial = owned
        return owned + owned

    def encode(self, current: tuple[float, ...]) -> tuple[float, ...]:
        if self._initial is None:
            raise ValueError("invariant public history requires reset for the current episode")
        return self._initial + canonicalize_public_frame(current)

    def clear(self) -> None:
        self._initial = None


def encode_invariant_history_episode(episode: DemonstrationEpisode) -> DemonstrationEpisode:
    """Detach original 148-frame records and preserve all metadata and outcomes.

    Next observations are transformed separately and never supply current input.
    Complete-record copying validates original continuity before canonicalization
    can remove any nuisance differences.
    """
    owned = copy_episode(episode)
    if len(owned.transitions[0].observation) != 148:
        raise ValueError("invariant history requires original 148-coordinate episode frames")
    history = InvariantInitialFrameHistory()
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
