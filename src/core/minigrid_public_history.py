"""Bounded initial-frame memory made only from public MiniGrid observations.

Inputs are normalized148-coordinate frames or complete demonstration episodes.
Outputs concatenate the episode's first frame and the current public frame.
No cue/context metadata, label, teacher, learner, future observation or IO is used.
"""

from dataclasses import replace
from math import isfinite

from src.core.demonstration_episode import DemonstrationEpisode, copy_episode


def _public_frame(frame: tuple[float, ...]) -> tuple[float, ...]:
    if type(frame) is not tuple or len(frame) != 148:
        raise ValueError("public frame must be an exact tuple with148 coordinates")
    if any(type(value) not in (int, float) or not -1 <= value <= 1 for value in frame):
        raise ValueError("public frame requires native finite numbers in [-1, 1]")
    owned = tuple(float(value) for value in frame)
    if any(not isfinite(value) for value in owned):
        raise ValueError("public frame coordinates must be finite")
    return owned


class PublicInitialFrameHistory:
    """Keep one owned public first frame, replaced at every episode reset."""

    def __init__(self) -> None:
        self._initial: tuple[float, ...] | None = None

    def reset(self, initial: tuple[float, ...]) -> tuple[float, ...]:
        # Why: a rejected new episode must not leave a previous episode's cue
        # available. Validate only after clearing the bounded retained frame.
        self.clear()
        owned = _public_frame(initial)
        self._initial = owned
        return owned + owned

    def encode(self, current: tuple[float, ...]) -> tuple[float, ...]:
        if self._initial is None:
            raise ValueError("public history requires reset for the current episode")
        return self._initial + _public_frame(current)

    def clear(self) -> None:
        self._initial = None


def encode_public_history_episode(episode: DemonstrationEpisode) -> DemonstrationEpisode:
    """Detach a complete148-frame episode, preserving its roles and all outcomes.

    Each observation uses only the first and its own current frame. Next-frame
    evidence is transformed separately; it never supplies the current encoding.
    """
    owned = copy_episode(episode)
    if len(owned.transitions[0].observation) != 148:
        raise ValueError("public history episode requires original148-coordinate frames")
    history = PublicInitialFrameHistory()
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
