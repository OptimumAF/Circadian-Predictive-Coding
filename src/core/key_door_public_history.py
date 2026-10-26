"""Bounded initial-frame memory from local public key/door observations.

Inputs are exact eight-coordinate frames or complete demonstration episodes.
Outputs concatenate the episode's first public frame and its current frame.
There is no cue metadata, context identifier, label, future-frame inference,
learner, task-law discovery, or IO here. The MiniGrid history contract is separate.
"""

from dataclasses import replace
from math import isfinite

from src.core.demonstration_episode import DemonstrationEpisode, copy_episode


def _public_frame(frame: tuple[float, ...]) -> tuple[float, ...]:
    if type(frame) is not tuple or len(frame) != 8:
        raise ValueError("local public frame must be an exact tuple with eight coordinates")
    if any(type(value) not in (int, float) or not -1 <= value <= 1 for value in frame):
        raise ValueError("local public frame requires native finite numbers in [-1, 1]")
    owned = tuple(float(value) for value in frame)
    if any(not isfinite(value) for value in owned):
        raise ValueError("local public frame coordinates must be finite")
    return owned


class PublicInitialFrameHistory:
    """Keep exactly one owned initial frame, reset independently each episode."""

    def __init__(self) -> None:
        self._initial: tuple[float, ...] | None = None

    def reset(self, initial: tuple[float, ...]) -> tuple[float, ...]:
        # Why: refusing a malformed new episode must not retain the preceding
        # episode's initial cue. Clear before validating the replacement.
        self.clear()
        owned = _public_frame(initial)
        self._initial = owned
        return owned + owned

    def encode(self, current: tuple[float, ...]) -> tuple[float, ...]:
        if self._initial is None:
            raise ValueError("local public history requires reset for the current episode")
        return self._initial + _public_frame(current)

    def clear(self) -> None:
        self._initial = None


def encode_public_history_episode(episode: DemonstrationEpisode) -> DemonstrationEpisode:
    """Detach a complete local episode and preserve its metadata and outcomes.

    Current and next observations are encoded independently. Next-frame evidence
    never supplies any current feature, and the initial frame never advances.
    """
    owned = copy_episode(episode)
    if len(owned.transitions[0].observation) != 8:
        raise ValueError("local public history requires original eight-coordinate episode frames")
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
