"""Owned complete demonstration episodes and a bounded replay-row codec.

Inputs are normalized demonstration transitions with 1..16, exactly 148 or 296
observation coordinates and a bound class count; outputs are
validated immutable records, learner feature/action arrays, or one fixed-width
float64 replay row. Rewards and next observations remain evidence, not targets.
There is no learner, objective, RNG, clock, filesystem or replay policy here.

Why this: encoding one whole episode into an existing replay row preserves
transition provenance without introducing a second retention framework.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from src.core.categorical_probability_scores import validate_class_count


def _bounded_number(value: object, name: str) -> float:
    if type(value) not in (int, float) or not -1 <= value <= 1:  # type: ignore[operator]
        raise ValueError(f"{name} must be a native finite number in [-1, 1]")
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _observation(values: object, name: str) -> tuple[float, ...]:
    if type(values) is not tuple:
        raise ValueError(f"{name} must be a tuple with 1..16, exactly 148 or 296 coordinates")
    _observation_dimension(len(values), name)
    return tuple(_bounded_number(value, name) for value in values)


def _text(value: object, name: str, limit: int) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty text with no edge whitespace")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be valid UTF-8 text") from error
    if len(encoded) > limit:
        raise ValueError(f"{name} exceeds {limit} UTF-8 bytes")
    return value


def _dimension(value: object, name: str) -> int:
    if type(value) is not int or not 1 <= value <= 16:
        raise ValueError(f"{name} must be a native integer in 1..16")
    return value


def _observation_dimension(value: object, name: str) -> int:
    # Why: retain the toy width/step bounds while admitting the full normalized
    # MiniGrid 7x7x3 image plus direction, optionally paired with its public first
    # frame. Intermediate widths are not new contracts.
    if type(value) is not int or not (1 <= value <= 16 or value in (148, 296)):
        raise ValueError(f"{name} must be a native integer in 1..16, exactly 148 or 296")
    return value


@dataclass(frozen=True)
class DemonstrationTransition:
    """One owned transition; complete-episode invariants live on the episode."""

    step: int
    observation: tuple[float, ...]
    action: int
    reward: float
    next_observation: tuple[float, ...]
    terminal: bool
    num_classes: int = 2

    def __post_init__(self) -> None:
        if type(self.step) is not int or not 0 <= self.step < 16:
            raise ValueError("transition step must be a native integer in 0..15")
        num_classes = validate_class_count(self.num_classes)
        if type(self.action) is not int or not 0 <= self.action < num_classes:
            raise ValueError(f"transition action must be native integer in 0..{num_classes - 1}")
        if type(self.terminal) is not bool:
            raise ValueError("transition terminal must be a native bool")
        observation = _observation(self.observation, "observation")
        next_observation = _observation(self.next_observation, "next_observation")
        if len(observation) != len(next_observation):
            raise ValueError("transition observations must have equal dimensions")
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "next_observation", next_observation)
        object.__setattr__(self, "reward", _bounded_number(self.reward, "reward"))


def _copy_transition(value: DemonstrationTransition) -> DemonstrationTransition:
    if type(value) is not DemonstrationTransition:
        raise ValueError("episode transitions must be exact DemonstrationTransition records")
    return DemonstrationTransition(
        value.step,
        value.observation,
        value.action,
        value.reward,
        value.next_observation,
        value.terminal,
        value.num_classes,
    )


@dataclass(frozen=True)
class DemonstrationEpisode:
    """A complete ordered demonstration with explicit data role and action count."""

    episode_id: str
    context: str
    role: str
    transitions: tuple[DemonstrationTransition, ...]
    num_classes: int = 2

    def __post_init__(self) -> None:
        _text(self.episode_id, "episode_id", 64)
        _text(self.context, "context", 32)
        num_classes = validate_class_count(self.num_classes)
        if type(self.role) is not str or self.role not in (
            "train",
            "final_test",
            "probe_train",
            "probe_test",
        ):
            raise ValueError("episode role must be train, final_test, probe_train or probe_test")
        if type(self.transitions) is not tuple or not 1 <= len(self.transitions) <= 16:
            raise ValueError("episode transitions must be a tuple with 1..16 records")
        transitions = tuple(_copy_transition(value) for value in self.transitions)
        dimension = len(transitions[0].observation)
        for index, transition in enumerate(transitions):
            if transition.num_classes != num_classes:
                raise ValueError("episode and transition class counts must agree")
            if transition.step != index or len(transition.observation) != dimension:
                raise ValueError(
                    "episode steps must be consecutive with fixed observation dimensions"
                )
            if transition.terminal != (index == len(transitions) - 1):
                raise ValueError("only the last transition must be terminal")
            if index and transitions[index - 1].next_observation != transition.observation:
                raise ValueError("adjacent episode observations must be continuous")
        object.__setattr__(self, "transitions", transitions)


@dataclass(frozen=True)
class EpisodeReplaySchema:
    """Fixed replay geometry; all row bytes count toward existing buffer caps."""

    observation_dim: int
    steps_per_episode: int
    num_classes: int = 2

    def __post_init__(self) -> None:
        _observation_dimension(self.observation_dim, "observation_dim")
        _dimension(self.steps_per_episode, "steps_per_episode")
        validate_class_count(self.num_classes)

    @property
    def encoded_dim(self) -> int:
        dimension = _observation_dimension(self.observation_dim, "observation_dim")
        steps = _dimension(self.steps_per_episode, "steps_per_episode")
        num_classes = validate_class_count(self.num_classes)
        return 98 + int(num_classes > 2) + steps * (2 * dimension + 3)


def copy_episode(episode: DemonstrationEpisode) -> DemonstrationEpisode:
    """Revalidate even frozen-record tampering and detach every transition."""
    if type(episode) is not DemonstrationEpisode:
        raise ValueError("expected an exact DemonstrationEpisode record")
    return DemonstrationEpisode(
        episode.episode_id, episode.context, episode.role, episode.transitions, episode.num_classes
    )


def episode_arrays(
    episode: DemonstrationEpisode,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return owned feature/action arrays; rewards remain episode evidence."""
    owned = copy_episode(episode)
    features = np.array([step.observation for step in owned.transitions], dtype=np.float64)
    targets = np.array([[step.action] for step in owned.transitions], dtype=np.float64)
    return features, targets


def _copy_schema(schema: EpisodeReplaySchema) -> EpisodeReplaySchema:
    if type(schema) is not EpisodeReplaySchema:
        raise ValueError("expected an exact EpisodeReplaySchema record")
    return EpisodeReplaySchema(schema.observation_dim, schema.steps_per_episode, schema.num_classes)


def _encode_text(value: str, limit: int) -> list[float]:
    encoded = value.encode("utf-8")
    return [float(len(encoded)), *map(float, encoded), *([0.0] * (limit - len(encoded)))]


def encode_episode(
    episode: DemonstrationEpisode, schema: EpisodeReplaySchema
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Encode a training episode into one existing replay-buffer row."""
    owned = copy_episode(episode)
    geometry = _copy_schema(schema)
    if owned.role != "train":
        raise ValueError("only training episodes may enter replay")
    if (
        len(owned.transitions) != geometry.steps_per_episode
        or len(owned.transitions[0].observation) != geometry.observation_dim
        or owned.num_classes != geometry.num_classes
    ):
        raise ValueError("episode geometry does not match replay schema")
    values = _encode_text(owned.episode_id, 64) + _encode_text(owned.context, 32)
    # Why: binary wire bytes stay stable; categorical rows bind the full action space.
    if geometry.num_classes > 2:
        values.append(float(geometry.num_classes))
    for transition in owned.transitions:
        values.extend(transition.observation)
        values.extend((float(transition.action), transition.reward))
        values.extend(transition.next_observation)
        values.append(float(transition.terminal))
    return np.array([values], dtype=np.float64), np.zeros((1, 1), dtype=np.float64)


def _decode_text(values: NDArray[np.float64], name: str, limit: int) -> str:
    if (
        np.any(values != np.floor(values))
        or not 1 <= values[0] <= limit
        or np.any(values[1:] < 0)
        or np.any(values[1:] > 255)
    ):
        raise ValueError(f"encoded {name} has invalid byte values or length")
    length = int(values[0])
    if np.any(values[1 + length :] != 0):
        raise ValueError(f"encoded {name} padding must be zero")
    try:
        decoded = bytes(int(value) for value in values[1 : 1 + length]).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"encoded {name} must contain valid UTF-8") from error
    return _text(decoded, name, limit)


def _row(value: object, width: int, name: str) -> NDArray[np.float64]:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float64)
        or value.shape != (1, width)
        or not np.all(np.isfinite(value))
    ):
        raise ValueError(f"{name} must be a finite float64 array of shape (1, {width})")
    return value[0].copy()


def _decode_transition(
    values: NDArray[np.float64], step: int, dimension: int, num_classes: int
) -> DemonstrationTransition:
    action = values[dimension]
    terminal = values[-1]
    if action != math.floor(action) or not 0 <= action < num_classes or terminal not in (0, 1):
        raise ValueError("encoded transition action or terminal is outside its declared range")
    return DemonstrationTransition(
        step,
        tuple(float(value) for value in values[:dimension]),
        int(action),
        float(values[dimension + 1]),
        tuple(float(value) for value in values[dimension + 2 : -1]),
        bool(terminal),
        num_classes,
    )


def decode_episode(
    encoded: NDArray[np.float64], dummy: NDArray[np.float64], schema: EpisodeReplaySchema
) -> DemonstrationEpisode:
    """Decode and revalidate one whole retained training episode."""
    geometry = _copy_schema(schema)
    values = _row(encoded, geometry.encoded_dim, "encoded episode")
    if np.any(_row(dummy, 1, "episode dummy target") != 0):
        raise ValueError("episode dummy target must be zero")
    episode_id = _decode_text(values[:65], "episode_id", 64)
    context = _decode_text(values[65:98], "context", 32)
    offset = 98
    if geometry.num_classes > 2:
        if values[offset] != geometry.num_classes:
            raise ValueError("encoded episode class count does not match replay schema")
        offset += 1
    stride = 2 * geometry.observation_dim + 3
    transitions = tuple(
        _decode_transition(
            values[offset + index * stride : offset + (index + 1) * stride],
            index,
            geometry.observation_dim,
            geometry.num_classes,
        )
        for index in range(geometry.steps_per_episode)
    )
    return DemonstrationEpisode(episode_id, context, "train", transitions, geometry.num_classes)
