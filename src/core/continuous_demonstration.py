"""Owned continuous demonstrations with declared units and exact bounds.

Inputs/outputs are immutable native records and training pairs. No simulator,
unit conversion, clipping, learner, budget, data release, replay or IO is provided.
Why: categorical action IDs cannot express a robot controller's continuous units;
keep that existing contract intact and require an explicit continuous boundary.
"""

from dataclasses import dataclass
import math

from src.core.experience import require_training_role


def _text(value: str, name: str, limit: int) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty native text without edge whitespace")
    try:
        size = len(value.encode("utf8"))
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be valid UTF-8") from error
    if size > limit:
        raise ValueError(f"{name} exceeds {limit} UTF-8 bytes")
    return value


def _number(value: float, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite native number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    # Why: rounding an integer action or bound could silently admit a value
    # outside the declared interval before the episode checks its bounds.
    if type(value) is int and int(result) != value:
        raise ValueError(f"{name} integer must be exactly representable as a float")
    return result


def _vector(values: tuple[float, ...], name: str, maximum: int) -> tuple[float, ...]:
    if type(values) is not tuple or not 1 <= len(values) <= maximum:
        raise ValueError(f"{name} must be an exact tuple with1..{maximum} finite numbers")
    return tuple(_number(value, name) for value in values)


@dataclass(frozen=True)
class ContinuousActionSpec:
    observation_dim: int
    action_units: tuple[str, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.observation_dim) is not int or not 1 <= self.observation_dim <= 1024:
            raise ValueError("observation_dim must be a native integer in1..1024")
        if type(self.action_units) is not tuple or not 1 <= len(self.action_units) <= 16:
            raise ValueError("action_units must be an exact tuple with1..16 declared axis units")
        units = tuple(_text(unit, "action unit", 32) for unit in self.action_units)
        lower = _vector(self.lower_bounds, "lower_bounds", 16)
        upper = _vector(self.upper_bounds, "upper_bounds", 16)
        if len(lower) != len(upper) or len(lower) != len(units):
            raise ValueError("each action axis requires one unit, lower bound and upper bound")
        if any(lo > hi for lo, hi in zip(lower, upper, strict=True)):
            raise ValueError("each lower bound must be <= its upper bound")
        object.__setattr__(self, "action_units", units)
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "upper_bounds", upper)


@dataclass(frozen=True)
class ContinuousTransition:
    step: int
    observation: tuple[float, ...]
    action: tuple[float, ...]
    reward: float
    next_observation: tuple[float, ...]
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        if type(self.step) is not int or not 0 <= self.step < 256:
            raise ValueError("transition step must be a native integer in0..255")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise ValueError("terminated and truncated must be exact bools")
        before = _vector(self.observation, "observation", 1024)
        after = _vector(self.next_observation, "next_observation", 1024)
        if len(before) != len(after):
            raise ValueError("transition observation dimensions must match")
        object.__setattr__(self, "observation", before)
        object.__setattr__(self, "next_observation", after)
        object.__setattr__(self, "action", _vector(self.action, "action", 16))
        object.__setattr__(self, "reward", _number(self.reward, "reward"))


def _copy_spec(value: ContinuousActionSpec) -> ContinuousActionSpec:
    if type(value) is not ContinuousActionSpec:
        raise ValueError("spec must be an exact ContinuousActionSpec")
    return ContinuousActionSpec(
        value.observation_dim, value.action_units, value.lower_bounds, value.upper_bounds
    )


def _copy_transition(value: ContinuousTransition) -> ContinuousTransition:
    if type(value) is not ContinuousTransition:
        raise ValueError("transitions must be exact ContinuousTransition records")
    return ContinuousTransition(
        value.step,
        value.observation,
        value.action,
        value.reward,
        value.next_observation,
        value.terminated,
        value.truncated,
    )


@dataclass(frozen=True)
class ContinuousEpisode:
    """A complete trajectory; termination and time-limit flags remain distinct."""

    episode_id: str
    context: str
    role: str
    spec: ContinuousActionSpec
    transitions: tuple[ContinuousTransition, ...]

    def __post_init__(self) -> None:
        _text(self.episode_id, "episode_id", 64)
        _text(self.context, "context", 64)
        if type(self.role) is not str or self.role not in (
            "train",
            "inner_guard",
            "probe_train",
            "probe_test",
            "final_test",
        ):
            raise ValueError(
                "episode role must be train, inner_guard, probe_train, probe_test or final_test"
            )
        spec = _copy_spec(self.spec)
        if type(self.transitions) is not tuple or not 1 <= len(self.transitions) <= 256:
            raise ValueError("transitions must be an exact tuple with1..256 records")
        owned = tuple(_copy_transition(value) for value in self.transitions)
        for index, transition in enumerate(owned):
            if transition.step != index or len(transition.observation) != spec.observation_dim:
                raise ValueError(
                    "episode steps must start at0 with fixed declared observation shape"
                )
            if len(transition.action) != len(spec.action_units) or any(
                action < lo or action > hi
                for action, lo, hi in zip(
                    transition.action, spec.lower_bounds, spec.upper_bounds, strict=True
                )
            ):
                raise ValueError("action shape or value violates declared per-axis bounds")
            if (transition.terminated or transition.truncated) != (index == len(owned) - 1):
                raise ValueError("only the final transition must terminate or truncate")
            if index and owned[index - 1].next_observation != transition.observation:
                raise ValueError("episode observations must join exactly between transitions")
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "transitions", owned)


def copy_continuous_episode(value: ContinuousEpisode) -> ContinuousEpisode:
    """Revalidate and own every field; never trust a previously frozen instance."""
    if type(value) is not ContinuousEpisode:
        raise ValueError("episode must be an exact ContinuousEpisode")
    return ContinuousEpisode(
        value.episode_id, value.context, value.role, value.spec, value.transitions
    )


def continuous_training_pairs(
    episode: ContinuousEpisode, *, role: str = "train"
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    """Extract only the matching training role; no release or role conversion.

    Why: final/probe evaluation labels may be stored as demonstrations but cannot
    enter this training conversion. Simulator adapters must explicitly convert
    their native arrays/scalars before constructing these validated records.
    """
    expected = require_training_role(role)
    if type(episode) is not ContinuousEpisode or episode.role != expected:
        raise ValueError("continuous training requires an episode with the matching training role")
    owned = copy_continuous_episode(episode)
    return (
        tuple(item.observation for item in owned.transitions),
        tuple(item.action for item in owned.transitions),
    )
