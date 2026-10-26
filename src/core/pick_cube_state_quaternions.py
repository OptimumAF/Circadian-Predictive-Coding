"""Owned sign-canonical views of official PickCube state42 observations.

Validate complete finite inputs and both unit quaternions before selecting a
deterministic representative of q/-q. Other coordinates and all trajectory
metadata stay unchanged. No normalization fit, clipping, IO, learner, simulator
or global continuity guarantee is supplied.
"""

from dataclasses import dataclass, fields
import math

from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
    _copy_spec,
    _vector,
    copy_continuous_episode,
)
from src.core.pick_cube_train_context import pick_cube_training_context


@dataclass(frozen=True)
class PickCubeStateQuaternionProjection:
    definition: str = "pick_cube_state42_largest_component_quaternion_sign_v1"
    source_dim: int = 42
    feature_dim: int = 42
    quaternion_starts: tuple[int, ...] = (22, 32)
    quaternion_width: int = 4
    sign_rule: str = "first_largest_abs_positive"
    unit_tolerance: float = 1e-4


def state_quaternion_projection() -> PickCubeStateQuaternionProjection:
    return PickCubeStateQuaternionProjection()


def validate_state_quaternion_projection(
    value: PickCubeStateQuaternionProjection,
) -> PickCubeStateQuaternionProjection:
    expected = state_quaternion_projection()
    if type(value) is not PickCubeStateQuaternionProjection:
        raise ValueError("expected the exact state42 quaternion projection")
    for field in fields(expected):
        actual, target = getattr(value, field.name), getattr(expected, field.name)
        if (
            type(actual) is not type(target)
            or (
                type(target) is tuple
                and (
                    len(actual) != len(target)
                    or any(type(a) is not type(b) for a, b in zip(actual, target, strict=True))
                )
            )
            or actual != target
        ):
            raise ValueError("declared state42 quaternion projection changed")
    return expected


def state_quaternion_spec(
    spec: ContinuousActionSpec, projection: PickCubeStateQuaternionProjection
) -> ContinuousActionSpec:
    config = validate_state_quaternion_projection(projection)
    owned = _copy_spec(spec)
    if owned.observation_dim != config.source_dim:
        raise ValueError("quaternion projection requires the complete raw42D state spec")
    return owned


def state_quaternion_observation(
    observation: tuple[float, ...], projection: PickCubeStateQuaternionProjection
) -> tuple[float, ...]:
    config = validate_state_quaternion_projection(projection)
    if type(observation) is not tuple or len(observation) != config.source_dim:
        raise ValueError("quaternion projection requires a complete raw42D state tuple")
    raw = _vector(observation, "raw state42 observation", config.source_dim)
    quaternions = tuple(
        raw[start : start + config.quaternion_width] for start in config.quaternion_starts
    )
    if any(
        abs(math.hypot(*quaternion) - 1.0) > config.unit_tolerance for quaternion in quaternions
    ):
        raise ValueError("state42 quaternion must be unit length within1e-4; no renormalization")
    result = list(raw)
    for start, quaternion in zip(config.quaternion_starts, quaternions, strict=True):
        # max returns the first equal maximum, making ties reproducible. Only
        # quaternion zeros are canonicalized; other state coordinates stay raw.
        dominant = max(range(config.quaternion_width), key=lambda index: abs(quaternion[index]))
        sign = -1.0 if quaternion[dominant] < 0.0 else 1.0
        result[start : start + config.quaternion_width] = tuple(
            0.0 if value == 0.0 else sign * value for value in quaternion
        )
    return tuple(result)


def state_quaternion_training_episode(
    episode: ContinuousEpisode, projection: PickCubeStateQuaternionProjection
) -> ContinuousEpisode:
    """Own one complete known TRAIN episode without changing its raw source."""
    config = validate_state_quaternion_projection(projection)
    owned = copy_continuous_episode(episode)
    if (
        owned.role != "train"
        or owned.context not in (pick_cube_training_context(17), pick_cube_training_context(19))
        or len(owned.transitions) > 50
    ):
        raise ValueError(
            "quaternion training projection requires a known complete TRAIN17/19 episode"
        )
    spec = state_quaternion_spec(owned.spec, config)
    transitions = tuple(
        ContinuousTransition(
            row.step,
            state_quaternion_observation(row.observation, config),
            row.action,
            row.reward,
            state_quaternion_observation(row.next_observation, config),
            row.terminated,
            row.truncated,
        )
        for row in owned.transitions
    )
    return ContinuousEpisode(owned.episode_id, owned.context, owned.role, spec, transitions)
