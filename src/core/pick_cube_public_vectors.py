"""Owned public21 views of known PickCube RGB records, without learning or IO.

Validate the full raw input before dropping RGB. Raw records, controls, labels,
roles and rendering costs stay intact. No teacher qualification, fitting, RNG,
geometry, clock feature, simulator, promotion or evaluation authority is added.
"""

from dataclasses import dataclass, fields

from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
    _copy_spec,
    _vector,
    copy_continuous_episode,
)
from src.core.pick_cube_camera_context import (
    PickCubeCameraContext,
    require_camera_binding,
    validate_camera_recipe,
    validate_pick_cube_camera_context,
)
from src.core.pick_cube_rgb_features import PickCubeRGBRecipe
from src.core.pick_cube_train_context import pick_cube_training_context


@dataclass(frozen=True)
class PickCubePublicVectorProjection:
    definition: str = "pick_cube_public_qpos9_qvel9_goal3_v1"
    source_dim: int = 789
    feature_dim: int = 21
    feature_order: tuple[str, ...] = ("qpos9", "qvel9", "public_goal3")


def public_vector_projection() -> PickCubePublicVectorProjection:
    return PickCubePublicVectorProjection()


def validate_public_vector_projection(
    value: PickCubePublicVectorProjection,
) -> PickCubePublicVectorProjection:
    expected = public_vector_projection()
    if type(value) is not PickCubePublicVectorProjection:
        raise ValueError("expected the exact public-vector21 projection")
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
            raise ValueError("declared public-vector21 projection changed")
    return expected


def public_vector_spec(
    spec: ContinuousActionSpec, projection: PickCubePublicVectorProjection
) -> ContinuousActionSpec:
    config = validate_public_vector_projection(projection)
    owned = _copy_spec(spec)
    if owned.observation_dim != config.source_dim:
        raise ValueError("public-vector21 projection requires the raw789D spec")
    return ContinuousActionSpec(
        config.feature_dim, owned.action_units, owned.lower_bounds, owned.upper_bounds
    )


def public_vector_observation(
    observation: tuple[float, ...],
    projection: PickCubePublicVectorProjection,
    *,
    source_recipe: PickCubeRGBRecipe,
    camera_context: PickCubeCameraContext | None = None,
) -> tuple[float, ...]:
    config = validate_public_vector_projection(projection)
    validate_camera_recipe(source_recipe, camera_context)
    if type(observation) is not tuple or len(observation) != config.source_dim:
        raise ValueError("public-vector21 projection requires a complete raw789D tuple")
    raw = _vector(observation, "raw RGB observation", config.source_dim)
    # Why: excluding pixels from distance computation must not hide invalid or
    # corrupt source evidence; it is an explicit view of a valid raw record.
    if any(not 0.0 <= pixel <= 1.0 for pixel in raw[config.feature_dim :]):
        raise ValueError("source RGB suffix must remain within [0,1]")
    return raw[: config.feature_dim]


def public_vector_training_episode(
    episode: ContinuousEpisode,
    projection: PickCubePublicVectorProjection,
    *,
    source_recipe: PickCubeRGBRecipe,
    camera_context: PickCubeCameraContext | None = None,
) -> ContinuousEpisode:
    """Derive an owned TRAIN view; callers keep explicit recipe metadata with it."""
    if type(episode) is not ContinuousEpisode or episode.role != "train":
        raise ValueError("public-vector21 training projection requires a complete TRAIN episode")
    config = validate_public_vector_projection(projection)
    validate_camera_recipe(source_recipe, camera_context)
    if camera_context is None:
        if episode.context not in (pick_cube_training_context(17), pick_cube_training_context(19)):
            raise ValueError("public-vector21 source requires a known TRAIN context")
    else:
        camera = validate_pick_cube_camera_context(camera_context)
        require_camera_binding(
            camera,
            context=episode.context,
            seed=camera.seed,
            role=episode.role,
            visual_observation=True,
        )
    owned = copy_continuous_episode(episode)
    if len(owned.transitions) > 50:
        raise ValueError("public-vector21 source exceeds the official50-step horizon")
    spec = public_vector_spec(owned.spec, config)
    transitions = tuple(
        ContinuousTransition(
            row.step,
            public_vector_observation(
                row.observation, config, source_recipe=source_recipe, camera_context=camera_context
            ),
            row.action,
            row.reward,
            public_vector_observation(
                row.next_observation,
                config,
                source_recipe=source_recipe,
                camera_context=camera_context,
            ),
            row.terminated,
            row.truncated,
        )
        for row in owned.transitions
    )
    return ContinuousEpisode(owned.episode_id, owned.context, owned.role, spec, transitions)
