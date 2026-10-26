"""One declared TRAIN camera displacement; no simulator, pixels or selection.

The physical reset stays17 and only the known camera eye moves5cm laterally.
Default RGB validation remains strict. This fixture does not qualify learning,
rendered effects, visual necessity, or general domain-drift robustness.
"""

from dataclasses import dataclass, fields, replace

from src.core.pick_cube_rgb_features import (
    PickCubeRGBRecipe,
    pick_cube_rgb_recipe,
    validate_pick_cube_rgb_recipe,
)


@dataclass(frozen=True)
class PickCubeCameraContext:
    context: str = "pick-cube-camera-y005-train-17"
    seed: int = 17


def pick_cube_camera_context() -> PickCubeCameraContext:
    """Return fresh immutable metadata for the sole supported camera fixture."""
    return PickCubeCameraContext()


def validate_pick_cube_camera_context(value: PickCubeCameraContext) -> PickCubeCameraContext:
    expected = pick_cube_camera_context()
    if (
        type(value) is not PickCubeCameraContext
        or type(value.context) is not str
        or type(value.seed) is not int
        or value != expected
    ):
        raise ValueError("expected the exact declared TRAIN17 camera context")
    return expected


def camera_context_recipe(value: PickCubeCameraContext) -> PickCubeRGBRecipe:
    validate_pick_cube_camera_context(value)
    # Why: a distinct known recipe preserves binding of all existing records;
    # accepting arbitrary camera metadata would silently change their meaning.
    return replace(
        pick_cube_rgb_recipe(),
        recipe_id="pick_cube_rgb_pool16_qpos_qvel_goal_camera_y005_v1",
        camera_eye=(0.3, 0.05, 0.6),
    )


def validate_camera_recipe(
    value: PickCubeRGBRecipe, camera_context: PickCubeCameraContext | None = None
) -> PickCubeRGBRecipe:
    if camera_context is None:
        return validate_pick_cube_rgb_recipe(value)
    expected = camera_context_recipe(camera_context)
    if type(value) is not PickCubeRGBRecipe:
        raise ValueError("expected the exact declared camera recipe")
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
            raise ValueError("declared camera recipe changed")
    return expected


def require_camera_binding(
    value: PickCubeCameraContext | None,
    *,
    context: str,
    seed: int,
    role: str,
    visual_observation: bool,
) -> PickCubeCameraContext | None:
    """Refuse a relabeled, held-out or nonvisual camera context before native IO."""
    if value is None:
        if context == pick_cube_camera_context().context:
            raise ValueError("camera context requires its explicit declared binding")
        return None
    expected = validate_pick_cube_camera_context(value)
    if (
        type(context) is not str
        or context != expected.context
        or type(seed) is not int
        or seed != expected.seed
        or type(role) is not str
        or role != "train"
        or visual_observation is not True
    ):
        raise ValueError("camera fixture requires its declared visual TRAIN17 binding")
    return expected
