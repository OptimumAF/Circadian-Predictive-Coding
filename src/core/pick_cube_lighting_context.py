"""One fixed visual TRAIN17 ambient-light intervention, without simulator IO.

Inputs/outputs are owned immutable declarations. Camera/feature transforms and
directional lights are not changed by this declaration; no learning claim follows.
"""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class PickCubeLightingContext:
    context: str = "pick-cube-ambient-half-train-17"
    seed: int = 17
    ambient_multiplier: float = 0.5
    semantics: str = "ambient_only_directional_lights_unchanged_by_this_port"


def pick_cube_lighting_context() -> PickCubeLightingContext:
    """Return a fresh declaration for the single supported lighting fixture."""
    return PickCubeLightingContext()


def validate_pick_cube_lighting_context(value: PickCubeLightingContext) -> PickCubeLightingContext:
    expected = pick_cube_lighting_context()
    if type(value) is not PickCubeLightingContext or any(
        type(getattr(value, field.name)) is not type(getattr(expected, field.name))
        or getattr(value, field.name) != getattr(expected, field.name)
        for field in fields(expected)
    ):
        raise ValueError("expected the exact declared visual TRAIN17 lighting context")
    return expected


def require_lighting_binding(
    value: PickCubeLightingContext | None,
    *,
    context: str,
    seed: int,
    role: str,
    visual_observation: bool,
) -> PickCubeLightingContext | None:
    """Refuse reserved-context relabeling or heldout/nonvisual IO before creation."""
    if value is None:
        if context == pick_cube_lighting_context().context:
            raise ValueError("lighting context requires its explicit declared binding")
        return None
    expected = validate_pick_cube_lighting_context(value)
    if (
        type(context) is not str
        or context != expected.context
        or type(seed) is not int
        or seed != expected.seed
        or type(role) is not str
        or role != "train"
        or visual_observation is not True
    ):
        raise ValueError("lighting fixture requires its declared visual TRAIN17 binding")
    return expected
