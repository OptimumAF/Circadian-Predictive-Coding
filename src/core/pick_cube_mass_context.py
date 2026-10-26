"""One fixed TRAIN17 mass-only intervention; no simulator or learning authority.

Metadata requests twice the observed reset mass, with no inertia/COM setter.
It does not assert observed inertia invariance, density scaling or robustness.
"""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class PickCubeMassContext:
    context: str = "pick-cube-mass-x2-train-17"
    seed: int = 17
    mass_multiplier: float = 2.0
    semantics: str = "mass_only_no_inertia_scaling"


def pick_cube_mass_context() -> PickCubeMassContext:
    """Return fresh immutable metadata for the single declared intervention."""
    return PickCubeMassContext()


def validate_pick_cube_mass_context(value: PickCubeMassContext) -> PickCubeMassContext:
    expected = pick_cube_mass_context()
    if type(value) is not PickCubeMassContext or any(
        type(getattr(value, field.name)) is not type(getattr(expected, field.name))
        or getattr(value, field.name) != getattr(expected, field.name)
        for field in fields(expected)
    ):
        raise ValueError("expected the exact declared TRAIN17 mass-only context")
    return expected


def require_mass_binding(
    value: PickCubeMassContext | None, *, context: str, seed: int, role: str
) -> PickCubeMassContext | None:
    """Reject relabeling and reserved roles before admitting native creation."""
    if value is None:
        if context == pick_cube_mass_context().context:
            raise ValueError("mass context requires its explicit declared binding")
        return None
    expected = validate_pick_cube_mass_context(value)
    if (
        type(context) is not str
        or context != expected.context
        or type(seed) is not int
        or seed != expected.seed
        or type(role) is not str
        or role != "train"
    ):
        raise ValueError("mass fixture requires its declared TRAIN17 binding")
    return expected
