"""Fixed official-state task-space prefix for one admitted Panda setup.

Selects the official grasp flag and two WORLD displacement vectors, scaled by
the declared .1 metre unit. The existing action-history module adds causal
executed commands. No teacher phase, target, clock, fit, rotation inference,
contact oracle, learner, simulator or IO occurs here. World/root vector axes
align only in the named default Panda table/controller setup.
"""

from dataclasses import asdict, dataclass
import math

from src.core.continuous_demonstration import _vector

DEFINITION = "pick_cube_official_state42_taskspace7_executed_history5_v1"


@dataclass(frozen=True)
class PickCubeTaskSpaceHistoryRecipe:
    mode: str = "executed"
    canonical_quaternions: bool = False
    definition: str = DEFINITION
    source_dim: int = 42
    feature_dim: int = 12
    closing_limit: int = 6
    displacement_scale: float = 0.1
    displacement_frame: str = "world"
    setup: str = "panda_table_pd_ee_delta_pos"

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in ("executed", "zero"):
            raise ValueError("taskspace history mode must be executed or zero")
        if type(self.canonical_quaternions) is not bool or self.canonical_quaternions:
            raise ValueError("taskspace history does not canonicalize unused quaternions")
        if (
            type(self.definition) is not str
            or self.definition != DEFINITION
            or type(self.displacement_scale) is not float
            or self.displacement_scale != 0.1
            or type(self.displacement_frame) is not str
            or self.displacement_frame != "world"
            or type(self.setup) is not str
            or self.setup != "panda_table_pd_ee_delta_pos"
            or any(
                type(value) is not int or value != expected
                for value, expected in (
                    (self.source_dim, 42),
                    (self.feature_dim, 12),
                    (self.closing_limit, 6),
                )
            )
        ):
            raise ValueError("fixed taskspace history recipe metadata changed")


def validate_taskspace_history_recipe(
    value: PickCubeTaskSpaceHistoryRecipe,
) -> PickCubeTaskSpaceHistoryRecipe:
    if type(value) is not PickCubeTaskSpaceHistoryRecipe:
        raise ValueError("expected an exact PickCubeTaskSpaceHistoryRecipe")
    return PickCubeTaskSpaceHistoryRecipe(
        value.mode,
        value.canonical_quaternions,
        value.definition,
        value.source_dim,
        value.feature_dim,
        value.closing_limit,
        value.displacement_scale,
        value.displacement_frame,
        value.setup,
    )


def taskspace_metadata(recipe: PickCubeTaskSpaceHistoryRecipe) -> dict:
    """Return detached fixed recipe metadata; no feature or fitted statistics."""
    return asdict(validate_taskspace_history_recipe(recipe))


def taskspace_observation(
    raw42: tuple[float, ...],
    recipe: PickCubeTaskSpaceHistoryRecipe | None = None,
) -> tuple[float, ...]:
    """Return (is_grasped, object−TCP/.1, goal−object/.1), all world axes."""
    config = validate_taskspace_history_recipe(
        PickCubeTaskSpaceHistoryRecipe() if recipe is None else recipe
    )
    if type(raw42) is not tuple or len(raw42) != config.source_dim:
        raise ValueError("taskspace history requires a complete raw42 observation tuple")
    raw = _vector(raw42, "official taskspace raw42", config.source_dim)
    if raw[18] not in (0.0, 1.0):
        raise ValueError("official taskspace is_grasped must be exactly zero or one")
    displacement = tuple(value / config.displacement_scale for value in raw[36:42])
    if not all(math.isfinite(value) for value in displacement):
        raise ValueError("taskspace physical scaling produced a nonfinite displacement")
    return (raw[18],) + displacement
