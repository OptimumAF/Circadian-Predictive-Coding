"""Physical cube observations shared across explicitly qualified task ports.

Inputs are an observed grasp signal, world positions in metres and causal
executed-command history. Output is the declared 12D model view, without raw
flattened offsets, teacher state, fitted statistics, simulator access or IO.
"""

from dataclasses import dataclass
import math

from src.core.continuous_demonstration import ContinuousActionSpec, _vector
from src.core.pick_cube_action_history import ActionHistoryState, validate_action_history_state


CUBE_TASKSPACE_DEFINITION = "cube_public_world_positions_grasp_executed_history12_v1"
ACTION_UNITS = (
    "normalized_root_dx",
    "normalized_root_dy",
    "normalized_root_dz",
    "normalized_gripper",
)


def _position(value: tuple[float, ...]) -> tuple[float, ...]:
    owned = _vector(value, "cube world position in metres", 3)
    if len(owned) != 3:
        raise ValueError("cube world position requires exactly three coordinates")
    return owned


@dataclass(frozen=True)
class CubeTaskSpaceState:
    """Observed physical values; absence of grasp evidence has no default."""

    is_grasped: bool
    tcp_position: tuple[float, ...]
    object_position: tuple[float, ...]
    goal_position: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.is_grasped) is not bool:
            raise ValueError("cube grasp evidence must be an exact observed bool")
        for name in ("tcp_position", "object_position", "goal_position"):
            object.__setattr__(self, name, _position(getattr(self, name)))


def copy_cube_taskspace_state(value: CubeTaskSpaceState) -> CubeTaskSpaceState:
    """Revalidate a frozen record before deriving model input."""
    if type(value) is not CubeTaskSpaceState:
        raise ValueError("expected an exact CubeTaskSpaceState")
    return CubeTaskSpaceState(
        value.is_grasped, value.tcp_position, value.object_position, value.goal_position
    )


def cube_taskspace_observation(
    state: CubeTaskSpaceState, history: ActionHistoryState | None = None
) -> tuple[float, ...]:
    """Return grasp, world displacements/.1 and actual command history.

    Why: compute named physical differences rather than reinterpret another
    task's flattened observation. Subtraction uses Python floats; it need not be
    bitwise equal to differences previously rounded by the native float32 SDK.
    A qualified port must establish world/root axis alignment independently.
    """
    owned = copy_cube_taskspace_state(state)
    prior = validate_action_history_state(ActionHistoryState() if history is None else history)
    differences = tuple(
        (end - start) / 0.1
        for starts, ends in (
            (owned.tcp_position, owned.object_position),
            (owned.object_position, owned.goal_position),
        )
        for start, end in zip(starts, ends, strict=True)
    )
    if not all(math.isfinite(value) for value in differences):
        raise ValueError("cube task-space displacement is not finite")
    return (
        float(owned.is_grasped),
        *differences,
        *prior.previous_action,
        prior.closing_count / 6,
    )


def cube_taskspace_spec() -> ContinuousActionSpec:
    """Preserve the existing physical model/action contract; no policy fitting."""
    return ContinuousActionSpec(12, ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
