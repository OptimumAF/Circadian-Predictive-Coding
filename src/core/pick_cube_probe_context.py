"""Fixed reserved PickCube probe split, declared before collecting either reset.

Outputs bind a role to its reset, shared task context and distinct episode ID.
No caller-selected allocation, collection, role conversion, release or IO occurs.
Why: two names for the same deterministic trajectory are not a held-out split;
23 retains its probe allocation and adjacent24 is the prospective test reset.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PickCubeProbeBinding:
    role: str
    seed: int
    context: str
    episode_id: str

    def __post_init__(self) -> None:
        seed, episode_id = _allocation(self.role)
        if (
            type(self.seed) is not int
            or self.seed != seed
            or type(self.context) is not str
            or self.context != "pick-cube-reserved-probe-23-24"
            or type(self.episode_id) is not str
            or self.episode_id != episode_id
        ):
            raise ValueError("reserved PickCube probe binding differs from the fixed split")


def _allocation(role: str) -> tuple[int, str]:
    if type(role) is not str:
        raise ValueError("reserved PickCube probe role must be probe_train or probe_test")
    if role == "probe_train":
        return 23, "pick-cube-probe-train-teacher-23"
    if role == "probe_test":
        return 24, "pick-cube-probe-test-teacher-24"
    raise ValueError("reserved PickCube probe role must be probe_train or probe_test")


def pick_cube_probe_binding(role: str) -> PickCubeProbeBinding:
    """Return fresh fixed metadata; never infer a role from a caller's seed."""
    seed, episode_id = _allocation(role)
    return PickCubeProbeBinding(role, seed, "pick-cube-reserved-probe-23-24", episode_id)
