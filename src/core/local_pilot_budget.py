"""Pure first-pilot resource preflight, independent of model objective.

Inputs are planned work and local/simulated execution context. Output is an
immutable accepted request or a useful refusal. This module performs no IO,
training, promotion, measurement, permission changes or runtime enforcement.
"""

from dataclasses import dataclass, fields
from math import isfinite

LOCAL_PILOT_BUDGET_ID = "local_small_head_pilot_budget_v1"


@dataclass(frozen=True)
class PilotResources:
    wall_seconds: float = 60.0
    training_updates: int = 512
    environment_steps: int = 1024
    cpu_threads: int = 1
    process_memory_bytes: int = 256 * 1024**2
    replay_bytes: int = 1024**2
    model_download_bytes: int = 0
    storage_bytes: int = 32 * 1024**2
    gpu_memory_bytes: int = 0

    def __post_init__(self) -> None:
        try:
            finite_seconds = type(self.wall_seconds) in {int, float} and isfinite(self.wall_seconds)
        except OverflowError:
            finite_seconds = False
        if not finite_seconds or self.wall_seconds <= 0:
            raise ValueError("pilot wall_seconds must be positive finite seconds")
        for descriptor in fields(self):
            if descriptor.name == "wall_seconds":
                continue
            value = getattr(self, descriptor.name)
            minimum = 1 if descriptor.name in {"cpu_threads", "process_memory_bytes"} else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f"pilot {descriptor.name} must be an integer >= {minimum}")


# Why this: controllers may request work, but this version's ceilings are fixed.
FIRST_LOCAL_PILOT_LIMITS = PilotResources()


@dataclass(frozen=True)
class PilotRequest:
    resources: PilotResources = FIRST_LOCAL_PILOT_LIMITS
    execution_target: str = "local"
    environment_kind: str = "simulation"
    device: str = "cpu"


class PilotBudgetExceeded(ValueError):
    def __init__(self, violations: tuple[str, ...]) -> None:
        self.violations = violations
        super().__init__("pilot resource limits exceeded: " + "; ".join(violations))


def validate_local_pilot_request(request: PilotRequest) -> PilotRequest:
    """Refuse unsupported context and all excess planned resources before IO."""
    if type(request) is not PilotRequest or type(request.resources) is not PilotResources:
        raise ValueError("pilot preflight requires a typed request and resources")
    request.resources.__post_init__()
    if (request.execution_target, request.environment_kind, request.device) != (
        "local",
        "simulation",
        "cpu",
    ):
        raise ValueError("first pilot requires local execution, simulation and CPU")
    violations = tuple(
        f"{descriptor.name}: requested {getattr(request.resources, descriptor.name)}, "
        f"limit {getattr(FIRST_LOCAL_PILOT_LIMITS, descriptor.name)}"
        for descriptor in fields(PilotResources)
        if getattr(request.resources, descriptor.name)
        > getattr(FIRST_LOCAL_PILOT_LIMITS, descriptor.name)
    )
    if violations:
        raise PilotBudgetExceeded(violations)
    return request
