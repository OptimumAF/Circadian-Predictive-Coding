"""Pure request timing validation, declared quantiles and native overlap populations.

Inputs are complete monotonic timing records. Outputs retain all samples and
report unavailable empty populations explicitly. No clock/thread/native model/IO
or favorable sample selection belongs here.
"""

from dataclasses import dataclass
from math import ceil
from typing import Literal

from src.core.experience import require_identifier, require_tick


@dataclass(frozen=True)
class ServingRequestTiming:
    phase: Literal["idle", "shared"]
    block: int | None
    index: int
    start_ns: int
    end_ns: int
    actor_version: str
    completed: bool

    def __post_init__(self):
        if self.phase not in ("idle", "shared"):
            raise ValueError("request phase must be idle or shared")
        if self.phase == "idle" and self.block is not None:
            raise ValueError("idle requests have no training block")
        if self.phase == "shared":
            if type(self.block) is not int:
                raise ValueError("shared requests require an integer training block")
            require_tick(self.block, "request block")
        for value in (self.index, self.start_ns, self.end_ns):
            require_tick(value, "request timing")
        require_identifier(self.actor_version, "request actor_version")
        if self.end_ns < self.start_ns or type(self.completed) is not bool:
            raise ValueError("request timing/status is invalid")


@dataclass(frozen=True)
class NativeCallTiming:
    block: int
    start_ns: int
    end_ns: int
    completed: bool

    def __post_init__(self):
        for value in (self.block, self.start_ns, self.end_ns):
            require_tick(value, "native timing")
        if self.end_ns < self.start_ns or type(self.completed) is not bool:
            raise ValueError("native timing/status is invalid")


@dataclass(frozen=True)
class LatencySummary:
    count: int
    p50_ns: int | None
    p95_ns: int | None
    max_ns: int | None


def summarize_latencies(requests: tuple[ServingRequestTiming, ...]) -> LatencySummary:
    if any(type(r) is not ServingRequestTiming or not r.completed for r in requests):
        raise ValueError("latency summaries require completed exact request records")
    if not requests:
        return LatencySummary(0, None, None, None)
    values = sorted(r.end_ns - r.start_ns for r in requests)
    return LatencySummary(
        len(values),
        values[ceil(0.5 * len(values)) - 1],
        values[ceil(0.95 * len(values)) - 1],
        values[-1],
    )


def select_native_overlap(
    requests: tuple[ServingRequestTiming, ...],
    calls: tuple[NativeCallTiming, ...],
    *,
    fully_contained: bool = False,
) -> tuple[ServingRequestTiming, ...]:
    if type(fully_contained) is not bool:
        raise ValueError("overlap population requires an exact containment boolean")
    selected = []
    for request in requests:
        for call in calls:
            overlaps = (
                call.completed
                and request.completed
                and request.phase == "shared"
                and request.block == call.block
                and request.start_ns < call.end_ns
                and request.end_ns > call.start_ns
            )
            contained = request.start_ns >= call.start_ns and request.end_ns <= call.end_ns
            if overlaps and (not fully_contained or contained):
                selected.append(request)
                break
    return tuple(selected)
