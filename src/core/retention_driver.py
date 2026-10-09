"""Typed bounded retention scheduling policy and payload-free observations.

No clocks, threads, locks, payload references, IO or deletion lives here.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Literal
from src.core.data_retention import DataCleanupReport
from src.core.experience import require_tick

RetentionDriverState = Literal["ready", "running", "stopping", "stopped", "exhausted", "failed"]


def require_seconds(value, name: str) -> None:
    if type(value) not in (int, float) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} requires positive finite seconds")


@dataclass(frozen=True)
class RetentionDriverLimits:
    poll_interval_seconds: float
    max_polls: int
    max_run_seconds: float
    join_timeout_seconds: float = 1.0

    def __post_init__(self) -> None:
        for value in (self.poll_interval_seconds, self.max_run_seconds, self.join_timeout_seconds):
            require_seconds(value, "driver timing")
        require_tick(self.max_polls, "max_polls")
        if self.max_polls == 0:
            raise ValueError("driver poll budget must be positive")


@dataclass(frozen=True)
class RetentionPollResult:
    outcome: Literal["idle", "purged", "busy", "failed"]
    report: DataCleanupReport | None = None

    def __post_init__(self) -> None:
        if self.outcome not in ("idle", "purged", "busy", "failed"):
            raise ValueError("unsupported retention outcome")
        if (self.outcome == "purged") != (type(self.report) is DataCleanupReport):
            raise ValueError("purged outcome requires cleanup report")


@dataclass(frozen=True)
class RetentionDriverSnapshot:
    limits: RetentionDriverLimits
    state: RetentionDriverState
    polls: int
    purges: int
    cleanup_attempts: int
    alive: bool
    pending_cleanup: bool
    error_type: str | None

    def __post_init__(self) -> None:
        if type(self.limits) is not RetentionDriverLimits or self.state not in (
            "ready",
            "running",
            "stopping",
            "stopped",
            "exhausted",
            "failed",
        ):
            raise ValueError("unsupported retention driver snapshot")
        require_tick(self.polls, "polls")
        require_tick(self.purges, "purges")
        require_tick(self.cleanup_attempts, "cleanup_attempts")
        if (
            self.polls > self.limits.max_polls
            or type(self.alive) is not bool
            or type(self.pending_cleanup) is not bool
        ):
            raise ValueError("invalid driver counters/state")
        if self.cleanup_attempts > self.limits.max_polls + 2:
            raise ValueError("cleanup attempts exceed original driver allowance")
        if self.error_type is not None and type(self.error_type) is not str:
            raise ValueError("invalid driver error type")
