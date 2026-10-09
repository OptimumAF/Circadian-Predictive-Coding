"""Declared cooperative work/serving limits and detached admission observations.

No scheduler thread, native model, resource sampler, persistence or timing metric
lives here. Limits count requests/attempts; payload bytes are a separate bound.
"""

from dataclasses import dataclass
from typing import Literal

from src.core.experience import AppliedExperience, require_tick

DeferReason = Literal[
    "paused", "serving_active", "training_active", "work_quota", "resource_unavailable"
]


@dataclass(frozen=True)
class SharingLimits:
    max_serving_requests: int
    max_admitted_updates: int
    max_updates_per_poll: int

    def __post_init__(self) -> None:
        for name in ("max_serving_requests", "max_admitted_updates", "max_updates_per_poll"):
            require_tick(getattr(self, name), name)
        if self.max_serving_requests == 0 or self.max_updates_per_poll == 0:
            raise ValueError("serving capacity and updates per poll must be positive")


@dataclass(frozen=True)
class TrainingAdmission:
    allowed: bool
    reason: DeferReason | None


@dataclass(frozen=True)
class SharingSnapshot:
    limits: SharingLimits
    active_serving_requests: int
    training_active: bool
    paused: bool
    admitted_updates: int
    deferrals: tuple[tuple[DeferReason, int], ...]


@dataclass(frozen=True)
class TrainingPoll:
    updates: tuple[AppliedExperience, ...]
    deferred_reason: DeferReason | None
