"""Trusted local experience metadata, declared permissions and logical clocks.

Payloads remain opaque. These records do not attest physical source provenance,
release final-test data, score roles or authorize a scientific experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Generic, Literal, Protocol, TypeVar

from src.core.learner_ports import TrainingDiagnostic

Features = TypeVar("Features")
Targets = TypeVar("Targets")
ExperienceRole = Literal["train", "inner_guard", "outer_selection", "final_test"]
SampleKey = tuple[str, str]


def require_identifier(value: str, name: str) -> None:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty identifier without surrounding whitespace")


def require_tick(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer logical tick")


def _require_role(role: ExperienceRole) -> None:
    if type(role) is not str or role not in (
        "train",
        "inner_guard",
        "outer_selection",
        "final_test",
    ):
        raise ValueError(
            "experience role must name train, inner_guard, outer_selection or final_test"
        )


class EventClock(Protocol):
    def now(self) -> int: ...


class LogicalClock:
    """Explicit monotonic event ticks, independent of budget wall-clock seconds."""

    def __init__(self, initial: int = 0) -> None:
        require_tick(initial, "initial time")
        self._time = initial

    def now(self) -> int:
        return self._time

    def advance_to(self, value: int) -> None:
        require_tick(value, "event time")
        if value < self._time:
            raise ValueError("logical clock cannot move backwards")
        self._time = value


@dataclass(frozen=True)
class ExperiencePermissions:
    training: bool = False
    replay: bool = False
    evaluation: bool = False

    def __post_init__(self) -> None:
        if any(type(flag) is not bool for flag in (self.training, self.replay, self.evaluation)):
            raise ValueError("experience permissions must be exact boolean flags")


@dataclass(frozen=True)
class Experience(Generic[Features]):
    sample_id: str
    episode_id: str
    observed_at: int
    model_version: str
    features: Features
    role: ExperienceRole
    permissions: ExperiencePermissions
    candidate_ids: tuple[str, ...] = ()
    action_id: str | None = None
    reward: float | None = None

    def __post_init__(self) -> None:
        for name in ("sample_id", "episode_id", "model_version"):
            require_identifier(getattr(self, name), name)
        require_tick(self.observed_at, "observed_at")
        _require_role(self.role)
        if type(self.permissions) is not ExperiencePermissions:
            raise ValueError("experience requires typed permissions")
        if self.role != "train" and (self.permissions.training or self.permissions.replay):
            raise ValueError("held-out roles cannot grant training or replay permission")
        if type(self.candidate_ids) is not tuple:
            raise ValueError("candidate_ids must be an immutable tuple")
        for candidate in self.candidate_ids:
            require_identifier(candidate, "candidate_id")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("candidate_ids must be unique")
        if self.action_id is not None:
            require_identifier(self.action_id, "action_id")
            if self.action_id not in self.candidate_ids:
                raise ValueError("action_id must identify a declared candidate")
        if self.reward is not None and (
            type(self.reward) not in (int, float) or not isfinite(self.reward)
        ):
            raise ValueError("reward must be finite and numeric")

    @property
    def key(self) -> SampleKey:
        return self.episode_id, self.sample_id


@dataclass(frozen=True)
class LabelArrival(Generic[Targets]):
    event_id: str
    sample_id: str
    episode_id: str
    arrived_at: int
    model_version: str
    targets: Targets
    role: ExperienceRole = "train"

    def __post_init__(self) -> None:
        for name in ("event_id", "sample_id", "episode_id", "model_version"):
            require_identifier(getattr(self, name), name)
        require_tick(self.arrived_at, "arrived_at")
        _require_role(self.role)

    @property
    def key(self) -> SampleKey:
        return self.episode_id, self.sample_id


@dataclass(frozen=True)
class AppliedExperience:
    sample_id: str
    episode_id: str
    event_id: str
    actor_version: str
    learner_version: str
    observed_at: int
    arrived_at: int
    applied_at: int
    update_number: int
    diagnostic: TrainingDiagnostic
