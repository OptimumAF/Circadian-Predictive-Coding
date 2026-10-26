"""Trusted local experience metadata, declared permissions and logical clocks.

Payloads remain opaque. These records do not attest physical source provenance,
release final-test data, score roles or authorize a scientific experiment.
Constructors and explicit validate_* helpers share the same metadata rules;
helpers neither copy payloads nor normalize or mutate their records.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Generic, Literal, Protocol, TypeVar

from src.core.learner_ports import TrainingDiagnostic

Features = TypeVar("Features")
Targets = TypeVar("Targets")
TrainingRole = Literal["train", "probe_train"]
ExperienceRole = Literal["train", "probe_train", "inner_guard", "outer_selection", "final_test"]
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
        "probe_train",
        "inner_guard",
        "outer_selection",
        "final_test",
    ):
        raise ValueError(
            "experience role must name train, probe_train, inner_guard, outer_selection or final_test"
        )


def require_training_role(role: str) -> TrainingRole:
    if type(role) is not str or role not in ("train", "probe_train"):
        raise ValueError("training role must be exactly train or probe_train")
    return "train" if role == "train" else "probe_train"


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
        validate_experience_permissions(self)


def validate_experience_permissions(value: ExperiencePermissions) -> None:
    """Validate declared flags without copying or changing their owner."""
    if any(type(flag) is not bool for flag in (value.training, value.replay, value.evaluation)):
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
        validate_experience(self)

    @property
    def key(self) -> SampleKey:
        return self.episode_id, self.sample_id


def validate_experience(value: Experience[Features]) -> None:
    """Validate metadata only; do not read opaque features or normalize fields."""
    for name in ("sample_id", "episode_id", "model_version"):
        require_identifier(getattr(value, name), name)
    require_tick(value.observed_at, "observed_at")
    _require_role(value.role)
    if type(value.permissions) is not ExperiencePermissions:
        raise ValueError("experience requires typed permissions")
    if value.role == "probe_train" and (value.permissions.replay or value.permissions.evaluation):
        raise ValueError("probe training cannot grant replay or evaluation permission")
    if value.role not in ("train", "probe_train") and (
        value.permissions.training or value.permissions.replay
    ):
        raise ValueError("held-out roles cannot grant training or replay permission")
    if type(value.candidate_ids) is not tuple:
        raise ValueError("candidate_ids must be an immutable tuple")
    for candidate in value.candidate_ids:
        require_identifier(candidate, "candidate_id")
    if len(value.candidate_ids) != len(set(value.candidate_ids)):
        raise ValueError("candidate_ids must be unique")
    if value.action_id is not None:
        require_identifier(value.action_id, "action_id")
        if value.action_id not in value.candidate_ids:
            raise ValueError("action_id must identify a declared candidate")
    if value.reward is not None and (
        type(value.reward) not in (int, float) or not isfinite(value.reward)
    ):
        raise ValueError("reward must be finite and numeric")


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
        validate_label_arrival(self)

    @property
    def key(self) -> SampleKey:
        return self.episode_id, self.sample_id


def validate_label_arrival(value: LabelArrival[Targets]) -> None:
    """Validate label metadata without copying or inspecting opaque targets."""
    for name in ("event_id", "sample_id", "episode_id", "model_version"):
        require_identifier(getattr(value, name), name)
    require_tick(value.arrived_at, "arrived_at")
    _require_role(value.role)


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


@dataclass(frozen=True)
class AcknowledgedExperience:
    """Correct arrived TRAIN consumed without an optimizer update or erasure."""

    sample_id: str
    episode_id: str
    event_id: str
    actor_version: str
    learner_version: str
    observed_at: int
    arrived_at: int
    acknowledged_at: int
    wake_updates: int
    candidate_revision: int

    def __post_init__(self) -> None:
        for name in ("sample_id", "episode_id", "event_id", "actor_version", "learner_version"):
            require_identifier(getattr(self, name), name)
        for name in (
            "observed_at",
            "arrived_at",
            "acknowledged_at",
            "wake_updates",
            "candidate_revision",
        ):
            require_tick(getattr(self, name), name)
        if not self.observed_at <= self.arrived_at <= self.acknowledged_at:
            raise ValueError("acknowledgement requires causal observation and label arrival")
