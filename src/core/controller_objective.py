"""Prospective outer utility and controller-stream metadata; no data access.

Wall execution seconds and sampled RSS are explicit cost proxies, not FLOPs or
complete memory. Unknown work has no score. Role admission belongs to the app
before it reads these metrics; records do not release evaluation data.
"""

from dataclasses import dataclass
from math import fsum, isfinite

from src.core.experience import require_identifier, require_tick
from src.core.reference_controller import ReferencePolicy

CONTROLLER_ROLES = ("controller_train", "controller_validation", "sealed_evaluation")


def _number(value, name, *, positive=False):
    try:
        valid = (
            type(value) in (int, float)
            and isfinite(value)
            and (value > 0 if positive else value >= 0)
        )
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


def _count(value, name, *, positive=False):
    if type(value) is not int or not (1 if positive else 0) <= value <= 2**53 - 1:
        raise ValueError(
            f"{name} must be a bounded {'positive' if positive else 'nonnegative'} integer"
        )


@dataclass(frozen=True)
class ControllerObjective:
    objective_id: str = "controller_outer_utility_v1"
    new_gain_weight: float = 1.0
    old_drop_weight: float = 1.0
    compute_weight: float = 1.0
    memory_weight: float = 1.0
    violation_weight: float = 1.0
    compute_reference_seconds: float = 120.0
    memory_reference_bytes: int = 512 * 1024**2
    violation_reference_count: int = 1

    def __post_init__(self):
        require_identifier(self.objective_id, "objective_id")
        weights = (
            self.new_gain_weight,
            self.old_drop_weight,
            self.compute_weight,
            self.memory_weight,
            self.violation_weight,
        )
        for w in weights:
            _number(w, "objective weight")
        if not any(w > 0 for w in weights):
            raise ValueError("at least one objective weight must be positive")
        _number(self.compute_reference_seconds, "compute reference", positive=True)
        _count(self.memory_reference_bytes, "memory reference", positive=True)
        _count(self.violation_reference_count, "violation reference", positive=True)


@dataclass(frozen=True)
class ControllerMeasures:
    new_before: int
    new_after: int
    new_count: int
    old_before: int
    old_after: int
    old_count: int
    execution_seconds: float | None
    peak_rss_bytes: int | None
    violations: int | None
    work_complete: bool


@dataclass(frozen=True)
class ControllerUtility:
    objective_id: str
    normalized_new_gain: float
    normalized_old_drop: float
    normalized_compute: float
    normalized_memory: float
    normalized_violations: float
    value: float


def controller_utility(
    objective: ControllerObjective, measures: ControllerMeasures
) -> ControllerUtility:
    if type(objective) is not ControllerObjective or type(measures) is not ControllerMeasures:
        raise ValueError("utility requires exact prospective objective and measures")
    objective.__post_init__()
    if type(measures.work_complete) is not bool or not measures.work_complete:
        raise ValueError("incomplete or unknown work has no outer utility")
    for group in ("new", "old"):
        total = getattr(measures, group + "_count")
        _count(total, group + " count", positive=True)
        for when in ("before", "after"):
            v = getattr(measures, group + "_" + when)
            _count(v, group + " " + when)
            if v > total:
                raise ValueError("correct counts exceed declared task count")
    _number(measures.execution_seconds, "trial execution seconds")
    _count(measures.peak_rss_bytes, "sampled peak RSS", positive=True)
    _count(measures.violations, "violations")
    assert measures.execution_seconds is not None and measures.peak_rss_bytes is not None
    assert measures.violations is not None
    gain = (measures.new_after - measures.new_before) / measures.new_count
    drop = max(0.0, (measures.old_before - measures.old_after) / measures.old_count)
    compute = measures.execution_seconds / objective.compute_reference_seconds
    memory = measures.peak_rss_bytes / objective.memory_reference_bytes
    violations = measures.violations / objective.violation_reference_count
    terms = (
        objective.new_gain_weight * gain,
        -objective.old_drop_weight * drop,
        -objective.compute_weight * compute,
        -objective.memory_weight * memory,
        -objective.violation_weight * violations,
    )
    if not all(isfinite(v) for v in (*terms, compute, memory, violations)):
        raise ValueError("normalized utility is nonfinite")
    try:
        value = fsum(terms)
    except OverflowError as error:
        raise ValueError("outer utility overflow") from error
    if not isfinite(value):
        raise ValueError("outer utility overflow")
    return ControllerUtility(objective.objective_id, gain, drop, compute, memory, violations, value)


@dataclass(frozen=True)
class ControllerStream:
    stream_id: str
    role: str
    source_group: str

    def __post_init__(self):
        if type(self.role) is not str or self.role not in CONTROLLER_ROLES:
            raise ValueError("unknown controller stream role")
        require_identifier(self.stream_id, "stream_id")
        require_identifier(self.source_group, "source_group")


@dataclass(frozen=True)
class ControllerProtocol:
    protocol_id: str
    objective: ControllerObjective
    policy: ReferencePolicy
    streams: tuple[ControllerStream, ...]
    declared_at: int = 0
    clock_scope: str = "controller_outer_trial_events_v1"

    def __post_init__(self):
        require_identifier(self.protocol_id, "protocol_id")
        require_identifier(self.clock_scope, "clock_scope")
        require_tick(self.declared_at, "declared_at")
        if (
            type(self.objective) is not ControllerObjective
            or type(self.policy) is not ReferencePolicy
        ):
            raise ValueError("protocol requires exact objective and frozen reference policy")
        self.objective.__post_init__()
        self.policy.__post_init__()
        if (
            type(self.streams) is not tuple
            or len(self.streams) != 3
            or any(type(s) is not ControllerStream for s in self.streams)
        ):
            raise ValueError("protocol requires three distinct controller streams")
        for stream in self.streams:
            stream.__post_init__()
        if (
            tuple(s.role for s in self.streams) != CONTROLLER_ROLES
            or len({s.stream_id for s in self.streams}) != 3
            or len({s.source_group for s in self.streams}) != 3
        ):
            raise ValueError(
                "controller streams need distinct IDs/groups and train/validation/sealed order"
            )


@dataclass(frozen=True)
class ControllerTrialMetadata:
    stream: ControllerStream
    policy: ReferencePolicy
    started_at: int
    labels_arrived_at: int
    completed_at: int


@dataclass(frozen=True)
class ControllerTrial:
    metadata: ControllerTrialMetadata
    measures: ControllerMeasures


@dataclass(frozen=True)
class ScoredControllerTrial:
    metadata: ControllerTrialMetadata
    utility: ControllerUtility


@dataclass(frozen=True, eq=False)
class ControllerFreeze:
    protocol: ControllerProtocol
    frozen_at: int
