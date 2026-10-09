"""Declared local promotion criteria and guard metadata, with no model or IO.

Evidence is measured by the app; decisions do not authorize a serving swap or
release final labels. Native training diagnostics remain separate from utility.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Generic, TypeVar

from src.core.experience import ExperienceRole, SampleKey, require_identifier, require_tick

Features = TypeVar("Features")
Targets = TypeVar("Targets")


def require_sample_keys(keys: tuple[SampleKey, ...], *, allow_empty: bool = False) -> None:
    if type(keys) is not tuple or (not keys and not allow_empty):
        raise ValueError("sample_keys must be a nonempty immutable tuple")
    for key in keys:
        if type(key) is not tuple or len(key) != 2:
            raise ValueError("sample identity requires episode and sample IDs")
        for value in key:
            require_identifier(value, "sample identity")
    if len(set(keys)) != len(keys):
        raise ValueError("sample_keys must be unique")


@dataclass(frozen=True)
class GuardBatch(Generic[Features, Targets]):
    task_id: str
    sample_keys: tuple[SampleKey, ...]
    labels_arrived_at: int
    features: Features
    targets: Targets
    role: ExperienceRole

    def __post_init__(self) -> None:
        require_identifier(self.task_id, "task_id")
        require_sample_keys(self.sample_keys)
        require_tick(self.labels_arrived_at, "labels_arrived_at")
        if type(self.role) is not str or self.role not in (
            "train",
            "inner_guard",
            "outer_selection",
            "final_test",
        ):
            raise ValueError("guard role must name an existing evaluation role")


@dataclass(frozen=True)
class PromotionPolicy:
    metric_id: str
    resource_metric_id: str
    min_new_utility: float
    min_new_gain: float
    max_old_drop: float
    max_prediction_seconds: float
    max_resource_bytes: int
    allowed_actions: tuple[str, ...]

    def __post_init__(self) -> None:
        require_identifier(self.metric_id, "metric_id")
        require_identifier(self.resource_metric_id, "resource_metric_id")
        for name in ("min_new_utility", "min_new_gain", "max_old_drop", "max_prediction_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value):
                raise ValueError(f"{name} must be finite and numeric")
        if self.max_old_drop < 0 or self.max_prediction_seconds < 0:
            raise ValueError("retention and latency limits must be nonnegative")
        require_tick(self.max_resource_bytes, "max_resource_bytes")
        if type(self.allowed_actions) is not tuple or not self.allowed_actions:
            raise ValueError("allowed_actions must be a nonempty immutable tuple")
        for action in self.allowed_actions:
            require_identifier(action, "allowed action")
        if len(set(self.allowed_actions)) != len(self.allowed_actions):
            raise ValueError("allowed_actions must be unique")


@dataclass(frozen=True)
class PromotionEvidence:
    actor_new_utility: float | None
    candidate_new_utility: float | None
    actor_old_utility: float | None
    candidate_old_utility: float | None
    candidate_max_prediction_seconds: float | None
    candidate_resource_bytes: int | None
    numerically_valid: bool
    actions_safe: bool

    def __post_init__(self) -> None:
        for value in (*self.utilities, self.candidate_max_prediction_seconds):
            if value is not None and type(value) not in (int, float):
                raise ValueError("measured utility/latency must be numeric or unavailable")
        if self.candidate_resource_bytes is not None:
            require_tick(self.candidate_resource_bytes, "candidate_resource_bytes")
        if type(self.numerically_valid) is not bool or type(self.actions_safe) is not bool:
            raise ValueError("numerical/action observations must be exact booleans")

    @property
    def utilities(self) -> tuple[float | None, ...]:
        return (
            self.actor_new_utility,
            self.candidate_new_utility,
            self.actor_old_utility,
            self.candidate_old_utility,
        )


@dataclass(frozen=True)
class PromotionDecision:
    accepted: bool
    reasons: tuple[str, ...]


def decide_promotion(policy: PromotionPolicy, evidence: PromotionEvidence) -> PromotionDecision:
    if type(policy) is not PromotionPolicy or type(evidence) is not PromotionEvidence:
        raise ValueError("promotion decision requires declared policy and measured evidence")
    reasons = []
    observed = (*evidence.utilities, evidence.candidate_max_prediction_seconds)
    if not evidence.numerically_valid or any(v is not None and not isfinite(v) for v in observed):
        reasons.append("numerical_validity")
    if any(v is None for v in observed) or evidence.candidate_resource_bytes is None:
        reasons.append("missing_measurement")
    if all(v is not None and isfinite(v) for v in evidence.utilities):
        _check_utility(policy, evidence, reasons)
    elapsed = evidence.candidate_max_prediction_seconds
    if elapsed is not None and (elapsed < 0 or elapsed > policy.max_prediction_seconds):
        reasons.append("latency")
    size = evidence.candidate_resource_bytes
    if size is not None and size > policy.max_resource_bytes:
        reasons.append("resource")
    if not evidence.actions_safe:
        reasons.append("action_safety")
    return PromotionDecision(not reasons, tuple(reasons))


def _check_utility(
    policy: PromotionPolicy, evidence: PromotionEvidence, reasons: list[str]
) -> None:
    actor_new, candidate_new = evidence.actor_new_utility, evidence.candidate_new_utility
    actor_old, candidate_old = evidence.actor_old_utility, evidence.candidate_old_utility
    assert actor_new is not None and candidate_new is not None
    assert actor_old is not None and candidate_old is not None
    if candidate_new < policy.min_new_utility:
        reasons.append("new_task_utility")
    if candidate_new - actor_new < policy.min_new_gain:
        reasons.append("new_task_gain")
    if actor_old - candidate_old > policy.max_old_drop:
        reasons.append("old_task_retention")


@dataclass(frozen=True)
class PromotionGuardReport:
    policy: PromotionPolicy
    actor_version: str
    learner_version: str
    candidate_revision: tuple[int, int, int]
    actor_snapshot_digest: str
    candidate_snapshot_digest: str
    new_guard_task_id: str
    old_guard_task_id: str
    new_guard_keys: tuple[SampleKey, ...]
    old_guard_keys: tuple[SampleKey, ...]
    label_arrivals: tuple[int, int]
    training_ids: tuple[SampleKey, ...]
    assessed_at: int
    evidence: PromotionEvidence
    decision: PromotionDecision
