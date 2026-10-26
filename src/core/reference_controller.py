"""Causal fixed adaptation references over aggregate ordinary TRAIN feedback.

Pure scalar choices only. No native/data handles, optimizer, clock mutation,
permission change, evaluation release, tuning or learned policy lives here.
"""

from dataclasses import dataclass
from math import isfinite

from src.core.experience import require_identifier, require_tick

REFERENCE_MODES = ("no_learning", "periodic", "drift", "budget_aware")


@dataclass(frozen=True)
class ReferencePolicy:
    mode: str
    period: int = 2
    drift_threshold: float = 0.1
    reserved_updates: int = 4

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in REFERENCE_MODES:
            raise ValueError("unknown fixed reference mode")
        require_tick(self.period, "period")
        if self.period == 0:
            raise ValueError("period must be positive")
        if (
            type(self.drift_threshold) not in (int, float)
            or not 0 <= self.drift_threshold <= 1
            or not isfinite(self.drift_threshold)
        ):
            raise ValueError("drift_threshold must be finite in [0,1]")
        require_tick(self.reserved_updates, "reserved_updates")


@dataclass(frozen=True)
class ArrivedTrainingFeedback:
    role: str
    sample_id: str
    actor_version: str
    observed_at: int
    arrived_at: int
    mistakes: int
    examples: int

    def __post_init__(self) -> None:
        if type(self.role) is not str or self.role != "train":
            raise ValueError("reference feedback requires ordinary TRAIN role")
        require_identifier(self.sample_id, "sample_id")
        require_identifier(self.actor_version, "actor_version")
        for name in ("observed_at", "arrived_at", "mistakes", "examples"):
            require_tick(getattr(self, name), name)
        if self.arrived_at < self.observed_at:
            raise ValueError("feedback cannot arrive before observation")
        if self.examples == 0 or self.mistakes > self.examples:
            raise ValueError("feedback requires positive examples and bounded mistakes")


@dataclass(frozen=True)
class ArrivedTaskFeedback:
    """Complete private controller-TRAIN task returns, never classifier labels."""

    role: str
    sample_id: str
    actor_version: str
    observed_at: int
    arrived_at: int
    failed_tasks: int
    tasks: int

    def __post_init__(self) -> None:
        if type(self.role) is not str or self.role != "controller_train":
            raise ValueError("task feedback requires controller_train role")
        require_identifier(self.sample_id, "sample_id")
        require_identifier(self.actor_version, "actor_version")
        for name in ("observed_at", "arrived_at", "failed_tasks", "tasks"):
            require_tick(getattr(self, name), name)
        if self.arrived_at < self.observed_at or self.tasks == 0 or self.failed_tasks > self.tasks:
            raise ValueError("task feedback requires causal bounded complete-task counts")

    @property
    def mistakes(self) -> int:
        return self.failed_tasks

    @property
    def examples(self) -> int:
        return self.tasks


@dataclass(frozen=True)
class ArrivedShiftFeedback:
    """Scalar categorical alarms from an arrived TRAIN packet, before a wake.

    The outer reader owns packet arrival and candidate-state binding. This
    record grants no authority to train, skip FIFO entries or renew credits.
    """

    role: str
    sample_id: str
    actor_version: str
    observed_at: int
    arrived_at: int
    candidate_version: str
    wake_updates: int
    batch_index: int
    alarms: tuple[bool, bool, bool]

    def __post_init__(self) -> None:
        if type(self.role) is not str or self.role != "train":
            raise ValueError("shift feedback requires ordinary TRAIN role")
        for name in ("sample_id", "actor_version", "candidate_version"):
            require_identifier(getattr(self, name), name)
        for name in ("observed_at", "arrived_at", "wake_updates", "batch_index"):
            require_tick(getattr(self, name), name)
        if self.arrived_at < self.observed_at or not 1 <= self.batch_index <= 512:
            raise ValueError("shift feedback requires causal clocks and a bounded batch index")
        if (
            type(self.alarms) is not tuple
            or len(self.alarms) != 3
            or any(type(value) is not bool for value in self.alarms)
        ):
            raise ValueError("shift alarms require three native booleans")


@dataclass(frozen=True)
class ReferenceObservation:
    tick: int
    remaining_opportunities: int
    ready_updates: int
    remaining_wake: int
    actor_version: str
    feedback: ArrivedTrainingFeedback | ArrivedTaskFeedback | ArrivedShiftFeedback | None = None

    def __post_init__(self) -> None:
        for name in ("tick", "remaining_opportunities", "ready_updates", "remaining_wake"):
            require_tick(getattr(self, name), name)
        if self.remaining_opportunities == 0:
            raise ValueError("selection requires a remaining opportunity")
        require_identifier(self.actor_version, "actor_version")
        if self.feedback is not None:
            if type(self.feedback) not in (
                ArrivedTrainingFeedback,
                ArrivedTaskFeedback,
                ArrivedShiftFeedback,
            ):
                raise ValueError("feedback must be exact arrived TRAIN metadata")
            self.feedback.__post_init__()
            if self.feedback.arrived_at > self.tick:
                raise ValueError("feedback labels have not arrived")
            if self.feedback.actor_version != self.actor_version:
                raise ValueError("feedback belongs to a stale actor version")


@dataclass(frozen=True)
class ReferenceDecision:
    action_id: str
    reason: str


def choose_reference(
    policy: ReferencePolicy, observation: ReferenceObservation
) -> ReferenceDecision:
    if type(policy) is not ReferencePolicy or type(observation) is not ReferenceObservation:
        raise ValueError("reference selection requires exact policy and causal observation")
    policy.__post_init__()
    observation.__post_init__()
    if policy.mode == "no_learning":
        return ReferenceDecision("wait", "no_learning")
    if observation.ready_updates == 0 or observation.remaining_wake == 0:
        return ReferenceDecision("wait", "no_ready_work_or_credit")
    if policy.mode == "periodic":
        wake = observation.tick % policy.period == 0
    elif policy.mode == "drift":
        feedback = observation.feedback
        if isinstance(feedback, ArrivedShiftFeedback):
            return ReferenceDecision("wake1" if any(feedback.alarms) else "wait", "arrived_shift")
        wake = (
            feedback is not None and feedback.mistakes / feedback.examples > policy.drift_threshold
        )
    else:
        # Spend only unreserved credit when there are no more future chances to
        # spread it over. This is a fixed budget heuristic, not optimized utility.
        usable = max(0, observation.remaining_wake - policy.reserved_updates)
        wake = usable >= observation.remaining_opportunities
    return ReferenceDecision("wake1" if wake else "wait", policy.mode)
