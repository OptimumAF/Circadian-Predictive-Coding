"""Owned forks and versioned local actor/candidate results.

Payload/state layouts remain native. Forks must own independent mutable model
state and preserve training policy; this contract is not a graph/source seal.
No thread, budget, promotion, IO or scientific release belongs in this module.
"""

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from src.core.learner_ports import NativeLearner, TrainingDiagnostic

Features = TypeVar("Features", contravariant=True)
Targets = TypeVar("Targets", contravariant=True)
Prediction = TypeVar("Prediction", covariant=True)
State = TypeVar("State")


class ForkableLearner(NativeLearner[Features, Targets, Prediction, State], Protocol):
    def fork(self) -> NativeLearner[Features, Targets, Prediction, State]:
        """Return an independent complete model and the same native training policy."""
        ...


@dataclass(frozen=True)
class VersionedPrediction(Generic[Prediction]):
    actor_version: str
    prediction: Prediction


@dataclass(frozen=True)
class VersionedState(Generic[State]):
    actor_version: str
    state: State


@dataclass(frozen=True)
class CandidateState(Generic[State]):
    actor_version: str
    learner_version: str
    state: State
    wake_updates: int
    consolidation_attempts: int
    consolidations_completed: int


@dataclass(frozen=True)
class ConsolidatedState(Generic[State]):
    state: State
    diagnostic: TrainingDiagnostic

    def __post_init__(self) -> None:
        if type(self.diagnostic) is not TrainingDiagnostic:
            raise ValueError("consolidation requires a native TrainingDiagnostic")


@dataclass(frozen=True)
class AppliedConsolidation:
    event_id: str
    actor_version: str
    learner_version: str
    attempt_number: int
    diagnostic: TrainingDiagnostic
