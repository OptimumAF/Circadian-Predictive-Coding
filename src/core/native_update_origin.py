"""Local update reference observations; no persistence or provenance certification.

Inputs are original managed records and the actual detached native inputs.
Observers own any references they retain; this contract grants no replay,
restore, consent, or recovery authority.
"""

from dataclasses import dataclass
from typing import Callable, Generic, Literal, TypeVar

from src.core.experience import AppliedExperience, Experience, LabelArrival

Features = TypeVar("Features")
Targets = TypeVar("Targets")
NativeUpdateStage = Literal["started", "completed", "refused", "uncertain", "committed_failure"]


@dataclass(frozen=True)
class NativeUpdateOrigin(Generic[Features, Targets]):
    learner: object
    source: Experience[Features]
    label: LabelArrival[Targets]
    features: Features
    targets: Targets
    learner_version: str
    receipt: AppliedExperience | None
    completed_updates: int


NativeUpdateObserver = Callable[
    [NativeUpdateStage, Callable[[], NativeUpdateOrigin[Features, Targets]]], None
]
