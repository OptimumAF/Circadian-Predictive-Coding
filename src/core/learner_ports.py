"""Native learner contracts with opaque inputs, predictions and model state.

Diagnostics retain each learner's definition; they are not a shared loss.
No clock, filesystem, data-role release, promotion or persistence belongs here.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Protocol, TypeVar

Features = TypeVar("Features", contravariant=True)
Targets = TypeVar("Targets", contravariant=True)
Prediction = TypeVar("Prediction", covariant=True)
State = TypeVar("State")


@dataclass(frozen=True)
class TrainingDiagnostic:
    definition: str
    value: float

    def __post_init__(self) -> None:
        if type(self.definition) is not str or not self.definition:
            raise ValueError("training diagnostic requires a native definition ID")
        if type(self.value) not in {int, float} or not isfinite(self.value):
            raise ValueError("training diagnostic must be finite")


class NativeLearner(Protocol[Features, Targets, Prediction, State]):
    """A complete native wake update and detached, model-owned state boundary."""

    def train_batch(self, features: Features, targets: Targets) -> TrainingDiagnostic: ...

    def predict(self, features: Features) -> Prediction: ...

    def snapshot_state(self) -> State: ...

    def restore_state(self, snapshot: State) -> None: ...
