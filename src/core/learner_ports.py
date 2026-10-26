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


def validate_learning_rate_multiplier(value: float) -> None:
    """Bound an optional per-update effect without granting a new base policy."""
    try:
        valid = type(value) in (int, float) and isfinite(value) and 0 < value <= 4
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("learning-rate multiplier must be a native finite scalar in (0,4]")


GATE_SETTINGS = ("unchanged", "neutral", "active")


def validate_gate_setting(value: str) -> None:
    if type(value) is not str or value not in GATE_SETTINGS:
        raise ValueError("gate setting must be unchanged, neutral or active")


class GateControlledLearner(Protocol[Features, Targets]):
    """Optional one-update plasticity effect, with model-owned capability checks."""

    def validate_gate_setting(self, setting: str) -> None: ...

    def train_batch_with_gate_setting(
        self, features: Features, targets: Targets, setting: str
    ) -> "TrainingDiagnostic": ...


class RateControlledLearner(Protocol[Features, Targets]):
    """Optional one-update rate effect; base policy/state schema stays unchanged."""

    def train_batch_with_rate_multiplier(
        self, features: Features, targets: Targets, multiplier: float
    ) -> "TrainingDiagnostic": ...


@dataclass(frozen=True)
class TrainingDiagnostic:
    definition: str
    value: float

    def __post_init__(self) -> None:
        if type(self.definition) is not str or not self.definition:
            raise ValueError("training diagnostic requires a native definition ID")
        if type(self.value) not in {int, float} or not isfinite(self.value):
            raise ValueError("training diagnostic must be finite")


class TrainingUpdateFailed(RuntimeError):
    """A native update failed after bounded work, distinct from app charges.

    Why: an unpublished physical optimizer step must survive a generic runner's
    error report without importing an outer adapter or inventing a budget charge.
    Work contains only owned native scalar counters/flags, never model payloads.
    """

    def __init__(self, reason: str, definition: str, work: dict[str, int | bool]) -> None:
        if type(reason) is not str or not reason or type(definition) is not str or not definition:
            raise ValueError("training failure requires native nonempty reason and definition")
        if type(work) is not dict or len(work) > 32:
            raise ValueError("training failure work must be an exact dict with at most32 counters")
        for key, value in work.items():
            if type(key) is not str or not key or key != key.strip():
                raise ValueError("training failure work keys must be native nonempty identifiers")
            try:
                valid_key = len(key.encode("utf8")) <= 64
            except UnicodeEncodeError:
                valid_key = False
            if not valid_key or not (
                type(value) is bool or (type(value) is int and 0 <= value <= 2**53 - 1)
            ):
                raise ValueError(
                    "training failure work must contain bounded native counters or bool flags"
                )
        self._definition = definition
        self._work_items = tuple(work.items())
        super().__init__(reason)

    @property
    def definition(self) -> str:
        return self._definition

    @property
    def work(self) -> dict[str, int | bool]:
        """Return an owned copy; mutating it cannot change the paid failure."""
        return dict(self._work_items)


class NativeLearner(Protocol[Features, Targets, Prediction, State]):
    """A complete native wake update and detached, model-owned state boundary."""

    def train_batch(self, features: Features, targets: Targets) -> TrainingDiagnostic: ...

    def predict(self, features: Features) -> Prediction: ...

    def snapshot_state(self) -> State: ...

    def restore_state(self, snapshot: State) -> None: ...
