"""Bounded incident transitions from validated descriptive shift decisions.

Inputs are ordered ``ShiftDecision`` records; outputs contain only channel
states and their onset batches, in uncertainty/feature/residual order. This
module owns no scores, thresholds, models, training policies, clocks, IO or RNG.
"""

from dataclasses import dataclass
from math import isfinite

from src.core.shift_signals import ShiftDecision


@dataclass(frozen=True)
class ShiftIncidentDecision:
    batch_index: int
    states: tuple[str, str, str]
    onset_batches: tuple[int | None, int | None, int | None]


def _validate_decision(decision: ShiftDecision, expected_batch: int) -> None:
    if type(decision) is not ShiftDecision:
        raise ValueError("shift incidents require an exact ShiftDecision record")
    if (
        type(decision.batch_index) is not int
        or not 1 <= decision.batch_index <= 512
        or decision.batch_index != expected_batch
    ):
        raise ValueError("shift incident batches must be consecutive native integers in 1..512")
    if type(decision.phase) is not str or decision.phase not in {
        "reference",
        "window",
        "calibration",
        "monitor",
    }:
        raise ValueError("shift incident phase must be reference, window, calibration or monitor")
    if (
        type(decision.alarms) is not tuple
        or len(decision.alarms) != 3
        or any(type(value) is not bool for value in decision.alarms)
    ):
        raise ValueError("shift incident alarms must be a tuple of three native booleans")
    if (
        type(decision.streaks) is not tuple
        or len(decision.streaks) != 3
        or any(type(value) is not int or not 0 <= value <= 512 for value in decision.streaks)
    ):
        raise ValueError(
            "shift incident streaks must be a tuple of three native integers in 0..512"
        )
    if any(alarm and streak == 0 for alarm, streak in zip(decision.alarms, decision.streaks)):
        raise ValueError("an alarm requires a positive shift streak")
    if decision.phase != "monitor" and (any(decision.alarms) or any(decision.streaks)):
        raise ValueError("only monitor decisions may have shift alarms or streaks")
    _validate_scores(decision)


def _validate_scores(decision: ShiftDecision) -> None:
    scores = (decision.uncertainty_change, decision.feature_drift, decision.residual_change)
    if decision.phase in {"reference", "window"}:
        if any(value is not None for value in scores):
            raise ValueError("reference and window shift scores must be absent")
        return
    if decision.phase == "calibration" and all(value is None for value in scores):
        return
    for value, lower, upper in zip(scores, (-1, 0, -1), (1, 2, 1)):
        if (
            type(value) not in {int, float}
            or value is None
            or not lower <= value <= upper
            or not isfinite(value)
        ):
            raise ValueError("shift incident scores must be finite native numbers in their domains")


class ShiftIncidentTracker:
    """Describe one incident per channel without retaining observation history.

    Why this: an alarm level is not a new trigger on every batch. A channel
    clears on its first non-alarm observation; no extra threshold or cooldown
    changes the detector's meaning. A clear record retains the old onset once.
    Monitoring cannot regress to a prefix phase and silently discard incidents.
    """

    def __init__(self) -> None:
        self._count = 0
        self._active_onsets: tuple[int | None, int | None, int | None] = (None, None, None)
        self._monitoring = False

    def observe(self, decision: ShiftDecision) -> ShiftIncidentDecision:
        _validate_decision(decision, self._count + 1)
        if self._monitoring and decision.phase != "monitor":
            raise ValueError("shift incidents cannot return to a prefix phase after monitoring")
        states: list[str] = []
        recorded_onsets: list[int | None] = []
        active_onsets: list[int | None] = []
        for alarm, onset in zip(decision.alarms, self._active_onsets):
            if decision.phase != "monitor":
                state, recorded, active = "inactive", None, None
            elif alarm:
                state = "onset" if onset is None else "continued"
                recorded = active = decision.batch_index if onset is None else onset
            else:
                state = "inactive" if onset is None else "cleared"
                recorded, active = onset, None
            states.append(state)
            recorded_onsets.append(recorded)
            active_onsets.append(active)
        result = ShiftIncidentDecision(
            decision.batch_index,
            (states[0], states[1], states[2]),
            (recorded_onsets[0], recorded_onsets[1], recorded_onsets[2]),
        )
        # Commit only after all input validation and the detached result exist.
        self._active_onsets = (active_onsets[0], active_onsets[1], active_onsets[2])
        self._count = decision.batch_index
        self._monitoring = decision.phase == "monitor"
        return result
