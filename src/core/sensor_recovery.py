"""Summarize one ordered phase's prearrival prediction accuracies.

Inputs are bounded, already scored accuracies and the phase's public timing.
Outputs describe the first observed deficit and its first sustained recovery.
This module does not run models, release targets, choose a shift policy, or
attribute recovery to learning. Labels counted here arrived before prediction;
the retrospective targets used to confirm accuracy can arrive later.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Literal


@dataclass(frozen=True)
class RecoverySummary:
    """First recovery, or explicit absence of a deficit/observed recovery."""

    status: Literal["no_deficit", "recovered", "censored"]
    first_deficit_tick: int | None
    confirmation_tick: int | None
    labels_to_recovery: int | None
    labels_available_before_last_prediction: int


def _bounded_integer(value: int, name: str, lower: int, upper: int) -> None:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be a native integer in [{lower}, {upper}]")


def _validate_inputs(
    accuracies: tuple[float, ...],
    start_tick: int,
    label_delay: int,
    rows_per_batch: int,
    accuracy_threshold: float,
    recovery_batches: int,
) -> None:
    if type(accuracies) is not tuple or not 1 <= len(accuracies) <= 512:
        raise ValueError("accuracies must be a tuple containing 1 to 512 values")
    for accuracy in accuracies:
        if type(accuracy) not in (int, float) or not 0 <= accuracy <= 1 or not isfinite(accuracy):
            raise ValueError("accuracies must contain finite native numbers in [0, 1]")
    _bounded_integer(start_tick, "start_tick", 0, 511)
    _bounded_integer(label_delay, "label_delay", 0, 512)
    _bounded_integer(rows_per_batch, "rows_per_batch", 1, 128)
    _bounded_integer(recovery_batches, "recovery_batches", 1, 32)
    if (
        type(accuracy_threshold) not in (int, float)
        or not 0 < accuracy_threshold <= 1
        or not isfinite(accuracy_threshold)
    ):
        raise ValueError("accuracy_threshold must be a finite native number in (0, 1]")
    if start_tick + len(accuracies) > 512:
        raise ValueError("the phase must end before tick 512")


def summarize_phase_recovery(
    accuracies: tuple[float, ...],
    *,
    start_tick: int,
    label_delay: int,
    rows_per_batch: int,
    accuracy_threshold: float = 0.75,
    recovery_batches: int = 2,
) -> RecoverySummary:
    """Find the first deficit followed by consecutive adequate predictions.

    Equality with the threshold is adequate. Phase labels count only when they
    arrive strictly before a prediction, excluding same-tick arrivals because
    the prediction hook runs first. This count includes labels from the phase
    start, rather than only those after a deficit; pre-phase labels are excluded.
    A zero-label recovery is possible and is not evidence of learning causality.
    A censored phase may have an adequate final prediction without enough
    consecutive adequate predictions to confirm recovery.
    """
    _validate_inputs(
        accuracies, start_tick, label_delay, rows_per_batch, accuracy_threshold, recovery_batches
    )
    labels_at_last = rows_per_batch * min(
        len(accuracies), max(0, len(accuracies) - 1 - label_delay)
    )
    first_deficit = None
    adequate_streak = 0
    for offset, accuracy in enumerate(accuracies):
        if accuracy < accuracy_threshold:
            if first_deficit is None:
                first_deficit = start_tick + offset
            adequate_streak = 0
        elif first_deficit is not None:
            adequate_streak += 1
            if adequate_streak == recovery_batches:
                tick = start_tick + offset
                labels = rows_per_batch * min(len(accuracies), max(0, offset - label_delay))
                return RecoverySummary("recovered", first_deficit, tick, labels, labels_at_last)
    status: Literal["no_deficit", "censored"] = (
        "no_deficit" if first_deficit is None else "censored"
    )
    return RecoverySummary(status, first_deficit, None, None, labels_at_last)
