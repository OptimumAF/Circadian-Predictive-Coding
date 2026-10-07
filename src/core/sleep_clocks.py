"""Typed epoch progress and read-only circadian training-clock snapshots."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SleepEpochProgress:
    """Runner epochs completed before a sleep attempt, within a fixed run cap."""

    completed_epochs: int
    total_epochs: int

    def __post_init__(self) -> None:
        if type(self.total_epochs) is not int or self.total_epochs <= 0:
            raise ValueError("total_epochs must be a positive integer")
        if (
            type(self.completed_epochs) is not int
            or self.completed_epochs < 0
            or self.completed_epochs > self.total_epochs
        ):
            raise ValueError("completed_epochs must be an integer in [0, total_epochs]")


@dataclass(frozen=True)
class SleepClockSnapshot:
    """Successful core work counts; runner attempts/epochs live outside the model."""

    wake_batches: int
    wake_examples: int
    wake_batches_since_sleep: int
    replay_updates: int
    sleep_events: int
