"""Bound physical step attempts across episodes without clocks or history.

An attempt is charged before physics and never refunded. Exactly one attempt
may be pending; a failed physical call permanently stops this quota. The caller
owns episode selection and must account for predictions and learning separately.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalInteractionQuotaSnapshot:
    maximum_steps: int
    attempted_steps: int
    completed_steps: int
    failed_steps: int


class PhysicalInteractionQuota:
    """One finite allowance, shared across consecutive fresh episodes."""

    __slots__ = (
        "_maximum_steps",
        "_birth_maximum_steps",
        "_attempted_steps",
        "_completed_steps",
        "_failed_steps",
    )

    def __init__(self, maximum_steps: int) -> None:
        if type(maximum_steps) is not int or not 1 <= maximum_steps <= 4096:
            raise ValueError("maximum_steps must be a native integer in 1..4096")
        self._maximum_steps = self._birth_maximum_steps = maximum_steps
        self._attempted_steps = self._completed_steps = self._failed_steps = 0

    def _validate(self) -> None:
        values = (self._attempted_steps, self._completed_steps, self._failed_steps)
        if (
            type(self._birth_maximum_steps) is not int
            or not 1 <= self._birth_maximum_steps <= 4096
            or type(self._maximum_steps) is not int
            or self._maximum_steps != self._birth_maximum_steps
            or any(type(value) is not int or value < 0 for value in values)
            or self._attempted_steps > self._maximum_steps
            or not 0 <= self._attempted_steps - self._completed_steps - self._failed_steps <= 1
            or self._failed_steps > 1
            or self._failed_steps
            and self._attempted_steps != self._completed_steps + self._failed_steps
        ):
            raise ValueError("physical interaction quota has invalid counters or maximum")

    def _pending(self) -> int:
        return self._attempted_steps - self._completed_steps - self._failed_steps

    @property
    def maximum_steps(self) -> int:
        self._validate()
        return self._maximum_steps

    @property
    def attempted_steps(self) -> int:
        self._validate()
        return self._attempted_steps

    @property
    def completed_steps(self) -> int:
        self._validate()
        return self._completed_steps

    @property
    def failed_steps(self) -> int:
        self._validate()
        return self._failed_steps

    @property
    def remaining_steps(self) -> int:
        self._validate()
        return self._maximum_steps - self._attempted_steps

    def begin_step(self) -> None:
        self._validate()
        if self._failed_steps or self._pending() or self._attempted_steps == self._maximum_steps:
            raise ValueError("physical interaction quota is failed, pending or exhausted")
        self._attempted_steps += 1

    def complete_step(self) -> None:
        self._validate()
        if self._failed_steps or self._pending() != 1:
            raise ValueError("physical interaction quota has no pending step")
        self._completed_steps += 1

    def fail_step(self) -> None:
        self._validate()
        if self._failed_steps or self._pending() != 1:
            raise ValueError("physical interaction quota has no pending step")
        self._failed_steps += 1

    def snapshot(self) -> PhysicalInteractionQuotaSnapshot:
        self._validate()
        return PhysicalInteractionQuotaSnapshot(
            self._maximum_steps,
            self._attempted_steps,
            self._completed_steps,
            self._failed_steps,
        )
