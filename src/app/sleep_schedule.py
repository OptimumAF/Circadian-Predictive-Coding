"""Resolve epoch-based sleep attempts independently of model mutation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SleepAttemptDecision:
    """One runner scheduling decision before a core sleep call."""

    periodic_due: bool
    adaptive_due: bool
    attempted: bool
    force_sleep: bool


@dataclass(frozen=True)
class SleepRollbackCooldownState:
    """Runner-owned retry state; never included in a model rollback snapshot."""

    format_version: int
    cooldown_epochs: int
    next_eligible_epoch: int
    last_rejected_epoch: int | None
    last_rejected_wake_batches: int | None
    rejected_attempts: int
    suppressed_due_attempts: int


class SleepRollbackCooldown:
    """Wait for both elapsed epochs and new wake work after a guard rejection."""

    def __init__(self, cooldown_epochs: int) -> None:
        if type(cooldown_epochs) is not int or cooldown_epochs < 0:
            raise ValueError("cooldown_epochs must be a non-negative integer")
        self.cooldown_epochs = cooldown_epochs
        self._next_eligible_epoch = 0
        self._last_rejected_epoch: int | None = None
        self._last_rejected_wake_batches: int | None = None
        self._rejected_attempts = 0
        self._suppressed_due_attempts = 0

    def allow_due_attempt(
        self, decision: SleepAttemptDecision, *, completed_epochs: int, wake_batches: int
    ) -> bool:
        self._validate_clocks(completed_epochs, wake_batches)
        if not decision.attempted:
            return False
        # Why this: zero explicitly retains the reviewed runner schedule.
        if self.cooldown_epochs == 0:
            return True
        if (
            completed_epochs < self._next_eligible_epoch
            or self._last_rejected_wake_batches is not None
            and wake_batches <= self._last_rejected_wake_batches
        ):
            self._suppressed_due_attempts += 1
            return False
        return True

    def record_rejection(self, *, completed_epochs: int, wake_batches: int) -> None:
        self._validate_clocks(completed_epochs, wake_batches)
        if self._last_rejected_epoch is not None and completed_epochs <= self._last_rejected_epoch:
            raise ValueError("rejected sleep epochs must increase")
        self._rejected_attempts += 1
        self._last_rejected_epoch = completed_epochs
        self._last_rejected_wake_batches = wake_batches
        self._next_eligible_epoch = completed_epochs + self.cooldown_epochs + 1

    def snapshot_state(self) -> SleepRollbackCooldownState:
        return SleepRollbackCooldownState(
            format_version=1,
            cooldown_epochs=self.cooldown_epochs,
            next_eligible_epoch=self._next_eligible_epoch,
            last_rejected_epoch=self._last_rejected_epoch,
            last_rejected_wake_batches=self._last_rejected_wake_batches,
            rejected_attempts=self._rejected_attempts,
            suppressed_due_attempts=self._suppressed_due_attempts,
        )

    def restore_state(self, state: SleepRollbackCooldownState) -> None:
        if (
            not isinstance(state, SleepRollbackCooldownState)
            or state.format_version != 1
            or state.cooldown_epochs != self.cooldown_epochs
        ):
            raise ValueError("incompatible sleep rollback cooldown state")
        if any(
            type(value) is not int or value < 0
            for value in (state.next_eligible_epoch, state.rejected_attempts, state.suppressed_due_attempts)
        ):
            raise ValueError("incompatible sleep rollback cooldown counters")
        if state.rejected_attempts == 0:
            valid_history = (
                state.last_rejected_epoch is None
                and state.last_rejected_wake_batches is None
                and state.next_eligible_epoch == 0
            )
        else:
            valid_history = (
                type(state.last_rejected_epoch) is int
                and state.last_rejected_epoch >= 0
                and type(state.last_rejected_wake_batches) is int
                and state.last_rejected_wake_batches >= 0
                and state.next_eligible_epoch
                == state.last_rejected_epoch + self.cooldown_epochs + 1
            )
        if not valid_history:
            raise ValueError("incompatible sleep rollback cooldown history")
        self._next_eligible_epoch = state.next_eligible_epoch
        self._last_rejected_epoch = state.last_rejected_epoch
        self._last_rejected_wake_batches = state.last_rejected_wake_batches
        self._rejected_attempts = state.rejected_attempts
        self._suppressed_due_attempts = state.suppressed_due_attempts

    @staticmethod
    def _validate_clocks(completed_epochs: int, wake_batches: int) -> None:
        if type(completed_epochs) is not int or completed_epochs < 0:
            raise ValueError("completed_epochs must be a non-negative integer")
        if type(wake_batches) is not int or wake_batches < 0:
            raise ValueError("wake_batches must be a non-negative integer")


def resolve_rollback_cooldown_epochs(sleep_mode: str, configured_epochs: int | None) -> int:
    """Keep legacy scheduling while enabling a fixed corrected-mode default."""
    if sleep_mode not in {"legacy", "components", "disabled"}:
        raise ValueError("sleep_mode must be one of: legacy, components, disabled")
    if configured_epochs is None:
        return 1 if sleep_mode == "components" else 0
    if type(configured_epochs) is not int or configured_epochs < 0:
        raise ValueError("cooldown_epochs must be a non-negative integer or None")
    return configured_epochs


def decide_sleep_attempt(
    *,
    sleep_mode: str,
    completed_epochs: int,
    interval_epochs: int,
    adaptive_due: bool,
    force_periodic: bool,
) -> SleepAttemptDecision:
    """Treat an interval as an attempt; only a forced periodic attempt bypasses adaptation."""
    if sleep_mode not in {"legacy", "components", "disabled"}:
        raise ValueError("sleep_mode must be one of: legacy, components, disabled")
    if type(completed_epochs) is not int or completed_epochs < 0:
        raise ValueError("completed_epochs must be a non-negative integer")
    if type(interval_epochs) is not int or interval_epochs < 0:
        raise ValueError("interval_epochs must be a non-negative integer")
    if sleep_mode == "disabled":
        return SleepAttemptDecision(False, False, False, False)

    periodic_due = (
        interval_epochs > 0 and completed_epochs > 0 and completed_epochs % interval_epochs == 0
    )
    return SleepAttemptDecision(
        periodic_due=periodic_due,
        adaptive_due=adaptive_due,
        attempted=periodic_due or adaptive_due,
        force_sleep=periodic_due and force_periodic,
    )
