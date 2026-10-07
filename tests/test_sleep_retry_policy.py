"""Operational sleep retries remain separate from restored model state."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.app.sleep_schedule import (
    SleepAttemptDecision,
    SleepRollbackCooldown,
    decide_sleep_attempt,
    resolve_rollback_cooldown_epochs,
)


@pytest.mark.parametrize(
    ("mode", "configured", "expected"),
    [
        ("legacy", None, 0),
        ("components", None, 1),
        ("disabled", None, 0),
        ("components", 0, 0),
        ("legacy", 2, 2),
    ],
)
def test_cooldown_resolution_preserves_legacy_and_accepts_explicit_override(
    mode: str, configured: int | None, expected: int
) -> None:
    assert resolve_rollback_cooldown_epochs(mode, configured) == expected


@pytest.mark.parametrize("invalid", [-1, 1.5, True])
def test_cooldown_resolution_rejects_invalid_config(invalid: object) -> None:
    with pytest.raises(ValueError, match="cooldown_epochs"):
        resolve_rollback_cooldown_epochs("components", invalid)  # type: ignore[arg-type]


def _due(epoch: int, *, adaptive: bool = False) -> SleepAttemptDecision:
    return decide_sleep_attempt(
        sleep_mode="components",
        completed_epochs=epoch,
        interval_epochs=1,
        adaptive_due=adaptive,
        force_periodic=True,
    )


def test_rejection_without_new_wake_blocks_identical_attempts_indefinitely() -> None:
    policy = SleepRollbackCooldown(1)
    assert policy.allow_due_attempt(_due(1), completed_epochs=1, wake_batches=0)
    policy.record_rejection(completed_epochs=1, wake_batches=0)
    for epoch in range(2, 7):
        assert not policy.allow_due_attempt(_due(epoch), completed_epochs=epoch, wake_batches=0)
    state = policy.snapshot_state()
    assert state.rejected_attempts == 1
    assert state.suppressed_due_attempts == 5
    assert state.next_eligible_epoch == 3
    assert state.last_rejected_wake_batches == 0


def test_retry_requires_both_epoch_cooldown_and_new_wake_batch() -> None:
    policy = SleepRollbackCooldown(1)
    policy.record_rejection(completed_epochs=1, wake_batches=2)
    assert not policy.allow_due_attempt(_due(2, adaptive=True), completed_epochs=2, wake_batches=3)
    assert not policy.allow_due_attempt(_due(3), completed_epochs=3, wake_batches=2)
    assert policy.allow_due_attempt(_due(3), completed_epochs=3, wake_batches=3)
    assert policy.snapshot_state().suppressed_due_attempts == 2


def test_non_due_and_zero_cooldown_preserve_existing_attempt_semantics() -> None:
    policy = SleepRollbackCooldown(0)
    policy.record_rejection(completed_epochs=1, wake_batches=0)
    not_due = decide_sleep_attempt(
        sleep_mode="legacy",
        completed_epochs=2,
        interval_epochs=3,
        adaptive_due=False,
        force_periodic=True,
    )
    assert not policy.allow_due_attempt(not_due, completed_epochs=2, wake_batches=0)
    assert policy.allow_due_attempt(_due(2), completed_epochs=2, wake_batches=0)
    assert policy.snapshot_state().suppressed_due_attempts == 0


def test_retry_snapshot_restores_deterministic_operational_continuation() -> None:
    policy = SleepRollbackCooldown(2)
    policy.record_rejection(completed_epochs=1, wake_batches=4)
    assert not policy.allow_due_attempt(_due(2), completed_epochs=2, wake_batches=5)
    saved = policy.snapshot_state()
    resumed = SleepRollbackCooldown(2)
    resumed.restore_state(saved)
    for epoch, wake_batches in [(3, 5), (4, 5), (5, 6)]:
        assert policy.allow_due_attempt(
            _due(epoch), completed_epochs=epoch, wake_batches=wake_batches
        ) == resumed.allow_due_attempt(
            _due(epoch), completed_epochs=epoch, wake_batches=wake_batches
        )
    assert policy.snapshot_state() == resumed.snapshot_state()

    before = resumed.snapshot_state()
    with pytest.raises(ValueError, match="incompatible"):
        resumed.restore_state(replace(saved, cooldown_epochs=1))
    assert resumed.snapshot_state() == before
