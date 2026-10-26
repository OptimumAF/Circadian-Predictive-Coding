"""Bounded, permanent accounting for attempted physical environment steps."""

from dataclasses import FrozenInstanceError, asdict, fields, is_dataclass

import numpy as np
import pytest

from src.core.physical_interaction_quota import PhysicalInteractionQuota


def counts(maximum, attempted=0, completed=0, failed=0):
    return dict(
        maximum_steps=maximum,
        attempted_steps=attempted,
        completed_steps=completed,
        failed_steps=failed,
    )


@pytest.mark.parametrize("maximum", (1, 12, 4096))
def test_should_start_with_exact_native_counts_and_an_owned_frozen_snapshot(maximum):
    quota = PhysicalInteractionQuota(maximum)
    snapshot = quota.snapshot()
    assert is_dataclass(snapshot)
    assert tuple(field.name for field in fields(snapshot)) == tuple(counts(maximum))
    assert asdict(snapshot) == counts(maximum)
    assert all(type(value) is int for value in asdict(snapshot).values())
    assert quota.maximum_steps == quota.remaining_steps == maximum
    assert quota.attempted_steps == quota.completed_steps == quota.failed_steps == 0
    assert getattr(type(snapshot), "__dataclass_params__").frozen is True
    with pytest.raises(FrozenInstanceError):
        setattr(snapshot, "attempted_steps", 99)
    quota.begin_step()
    quota.complete_step()
    assert asdict(snapshot) == counts(maximum)
    assert asdict(quota.snapshot()) == counts(maximum, 1, 1)


@pytest.mark.parametrize("invalid", (None, False, True, 0, -1, 4097, 1.0, "12", np.int64(12)))
def test_should_refuse_non_native_or_out_of_range_maximum_before_creating_a_quota(invalid):
    with pytest.raises(ValueError):
        PhysicalInteractionQuota(invalid)


def test_should_charge_before_completion_and_never_allow_two_pending_steps():
    quota = PhysicalInteractionQuota(2)
    assert getattr(quota, "begin_step")() is None
    pending = quota.snapshot()
    assert asdict(pending) == counts(2, 1)
    assert quota.remaining_steps == 1
    with pytest.raises(ValueError):
        quota.begin_step()
    assert quota.snapshot() == pending
    assert getattr(quota, "complete_step")() is None
    assert asdict(quota.snapshot()) == counts(2, 1, 1)
    quota.begin_step()
    assert quota.remaining_steps == 0
    quota.complete_step()
    exhausted = quota.snapshot()
    assert asdict(exhausted) == counts(2, 2, 2)
    with pytest.raises(ValueError):
        quota.begin_step()
    assert quota.snapshot() == exhausted


@pytest.mark.parametrize("operation", ("complete_step", "fail_step"))
def test_should_refuse_a_return_without_a_pending_attempt(operation):
    quota = PhysicalInteractionQuota(3)
    initial = quota.snapshot()
    with pytest.raises(ValueError):
        getattr(quota, operation)()
    assert quota.snapshot() == initial
    quota.begin_step()
    quota.complete_step()
    completed = quota.snapshot()
    with pytest.raises(ValueError):
        getattr(quota, operation)()
    assert quota.snapshot() == completed


def test_should_keep_a_failed_attempt_paid_and_latch_failure_with_unused_capacity():
    quota = PhysicalInteractionQuota(12)
    quota.begin_step()
    quota.complete_step()
    quota.begin_step()
    assert getattr(quota, "fail_step")() is None
    failed = quota.snapshot()
    assert asdict(failed) == counts(12, 2, 1, 1)
    assert quota.remaining_steps == 10
    for operation in ("begin_step", "complete_step", "fail_step"):
        with pytest.raises(ValueError):
            getattr(quota, operation)()
        assert quota.snapshot() == failed


@pytest.mark.parametrize(
    "attribute",
    ("maximum_steps", "attempted_steps", "completed_steps", "failed_steps", "remaining_steps"),
)
def test_should_expose_read_only_counts_without_a_reset_interface(attribute):
    quota = PhysicalInteractionQuota(12)
    before = quota.snapshot()
    with pytest.raises(AttributeError):
        setattr(quota, attribute, 0)
    assert quota.snapshot() == before
    assert not hasattr(quota, "reset")


CORRUPTIONS = (
    {"_maximum_steps": 13},
    {"_maximum_steps": True},
    {"_attempted_steps": -1},
    {"_attempted_steps": True},
    {"_attempted_steps": 1.0},
    {"_attempted_steps": 13},
    {"_completed_steps": 1},
    {"_failed_steps": 1},
    {"_failed_steps": False},
    {"_attempted_steps": 2},
    {"_attempted_steps": 1, "_completed_steps": 1, "_failed_steps": 1},
    {"_attempted_steps": 2, "_failed_steps": 2},
)


@pytest.mark.parametrize("changes", CORRUPTIONS)
def test_should_refuse_bypassed_invalid_state_on_every_public_read_and_mutation(changes):
    # Deliberate private corruption checks admission, rather than a reset feature.
    quota = PhysicalInteractionQuota(12)
    for name, value in changes.items():
        object.__setattr__(quota, name, value)
    for read in (
        lambda: quota.maximum_steps,
        lambda: quota.attempted_steps,
        lambda: quota.completed_steps,
        lambda: quota.failed_steps,
        lambda: quota.remaining_steps,
        quota.snapshot,
        quota.begin_step,
        quota.complete_step,
        quota.fail_step,
    ):
        with pytest.raises(ValueError):
            read()


def test_should_not_let_caller_mutation_of_a_snapshot_rewrite_quota_accounting():
    quota = PhysicalInteractionQuota(12)
    snapshot = quota.snapshot()
    object.__setattr__(snapshot, "maximum_steps", 4096)
    object.__setattr__(snapshot, "attempted_steps", 0)
    quota.begin_step()
    quota.fail_step()
    assert asdict(quota.snapshot()) == counts(12, 1, 0, 1)
    with pytest.raises(ValueError):
        quota.begin_step()
