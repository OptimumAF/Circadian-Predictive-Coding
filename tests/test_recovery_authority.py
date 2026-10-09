"""Pure authority changes from bounded trusted fake host observations."""

from dataclasses import replace
from typing import Callable

import pytest

from src.core.recovery_authority import (
    AuthorityRecord,
    plan_handoff,
    plan_reservation,
    validate_authority_change,
)
from src.core.recovery_admission import RecoveryMetadata
from src.core.recovery_observation import (
    RecoveryHostObservation,
    RecoveryProcessIdentity,
    anchored_clock_epoch,
)
from test_recovery_admission import fixture as metadata_fixture


def fixture():
    metadata, _ = metadata_fixture()
    anchor = RecoveryProcessIdentity(10, 100)
    worker = RecoveryProcessIdentity(20, 200)
    metadata = replace(metadata, clock_epoch=anchored_clock_epoch(anchor))
    record = AuthorityRecord(metadata, anchor, worker)
    observation = RecoveryHostObservation(
        metadata.clock_epoch, 600, 640, 640, anchor, worker, False
    )
    return record, observation


def test_should_charge_reservations_before_work_and_preserve_original_policy():
    record, observation = fixture()
    change = plan_reservation(
        record, observation, updates=1, copy_bytes=10, grants=1, checkpoint_attempts=1
    )
    proposed = change.proposed
    assert proposed.metadata.sequence == record.metadata.sequence + 1
    assert proposed.metadata.usage.updates_admitted == 4
    assert proposed.metadata.usage.updates_completed == 3
    assert proposed.metadata.usage.copied_bytes == 50
    assert proposed.metadata.usage.grants == 4
    assert proposed.metadata.usage.checkpoint_attempts == 3
    assert proposed.metadata.usage.peak_rss_bytes == 640
    assert proposed.metadata.uncertain_work
    assert proposed.metadata.manifest == record.metadata.manifest
    assert proposed.metadata.limits == record.metadata.limits


def test_should_preserve_all_spent_work_when_issuing_next_owner_generation():
    record, observation = fixture()
    worker = RecoveryProcessIdentity(30, 300)
    change = plan_handoff(
        record, replace(observation, previous_owner_ended=True), worker, "owner-b"
    )
    proposed = change.proposed
    assert proposed.worker == worker and proposed.anchor == record.anchor
    assert proposed.metadata.owner_epoch == record.metadata.owner_epoch + 1
    assert proposed.metadata.owner_id == "owner-b"
    assert proposed.metadata.started_ns == record.metadata.started_ns
    assert proposed.metadata.usage.updates_admitted == record.metadata.usage.updates_admitted
    assert proposed.metadata.usage.copied_bytes == record.metadata.usage.copied_bytes
    assert proposed.metadata.usage.checkpoint_attempts == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("updates", -1),
        ("updates", True),
        ("updates", 2),
        ("copy_bytes", -1),
        ("copy_bytes", 61),
        ("grants", 8),
        ("checkpoint_attempts", 3),
    ],
)
def test_should_refuse_bad_or_over_cap_reservations_without_changing_record(field, value):
    record, observation = fixture()
    original = replace(record)
    with pytest.raises(ValueError):
        plan_reservation(record, observation, **{field: value})
    assert record == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("clock_epoch", "other"),
        ("now_ns", 399),
        ("now_ns", 1100),
        ("rss_bytes", 1025),
        ("previous_owner", None),
        ("previous_owner_ended", True),
        ("observer", RecoveryProcessIdentity(99, 999)),
    ],
)
def test_should_refuse_unsupported_host_authority_before_reservation(field, value):
    record, observation = fixture()
    with pytest.raises(ValueError):
        if field == "previous_owner":
            observation = replace(observation, previous_owner=None, previous_owner_ended=None)
        elif field == "rss_bytes":
            observation = replace(observation, rss_bytes=value, peak_rss_bytes=value)
        else:
            observation = replace(observation, **{field: value})
        plan_reservation(record, observation, copy_bytes=1)


@pytest.mark.parametrize(
    "field",
    [
        "started_ns",
        "event_tick",
        "clock_epoch",
        "owner_epoch",
        "sequence",
        "manifest",
        "limits",
    ],
)
def test_should_refuse_forged_proposal_or_quota_renewal(field):
    record, observation = fixture()
    change = plan_reservation(record, observation, copy_bytes=1)
    metadata = change.proposed.metadata
    mutations: dict[str, Callable[[RecoveryMetadata], RecoveryMetadata]] = {
        "started_ns": lambda m: replace(m, started_ns=101),
        "event_tick": lambda m: replace(m, event_tick=13),
        "clock_epoch": lambda m: replace(m, clock_epoch="other"),
        "owner_epoch": lambda m: replace(m, owner_epoch=5),
        "sequence": lambda m: replace(m, sequence=m.sequence + 1),
        "manifest": lambda m: replace(m, manifest=replace(m.manifest, source_sha256="f" * 64)),
        "limits": lambda m: replace(m, limits=replace(m.limits, max_copy_bytes=101)),
    }
    with pytest.raises(ValueError):
        bad = replace(change.proposed, metadata=mutations[field](metadata))
        validate_authority_change(replace(change, proposed=bad))


@pytest.mark.parametrize(
    "field",
    [
        "copied_bytes",
        "grants",
        "checkpoint_attempts",
        "updates_admitted",
        "updates_completed",
        "peak_rss_bytes",
    ],
)
def test_should_refuse_refund_or_unproved_completion(field):
    record, observation = fixture()
    change = plan_reservation(record, observation, copy_bytes=1)
    usage = change.proposed.metadata.usage
    value = getattr(record.metadata.usage, field) - 1
    if field == "updates_completed":
        value += 2
    with pytest.raises(ValueError):
        bad = replace(
            change.proposed,
            metadata=replace(change.proposed.metadata, usage=replace(usage, **{field: value})),
        )
        validate_authority_change(replace(change, proposed=bad))


def test_should_block_ambiguous_native_work_and_never_refund_it():
    record, observation = fixture()
    pending = plan_reservation(record, observation, updates=1).proposed
    for action in (
        lambda: plan_reservation(pending, observation, updates=1),
        lambda: plan_handoff(
            pending,
            replace(observation, previous_owner_ended=True),
            RecoveryProcessIdentity(30, 300),
            "owner-b",
        ),
    ):
        with pytest.raises(ValueError, match="uncertain"):
            action()
    assert pending.metadata.usage.updates_admitted == 4


def test_should_refuse_handoff_without_matching_ended_registered_owner():
    record, observation = fixture()
    with pytest.raises(ValueError):
        plan_handoff(record, observation, RecoveryProcessIdentity(30, 300), "owner-b")
    with pytest.raises(ValueError):
        plan_handoff(
            record, replace(observation, previous_owner_ended=True), record.worker, "owner-b"
        )


def test_should_revalidate_corrupted_nested_records_without_opaque_hooks():
    record, observation = fixture()
    object.__setattr__(record.metadata.usage, "copied_bytes", False)
    with pytest.raises(ValueError):
        plan_reservation(record, observation)
    with pytest.raises(ValueError):
        validate_authority_change(object())


@pytest.mark.parametrize("resource", ["updates", "copy_bytes", "grants", "checkpoint_attempts"])
def test_should_preserve_zero_allowance_and_refuse_resource_admission(resource):
    record, observation = fixture()
    used, limits = record.metadata.usage, record.metadata.limits
    if resource == "updates":
        used = replace(used, updates_admitted=0, updates_completed=0)
        limits = replace(limits, max_updates=0)
    elif resource == "copy_bytes":
        used = replace(used, copied_bytes=0)
        limits = replace(limits, max_copy_bytes=0)
    elif resource == "grants":
        used = replace(used, grants=0)
        limits = replace(limits, max_grants=0)
    else:
        used = replace(used, checkpoint_attempts=0)
        limits = replace(limits, max_checkpoint_attempts=0)
    record = replace(record, metadata=replace(record.metadata, usage=used, limits=limits))
    assert plan_reservation(record, observation).proposed.metadata.limits == limits
    with pytest.raises(ValueError):
        plan_reservation(record, observation, **{resource: 1})
