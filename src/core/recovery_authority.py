"""Monotone coordinator metadata/port; no IO, payloads or live owner capability.

Trusted OS observations and independently known original state are required.
Reservations remain uncertain until a future native completion/codec protocol.
"""

from dataclasses import dataclass, replace
from typing import Protocol

from src.core.recovery_admission import RecoveryMetadata
from src.core.recovery_observation import (
    RecoveryHostObservation,
    RecoveryProcessIdentity,
    anchored_clock_epoch,
    validate_process_identity,
)


@dataclass(frozen=True)
class AuthorityRecord:
    metadata: RecoveryMetadata
    anchor: RecoveryProcessIdentity
    worker: RecoveryProcessIdentity

    def __post_init__(self) -> None:
        if type(self.metadata) is not RecoveryMetadata:
            raise ValueError("authority requires exact recovery metadata")
        RecoveryMetadata.__post_init__(self.metadata)
        validate_process_identity(self.anchor)
        validate_process_identity(self.worker)
        if self.worker == self.anchor or self.metadata.clock_epoch != anchored_clock_epoch(
            self.anchor
        ):
            raise ValueError("authority requires distinct worker and original anchored clock")
        used, limits = self.metadata.usage, self.metadata.limits
        if used.updates_completed > used.updates_admitted:
            raise ValueError("authority completed work exceeds admissions")
        if used.updates_completed < used.updates_admitted and not self.metadata.uncertain_work:
            raise ValueError("unfinished admissions require uncertain work")
        for value, cap in (
            (used.updates_admitted, limits.max_updates),
            (used.copied_bytes, limits.max_copy_bytes),
            (used.grants, limits.max_grants),
            (used.checkpoint_attempts, limits.max_checkpoint_attempts),
        ):
            if value > cap:
                raise ValueError("original authority allowance exceeded")
        # Retain a measured RSS overshoot only in a terminal record; no work is granted.
        if used.peak_rss_bytes > limits.max_rss_bytes and not self.metadata.stopped:
            raise ValueError("original authority RSS allowance exceeded")


def validate_authority_record(record: object) -> None:
    if type(record) is not AuthorityRecord:
        raise ValueError("authority requires exact record")
    AuthorityRecord.__post_init__(record)


@dataclass(frozen=True)
class AuthorityChange:
    expected: AuthorityRecord
    proposed: AuthorityRecord
    observation: RecoveryHostObservation


def _require_original(old: AuthorityRecord, new: AuthorityRecord) -> None:
    if old.anchor != new.anchor:
        raise ValueError("original authority anchor differs")
    for name in (
        "session_id",
        "schema_version",
        "clock_epoch",
        "started_ns",
        "limits",
        "manifest",
        "event_tick",
    ):
        if getattr(old.metadata, name) != getattr(new.metadata, name):
            raise ValueError(f"original authority {name} differs")


def _require_host(record: AuthorityRecord, observation: RecoveryHostObservation) -> None:
    if type(observation) is not RecoveryHostObservation:
        raise ValueError("authority requires exact host observation")
    RecoveryHostObservation.__post_init__(observation)
    metadata = record.metadata
    if observation.observer != record.anchor or observation.previous_owner != record.worker:
        raise ValueError("authority observation does not bind original coordinator/worker")
    if observation.clock_epoch != metadata.clock_epoch or observation.now_ns < metadata.observed_ns:
        raise ValueError("authority clock epoch changed or time went backwards")
    if observation.now_ns - metadata.started_ns >= metadata.limits.max_elapsed_ns:
        raise ValueError("original elapsed authority allowance exhausted")
    if (
        max(observation.peak_rss_bytes, metadata.usage.peak_rss_bytes)
        > metadata.limits.max_rss_bytes
    ):
        raise ValueError("original RSS authority allowance exhausted")
    if metadata.stopped or metadata.uncertain_work:
        raise ValueError("stopped or uncertain authority cannot retry or hand off")


def validate_authority_change(change: object) -> None:
    if type(change) is not AuthorityChange:
        raise ValueError("authority requires exact change")
    old, new, observation = change.expected, change.proposed, change.observation
    validate_authority_record(old)
    validate_authority_record(new)
    _require_original(old, new)
    _require_host(old, observation)
    previous, proposed = old.metadata, new.metadata
    if proposed.sequence != previous.sequence + 1 or proposed.observed_ns != observation.now_ns:
        raise ValueError("authority must advance exact next sequence and observed clock")
    if (
        proposed.stopped != previous.stopped
        or proposed.usage.updates_completed != previous.usage.updates_completed
    ):
        raise ValueError("unproved stop/completion transition is unsupported")
    if proposed.usage.peak_rss_bytes != max(
        previous.usage.peak_rss_bytes, observation.peak_rss_bytes
    ):
        raise ValueError("authority absolute RSS high water differs")
    handoff = new.worker != old.worker or proposed.owner_id != previous.owner_id
    if handoff:
        _require_handoff(old, new, observation)
    else:
        _require_reservation(old, new, observation)


def _require_handoff(
    old: AuthorityRecord, new: AuthorityRecord, observation: RecoveryHostObservation
) -> None:
    before, after = old.metadata, new.metadata
    if (
        observation.previous_owner_ended is not True
        or new.worker == old.worker
        or after.owner_id == before.owner_id
        or after.owner_epoch != before.owner_epoch + 1
        or after.uncertain_work
    ):
        raise ValueError("handoff requires ended registered owner and exact next owner epoch")
    expected = replace(
        before.usage,
        checkpoint_attempts=before.usage.checkpoint_attempts + 1,
        peak_rss_bytes=after.usage.peak_rss_bytes,
    )
    if after.usage != expected:
        raise ValueError("handoff cannot renew or alter spent resources")


def _require_reservation(
    old: AuthorityRecord, new: AuthorityRecord, observation: RecoveryHostObservation
) -> None:
    before, after = old.metadata, new.metadata
    if observation.previous_owner_ended is not False or after.owner_epoch != before.owner_epoch:
        raise ValueError("reservation requires original live owner generation")
    increments = {
        name: getattr(after.usage, name) - getattr(before.usage, name)
        for name in ("updates_admitted", "copied_bytes", "grants", "checkpoint_attempts")
    }
    if any(value < 0 for value in increments.values()) or increments["updates_admitted"] > 1:
        raise ValueError("authority counters cannot refund or batch uncertain updates")
    if after.uncertain_work != (increments["updates_admitted"] > 0):
        raise ValueError("admitted native work must remain uncertain before completion")


def plan_reservation(
    record: AuthorityRecord,
    observation: RecoveryHostObservation,
    *,
    updates: int = 0,
    copy_bytes: int = 0,
    grants: int = 0,
    checkpoint_attempts: int = 0,
) -> AuthorityChange:
    validate_authority_record(record)
    _require_host(record, observation)
    for value in (updates, copy_bytes, grants, checkpoint_attempts):
        if type(value) is not int or not 0 <= value < 2**63:
            raise ValueError("reservations require bounded nonnegative exact integers")
    old = record.metadata
    used = old.usage
    proposed = replace(
        record,
        metadata=replace(
            old,
            sequence=old.sequence + 1,
            observed_ns=observation.now_ns,
            uncertain_work=updates > 0,
            usage=replace(
                used,
                updates_admitted=used.updates_admitted + updates,
                copied_bytes=used.copied_bytes + copy_bytes,
                grants=used.grants + grants,
                checkpoint_attempts=used.checkpoint_attempts + checkpoint_attempts,
                peak_rss_bytes=max(used.peak_rss_bytes, observation.peak_rss_bytes),
            ),
        ),
    )
    change = AuthorityChange(record, proposed, observation)
    validate_authority_change(change)
    return change


def plan_handoff(
    record: AuthorityRecord,
    observation: RecoveryHostObservation,
    worker: RecoveryProcessIdentity,
    owner_id: str,
) -> AuthorityChange:
    validate_authority_record(record)
    _require_host(record, observation)
    old = record.metadata
    proposed = replace(
        record,
        worker=worker,
        metadata=replace(
            old,
            sequence=old.sequence + 1,
            observed_ns=observation.now_ns,
            owner_epoch=old.owner_epoch + 1,
            owner_id=owner_id,
            usage=replace(
                old.usage,
                checkpoint_attempts=old.usage.checkpoint_attempts + 1,
                peak_rss_bytes=max(old.usage.peak_rss_bytes, observation.peak_rss_bytes),
            ),
        ),
    )
    change = AuthorityChange(record, proposed, observation)
    validate_authority_change(change)
    return change


def validate_authority_floor(known: AuthorityRecord, current: AuthorityRecord) -> None:
    """Known live coordinator witness; disk state alone cannot recreate this floor."""
    validate_authority_record(known)
    validate_authority_record(current)
    _require_original(known, current)
    old, new = known.metadata, current.metadata
    if new.sequence < old.sequence or (new.sequence == old.sequence and current != known):
        raise ValueError("authority disk rollback or same-sequence mutation")
    for name in ("owner_epoch", "observed_ns"):
        if getattr(new, name) < getattr(old, name):
            raise ValueError("authority rollback")
    for name in (
        "updates_admitted",
        "updates_completed",
        "copied_bytes",
        "grants",
        "checkpoint_attempts",
        "peak_rss_bytes",
    ):
        if getattr(new.usage, name) < getattr(old.usage, name):
            raise ValueError("authority rollback")
    if (old.uncertain_work and not new.uncertain_work) or (old.stopped and not new.stopped):
        raise ValueError("authority uncertain/stopped history cannot reopen")


class RecoveryAuthorityPort(Protocol):
    def read(self) -> AuthorityRecord: ...
    def advance(self, change: AuthorityChange) -> bool: ...
