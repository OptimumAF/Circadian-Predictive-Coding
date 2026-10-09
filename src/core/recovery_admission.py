"""Pure pre-payload durable recovery metadata relationships.

Inputs: bounded saved metadata and an independently trusted coordinator fence.
Output: validated accounting facts, never a restore/training capability. No IO,
hashing, native state, payload copies, authentication, lease or clock adapters.
"""

from dataclasses import dataclass
import re
from typing import Protocol, cast


class _ValidatedRecord(Protocol):
    def __post_init__(self) -> None: ...


def _counter(value: int, name: str, *, positive: bool = False) -> None:
    if type(value) is not int or not (int(positive) <= value < 2**63):
        raise ValueError(f"{name} requires a bounded exact integer")


def _identifier(value: str, name: str) -> None:
    if type(value) is not str or not value or len(value) > 128 or value.strip() != value:
        raise ValueError(f"{name} requires a bounded nonempty identifier")


def _flag(value: bool, name: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{name} requires an exact boolean")


@dataclass(frozen=True)
class RecoveryManifest:
    """Digests bind complete component bytes; their presence proves no authenticity."""

    source_sha256: str
    policy_sha256: str
    payload_sha256: str
    native_sha256: str
    inbox_sha256: str
    consolidation_sha256: str
    lifecycle_sha256: str
    actor_sha256: str
    sharing_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "source_sha256",
            "policy_sha256",
            "payload_sha256",
            "native_sha256",
            "inbox_sha256",
            "consolidation_sha256",
            "lifecycle_sha256",
            "actor_sha256",
            "sharing_sha256",
        ):
            value = getattr(self, name)
            if type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None:
                raise ValueError(f"{name} requires lowercase SHA-256")


@dataclass(frozen=True)
class RecoveryLimits:
    """Restricted initial policy: every cumulative or absolute ceiling is required."""

    max_updates: int
    max_elapsed_ns: int
    max_copy_bytes: int
    max_grants: int
    max_checkpoint_attempts: int
    max_rss_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "max_updates",
            "max_elapsed_ns",
            "max_copy_bytes",
            "max_grants",
            "max_checkpoint_attempts",
            "max_rss_bytes",
        ):
            _counter(getattr(self, name), name, positive=name == "max_rss_bytes")


@dataclass(frozen=True)
class RecoveryUsage:
    """Spent reservations survive failure; RSS is an observed absolute high water."""

    updates_admitted: int
    updates_completed: int
    copied_bytes: int
    grants: int
    checkpoint_attempts: int
    peak_rss_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "updates_admitted",
            "updates_completed",
            "copied_bytes",
            "grants",
            "checkpoint_attempts",
            "peak_rss_bytes",
        ):
            _counter(getattr(self, name), name, positive=name == "peak_rss_bytes")


def _record(value: object, expected: type[_ValidatedRecord]) -> None:
    # Revalidate all bounded fields: a frozen record is not a trust certificate.
    if type(value) is not expected:
        raise ValueError(f"recovery requires exact {expected.__name__}")
    expected.__post_init__(value)


@dataclass(frozen=True)
class RecoveryMetadata:
    schema_version: int
    session_id: str
    clock_epoch: str
    owner_id: str
    sequence: int
    owner_epoch: int
    started_ns: int
    observed_ns: int
    event_tick: int
    manifest: RecoveryManifest
    limits: RecoveryLimits
    usage: RecoveryUsage
    stopped: bool
    uncertain_work: bool

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported recovery schema_version")
        for name in ("session_id", "clock_epoch", "owner_id"):
            _identifier(getattr(self, name), name)
        for name in ("sequence", "owner_epoch", "started_ns", "observed_ns", "event_tick"):
            _counter(getattr(self, name), name)
        for name in ("stopped", "uncertain_work"):
            _flag(getattr(self, name), name)
        _record(self.manifest, RecoveryManifest)
        _record(self.limits, RecoveryLimits)
        _record(self.usage, RecoveryUsage)
        if self.observed_ns < self.started_ns:
            raise ValueError("recovery monotonic observation precedes original start")


@dataclass(frozen=True)
class RecoveryFence:
    """Trusted adapter facts after atomic next-owner acquisition, not caller claims.

    The future coordinator must independently persist expected, prove old owner
    ended, issue exactly the next epoch and retain a live lease through publish.
    Constructing this record does not perform or attest those operations.
    """

    expected: RecoveryMetadata
    owner_id: str
    owner_epoch: int
    clock_epoch: str
    now_ns: int
    rss_bytes: int
    previous_owner_ended: bool
    lease_live: bool

    def __post_init__(self) -> None:
        _record(self.expected, RecoveryMetadata)
        for name in ("owner_id", "clock_epoch"):
            _identifier(getattr(self, name), name)
        for name in ("owner_epoch", "now_ns", "rss_bytes"):
            _counter(getattr(self, name), name, positive=name == "rss_bytes")
        for name in ("previous_owner_ended", "lease_live"):
            _flag(getattr(self, name), name)


@dataclass(frozen=True)
class RecoveryAdmission:
    """Accounting observation only; an exhausted quota stays zero."""

    owner_id: str
    owner_epoch: int
    elapsed_ns: int
    peak_rss_bytes: int
    remaining_updates: int
    remaining_copy_bytes: int


def _require_fence(metadata: RecoveryMetadata, fence: RecoveryFence) -> None:
    if metadata != fence.expected:
        raise ValueError("saved metadata differs from independent authoritative record")
    if (
        not fence.previous_owner_ended
        or not fence.lease_live
        or fence.owner_id == metadata.owner_id
        or fence.owner_epoch != metadata.owner_epoch + 1
    ):
        raise ValueError("recovery requires one live next-owner fence and ended prior owner")
    if fence.clock_epoch != metadata.clock_epoch or fence.now_ns < metadata.observed_ns:
        raise ValueError("unsupported monotonic clock epoch or backwards recovery time")


def _require_work(metadata: RecoveryMetadata) -> None:
    usage, limits = metadata.usage, metadata.limits
    if metadata.stopped or metadata.uncertain_work:
        raise ValueError("stopped or uncertain candidate cannot resume")
    # Why: uncertain partial native work cannot be safely retried or refunded.
    if usage.updates_admitted != usage.updates_completed:
        raise ValueError("in-flight or inconsistent work requires separate reconciliation")
    for used, cap in (
        (usage.updates_admitted, limits.max_updates),
        (usage.copied_bytes, limits.max_copy_bytes),
        (usage.grants, limits.max_grants),
        (usage.checkpoint_attempts, limits.max_checkpoint_attempts),
    ):
        if used > cap:
            raise ValueError("original cumulative recovery quota exceeded")


def validate_recovery_admission(metadata: object, fence: object) -> RecoveryAdmission:
    """Refuse unsupported/changed relationships before any payload API is involved.

    Every authority, time, RSS and ownership fact must come from the independently
    trusted adapter. This pure check neither loads bytes nor authenticates facts.
    A future restore must recheck the fence before publishing the replacement.
    """
    _record(metadata, RecoveryMetadata)
    _record(fence, RecoveryFence)
    metadata = cast(RecoveryMetadata, metadata)
    fence = cast(RecoveryFence, fence)
    _require_fence(metadata, fence)
    _require_work(metadata)
    elapsed = fence.now_ns - metadata.started_ns
    peak = max(metadata.usage.peak_rss_bytes, fence.rss_bytes)
    if elapsed >= metadata.limits.max_elapsed_ns or peak > metadata.limits.max_rss_bytes:
        raise ValueError("original elapsed or absolute RSS recovery allowance exhausted")
    return RecoveryAdmission(
        fence.owner_id,
        fence.owner_epoch,
        elapsed,
        peak,
        metadata.limits.max_updates - metadata.usage.updates_admitted,
        metadata.limits.max_copy_bytes - metadata.usage.copied_bytes,
    )
