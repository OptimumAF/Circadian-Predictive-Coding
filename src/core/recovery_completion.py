"""Explicit completion of one original native-only update reservation.

A trusted surviving coordinator validates a complete parent-owned checkpoint
before proposing completion. Why a separate transition: ordinary floor checks
must still refuse disk-asserted manifest changes or cleared uncertainty. These
records bind accounting and known checkpoint digests; they do not prove native
execution, authenticate bytes, publish an actor, retry work or restore authority.
No IO, native models, reporting dependency or automatic reconciliation lives here.
"""

from dataclasses import dataclass, replace
from typing import Protocol

from src.core.recovery_admission import RecoveryManifest
from src.core.recovery_authority import (
    AuthorityChange,
    AuthorityRecord,
    validate_authority_change,
    validate_authority_floor,
    validate_authority_record,
)
from src.core.recovery_coordination import validate_coordination_observation
from src.core.recovery_observation import RecoveryHostObservation


@dataclass(frozen=True)
class AuthorityCompletion:
    admission: AuthorityChange
    expected: AuthorityRecord
    proposed: AuthorityRecord
    observation: RecoveryHostObservation


def _require_admission(admission: AuthorityChange, record: AuthorityRecord) -> None:
    validate_authority_change(admission)
    validate_authority_record(record)
    original, pending = admission.expected.metadata, admission.proposed.metadata
    if (
        original.usage.updates_completed != original.usage.updates_admitted
        or pending.usage.updates_admitted != original.usage.updates_admitted + 1
        or not pending.uncertain_work
        or original.manifest.payload_sha256 != original.manifest.native_sha256
    ):
        raise ValueError("completion requires one original native-only update reservation")
    # The generic floor remains restrictive. Pending reports may advance only
    # sequence/time/RSS; they cannot change the original admission or spent costs.
    validate_authority_floor(admission.proposed, record)
    current = record.metadata
    reported = replace(
        admission.proposed,
        metadata=replace(
            pending,
            sequence=current.sequence,
            observed_ns=current.observed_ns,
            usage=replace(pending.usage, peak_rss_bytes=current.usage.peak_rss_bytes),
        ),
    )
    if reported != record:
        raise ValueError(
            "pending completion record differs from original reservation/report fields"
        )


def _require_manifest(previous: RecoveryManifest, manifest: RecoveryManifest) -> None:
    if type(manifest) is not RecoveryManifest:
        raise ValueError("native completion requires an exact RecoveryManifest")
    RecoveryManifest.__post_init__(manifest)
    if (
        manifest.native_sha256 != manifest.payload_sha256
        or replace(
            previous, native_sha256=manifest.native_sha256, payload_sha256=manifest.payload_sha256
        )
        != manifest
    ):
        raise ValueError("completion may change only the equal native/payload digest pair")


def _validate_inputs(admission, record, observation, manifest) -> None:
    _require_admission(admission, record)
    metadata = record.metadata
    validate_coordination_observation(
        record,
        observation,
        metadata.observed_ns,
        metadata.usage.peak_rss_bytes,
        ended=False,
        pending=True,
    )
    _require_manifest(metadata.manifest, manifest)


def _completed_record(record, observation, manifest) -> AuthorityRecord:
    metadata = record.metadata
    return replace(
        record,
        metadata=replace(
            metadata,
            sequence=metadata.sequence + 1,
            observed_ns=observation.now_ns,
            manifest=replace(manifest),
            uncertain_work=False,
            usage=replace(
                metadata.usage,
                updates_completed=metadata.usage.updates_completed + 1,
                peak_rss_bytes=observation.peak_rss_bytes,
            ),
        ),
    )


def validate_authority_completion(change: object) -> None:
    if type(change) is not AuthorityCompletion:
        raise ValueError("native completion requires an exact AuthorityCompletion")
    if vars(change).keys() != {"admission", "expected", "proposed", "observation"}:
        raise ValueError("native completion differs from the complete supported record schema")
    validate_authority_record(change.proposed)
    manifest = change.proposed.metadata.manifest
    _validate_inputs(change.admission, change.expected, change.observation, manifest)
    if _completed_record(change.expected, change.observation, manifest) != change.proposed:
        raise ValueError("completion differs from the exact completed accounting transition")


def plan_completion(
    admission: AuthorityChange,
    record: AuthorityRecord,
    observation: RecoveryHostObservation,
    manifest: RecoveryManifest,
) -> AuthorityCompletion:
    """Bind a trusted validated checkpoint to this live original pending update."""
    _validate_inputs(admission, record, observation, manifest)
    completion = AuthorityCompletion(
        admission, record, _completed_record(record, observation, manifest), observation
    )
    validate_authority_completion(completion)
    return completion


class RecoveryCompletionPort(Protocol):
    def complete(self, completion: AuthorityCompletion) -> bool: ...
