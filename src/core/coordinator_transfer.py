"""Explicit pure coordinator transfer and terminal-only ambiguity records.

Inputs must come from an independently trusted surviving supervisor, completed
native-only checkpoint and physical process observations on the SAME running
Windows interrupt-time domain. Why retain the absolute start: replacing a clock
anchor must never renew elapsed time or any cumulative allowance. The records
below validate relationships, not that continuous supervision or native work
actually happened. Disk bytes and dataclass construction confer no authority.
There is no IO, OS capability, restore, retry or live-floor bootstrap here.
"""

from dataclasses import dataclass, fields, replace
from typing import Any, cast

from src.core.recovery_admission import (
    RecoveryLimits,
    RecoveryManifest,
    RecoveryMetadata,
    RecoveryUsage,
    _identifier,
)
from src.core.recovery_authority import AuthorityRecord, validate_authority_record
from src.core.recovery_observation import (
    RecoveryHostObservation,
    RecoveryProcessIdentity,
    anchored_clock_epoch,
    validate_process_identity,
)


def _schema(value: object, kind: type[Any]) -> None:
    # Frozen dataclasses can still be tampered with; check before field access,
    # including the nested old records whose original validators omit schemas.
    if type(value) is not kind or vars(value).keys() != {
        field.name for field in fields(cast(Any, kind))
    }:
        raise ValueError(f"coordinator transfer requires the exact complete {kind.__name__}")


def _identity(value: RecoveryProcessIdentity) -> None:
    _schema(value, RecoveryProcessIdentity)
    validate_process_identity(value)


def _authority(value: AuthorityRecord) -> None:
    _schema(value, AuthorityRecord)
    metadata = value.metadata
    _schema(metadata, RecoveryMetadata)
    _schema(metadata.manifest, RecoveryManifest)
    _schema(metadata.limits, RecoveryLimits)
    _schema(metadata.usage, RecoveryUsage)
    _identity(value.anchor)
    _identity(value.worker)
    validate_authority_record(value)


def _host(value: RecoveryHostObservation) -> None:
    _schema(value, RecoveryHostObservation)
    _identity(value.observer)
    if value.previous_owner is not None:
        _identity(value.previous_owner)
    RecoveryHostObservation.__post_init__(value)


@dataclass(frozen=True)
class SupervisorRecoveryWitness:
    authority: AuthorityRecord
    supervisor: RecoveryProcessIdentity

    def __post_init__(self) -> None:
        _validate_witness(self)


def _validate_witness(witness: SupervisorRecoveryWitness) -> None:
    _schema(witness, SupervisorRecoveryWitness)
    _authority(witness.authority)
    _identity(witness.supervisor)
    old = witness.authority
    metadata = old.metadata
    if (
        metadata.stopped
        or metadata.uncertain_work
        or metadata.usage.updates_admitted != metadata.usage.updates_completed
        or metadata.manifest.payload_sha256 != metadata.manifest.native_sha256
        or witness.supervisor in (old.anchor, old.worker)
    ):
        raise ValueError(
            "transfer requires completed native-only authority and a distinct supervisor"
        )


@dataclass(frozen=True)
class CoordinatorTransferObservation:
    anchor_ended: RecoveryHostObservation
    worker_ended: RecoveryHostObservation
    replacement: RecoveryHostObservation
    supervisor_after: RecoveryHostObservation | None = None

    def __post_init__(self) -> None:
        _schema(self, CoordinatorTransferObservation)
        for observation in (self.anchor_ended, self.worker_ended, self.replacement):
            _host(observation)
        if self.supervisor_after is not None:
            _host(self.supervisor_after)


@dataclass(frozen=True)
class CoordinatorTransfer:
    witness: SupervisorRecoveryWitness
    expected: AuthorityRecord
    proposed: AuthorityRecord
    observation: CoordinatorTransferObservation

    def __post_init__(self) -> None:
        validate_coordinator_transfer(self)


def _validate_observations(
    witness: SupervisorRecoveryWitness, observation: CoordinatorTransferObservation
) -> None:
    _validate_witness(witness)
    _schema(observation, CoordinatorTransferObservation)
    CoordinatorTransferObservation.__post_init__(observation)
    old, supervisor = witness.authority, witness.supervisor
    anchor, worker, replacement = (
        observation.anchor_ended,
        observation.worker_ended,
        observation.replacement,
    )
    for ended, owner in ((anchor, old.anchor), (worker, old.worker)):
        if (
            ended.observer != supervisor
            or ended.previous_owner != owner
            or ended.previous_owner_ended is not True
            or ended.clock_epoch != anchored_clock_epoch(supervisor)
        ):
            raise ValueError("transfer requires supervisor-bound ended original coordinator/worker")
    if (
        replacement.previous_owner is None
        or replacement.previous_owner_ended is not False
        or replacement.clock_epoch != anchored_clock_epoch(replacement.observer)
    ):
        raise ValueError("transfer requires a new physical coordinator and live registered worker")
    identities = (
        old.anchor,
        old.worker,
        supervisor,
        replacement.observer,
        replacement.previous_owner,
    )
    if len(set(identities)) != 5:
        raise ValueError("transfer requires five distinct full process identities")
    if len({supervisor.pid, replacement.observer.pid, replacement.previous_owner.pid}) != 3:
        raise ValueError("simultaneously live transfer processes require distinct PIDs")
    metadata = old.metadata
    if not metadata.observed_ns <= anchor.now_ns <= worker.now_ns <= replacement.now_ns:
        raise ValueError("transfer observations must preserve original absolute monotone time")
    after = observation.supervisor_after
    if after is not None and (
        after.observer != supervisor
        or after.clock_epoch != anchored_clock_epoch(supervisor)
        or after.previous_owner != replacement.observer
        or after.previous_owner_ended is not False
        or after.now_ns < replacement.now_ns
    ):
        raise ValueError("final transfer observation must bind the supervisor and live new anchor")
    final_ns = replacement.now_ns if after is None else after.now_ns
    if final_ns - metadata.started_ns >= metadata.limits.max_elapsed_ns:
        raise ValueError("original elapsed coordinator transfer allowance exhausted")
    if _peak(witness, observation) > metadata.limits.max_rss_bytes:
        raise ValueError("original RSS coordinator transfer allowance exhausted")


def _peak(witness: SupervisorRecoveryWitness, observation: CoordinatorTransferObservation) -> int:
    return max(
        witness.authority.metadata.usage.peak_rss_bytes,
        observation.anchor_ended.peak_rss_bytes,
        observation.worker_ended.peak_rss_bytes,
        observation.replacement.peak_rss_bytes,
        0 if observation.supervisor_after is None else observation.supervisor_after.peak_rss_bytes,
    )


def _proposed(
    witness: SupervisorRecoveryWitness, observation: CoordinatorTransferObservation, owner_id: str
) -> AuthorityRecord:
    _identifier(owner_id, "new owner_id")
    old, replacement = witness.authority, observation.replacement
    if owner_id == old.metadata.owner_id:
        raise ValueError("coordinator transfer requires a distinct owner_id")
    metadata = old.metadata
    # Validation establishes non-None; this assertion documents the owned pair.
    assert replacement.previous_owner is not None
    return replace(
        old,
        anchor=replacement.observer,
        worker=replacement.previous_owner,
        metadata=replace(
            metadata,
            clock_epoch=replacement.clock_epoch,
            # A native after-RPC sample is a separate fact, not a relabeled
            # remote observation. Preserve its later time and measured peak.
            observed_ns=(
                replacement.now_ns
                if observation.supervisor_after is None
                else observation.supervisor_after.now_ns
            ),
            sequence=metadata.sequence + 1,
            owner_epoch=metadata.owner_epoch + 1,
            owner_id=owner_id,
            usage=replace(
                metadata.usage,
                checkpoint_attempts=metadata.usage.checkpoint_attempts + 1,
                peak_rss_bytes=_peak(witness, observation),
            ),
        ),
    )


def validate_coordinator_transfer(change: object) -> None:
    _schema(change, CoordinatorTransfer)
    assert isinstance(change, CoordinatorTransfer)
    _validate_observations(change.witness, change.observation)
    _authority(change.expected)
    _authority(change.proposed)
    if change.expected != change.witness.authority:
        raise ValueError("transfer expected state differs from independent supervisor witness")
    expected = _proposed(change.witness, change.observation, change.proposed.metadata.owner_id)
    if change.proposed != expected:
        raise ValueError("transfer differs from exact ownership/attempt/clock transition")


def plan_coordinator_transfer(
    witness: SupervisorRecoveryWitness, observation: CoordinatorTransferObservation, owner_id: str
) -> CoordinatorTransfer:
    """Plan from trusted same-domain observations without minting physical authority."""
    _validate_observations(witness, observation)
    return CoordinatorTransfer(
        witness, witness.authority, _proposed(witness, observation, owner_id), observation
    )


@dataclass(frozen=True)
class CoordinatorTransferFailure:
    transfer: CoordinatorTransfer
    observation: RecoveryHostObservation

    def __post_init__(self) -> None:
        validate_transfer_failure(self)


def validate_transfer_failure(failure: object) -> None:
    _schema(failure, CoordinatorTransferFailure)
    assert isinstance(failure, CoordinatorTransferFailure)
    validate_coordinator_transfer(failure.transfer)
    _host(failure.observation)
    transfer, observation = failure.transfer, failure.observation
    supervisor = transfer.witness.supervisor
    if (
        observation.observer != supervisor
        or observation.clock_epoch != anchored_clock_epoch(supervisor)
        or observation.previous_owner != transfer.proposed.anchor
        or type(observation.previous_owner_ended) is not bool
        or observation.now_ns < transfer.proposed.metadata.observed_ns
    ):
        raise ValueError(
            "terminal transfer observation must bind the live supervisor and new anchor"
        )


def transfer_failure_states(
    failure: CoordinatorTransferFailure,
) -> tuple[AuthorityRecord, AuthorityRecord]:
    """Keep either CAS outcome stopped, charging the attempted transfer only once."""
    validate_transfer_failure(failure)
    transfer, observation = failure.transfer, failure.observation
    attempts = transfer.expected.metadata.usage.checkpoint_attempts + 1
    peak = max(transfer.proposed.metadata.usage.peak_rss_bytes, observation.peak_rss_bytes)

    def terminal(base: AuthorityRecord) -> AuthorityRecord:
        metadata = base.metadata
        stopped = replace(
            base,
            metadata=replace(
                metadata,
                sequence=metadata.sequence + 1,
                observed_ns=observation.now_ns,
                stopped=True,
                usage=replace(metadata.usage, checkpoint_attempts=attempts, peak_rss_bytes=peak),
            ),
        )
        _authority(stopped)
        return stopped

    return terminal(transfer.expected), terminal(transfer.proposed)
