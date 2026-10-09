"""Bounded coordination costs/host relationships; no IO or live lease authority."""

from dataclasses import dataclass
from typing import Protocol

from src.core.recovery_authority import AuthorityRecord, validate_authority_record
from src.core.recovery_observation import RecoveryHostObservation, RecoveryProcessProbe


class RecoveryRegistration(RecoveryProcessProbe, Protocol):
    def close(self) -> None: ...


@dataclass(frozen=True)
class RecoveryCosts:
    updates: int = 0
    copy_bytes: int = 0
    grants: int = 0
    checkpoint_attempts: int = 0

    def __post_init__(self) -> None:
        for value in (self.updates, self.copy_bytes, self.grants, self.checkpoint_attempts):
            if type(value) is not int or not 0 <= value < 2**63:
                raise ValueError("coordinator costs require bounded exact nonnegative integers")
        if self.updates > 1:
            raise ValueError("coordinator cannot batch uncertain updates")


def validate_coordination_observation(
    record: AuthorityRecord,
    observation: RecoveryHostObservation,
    previous_ns: int,
    previous_peak: int,
    *,
    ended: bool,
    pending: bool,
) -> None:
    validate_authority_record(record)
    if type(observation) is not RecoveryHostObservation:
        raise ValueError("coordinator requires exact trusted host observation")
    RecoveryHostObservation.__post_init__(observation)
    for value in (previous_ns, previous_peak):
        if type(value) is not int or not 0 <= value < 2**63:
            raise ValueError("coordinator high water requires bounded exact values")
    if type(ended) is not bool or type(pending) is not bool:
        raise ValueError("coordinator observation mode requires exact flags")
    metadata = record.metadata
    if (
        observation.observer != record.anchor
        or observation.previous_owner != record.worker
        or observation.previous_owner_ended is not ended
        or observation.clock_epoch != metadata.clock_epoch
    ):
        raise ValueError("coordinator observation does not bind registered original authority")
    if observation.now_ns < max(previous_ns, metadata.observed_ns):
        raise ValueError("coordinator observed clock went backwards")
    if observation.peak_rss_bytes < previous_peak:
        raise ValueError("coordinator observed RSS high water went backwards")
    if observation.now_ns - metadata.started_ns >= metadata.limits.max_elapsed_ns:
        raise ValueError("coordinator original elapsed allowance exhausted")
    if (
        max(observation.peak_rss_bytes, metadata.usage.peak_rss_bytes)
        > metadata.limits.max_rss_bytes
    ):
        raise ValueError("coordinator original RSS allowance exhausted")
    if metadata.stopped or (metadata.uncertain_work and not pending):
        raise ValueError("coordinator stopped or uncertain work cannot publish or retry")
    if pending and (
        not metadata.uncertain_work
        or metadata.usage.updates_admitted != metadata.usage.updates_completed + 1
    ):
        raise ValueError("coordinator pending mode requires its single reserved uncertain update")
