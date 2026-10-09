"""Monotone observed/terminal facts and independent terminal-only witnesses.

No IO, live lease, native completion or restart admission. Unknown facts may stop
work using the last known stamp; they never authorize resumed work.
"""

from dataclasses import dataclass, replace
from typing import Protocol

from src.core.recovery_authority import (
    AuthorityChange,
    AuthorityRecord,
    RecoveryAuthorityPort,
    validate_authority_change,
    validate_authority_record,
)
from src.core.recovery_observation import RecoveryHostObservation


@dataclass(frozen=True)
class AuthorityReport:
    expected: AuthorityRecord
    proposed: AuthorityRecord
    observation: RecoveryHostObservation | None


def _require_report_observation(
    record: AuthorityRecord, observation: RecoveryHostObservation
) -> None:
    if type(observation) is not RecoveryHostObservation:
        raise ValueError("authority report requires exact host facts")
    RecoveryHostObservation.__post_init__(observation)
    metadata = record.metadata
    if (
        observation.observer != record.anchor
        or observation.previous_owner != record.worker
        or observation.clock_epoch != metadata.clock_epoch
    ):
        raise ValueError("authority report host facts do not bind original registered authority")
    if (
        observation.now_ns < metadata.observed_ns
        or observation.peak_rss_bytes < metadata.usage.peak_rss_bytes
    ):
        raise ValueError("authority report clock/RSS observations cannot regress")


def validate_authority_report(report: object) -> None:
    if type(report) is not AuthorityReport:
        raise ValueError("authority reporting requires an exact typed report")
    old, new = report.expected, report.proposed
    validate_authority_record(old)
    validate_authority_record(new)
    before, after = old.metadata, new.metadata
    unchanged = replace(
        old,
        metadata=replace(
            before,
            sequence=after.sequence,
            observed_ns=after.observed_ns,
            stopped=after.stopped,
            usage=replace(before.usage, peak_rss_bytes=after.usage.peak_rss_bytes),
        ),
    )
    if unchanged != new or after.sequence != before.sequence + 1:
        raise ValueError("authority report cannot change original ownership/policy/caps/spent work")
    if before.stopped and not after.stopped:
        raise ValueError("authority report cannot reopen stopped history")
    observation = report.observation
    if observation is None:
        if (
            not after.stopped
            or after.observed_ns != before.observed_ns
            or after.usage != before.usage
        ):
            raise ValueError("unavailable host facts may only stop at the last known stamp")
        return
    _require_report_observation(old, observation)
    if (
        after.observed_ns != observation.now_ns
        or after.usage.peak_rss_bytes != observation.peak_rss_bytes
    ):
        raise ValueError("authority report must preserve the exact observed high water")
    if not after.stopped and (
        observation.previous_owner_ended is not False
        or observation.now_ns - before.started_ns >= before.limits.max_elapsed_ns
        or observation.peak_rss_bytes > before.limits.max_rss_bytes
    ):
        raise ValueError("ended/over-cap observations require terminal authority")


def plan_report(
    record: AuthorityRecord, observation: RecoveryHostObservation | None, *, terminal: bool = False
) -> AuthorityReport:
    validate_authority_record(record)
    if type(terminal) is not bool:
        raise ValueError("authority terminal mode requires an exact flag")
    if observation is not None:
        _require_report_observation(record, observation)
    metadata = record.metadata
    proposed = replace(
        record,
        metadata=replace(
            metadata,
            sequence=metadata.sequence + 1,
            observed_ns=metadata.observed_ns if observation is None else observation.now_ns,
            stopped=metadata.stopped or terminal,
            usage=replace(
                metadata.usage,
                peak_rss_bytes=metadata.usage.peak_rss_bytes
                if observation is None
                else observation.peak_rss_bytes,
            ),
        ),
    )
    report = AuthorityReport(record, proposed, observation)
    validate_authority_report(report)
    return report


def select_report_observation(
    record: AuthorityRecord, observation: RecoveryHostObservation | None
) -> RecoveryHostObservation | None:
    """Keep only identity-bound monotone facts; never invent missing observations."""
    if observation is None:
        return None
    try:
        _require_report_observation(record, observation)
    except ValueError:
        return None
    return observation


@dataclass(frozen=True)
class FailureWitness:
    expected: AuthorityRecord
    attempted: AuthorityChange | AuthorityReport | None = None
    terminal_attempt: AuthorityReport | None = None
    observation: RecoveryHostObservation | None = None

    def __post_init__(self) -> None:
        validate_failure_witness(self)


def validate_failure_witness(witness: object) -> None:
    if type(witness) is not FailureWitness:
        raise ValueError("terminal reconciliation requires exact independent failure witness")
    validate_authority_record(witness.expected)
    states = [witness.expected]
    attempt = witness.attempted
    if attempt is not None:
        if type(attempt) is AuthorityChange:
            validate_authority_change(attempt)
        else:
            validate_authority_report(attempt)
        if attempt.expected != witness.expected:
            raise ValueError("failure witness attempt is not bound to original known state")
        states.append(attempt.proposed)
    terminal = witness.terminal_attempt
    if terminal is not None:
        validate_authority_report(terminal)
        if terminal.expected not in states or not terminal.proposed.metadata.stopped:
            raise ValueError("failure witness terminal attempt is not bound to known states")
        states.append(terminal.proposed)
    if witness.observation is not None and not any(
        select_report_observation(state, witness.observation) is not None for state in states
    ):
        raise ValueError("failure witness observation is unbound or regressing")


def witnessed_terminal_states(witness: FailureWitness) -> tuple[AuthorityRecord, ...]:
    validate_failure_witness(witness)
    states = [witness.expected]
    if witness.attempted is not None:
        states.append(witness.attempted.proposed)
    if witness.terminal_attempt is not None:
        states.append(witness.terminal_attempt.proposed)
    terminals = [
        plan_report(
            state, select_report_observation(state, witness.observation), terminal=True
        ).proposed
        for state in states
    ]
    return tuple(states + terminals)


class TerminalReportingFailure(ValueError):
    def __init__(self, witness: FailureWitness) -> None:
        super().__init__(
            "durable terminal authority unconfirmed; independent reconciliation required"
        )
        self.witness = witness


class RecoveryReportingPort(RecoveryAuthorityPort, Protocol):
    def report(self, report: AuthorityReport) -> bool: ...
