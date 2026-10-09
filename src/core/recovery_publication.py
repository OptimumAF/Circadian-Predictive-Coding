"""Trusted publication guard port and pure host validation.

Inputs: independent original authority and fresh registered host facts.
No IO, authentication, native completion, callback preemption or side-effect undo.
"""

from typing import Callable, ContextManager, Protocol

from src.core.recovery_authority import AuthorityRecord, validate_authority_record
from src.core.recovery_coordination import validate_coordination_observation
from src.core.recovery_observation import RecoveryHostObservation
from src.core.recovery_reporting import AuthorityReport, RecoveryReportingPort


class RecoveryPublicationLease(Protocol):
    """Held publication ownership; only durable observation reports, never work."""

    def read(self) -> AuthorityRecord: ...

    def report(self, report: AuthorityReport) -> bool: ...


def validate_publication_observation(
    expected: object,
    observation: RecoveryHostObservation,
    previous: RecoveryHostObservation | None = None,
) -> None:
    # Validate the original record before reading its nested fields.
    if type(expected) is not AuthorityRecord:
        raise ValueError("publication requires exact independent authority")
    validate_authority_record(expected)
    if previous is not None:
        validate_publication_observation(expected, previous)
    validate_coordination_observation(
        expected,
        observation,
        expected.metadata.observed_ns if previous is None else previous.now_ns,
        expected.metadata.usage.peak_rss_bytes if previous is None else previous.peak_rss_bytes,
        ended=False,
        pending=False,
    )


class RecoveryLeasedPublicationPort(RecoveryReportingPort, Protocol):
    def publication_lease(
        self, expected: AuthorityRecord
    ) -> ContextManager[RecoveryPublicationLease]: ...


class RecoveryPublicationPort(RecoveryReportingPort, Protocol):
    def publication_guard(
        self,
        expected: AuthorityRecord,
        observe: Callable[[], RecoveryHostObservation],
    ) -> ContextManager[None]:
        """Exclude supported writers; fresh observations on entry and successful exit.

        The trusted observe callback must bind retained anchor/worker registrations.
        An exception closes/disables this adapter; no automatic retry or refunds.
        Callbacks cannot use this guard to advance/report the locked journal.
        """
        ...
