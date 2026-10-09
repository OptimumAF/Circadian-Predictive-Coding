"""Held private writer lease allowing durable observation-only reports.

No work admission, terminal retry, OS observation, model completion or side-effect
rollback. Trusted canonical private journal paths/retained coordinator are required.
"""

from contextlib import contextmanager
from threading import get_ident
from typing import Iterator, TYPE_CHECKING

from src.core.recovery_authority import AuthorityRecord, validate_authority_record
from src.core.recovery_publication import RecoveryPublicationLease, validate_publication_observation
from src.core.recovery_reporting import AuthorityReport, validate_authority_report
from src.infra.sqlite_recovery_lock import authority_writer_lock

if TYPE_CHECKING:
    from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal


class _ObservationLease:
    def __init__(self, journal: "SqliteRecoveryJournal", expected: AuthorityRecord) -> None:
        self._journal, self._current = journal, expected
        self._active, self._thread = True, get_ident()

    def _require_active(self) -> None:
        if not self._active or self._thread != get_ident():
            raise ValueError("publication lease inactive or called from another thread")
        self._journal._require_ready()

    def read(self) -> AuthorityRecord:
        self._require_active()
        current = self._journal.read()
        if current != self._current:
            raise ValueError("publication leased authority differs from independent witness")
        return current

    def report(self, report: AuthorityReport) -> bool:
        self._require_active()
        validate_authority_report(report)
        if report.proposed.metadata.stopped or report.proposed.metadata.uncertain_work:
            raise ValueError("publication lease permits only nonterminal observation reports")
        if self.read() != report.expected:
            return False
        if report.observation is None:
            raise ValueError("publication lease requires fresh registered facts")
        validate_publication_observation(self._current, report.observation)
        committed = self._journal._swap_unlocked(report.expected, report.proposed)
        if committed is True:
            self._current = report.proposed
        return committed


@contextmanager
def publication_lease(
    journal: "SqliteRecoveryJournal", expected: AuthorityRecord
) -> Iterator[RecoveryPublicationLease]:
    journal._require_ready()
    validate_authority_record(expected)
    if expected.metadata.stopped or expected.metadata.uncertain_work:
        raise ValueError("stopped or uncertain authority cannot publish")
    try:
        with authority_writer_lock(journal.path):
            if journal.read() != expected:
                raise ValueError("publication persisted authority differs from independent witness")
            lease = _ObservationLease(journal, expected)
            try:
                yield lease
                lease.read()
            finally:
                lease._active = False
    except BaseException:
        journal._failed = True
        raise
