"""Single-session private SQLite metadata CAS; no native restore or owner lease.

Requires independently known live-coordinator authority floor, trusted local disk
and OS observations supplied by the coordinator. No automatic retry/bootstrap.
"""

from pathlib import Path
import sqlite3
from contextlib import contextmanager
from typing import Callable, Iterator

from src.core.recovery_authority import (
    AuthorityRecord,
    AuthorityChange,
    validate_authority_change,
    validate_authority_record,
    validate_authority_floor,
)
from src.infra.recovery_authority_codec import encode_authority, decode_authority, MAX_RECORD_BYTES
from src.core.recovery_reporting import (
    AuthorityReport,
    FailureWitness,
    plan_report,
    select_report_observation,
    validate_authority_report,
    validate_failure_witness,
    witnessed_terminal_states,
)
from src.core.recovery_observation import RecoveryHostObservation
from src.core.recovery_publication import RecoveryPublicationLease, validate_publication_observation
from src.infra.sqlite_recovery_lock import authority_writer_lock

SCHEMA = (
    "CREATE TABLE authority(singleton INTEGER PRIMARY KEY CHECK(singleton=1), record BLOB NOT NULL)"
)
MAX_DB_BYTES = 256 * 1024


def _configure(connection: sqlite3.Connection) -> sqlite3.Connection:
    try:
        if connection.execute("PRAGMA page_size").fetchone() != (4096,):
            raise ValueError("authority requires bounded 4096-byte pages")
        if connection.execute("PRAGMA journal_mode").fetchone() != ("delete",):
            raise ValueError("authority requires DELETE journal policy")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA max_page_count=64")
        if connection.execute("PRAGMA synchronous").fetchone() != (2,):
            raise ValueError("authority requires FULL synchronous policy")
        return connection
    except BaseException:
        connection.close()
        raise


def _connect(path: Path) -> sqlite3.Connection:
    if path.exists() and path.stat().st_size > MAX_DB_BYTES:
        raise ValueError("authority database exceeds byte bound")
    return _configure(
        sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=0, isolation_level=None)
    )


def _read(connection: sqlite3.Connection) -> AuthorityRecord:
    if connection.execute("PRAGMA user_version").fetchone() != (1,):
        raise ValueError("unsupported authority database schema version")
    schema = connection.execute(
        "SELECT sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
    ).fetchall()
    if schema != [(SCHEMA,)]:
        raise ValueError("authority database schema differs")
    row = connection.execute(
        "SELECT singleton,typeof(record),length(record) FROM authority"
    ).fetchall()
    if len(row) != 1 or row[0][0] != 1 or row[0][1] != "blob" or row[0][2] > MAX_RECORD_BYTES:
        raise ValueError("authority database requires one bounded exact record")
    return decode_authority(
        connection.execute("SELECT record FROM authority WHERE singleton=1").fetchone()[0]
    )


class SqliteRecoveryJournal:
    def __init__(self, path: str | Path, known_authority: AuthorityRecord) -> None:
        validate_authority_record(known_authority)
        self.path = Path(path).resolve()
        self._floor = known_authority
        self._failed = False

    @classmethod
    def create(cls, path: str | Path, initial: AuthorityRecord) -> "SqliteRecoveryJournal":
        validate_authority_record(initial)
        raw = encode_authority(initial)
        resolved = Path(path).resolve()
        with resolved.open("xb"):
            pass
        connection = _connect(resolved)
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(SCHEMA)
            connection.execute("PRAGMA user_version=1")
            connection.execute("INSERT INTO authority VALUES(1,?)", (raw,))
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        return cls(resolved, initial)

    def _require_ready(self) -> None:
        if self._failed:
            raise ValueError(
                "authority adapter previously failed; explicit reconciliation required"
            )

    def _open(self) -> sqlite3.Connection:
        try:
            return _connect(self.path)
        except BaseException:
            self._failed = True
            raise

    def read(self) -> AuthorityRecord:
        self._require_ready()
        connection = self._open()
        try:
            current = _read(connection)
            validate_authority_floor(self._floor, current)
            self._floor = current
            return current
        except BaseException:
            self._failed = True
            raise
        finally:
            connection.close()

    def advance(self, change: AuthorityChange) -> bool:
        self._require_ready()
        validate_authority_change(change)
        return self._swap(change.expected, change.proposed)

    def report(self, report: AuthorityReport) -> bool:
        self._require_ready()
        validate_authority_report(report)
        return self._swap(report.expected, report.proposed)

    def _swap(self, before: AuthorityRecord, after: AuthorityRecord) -> bool:
        try:
            with authority_writer_lock(self.path):
                return self._swap_unlocked(before, after)
        except BaseException:
            self._failed = True
            raise

    def _swap_unlocked(self, before: AuthorityRecord, after: AuthorityRecord) -> bool:
        expected, proposed = encode_authority(before), encode_authority(after)
        connection = self._open()
        try:
            connection.execute("BEGIN IMMEDIATE")
            current = _read(connection)
            validate_authority_floor(self._floor, current)
            if current != before:
                connection.execute("ROLLBACK")
                self._floor = current
                return False
            changed = connection.execute(
                "UPDATE authority SET record=? WHERE singleton=1 AND record=?", (proposed, expected)
            )
            if changed.rowcount != 1:
                raise ValueError("authority compare-and-swap row changed")
            connection.execute("COMMIT")
            self._floor = after
            return True
        except BaseException:
            self._failed = True
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @classmethod
    def reconcile_terminal(cls, path: str | Path, witness: FailureWitness) -> AuthorityRecord:
        resolved = Path(path).resolve()
        validate_failure_witness(witness)
        with authority_writer_lock(resolved):
            return cls._reconcile_terminal_unlocked(resolved, witness)

    @classmethod
    def _reconcile_terminal_unlocked(
        cls, path: str | Path, witness: FailureWitness
    ) -> AuthorityRecord:
        """Explicit surviving-coordinator witness; only stop, never resumed work."""
        validate_failure_witness(witness)
        allowed = witnessed_terminal_states(witness)
        connection = _connect(Path(path).resolve())
        try:
            connection.execute("BEGIN IMMEDIATE")
            current = _read(connection)
            if current not in allowed:
                raise ValueError("terminal reconciliation requires exact witnessed disk state")
            observation = select_report_observation(current, witness.observation)
            if current.metadata.stopped and (
                observation is None
                or (
                    observation.now_ns == current.metadata.observed_ns
                    and observation.peak_rss_bytes == current.metadata.usage.peak_rss_bytes
                )
            ):
                connection.execute("ROLLBACK")
                return current
            report = plan_report(current, observation, terminal=True)
            changed = connection.execute(
                "UPDATE authority SET record=? WHERE singleton=1 AND record=?",
                (encode_authority(report.proposed), encode_authority(current)),
            )
            if changed.rowcount != 1:
                raise ValueError("terminal reconciliation compare-and-swap changed")
            connection.execute("COMMIT")
            return report.proposed
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @contextmanager
    def publication_guard(
        self,
        expected: AuthorityRecord,
        observe: Callable[[], RecoveryHostObservation],
    ) -> Iterator[None]:
        self._require_ready()
        try:
            with authority_writer_lock(self.path):
                with self._publication_guard_unlocked(expected, observe):
                    yield
        except BaseException:
            self._failed = True
            raise

    @contextmanager
    def _publication_guard_unlocked(
        self,
        expected: AuthorityRecord,
        observe: Callable[[], RecoveryHostObservation],
    ) -> Iterator[None]:
        """Hold a private-database writer reservation across a trusted callback.

        Why this: full-record CAS alone leaves a gap between validation and
        publication. Every supported writer uses BEGIN IMMEDIATE on this file.
        This transaction never mutates authority or refunds committed work.
        """
        self._require_ready()
        validate_authority_record(expected)
        if expected.metadata.stopped or expected.metadata.uncertain_work:
            raise ValueError("stopped or uncertain authority cannot publish")
        if not callable(observe):
            raise ValueError("publication requires trusted fresh observation callback")
        connection = self._open()
        try:
            connection.execute("BEGIN IMMEDIATE")
            current = _read(connection)
            validate_authority_floor(self._floor, current)
            if current != expected:
                raise ValueError("publication persisted authority differs from independent witness")
            before = observe()
            validate_publication_observation(expected, before)
            yield
            self._require_ready()
            after = observe()
            validate_publication_observation(expected, after, before)
            if _read(connection) != expected:
                raise ValueError("publication guarded authority changed")
            connection.execute("ROLLBACK")
            self._floor = expected
        except BaseException:
            self._failed = True
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @contextmanager
    def publication_lease(self, expected: AuthorityRecord) -> Iterator[RecoveryPublicationLease]:
        from src.infra.sqlite_recovery_publication import publication_lease

        with publication_lease(self, expected) as lease:
            yield lease
