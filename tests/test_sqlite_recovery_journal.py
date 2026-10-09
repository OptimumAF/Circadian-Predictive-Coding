"""Local SQLite metadata transactions only; no live worker or native model."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import sqlite3
from threading import Barrier

import pytest

from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal
from src.infra.recovery_authority_codec import encode_authority, decode_authority
from src.core.recovery_authority import plan_reservation, plan_handoff
from src.core.recovery_observation import RecoveryProcessIdentity
from test_recovery_authority import fixture


def test_should_persist_before_work_and_reopen_with_independent_known_floor(tmp_path):
    record, observation = fixture()
    path = tmp_path / "authority.sqlite"
    journal = SqliteRecoveryJournal.create(path, record)
    change = plan_reservation(record, observation, copy_bytes=10, checkpoint_attempts=1)
    assert journal.advance(change)
    reopened = SqliteRecoveryJournal(path, record)
    assert reopened.read() == change.proposed
    assert not reopened.advance(change)
    assert reopened.read() == change.proposed
    assert path.stat().st_size <= 256 * 1024


def test_should_not_recreate_or_overwrite_missing_or_existing_authority(tmp_path):
    record, _ = fixture()
    path = tmp_path / "authority.sqlite"
    with pytest.raises(sqlite3.Error):
        SqliteRecoveryJournal(path, record).read()
    assert not path.exists()
    SqliteRecoveryJournal.create(path, record)
    with pytest.raises(FileExistsError):
        SqliteRecoveryJournal.create(path, record)


def test_should_fence_old_owner_after_atomic_handoff_without_quota_renewal(tmp_path):
    record, observation = fixture()
    journal = SqliteRecoveryJournal.create(tmp_path / "a.sqlite", record)
    handoff = plan_handoff(
        record,
        replace(observation, previous_owner_ended=True),
        RecoveryProcessIdentity(30, 300),
        "owner-b",
    )
    assert journal.advance(handoff)
    old = plan_reservation(record, observation, copy_bytes=1)
    assert not journal.advance(old)
    assert journal.read() == handoff.proposed


def test_should_allow_exactly_one_two_thread_cas_contender(tmp_path):
    record, observation = fixture()
    path = tmp_path / "a.sqlite"
    SqliteRecoveryJournal.create(path, record)
    barrier = Barrier(2)

    def contender():
        journal = SqliteRecoveryJournal(path, record)
        barrier.wait(timeout=3)
        try:
            return journal.advance(plan_reservation(record, observation, copy_bytes=1))
        except sqlite3.OperationalError:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: contender(), range(2)))
    assert results.count(True) == 1
    assert SqliteRecoveryJournal(path, record).read().metadata.usage.copied_bytes == 41


def test_should_refuse_lock_without_retry_and_leave_spent_record_unchanged(tmp_path):
    record, observation = fixture()
    path = tmp_path / "a.sqlite"
    journal = SqliteRecoveryJournal.create(path, record)
    with sqlite3.connect(path, isolation_level=None) as lock:
        lock.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError):
            journal.advance(plan_reservation(record, observation, copy_bytes=1))
        lock.execute("ROLLBACK")
    assert SqliteRecoveryJournal(path, record).read() == record


def test_should_rollback_write_on_precommit_failure_and_disable_uncertain_adapter(
    tmp_path, monkeypatch
):
    record, observation = fixture()
    path = tmp_path / "a.sqlite"
    journal = SqliteRecoveryJournal.create(path, record)
    import src.infra.sqlite_recovery_journal as module

    original = module._connect

    class Failing(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql == "COMMIT":
                raise sqlite3.OperationalError("injected before commit")
            return super().execute(sql, parameters)

    def connect(path):
        return module._configure(
            sqlite3.connect(
                path.as_uri() + "?mode=rw",
                uri=True,
                timeout=0,
                isolation_level=None,
                factory=Failing,
            )
        )

    monkeypatch.setattr(module, "_connect", connect)
    with pytest.raises(sqlite3.Error):
        journal.advance(plan_reservation(record, observation, copy_bytes=1))
    monkeypatch.setattr(module, "_connect", original)
    assert SqliteRecoveryJournal(path, record).read() == record
    with pytest.raises(ValueError, match="failed"):
        journal.read()


def test_should_refuse_same_coordinator_disk_rollback_against_known_high_water(tmp_path):
    record, observation = fixture()
    path = tmp_path / "a.sqlite"
    journal = SqliteRecoveryJournal.create(path, record)
    assert journal.advance(plan_reservation(record, observation, copy_bytes=1))
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE authority SET record=?", (encode_authority(record),))
    with pytest.raises(ValueError, match="rollback"):
        journal.read()


def test_should_preserve_committed_charge_when_commit_acknowledgment_is_lost(tmp_path, monkeypatch):
    record, observation = fixture()
    path = tmp_path / "a.sqlite"
    journal = SqliteRecoveryJournal.create(path, record)
    import src.infra.sqlite_recovery_journal as module

    original = module._connect

    class LostAcknowledgment(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            result = super().execute(sql, parameters)
            if sql == "COMMIT":
                raise sqlite3.OperationalError("injected after commit")
            return result

    def connect(path):
        return module._configure(
            sqlite3.connect(
                path.as_uri() + "?mode=rw",
                uri=True,
                timeout=0,
                isolation_level=None,
                factory=LostAcknowledgment,
            )
        )

    change = plan_reservation(record, observation, updates=1, copy_bytes=1)
    monkeypatch.setattr(module, "_connect", connect)
    with pytest.raises(sqlite3.Error):
        journal.advance(change)
    monkeypatch.setattr(module, "_connect", original)
    with pytest.raises(ValueError, match="failed"):
        journal.advance(change)
    reopened = SqliteRecoveryJournal(path, record)
    assert reopened.read() == change.proposed
    assert not reopened.advance(change)
    with pytest.raises(ValueError, match="uncertain"):
        plan_reservation(reopened.read(), observation, updates=1)


def test_should_disable_adapter_when_original_database_is_missing(tmp_path):
    record, _ = fixture()
    journal = SqliteRecoveryJournal(tmp_path / "missing.sqlite", record)
    with pytest.raises(sqlite3.Error):
        journal.read()
    with pytest.raises(ValueError, match="failed"):
        journal.read()


def test_should_refuse_database_with_unbounded_page_policy(tmp_path):
    record, _ = fixture()
    path = tmp_path / "large-pages.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA page_size=65536")
        connection.execute("CREATE TABLE other(value)")
    with pytest.raises(ValueError, match="pages"):
        SqliteRecoveryJournal(path, record).read()


def test_should_refuse_changed_journal_policy_without_silently_migrating_it(tmp_path):
    record, _ = fixture()
    path = tmp_path / "wal.sqlite"
    journal = SqliteRecoveryJournal.create(path, record)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
    with pytest.raises(ValueError, match="DELETE"):
        journal.read()
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)


@pytest.mark.parametrize(
    "raw", [b"{}", b'{"metadata":{},"metadata":{}}', b"[]", b"{" * 2000, b"x" * (16 * 1024 + 1)]
)
def test_should_refuse_bounded_codec_corruption_or_duplicate_fields(raw):
    with pytest.raises(ValueError):
        decode_authority(raw)


def test_should_roundtrip_exact_complete_metadata_without_pickle():
    record, _ = fixture()
    raw = encode_authority(record)
    assert decode_authority(raw) == record and raw == encode_authority(decode_authority(raw))
    parsed = json.loads(raw)
    parsed["extra"] = False
    with pytest.raises(ValueError):
        decode_authority(json.dumps(parsed).encode())


@pytest.mark.parametrize("failure", ["corrupt", "schema", "oversize"])
def test_should_refuse_bad_authority_db_before_any_advancement(tmp_path, failure):
    record, observation = fixture()
    path = tmp_path / "a.sqlite"
    journal = SqliteRecoveryJournal.create(path, record)
    with sqlite3.connect(path) as connection:
        if failure == "corrupt":
            connection.execute("UPDATE authority SET record=x'7B7D'")
        if failure == "schema":
            connection.execute("PRAGMA user_version=99")
        if failure == "oversize":
            connection.execute("UPDATE authority SET record=?", (b"x" * 17000,))
    with pytest.raises(ValueError):
        journal.advance(plan_reservation(record, observation, copy_bytes=1))
