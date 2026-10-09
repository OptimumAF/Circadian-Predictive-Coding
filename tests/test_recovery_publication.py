"""Trusted fake observations and actual local SQLite competing writers; no workers."""

from dataclasses import replace
from threading import Thread
import sqlite3

import pytest

from src.core.recovery_authority import plan_reservation
from src.core.recovery_reporting import FailureWitness, plan_report
from src.core.recovery_publication import validate_publication_observation
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal
from test_recovery_authority import fixture


def test_should_preserve_every_spent_binding_and_release_normal_guard(tmp_path):
    record, observation = fixture()
    journal = SqliteRecoveryJournal.create(tmp_path / "authority.db", record)
    before = journal.path.read_bytes()
    calls = []

    def observe():
        calls.append(True)
        return replace(observation, now_ns=600 + len(calls))

    with journal.publication_guard(record, observe):
        assert calls == [True]
        assert journal.read() == record
    assert calls == [True, True]
    assert journal.read() == record and journal.path.read_bytes() == before
    assert journal.report(plan_report(record, observation)) is True


@pytest.mark.parametrize("kind", ["reservation", "report", "terminal"])
def test_should_exclude_separate_supported_writer_until_guard_release(tmp_path, kind):
    record, observation = fixture()
    journal = SqliteRecoveryJournal.create(tmp_path / "authority.db", record)
    outcomes = []

    def compete():
        other = SqliteRecoveryJournal(journal.path, record)
        try:
            if kind == "reservation":
                other.advance(plan_reservation(record, observation, copy_bytes=1))
            elif kind == "report":
                other.report(plan_report(record, observation, terminal=True))
            else:
                other.reconcile_terminal(journal.path, FailureWitness(record, None, None, None))
        except sqlite3.OperationalError as error:
            outcomes.append(str(error))

    with journal.publication_guard(record, lambda: observation):
        worker = Thread(target=compete)
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert len(outcomes) == 1 and "locked" in outcomes[0]
        assert journal.read() == record
    assert journal.read() == record
    assert SqliteRecoveryJournal(journal.path, record).report(
        plan_report(record, observation, terminal=True)
    )


@pytest.mark.parametrize("boundary", ["before", "after"])
@pytest.mark.parametrize(
    "fault", ["expired", "rss", "ended", "epoch", "owner", "anchor", "unknown", "regression"]
)
def test_should_refuse_bad_host_facts_at_both_callback_boundaries(tmp_path, boundary, fault):
    record, observation = fixture()
    journal = SqliteRecoveryJournal.create(tmp_path / "authority.db", record)
    bad = observation
    if fault == "expired":
        bad = replace(observation, now_ns=1100)
    elif fault == "rss":
        bad = replace(observation, rss_bytes=1025, peak_rss_bytes=1025)
    elif fault == "ended":
        bad = replace(observation, previous_owner_ended=True)
    elif fault == "epoch":
        bad = replace(observation, clock_epoch="foreign")
    elif fault in ("owner", "anchor"):
        name = "previous_owner" if fault == "owner" else "observer"
        bad = replace(observation, **{name: replace(getattr(observation, name), pid=99)})
    elif fault == "regression":
        bad = replace(observation, now_ns=399 if boundary == "before" else 599)
    calls = []
    published = []

    def observe():
        calls.append(True)
        broken = boundary == "before" or len(calls) == 2
        if broken and fault == "unknown":
            raise RuntimeError("host facts unavailable")
        return bad if broken else observation

    with pytest.raises((ValueError, RuntimeError)):
        with journal.publication_guard(record, observe):
            published.append(True)
    assert published == ([] if boundary == "before" else [True])
    with pytest.raises(ValueError, match="previously failed"):
        journal.read()
    # Post-callback failure cannot undo publication. It cannot refund authority either.
    other = SqliteRecoveryJournal(journal.path, record)
    assert other.read() == record
    assert other.report(plan_report(record, observation, terminal=True))


@pytest.mark.parametrize("flag", ["stopped", "uncertain"])
def test_should_refuse_terminal_or_uncertain_authority_without_observing_or_publishing(
    tmp_path, flag
):
    record, observation = fixture()
    record = (
        plan_report(record, None, terminal=True)
        if flag == "stopped"
        else plan_reservation(record, observation, updates=1)
    ).proposed
    journal = SqliteRecoveryJournal.create(tmp_path / "authority.db", record)
    calls = []

    def observe():
        calls.append(True)
        return observation

    with pytest.raises(ValueError):
        with journal.publication_guard(record, observe):
            pytest.fail("terminal/uncertain publication must be refused")
    assert not calls
    assert SqliteRecoveryJournal(journal.path, record).read() == record


def test_should_refuse_stale_exact_authority_under_transaction(tmp_path):
    record, observation = fixture()
    journal = SqliteRecoveryJournal.create(tmp_path / "authority.db", record)
    newer = plan_report(record, observation)
    assert SqliteRecoveryJournal(journal.path, record).report(newer)
    with pytest.raises(ValueError, match="differs"):
        with journal.publication_guard(record, lambda: observation):
            pytest.fail("stale callback")
    assert SqliteRecoveryJournal(journal.path, newer.proposed).read() == newer.proposed


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_should_release_guard_on_base_exception_without_refund(tmp_path, error):
    record, observation = fixture()
    record = plan_reservation(record, observation, copy_bytes=1, grants=1).proposed
    journal = SqliteRecoveryJournal.create(tmp_path / "authority.db", record)
    with pytest.raises(error):
        with journal.publication_guard(record, lambda: observation):
            raise error("callback failure")
    other = SqliteRecoveryJournal(journal.path, record)
    assert other.read() == record
    assert other.report(plan_report(record, observation, terminal=True))
    with pytest.raises(ValueError, match="previously failed"):
        with journal.publication_guard(record, lambda: observation):
            pytest.fail("failed adapter retry")


def test_should_refuse_nested_guard_and_disable_outer_publication(tmp_path):
    record, observation = fixture()
    journal = SqliteRecoveryJournal.create(tmp_path / "authority.db", record)
    with pytest.raises(ValueError, match="previously failed"):
        with journal.publication_guard(record, lambda: observation):
            with pytest.raises(sqlite3.OperationalError):
                with journal.publication_guard(record, lambda: observation):
                    pytest.fail("nested guard")
    assert SqliteRecoveryJournal(journal.path, record).read() == record


def test_should_refuse_inexact_or_corrupted_guard_authority():
    record, observation = fixture()
    with pytest.raises(ValueError):
        validate_publication_observation(None, observation)
    object.__setattr__(record.metadata.usage, "copied_bytes", False)
    with pytest.raises(ValueError):
        validate_publication_observation(record, observation)


@pytest.mark.parametrize("after_release", [False, True])
def test_should_close_connection_even_when_guard_release_fails(
    tmp_path, monkeypatch, after_release
):
    from src.infra import sqlite_recovery_journal as module

    record, observation = fixture()
    journal = SqliteRecoveryJournal.create(tmp_path / "authority.db", record)
    original = module._connect
    closed = []

    class Fault(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql == "ROLLBACK" and not after_release:
                raise sqlite3.OperationalError("before release")
            result = super().execute(sql, parameters)
            if sql == "ROLLBACK":
                raise sqlite3.OperationalError("after release")
            return result

        def close(self):
            super().close()
            closed.append(True)

    def connect(path):
        return module._configure(
            sqlite3.connect(
                path.as_uri() + "?mode=rw", uri=True, timeout=0, isolation_level=None, factory=Fault
            )
        )

    monkeypatch.setattr(module, "_connect", connect)
    with pytest.raises(sqlite3.OperationalError):
        with journal.publication_guard(record, lambda: observation):
            pass
    assert closed == [True]
    with pytest.raises(ValueError, match="previously failed"):
        journal.read()
    monkeypatch.setattr(module, "_connect", original)
    other = SqliteRecoveryJournal(journal.path, record)
    assert other.read() == record
    assert other.report(plan_report(record, observation, terminal=True))
