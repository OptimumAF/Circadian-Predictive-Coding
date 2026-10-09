"""Actual local journal/OS writer exclusion with trusted fake process facts."""

from contextlib import contextmanager
from dataclasses import replace
import sqlite3
from threading import Thread

import pytest

from src.core.recovery_publication import RecoveryPublicationLease
from src.core.recovery_coordination import RecoveryCosts
from src.core.recovery_authority import plan_reservation
from src.core.recovery_reporting import FailureWitness, plan_report, TerminalReportingFailure
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal
from test_recovery_authority import fixture as authority_fixture
from test_recovery_coordinator import fixture


def local(tmp_path):
    return fixture(lambda r: SqliteRecoveryJournal.create(tmp_path / "authority.db", r))


@pytest.mark.parametrize("kind", ["reserve", "report", "terminal", "legacy-guard"])
def test_should_exclude_competing_writers_and_commit_fresh_facts_before_publication(tmp_path, kind):
    coordinator, journal, observer, anchor, worker, _ = local(tmp_path)
    published = []
    errors = []

    def compete(record, observation):
        other = SqliteRecoveryJournal(journal.path, record)
        try:
            if kind == "reserve":
                other.advance(plan_reservation(record, observation, copy_bytes=1))
            elif kind == "report":
                other.report(plan_report(record, observation, terminal=True))
            elif kind == "terminal":
                other.reconcile_terminal(journal.path, FailureWitness(record))
            else:
                with other.publication_guard(record, lambda: observation):
                    errors.append("unexpected guard")
        except sqlite3.OperationalError as error:
            errors.append(str(error))

    def publish(value):
        record = journal.read()
        assert record.metadata.observed_ns == observer.now
        assert record.metadata.usage.peak_rss_bytes == 700
        assert record.metadata.usage.copied_bytes == 41
        observation = replace(
            authority_fixture()[1], now_ns=observer.now, rss_bytes=700, peak_rss_bytes=700
        )
        thread = Thread(target=compete, args=(record, observation))
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert len(errors) == 1 and "locked" in errors[0]
        published.append(value)

    observer.hook = lambda o: replace(o, rss_bytes=700, peak_rss_bytes=700)
    assert (
        coordinator.execute(RecoveryCosts(copy_bytes=1), lambda: "prepared", publish) == "prepared"
    )
    current = journal.read()
    assert current.metadata.observed_ns == observer.now
    assert current.metadata.usage.copied_bytes == 41 and published == ["prepared"]
    assert not current.metadata.stopped and not current.metadata.uncertain_work
    coordinator.close()
    assert anchor.closed and worker.closed and journal.read().metadata.stopped


@pytest.mark.parametrize("boundary", ["entry", "exit"])
@pytest.mark.parametrize("fault", ["expired", "rss", "ended", "unknown", "foreign", "regression"])
def test_should_refuse_bad_guarded_host_facts_and_reconcile_without_refunds(
    tmp_path, boundary, fault
):
    coordinator, journal, observer, _, worker, _ = local(tmp_path)
    original = journal.publication_lease
    published = []

    def break_facts():
        if fault == "expired":
            observer.now = 1100
        elif fault == "rss":
            observer.hook = lambda o: replace(o, rss_bytes=1025, peak_rss_bytes=1025)
        elif fault == "ended":
            worker.ended = True
        elif fault == "unknown":
            worker.ended = None
        elif fault == "foreign":
            observer.hook = lambda o: replace(o, clock_epoch="foreign")
        else:
            observer.now = 399

    @contextmanager
    def guarded(expected):
        with original(expected) as lease:
            if boundary == "entry":
                break_facts()
            yield lease

    journal.publication_lease = guarded

    def publish(value):
        published.append(value)
        if boundary == "exit":
            break_facts()

    with pytest.raises(ValueError):
        coordinator.execute(RecoveryCosts(copy_bytes=1), lambda: "prepared", publish)
    assert published == ([] if boundary == "entry" else ["prepared"])
    assert not coordinator.terminal_confirmed
    witness = coordinator.failure_witness
    assert witness is not None
    stopped = SqliteRecoveryJournal.reconcile_terminal(journal.path, witness)
    assert stopped.metadata.stopped and stopped.metadata.usage.copied_bytes == 41
    assert stopped.metadata.usage.updates_admitted == stopped.metadata.usage.updates_completed == 3
    if fault == "rss":
        assert stopped.metadata.usage.peak_rss_bytes == 1025
    if fault == "expired":
        assert stopped.metadata.observed_ns >= 1100
    with pytest.raises(ValueError, match="failed"):
        coordinator.execute(RecoveryCosts(), lambda: None)
    coordinator.close()


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_should_release_publication_lease_and_keep_callback_failure_witness(tmp_path, error):
    coordinator, journal, _, anchor, worker, _ = local(tmp_path)

    def publish(value):
        raise error("publication failed")

    with pytest.raises(error) as caught:
        coordinator.execute(RecoveryCosts(copy_bytes=1), lambda: None, publish)
    assert isinstance(caught.value.__cause__, TerminalReportingFailure)
    witness = coordinator.failure_witness
    assert witness is not None
    stopped = SqliteRecoveryJournal.reconcile_terminal(journal.path, witness)
    assert stopped.metadata.stopped and stopped.metadata.usage.copied_bytes == 41
    coordinator.close()
    assert anchor.closed and worker.closed


def test_should_refuse_stale_acquisition_and_never_publish_or_refund(tmp_path):
    coordinator, journal, observer, _, _, _ = local(tmp_path)
    original = journal.publication_lease

    @contextmanager
    def guarded(expected):
        other = SqliteRecoveryJournal(journal.path, expected)
        proposal = plan_report(expected, observer.observe(coordinator._worker), terminal=True)
        assert other.report(proposal)
        with original(expected) as lease:
            yield lease

    journal.publication_lease = guarded
    with pytest.raises(ValueError):
        coordinator.execute(
            RecoveryCosts(copy_bytes=1), lambda: None, lambda v: pytest.fail("stale publication")
        )
    current = SqliteRecoveryJournal(journal.path, authority_fixture()[0]).read()
    assert current.metadata.stopped and current.metadata.usage.copied_bytes == 41
    assert coordinator.failure_witness is not None and not coordinator.terminal_confirmed
    coordinator.close()


def test_should_restrict_lease_to_observation_reports_and_invalidate_after_release(tmp_path):
    record, observation = authority_fixture()
    journal = SqliteRecoveryJournal.create(tmp_path / "authority.db", record)
    with journal.publication_lease(record) as lease:
        typed: RecoveryPublicationLease = lease
        assert typed.read() == record
        assert typed.report(plan_report(record, observation))
        saved = typed.read()
        with pytest.raises(ValueError):
            typed.report(plan_report(saved, None, terminal=True))
        assert typed.read() == saved
    with pytest.raises(ValueError, match="inactive"):
        typed.read()
    with pytest.raises(ValueError, match="inactive"):
        typed.report(plan_report(saved, observation))
    assert journal.read() == saved


@pytest.mark.parametrize("after_commit", [False, True])
def test_should_keep_guarded_report_commit_ambiguity_and_reconcile_exact_spent_state(
    tmp_path, monkeypatch, after_commit
):
    from src.infra import sqlite_recovery_journal as module

    coordinator, journal, observer, _, _, _ = local(tmp_path)
    original_connect = module._connect
    armed = False

    class Fault(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if armed and sql == "COMMIT" and not after_commit:
                raise sqlite3.OperationalError("before guarded commit")
            result = super().execute(sql, parameters)
            if armed and sql == "COMMIT":
                raise sqlite3.OperationalError("after guarded commit")
            return result

    def connect(path):
        return module._configure(
            sqlite3.connect(
                path.as_uri() + "?mode=rw", uri=True, timeout=0, isolation_level=None, factory=Fault
            )
        )

    original_lease = journal.publication_lease

    @contextmanager
    def guarded(expected):
        nonlocal armed
        with original_lease(expected) as lease:
            armed = True
            yield lease

    monkeypatch.setattr(module, "_connect", connect)
    journal.publication_lease = guarded
    with pytest.raises(sqlite3.OperationalError):
        coordinator.execute(
            RecoveryCosts(copy_bytes=1),
            lambda: None,
            lambda v: pytest.fail("uncertain report must not publish"),
        )
    monkeypatch.setattr(module, "_connect", original_connect)
    witness = coordinator.failure_witness
    assert witness is not None and witness.attempted is not None
    disk = SqliteRecoveryJournal(journal.path, authority_fixture()[0]).read()
    assert disk == (witness.attempted.proposed if after_commit else witness.attempted.expected)
    stopped = SqliteRecoveryJournal.reconcile_terminal(journal.path, witness)
    assert stopped.metadata.stopped and stopped.metadata.usage.copied_bytes == 41
    assert stopped.metadata.observed_ns == observer.now
    coordinator.close()


@pytest.mark.parametrize("after_release", [False, True])
def test_should_release_native_ownership_on_unlock_fault_and_reconcile_spent_state(
    tmp_path, monkeypatch, after_release
):
    from src.infra import sqlite_recovery_lock as module

    coordinator, journal, _, _, _, _ = local(tmp_path)
    original_lock = module._native_lock
    original_lease = journal.publication_lease
    armed = False
    published: list[str] = []

    def lock(stream, *, release=False):
        if armed and release and not after_release:
            raise OSError("before native release")
        original_lock(stream, release=release)
        if armed and release:
            raise OSError("after native release")

    @contextmanager
    def guarded(expected):
        nonlocal armed
        with original_lease(expected) as lease:
            armed = True
            yield lease

    monkeypatch.setattr(module, "_native_lock", lock)
    journal.publication_lease = guarded
    with pytest.raises(OSError):
        coordinator.execute(RecoveryCosts(copy_bytes=1), lambda: "prepared", published.append)
    assert published == ["prepared"]
    monkeypatch.setattr(module, "_native_lock", original_lock)
    witness = coordinator.failure_witness
    assert witness is not None and not coordinator.terminal_confirmed
    stopped = SqliteRecoveryJournal.reconcile_terminal(journal.path, witness)
    assert stopped.metadata.stopped and stopped.metadata.usage.copied_bytes == 41
    coordinator.close()


@pytest.mark.parametrize("flag", ["stopped", "uncertain"])
def test_should_refuse_terminal_or_uncertain_publication_lease_without_reopening(tmp_path, flag):
    record, observation = authority_fixture()
    record = (
        plan_report(record, None, terminal=True)
        if flag == "stopped"
        else plan_reservation(record, observation, updates=1)
    ).proposed
    journal = SqliteRecoveryJournal.create(tmp_path / "authority.db", record)
    with pytest.raises(ValueError):
        with journal.publication_lease(record):
            pytest.fail("terminal/uncertain lease")
    assert journal.read() == record


def test_should_refuse_cross_thread_lease_use_and_nested_writer_without_corrupting_owner(tmp_path):
    record, observation = authority_fixture()
    journal = SqliteRecoveryJournal.create(tmp_path / "authority.db", record)
    errors = []
    with journal.publication_lease(record) as lease:

        def misuse():
            try:
                lease.report(plan_report(record, observation))
            except ValueError as error:
                errors.append(str(error))

        thread = Thread(target=misuse)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive() and len(errors) == 1
        other = SqliteRecoveryJournal(journal.path, record)
        with pytest.raises(sqlite3.OperationalError):
            with other.publication_lease(record):
                pytest.fail("nested lease")
        assert lease.read() == record
    assert journal.read() == record


@pytest.mark.parametrize("concurrent", [False, True])
def test_should_refuse_nested_or_concurrent_coordinator_dispatch_during_publication(
    tmp_path, concurrent
):
    coordinator, journal, _, _, _, _ = local(tmp_path)
    errors = []

    def dispatch():
        try:
            coordinator.execute(RecoveryCosts(copy_bytes=1), lambda: pytest.fail("nested work"))
        except ValueError as error:
            errors.append(str(error))

    def publish(value):
        if concurrent:
            thread = Thread(target=dispatch)
            thread.start()
            thread.join(timeout=2)
            assert not thread.is_alive()
        else:
            dispatch()
        assert len(errors) == 1 and "busy" in errors[0]

    coordinator.execute(RecoveryCosts(copy_bytes=1), lambda: None, publish)
    assert journal.read().metadata.usage.copied_bytes == 41
    assert not journal.read().metadata.stopped
    coordinator.close()
