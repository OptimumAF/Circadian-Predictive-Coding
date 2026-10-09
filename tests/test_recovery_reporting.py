"""Durable terminal/high-water transitions, fake coordinator and local SQLite."""

from dataclasses import replace
import sqlite3

import pytest

from src.core.recovery_authority import plan_reservation, validate_authority_floor
from src.core.recovery_authority import AuthorityRecord
from src.core.recovery_reporting import (
    AuthorityReport,
    FailureWitness,
    TerminalReportingFailure,
    plan_report,
    validate_authority_report,
    validate_failure_witness,
)
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal
from src.app.recovery_coordinator import RecoveryCoordinator
from src.core.recovery_coordination import RecoveryCosts
from test_recovery_authority import fixture as authority_fixture
from test_recovery_coordinator import fixture, Probe, Observer


def test_should_report_high_water_without_changing_owner_caps_or_spent_work():
    record, observation = authority_fixture()
    change = plan_report(record, observation)
    assert type(change) is AuthorityReport
    assert change.proposed.metadata.sequence == record.metadata.sequence + 1
    assert change.proposed.metadata.observed_ns == observation.now_ns
    assert change.proposed.metadata.usage.peak_rss_bytes == observation.peak_rss_bytes
    assert (
        replace(change.proposed.metadata.usage, peak_rss_bytes=record.metadata.usage.peak_rss_bytes)
        == record.metadata.usage
    )
    assert change.proposed.metadata.limits == record.metadata.limits
    assert change.proposed.metadata.manifest == record.metadata.manifest
    assert change.proposed.worker == record.worker
    validate_authority_floor(record, change.proposed)


@pytest.mark.parametrize("observed", [True, False])
def test_should_stop_with_known_or_unavailable_host_facts_without_inventing_time(observed):
    record, observation = authority_fixture()
    change = plan_report(record, observation if observed else None, terminal=True)
    assert change.proposed.metadata.stopped
    assert change.proposed.metadata.usage.copied_bytes == record.metadata.usage.copied_bytes
    assert change.proposed.metadata.observed_ns == (600 if observed else 400)
    with pytest.raises(ValueError):
        plan_reservation(change.proposed, observation)
    with pytest.raises(ValueError):
        plan_report(record, None)


def test_should_preserve_uncertain_admission_while_reporting_and_stopping():
    record, observation = authority_fixture()
    pending = plan_reservation(record, observation, updates=1, copy_bytes=1).proposed
    report = plan_report(
        pending, replace(observation, now_ns=700, rss_bytes=700, peak_rss_bytes=700)
    )
    stopped = plan_report(report.proposed, None, terminal=True).proposed
    assert stopped.metadata.uncertain_work and stopped.metadata.stopped
    assert stopped.metadata.usage.updates_admitted == 4
    assert stopped.metadata.usage.updates_completed == 3
    assert stopped.metadata.usage.copied_bytes == 41


@pytest.mark.parametrize("failure", ["elapsed", "rss", "ended"])
def test_should_preserve_over_cap_or_ended_observations_only_as_terminal_facts(failure):
    record, observation = authority_fixture()
    if failure == "elapsed":
        observation = replace(observation, now_ns=1100)
    if failure == "rss":
        observation = replace(observation, rss_bytes=1025, peak_rss_bytes=1025)
    if failure == "ended":
        observation = replace(observation, previous_owner_ended=True)
    with pytest.raises(ValueError):
        plan_report(record, observation)
    stopped = plan_report(record, observation, terminal=True).proposed
    assert stopped.metadata.stopped and stopped.metadata.limits == record.metadata.limits
    assert stopped.metadata.observed_ns == observation.now_ns
    assert stopped.metadata.usage.peak_rss_bytes == observation.peak_rss_bytes
    with pytest.raises(ValueError):
        plan_reservation(stopped, observation)


@pytest.mark.parametrize("failure", ["clock", "observer", "owner", "time", "peak"])
def test_should_refuse_unbound_or_regressing_terminal_facts(failure):
    record, observation = authority_fixture()
    if failure == "clock":
        observation = replace(observation, clock_epoch="foreign")
    if failure == "observer":
        observation = replace(observation, observer=replace(record.anchor, pid=99))
    if failure == "owner":
        observation = replace(observation, previous_owner=replace(record.worker, pid=99))
    if failure == "time":
        observation = replace(observation, now_ns=399)
    if failure == "peak":
        observation = replace(observation, rss_bytes=511, peak_rss_bytes=511)
    with pytest.raises(ValueError):
        plan_report(record, observation, terminal=True)


@pytest.mark.parametrize(
    "field",
    ["copied_bytes", "grants", "checkpoint_attempts", "updates_admitted", "updates_completed"],
)
def test_should_refuse_refund_or_invented_completion_in_report(field):
    record, observation = authority_fixture()
    report = plan_report(record, observation, terminal=True)
    value = getattr(report.proposed.metadata.usage, field) - 1
    if field == "updates_completed":
        value += 2
    with pytest.raises(ValueError):
        bad = replace(
            report.proposed,
            metadata=replace(
                report.proposed.metadata,
                usage=replace(report.proposed.metadata.usage, **{field: value}),
            ),
        )
        validate_authority_report(replace(report, proposed=bad))


@pytest.mark.parametrize("field", ["manifest", "limits", "owner", "start", "sequence", "reopen"])
def test_should_refuse_source_policy_owner_quota_or_stopped_history_changes(field):
    record, observation = authority_fixture()
    report = plan_report(record, observation, terminal=True)
    metadata = report.proposed.metadata
    if field == "manifest":
        metadata = replace(metadata, manifest=replace(metadata.manifest, source_sha256="f" * 64))
    if field == "limits":
        metadata = replace(metadata, limits=replace(metadata.limits, max_copy_bytes=101))
    if field == "owner":
        metadata = replace(metadata, owner_epoch=5)
    if field == "start":
        metadata = replace(metadata, started_ns=101)
    if field == "sequence":
        metadata = replace(metadata, sequence=9)
    if field == "reopen":
        record = report.proposed
        metadata = replace(record.metadata, sequence=record.metadata.sequence + 1, stopped=False)
        report = replace(report, expected=record)
    with pytest.raises(ValueError):
        validate_authority_report(
            replace(report, proposed=replace(report.proposed, metadata=metadata))
        )


def test_should_revalidate_corrupted_report_and_witness_exact_types():
    record, observation = authority_fixture()
    report = plan_report(record, observation)
    object.__setattr__(report.proposed.metadata.usage, "peak_rss_bytes", False)
    with pytest.raises(ValueError):
        validate_authority_report(report)
    with pytest.raises(ValueError):
        validate_authority_report(object())
    with pytest.raises(ValueError):
        validate_failure_witness(object())


def test_should_persist_reports_and_terminal_state_across_adapter_instances(tmp_path):
    record, observation = authority_fixture()
    path = tmp_path / "a.sqlite"
    journal = SqliteRecoveryJournal.create(path, record)
    report = plan_report(record, observation)
    assert journal.report(report)
    assert not journal.report(report)
    stop = plan_report(report.proposed, None, terminal=True)
    assert journal.report(stop)
    current = SqliteRecoveryJournal(path, record).read()
    assert current == stop.proposed
    assert current.metadata.limits == record.metadata.limits
    with pytest.raises(ValueError):
        plan_reservation(current, observation)


def test_should_persist_callback_failure_and_refuse_new_coordinator(tmp_path):
    path = tmp_path / "a.sqlite"
    coordinator, journal, observer, anchor, worker, _ = fixture(
        lambda r: SqliteRecoveryJournal.create(path, r)
    )

    def fail():
        observer.now = 700
        raise OSError("preparation failed")

    with pytest.raises(OSError):
        coordinator.execute(RecoveryCosts(copy_bytes=1), fail)
    assert coordinator.terminal_confirmed
    stopped = journal.read()
    assert stopped.metadata.stopped and stopped.metadata.usage.copied_bytes == 41
    coordinator.close()
    log: list[tuple[str, object]] = []
    a, w = Probe(stopped.anchor, log), Probe(stopped.worker, log)
    reopened_observer = Observer(stopped, log)
    reopened_observer.now = stopped.metadata.observed_ns
    with pytest.raises(ValueError, match="stopped"):
        RecoveryCoordinator(stopped, SqliteRecoveryJournal(path, stopped), reopened_observer, a, w)
    assert a.closed and w.closed
    assert SqliteRecoveryJournal(path, stopped).read().metadata.stopped


def test_should_persist_latest_validated_clock_and_peak_before_publication(tmp_path):
    path = tmp_path / "a.sqlite"
    coordinator, journal, observer, _, _, _ = fixture(
        lambda r: SqliteRecoveryJournal.create(path, r)
    )
    from src.core.recovery_coordination import RecoveryCosts

    def prepare():
        observer.hook = lambda o: replace(o, rss_bytes=700, peak_rss_bytes=700)
        return "prepared"

    def publish(value):
        current = SqliteRecoveryJournal(path, journal.read()).read()
        assert current.metadata.observed_ns == observer.now
        assert current.metadata.usage.peak_rss_bytes == 700

    coordinator.execute(RecoveryCosts(copy_bytes=1), prepare, publish)
    coordinator.close()
    assert journal.read().metadata.stopped


@pytest.mark.parametrize("failure", ["rss", "elapsed", "foreign"])
def test_should_stop_on_failed_observation_and_keep_only_trustworthy_facts(tmp_path, failure):
    path = tmp_path / "a.sqlite"
    coordinator, journal, observer, _, _, _ = fixture(
        lambda r: SqliteRecoveryJournal.create(path, r)
    )
    from src.core.recovery_coordination import RecoveryCosts

    previous = journal.read()
    if failure == "rss":
        observer.hook = lambda o: replace(o, rss_bytes=1025, peak_rss_bytes=1025)
    if failure == "elapsed":
        observer.now = 1100
    if failure == "foreign":
        observer.hook = lambda o: replace(o, clock_epoch="foreign")
    with pytest.raises(ValueError):
        coordinator.execute(RecoveryCosts(copy_bytes=1), lambda: None)
    stopped = journal.read()
    assert stopped.metadata.stopped and coordinator.terminal_confirmed
    assert stopped.metadata.usage.copied_bytes == 40
    if failure == "rss":
        assert stopped.metadata.usage.peak_rss_bytes == 1025
    if failure == "elapsed":
        assert stopped.metadata.observed_ns >= 1100
    if failure == "foreign":
        assert stopped.metadata.observed_ns == previous.metadata.observed_ns
    coordinator.close()


@pytest.mark.parametrize("after_commit", [False, True])
def test_should_require_explicit_terminal_reconciliation_after_commit_ambiguity(
    tmp_path, monkeypatch, after_commit
):
    path = tmp_path / "a.sqlite"
    coordinator, journal, _, _, _, log = fixture(lambda r: SqliteRecoveryJournal.create(path, r))
    from src.core.recovery_coordination import RecoveryCosts
    import src.infra.sqlite_recovery_journal as module

    original = module._connect

    class Fault(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql == "COMMIT" and not after_commit:
                raise sqlite3.OperationalError("before commit")
            result = super().execute(sql, parameters)
            if sql == "COMMIT":
                raise sqlite3.OperationalError("after commit")
            return result

    def connect(path):
        return module._configure(
            sqlite3.connect(
                path.as_uri() + "?mode=rw", uri=True, timeout=0, isolation_level=None, factory=Fault
            )
        )

    monkeypatch.setattr(module, "_connect", connect)
    with pytest.raises(sqlite3.OperationalError) as result:
        coordinator.execute(
            RecoveryCosts(updates=1, copy_bytes=1), lambda: log.append(("prepare", 0))
        )
    assert isinstance(result.value.__cause__, TerminalReportingFailure)
    assert not coordinator.terminal_confirmed
    witness = coordinator.failure_witness
    assert witness is not None and witness.attempted is not None
    assert not any(name == "prepare" for name, _ in log)
    monkeypatch.setattr(module, "_connect", original)
    stopped = SqliteRecoveryJournal.reconcile_terminal(path, witness)
    assert stopped.metadata.stopped
    assert stopped.metadata.usage.copied_bytes == (41 if after_commit else 40)
    assert stopped.metadata.usage.updates_admitted == (4 if after_commit else 3)
    assert stopped.metadata.uncertain_work is after_commit
    assert SqliteRecoveryJournal.reconcile_terminal(path, witness) == stopped
    with pytest.raises(ValueError):
        plan_reservation(stopped, authority_fixture()[1])
    coordinator.close()


def test_should_refuse_reconciliation_from_foreign_or_unwitnessed_disk_state(tmp_path):
    record, observation = authority_fixture()
    path = tmp_path / "a.sqlite"
    journal = SqliteRecoveryJournal.create(path, record)
    witness = FailureWitness(record, observation=observation)
    changed = plan_reservation(record, observation, copy_bytes=1)
    assert journal.advance(changed)
    with pytest.raises(ValueError, match="witness"):
        SqliteRecoveryJournal.reconcile_terminal(path, witness)
    assert SqliteRecoveryJournal(path, record).read() == changed.proposed


def test_should_reject_unrelated_attempt_in_failure_witness():
    record, observation = authority_fixture()
    unrelated = replace(record, metadata=replace(record.metadata, owner_id="foreign"))
    attempt = plan_reservation(unrelated, observation, copy_bytes=1)
    with pytest.raises(ValueError):
        FailureWitness(record, attempted=attempt)


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("after_commit", [False, True])
def test_should_reconcile_report_commit_faults_without_losing_measured_high_water(
    tmp_path, monkeypatch, terminal, after_commit
):
    record, observation = authority_fixture()
    path = tmp_path / "a.sqlite"
    journal = SqliteRecoveryJournal.create(path, record)
    report = plan_report(record, observation, terminal=terminal)
    witness = FailureWitness(record, attempted=report, observation=observation)
    import src.infra.sqlite_recovery_journal as module

    original = module._connect

    class Fault(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql == "COMMIT" and not after_commit:
                raise sqlite3.OperationalError("before report commit")
            result = super().execute(sql, parameters)
            if sql == "COMMIT":
                raise sqlite3.OperationalError("after report commit")
            return result

    def connect(path):
        return module._configure(
            sqlite3.connect(
                path.as_uri() + "?mode=rw", uri=True, timeout=0, isolation_level=None, factory=Fault
            )
        )

    monkeypatch.setattr(module, "_connect", connect)
    with pytest.raises(sqlite3.Error):
        journal.report(report)
    with pytest.raises(ValueError, match="failed"):
        journal.read()
    monkeypatch.setattr(module, "_connect", original)
    stopped = SqliteRecoveryJournal.reconcile_terminal(path, witness)
    assert stopped.metadata.stopped
    assert stopped.metadata.observed_ns == observation.now_ns
    assert stopped.metadata.usage.peak_rss_bytes == observation.peak_rss_bytes
    assert stopped.metadata.usage.copied_bytes == 40
    assert SqliteRecoveryJournal.reconcile_terminal(path, witness) == stopped


@pytest.mark.parametrize(
    "field", ["updates_admitted", "copied_bytes", "grants", "checkpoint_attempts"]
)
def test_should_not_use_terminal_rss_exception_to_bypass_other_original_caps(field):
    record, _ = authority_fixture()
    cap = {"updates_admitted": 10, "copied_bytes": 100, "grants": 10, "checkpoint_attempts": 4}[
        field
    ]
    with pytest.raises(ValueError):
        metadata = replace(
            record.metadata,
            stopped=True,
            uncertain_work=True,
            usage=replace(record.metadata.usage, peak_rss_bytes=1025, **{field: cap + 1}),
        )
        AuthorityRecord(metadata, record.anchor, record.worker)


@pytest.mark.parametrize(
    "failure",
    [
        "terminal-live",
        "unrelated-terminal",
        "foreign-observation",
        "unknown-attempt",
        "unknown-terminal",
    ],
)
def test_should_refuse_malformed_witness_relationships_before_storage(failure):
    record, observation = authority_fixture()
    if failure == "terminal-live":
        with pytest.raises(ValueError):
            FailureWitness(record, terminal_attempt=plan_report(record, observation))
    elif failure == "unrelated-terminal":
        other = replace(record, metadata=replace(record.metadata, owner_id="foreign"))
        with pytest.raises(ValueError):
            FailureWitness(record, terminal_attempt=plan_report(other, None, terminal=True))
    elif failure == "foreign-observation":
        with pytest.raises(ValueError):
            FailureWitness(record, observation=replace(observation, clock_epoch="foreign"))
    elif failure == "unknown-attempt":
        witness = FailureWitness(record)
        object.__setattr__(witness, "attempted", object())
        with pytest.raises(ValueError):
            validate_failure_witness(witness)
    else:
        witness = FailureWitness(record)
        object.__setattr__(witness, "terminal_attempt", object())
        with pytest.raises(ValueError):
            validate_failure_witness(witness)


def test_should_not_create_missing_database_during_terminal_reconciliation(tmp_path):
    record, observation = authority_fixture()
    path = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.Error):
        SqliteRecoveryJournal.reconcile_terminal(
            path, FailureWitness(record, observation=observation)
        )
    assert not path.exists()
