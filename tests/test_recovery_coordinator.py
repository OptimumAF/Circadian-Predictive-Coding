"""Trusted fake registrations/observations and local SQLite; no native worker."""

from dataclasses import replace
from contextlib import contextmanager
from threading import Event, Thread

import pytest

from src.app.recovery_coordinator import RecoveryCoordinator
from src.core.recovery_coordination import RecoveryCosts
from src.core.recovery_authority import plan_reservation, validate_authority_change
from src.core.recovery_observation import RecoveryProcessIdentity, RecoveryHostObservation
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal
from src.core.recovery_reporting import validate_authority_report
from test_recovery_authority import fixture as authority_fixture


class Probe:
    def __init__(self, identity, log):
        self.identity = identity
        self.log = log
        self.ended: bool | None = False
        self.closed = False
        self.close_error = None

    def is_ended(self):
        if self.closed:
            raise ValueError("probe closed")
        return self.ended

    def close(self):
        self.log.append(("close", self.identity.pid))
        self.closed = True
        if self.close_error:
            raise self.close_error


class Observer:
    def __init__(self, record, log):
        self.record, self.log = record, log
        self.now = 600
        self.hook = None

    def observe(self, previous_owner=None):
        self.now += 10
        self.log.append(("observe", previous_owner.identity.pid))
        observation = RecoveryHostObservation(
            self.record.metadata.clock_epoch,
            self.now,
            640,
            640,
            self.record.anchor,
            previous_owner.identity,
            previous_owner.is_ended(),
        )
        return observation if self.hook is None else self.hook(observation)


class MemoryJournal:
    def __init__(self, record, log):
        self.record, self.log = record, log
        self.hook = None

    def read(self):
        self.log.append(("read", self.record.metadata.sequence))
        return self.record

    def advance(self, change):
        self.log.append(("advance", change.proposed.metadata.sequence))
        validate_authority_change(change)
        if self.hook is not None:
            return self.hook(change)
        if self.record != change.expected:
            return False
        self.record = change.proposed
        return True

    def report(self, report):
        self.log.append(("report", report.proposed.metadata.sequence))
        validate_authority_report(report)
        if self.record != report.expected:
            return False
        self.record = report.proposed
        return True

    @contextmanager
    def publication_lease(self, expected):
        if self.record != expected or expected.metadata.stopped or expected.metadata.uncertain_work:
            raise ValueError("fake publication authority differs or cannot publish")
        yield self


def fixture(journal_factory=None):
    record, _ = authority_fixture()
    log: list[tuple[str, object]] = []
    anchor, worker = Probe(record.anchor, log), Probe(record.worker, log)
    observer = Observer(record, log)
    journal = MemoryJournal(record, log) if journal_factory is None else journal_factory(record)
    coordinator = RecoveryCoordinator(record, journal, observer, anchor, worker)
    return coordinator, journal, observer, anchor, worker, log


def test_should_commit_before_preparation_and_check_before_and_after_publication():
    coordinator, journal, _, anchor, worker, log = fixture()

    def prepare():
        log.append(("prepare", 0))
        assert journal.record.metadata.usage.copied_bytes == 45
        return "prepared"

    def publish(value):
        assert value == "prepared"
        log.append(("publish", 0))

    assert coordinator.execute(RecoveryCosts(copy_bytes=5), prepare, publish) == "prepared"
    names = [item[0] for item in log]
    assert names.index("advance") < names.index("prepare") < names.index("publish")
    assert "observe" in names[names.index("prepare") + 1 : names.index("publish")]
    assert "observe" in names[names.index("publish") + 1 :]
    assert journal.record.metadata.usage.copied_bytes == 45
    coordinator.close()
    assert anchor.closed and worker.closed


@pytest.mark.parametrize("sqlite", [False, True])
def test_should_keep_admission_uncertain_and_refuse_publication_or_retry(tmp_path, sqlite):
    factory = (
        (lambda record: SqliteRecoveryJournal.create(tmp_path / "a.sqlite", record))
        if sqlite
        else None
    )
    coordinator, journal, _, anchor, worker, log = fixture(factory)

    def prepare():
        assert journal.read().metadata.uncertain_work
        log.append(("prepare", 0))
        return "native-result"

    with pytest.raises(ValueError, match="uncertain"):
        coordinator.execute(
            RecoveryCosts(updates=1, copy_bytes=1),
            prepare,
            lambda value: log.append(("publish", value)),
        )
    record = journal.read()
    assert record.metadata.usage.updates_admitted == 4
    assert record.metadata.usage.updates_completed == 3
    assert record.metadata.usage.copied_bytes == 41
    assert not any(name == "publish" for name, _ in log)
    with pytest.raises(ValueError, match="failed"):
        coordinator.execute(RecoveryCosts(), lambda: None)
    coordinator.close()
    assert anchor.closed and worker.closed


@pytest.mark.parametrize("failure", ["stale", "before-commit", "after-commit", "unknown-result"])
def test_should_not_dispatch_after_stale_or_uncertain_commit(failure):
    coordinator, journal, _, _, _, log = fixture()

    def advance(change):
        if failure == "after-commit":
            journal.record = change.proposed
        if failure == "stale":
            return False
        if failure == "unknown-result":
            return 1
        raise OSError(failure)

    journal.hook = advance
    with pytest.raises((ValueError, OSError)):
        coordinator.execute(RecoveryCosts(copy_bytes=1), lambda: log.append(("prepare", 0)))
    assert not any(name == "prepare" for name, _ in log)
    assert journal.record.metadata.usage.copied_bytes == (41 if failure == "after-commit" else 40)
    with pytest.raises(ValueError, match="failed"):
        coordinator.execute(RecoveryCosts(), lambda: None)
    coordinator.close()


@pytest.mark.parametrize("phase", ["prepare", "publish"])
@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt])
def test_should_preserve_charges_and_disable_lane_on_callback_base_exception(phase, error):
    coordinator, journal, _, _, _, _ = fixture()

    def fail(*args):
        raise error("injected callback")

    with pytest.raises(error):
        coordinator.execute(
            RecoveryCosts(copy_bytes=1),
            fail if phase == "prepare" else lambda: None,
            fail if phase == "publish" else None,
        )
    assert journal.record.metadata.usage.copied_bytes == 41
    with pytest.raises(ValueError, match="failed"):
        coordinator.execute(RecoveryCosts(), lambda: None)
    coordinator.close()


@pytest.mark.parametrize(
    "failure", ["anchor", "worker", "expired", "rss", "journal", "unknown", "identity"]
)
def test_should_refuse_publication_when_authority_changes_during_preparation(failure):
    coordinator, journal, observer, anchor, worker, log = fixture()

    def prepare():
        if failure == "anchor":
            anchor.ended = True
        if failure == "worker":
            worker.ended = True
        if failure == "unknown":
            worker.ended = None
        if failure == "identity":
            worker.identity = RecoveryProcessIdentity(99, 999)
        if failure == "expired":
            observer.now = 1100
        if failure == "rss":
            observer.hook = lambda o: replace(o, rss_bytes=1025, peak_rss_bytes=1025)
        if failure == "journal":
            journal.record = replace(
                journal.record, metadata=replace(journal.record.metadata, owner_id="external")
            )
        return "prepared"

    with pytest.raises(ValueError):
        coordinator.execute(
            RecoveryCosts(copy_bytes=1), prepare, lambda v: log.append(("publish", v))
        )
    assert not any(name == "publish" for name, _ in log)
    assert journal.record.metadata.usage.copied_bytes == 41
    coordinator.close()


def test_should_detect_postpublication_loss_without_claiming_side_effect_rollback():
    coordinator, journal, _, anchor, _, log = fixture()

    def publish(value):
        log.append(("publish", value))
        anchor.ended = True

    with pytest.raises(ValueError):
        coordinator.execute(RecoveryCosts(copy_bytes=1), lambda: "prepared", publish)
    assert ("publish", "prepared") in log
    assert journal.record.metadata.usage.copied_bytes == 41
    coordinator.close()


def test_should_reject_nested_lane_without_corrupting_outer_sequence():
    coordinator, journal, _, _, _, _ = fixture()

    def prepare():
        with pytest.raises(ValueError, match="busy"):
            coordinator.execute(RecoveryCosts(), lambda: None)
        with pytest.raises(ValueError, match="busy"):
            coordinator.close()
        return "outer"

    assert coordinator.execute(RecoveryCosts(copy_bytes=1), prepare) == "outer"
    assert journal.record.metadata.usage.copied_bytes == 41
    coordinator.close()


def test_should_refuse_simultaneous_lane_without_queue_or_duplicate_charge():
    coordinator, journal, _, _, _, _ = fixture()
    entered, release = Event(), Event()
    outcomes = []

    def prepare():
        entered.set()
        assert release.wait(timeout=2)
        return "first"

    def run():
        try:
            outcomes.append(coordinator.execute(RecoveryCosts(copy_bytes=1), prepare))
        except BaseException as error:
            outcomes.append(error)

    thread = Thread(target=run)
    thread.start()
    try:
        assert entered.wait(timeout=2)
        with pytest.raises(ValueError, match="busy"):
            coordinator.execute(RecoveryCosts(copy_bytes=1), lambda: "second")
    finally:
        release.set()
        thread.join(timeout=2)
    assert not thread.is_alive() and outcomes == ["first"]
    assert journal.record.metadata.usage.copied_bytes == 41
    coordinator.close()


def test_should_require_ended_old_registration_and_commit_next_live_owner():
    coordinator, journal, _, anchor, worker, log = fixture()
    worker.ended = True
    replacement = Probe(RecoveryProcessIdentity(30, 300), log)
    coordinator.handoff(replacement, "owner-b")
    assert worker.closed and not replacement.closed
    assert journal.record.worker == replacement.identity
    assert journal.record.metadata.owner_epoch == 5
    assert journal.record.metadata.usage.checkpoint_attempts == 3
    assert coordinator.execute(RecoveryCosts(copy_bytes=1), lambda: "new") == "new"
    coordinator.close()
    assert replacement.closed and anchor.closed


@pytest.mark.parametrize(
    "failure", ["old-live", "new-ended", "same-worker", "same-anchor", "unknown-new"]
)
def test_should_refuse_unsupported_handoff_without_dispatch_or_new_generation(failure):
    coordinator, journal, _, anchor, worker, log = fixture()
    worker.ended = failure != "old-live"
    identity = (
        worker.identity
        if failure == "same-worker"
        else anchor.identity
        if failure == "same-anchor"
        else RecoveryProcessIdentity(30, 300)
    )
    replacement = Probe(identity, log)
    if failure == "new-ended":
        replacement.ended = True
    if failure == "unknown-new":
        replacement.ended = None
    with pytest.raises(ValueError):
        coordinator.handoff(replacement, "owner-b")
    assert journal.record.metadata.owner_epoch == 4
    coordinator.close()
    assert replacement.closed and anchor.closed and worker.closed


@pytest.mark.parametrize("failure", ["identity", "journal", "anchor-ended", "observation"])
def test_should_close_owned_registrations_when_initial_authority_is_unproved(failure):
    record, _ = authority_fixture()
    log: list[tuple[str, object]] = []
    anchor, worker = Probe(record.anchor, log), Probe(record.worker, log)
    observer, journal = Observer(record, log), MemoryJournal(record, log)
    if failure == "identity":
        worker.identity = RecoveryProcessIdentity(99, 999)
    if failure == "journal":
        journal.record = replace(record, metadata=replace(record.metadata, sequence=8))
    if failure == "anchor-ended":
        anchor.ended = True
    if failure == "observation":
        observer.hook = lambda o: replace(o, clock_epoch="foreign")
    with pytest.raises(ValueError):
        RecoveryCoordinator(record, journal, observer, anchor, worker)
    assert anchor.closed and worker.closed


def test_should_attempt_all_terminal_closures_and_report_failure_without_reopening():
    coordinator, _, _, anchor, worker, _ = fixture()
    anchor.close_error = OSError("injected close")
    with pytest.raises(BaseExceptionGroup):
        coordinator.close()
    assert anchor.closed and worker.closed
    coordinator.close()
    with pytest.raises(ValueError, match="closed"):
        coordinator.execute(RecoveryCosts(), lambda: None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("updates", True),
        ("updates", 2),
        ("copy_bytes", -1),
        ("grants", 2**63),
        ("checkpoint_attempts", False),
    ],
)
def test_should_refuse_invalid_exact_costs(field, value):
    with pytest.raises(ValueError):
        RecoveryCosts(**{field: value})


def test_should_revalidate_corrupted_costs_and_external_stale_state_before_work():
    coordinator, journal, observer, _, _, log = fixture()
    costs = RecoveryCosts(copy_bytes=1)
    object.__setattr__(costs, "copy_bytes", False)
    with pytest.raises(ValueError):
        coordinator.execute(costs, lambda: None)
    coordinator.close()
    coordinator, journal, observer, _, _, log = fixture()
    journal.record = plan_reservation(
        journal.record, observer.observe(coordinator._worker), copy_bytes=1
    ).proposed
    with pytest.raises(ValueError, match="authority"):
        coordinator.execute(RecoveryCosts(copy_bytes=1), lambda: log.append(("prepare", 0)))
    assert not any(name == "prepare" for name, _ in log)
    coordinator.close()


@pytest.mark.parametrize("failure", ["clock", "observer", "owner", "time", "peak"])
def test_should_refuse_foreign_or_regressing_fresh_observation_before_charge(failure):
    coordinator, journal, observer, _, _, log = fixture()

    def mutate(observation):
        if failure == "clock":
            return replace(observation, clock_epoch="foreign")
        if failure == "observer":
            return replace(observation, observer=RecoveryProcessIdentity(99, 999))
        if failure == "owner":
            return replace(observation, previous_owner=RecoveryProcessIdentity(99, 999))
        if failure == "time":
            return replace(observation, now_ns=500)
        return replace(observation, rss_bytes=512, peak_rss_bytes=512)

    observer.hook = mutate
    with pytest.raises(ValueError):
        coordinator.execute(RecoveryCosts(copy_bytes=1), lambda: log.append(("prepare", 0)))
    assert journal.record.metadata.usage.copied_bytes == 40
    assert not any(name == "prepare" for name, _ in log)
    coordinator.close()


def test_should_refuse_anchor_loss_inside_observation_before_charge():
    coordinator, journal, observer, anchor, _, log = fixture()

    def lose_anchor(observation):
        anchor.ended = True
        return observation

    observer.hook = lose_anchor
    with pytest.raises(ValueError):
        coordinator.execute(RecoveryCosts(copy_bytes=1), lambda: log.append(("prepare", 0)))
    assert journal.record.metadata.usage.copied_bytes == 40
    assert not any(name == "prepare" for name, _ in log)
    coordinator.close()


@pytest.mark.parametrize("failure", ["unpersisted-success", "worker-ended"])
def test_should_recheck_commit_and_worker_before_dispatch(failure):
    coordinator, journal, _, _, worker, log = fixture()

    def advance(change):
        if failure == "worker-ended":
            journal.record = change.proposed
            worker.ended = True
        return True

    journal.hook = advance
    with pytest.raises(ValueError):
        coordinator.execute(RecoveryCosts(copy_bytes=1), lambda: log.append(("prepare", 0)))
    assert not any(name == "prepare" for name, _ in log)
    assert journal.record.metadata.usage.copied_bytes == (41 if failure == "worker-ended" else 40)
    coordinator.close()


def test_should_keep_committed_handoff_charge_when_new_worker_ends_during_commit():
    coordinator, journal, _, anchor, worker, log = fixture()
    worker.ended = True
    replacement = Probe(RecoveryProcessIdentity(30, 300), log)

    def advance(change):
        journal.record = change.proposed
        replacement.ended = True
        return True

    journal.hook = advance
    with pytest.raises(ValueError):
        coordinator.handoff(replacement, "owner-b")
    assert journal.record.metadata.owner_epoch == 5
    assert journal.record.metadata.usage.checkpoint_attempts == 3
    with pytest.raises(ValueError, match="failed"):
        coordinator.execute(RecoveryCosts(), lambda: None)
    coordinator.close()
    assert anchor.closed and worker.closed and replacement.closed


def test_should_dispatch_single_reserved_update_without_inventing_completion():
    coordinator, journal, _, _, _, log = fixture()
    assert coordinator.execute(RecoveryCosts(updates=1), lambda: "pending") == "pending"
    assert journal.record.metadata.uncertain_work
    assert journal.record.metadata.usage.updates_completed == 3
    with pytest.raises(ValueError, match="uncertain"):
        coordinator.execute(RecoveryCosts(), lambda: log.append(("prepare", 0)))
    assert not any(name == "prepare" for name, _ in log)
    coordinator.close()


def test_should_preserve_initial_failure_and_attempt_both_cleanup_failures():
    record, _ = authority_fixture()
    log: list[tuple[str, object]] = []
    anchor, worker = Probe(record.anchor, log), Probe(record.worker, log)
    anchor.ended = True
    anchor.close_error, worker.close_error = OSError("anchor close"), OSError("worker close")
    with pytest.raises(BaseExceptionGroup) as result:
        RecoveryCoordinator(
            record, MemoryJournal(record, log), Observer(record, log), anchor, worker
        )
    assert isinstance(result.value.exceptions[0], ValueError)
    assert anchor.closed and worker.closed


def test_should_close_context_after_observer_keyboard_interrupt_and_refuse_retry():
    coordinator, _, observer, anchor, worker, _ = fixture()

    def fail(observation):
        raise KeyboardInterrupt("observer interrupted")

    observer.hook = fail
    with pytest.raises(KeyboardInterrupt):
        with coordinator:
            coordinator.execute(RecoveryCosts(), lambda: None)
    assert anchor.closed and worker.closed
    with pytest.raises(ValueError, match="closed"):
        coordinator.execute(RecoveryCosts(), lambda: None)
