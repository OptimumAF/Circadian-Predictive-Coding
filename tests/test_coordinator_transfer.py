"""Synthetic trusted supervisor transitions and real bounded SQLite outcomes.

These values test metadata semantics; they provide no physical process proof.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import sqlite3
from typing import Any

import pytest

from src.app.recovery_coordinator import RecoveryCoordinator
from src.core.coordinator_transfer import (
    SupervisorRecoveryWitness,
    CoordinatorTransferObservation,
    CoordinatorTransfer,
    CoordinatorTransferFailure,
    plan_coordinator_transfer,
    validate_coordinator_transfer,
    validate_transfer_failure,
    transfer_failure_states,
)
from src.core.recovery_authority import plan_reservation, validate_authority_floor
from src.core.recovery_coordination import RecoveryCosts
from src.core.recovery_observation import (
    RecoveryProcessIdentity,
    RecoveryHostObservation,
    anchored_clock_epoch,
)
from src.core.recovery_reporting import plan_report
import src.infra.sqlite_recovery_journal as storage
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal
from test_recovery_authority import fixture as old_fixture
from test_recovery_coordinator import Probe, Observer


def fixture(*, zero=False, reuse_pid=False):
    old, _ = old_fixture()
    m = old.metadata
    old = replace(
        old,
        metadata=replace(
            m,
            manifest=replace(m.manifest, native_sha256="a" * 64, payload_sha256="a" * 64),
            usage=replace(
                m.usage, updates_admitted=0 if zero else 3, updates_completed=0 if zero else 3
            ),
        ),
    )
    supervisor = RecoveryProcessIdentity(30, 300)
    anchor = RecoveryProcessIdentity(10 if reuse_pid else 40, 400)
    worker = RecoveryProcessIdentity(20 if reuse_pid else 50, 500)
    witness = SupervisorRecoveryWitness(old, supervisor)
    observation = CoordinatorTransferObservation(
        RecoveryHostObservation(
            anchored_clock_epoch(supervisor), 500, 580, 600, supervisor, old.anchor, True
        ),
        RecoveryHostObservation(
            anchored_clock_epoch(supervisor), 600, 650, 700, supervisor, old.worker, True
        ),
        RecoveryHostObservation(anchored_clock_epoch(anchor), 700, 640, 680, anchor, worker, False),
    )
    return witness, observation


def change():
    witness, observation = fixture()
    return plan_coordinator_transfer(witness, observation, "owner-b")


def failure(transfer=None, *, ended=True, overshoot=False):
    transfer = change() if transfer is None else transfer
    supervisor = transfer.witness.supervisor
    terminal = RecoveryHostObservation(
        anchored_clock_epoch(supervisor),
        2000 if overshoot else 750,
        1800 if overshoot else 720,
        2048 if overshoot else 800,
        supervisor,
        transfer.proposed.anchor,
        ended,
    )
    return CoordinatorTransferFailure(transfer, terminal)


def tamper(value, path, replacement):
    result = deepcopy(value)
    names = path.split(".")
    target = result
    for name in names[:-1]:
        # Each branch is an independent claimed record. Breaking preserved
        # deepcopy aliases keeps a proposal-only fault off the trusted witness.
        child = deepcopy(getattr(target, name))
        object.__setattr__(target, name, child)
        target = child
    object.__setattr__(target, names[-1], replacement)
    return result


@pytest.mark.parametrize("zero,reuse_pid", [(False, False), (True, False), (False, True)])
def test_should_transfer_only_declared_fields_without_renewing_any_original_allowance(
    zero, reuse_pid
):
    witness, observation = fixture(zero=zero, reuse_pid=reuse_pid)
    before = asdict(witness)
    result = plan_coordinator_transfer(witness, observation, "owner-b")
    old = witness.authority
    expected = replace(
        old,
        anchor=observation.replacement.observer,
        worker=observation.replacement.previous_owner,
        metadata=replace(
            old.metadata,
            clock_epoch=observation.replacement.clock_epoch,
            observed_ns=700,
            sequence=old.metadata.sequence + 1,
            owner_epoch=old.metadata.owner_epoch + 1,
            owner_id="owner-b",
            usage=replace(
                old.metadata.usage,
                checkpoint_attempts=old.metadata.usage.checkpoint_attempts + 1,
                peak_rss_bytes=700,
            ),
        ),
    )
    assert type(result) is CoordinatorTransfer
    assert result.expected == old and result.proposed == expected
    assert result.witness == witness and result.observation == observation
    validate_coordinator_transfer(result)
    assert asdict(witness) == before
    with pytest.raises(ValueError):
        validate_authority_floor(old, result.proposed)
    if reuse_pid:
        assert old.anchor.pid == result.proposed.anchor.pid and old.anchor != result.proposed.anchor
        assert old.worker.pid == result.proposed.worker.pid and old.worker != result.proposed.worker


@pytest.mark.parametrize(
    "fault",
    [
        "stopped",
        "uncertain",
        "managed",
        "supervisor_anchor",
        "supervisor_worker",
        "same_owner",
        "exhausted_attempts",
    ],
)
def test_should_refuse_unsupported_original_witness_before_planning(fault):
    witness, observation = fixture()
    old = witness.authority
    with pytest.raises(ValueError):
        if fault == "same_owner":
            plan_coordinator_transfer(witness, observation, old.metadata.owner_id)
            return
        if fault == "stopped":
            old = replace(old, metadata=replace(old.metadata, stopped=True))
        elif fault == "uncertain":
            old = replace(
                old,
                metadata=replace(
                    old.metadata,
                    uncertain_work=True,
                    usage=replace(old.metadata.usage, updates_admitted=4),
                ),
            )
        elif fault == "managed":
            old = replace(
                old,
                metadata=replace(
                    old.metadata, manifest=replace(old.metadata.manifest, native_sha256="b" * 64)
                ),
            )
        elif fault == "exhausted_attempts":
            old = replace(
                old,
                metadata=replace(
                    old.metadata, usage=replace(old.metadata.usage, checkpoint_attempts=4)
                ),
            )
        supervisor = (
            old.anchor
            if fault == "supervisor_anchor"
            else old.worker
            if fault == "supervisor_worker"
            else witness.supervisor
        )
        plan_coordinator_transfer(
            SupervisorRecoveryWitness(old, supervisor), observation, "owner-b"
        )


@pytest.mark.parametrize(
    "path,value",
    [
        ("anchor_ended.previous_owner_ended", False),
        ("worker_ended.previous_owner_ended", None),
        ("replacement.previous_owner_ended", True),
        ("replacement.previous_owner_ended", 0),
        ("anchor_ended.observer", RecoveryProcessIdentity(31, 301)),
        ("worker_ended.previous_owner", RecoveryProcessIdentity(21, 201)),
        ("anchor_ended.clock_epoch", "foreign"),
        ("replacement.clock_epoch", "foreign"),
        ("anchor_ended.now_ns", 399),
        ("worker_ended.now_ns", 499),
        ("replacement.now_ns", 599),
        ("replacement.now_ns", 1100),
        ("replacement.now_ns", True),
        ("worker_ended.peak_rss_bytes", 1025),
        ("replacement.rss_bytes", True),
        ("replacement.observer", RecoveryProcessIdentity(30, 999)),
        ("replacement.previous_owner", RecoveryProcessIdentity(40, 999)),
    ],
)
def test_should_refuse_wrong_observation_binding_flags_time_or_original_resource_caps(path, value):
    witness, observation = fixture()
    invalid = tamper(observation, path, value)
    with pytest.raises(ValueError):
        plan_coordinator_transfer(witness, invalid, "owner-b")


@pytest.mark.parametrize("field", ["observer", "previous_owner"])
def test_should_refuse_reusing_any_exact_old_identity_as_replacement(field):
    witness, observation = fixture()
    identity = witness.authority.anchor if field == "observer" else witness.authority.worker
    replacement = replace(
        observation.replacement,
        **{
            field: identity,
            **({"clock_epoch": anchored_clock_epoch(identity)} if field == "observer" else {}),
        },
    )
    with pytest.raises(ValueError):
        plan_coordinator_transfer(witness, replace(observation, replacement=replacement), "owner-b")


@pytest.mark.parametrize("collision", ["supervisor_anchor", "anchor_worker"])
def test_should_refuse_live_pid_collision_even_when_full_ids_and_clock_tags_are_distinct(collision):
    witness, observation = fixture()
    replacement = observation.replacement
    if collision == "supervisor_anchor":
        anchor = RecoveryProcessIdentity(witness.supervisor.pid, 999)
        replacement = replace(
            replacement, observer=anchor, clock_epoch=anchored_clock_epoch(anchor)
        )
    else:
        replacement = replace(
            replacement, previous_owner=RecoveryProcessIdentity(replacement.observer.pid, 999)
        )
    with pytest.raises(ValueError):
        plan_coordinator_transfer(witness, replace(observation, replacement=replacement), "owner-b")


def test_should_accept_inclusive_attempt_and_rss_caps_and_last_admitted_time():
    witness, observation = fixture()
    old = witness.authority
    witness = SupervisorRecoveryWitness(
        replace(
            old,
            metadata=replace(
                old.metadata, usage=replace(old.metadata.usage, checkpoint_attempts=3)
            ),
        ),
        witness.supervisor,
    )
    replacement = replace(observation.replacement, now_ns=1099, rss_bytes=1024, peak_rss_bytes=1024)
    result = plan_coordinator_transfer(
        witness, replace(observation, replacement=replacement), "owner-b"
    )
    assert result.proposed.metadata.usage.checkpoint_attempts == 4
    assert result.proposed.metadata.usage.peak_rss_bytes == 1024
    assert result.proposed.metadata.started_ns == 100


@pytest.mark.parametrize(
    "path,value",
    [
        ("proposed.metadata.started_ns", 101),
        ("proposed.metadata.session_id", "other"),
        ("proposed.metadata.sequence", 9),
        ("proposed.metadata.owner_epoch", 6),
        ("proposed.metadata.owner_id", "owner-a"),
        ("proposed.metadata.observed_ns", 701),
        ("proposed.metadata.event_tick", 13),
        ("proposed.metadata.clock_epoch", "foreign"),
        ("proposed.metadata.usage.copied_bytes", 39),
        ("proposed.metadata.usage.grants", 2),
        ("proposed.metadata.usage.checkpoint_attempts", 2),
        ("proposed.metadata.usage.peak_rss_bytes", 699),
        ("proposed.metadata.usage.updates_admitted", 4),
        ("proposed.metadata.usage.updates_completed", 2),
        ("proposed.metadata.stopped", True),
        ("proposed.metadata.uncertain_work", True),
        ("proposed.metadata.limits.max_updates", 11),
        ("proposed.metadata.manifest.actor_sha256", "f" * 64),
        ("expected.metadata.sequence", 6),
        ("witness.supervisor.pid", True),
        ("witness.authority.metadata.usage.grants", True),
        ("proposed.metadata.uncertain_work", 0),
    ],
)
def test_should_revalidate_tampered_transfer_and_refuse_every_undeclared_change(path, value):
    original = change()
    before = asdict(original)
    invalid = tamper(original, path, value)
    assert asdict(original) == before
    if path.startswith("proposed."):
        assert invalid.expected == original.expected
        assert invalid.witness == original.witness
    with pytest.raises(ValueError):
        validate_coordinator_transfer(invalid)


@pytest.mark.parametrize(
    "path",
    [
        "witness",
        "expected",
        "proposed",
        "observation",
        "witness.authority.metadata",
        "witness.authority.metadata.usage",
        "observation.anchor_ended",
        "observation.worker_ended",
        "observation.replacement",
    ],
)
@pytest.mark.parametrize("fault", ["missing", "extra", "mapping"])
def test_should_refuse_incomplete_nested_schema_with_value_error_before_field_access(path, fault):
    valid = change()
    target = valid
    for name in path.split("."):
        target = getattr(target, name)
    if fault == "mapping":
        invalid = tamper(valid, path, asdict(target))
    else:
        invalid = deepcopy(valid)
        node = invalid
        for name in path.split("."):
            node = getattr(node, name)
        if fault == "missing":
            object.__delattr__(node, next(iter(vars(node))))
        else:
            object.__setattr__(node, "unexpected", False)
    with pytest.raises(ValueError):
        validate_coordinator_transfer(invalid)


@pytest.mark.parametrize("ended", [False, True])
def test_should_reconcile_both_known_outcomes_terminal_only_with_one_transfer_charge(ended):
    witness = failure(ended=ended, overshoot=True)
    validate_transfer_failure(witness)
    stopped = transfer_failure_states(witness)
    for base, actual in zip(
        (witness.transfer.expected, witness.transfer.proposed), stopped, strict=True
    ):
        expected = replace(
            base,
            metadata=replace(
                base.metadata,
                sequence=base.metadata.sequence + 1,
                observed_ns=2000,
                stopped=True,
                usage=replace(base.metadata.usage, checkpoint_attempts=3, peak_rss_bytes=2048),
            ),
        )
        assert actual == expected and actual.metadata.uncertain_work is False
        assert (
            actual.metadata.usage.updates_admitted == actual.metadata.usage.updates_completed == 3
        )
        with pytest.raises(ValueError):
            plan_reservation(actual, witness.transfer.observation.replacement, copy_bytes=1)


@pytest.mark.parametrize(
    "path,value",
    [
        ("observation.observer", RecoveryProcessIdentity(31, 301)),
        ("observation.previous_owner", RecoveryProcessIdentity(50, 500)),
        ("observation.clock_epoch", "foreign"),
        ("observation.now_ns", 699),
        ("observation.previous_owner_ended", None),
        ("observation.previous_owner_ended", 1),
        ("transfer.proposed.metadata.owner_epoch", 99),
    ],
)
def test_should_refuse_invalid_failure_witness_before_path_lock_or_sqlite(
    monkeypatch, tmp_path, path, value
):
    invalid = tamper(failure(), path, value)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid witness performed IO")

    monkeypatch.setattr(storage, "_connect", forbidden)
    monkeypatch.setattr(storage, "authority_writer_lock", forbidden)
    with pytest.raises(ValueError):
        SqliteRecoveryJournal.reconcile_coordinator_transfer(tmp_path / "absent.sqlite", invalid)
    assert list(tmp_path.iterdir()) == []


def test_should_transfer_explicit_sqlite_floor_once_and_fence_old_adapters_before_io(
    tmp_path, monkeypatch
):
    transition = change()
    path = tmp_path / "authority.sqlite"
    journal = SqliteRecoveryJournal.create(path, transition.expected)
    stale = SqliteRecoveryJournal(path, transition.expected)
    assert journal.transfer_coordinator(transition) is True
    assert journal.read() == transition.proposed
    with pytest.raises(ValueError):
        stale.transfer_coordinator(transition)
    with pytest.raises(ValueError, match="failed"):
        stale.read()

    def forbidden(*args, **kwargs):
        pytest.fail("mismatched independent floor performed IO")

    monkeypatch.setattr(storage, "_connect", forbidden)
    monkeypatch.setattr(storage, "authority_writer_lock", forbidden)
    with pytest.raises(ValueError):
        journal.transfer_coordinator(transition)


def test_should_return_false_only_for_same_anchor_newer_cas_without_transferring(tmp_path):
    transition = change()
    path = tmp_path / "authority.sqlite"
    journal = SqliteRecoveryJournal.create(path, transition.expected)
    stale = SqliteRecoveryJournal(path, transition.expected)
    old = transition.expected
    host = RecoveryHostObservation(
        old.metadata.clock_epoch, 450, 550, 550, old.anchor, old.worker, False
    )
    reported = plan_report(old, host)
    assert journal.report(reported)
    assert stale.transfer_coordinator(transition) is False
    assert stale.read() == reported.proposed
    assert reported.proposed.anchor == old.anchor


def test_should_continue_existing_coordinator_ports_using_new_anchor_and_original_quotas(tmp_path):
    transition = change()
    journal = SqliteRecoveryJournal.create(tmp_path / "authority.sqlite", transition.expected)
    assert journal.transfer_coordinator(transition)
    current = transition.proposed
    log: list[Any] = []
    anchor, worker = Probe(current.anchor, log), Probe(current.worker, log)
    observer = Observer(current, log)
    observer.now = 700
    observer.hook = lambda observation: replace(observation, rss_bytes=720, peak_rss_bytes=720)
    calls = []
    coordinator = RecoveryCoordinator(current, journal, observer, anchor, worker)

    def prepare():
        calls.append("prepare")
        return "ready"

    try:
        result = coordinator.execute(
            RecoveryCosts(copy_bytes=5, checkpoint_attempts=1),
            prepare,
            lambda value: calls.append(value),
        )
        assert result == "ready" and calls == ["prepare", "ready"]
        saved = journal.read()
        assert saved.anchor == current.anchor and saved.worker == current.worker
        assert saved.metadata.session_id == current.metadata.session_id
        assert saved.metadata.started_ns == 100 and saved.metadata.limits == current.metadata.limits
        assert (
            saved.metadata.usage.copied_bytes == 45
            and saved.metadata.usage.checkpoint_attempts == 4
        )
        assert saved.metadata.usage.updates_admitted == saved.metadata.usage.updates_completed == 3
        with pytest.raises(ValueError):
            coordinator.execute(RecoveryCosts(checkpoint_attempts=1), lambda: calls.append("retry"))
        assert calls == ["prepare", "ready"]
        assert journal.read().metadata.stopped is True
    finally:
        coordinator.close()
    assert anchor.closed and worker.closed


@pytest.mark.parametrize("after_commit", [False, True])
def test_should_keep_ambiguous_adapter_failed_and_reconcile_only_known_terminal_branch(
    tmp_path, monkeypatch, after_commit
):
    transition = change()
    path = tmp_path / "authority.sqlite"
    journal = SqliteRecoveryJournal.create(path, transition.expected)
    original = storage._connect

    class Fault(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql == "COMMIT" and not after_commit:
                raise sqlite3.OperationalError("before transfer COMMIT")
            result = super().execute(sql, parameters)
            if sql == "COMMIT":
                raise sqlite3.OperationalError("after transfer COMMIT")
            return result

    def connect(target):
        return storage._configure(
            sqlite3.connect(
                target.as_uri() + "?mode=rw",
                uri=True,
                timeout=0,
                isolation_level=None,
                factory=Fault,
            )
        )

    monkeypatch.setattr(storage, "_connect", connect)
    with pytest.raises(sqlite3.OperationalError):
        journal.transfer_coordinator(transition)
    monkeypatch.setattr(storage, "_connect", original)
    with pytest.raises(ValueError, match="failed"):
        journal.transfer_coordinator(transition)
    witness = failure(transition, overshoot=True)
    expected = transfer_failure_states(witness)[int(after_commit)]
    actual = SqliteRecoveryJournal.reconcile_coordinator_transfer(path, witness)
    assert actual == expected and actual.metadata.stopped
    assert SqliteRecoveryJournal.reconcile_coordinator_transfer(path, witness) == actual
    assert SqliteRecoveryJournal(path, actual).read() == actual
    with pytest.raises(ValueError, match="failed"):
        journal.read()


@pytest.mark.parametrize("branch", [0, 1])
def test_should_allow_exact_terminal_outcome_idempotently_without_second_attempt_charge(
    tmp_path, branch
):
    witness = failure()
    expected = transfer_failure_states(witness)[branch]
    path = tmp_path / "authority.sqlite"
    SqliteRecoveryJournal.create(path, expected)
    assert SqliteRecoveryJournal.reconcile_coordinator_transfer(path, witness) == expected
    assert SqliteRecoveryJournal.reconcile_coordinator_transfer(path, witness) == expected
    assert expected.metadata.usage.checkpoint_attempts == 3


@pytest.mark.parametrize("fault", ["newer", "forged", "foreign"])
def test_should_refuse_unknown_disk_outcome_without_rewriting_it(tmp_path, fault):
    witness = failure()
    current = witness.transfer.proposed
    if fault == "newer":
        current = replace(
            current,
            metadata=replace(
                current.metadata, sequence=current.metadata.sequence + 1, observed_ns=701
            ),
        )
    elif fault == "forged":
        current = replace(
            current,
            metadata=replace(
                current.metadata, manifest=replace(current.metadata.manifest, actor_sha256="f" * 64)
            ),
        )
    else:
        current = replace(current, metadata=replace(current.metadata, session_id="foreign"))
    path = tmp_path / "authority.sqlite"
    SqliteRecoveryJournal.create(path, current)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        SqliteRecoveryJournal.reconcile_coordinator_transfer(path, witness)
    assert path.read_bytes() == before
    assert SqliteRecoveryJournal(path, current).read() == current
