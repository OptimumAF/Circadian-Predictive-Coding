"""One-admission completion, explicit floor advancement and terminal ambiguity.

Scalar fake authority and bounded SQLite fixtures execute no native models or
processes. Legacy uncertain-update behavior remains a separate explicit control.
"""

from dataclasses import asdict, replace
import sqlite3
from typing import Any

import pytest

from src.app.recovery_coordinator import RecoveryCoordinator
from src.core.recovery_authority import (
    plan_handoff,
    plan_reservation,
    validate_authority_floor,
)
from src.core.recovery_completion import (
    AuthorityCompletion,
    plan_completion,
    validate_authority_completion,
)
from src.core.recovery_coordination import RecoveryCosts
from src.core.recovery_observation import RecoveryProcessIdentity
from src.core.recovery_reporting import (
    FailureWitness,
    TerminalReportingFailure,
    plan_report,
    validate_failure_witness,
    witnessed_terminal_states,
)
import src.infra.sqlite_recovery_journal as sqlite_module
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal
from test_recovery_authority import fixture as authority_fixture
from test_recovery_coordinator import MemoryJournal, Observer, Probe

A_HASH, B_HASH = "a" * 64, "b" * 64


def original():
    record, observation = authority_fixture()
    manifest = replace(record.metadata.manifest, payload_sha256=A_HASH, native_sha256=A_HASH)
    return replace(record, metadata=replace(record.metadata, manifest=manifest)), observation


def manifest_for(record, digest=B_HASH):
    return replace(record.metadata.manifest, payload_sha256=digest, native_sha256=digest)


def pending_fixture():
    record, observation = original()
    admission = plan_reservation(
        record, observation, updates=1, copy_bytes=5, checkpoint_attempts=1
    )
    reported = plan_report(
        admission.proposed, replace(observation, now_ns=700, rss_bytes=700, peak_rss_bytes=700)
    ).proposed
    final = replace(observation, now_ns=800, rss_bytes=720, peak_rss_bytes=720)
    return admission, reported, final


@pytest.mark.parametrize("digest", (A_HASH, B_HASH))
def test_should_complete_only_one_admission_after_reports_and_preserve_all_spent_costs(digest):
    admission, pending, observation = pending_fixture()
    manifest = manifest_for(pending, digest)
    change = plan_completion(admission, pending, observation, manifest)
    expected = replace(
        pending,
        metadata=replace(
            pending.metadata,
            sequence=pending.metadata.sequence + 1,
            observed_ns=observation.now_ns,
            manifest=manifest,
            uncertain_work=False,
            usage=replace(
                pending.metadata.usage,
                updates_completed=pending.metadata.usage.updates_completed + 1,
                peak_rss_bytes=observation.peak_rss_bytes,
            ),
        ),
    )
    assert type(change) is AuthorityCompletion
    assert (
        change.admission == admission and change.expected == pending and change.proposed == expected
    )
    validate_authority_completion(change)
    assert (
        change.proposed.metadata.usage.updates_completed
        == change.proposed.metadata.usage.updates_admitted
        == 4
    )
    assert pending.metadata.usage.updates_completed == 3 and pending.metadata.uncertain_work
    with pytest.raises(ValueError):
        validate_authority_floor(pending, change.proposed)


@pytest.mark.parametrize(
    "fault",
    (
        "admission_family",
        "no_update",
        "handoff",
        "managed_original",
        "not_pending",
        "stopped",
        "sequence_regression",
        "time_regression",
        "peak_regression",
        "copy_change",
        "extra_admission",
        "completed_change",
        "manifest_change",
        "owner_change",
        "epoch_change",
        "session_change",
        "caps_change",
    ),
)
def test_should_refuse_wrong_admission_or_nonreport_changes_to_pending_state(fault):
    admission, pending, observation = pending_fixture()
    with pytest.raises(ValueError):
        if fault == "admission_family":
            admission = asdict(admission)
        elif fault == "no_update":
            record, initial = original()
            admission = plan_reservation(record, initial, copy_bytes=5)
            pending = admission.proposed
        elif fault == "handoff":
            record, initial = original()
            admission = plan_handoff(
                record,
                replace(initial, previous_owner_ended=True),
                RecoveryProcessIdentity(30, 300),
                "owner-b",
            )
            pending = admission.proposed
        elif fault == "managed_original":
            record, initial = authority_fixture()
            admission = plan_reservation(record, initial, updates=1)
            pending = admission.proposed
        elif fault == "not_pending":
            pending = admission.expected
        elif fault == "stopped":
            pending = plan_report(pending, None, terminal=True).proposed
        else:
            m = pending.metadata
            if fault == "sequence_regression":
                m = replace(m, sequence=admission.proposed.metadata.sequence - 1)
            elif fault == "time_regression":
                m = replace(m, observed_ns=admission.proposed.metadata.observed_ns - 1)
            elif fault == "peak_regression":
                m = replace(m, usage=replace(m.usage, peak_rss_bytes=600))
            elif fault == "copy_change":
                m = replace(m, usage=replace(m.usage, copied_bytes=m.usage.copied_bytes + 1))
            elif fault == "extra_admission":
                m = replace(m, usage=replace(m.usage, updates_admitted=5))
            elif fault == "completed_change":
                m = replace(m, usage=replace(m.usage, updates_completed=4))
            elif fault == "manifest_change":
                m = replace(m, manifest=manifest_for(pending))
            elif fault == "owner_change":
                m = replace(m, owner_id="foreign")
            elif fault == "epoch_change":
                m = replace(m, owner_epoch=m.owner_epoch + 1)
            elif fault == "session_change":
                m = replace(m, session_id="foreign")
            else:
                m = replace(m, limits=replace(m.limits, max_updates=11))
            pending = replace(pending, metadata=m)
        plan_completion(admission, pending, observation, manifest_for(pending))


@pytest.mark.parametrize(
    "field",
    (
        "source_sha256",
        "policy_sha256",
        "inbox_sha256",
        "consolidation_sha256",
        "lifecycle_sha256",
        "actor_sha256",
        "sharing_sha256",
        "pair_mismatch",
        "family",
    ),
)
def test_should_update_only_matching_native_payload_hash_pair(field):
    admission, pending, observation = pending_fixture()
    manifest: Any = manifest_for(pending)
    if field == "family":
        manifest = asdict(manifest)
    else:
        manifest = replace(
            manifest, **{("native_sha256" if field == "pair_mismatch" else field): "c" * 64}
        )
    with pytest.raises(ValueError):
        plan_completion(admission, pending, observation, manifest)


@pytest.mark.parametrize(
    "fault",
    ("ended", "unknown", "owner", "anchor", "epoch", "time", "peak", "elapsed", "rss", "family"),
)
def test_should_require_fresh_live_bound_host_and_preserve_original_caps(fault):
    admission, pending, observation = pending_fixture()
    with pytest.raises(ValueError):
        if fault == "ended":
            observation = replace(observation, previous_owner_ended=True)
        elif fault == "unknown":
            observation = replace(observation, previous_owner=None, previous_owner_ended=None)
        elif fault == "owner":
            observation = replace(observation, previous_owner=RecoveryProcessIdentity(30, 300))
        elif fault == "anchor":
            observation = replace(observation, observer=RecoveryProcessIdentity(30, 300))
        elif fault == "epoch":
            observation = replace(observation, clock_epoch="foreign")
        elif fault == "time":
            observation = replace(observation, now_ns=pending.metadata.observed_ns - 1)
        elif fault == "peak":
            observation = replace(observation, rss_bytes=690, peak_rss_bytes=690)
        elif fault == "elapsed":
            observation = replace(observation, now_ns=1100)
        elif fault == "rss":
            observation = replace(observation, rss_bytes=1025, peak_rss_bytes=1025)
        else:
            observation = asdict(observation)
        plan_completion(admission, pending, observation, manifest_for(pending))


@pytest.mark.parametrize(
    "fault",
    (
        "family",
        "no_completion",
        "extra_completion",
        "uncertain",
        "stopped",
        "sequence",
        "copied",
        "grants",
        "attempts",
        "actor",
        "worker",
    ),
)
def test_should_revalidate_exact_completion_proposal_instead_of_trusting_its_record(fault):
    admission, pending, observation = pending_fixture()
    change: Any = plan_completion(admission, pending, observation, manifest_for(pending))
    with pytest.raises(ValueError):
        if fault == "family":
            change = asdict(change)
        else:
            proposed = change.proposed
            m = proposed.metadata
            if fault in ("no_completion", "extra_completion"):
                m = replace(
                    m,
                    usage=replace(m.usage, updates_completed=3 if fault == "no_completion" else 5),
                )
            elif fault == "uncertain":
                m = replace(m, uncertain_work=True)
            elif fault == "stopped":
                m = replace(m, stopped=True)
            elif fault == "sequence":
                m = replace(m, sequence=m.sequence + 1)
            elif fault in ("copied", "grants", "attempts"):
                name = {
                    "copied": "copied_bytes",
                    "grants": "grants",
                    "attempts": "checkpoint_attempts",
                }[fault]
                m = replace(m, usage=replace(m.usage, **{name: getattr(m.usage, name) - 1}))
            elif fault == "actor":
                m = replace(m, manifest=replace(m.manifest, actor_sha256="c" * 64))
            else:
                proposed = replace(proposed, worker=RecoveryProcessIdentity(30, 300))
            change = replace(change, proposed=replace(proposed, metadata=m))
        validate_authority_completion(change)


class CompletingJournal(MemoryJournal):
    def complete(self, change):
        validate_authority_completion(change)
        self.log.append(("complete", change.proposed.metadata.sequence))
        if self.record != change.expected:
            return False
        self.record = change.proposed
        return True


def coordinator_fixture(factory=None, managed=False):
    record, _ = authority_fixture() if managed else original()
    log: list[tuple[str, object]] = []
    anchor, worker = Probe(record.anchor, log), Probe(record.worker, log)
    observer = Observer(record, log)
    journal = CompletingJournal(record, log) if factory is None else factory(record)
    coordinator = RecoveryCoordinator(record, journal, observer, anchor, worker)
    return coordinator, journal, observer, anchor, worker, log


@pytest.mark.parametrize("sqlite", (False, True))
def test_should_reserve_prepare_validate_complete_and_only_then_allow_publication_probe(
    tmp_path, sqlite
):
    factory = (
        (lambda record: SqliteRecoveryJournal.create(tmp_path / "authority.sqlite", record))
        if sqlite
        else None
    )
    coordinator, journal, _, anchor, worker, log = coordinator_fixture(factory)
    checkpoint = object()
    before = journal.read()

    def prepare():
        pending = journal.read()
        assert pending.metadata.uncertain_work and pending.metadata.usage.updates_admitted == 4
        assert (
            pending.metadata.usage.updates_completed == 3
            and pending.metadata.usage.copied_bytes == 45
        )
        log.append(("prepare", 1))
        return checkpoint

    def validate(value):
        assert value is checkpoint and journal.read().metadata.uncertain_work
        log.append(("validate", 1))
        return manifest_for(before)

    try:
        assert (
            coordinator.execute_native_update(
                RecoveryCosts(updates=1, copy_bytes=5, checkpoint_attempts=1), prepare, validate
            )
            is checkpoint
        )
        complete = journal.read()
        assert (
            complete.metadata.usage.updates_completed
            == complete.metadata.usage.updates_admitted
            == 4
        )
        assert not complete.metadata.uncertain_work and complete.metadata.manifest == manifest_for(
            before
        )
        assert complete.metadata.manifest.actor_sha256 == before.metadata.manifest.actor_sha256
        assert [name for name, _ in log if name in ("prepare", "validate")] == [
            "prepare",
            "validate",
        ]
        published: list[str] = []
        coordinator.execute(RecoveryCosts(), lambda: "probe", published.append)
        assert published == ["probe"]
    finally:
        coordinator.close()
    assert anchor.closed and worker.closed


@pytest.mark.parametrize(
    "fault",
    (
        "zero_updates",
        "two_updates",
        "bool_updates",
        "cost_family",
        "prepare",
        "validate",
        "capability",
        "managed",
    ),
)
def test_should_refuse_invalid_native_update_admission_before_callbacks_or_costs(fault):
    coordinator, journal, _, _, _, log = coordinator_fixture(managed=fault == "managed")
    before = journal.read().metadata.usage
    costs: Any = RecoveryCosts(updates=1)
    called = []
    prepare: Any = lambda: called.append("prepare")
    validate: Any = lambda value: manifest_for(journal.read())
    if fault.endswith("updates"):
        object.__setattr__(
            costs, "updates", {"zero_updates": 0, "two_updates": 2, "bool_updates": True}[fault]
        )
    elif fault == "cost_family":
        costs = asdict(costs)
    elif fault == "prepare":
        prepare = None
    elif fault == "validate":
        validate = None
    elif fault == "capability":
        journal.complete = None
    try:
        with pytest.raises(ValueError):
            coordinator.execute_native_update(costs, prepare, validate)
        assert journal.read().metadata.usage == before and called == []
        assert not any(name == "advance" for name, _ in log)
    finally:
        coordinator.close()


@pytest.mark.parametrize(
    "fault",
    (
        "prepare",
        "validator",
        "validator_type",
        "validator_pair",
        "worker_ended",
        "host_epoch",
        "host_rss",
    ),
)
def test_should_keep_native_admission_uncertain_and_stop_without_callback_retry(fault):
    coordinator, journal, observer, _, worker, _ = coordinator_fixture()
    calls = {"prepare": 0, "validate": 0}

    def prepare():
        calls["prepare"] += 1
        if fault == "prepare":
            raise RuntimeError("preparation failure")
        if fault == "worker_ended":
            worker.ended = True
        return "checkpoint"

    def validate(value):
        calls["validate"] += 1
        if fault == "validator":
            raise RuntimeError("checkpoint validation failure")
        if fault == "validator_type":
            return asdict(manifest_for(journal.record))
        if fault == "validator_pair":
            return replace(manifest_for(journal.record), native_sha256="c" * 64)
        if fault == "host_epoch":
            observer.hook = lambda value: replace(value, clock_epoch="foreign")
        if fault == "host_rss":
            observer.hook = lambda value: replace(value, rss_bytes=1025, peak_rss_bytes=1025)
        return manifest_for(journal.record)

    try:
        with pytest.raises((ValueError, RuntimeError)):
            coordinator.execute_native_update(
                RecoveryCosts(updates=1, copy_bytes=5), prepare, validate
            )
        current = journal.read()
        assert current.metadata.stopped and current.metadata.uncertain_work
        assert (
            current.metadata.usage.updates_admitted == 4
            and current.metadata.usage.updates_completed == 3
        )
        assert current.metadata.usage.copied_bytes == 45
        saved = calls.copy()
        with pytest.raises(ValueError):
            coordinator.execute_native_update(RecoveryCosts(updates=1), prepare, validate)
        assert calls == saved and calls["prepare"] == 1
    finally:
        coordinator.close()


@pytest.mark.parametrize("failure", ("false", "truthy", "before", "after", "readback"))
def test_should_preserve_completion_attempt_witness_and_disable_ambiguous_lane(
    monkeypatch, failure
):
    coordinator, journal, _, _, _, _ = coordinator_fixture()
    attempted = []
    primary = OSError("completion acknowledgment failure")
    original_read = journal.read
    read_fail = [False]

    def read():
        if read_fail[0]:
            read_fail[0] = False
            raise primary
        return original_read()

    def complete(change):
        validate_authority_completion(change)
        attempted.append(change)
        if failure == "before":
            raise primary
        if failure in ("after", "readback"):
            journal.record = change.proposed
        if failure == "after":
            raise primary
        if failure == "readback":
            read_fail[0] = True
            return True
        return False if failure == "false" else 1

    monkeypatch.setattr(journal, "complete", complete)
    monkeypatch.setattr(journal, "read", read)
    calls = []
    try:
        with pytest.raises((ValueError, OSError)):
            coordinator.execute_native_update(
                RecoveryCosts(updates=1, copy_bytes=5),
                lambda: calls.append("prepared"),
                lambda value: manifest_for(journal.record),
            )
        assert calls == ["prepared"] and len(attempted) == 1
        witness = coordinator.failure_witness
        assert witness is not None and type(witness.attempted) is AuthorityCompletion
        validate_failure_witness(witness)
        allowed = witnessed_terminal_states(witness)
        assert attempted[0].expected in allowed and attempted[0].proposed in allowed
        assert (
            journal.record.metadata.usage.updates_admitted == 4
            and journal.record.metadata.usage.copied_bytes == 45
        )
        with pytest.raises(ValueError):
            coordinator.execute_native_update(
                RecoveryCosts(updates=1),
                lambda: calls.append("retry"),
                lambda value: manifest_for(journal.record),
            )
        assert calls == ["prepared"]
    finally:
        coordinator.close()


def test_should_advance_only_explicit_sqlite_completion_floor_and_refuse_stale_readers(tmp_path):
    admission, pending, observation = pending_fixture()
    path = tmp_path / "authority.sqlite"
    journal = SqliteRecoveryJournal.create(path, admission.expected)
    assert journal.advance(admission)
    assert journal.report(
        plan_report(
            admission.proposed, replace(observation, now_ns=700, rss_bytes=700, peak_rss_bytes=700)
        )
    )
    assert journal.read() == pending
    stale = SqliteRecoveryJournal(path, pending)
    change = plan_completion(admission, pending, observation, manifest_for(pending))
    assert journal.complete(change) is True and journal.read() == change.proposed
    assert journal.complete(change) is False and journal.read() == change.proposed
    with pytest.raises(ValueError):
        stale.read()
    with pytest.raises(ValueError):
        validate_authority_floor(pending, change.proposed)
    assert SqliteRecoveryJournal(path, change.proposed).read() == change.proposed


@pytest.mark.parametrize("after_commit", (False, True))
def test_should_reconcile_completion_commit_ambiguity_only_to_terminal_known_outcome(
    monkeypatch, tmp_path, after_commit
):
    path = tmp_path / "authority.sqlite"
    coordinator, journal, _, _, _, _ = coordinator_fixture(
        lambda record: SqliteRecoveryJournal.create(path, record)
    )
    native_complete = journal.complete
    complete_attempts = []

    class Fault(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql == "COMMIT" and not after_commit:
                raise sqlite3.OperationalError("before completion COMMIT")
            result = super().execute(sql, parameters)
            if sql == "COMMIT":
                raise sqlite3.OperationalError("after completion COMMIT")
            return result

    def connect(target):
        return sqlite_module._configure(
            sqlite3.connect(
                target.as_uri() + "?mode=rw",
                uri=True,
                timeout=0,
                isolation_level=None,
                factory=Fault,
            )
        )

    def complete(change):
        complete_attempts.append(change)
        with monkeypatch.context() as local:
            local.setattr(sqlite_module, "_connect", connect)
            return native_complete(change)

    monkeypatch.setattr(journal, "complete", complete)
    try:
        with pytest.raises(sqlite3.OperationalError) as captured:
            coordinator.execute_native_update(
                RecoveryCosts(updates=1, copy_bytes=5),
                lambda: "B",
                lambda value: manifest_for(original()[0]),
            )
        assert len(complete_attempts) == 1 and isinstance(
            captured.value.__cause__, TerminalReportingFailure
        )
        witness = coordinator.failure_witness
        assert witness is not None and type(witness.attempted) is AuthorityCompletion
        stopped = SqliteRecoveryJournal.reconcile_terminal(path, witness)
        assert stopped.metadata.stopped and stopped.metadata.uncertain_work is not after_commit
        assert stopped.metadata.usage.updates_admitted == 4
        assert stopped.metadata.usage.updates_completed == (4 if after_commit else 3)
        assert stopped.metadata.usage.copied_bytes == 45
        assert stopped.metadata.manifest.native_sha256 == (B_HASH if after_commit else A_HASH)
        assert SqliteRecoveryJournal.reconcile_terminal(path, witness) == stopped
        with pytest.raises(ValueError):
            plan_reservation(stopped, original()[1], updates=1)
    finally:
        coordinator.close()


def test_should_refuse_foreign_or_rebound_completion_failure_witness():
    admission, pending, observation = pending_fixture()
    change = plan_completion(admission, pending, observation, manifest_for(pending))
    with pytest.raises(ValueError):
        FailureWitness(admission.expected, attempted=change)


def test_should_preserve_legacy_uncertain_update_refusal_without_implicit_completion():
    coordinator, journal, _, _, _, _ = coordinator_fixture()
    calls = []
    try:
        with pytest.raises(ValueError):
            coordinator.execute(
                RecoveryCosts(updates=1),
                lambda: calls.append("prepared"),
                lambda value: calls.append("published"),
            )
        assert calls == ["prepared"]
        assert (
            journal.read().metadata.uncertain_work
            and journal.read().metadata.usage.updates_completed == 3
        )
    finally:
        coordinator.close()
