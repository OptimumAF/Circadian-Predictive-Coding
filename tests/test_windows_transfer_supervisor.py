"""Retained Windows transfer composition with deterministic handles and no processes."""

from copy import deepcopy
from dataclasses import asdict, replace
import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest

from src.core.coordinator_transfer import (
    CoordinatorTransferFailure,
    plan_coordinator_transfer,
    validate_coordinator_transfer,
    transfer_failure_states,
)
from src.core.recovery_observation import RecoveryHostObservation, anchored_clock_epoch
from src.core.recovery_reporting import plan_report
from src.infra.bounded_json_child import BoundedJsonChild
import src.infra.sqlite_recovery_journal as storage
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal
from src.infra.windows_process_handles import WindowsProcessHandle, WindowsRecoveryApi
from src.infra.windows_transfer_supervisor import WindowsTransferSupervisor
from test_coordinator_transfer import fixture as core_fixture, tamper
from test_windows_recovery_observer import Kernel


@pytest.fixture
def harness():
    kernel = Kernel()
    kernel.now = 6
    kernel.processes.update({40: [400, False], 50: [500, False]})
    api = WindowsRecoveryApi(_kernel=kernel, rss_reader=lambda: kernel.rss)
    handles = {pid: WindowsProcessHandle.pin(pid, api=api) for pid in (30, 10, 20, 40, 50)}
    known = core_fixture(zero=True)[0].authority
    state = SimpleNamespace(
        kernel=kernel, api=api, handles=handles, known=known, sends=[], owner=None
    )
    peer: Any = object.__new__(BoundedJsonChild)
    process = SimpleNamespace(pid=40, code=None)
    process.poll = lambda: process.code
    peer._process = process
    state.peer = peer
    state.process = process
    state.payload = dict(
        event="observed",
        observation=asdict(
            RecoveryHostObservation(
                anchored_clock_epoch(handles[40].identity),
                750,
                700,
                800,
                handles[40].identity,
                handles[50].identity,
                False,
            )
        ),
    )

    def send(value):
        state.sends.append(deepcopy(value))

    def receive():
        kernel.now, kernel.rss = 8, 900
        return deepcopy(state.payload)

    peer.send = send
    peer.receive = receive

    def forbidden(*args, **kwargs):
        pytest.fail("supervisor tried to terminate or own the child transport")

    peer.kill = forbidden
    peer.close = forbidden
    try:
        yield state
    finally:
        kernel.close_ok = True
        if state.owner is not None:
            state.owner.close()
        for handle in handles.values():
            handle.close()
        # Failed-close fixtures deliberately leave simulated uncertain native slots.
        kernel.handles.clear()


def construct(h):
    h.owner = WindowsTransferSupervisor(h.known, h.handles[30], h.handles[10], h.handles[20])
    return h.owner


def plan(h):
    owner = construct(h)
    h.kernel.processes[10][1] = h.kernel.processes[20][1] = True
    h.kernel.now, h.kernel.rss = 7, 650
    return owner.plan_transfer(h.peer, h.handles[40], h.handles[50], "owner-b")


def test_should_retain_final_supervisor_time_and_all_four_peaks_without_changing_old_defaults():
    witness, old = core_fixture()
    unchanged = plan_coordinator_transfer(witness, old, "owner-b")
    assert unchanged.proposed.metadata.observed_ns == 700
    after = RecoveryHostObservation(
        anchored_clock_epoch(witness.supervisor),
        800,
        850,
        900,
        witness.supervisor,
        old.replacement.observer,
        False,
    )
    observation = replace(old, supervisor_after=after)
    result = plan_coordinator_transfer(witness, observation, "owner-b")
    expected = replace(
        unchanged.proposed,
        metadata=replace(
            unchanged.proposed.metadata,
            observed_ns=800,
            usage=replace(unchanged.proposed.metadata.usage, peak_rss_bytes=900),
        ),
    )
    assert result.proposed == expected
    validate_coordinator_transfer(result)
    with pytest.raises(ValueError):
        validate_coordinator_transfer(tamper(result, "proposed.metadata.observed_ns", 700))
    with pytest.raises(ValueError):
        CoordinatorTransferFailure(result, replace(after, now_ns=799))


@pytest.mark.parametrize(
    "fault", ["before_remote", "ended", "observer", "owner", "epoch", "elapsed", "rss", "flag"]
)
def test_should_refuse_bad_fourth_observation_and_original_late_resource_overshoot(fault):
    witness, observation = core_fixture()
    after = RecoveryHostObservation(
        anchored_clock_epoch(witness.supervisor),
        800,
        800,
        900,
        witness.supervisor,
        observation.replacement.observer,
        False,
    )
    changes: dict[str, dict[str, Any]] = dict(
        before_remote={"now_ns": 699},
        ended={"previous_owner_ended": True},
        observer={"observer": observation.replacement.observer},
        owner={"previous_owner": observation.replacement.previous_owner},
        epoch={"clock_epoch": "foreign"},
        elapsed={"now_ns": 1100},
        rss={"rss_bytes": 1025, "peak_rss_bytes": 1025},
        flag={"previous_owner_ended": 0},
    )
    with pytest.raises(ValueError):
        plan_coordinator_transfer(
            witness,
            replace(observation, supervisor_after=replace(after, **changes[fault])),
            "owner-b",
        )


def test_should_bracket_exact_owned_peer_observation_and_preserve_true_time_rss_and_quota(harness):
    h = harness
    before = asdict(h.known)
    result = plan(h)
    assert h.sends == [
        dict(op="observe", worker=asdict(h.handles[50].identity), peak_rss_bytes=650)
    ]
    observation = result.observation
    assert observation.worker_ended.now_ns == 700
    assert observation.replacement.now_ns == 750
    assert observation.supervisor_after is not None and observation.supervisor_after.now_ns == 800
    assert observation.supervisor_after.observer == h.handles[30].identity
    assert result.proposed.metadata.observed_ns == 800
    assert result.proposed.metadata.usage.peak_rss_bytes == 900
    assert result.proposed.metadata.usage.checkpoint_attempts == 3
    assert result.proposed.metadata.limits == h.known.metadata.limits
    assert result.proposed.metadata.started_ns == h.known.metadata.started_ns
    assert result.proposed.metadata.manifest == h.known.metadata.manifest
    assert asdict(h.known) == before
    with pytest.raises(ValueError):
        h.owner.plan_transfer(h.peer, h.handles[40], h.handles[50], "owner-c")
    assert len(h.sends) == 1


@pytest.mark.parametrize(
    "fault",
    [
        "record",
        "supervisor_type",
        "anchor_type",
        "current_pid",
        "foreign_api",
        "ended_anchor",
        "ended_worker",
        "ended_supervisor",
        "unknown_wait",
    ],
)
def test_should_validate_all_initial_live_bindings_before_taking_handle_ownership(harness, fault):
    h = harness
    known, supervisor, anchor, worker = h.known, h.handles[30], h.handles[10], h.handles[20]
    if fault == "record":
        known = asdict(known)
    elif fault == "supervisor_type":
        supervisor = object()
    elif fault == "anchor_type":
        anchor = object()
    elif fault == "current_pid":
        h.kernel.GetCurrentProcessId.callback = lambda: 10
    elif fault == "foreign_api":
        worker._api = WindowsRecoveryApi(_kernel=h.kernel, rss_reader=lambda: 512)
    elif fault.startswith("ended_"):
        pid = {"ended_anchor": 10, "ended_worker": 20, "ended_supervisor": 30}[fault]
        h.kernel.processes[pid][1] = True
    else:
        h.kernel.wait_override = 0xFFFFFFFF
    with pytest.raises((ValueError, OSError)):
        WindowsTransferSupervisor(known, supervisor, anchor, worker)
    assert len(h.kernel.handles) == 5


@pytest.mark.parametrize("which", [10, 20])
def test_should_require_original_retained_death_and_never_repin_after_exit(harness, which):
    h = harness
    owner = construct(h)
    h.kernel.processes[which][1] = True
    h.kernel.now = 7
    before = len(h.kernel.opens)
    with pytest.raises(ValueError):
        owner.plan_transfer(h.peer, h.handles[40], h.handles[50], "owner-b")
    assert h.sends == [] and len(h.kernel.opens) == before
    h.kernel.processes[10][1] = h.kernel.processes[20][1] = True
    with pytest.raises(ValueError):
        owner.plan_transfer(h.peer, h.handles[40], h.handles[50], "owner-b")


@pytest.mark.parametrize(
    "fault",
    [
        "extra",
        "missing",
        "event",
        "time_lower",
        "time_upper",
        "bool_time",
        "bool_pid",
        "bool_flag",
        "owner",
        "observer",
        "epoch",
        "nested_extra",
        "nested_missing",
    ],
)
def test_should_refuse_untrusted_peer_fields_or_unbracketed_time_permanently(harness, fault):
    h = harness
    data = h.payload
    observation = data["observation"]
    if fault == "extra":
        data["extra"] = False
    elif fault == "missing":
        data.pop("observation")
    elif fault == "event":
        data["event"] = "other"
    elif fault == "time_lower":
        observation["now_ns"] = 699
    elif fault == "time_upper":
        observation["now_ns"] = 801
    elif fault == "bool_time":
        observation["now_ns"] = True
    elif fault == "bool_pid":
        observation["observer"]["pid"] = True
    elif fault == "bool_flag":
        observation["previous_owner_ended"] = 0
    elif fault == "owner":
        observation["previous_owner"]["created_filetime"] = 999
    elif fault == "observer":
        observation["observer"]["created_filetime"] = 999
    elif fault == "epoch":
        observation["clock_epoch"] = "foreign"
    elif fault == "nested_extra":
        observation["observer"]["extra"] = 0
    else:
        observation.pop("rss_bytes")
    with pytest.raises(ValueError):
        plan(h)
    assert len(h.sends) == 1
    with pytest.raises(ValueError):
        h.owner.plan_transfer(h.peer, h.handles[40], h.handles[50], "owner-b")
    assert len(h.sends) == 1


@pytest.mark.parametrize(
    "fault",
    [
        "peer_pid",
        "peer_ended",
        "new_anchor_ended",
        "new_worker_ended",
        "current_pid_changed",
        "io_failure",
    ],
)
def test_should_refuse_channel_or_physical_liveness_failure_before_or_after_rpc(harness, fault):
    h = harness
    if fault == "peer_pid":
        h.process.pid = 99
    elif fault == "peer_ended":
        h.process.code = 0
    else:
        original = h.peer.receive

        def receive():
            result = original()
            if fault == "new_anchor_ended":
                h.kernel.processes[40][1] = True
            elif fault == "new_worker_ended":
                h.kernel.processes[50][1] = True
            elif fault == "current_pid_changed":
                h.kernel.GetCurrentProcessId.callback = lambda: 40
            else:
                raise OSError("peer failed")
            return result

        h.peer.receive = receive
    with pytest.raises((ValueError, OSError)):
        plan(h)
    assert len(h.sends) == (0 if fault in ("peer_pid", "peer_ended") else 1)


def test_should_commit_once_through_exact_independent_sqlite_floor_and_return_retained_plan(
    harness, tmp_path
):
    h = harness
    result = plan(h)
    journal = SqliteRecoveryJournal.create(tmp_path / "authority.sqlite", h.known)
    h.kernel.now = 9
    assert h.owner.commit_transfer(journal) is result
    assert journal.read() == result.proposed
    with pytest.raises(ValueError):
        h.owner.commit_transfer(journal)
    assert journal.read() == result.proposed


@pytest.mark.parametrize(
    "fault",
    [
        "time_backwards",
        "elapsed",
        "new_anchor_ended",
        "new_worker_ended",
        "peer_ended",
        "wrong_floor",
    ],
)
def test_should_refuse_failed_commit_without_retry_and_keep_terminal_witness(
    harness, tmp_path, fault
):
    h = harness
    result = plan(h)
    path = tmp_path / "authority.sqlite"
    known = (
        h.known
        if fault != "wrong_floor"
        else replace(h.known, metadata=replace(h.known.metadata, sequence=6))
    )
    journal = SqliteRecoveryJournal.create(path, known)
    h.kernel.now = 9
    if fault == "time_backwards":
        h.kernel.now = 7
    elif fault == "elapsed":
        h.kernel.now = 11
    elif fault == "new_anchor_ended":
        h.kernel.processes[40][1] = True
    elif fault == "new_worker_ended":
        h.kernel.processes[50][1] = True
    elif fault == "peer_ended":
        h.process.code = 0
    with pytest.raises(ValueError):
        h.owner.commit_transfer(journal)
    h.kernel.now = 12
    h.kernel.rss = 1000
    witness = h.owner.failure_witness()
    assert witness.transfer is result
    assert witness.observation.now_ns == 1200
    assert witness.observation.peak_rss_bytes == 1000
    if fault != "wrong_floor":
        stopped = SqliteRecoveryJournal.reconcile_coordinator_transfer(path, witness)
        assert stopped == transfer_failure_states(witness)[0]
    with pytest.raises(ValueError):
        h.owner.commit_transfer(journal)


def test_should_preserve_false_cas_failure_and_refuse_unknown_newer_terminal_state(
    harness, tmp_path
):
    h = harness
    plan(h)
    path = tmp_path / "authority.sqlite"
    journal = SqliteRecoveryJournal.create(path, h.known)
    current = plan_report(
        h.known,
        RecoveryHostObservation(
            h.known.metadata.clock_epoch, 650, 600, 600, h.known.anchor, h.known.worker, False
        ),
    )
    assert SqliteRecoveryJournal(path, h.known).report(current)
    h.kernel.now = 9
    with pytest.raises(ValueError):
        h.owner.commit_transfer(journal)
    h.kernel.now = 10
    witness = h.owner.failure_witness()
    before = path.read_bytes()
    with pytest.raises(ValueError):
        SqliteRecoveryJournal.reconcile_coordinator_transfer(path, witness)
    assert path.read_bytes() == before


def test_should_refuse_nested_plan_nonblockingly_and_keep_failure_permanent(harness):
    h = harness

    def receive():
        return h.owner.plan_transfer(h.peer, h.handles[40], h.handles[50], "nested")

    h.peer.receive = receive
    with pytest.raises(ValueError):
        plan(h)
    assert len(h.sends) == 1
    with pytest.raises(ValueError):
        h.owner.plan_transfer(h.peer, h.handles[40], h.handles[50], "retry")


@pytest.mark.parametrize("after_commit", [False, True])
def test_should_retain_physical_transfer_witness_across_commit_ambiguity(
    harness, tmp_path, monkeypatch, after_commit
):
    h = harness
    planned = plan(h)
    path = tmp_path / "authority.sqlite"
    journal = SqliteRecoveryJournal.create(path, h.known)
    original = storage._connect

    class Fault(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql == "COMMIT" and not after_commit:
                raise sqlite3.OperationalError("injected before commit")
            result = super().execute(sql, parameters)
            if sql == "COMMIT":
                raise sqlite3.OperationalError("injected after commit")
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

    h.kernel.now = 9
    monkeypatch.setattr(storage, "_connect", connect)
    with pytest.raises(sqlite3.OperationalError):
        h.owner.commit_transfer(journal)
    monkeypatch.setattr(storage, "_connect", original)
    h.kernel.now = 10
    witness = h.owner.failure_witness()
    assert witness.transfer is planned
    assert (
        SqliteRecoveryJournal.reconcile_coordinator_transfer(path, witness)
        == (transfer_failure_states(witness)[int(after_commit)])
    )
    with pytest.raises(ValueError):
        h.owner.commit_transfer(journal)
    with pytest.raises(ValueError, match="failed"):
        journal.read()


@pytest.mark.parametrize("message", ["clock failed", "RSS failed", "SQLite failed"])
def test_should_preserve_non_cap_continuation_errors_and_close_coordinator(monkeypatch, message):
    import src.adapters.windows_transfer_worker as worker
    from src.infra.recovery_authority_codec import encode_authority

    known = core_fixture(zero=True)[0].authority
    events: list[str] = []
    primary = ValueError(message)

    class Coordinator:
        def execute(self, costs, prepare, publish=None):
            if events:
                raise primary
            result = prepare()
            assert publish is not None
            publish(result)
            events.append("executed")
            return result

        def close(self):
            events.append("closed")

    coordinator = Coordinator()
    monkeypatch.setattr(worker, "compose_windows_recovery", lambda *args: coordinator)
    monkeypatch.setattr(
        worker, "SqliteRecoveryJournal", lambda *args: SimpleNamespace(read=lambda: known)
    )
    request = dict(
        op="continue", authority_utf8=encode_authority(known).decode("utf-8"), journal_path="unused"
    )
    with pytest.raises(ValueError) as caught:
        worker._continue(
            request,
            SimpleNamespace(identity=known.anchor),
            SimpleNamespace(identity=known.worker),
            SimpleNamespace(now_ns=known.metadata.observed_ns),
            [0],
        )
    assert caught.value is primary
    assert events == ["executed", "closed"]


def test_should_close_each_owned_handle_once_and_never_terminate_the_caller_owned_peer(harness):
    h = harness
    plan(h)
    owned = [handle._handle for handle in h.handles.values()]
    h.owner.close()
    h.owner.close()
    assert all(h.kernel.closed.count(handle) == 1 for handle in owned)
    assert h.kernel.handles == {}


def test_should_preserve_primary_exception_and_all_failed_close_notes_without_retry(harness):
    h = harness
    plan(h)
    attempts = []
    native_close = h.kernel.CloseHandle.callback

    def close(handle):
        attempts.append(handle)
        return native_close(handle)

    h.kernel.CloseHandle.callback = close
    h.kernel.close_ok = False
    primary = RuntimeError("body primary")
    with pytest.raises(RuntimeError) as caught:
        with h.owner:
            raise primary
    assert caught.value is primary
    assert getattr(primary, "__notes__", [])
    assert len(attempts) == len(set(attempts)) == 5
    h.owner.close()
    assert len(attempts) == 5


@pytest.mark.parametrize(
    "case", ["success", "wrong_code", "stderr", "other_error", "cleanup_error"]
)
def test_should_accept_only_the_expected_crash_and_preserve_wait_or_cleanup_faults(case):
    from src.adapters.windows_transfer_example import _crash_original

    events: list[str] = []
    code = 18 if case == "wrong_code" else 17
    stderr = b"unexpected stderr" if case == "stderr" else b""
    message = (
        "unrelated pipe failure"
        if case == "other_error"
        else f"child exit failed: returncode={code}, stderr={stderr!r}"
    )
    primary = RuntimeError(message)
    cleanup = OSError("close failed")

    class Peer:
        returncode = code
        stderr: bytes

        def send(self, value):
            assert value == {"op": "crash"}
            events.append("send")

        def wait(self, *args, **kwargs):
            events.append("wait")
            raise primary

        def close(self):
            events.append("close")
            if case == "cleanup_error":
                raise cleanup

    peer = Peer()
    peer.stderr = stderr
    if case == "success":
        _crash_original(peer)
        assert events == ["send", "wait", "close"]
    else:
        expected = cleanup if case == "cleanup_error" else primary
        with pytest.raises(type(expected)) as caught:
            _crash_original(peer)
        assert caught.value is expected
        assert events == (
            ["send", "wait", "close"] if case == "cleanup_error" else ["send", "wait"]
        )
