"""Paused native telemetry is data, never checkpoint completion or authority."""

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters.native_transfer_interruption import stop_interrupted, validate_pause
from src.adapters.native_transfer_pause import PausedNativeModel
from src.adapters.native_transfer_protocol import LIMITS
from src.core.coupled_predictive_coding import CoupledTrainResult
from src.core.recovery_admission import RecoveryUsage
from src.core.recovery_authority import plan_reservation
from src.core.recovery_observation import RecoveryHostObservation
from src.core.recovery_reporting import plan_report
from src.infra.recovery_authority_codec import encode_authority
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal
from test_native_transfer_session import advance, encode, known_for, snapshot
from test_native_transfer_session import harness as _transfer_harness

transfer_harness = _transfer_harness


def fixture(family="pc"):
    b = advance(snapshot(family))
    c = advance(b)
    before = known_for(b)
    before = replace(
        before,
        metadata=replace(
            before.metadata,
            observed_ns=800,
            usage=RecoveryUsage(1, 1, 6 * LIMITS.max_encoded_bytes, 0, 3, 1024),
        ),
    )
    observation = RecoveryHostObservation(
        before.metadata.clock_epoch, 900, 1024, 1024, before.anchor, before.worker, False
    )
    admitted = plan_reservation(
        before,
        observation,
        updates=1,
        copy_bytes=4 * LIMITS.max_encoded_bytes,
        checkpoint_attempts=1,
    )
    pending = plan_report(
        admitted.proposed, replace(observation, now_ns=910, rss_bytes=1100, peak_rss_bytes=1100)
    ).proposed
    counts = dict(
        constructions=1,
        train_calls=1,
        native_sgd=1,
        raw_scaling_sgd=0,
        energy=4,
        predictions=1,
        split=0,
        guarded_inference=0,
        popen=0,
    )
    response = dict(
        event="native_paused",
        authority_utf8=encode_authority(pending).decode(),
        payload_utf8=encode(c).decode(),
        digest=sha256(encode(c)).hexdigest(),
        energy=0.75,
        budget=counts,
    )
    return SimpleNamespace(
        before=before, pending=pending, b=b, c=c, counts=counts, response=response
    )


class FakeNative:
    def __init__(self, state):
        self.state = deepcopy(state)
        self.calls: list[Any] = []
        self.prediction = np.array([[0.2], [0.8]])

    def snapshot_state(self):
        self.calls.append(("snapshot",))
        return deepcopy(self.state)

    def predict_proba(self, features):
        self.calls.append(("predict", features))
        return self.prediction

    def train_epoch(self, *args, **kwargs):
        self.calls.append(("train", args, kwargs))
        self.state = advance(self.state)
        return CoupledTrainResult(0.75)


@pytest.mark.parametrize("family", ["pc", "neutral_cpc", "active_cpc"])
@pytest.mark.parametrize("wait_kind", ["return", "eof", "io"])
def test_should_emit_one_complete_postmutation_receipt_under_pending_authority_and_never_return_training(
    family, wait_kind
):
    f = fixture(family)
    model = FakeNative(f.b)
    events: list[Any] = []
    failure = EOFError("closed pipe") if wait_kind == "eof" else OSError("wait failed")

    def authority():
        assert encode(model.state) == encode(f.c)
        events.append("authority")
        return f.pending

    def emit(value):
        events.append(value)

    def wait():
        events.append("wait")
        if wait_kind != "return":
            raise failure
        return {"op": "stop"}

    wrapper = PausedNativeModel(model, authority, f.counts, emit=emit, wait=wait)
    assert model.calls == [] and events == []
    x, y, policy = np.zeros((2, 2)), np.ones((2, 1)), object()
    if wait_kind == "return":
        with pytest.raises((ValueError, RuntimeError)):
            wrapper.train_epoch(x, y, 0.02, 2, 0.05, inference_policy=policy)
    else:
        with pytest.raises(type(failure)) as caught:
            wrapper.train_epoch(x, y, 0.02, 2, 0.05, inference_policy=policy)
        assert caught.value is failure
    train = next(call for call in model.calls if call[0] == "train")
    assert train[1][0] is x and train[1][1] is y
    assert train[1][2:] == (0.02, 2, 0.05) and train[2] == {"inference_policy": policy}
    assert [call[0] for call in model.calls].count("train") == 1
    assert events[0] == "authority" and events[-1] == "wait"
    assert events[1:-1] == [f.response]
    assert validate_pause(f.before, events[1], encode(f.c), 0.75) == f.pending
    expected = deepcopy(f.response)
    f.counts["train_calls"] = 99
    model.state.parameters.output_bias.fill(99)
    assert events[1] == expected


def test_should_delegate_snapshot_and_prediction_without_training_or_emission():
    f = fixture()
    model = FakeNative(f.b)

    def forbidden(*args):
        raise AssertionError("delegation reached pause callback")

    wrapper = PausedNativeModel(model, forbidden, f.counts, emit=forbidden, wait=forbidden)
    x = np.zeros((2, 2))
    assert encode(wrapper.snapshot_state()) == encode(f.b)
    assert wrapper.predict_proba(x) is model.prediction
    assert [call[0] for call in model.calls] == ["snapshot", "predict"]
    assert model.calls[1][1] is x


@pytest.mark.parametrize("fault", ["delegate", "authority", "encode", "emit"])
def test_should_preserve_pre_wait_fault_and_never_publish_or_retry(fault):
    f = fixture()
    model: Any = FakeNative(f.b)
    events = []
    primary = OSError("injected failure")

    def authority():
        events.append("authority")
        if fault == "authority":
            raise primary
        return f.pending

    def emit(value):
        events.append("emit")
        if fault == "emit":
            raise primary

    def wait():
        events.append("wait")

    if fault == "delegate":

        def train(*args, **kwargs):
            events.append("train")
            raise primary

        model.train_epoch = train
    elif fault == "encode":
        original = model.train_epoch

        def train(*args, **kwargs):
            result = original(*args, **kwargs)
            model.state.parameters.output_bias.fill(np.nan)
            return result

        model.train_epoch = train
    wrapper = PausedNativeModel(model, authority, f.counts, emit=emit, wait=wait)
    if fault == "encode":
        with pytest.raises(ValueError):
            wrapper.train_epoch(None)
    else:
        with pytest.raises(OSError) as caught:
            wrapper.train_epoch(None)
        assert caught.value is primary
    assert "wait" not in events
    if fault != "emit":
        assert "emit" not in events


def test_should_accept_only_exact_pending_charge_and_keep_c_telemetry_non_authoritative():
    f = fixture()
    before = encode_authority(f.before)
    response = deepcopy(f.response)
    assert validate_pause(f.before, response, encode(f.c), 0.75) == f.pending
    assert encode_authority(f.before) == before
    assert f.pending.metadata.manifest == f.before.metadata.manifest
    assert f.pending.metadata.manifest.native_sha256 != response["digest"]
    assert f.pending.metadata.usage.updates_admitted == 2
    assert f.pending.metadata.usage.updates_completed == 1


@pytest.mark.parametrize(
    "fault",
    [
        "mapping",
        "extra",
        "missing",
        "event",
        "event_subclass",
        "authority_bytes",
        "authority_noncanonical",
        "payload_bytes",
        "payload_bad",
        "digest_bool",
        "digest_foreign",
        "energy_bool",
        "energy_nan",
        "energy_mismatch",
        "budget_missing",
        "budget_extra",
        "budget_bool",
        "budget_train",
        "budget_sgd",
        "budget_energy",
        "budget_predictions",
        "budget_split",
    ],
)
def test_should_refuse_invalid_pause_wire_payload_energy_and_exact_native_budget(fault):
    f = fixture()
    response: Any = deepcopy(f.response)
    if fault == "mapping":
        response = SimpleNamespace(**response)
    elif fault == "extra":
        response["extra"] = 0
    elif fault == "missing":
        response.pop("digest")
    elif fault == "event":
        response["event"] = "updated"
    elif fault == "event_subclass":

        class Event(str):
            pass

        response["event"] = Event("native_paused")
    elif fault == "authority_bytes":
        response["authority_utf8"] = response["authority_utf8"].encode()
    elif fault == "authority_noncanonical":
        response["authority_utf8"] += " "
    elif fault == "payload_bytes":
        response["payload_utf8"] = response["payload_utf8"].encode()
    elif fault == "payload_bad":
        response["payload_utf8"] += " "
    elif fault == "digest_bool":
        response["digest"] = True
    elif fault == "digest_foreign":
        response["digest"] = "f" * 64
    elif fault == "energy_bool":
        response["energy"] = True
    elif fault == "energy_nan":
        response["energy"] = float("nan")
    elif fault == "energy_mismatch":
        response["energy"] = 0.76
    elif fault == "budget_missing":
        response["budget"].pop("popen")
    elif fault == "budget_extra":
        response["budget"]["extra"] = 0
    else:
        key = dict(
            budget_bool="constructions",
            budget_train="train_calls",
            budget_sgd="native_sgd",
            budget_energy="energy",
            budget_predictions="predictions",
            budget_split="split",
        )[fault]
        response["budget"][key] = True if fault == "budget_bool" else response["budget"][key] + 1
    with pytest.raises(ValueError):
        validate_pause(f.before, response, encode(f.c), 0.75)


@pytest.mark.parametrize(
    "fault",
    [
        "sequence",
        "admitted",
        "completed",
        "copy",
        "attempt",
        "grant",
        "uncertain",
        "stopped",
        "time",
        "peak",
        "owner",
        "epoch",
        "anchor",
        "worker",
        "start",
        "limits",
        "manifest",
    ],
)
def test_should_refuse_any_non_exact_pending_transition_or_protected_authority_change(fault):
    f = fixture()
    response = deepcopy(f.response)
    pending = json.loads(response["authority_utf8"])
    metadata = pending["metadata"]
    if fault in ("admitted", "completed", "copy", "attempt", "grant", "peak"):
        name = dict(
            admitted="updates_admitted",
            completed="updates_completed",
            copy="copied_bytes",
            attempt="checkpoint_attempts",
            grant="grants",
            peak="peak_rss_bytes",
        )[fault]
        value = (
            1023
            if fault == "peak"
            else metadata["usage"][name] - 1
            if fault in ("admitted", "copy", "attempt")
            else metadata["usage"][name] + 1
        )
        metadata["usage"][name] = value
    elif fault == "sequence":
        metadata["sequence"] += 1
    elif fault == "uncertain":
        metadata["uncertain_work"] = False
    elif fault == "stopped":
        metadata["stopped"] = True
    elif fault == "time":
        metadata["observed_ns"] = f.before.metadata.observed_ns - 1
    elif fault == "owner":
        metadata["owner_id"] = "foreign"
    elif fault == "epoch":
        metadata["clock_epoch"] = "foreign"
    elif fault == "anchor":
        pending["anchor"]["created_filetime"] = 999
    elif fault == "worker":
        pending["worker"]["created_filetime"] = 999
    elif fault == "start":
        metadata["started_ns"] += 1
    elif fault == "limits":
        metadata["limits"]["max_elapsed_ns"] += 1
    elif fault == "manifest":
        metadata["manifest"]["native_sha256"] = f.response["digest"]
        metadata["manifest"]["payload_sha256"] = f.response["digest"]
    response["authority_utf8"] = json.dumps(pending, sort_keys=True, separators=(",", ":"))
    # Keep malformed data at the wire boundary instead of failing in fixture construction.
    with pytest.raises(ValueError):
        validate_pause(f.before, response, encode(f.c), 0.75)


def test_should_stop_only_the_known_pending_record_idempotently_without_inventing_observations(
    tmp_path,
):
    f = fixture()
    path = tmp_path / "pending.sqlite"
    SqliteRecoveryJournal.create(path, f.pending)
    expected = plan_report(f.pending, None, terminal=True).proposed
    assert stop_interrupted(path, f.pending) == expected
    before = path.read_bytes()
    assert stop_interrupted(path, f.pending) == expected
    assert path.read_bytes() == before
    assert expected.metadata.observed_ns == f.pending.metadata.observed_ns
    assert expected.metadata.usage == f.pending.metadata.usage
    assert expected.metadata.manifest == f.before.metadata.manifest


@pytest.mark.parametrize("fault", ["newer", "foreign", "guessed_c", "unrelated_terminal"])
def test_should_not_reconcile_unknown_or_guessed_completed_disk_authority(tmp_path, fault):
    f = fixture()
    current = f.pending
    if fault == "newer":
        current = plan_report(current, None, terminal=True).proposed
        current = replace(
            current, metadata=replace(current.metadata, sequence=current.metadata.sequence + 1)
        )
    elif fault == "foreign":
        current = replace(current, metadata=replace(current.metadata, owner_id="foreign"))
    elif fault == "guessed_c":
        current = replace(
            current,
            metadata=replace(
                current.metadata,
                manifest=replace(
                    current.metadata.manifest,
                    native_sha256=f.response["digest"],
                    payload_sha256=f.response["digest"],
                ),
                uncertain_work=False,
                usage=replace(current.metadata.usage, updates_completed=2),
            ),
        )
    else:
        current = replace(
            plan_report(current, None, terminal=True).proposed,
            metadata=replace(
                plan_report(current, None, terminal=True).proposed.metadata,
                observed_ns=current.metadata.observed_ns + 1,
            ),
        )
    path = tmp_path / "unknown.sqlite"
    SqliteRecoveryJournal.create(path, current)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        stop_interrupted(path, f.pending)
    assert path.read_bytes() == before


@pytest.mark.parametrize("role,pause", [("old", False), ("new", False), ("new", True)])
def test_should_route_default_and_optional_new_worker_modes_without_os_or_profile_changes(
    monkeypatch, role, pause
):
    import src.adapters.native_transfer_worker as worker

    events: list[Any] = []
    argv = ["worker", "--role", role] + (["--pause-after-train"] if pause else [])
    monkeypatch.setattr(worker.sys, "argv", argv)
    monkeypatch.setattr(worker.sys, "platform", "win32")
    # Keep the outer before-work profile active throughout this pure CLI control.
    monkeypatch.setattr(worker.sys, "setprofile", lambda profile: None)
    monkeypatch.setattr(worker.threading, "setprofile", lambda profile: None)

    def budget(value, **kwargs):
        assert value == role
        assert kwargs == ({"pause_after_train": True} if pause else {})
        return {"role": value}, None

    monkeypatch.setattr(worker, "_budget", budget)
    api = SimpleNamespace(current_pid=lambda: 40)
    monkeypatch.setattr(worker, "WindowsRecoveryApi", lambda: api)

    class Handle:
        def __enter__(self):
            events.append("entered")
            return self

        def __exit__(self, *args):
            events.append("closed")

    own = Handle()

    def pin(pid, **kwargs):
        assert pid == 40 and kwargs == {"api": api}
        events.append("pin")
        return own

    monkeypatch.setattr(worker, "WindowsProcessHandle", SimpleNamespace(pin=pin))

    def serve(stack, captured, selected_api, selected_role, counts, **kwargs):
        assert captured is own and selected_api is api and selected_role == role
        assert counts == {"role": role}
        assert kwargs == ({"pause_after_train": True} if pause else {})
        events.append("served")

    monkeypatch.setattr(worker, "_serve", serve)
    assert worker.main() == 0
    assert events == ["pin", "entered", "served", "closed"]


def test_should_refuse_old_role_pause_flag_before_any_operational_boundary(monkeypatch):
    import src.adapters.native_transfer_worker as worker

    def forbidden(*args, **kwargs):
        raise AssertionError("old-role pause reached operational boundary")

    monkeypatch.setattr(worker.sys, "argv", ["worker", "--role", "old", "--pause-after-train"])
    monkeypatch.setattr(worker, "_budget", forbidden)
    monkeypatch.setattr(worker, "WindowsRecoveryApi", forbidden)
    with pytest.raises(SystemExit) as caught:
        worker.main()
    assert caught.value.code == 2


@pytest.mark.parametrize("role,pause,cap", [("old", False, 1), ("new", False, 2), ("new", True, 1)])
def test_should_refuse_prediction_entry_at_the_exact_mode_specific_cap_without_executing_it(
    role, pause, cap
):
    from src.adapters.native_transfer_worker import _budget
    from src.core.coupled_energy import predict_coupled

    counts, profile = _budget(role, pause_after_train=True) if pause else _budget(role)
    frame = SimpleNamespace(f_code=predict_coupled.__code__, f_locals={})
    profile(frame, "return", None)
    assert counts["predictions"] == 0
    for expected in range(1, cap + 1):
        profile(frame, "call", None)
        assert counts["predictions"] == expected
    with pytest.raises(RuntimeError, match="predictions budget exhausted BEFORE work"):
        profile(frame, "call", None)
    assert counts["predictions"] == cap + 1
    assert all(value == 0 for key, value in counts.items() if key != "predictions")


def test_should_pause_inside_persisted_real_session_admission_without_saving_or_completing(
    transfer_harness,
):
    h = transfer_harness
    factory = h.factory
    emitted: list[dict[str, Any]] = []
    counts = fixture().counts
    primary = EOFError("owned pipe ended")

    def wait():
        raise primary

    def paused_factory(value):
        return PausedNativeModel(
            factory(value), lambda: h.session.authority(), counts, emit=emitted.append, wait=wait
        )

    h.factory = paused_factory
    session = h.adopt(h.known)
    adopted = session.authority()
    with pytest.raises(EOFError) as caught:
        session.update(
            h.source,
            h.target,
            sha256(encode(h.value)).hexdigest(),
            sha256(encode(advance(h.value))).hexdigest(),
        )
    assert caught.value is primary
    assert h.events.index("load") < h.events.index("factory") < h.events.index("train")
    assert "save" not in h.events and "predict" not in h.events
    assert len(emitted) == 1 and not h.target.exists()
    pending = validate_pause(adopted, emitted[0], encode(advance(h.value)), 0.75)
    terminal = session.authority()
    assert terminal == plan_report(pending, None, terminal=True).proposed
    assert (
        terminal.metadata.usage.updates_admitted == 1
        and terminal.metadata.usage.updates_completed == 0
    )
    assert terminal.metadata.stopped and terminal.metadata.uncertain_work
    events = list(h.events)
    with pytest.raises(ValueError):
        session.update(
            h.source,
            h.target,
            sha256(encode(h.value)).hexdigest(),
            sha256(encode(advance(h.value))).hexdigest(),
        )
    assert h.events == events


def test_should_preserve_successful_default_replacement_completion_and_original_cap_refusal(
    tmp_path, monkeypatch
):
    import src.adapters.native_transfer_example as example

    f = fixture()
    completed = replace(
        f.pending,
        metadata=replace(
            f.pending.metadata,
            uncertain_work=False,
            usage=replace(f.pending.metadata.usage, updates_completed=2),
            manifest=replace(
                f.pending.metadata.manifest,
                native_sha256=f.response["digest"],
                payload_sha256=f.response["digest"],
            ),
        ),
    )
    terminal = plan_report(completed, None, terminal=True).proposed
    path = tmp_path / "completed.sqlite"
    SqliteRecoveryJournal.create(path, terminal)
    budget = dict(f.counts, predictions=2)
    exhausted = dict(
        event="exhausted",
        authority_utf8=encode_authority(terminal).decode(),
        reason="original authority allowance exceeded",
        callbacks=0,
        budget=budget,
    )
    events: list[Any] = []
    paths = {name: tmp_path / f"{name}.bin" for name in ("b", "c")}
    control, marker = object(), {"marker": "complete"}

    class Peer:
        def send(self, value):
            events.append(("send", value))

        def receive(self):
            events.append(("receive",))
            return deepcopy(exhausted)

    peer = Peer()

    def update(*args):
        assert args == (peer, paths["b"], paths["c"], f.before, control, "b", "c")
        events.append(("update",))
        return completed, marker

    monkeypatch.setattr(example, "_update", update)
    outcome, actual = example._complete_replacement(peer, paths, f.before, control, path)
    assert actual == terminal
    assert outcome == dict(
        completed_c=asdict(completed),
        terminal=asdict(terminal),
        new_update=marker,
        exhausted=exhausted,
    )
    assert events == [("update",), ("send", {"op": "exhaust"}), ("receive",)]
