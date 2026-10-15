"""Independent paused-worker protocol controls and one bounded interruption demo."""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import src.adapters.coupled_update_interruption as example
import src.adapters.coupled_update_worker as worker
from src.adapters.coupled_checkpoint_codec import CoupledCheckpointCodec
from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.coupled_circadian_predictive_coding import CoupledCircadianPredictiveCodingNetwork
from src.core.coupled_energy import predict_coupled
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork


def request():
    return dict(
        op="update",
        binding=dict(source_sha256="a" * 64, policy_sha256="b" * 64),
        limits=dict(
            max_encoded_bytes=65536, max_array_bytes=65536, max_layers=16, max_dimension=64
        ),
        expected_sha256="c" * 64,
        payload_utf8="original A",
    )


def fake_worker(monkeypatch):
    events: list[dict[str, Any]] = []
    calls: list[tuple[Any, ...]] = []
    original = request()
    state = SimpleNamespace(
        parameters=SimpleNamespace(hidden_weights=(np.zeros((2, 3)), np.zeros((3, 2)))),
        precisions=(1.0, 1.0),
    )

    class Model:
        def train_epoch(self, x, y, *settings):
            calls.append(("train", x.copy(), y.copy(), settings))
            x.fill(99)
            y.fill(99)
            return SimpleNamespace(energy=0.75)

        def snapshot_state(self):
            calls.append(("snapshot",))
            return state

        def predict_proba(self, x):
            calls.append(("predict", x.copy()))
            return np.array([[0.6], [0.4]])

    class Codec:
        def decode(self, raw, **controls):
            calls.append(("decode", raw, controls))
            return state

        def encode(self, snapshot, **controls):
            assert snapshot is state
            calls.append(("encode", controls))
            return b"encoded B"

    def stop():
        calls.append(("stop",))

    monkeypatch.setattr(worker, "_emit", lambda value: events.append(deepcopy(value)))
    monkeypatch.setattr(worker, "_request", lambda: original)
    monkeypatch.setattr(worker, "CoupledCheckpointCodec", Codec)
    monkeypatch.setattr(worker, "_model", lambda snapshot: Model())
    monkeypatch.setattr(worker, "_stop", stop)
    return original, state, calls, events


@pytest.mark.parametrize("flag", [None, 0, 1, "true", np.bool_(True), [], {}])
def test_should_refuse_nonboolean_pause_before_any_operational_callback(monkeypatch, flag):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid option reached operational work")

    for name in ("_emit", "_request", "_model", "CoupledCheckpointCodec"):
        monkeypatch.setattr(worker, name, forbidden)
    with pytest.raises(ValueError):
        worker.main(pause_before_result=flag)


def test_should_compute_encode_and_pause_without_disclosing_or_predicting_b(monkeypatch):
    original, _, calls, events = fake_worker(monkeypatch)
    before = deepcopy(original)
    assert worker.main(pause_before_result=True) == 0
    assert events[0]["event"] == "hello"
    assert events[1:] == [dict(event="computed", training_calls=1)]
    assert [value[0] for value in calls] == ["decode", "train", "snapshot", "encode", "stop"]
    np.testing.assert_array_equal(calls[1][1], [[0.2, -0.3], [-0.4, 0.6]])
    np.testing.assert_array_equal(calls[1][2], [[1.0], [0.0]])
    assert calls[1][3] == (0.02, 2, 0.05)
    assert original == before


@pytest.mark.parametrize("explicit", [False, True])
def test_should_preserve_default_healthy_updated_protocol(monkeypatch, explicit):
    _, _, calls, events = fake_worker(monkeypatch)
    result = worker.main(pause_before_result=False) if explicit else worker.main()
    assert result == 0
    response = events[1]
    assert len(events) == 2 and events[0]["event"] == "hello"
    assert response == dict(
        event="updated",
        payload_utf8="encoded B",
        digest=sha256(b"encoded B").hexdigest(),
        energy=0.75,
        predictions=[[0.6], [0.4]],
        training_calls=1,
    )
    assert [value[0] for value in calls] == [
        "decode",
        "train",
        "snapshot",
        "encode",
        "predict",
        "stop",
    ]
    np.testing.assert_array_equal(calls[-2][1], [[0.2, -0.3], [-0.4, 0.6]])


@pytest.mark.parametrize("stop", [{}, {"op": "update"}, {"op": True}, {"op": "stop", "extra": 1}])
def test_should_reject_nonexact_stop_after_computed_without_publishing_b(monkeypatch, stop):
    original, _, calls, events = fake_worker(monkeypatch)
    from src.adapters.coupled_checkpoint_worker import _stop

    # The shared exact-stop parser reads its own module's request hook.
    import src.adapters.coupled_checkpoint_worker as transport

    monkeypatch.setattr(transport, "_request", lambda: stop)
    monkeypatch.setattr(worker, "_stop", _stop)
    monkeypatch.setattr(worker, "_request", lambda: original)
    with pytest.raises(ValueError):
        worker.main(pause_before_result=True)
    assert events[1:] == [dict(event="computed", training_calls=1)]
    assert not any(value[0] == "predict" for value in calls)


def test_should_accept_exact_stop_after_computed(monkeypatch):
    fake_worker(monkeypatch)
    import src.adapters.coupled_checkpoint_worker as transport

    monkeypatch.setattr(transport, "_request", lambda: {"op": "stop"})
    monkeypatch.setattr(worker, "_stop", transport._stop)
    assert worker.main(pause_before_result=True) == 0


def test_should_refuse_eof_after_computed_without_disclosing_b(monkeypatch):
    _, _, calls, events = fake_worker(monkeypatch)
    import src.adapters.coupled_checkpoint_worker as transport

    monkeypatch.setattr(transport.sys, "stdin", SimpleNamespace(buffer=BytesIO(b"")))
    monkeypatch.setattr(worker, "_stop", transport._stop)
    with pytest.raises(ValueError):
        worker.main(pause_before_result=True)
    assert events[1:] == [dict(event="computed", training_calls=1)]
    assert not any(value[0] == "predict" for value in calls)


@pytest.mark.parametrize("phase", ["read", "decode", "train", "encode", "stop"])
def test_should_propagate_failure_without_retry_or_updated_response(monkeypatch, phase):
    _, _, calls, events = fake_worker(monkeypatch)

    def fail(*args, **kwargs):
        calls.append(("failure", phase))
        raise OSError(phase)

    if phase == "read":
        monkeypatch.setattr(worker, "_request", fail)
    elif phase == "stop":
        monkeypatch.setattr(worker, "_stop", fail)
    elif phase == "train":
        monkeypatch.setattr(worker, "_model", lambda state: SimpleNamespace(train_epoch=fail))
    else:
        codec = worker.CoupledCheckpointCodec()
        monkeypatch.setattr(codec, phase, fail)
        monkeypatch.setattr(worker, "CoupledCheckpointCodec", lambda: codec)
    with pytest.raises(OSError, match=phase):
        worker.main(pause_before_result=True)
    assert sum(value[0] == "failure" for value in calls) == 1
    assert not any(value.get("event") == "updated" for value in events)
    assert [value["event"] for value in events] == (
        ["hello", "computed"] if phase == "stop" else ["hello"]
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda value: value.update(op="read"),
        lambda value: value.update(op=True),
        lambda value: value.update(extra=1),
        lambda value: value.pop("payload_utf8"),
        lambda value: value.update(payload_utf8=b"not text"),
        lambda value: value.update(expected_sha256="bad"),
        lambda value: value["limits"].update(max_encoded_bytes=65537),
        lambda value: value["limits"].update(max_array_bytes=65537),
    ],
)
def test_should_reject_bad_request_before_decode_or_training(monkeypatch, change):
    original, _, calls, events = fake_worker(monkeypatch)
    change(original)
    with pytest.raises(ValueError):
        worker.main(pause_before_result=True)
    assert calls == [] and [value["event"] for value in events] == ["hello"]


@pytest.mark.parametrize("geometry", ["input", "width", "depth", "precision"])
def test_should_refuse_unfrozen_native_geometry_before_model_or_training(monkeypatch, geometry):
    _, state, calls, events = fake_worker(monkeypatch)
    if geometry == "precision":
        state.precisions = (1.0, 2.0)
    else:
        shapes = {
            "input": ((3, 3), (3, 2)),
            "width": ((2, 4), (4, 2)),
            "depth": ((2, 3),),
        }[geometry]
        state.parameters.hidden_weights = tuple(np.zeros(shape) for shape in shapes)
    with pytest.raises(ValueError):
        worker.main(pause_before_result=True)
    assert [value[0] for value in calls] == ["decode"]
    assert [value["event"] for value in events] == ["hello"]


def forbidden_work(*args, **kwargs):
    pytest.fail("admission/description dispatched operational work")


def no_native_or_process(monkeypatch):
    monkeypatch.setattr(CoupledPredictiveCodingNetwork, "__init__", forbidden_work)
    monkeypatch.setattr(CoupledCircadianPredictiveCodingNetwork, "__init__", forbidden_work)
    monkeypatch.setattr(subprocess, "Popen", forbidden_work)


def test_should_describe_frozen_work_without_io_models_or_children(monkeypatch):
    no_native_or_process(monkeypatch)
    monkeypatch.setattr(Path, "mkdir", forbidden_work)
    monkeypatch.setattr(Path, "open", forbidden_work)
    result = example.describe_example()
    assert result["protocol_id"] == "coupled_update_interruption_engineering_v1"
    assert result["cases"] == ["pc", "neutral_cpc", "active_cpc"]
    assert result["native_training_calls"] == 9 and result["native_energy_evaluations"] == 36
    assert result["owned_child_launches"] == 3


@pytest.mark.skipif(sys.platform != "win32", reason="Windows directory admission")
def test_should_refuse_existing_directory_before_models_or_children(monkeypatch, tmp_path):
    no_native_or_process(monkeypatch)
    with pytest.raises(FileExistsError):
        example.run_example(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_should_refuse_unsupported_platform_before_output_or_native_work(monkeypatch, tmp_path):
    no_native_or_process(monkeypatch)
    monkeypatch.setattr(example.sys, "platform", "unsupported")
    monkeypatch.setattr(Path, "mkdir", forbidden_work)
    with pytest.raises(OSError):
        example.run_example(tmp_path / "unused")


def decode(checkpoint):
    raw = checkpoint["payload_utf8"].encode()
    assert sha256(raw).hexdigest() == checkpoint["digest"]
    state = CoupledCheckpointCodec().decode(
        raw,
        binding=CodecBinding(**json.loads(raw)["binding"]),
        expected_sha256=checkpoint["digest"],
        limits=CodecLimits(65536, 65536, 16, 64),
    )
    plain = json.loads(json.dumps(asdict(state), default=lambda value: value.tolist()))
    assert plain == checkpoint["snapshot"]
    return state


@pytest.mark.skipif(sys.platform != "win32", reason="retained Windows worker identity integration")
def test_should_never_complete_an_update_from_computed_telemetry_after_owned_child_exit(tmp_path):
    directory = tmp_path / "native-interruption"
    report = example.run_example(directory)
    assert [row["family"] for row in report["rows"]] == ["pc", "neutral_cpc", "active_cpc"]
    assert report["remaining_temporary_files"] == []
    states = {}
    for row in report["rows"]:
        state = decode(row["checkpoint_a"])
        states[row["family"]] = state
        path = Path(row["path_a"])
        assert path.is_relative_to(directory)
        assert path.read_bytes() == row["checkpoint_a"]["payload_utf8"].encode()
        path_b = Path(row["path_b"])
        assert path_b.is_relative_to(directory) and not path_b.exists()
        assert row["actor_after"] == row["checkpoint_a"]["snapshot"]
        prediction = predict_coupled(state.parameters, np.array([[0.2, -0.3], [-0.4, 0.6]]))
        assert set(row["predictions"]) == {"before", "pending", "after_result", "after_refusal"}
        for value in row["predictions"].values():
            np.testing.assert_array_equal(value, prediction)
        assert row["response"] == dict(event="computed", training_calls=1)
        assert row["callbacks"] == dict(
            prepare=1, validate=0, refused_prepare=0, refused_validate=0
        )
        assert row["native_failure"] and row["final_attempt_refusal"]
        assert row["terminal_confirmed"] is True
        assert row["exit_code"] != 0
        assert row["child_pid"] == row["worker_identity"]["pid"]
        ordered = [row["authority"][key] for key in ("initial", "pending", "result", "terminal")]
        original = ordered[0]["metadata"]
        for index, record in enumerate(ordered):
            current = record["metadata"]
            assert record["worker"] == row["worker_identity"]
            assert record["anchor"] == ordered[0]["anchor"]
            for name in (
                "session_id",
                "clock_epoch",
                "started_ns",
                "owner_id",
                "owner_epoch",
                "limits",
                "manifest",
                "event_tick",
            ):
                assert current[name] == original[name]
            assert (
                current["manifest"]["native_sha256"]
                == current["manifest"]["payload_sha256"]
                == row["checkpoint_a"]["digest"]
            )
            usage = current["usage"]
            assert (
                usage["updates_admitted"]
                == usage["checkpoint_attempts"]
                == (0 if index == 0 else 1)
            )
            assert usage["updates_completed"] == usage["grants"] == 0
            assert usage["copied_bytes"] == (0 if index == 0 else 4 * 65536)
            assert current["uncertain_work"] is (index != 0)
        assert ordered[-1]["metadata"]["stopped"] is True
        observations = row["observations"]
        assert observations["initial"]["previous_owner_ended"] is False
        assert observations["computed_live"]["previous_owner_ended"] is False
        assert observations["ended"]["previous_owner_ended"] is True
        assert observations["ended"]["previous_owner"] == row["worker_identity"]
        if state.wake_state is not None:
            assert state.wake_state.wake_updates == 2
    np.testing.assert_equal(
        asdict(states["pc"].parameters), asdict(states["neutral_cpc"].parameters)
    )
