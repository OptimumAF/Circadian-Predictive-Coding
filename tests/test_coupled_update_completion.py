"""One bounded native four-case completion example and two pure API controls."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import src.adapters.coupled_update_completion as example
from src.adapters.coupled_checkpoint_codec import CoupledCheckpointCodec
from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.coupled_circadian_predictive_coding import CoupledCircadianPredictiveCodingNetwork
from src.core.coupled_energy import predict_coupled
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork


def native_json(state):
    return json.loads(
        json.dumps(asdict(state), default=lambda value: value.tolist(), allow_nan=False)
    )


def decode(checkpoint):
    raw = checkpoint["payload_utf8"].encode()
    assert sha256(raw).hexdigest() == checkpoint["digest"]
    binding = CodecBinding(**json.loads(raw)["binding"])
    value = CoupledCheckpointCodec().decode(
        raw,
        binding=binding,
        expected_sha256=checkpoint["digest"],
        limits=CodecLimits(65536, 65536, 16, 64),
    )
    assert native_json(value) == checkpoint["snapshot"]
    return value


def test_should_describe_fixed_work_without_io_models_or_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("description dispatched operational work")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(CoupledPredictiveCodingNetwork, "__init__", forbidden)
    monkeypatch.setattr(CoupledCircadianPredictiveCodingNetwork, "__init__", forbidden)
    result = example.describe_example()
    assert result["protocol_id"] == "coupled_update_completion_engineering_v1"
    assert result["cases"] == ["pc", "neutral_cpc", "active_cpc", "active_cpc_validation_failure"]
    assert result["native_training_calls"] == 16 and result["native_energy_evaluations"] == 64
    assert result["owned_child_launches"] == 4
    assert result["preparatory_training_calls"] == 12 and result["admitted_child_updates"] == 4


@pytest.mark.skipif(sys.platform != "win32", reason="directory admission follows Windows admission")
def test_should_refuse_existing_directory_before_constructing_native_models_or_children(
    monkeypatch, tmp_path
):
    def forbidden(*args, **kwargs):
        pytest.fail("existing directory dispatched work")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(CoupledPredictiveCodingNetwork, "__init__", forbidden)
    monkeypatch.setattr(CoupledCircadianPredictiveCodingNetwork, "__init__", forbidden)
    with pytest.raises(FileExistsError):
        example.run_example(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(sys.platform != "win32", reason="live retained Windows coordinator integration")
def test_should_complete_three_native_updates_but_never_infer_completion_from_valid_failed_bytes(
    tmp_path,
):
    directory = tmp_path / "native-completion"
    result = example.run_example(directory)
    rows = result["rows"]
    assert [row["case"] for row in rows] == [
        "pc",
        "neutral_cpc",
        "active_cpc",
        "active_cpc_validation_failure",
    ]
    features = np.array([[0.2, -0.3], [-0.4, 0.6]])
    states = {}
    for row in rows:
        failed = row["case"] == "active_cpc_validation_failure"
        assert row["expected_failure"] is failed
        a, b = decode(row["checkpoint_a"]), decode(row["actual_b"])
        assert row["actual_b"] == row["expected_b"]
        states[row["case"]] = (a, b)
        path_a, path_b = Path(row["path_a"]), Path(row["path_b"])
        assert (
            path_a.is_relative_to(directory)
            and path_b.is_relative_to(directory)
            and path_a != path_b
        )
        assert path_a.read_bytes() == row["checkpoint_a"]["payload_utf8"].encode()
        assert path_b.read_bytes() == row["actual_b"]["payload_utf8"].encode()
        predicted_a, predicted_b = (
            predict_coupled(a.parameters, features),
            predict_coupled(b.parameters, features),
        )
        assert set(row["predictions"]) == {
            "before",
            "pending",
            "validated",
            "after_result",
            "after_refusal",
        }
        for prediction in row["predictions"].values():
            np.testing.assert_array_equal(prediction, predicted_a)
        np.testing.assert_array_equal(row["expected_b_predictions"], predicted_b)
        response = row["response"]
        assert response["event"] == "updated" and response["training_calls"] == 1
        assert (
            response["digest"] == row["actual_b"]["digest"]
            and response["payload_utf8"] == row["actual_b"]["payload_utf8"]
        )
        assert (
            type(response["energy"]) is float
            and np.isfinite(response["energy"])
            and response["energy"] >= 0
        )
        np.testing.assert_array_equal(response["predictions"], predicted_b)
        assert row["child_pid"] == row["worker_identity"]["pid"] and row["exit_code"] == 0
        assert row["callbacks"] == dict(
            prepare=1, validate=1, probe=0 if failed else 1, refused_prepare=0, refused_validate=0
        )
        assert row["final_attempt_refusal"]
        authority = row["authority"]
        assert (authority["probe"] is None) == failed
        keys = ("initial", "pending", "validated", "result", "terminal")
        records = [authority[key] for key in keys]
        metadata = [record["metadata"] for record in records]
        initial = metadata[0]
        assert initial["limits"]["max_updates"] == initial["limits"]["max_checkpoint_attempts"] == 1
        assert initial["limits"]["max_copy_bytes"] == 4 * 65536
        for record, current in zip(records, metadata, strict=True):
            assert (
                record["worker"] == row["worker_identity"]
                and record["anchor"] == records[0]["anchor"]
            )
            for key in (
                "session_id",
                "clock_epoch",
                "started_ns",
                "owner_id",
                "owner_epoch",
                "limits",
                "event_tick",
            ):
                assert current[key] == initial[key]
            for name in set(initial["manifest"]) - {"native_sha256", "payload_sha256"}:
                assert current["manifest"][name] == initial["manifest"][name]
        assert [current["usage"]["updates_admitted"] for current in metadata] == [0, 1, 1, 1, 1]
        assert [current["usage"]["checkpoint_attempts"] for current in metadata] == [0, 1, 1, 1, 1]
        assert [current["usage"]["copied_bytes"] for current in metadata] == [
            0,
            4 * 65536,
            4 * 65536,
            4 * 65536,
            4 * 65536,
        ]
        assert [current["usage"]["updates_completed"] for current in metadata] == [
            0,
            0,
            0,
            0 if failed else 1,
            0 if failed else 1,
        ]
        assert [current["uncertain_work"] for current in metadata] == [
            False,
            True,
            True,
            failed,
            failed,
        ]
        assert metadata[-1]["stopped"] is True
        for index, current in enumerate(metadata):
            expected = (
                row["checkpoint_a"]["digest"] if index < 3 or failed else row["actual_b"]["digest"]
            )
            assert (
                current["manifest"]["native_sha256"]
                == current["manifest"]["payload_sha256"]
                == expected
            )
        for name in ("sequence", "observed_ns"):
            values = [current[name] for current in metadata]
            assert values == sorted(values)
        if failed:
            assert row["native_failure"] and row["stale_floor_refusal"] is None
        else:
            assert row["native_failure"] is None and row["stale_floor_refusal"]
            probe_usage = authority["probe"]["metadata"]["usage"]
            completed_usage = authority["result"]["metadata"]["usage"]
            for name in (
                "updates_admitted",
                "updates_completed",
                "copied_bytes",
                "grants",
                "checkpoint_attempts",
            ):
                assert probe_usage[name] == completed_usage[name]
            assert probe_usage["peak_rss_bytes"] >= completed_usage["peak_rss_bytes"]
            assert (
                authority["probe"]["metadata"]["manifest"]
                == authority["result"]["metadata"]["manifest"]
            )
        if row["family"] != "pc":
            assert a.wake_state is not None and b.wake_state is not None
            assert a.wake_state.wake_updates == 2 and b.wake_state.wake_updates == 3
    for index in (0, 1):
        np.testing.assert_equal(
            asdict(states["pc"][index].parameters), asdict(states["neutral_cpc"][index].parameters)
        )
