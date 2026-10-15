"""One three-family interruption exercise; children never train or publish B."""

from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import src.adapters.coupled_checkpoint_interruption as interruption
from src.adapters.coupled_checkpoint_codec import CoupledCheckpointCodec
from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.coupled_circadian_predictive_coding import CoupledCircadianPredictiveCodingNetwork
from src.core.coupled_energy import predict_coupled
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork


def native_json(state):
    return json.loads(
        json.dumps(asdict(state), default=lambda value: value.tolist(), allow_nan=False)
    )


def decode_payload(checkpoint):
    raw = checkpoint["payload_utf8"].encode("utf-8")
    assert sha256(raw).hexdigest() == checkpoint["digest"]
    body = json.loads(raw)
    # This verifies the reported representation, not source authentication. The
    # actual coordinator must retain its independently supplied original binding.
    binding = CodecBinding(**body["binding"])
    state = CoupledCheckpointCodec().decode(
        raw,
        binding=binding,
        expected_sha256=checkpoint["digest"],
        limits=CodecLimits(65536, 65536, 16, 64),
    )
    assert checkpoint["snapshot"] == native_json(state)
    return state


def test_should_describe_frozen_work_without_io_native_models_or_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("describe performed operational work")

    monkeypatch.setattr(CoupledPredictiveCodingNetwork, "__init__", forbidden)
    monkeypatch.setattr(CoupledCircadianPredictiveCodingNetwork, "__init__", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    result = interruption.describe_example()
    assert result["protocol_id"] == "coupled_checkpoint_interruption_engineering_v1"
    assert tuple(result["families"]) == ("pc", "neutral_cpc", "active_cpc")
    assert result["native_training_calls"] == 9 and result["native_energy_evaluations"] == 36
    assert result["owned_child_launches"] == 6


@pytest.mark.skipif(sys.platform != "win32", reason="directory admission follows Windows admission")
def test_should_refuse_existing_output_directory_before_models_or_processes(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid directory dispatched work")

    monkeypatch.setattr(CoupledPredictiveCodingNetwork, "__init__", forbidden)
    monkeypatch.setattr(CoupledCircadianPredictiveCodingNetwork, "__init__", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with pytest.raises(FileExistsError):
        interruption.run_example(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(sys.platform != "win32", reason="retained Windows process identity integration")
def test_should_keep_actor_a_and_spent_authority_across_six_owned_child_processes(tmp_path):
    directory = tmp_path / "interruption"
    report = interruption.run_example(directory)
    rows = report["rows"]
    assert [row["family"] for row in rows] == ["pc", "neutral_cpc", "active_cpc"]
    assert report["remaining_temporary_files"] == []
    seen: set[tuple[int, int]] = set()
    states = {}
    features = np.array([[0.2, -0.3], [-0.4, 0.6]])
    for row in rows:
        family = row["family"]
        a, b = decode_payload(row["checkpoint_a"]), decode_payload(row["candidate_b"])
        states[family] = (a, b)
        path = Path(row["path"])
        assert path.is_absolute() and path.is_relative_to(directory)
        assert path.read_bytes() == row["checkpoint_a"]["payload_utf8"].encode()
        assert row["checkpoint_a"]["digest"] != row["candidate_b"]["digest"]
        assert row["staged_payload_utf8"] == row["candidate_b"]["payload_utf8"]
        staged = row["staged"]
        assert staged["event"] == "staged" and staged["digest"] == row["candidate_b"]["digest"]
        basename = staged["temporary"]
        assert (
            type(basename) is str
            and basename not in ("", ".", "..")
            and Path(basename).name == basename
        )
        assert row["staged_file_removed"] is True and not (path.parent / basename).exists()
        predicted = predict_coupled(a.parameters, features)
        assert set(row["predictions"]) == {
            "before",
            "during_staging",
            "after_exit",
            "after_handoff",
            "after_refusal",
        }
        for prediction in row["predictions"].values():
            np.testing.assert_array_equal(prediction, predicted)
        recovered = row["recovered"]
        assert recovered["event"] == "restored" and recovered["snapshot"] == native_json(a)
        np.testing.assert_array_equal(recovered["predictions"], predicted)
        pids = row["child_pids"]
        assert len(pids) == 2 and all(
            type(pid) is int and pid > 0 and pid != os.getpid() for pid in pids
        )
        assert (
            len(row["child_exit_codes"]) == 2
            and row["child_exit_codes"][0] != 0
            and row["child_exit_codes"][1] == 0
        )
        assert row["attempt_refusal"] and row["refused_callback_calls"] == 0
        records = row["authority"]
        assert set(records) == {"initial", "reserved", "handed_off", "read_done", "terminal"}
        ordered = [
            records[key] for key in ("initial", "reserved", "handed_off", "read_done", "terminal")
        ]
        identities = [
            (record["worker"]["pid"], record["worker"]["created_filetime"])
            for record in (ordered[0], ordered[2])
        ]
        # A process ID can be reused after a prior family closes its handles.
        # Registration identity includes creation time and must remain distinct.
        assert len(set(identities)) == 2 and not seen.intersection(identities)
        seen.update(identities)
        metadata = [value["metadata"] for value in ordered]
        original = metadata[0]
        for record, value in zip(ordered, metadata, strict=True):
            for key in (
                "session_id",
                "clock_epoch",
                "started_ns",
                "limits",
                "manifest",
                "event_tick",
            ):
                assert value[key] == original[key]
            assert record["anchor"] == ordered[0]["anchor"]
            assert value["uncertain_work"] is False
            assert value["usage"]["updates_admitted"] == value["usage"]["updates_completed"] == 0
            assert value["usage"]["grants"] == 0
        assert original["limits"]["max_checkpoint_attempts"] == 2
        assert original["limits"]["max_updates"] == 0
        assert original["limits"]["max_copy_bytes"] == 4 * 65536
        assert [value["usage"]["checkpoint_attempts"] for value in metadata] == [0, 1, 2, 2, 2]
        assert [value["usage"]["copied_bytes"] for value in metadata] == [
            0,
            2 * 65536,
            2 * 65536,
            4 * 65536,
            4 * 65536,
        ]
        assert [value["owner_epoch"] for value in metadata] == [0, 0, 1, 1, 1]
        assert metadata[1]["owner_id"] == original["owner_id"] != metadata[2]["owner_id"]
        assert [record["worker"]["pid"] for record in ordered] == [
            pids[0],
            pids[0],
            pids[1],
            pids[1],
            pids[1],
        ]
        assert [value["stopped"] for value in metadata] == [False, False, False, False, True]
        for key in ("sequence", "observed_ns"):
            values = [value[key] for value in metadata]
            assert values == sorted(values)
        peaks = [value["usage"]["peak_rss_bytes"] for value in metadata]
        assert peaks == sorted(peaks)
        observations = [row["observations"][key] for key in ("initial", "staged", "ended")]
        assert [value["now_ns"] for value in observations] == sorted(
            value["now_ns"] for value in observations
        )
        assert (
            observations[1]["previous_owner_ended"] is False
            and observations[2]["previous_owner_ended"] is True
        )
        assert observations[2]["previous_owner"] == ordered[0]["worker"]
        assert observations[2]["clock_epoch"] == original["clock_epoch"]
        if family != "pc":
            assert a.wake_state is not None and b.wake_state is not None
            assert a.wake_state.wake_updates == 2 and b.wake_state.wake_updates == 3
            assert np.any(a.wake_state.chemical_state > 0)
            if family == "neutral_cpc":
                np.testing.assert_array_equal(a.wake_state.plasticity_state, np.ones(2))
    for index in (0, 1):
        np.testing.assert_equal(
            asdict(states["pc"][index].parameters), asdict(states["neutral_cpc"][index].parameters)
        )
