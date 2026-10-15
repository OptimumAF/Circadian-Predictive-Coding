"""Owned candidate corruption must never become native completion authority."""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts import run_coupled_update_completion as cli
from src.adapters import coupled_update_completion as demo
from src.adapters.coupled_checkpoint_codec import CoupledCheckpointCodec
from src.core.checkpoint_codec import CodecBinding, CodecLimits


def forbidden(*args, **kwargs):
    pytest.fail("invalid input or description dispatched operational work")


@pytest.mark.parametrize("flag", [None, 0, 1, "true", [], {}, np.bool_(True)])
@pytest.mark.parametrize("operation", ["describe", "run"])
def test_should_refuse_nonboolean_mode_before_io_or_native_work(
    monkeypatch, tmp_path, flag, operation
):
    monkeypatch.setattr(demo, "_model", forbidden)
    monkeypatch.setattr(demo, "BoundedJsonChild", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(ValueError):
        if operation == "describe":
            demo.describe_example(corrupt_checkpoint=flag)
        else:
            demo.run_example(tmp_path / "unused", corrupt_checkpoint=flag)


def test_should_describe_selected_work_without_changing_scientific_controls(monkeypatch):
    monkeypatch.setattr(demo, "_model", forbidden)
    monkeypatch.setattr(demo, "BoundedJsonChild", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    default = demo.describe_example()
    assert default == demo.describe_example(corrupt_checkpoint=False)
    selected = demo.describe_example(corrupt_checkpoint=True)
    assert default["protocol_id"] == "coupled_update_completion_engineering_v1"
    assert selected["protocol_id"] == "coupled_checkpoint_corruption_engineering_v1"
    assert selected["cases"] == ["pc", "neutral_cpc", "active_cpc"]
    assert selected["native_training_calls"] == 12
    assert selected["native_energy_evaluations"] == 48
    assert selected["owned_child_launches"] == selected["admitted_child_updates"] == 3
    assert selected["preparatory_training_calls"] == 9
    for name in ("seed", "geometry", "training", "external_training_settings", "limits", "binding"):
        assert selected[name] == default[name]


@pytest.mark.parametrize("selected", [False, True])
def test_should_describe_cli_selection_without_operational_work(monkeypatch, capsys, selected):
    monkeypatch.setattr(demo, "_model", forbidden)
    monkeypatch.setattr(demo, "BoundedJsonChild", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    flags = ["native-completion", "--describe"] + (["--corrupt-checkpoint"] if selected else [])
    monkeypatch.setattr(sys, "argv", flags)
    cli.main()
    assert json.loads(capsys.readouterr().out) == demo.describe_example(corrupt_checkpoint=selected)


@pytest.mark.parametrize("selected", [False, True])
def test_should_forward_cli_mode_and_directory_exactly_once(
    monkeypatch, capsys, tmp_path, selected
):
    seen = []

    def run(directory, *, corrupt_checkpoint=False):
        seen.append((directory, corrupt_checkpoint))
        return {"selected": corrupt_checkpoint}

    monkeypatch.setattr(cli, "run_example", run)
    flags = ["native-completion", "--directory", str(tmp_path)]
    if selected:
        flags.append("--corrupt-checkpoint")
    monkeypatch.setattr(sys, "argv", flags)
    cli.main()
    assert seen == [(tmp_path, selected)]
    assert json.loads(capsys.readouterr().out) == {"selected": selected}


def test_should_refuse_selected_mode_on_unsupported_platform_before_io(monkeypatch, tmp_path):
    monkeypatch.setattr(demo, "_model", forbidden)
    monkeypatch.setattr(demo, "BoundedJsonChild", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(demo.sys, "platform", "unsupported")
    with pytest.raises(OSError):
        demo.run_example(tmp_path / "unused", corrupt_checkpoint=True)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows directory admission")
def test_should_preserve_existing_directory_before_corruption_work(monkeypatch, tmp_path):
    marker = tmp_path / "unrelated.txt"
    marker.write_bytes(b"preserve")
    monkeypatch.setattr(demo, "_model", forbidden)
    monkeypatch.setattr(demo, "BoundedJsonChild", forbidden)
    with pytest.raises(FileExistsError):
        demo.run_example(tmp_path, corrupt_checkpoint=True)
    assert list(tmp_path.iterdir()) == [marker] and marker.read_bytes() == b"preserve"


def test_should_damage_only_owned_last_byte_and_flush_before_fsync(monkeypatch, tmp_path):
    path = tmp_path / "candidate.json"
    raw = b'{"bounded":"owned"}'
    path.write_bytes(raw)
    calls = []
    original_fsync = demo.os.fsync

    def fsync(descriptor):
        # A separate descriptor sees the changed byte only after explicit flush.
        assert path.read_bytes() == raw[:-1] + b"!"
        calls.append(descriptor)
        original_fsync(descriptor)

    monkeypatch.setattr(demo.os, "fsync", fsync)
    digest = sha256(raw).hexdigest()
    result = demo._corrupt_checkpoint(path, raw, digest)
    assert len(calls) == 1
    assert path.read_bytes() == raw[:-1] + b"!"
    assert result == dict(
        offset=len(raw) - 1,
        original_byte=125,
        replacement_byte=33,
        expected_digest=digest,
        actual_digest=sha256(raw[:-1] + b"!").hexdigest(),
        payload_utf8=(raw[:-1] + b"!").decode(),
    )


@pytest.mark.parametrize("fault", ["digest", "bytes", "closing_byte"])
def test_should_refuse_invalid_owned_corruption_input_without_mutation(tmp_path, fault):
    path = tmp_path / "candidate.json"
    raw = b'{"bounded":1}'
    disk = b'{"bounded":2}' if fault == "bytes" else raw
    supplied = raw[:-1] + b"]" if fault == "closing_byte" else raw
    if fault == "closing_byte":
        disk = supplied
    path.write_bytes(disk)
    digest = "a" * 64 if fault == "digest" else sha256(supplied).hexdigest()
    with pytest.raises((ValueError, AssertionError)):
        demo._corrupt_checkpoint(path, supplied, digest)
    assert path.read_bytes() == disk


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


@pytest.mark.skipif(sys.platform != "win32", reason="retained Windows coordinator integration")
def test_should_reject_damaged_b_under_original_digest_and_preserve_pending_authority(tmp_path):
    directory = tmp_path / "corruption"
    report = demo.run_example(directory, corrupt_checkpoint=True)
    assert [row["case"] for row in report["rows"]] == ["pc", "neutral_cpc", "active_cpc"]
    assert report["remaining_temporary_files"] == []
    states = {}
    for row in report["rows"]:
        a, b = decode(row["checkpoint_a"]), decode(row["expected_b"])
        states[row["family"]] = (a, b)
        assert row["expected_failure"] is True and row["actual_b"] is None
        path_a, path_b = Path(row["path_a"]), Path(row["path_b"])
        assert path_a.is_relative_to(directory) and path_b.is_relative_to(directory)
        assert path_a.read_bytes() == row["checkpoint_a"]["payload_utf8"].encode()
        valid = row["expected_b"]["payload_utf8"].encode()
        damaged = path_b.read_bytes()
        assert valid[-1:] == b"}" and damaged == valid[:-1] + b"!"
        corruption = row["corruption"]
        assert corruption == dict(
            offset=len(valid) - 1,
            original_byte=125,
            replacement_byte=33,
            expected_digest=row["expected_b"]["digest"],
            actual_digest=sha256(damaged).hexdigest(),
            payload_utf8=damaged.decode(),
        )
        assert corruption["actual_digest"] != corruption["expected_digest"]
        response = row["response"]
        assert response["event"] == "updated" and response["training_calls"] == 1
        assert response["payload_utf8"] == row["expected_b"]["payload_utf8"]
        assert response["digest"] == row["expected_b"]["digest"]
        np.testing.assert_array_equal(response["predictions"], row["expected_b_predictions"])
        assert set(row["predictions"]) == {
            "before",
            "pending",
            "checkpoint_written",
            "after_result",
            "after_refusal",
        }
        for value in row["predictions"].values():
            np.testing.assert_array_equal(value, row["predictions"]["before"])
        assert row["callbacks"] == dict(
            prepare=1, validate=1, probe=0, refused_prepare=0, refused_validate=0
        )
        assert row["native_failure"] == (
            f"checkpoint content differs from independently expected digest: {path_b}"
        )
        assert row["final_attempt_refusal"] == (
            "coordinator previously failed; independent reconciliation required"
        )
        assert row["stale_floor_refusal"] is None
        assert row["exit_code"] == 0 and row["worker_identity"]["pid"] == row["child_pid"]
        authority = row["authority"]
        assert authority["probe"] is None
        records = [
            authority[key]
            for key in ("initial", "pending", "checkpoint_written", "result", "terminal")
        ]
        original = deepcopy(records[0]["metadata"])
        for index, record in enumerate(records):
            current = record["metadata"]
            assert (
                record["anchor"] == records[0]["anchor"]
                and record["worker"] == records[0]["worker"]
            )
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
            usage = current["usage"]
            assert (
                usage["updates_admitted"]
                == usage["checkpoint_attempts"]
                == (0 if index == 0 else 1)
            )
            assert usage["updates_completed"] == usage["grants"] == 0
            assert usage["copied_bytes"] == (0 if index == 0 else 262144)
            assert current["uncertain_work"] is (index != 0)
        assert records[-1]["metadata"]["stopped"] is True
        for observation in row["observations"].values():
            assert observation["previous_owner_ended"] is False
            assert observation["previous_owner"] == row["worker_identity"]
        if a.wake_state is not None:
            assert a.wake_state.wake_updates == 2 and b.wake_state.wake_updates == 3
    for index in (0, 1):
        np.testing.assert_equal(
            asdict(states["pc"][index].parameters), asdict(states["neutral_cpc"][index].parameters)
        )
