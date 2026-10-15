"""Actual bounded files, precommit recovery and unchanged native continuation."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_coupled_checkpoint_files as demo


def test_should_describe_without_io_or_native_models(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("description performed IO or model work")

    monkeypatch.setattr(demo, "_model", forbidden)
    monkeypatch.setattr(demo.Path, "mkdir", forbidden)
    monkeypatch.setattr(sys, "argv", ["checkpoint-files", "--describe"])
    demo.main()
    assert json.loads(capsys.readouterr().out) == demo.describe_example()
    assert demo.describe_example()["native_training_calls"] == 12
    assert demo.describe_example()["native_energy_evaluations"] == 48


def test_should_preserve_existing_directory_before_constructing_models(monkeypatch, tmp_path):
    marker = tmp_path / "unrelated.txt"
    marker.write_bytes(b"preserve user file")

    def forbidden(*args, **kwargs):
        raise AssertionError("existing directory allowed model construction")

    monkeypatch.setattr(demo, "_model", forbidden)
    with pytest.raises(FileExistsError):
        demo.run_example(tmp_path)
    assert marker.read_bytes() == b"preserve user file"
    assert list(tmp_path.iterdir()) == [marker]


def test_should_write_complete_files_and_recover_old_snapshot_after_failed_replacement(tmp_path):
    directory = tmp_path / "new-native-files"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_coupled_checkpoint_files",
            "--directory",
            str(directory),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert not result.stderr and len(result.stdout) <= 2 * 1024**2
    report = json.loads(result.stdout)
    assert report["training"]["role"] == "train" and report["remaining_temporary_files"] == []
    assert [row["family"] for row in report["rows"]] == report["families"]
    assert sorted(path.name for path in directory.iterdir()) == [
        "active_cpc.json",
        "neutral_cpc.json",
        "pc.json",
    ]
    for row in report["rows"]:
        a, b = row["checkpoint_a"], row["checkpoint_b"]
        for stage in (a, b):
            raw = stage["payload_utf8"].encode("utf-8")
            assert sha256(raw).hexdigest() == stage["digest"]
            assert len(raw) <= report["limits"]["max_encoded_bytes"]
            assert json.loads(raw)["binding"] == report["binding"]
        assert a["snapshot"] == row["loaded_a"] == row["recovered_a"]
        assert (
            b["snapshot"]
            == row["reloaded_b"]
            == row["continued_original"]
            == row["continued_restored"]
        )
        assert a["snapshot"]["parameters"] != b["snapshot"]["parameters"]
        assert a["digest"] != b["digest"]
        assert Path(row["path"]).read_bytes() == b["payload_utf8"].encode("utf-8")
        assert row["previous_file_preserved"] is True and row["precommit_failure"]
        assert set(row["refusals"]) == {"stale_digest", "wrong_policy_binding", "encoded_cap"}
        assert all(row["refusals"].values())
        wake = a["snapshot"]["wake_state"]
        if row["family"] == "pc":
            assert wake is b["snapshot"]["wake_state"] is None
        else:
            assert wake["wake_updates"] == 2 and b["snapshot"]["wake_state"]["wake_updates"] == 3
            assert any(value > 0 for value in wake["chemical_state"])
    for phase in ("checkpoint_a", "checkpoint_b"):
        assert (
            report["rows"][0][phase]["snapshot"]["parameters"]
            == report["rows"][1][phase]["snapshot"]["parameters"]
        )
