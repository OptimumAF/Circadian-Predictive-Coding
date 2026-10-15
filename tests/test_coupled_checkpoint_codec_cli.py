"""Actual complete native snapshot bytes, refusal and continued learning."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_coupled_checkpoint_codec as demo


def test_should_describe_without_constructing_or_training_models(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("description constructed a model")

    monkeypatch.setattr(demo, "_model", forbidden)
    monkeypatch.setattr(sys, "argv", ["coupled-codec", "--describe"])
    demo.main()
    assert json.loads(capsys.readouterr().out) == demo.describe_example()
    assert demo.describe_example()["native_training_calls"] == 12
    assert demo.describe_example()["native_energy_evaluations"] == 48


def test_should_roundtrip_complete_bytes_and_continue_all_three_native_families():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_coupled_checkpoint_codec"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert not result.stderr and len(result.stdout) <= 2 * 1024**2
    report = json.loads(result.stdout)
    assert report["training"]["role"] == "train"
    assert [row["family"] for row in report["rows"]] == report["families"]
    for row in report["rows"]:
        payload = row["encoded_utf8"].encode("utf-8")
        assert len(payload) == row["encoded_bytes"] <= report["limits"]["max_encoded_bytes"]
        assert sha256(payload).hexdigest() == row["encoded_sha256"]
        wire = json.loads(payload)
        assert wire["binding"] == report["binding"]
        assert wire["snapshot_version"] == row["source_snapshot"]["format_version"] == 1
        assert wire["precisions"] == row["source_snapshot"]["precisions"] == [1.0, 1.0]
        assert row["source_snapshot"] == row["decoded_snapshot"]
        assert row["continued_source"] == row["continued_restored"]
        assert row["continued_source"]["parameters"] != row["source_snapshot"]["parameters"]
        assert row["decoded_storage_detached"] is True
        assert all(row["refusals"].values()) and len(row["refusals"]) == 2
        wake = row["source_snapshot"]["wake_state"]
        if row["family"] == "pc":
            assert wire["kind"] == "coupled_pc" and wire["wake_state"] is wake is None
        else:
            assert wire["kind"] == "coupled_cpc"
            assert wake["wake_updates"] == wire["wake_state"]["wake_updates"] == 2
            assert row["continued_restored"]["wake_state"]["wake_updates"] == 3
            assert any(value > 0 for value in wake["chemical_state"])
            if row["family"] == "neutral_cpc":
                assert wake["min_plasticity"] == 1 and wake["plasticity_state"] == [1.0, 1.0]
            else:
                assert wake["min_plasticity"] == pytest.approx(0.2)
                assert any(value < 1 for value in wake["plasticity_state"])
    for stage in ("source_snapshot", "continued_source"):
        assert report["rows"][0][stage]["parameters"] == report["rows"][1][stage]["parameters"]
