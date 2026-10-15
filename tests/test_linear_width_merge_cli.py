"""Actual observed merge damage, source recovery and detached continuation."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_linear_width_merge as demo


def _run(*args):
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_linear_width_merge", *args],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        timeout=30,
        check=True,
    )
    assert not result.stderr
    assert len(result.stdout) <= 2 * 1024**2
    return json.loads(result.stdout)


def test_should_describe_without_model_work(monkeypatch: pytest.MonkeyPatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("description must not construct or evaluate models")

    monkeypatch.setattr(demo, "_source", forbidden)
    assert _run("--describe") == demo.describe_example()
    assert demo.describe_example()["sgd_calls"] == 2
    assert demo.describe_example()["equilibrium_evaluations"] == 20


def test_should_retain_all_outcomes_and_original_amplification_with_full_fallbacks():
    report = _run()
    assert len(report["controls"]) == 6
    assert report["continuation_training"]["role"] == "train"
    assert report["merge_rtol"] == 1e-10 and report["merge_atol"] == 1e-12
    for row in report["controls"]:
        decision = row["decision"]
        assert decision["planned_probe_count"] == 2
        if row["scenario"] == "compatible_accepted":
            assert decision["status"] == "accepted" and decision["reason"] is None
            assert decision["accepted_lineage"] == row["declared_lineage"]
            assert decision["returned_state"]["config"]["width"] == 2
            assert row["source"]["config"]["width"] == 3
            assert row["continuation"]["state"]["config"] == decision["returned_state"]["config"]
            assert (
                row["continuation"]["state"]["raw_weights"]
                != decision["returned_state"]["raw_weights"]
            )
            assert len(decision["diagnostics"]) == 2
        else:
            assert decision["returned_state"] == row["source"]
            assert decision["accepted_lineage"] is None and row["continuation"] is None
            assert decision["reason"]
            if row["scenario"] == "amplification_rejected":
                assert decision["status"] == "rejected" and row["pure_merge_candidate"] is not None
                assert len(decision["diagnostics"]) == 2
                assert all(
                    d["max_prediction_change"] > row["limits"]["max_prediction_change"]
                    for d in decision["diagnostics"]
                )
                assert all(
                    d["half_mse_increase"] == d["energy_increase"] == 0
                    for d in decision["diagnostics"]
                )
            else:
                assert decision["status"] == "failed" and decision["diagnostics"] == []
