"""Actual fixed local gate example: four SGD calls, twenty equilibrium calls."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def _invoke(*arguments):
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_linear_width_pruning", *arguments],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    assert result.stderr == "" and len(result.stdout.encode("utf-8")) < 1024**2
    return json.loads(result.stdout)


def test_should_describe_without_evaluation_or_training():
    report = _invoke("--describe")
    assert report["sgd_calls"] == 4 and report["equilibrium_evaluations"] == 20
    assert "controls" not in report


def test_should_gate_both_families_and_continue_from_accepted_or_retained_states():
    report = _invoke()
    assert report["continuation_training"]["role"] == "train"
    assert len(report["controls"]) == 6
    assert [(c["family"], c["scenario"]) for c in report["controls"]] == [
        (family, scenario)
        for family in ("mlp", "residual")
        for scenario in ("disconnected_accepted", "energy_damage_rejected", "budget_failed")
    ]
    for control in report["controls"]:
        decision = control["decision"]
        assert all(p["role"] in ("train", "validation") for p in control["probes"])
        if control["scenario"] == "disconnected_accepted":
            assert decision["status"] == "accepted" and decision["reason"] is None
            assert decision["returned_state"]["config"]["width"] == 1
            assert decision["accepted_lineage"]["prune_index"] == 1
            assert len(decision["diagnostics"]) == decision["planned_probe_count"] == 2
            for item in decision["diagnostics"]:
                assert item["max_prediction_change"] <= 1e-12
                assert item["half_mse_increase"] <= 1e-12 and item["energy_increase"] <= 1e-12
        else:
            assert decision["status"] == (
                "rejected" if control["scenario"] == "energy_damage_rejected" else "failed"
            )
            assert decision["accepted_lineage"] is None and decision["reason"]
            assert decision["returned_state"] == control["source"]
            if control["scenario"] == "energy_damage_rejected":
                item = decision["diagnostics"][0]
                assert item["source_energy"] == pytest.approx(0.25)
                assert item["candidate_energy"] == pytest.approx(0.5)
                assert item["max_prediction_change"] == 0 and item["half_mse_increase"] == 0
                assert item["energy_increase"] == pytest.approx(0.25)
            else:
                assert decision["diagnostics"] == [] and control["continuation"] is None
        continuation = control["continuation"]
        if continuation is not None:
            assert np.isfinite(continuation["preupdate_energy"])
            assert continuation["state"]["config"] == decision["returned_state"]["config"]
            for raw in continuation["state"]["raw_weights"]:
                assert np.isfinite(raw).all()
