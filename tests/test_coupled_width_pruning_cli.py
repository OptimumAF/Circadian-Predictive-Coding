"""Actual native transfer/gate CLI: twelve wakes, twenty-four guarded inferences."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def _invoke(*arguments):
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_coupled_width_pruning", *arguments],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    assert result.stderr == "" and len(result.stdout.encode("utf-8")) < 1024**2
    return json.loads(result.stdout)


def test_should_describe_without_training_or_inference():
    report = _invoke("--describe")
    assert (
        "controls" not in report
        and report["train_calls"] == 12
        and report["guarded_infer_calls"] == 24
    )


def test_should_transfer_native_wake_state_gate_damage_and_keep_neutral_continuation():
    report = _invoke()
    assert len(report["controls"]) == 9 and all(report["pc_neutral_parameter_parity"].values())
    assert all(p["role"] in ("train", "validation") for p in report["probes"])
    for control in report["controls"]:
        decision, source = control["decision"], control["source"]
        model = decision["returned_model"]
        if control["scenario"] == "disconnected_accepted":
            assert decision["status"] == "accepted" and model["hidden_dims"] == [3, 2]
            assert decision["accepted_lineage"]["unit_index"] == 2
            assert len(decision["diagnostics"]) == decision["planned_probe_count"] == 2
            for item in decision["diagnostics"]:
                assert item["max_prediction_change"] <= 1e-12
                assert item["bce_increase"] <= 1e-12 and item["inferred_energy_increase"] <= 1e-12
                for key in ("source_inference", "candidate_inference"):
                    assert item[key]["reason"] == "max_steps"
                    assert item[key]["attempted_steps"] == item[key]["accepted_steps"] == 2
                    assert not item[key]["used_fallback"]
            if model["wake_state"] is not None:
                assert (
                    model["wake_state"]["chemical_state"]
                    == source["wake_state"]["chemical_state"][:2]
                )
                assert (
                    model["wake_state"]["plasticity_state"]
                    == source["wake_state"]["plasticity_state"][:2]
                )
        else:
            assert decision["status"] == (
                "rejected" if control["scenario"] == "damage_rejected" else "failed"
            )
            assert decision["accepted_lineage"] is None and model == source
        if model["wake_state"] is not None:
            assert model["wake_state"]["wake_updates"] == 2
            assert any(x > 0 for x in model["wake_state"]["chemical_state"])
            if control["variant"] == "active_cpc":
                assert any(x < 1 for x in model["wake_state"]["plasticity_state"])
            else:
                assert all(x == 1 for x in model["wake_state"]["plasticity_state"])
        continuation = control["continuation"]
        if continuation is not None:
            assert continuation["training_role"] == "train" and np.isfinite(
                continuation["preupdate_energy"]
            )
            if continuation["model"]["wake_state"] is not None:
                assert continuation["model"]["wake_state"]["wake_updates"] == 3
