"""Actual raw pruning CLI outcomes and selected geometry; no scientific study."""

import json
from pathlib import Path
import subprocess
import sys


def _run(*arguments):
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_coupled_scaling_pruning", *arguments],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert not result.stderr
    return json.loads(result.stdout)


def test_should_describe_pruning_fixture_without_model_work():
    description = _run("--describe")
    assert description["source_width"] == 3 and description["hidden_layers"] == 2
    assert description["removed_units"] == [0, 1]
    assert description["training_calls"] == 6
    assert description["guarded_inference_calls"] == 24
    assert description["objective_evaluations"] == 96
    assert "not_scientific" in description["classification"]


def test_should_show_full_raw_fallbacks_accepted_continuation_and_neutral_parity():
    report = _run()
    assert report["exact_pc_neutral_continuation"] is True
    rows = report["controls"]
    assert len(rows) == 9
    assert {row["variant"] for row in rows} == {"pc", "neutral_cpc", "active_cpc"}
    for row in rows:
        decision, source = row["decision"], row["source"]["state"]
        state = decision["state"]
        assert source["config"]["width"] == 3
        if row["scenario"] == "disconnected_accepted":
            assert decision["status"] == "accepted" and state["config"]["width"] == 2
            assert decision["lineage"]["removed_units"] == [0, 1]
            assert (
                0
                <= decision["max_effective_relative_error"]
                <= report["description"]["max_effective_relative_error"]
            )
            assert len(decision["diagnostics"]) == decision["planned_probe_count"] == 2
            assert row["continuation"]["exact_fork"] is True
            assert row["continuation"]["strict_old_restore_refused"] is True
            assert row["continuation"]["state"]["state"]["config"]["width"] == 2
            if state["wake_state"] is not None:
                assert state["wake_state"]["wake_updates"] == 7
                assert len(state["wake_state"]["chemical_state"]) == 2
                assert row["continuation"]["state"]["state"]["wake_state"]["wake_updates"] == 8
        else:
            assert decision["status"] == (
                "rejected" if row["scenario"] == "damage_rejected" else "failed"
            )
            assert decision["reason"] and state == source
            assert decision["lineage"] is decision["max_effective_relative_error"] is None
            assert row["continuation"] is None
