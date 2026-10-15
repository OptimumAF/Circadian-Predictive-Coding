"""Bounded actual split/merge example and complete continued state."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_coupled_scaling_structure as demo


def _run(*arguments):
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_coupled_scaling_structure", *arguments],
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
        raise AssertionError("description must not initialize or train models")

    monkeypatch.setattr(demo, "_source", forbidden)
    assert demo.describe_example() == _run("--describe")
    assert demo.describe_example()["training_calls"] == 12
    assert demo.describe_example()["objective_evaluations"] == 60


def test_should_split_continue_fork_merge_and_preserve_neutral_parity():
    result = _run()
    assert result["exact_pc_neutral_continuation"]
    assert len(result["records"]) == 3
    assert {row["variant"] for row in result["records"]} == {"pc", "neutral_cpc", "active_cpc"}
    for row in result["records"]:
        source = row["source"]["state"]
        split = row["split"]["state"]
        continued = row["continued"]["state"]
        merged = row["continued_merge"]["state"]
        assert source["config"]["width"] == 3
        assert split["config"] == {**source["config"], "width": 6, "precisions": [0.4, 0.6]}
        assert row["immediate_merge"]["state"]["config"] == merged["config"] == source["config"]
        assert row["lineage"]["source_config"] == source["config"]
        assert row["exact_fork"] and row["strict_restore_refused"]
        assert row["topology_precision_restore_refused"] and row["incompatible_merge_refusal"]
        for check in (row["diagonal_check"], row["continued_diagonal_check"]):
            assert check["prediction_max_error"] <= 1e-12
            assert max(check["latent_gradient_max_errors"]) <= 1e-12
            a, b = check["source_evaluation"], check["split_evaluation"]
            assert abs(a["energy"] - b["energy"]) <= 1e-12
            assert all(value > 0 for value in a["hidden_penalties"])
        if source["wake_state"] is not None:
            assert source["wake_state"]["wake_updates"] == split["wake_state"]["wake_updates"] == 2
            assert any(c > 0 for c in source["wake_state"]["chemical_state"])
            for field in ("chemical_state", "plasticity_state"):
                assert split["wake_state"][field] == [
                    x for x in source["wake_state"][field] for _ in range(2)
                ]
                assert merged["wake_state"][field] == continued["wake_state"][field][::2]
            assert (
                merged["wake_state"]["wake_updates"] == continued["wake_state"]["wake_updates"] == 3
            )
    pc, neutral = result["records"][:2]
    assert (
        pc["continued"]["state"]["raw_parameters"]
        == neutral["continued"]["state"]["raw_parameters"]
    )
