"""Real native fault injection and the public CLI scenario selection."""

import json
from pathlib import Path
import subprocess
import sys
from typing import cast

import pytest

from src.adapters.promotion_rejection_demo import run_rejection_demo


def assert_rejection(report: dict) -> None:
    assert report["protocol_id"] == "native_promotion_rejection_v1"
    assert report["scope"] == "integration_fault_injection"
    assert report["seed"] == 23 and report["method"] == "backprop"
    assert report["completed_updates"] == 1
    assert report["guard_roles"] == ["inner_guard", "inner_guard"]
    guard = report["guard_report"]
    assert guard["decision"]["accepted"] is False
    assert list(guard["decision"]["reasons"]) == ["resource"]
    assert guard["policy"]["max_resource_bytes"] == 0
    assert guard["evidence"]["candidate_resource_bytes"] > 0
    assert guard["evidence"]["numerically_valid"] is True
    assert guard["evidence"]["actions_safe"] is True
    assert list(guard["candidate_revision"]) == [1, 0, 0]
    training = {tuple(key) for key in guard["training_ids"]}
    new = {tuple(key) for key in guard["new_guard_keys"]}
    old = {tuple(key) for key in guard["old_guard_keys"]}
    assert training == {("train", "s0")}
    assert not (training & new or training & old or new & old)
    assert all(tick <= guard["assessed_at"] for tick in guard["label_arrivals"])
    assert guard["actor_snapshot_digest"] != guard["candidate_snapshot_digest"]
    assert report["serving_unchanged"] is True
    assert report["actor_version_before"] == report["actor_version_after"] == "actor-0"
    assert report["actor_predictions_before"] == report["actor_predictions_after"]
    assert report["cache_hit_after"] is True


def test_should_expose_native_rejection_with_disjoint_guards_and_unchanged_serving() -> None:
    assert_rejection(run_rejection_demo())


def test_should_run_rejection_scenario_through_existing_cli() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_native_stream_demo",
            "--scenario",
            "promotion-rejection",
            "--seed",
            "23",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert completed.stderr == ""
    assert_rejection(json.loads(completed.stdout))


@pytest.mark.parametrize("seed", [True, -1, 1.5, 2**32])
def test_should_refuse_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        run_rejection_demo(cast(int, seed))
