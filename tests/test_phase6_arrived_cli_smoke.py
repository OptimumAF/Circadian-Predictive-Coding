"""Real-process v6/v7 stdout artifacts on their fixed bounded fixtures.

These checks establish complete public reports, not a candidate ranking.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
ROLE_NAMES = {
    f"phase_{phase}_{role}"
    for phase in ("a", "b")
    for role in ("train", "inner_guard", "outer_selection", "final_test")
}
METHODS = {"backprop", "predictive_coding", "circadian_predictive_coding"}


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"nonfinite arrived artifact value: {value}")


def _run_stdout_json(tmp_path: Path, module: str) -> dict[str, Any]:
    path = tmp_path / f"{module.rsplit('.', 1)[-1]}.json"
    with path.open("x", encoding="utf-8", newline="\n") as output:
        completed = subprocess.run(
            [sys.executable, "-m", module],
            cwd=REPOSITORY,
            stdout=output,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            check=False,
        )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    assert isinstance(report, dict)
    assert path.read_text(encoding="utf-8").endswith("\n")
    return report


def test_should_publish_v6_arrived_role_and_guard_stdout(tmp_path: Path) -> None:
    report = _run_stdout_json(tmp_path, "scripts.run_continual_arrived_smoke")
    assert report["protocol_id"] == "continual_arrived_roles_v6"
    assert report["seeds"] == [17, 19]
    assert [row["seed"] for row in report["seed_results"]] == [17, 19]
    for row in report["seed_results"]:
        assert set(row["role_hashes"]) == ROLE_NAMES
        assert len(set(row["role_hashes"].values())) == len(ROLE_NAMES)
        assert len(row["guard_decisions"]) == 4
        events = row["sleep_events"]
        assert [event["completed_epoch"] for event in events] == [1, 2, 3, 4]
        assert all(event["trigger_reason"] == "periodic" for event in events)
        assert all(event["outcome"] in {"accepted", "rolled_back"} for event in events)
        for phase, phase_events in (("a", events[:2]), ("b", events[2:])):
            assert all(
                event["guard"]["role_hash"] == row["role_hashes"][f"phase_{phase}_inner_guard"]
                for event in phase_events
            )
        assert row["role_access_counts"]["train"] == 12
        assert row["role_access_counts"]["guard"] == 4
        assert row["role_access_counts"]["label_release"] == 8
        assert row["final_release_events"] == 4
        assert len(row["method_task_information"]) == 6
        assert {item["method"] for item in row["method_task_information"]} == METHODS


def test_should_publish_v7_all_outer_trials_then_selected_final_stdout(tmp_path: Path) -> None:
    report = _run_stdout_json(tmp_path, "scripts.run_continual_arrived_selection_smoke")
    assert report["protocol_id"] == "continual_arrived_outer_selection_v7"
    assert report["seeds"] == [17, 19]
    assert set(report["candidate_rates"]) == {"default", "lower_rate"}
    assert len(report["trials"]) == 12
    assert {
        (trial["candidate_id"], trial["seed"], trial["method"]) for trial in report["trials"]
    } == {
        (candidate, seed, method)
        for candidate in ("default", "lower_rate")
        for seed in (17, 19)
        for method in METHODS
    }
    for trial in report["trials"]:
        assert trial["train_updates"] == 2
        assert trial["outer_examples_scored"] > 0
        assert len(trial["development_role_hashes"]) == 6
        assert all(event["role"] != "final_test" for event in trial["role_accesses"])
        assert len(trial["sleep_events"]) == (
            2 if trial["method"] == "circadian_predictive_coding" else 0
        )
    selections = report["selections"]
    assert len(selections) == 3
    assert {item["method"] for item in selections} == METHODS
    assert report["freeze"]["choices"] == selections
    assert len(report["freeze"]["freeze_digest"]) == 64
    histories = report["candidate_sleep_histories"]
    assert len(histories) == 4
    assert {(row["candidate_id"], row["seed"]) for row in histories} == {
        (candidate, seed) for candidate in ("default", "lower_rate") for seed in (17, 19)
    }
    assert all(len(row["sleep_events"]) == 2 for row in histories)
    circadian_choice = next(
        item["candidate_id"]
        for item in selections
        if item["method"] == "circadian_predictive_coding"
    )
    assert [row["seed"] for row in report["final_seed_scores"]] == [17, 19]
    for row in report["final_seed_scores"]:
        assert set(row["final_role_hashes"]) == {"phase_a_final_test", "phase_b_final_test"}
        selected = next(
            history["sleep_events"]
            for history in histories
            if (history["candidate_id"], history["seed"]) == (circadian_choice, row["seed"])
        )
        assert row["selected_sleep_events"] == selected
