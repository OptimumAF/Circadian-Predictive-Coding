"""Bounded public v9 schedule, unscored training, and scored output checks.

The fixed FIFO/reservoir rows are all retained; no winner is selected.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts import run_continual_matched_replay_outcomes as outcome_cli
from scripts import run_continual_matched_replay_schedule_smoke as schedule_cli
from scripts import run_continual_matched_replay_training_smoke as training_cli


REPOSITORY = Path(__file__).resolve().parents[1]
COMMANDS = (
    (schedule_cli, "MatchedReplayScheduleSession"),
    (training_cli, "run_matched_replay_training"),
    (outcome_cli, "run_matched_replay_outcomes"),
)


@pytest.mark.parametrize(("module", "entrypoint"), COMMANDS)
def test_should_reject_occupied_v9_result_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: Any, entrypoint: str
) -> None:
    result_path = tmp_path / "occupied.json"
    result_path.write_bytes(b"prior result")
    monkeypatch.setattr(sys, "argv", ["matched", "--result", str(result_path)])
    monkeypatch.setattr(
        module,
        entrypoint,
        lambda *_args, **_kwargs: pytest.fail("worked before occupied result preflight"),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        module.main()
    assert result_path.read_bytes() == b"prior result"


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"nonfinite matched artifact value: {value}")


def _run_cli(tmp_path: Path, module: str) -> dict[str, Any]:
    result_path = tmp_path / f"{module.rsplit('.', 1)[-1]}.json"
    completed = subprocess.run(
        [sys.executable, "-m", module, "--result", str(result_path)],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads(result_path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    printed = json.loads(completed.stdout, parse_constant=_reject_nonfinite)
    assert isinstance(record, dict)
    assert printed["result"] == str(result_path)
    original = result_path.read_bytes()
    assert printed["sha256"] == sha256(original).hexdigest()
    duplicate = subprocess.run(
        [sys.executable, "-m", module, "--result", str(result_path)],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert duplicate.returncode != 0
    assert "already exists" in duplicate.stderr
    assert result_path.read_bytes() == original
    return record


def _rows_by_policy_and_seed(record: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    rows = record["rows"]
    assert len(rows) == 4
    indexed = {(row["policy"]["name"], row["seed"]): row for row in rows}
    assert set(indexed) == {
        (policy, seed) for policy in ("recent_fifo", "seeded_reservoir") for seed in (17, 19)
    }
    return indexed


def _assert_schedule(row: dict[str, Any]) -> None:
    assert row["resolved_manifest"]["seeds"] == [17, 19]
    assert len(row["manifest_digest"]) == 64
    boundaries = row["boundaries"]
    assert [(item["phase"], item["epoch"]) for item in boundaries] == [
        ("a", 1),
        ("a", 2),
        ("b", 1),
        ("b", 2),
    ]
    for boundary in boundaries:
        assert len(boundary["selected_ids"]) == 2
        assert boundary["selected_ids"] == boundary["retained_order_ids"][-2:]
        assert boundary["retention"]["example_count"] <= 4
        assert boundary["retention"]["retained_bytes"] <= 96
        assert {work["method"] for work in boundary["method_work"]} == {
            "backprop",
            "predictive_coding",
            "circadian_predictive_coding",
        }
        assert all(
            work["sample_ids"] == boundary["selected_ids"]
            and work["planned_optimizer_updates"] == 2
            for work in boundary["method_work"]
        )
    assert "metrics" not in row
    assert "final_role_hashes" not in row


def _assert_training(row: dict[str, Any], planned: dict[str, Any]) -> None:
    assert row["resolved_manifest"] == planned["resolved_manifest"]
    assert row["manifest_digest"] == planned["manifest_digest"]
    assert set(row["train_role_hashes"]) == set(row["guard_role_hashes"]) == {"a", "b"}
    assert len(row["boundaries"]) == 4
    for trained, scheduled in zip(row["boundaries"], planned["boundaries"], strict=True):
        assert (trained["phase"], trained["epoch"]) == (scheduled["phase"], scheduled["epoch"])
        assert trained["retained_order_ids"] == scheduled["retained_order_ids"]
        assert trained["selected_ids"] == scheduled["selected_ids"]
        assert trained["sleep_outcome"] in {"accepted", "rolled_back"}
        assert {work["method"] for work in trained["applied_by_method"]} == {
            "backprop",
            "predictive_coding",
            "circadian_predictive_coding",
        }
        if trained["sleep_outcome"] == "accepted":
            assert all(
                work["sample_ids"] == scheduled["selected_ids"]
                for work in trained["applied_by_method"]
            )
        else:
            assert all(work["sample_ids"] == [] for work in trained["applied_by_method"])
    assert "metrics" not in row
    assert "final_role_hashes" not in row


def _assert_outcomes(
    record: dict[str, Any], training: dict[tuple[str, int], dict[str, Any]]
) -> None:
    assert record["manifest"]["seeds"] == [17, 19]
    assert [policy["policy"]["name"] for policy in record["policies"]] == [
        "recent_fifo",
        "seeded_reservoir",
    ]
    assert len(record["manifest_digest"]) == 64
    by_seed: dict[int, list[dict[str, Any]]] = {17: [], 19: []}
    for policy in record["policies"]:
        name = policy["policy"]["name"]
        assert [row["seed"] for row in policy["seeds"]] == [17, 19]
        for row in policy["seeds"]:
            by_seed[row["seed"]].append(row)
            unscored = training[name, row["seed"]]
            assert row["schedule_manifest_digest"] == unscored["manifest_digest"]
            assert row["boundaries"] == unscored["boundaries"]
            assert row["role_hashes"]["phase_a_train"] == unscored["train_role_hashes"]["a"]
            assert row["role_hashes"]["phase_b_train"] == unscored["train_role_hashes"]["b"]
            assert row["role_hashes"]["phase_a_inner_guard"] == unscored["guard_role_hashes"]["a"]
            assert row["role_hashes"]["phase_b_inner_guard"] == unscored["guard_role_hashes"]["b"]
            assert {key for key in row["role_hashes"] if key.endswith("final_test")} == {
                "phase_a_final_test",
                "phase_b_final_test",
            }
            assert set(row["metrics"]) == {
                "backprop",
                "predictive_coding",
                "circadian_predictive_coding",
                "replay_retention",
            }
            assert len(row["sleep_events_without_durations"]) == 4
            assert [
                event["completed_epoch"] for event in row["sleep_events_without_durations"]
            ] == [1, 2, 3, 4]
            assert "sleep_events" not in row["metrics"]["circadian_predictive_coding"]
            assert {work["method"] for work in row["applied_work"]} == {
                "backprop",
                "predictive_coding",
                "circadian_predictive_coding",
            }
    for paired in by_seed.values():
        assert paired[0]["role_hashes"] == paired[1]["role_hashes"]


def test_should_bind_v9_schedule_training_and_all_final_outcomes(tmp_path: Path) -> None:
    schedule = _run_cli(tmp_path, "scripts.run_continual_matched_replay_schedule_smoke")
    training = _run_cli(tmp_path, "scripts.run_continual_matched_replay_training_smoke")
    outcomes = _run_cli(tmp_path, "scripts.run_continual_matched_replay_outcomes")
    assert schedule["protocol_id"] == "continual_matched_replay_schedule_v9"
    assert training["protocol_id"] == "continual_matched_replay_training_v9"
    assert outcomes["protocol_id"] == "continual_matched_replay_outcomes_v9"
    planned = _rows_by_policy_and_seed(schedule)
    trained = _rows_by_policy_and_seed(training)
    for key in planned:
        _assert_schedule(planned[key])
        _assert_training(trained[key], planned[key])
    _assert_outcomes(outcomes, trained)
