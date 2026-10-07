"""Read fixed v14 schedule, train-only, and scored public artifacts."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts import run_continual_trigger_replay_outcomes as outcome_cli
from scripts import run_continual_trigger_replay_schedule as schedule_cli
from scripts import run_continual_trigger_replay_training as training_cli


REPOSITORY = Path(__file__).resolve().parents[1]
COMMANDS = (schedule_cli, training_cli, outcome_cli)
METHODS = {"backprop", "predictive_coding", "circadian_predictive_coding"}
ARMS = ("periodic", "adaptive", "no_sleep")


@pytest.mark.parametrize("module", COMMANDS)
def test_should_reject_occupied_v14_result_before_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: Any
) -> None:
    path = tmp_path / "occupied.json"
    path.write_bytes(b"prior result")
    monkeypatch.setattr(sys, "argv", ["v14", "--result", str(path)])
    monkeypatch.setattr(
        module, "build_payload", lambda: pytest.fail("worked before occupied result preflight")
    )
    with pytest.raises(FileExistsError, match="already exists"):
        module.main()
    assert path.read_bytes() == b"prior result"


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"nonfinite v14 value: {value}")


def _run_cli(tmp_path: Path, module: str) -> dict[str, Any]:
    path = tmp_path / f"{module.rsplit('.', 1)[-1]}.json"
    command = [sys.executable, "-m", module, "--result", str(path)]
    completed = subprocess.run(
        command, cwd=REPOSITORY, capture_output=True, text=True, timeout=120, check=False
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    printed = json.loads(completed.stdout, parse_constant=_reject_nonfinite)
    assert isinstance(record, dict)
    original = path.read_bytes()
    assert printed == {"result": str(path), "sha256": sha256(original).hexdigest()}
    duplicate = subprocess.run(
        command, cwd=REPOSITORY, capture_output=True, text=True, timeout=30, check=False
    )
    assert duplicate.returncode != 0
    assert "already exists" in duplicate.stderr
    assert path.read_bytes() == original
    return record


def _schedule_by_seed(record: dict[str, Any]) -> dict[int, dict[str, Any]]:
    assert record["protocol_id"] == "continual_trigger_opportunities_v14"
    rows = record["rows"]
    assert [row["seed"] for row in rows] == [47, 53]
    indexed = {row["seed"]: row for row in rows}
    for row in rows:
        assert len(row["opportunities"]) == 24
        for index, item in enumerate(row["opportunities"]):
            assert (item["phase"], item["epoch"], item["global_epoch"]) == (
                "a" if index < 12 else "b",
                index % 12 + 1,
                index + 1,
            )
            assert item["retention"]["example_count"] <= 8
            assert item["retention"]["retained_bytes"] <= 192
            assert len(item["selected_ids"]) == 2
            assert item["selected_ids"] == item["retained_order_ids"][-2:]
            assert {work["method"] for work in item["method_work"]} == METHODS
            assert all(
                work["sample_ids"] == item["selected_ids"]
                and work["planned_optimizer_updates"] == 2
                for work in item["method_work"]
            )
    return indexed


def _training_by_cell(record: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    assert record["protocol_id"] == "continual_trigger_replay_train_only_v14"
    rows = record["rows"]
    assert [(row["seed"], row["arm"]) for row in rows] == [
        (seed, arm) for seed in (47, 53) for arm in ARMS
    ]
    indexed = {(row["seed"], row["arm"]): row for row in rows}
    for row in rows:
        assert len(row["opportunities"]) == 24
        assert set(row["phase_a_development_role_hashes"]) == {
            "train",
            "inner_guard",
            "outer_selection",
        }
        assert set(row["phase_b_development_role_hashes"]) == {
            "train",
            "inner_guard",
            "outer_selection",
        }
        assert all(access["role"] != "final_test" for access in row["role_accesses"])
        assert "final_role_hashes" not in row
        assert "methods" not in row
    return indexed


def _check_matched_opportunities(
    planned: dict[int, dict[str, Any]], trained: dict[tuple[int, str], dict[str, Any]]
) -> None:
    for (seed, arm), row in trained.items():
        assert row["manifest_digest"] == planned[seed]["manifest_digest"]
        accepted = 0
        for scheduled, actual in zip(
            planned[seed]["opportunities"], row["opportunities"], strict=True
        ):
            for name in (
                "phase",
                "epoch",
                "global_epoch",
                "train_role_hash",
                "retention",
                "retained_order_ids",
                "selected_ids",
            ):
                assert actual[name] == scheduled[name]
            assert {work["method"] for work in actual["applied_by_method"]} == METHODS
            if actual["event"]["outcome"] == "accepted":
                accepted += 1
                assert arm == "periodic" and actual["global_epoch"] % 4 == 0
                assert all(
                    work["sample_ids"] == scheduled["selected_ids"]
                    and work["optimizer_updates"] == 2
                    for work in actual["applied_by_method"]
                )
            else:
                assert all(work["optimizer_updates"] == 0 for work in actual["applied_by_method"])
        assert accepted == (6 if arm == "periodic" else 0)
    for seed in (47, 53):
        same_seed = [trained[seed, arm] for arm in ARMS]
        assert len({tuple(map(tuple, row["initial_parameter_digests"])) for row in same_seed}) == 1
        assert (
            len({tuple(row["phase_a_development_role_hashes"].items()) for row in same_seed}) == 1
        )
        assert (
            len({tuple(row["phase_b_development_role_hashes"].items()) for row in same_seed}) == 1
        )


def _check_outcomes(record: dict[str, Any], trained: dict[tuple[int, str], dict[str, Any]]) -> None:
    assert record["protocol_id"] == "continual_trigger_replay_outcomes_v14"
    assert [(row["seed"], row["arm"]) for row in record["outcomes"]] == [
        (seed, arm) for seed in (47, 53) for arm in ARMS
    ]
    assert len(record["contrasts"]) == 18
    for row in record["outcomes"]:
        unscored = trained[row["seed"], row["arm"]]
        assert (row["sleep_attempts"], row["sleep_accepted"], row["sleep_rolled_back"]) == (
            (6, 6, 0) if row["arm"] == "periodic" else (0, 0, 0)
        )
        assert row["circadian_replay_exposure"] == unscored["circadian_replay_exposure"]
        assert set(dict(row["final_role_hashes"])) == {"a", "b"}
        assert {item["method"] for item in row["methods"]} == METHODS
    for seed in (47, 53):
        rows = [row for row in record["outcomes"] if row["seed"] == seed]
        assert all(row["final_role_hashes"] == rows[0]["final_role_hashes"] for row in rows)
        assert all(row["final_role_ids"] == rows[0]["final_role_ids"] for row in rows)
    assert "winner" not in record


def test_should_bind_v14_schedule_train_only_and_all_scored_cells(tmp_path: Path) -> None:
    schedule = _run_cli(tmp_path, "scripts.run_continual_trigger_replay_schedule")
    training = _run_cli(tmp_path, "scripts.run_continual_trigger_replay_training")
    outcomes = _run_cli(tmp_path, "scripts.run_continual_trigger_replay_outcomes")
    assert schedule["resolved_manifest"] == training["resolved_manifest"] == outcomes["manifest"]
    assert training["manifest_digest"] == outcomes["manifest_digest"]
    planned = _schedule_by_seed(schedule)
    trained = _training_by_cell(training)
    _check_matched_opportunities(planned, trained)
    _check_outcomes(outcomes, trained)
