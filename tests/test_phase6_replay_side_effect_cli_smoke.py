"""Read the public fixed v10 ablation without selecting a favorable policy."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts import run_continual_replay_side_effect_ablation as side_effect_cli


REPOSITORY = Path(__file__).resolve().parents[1]


def test_should_reject_occupied_v10_result_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_path = tmp_path / "occupied.json"
    result_path.write_bytes(b"prior result")
    monkeypatch.setattr(sys, "argv", ["ablation", "--result", str(result_path)])
    monkeypatch.setattr(
        side_effect_cli,
        "run_replay_side_effect_ablation",
        lambda *_args, **_kwargs: pytest.fail("trained before occupied result preflight"),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        side_effect_cli.main()
    assert result_path.read_bytes() == b"prior result"


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"nonfinite ablation value: {value}")


def _run_cli(tmp_path: Path, module: str) -> dict[str, Any]:
    result_path = tmp_path / f"{module.rsplit('.', 1)[-1]}.json"
    command = [sys.executable, "-m", module, "--result", str(result_path)]
    completed = subprocess.run(
        command, cwd=REPOSITORY, capture_output=True, text=True, timeout=60, check=False
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads(result_path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    printed = json.loads(completed.stdout, parse_constant=_reject_nonfinite)
    assert isinstance(record, dict)
    assert printed == {
        "result": str(result_path),
        "sha256": sha256(result_path.read_bytes()).hexdigest(),
    }
    original = result_path.read_bytes()
    duplicate = subprocess.run(
        command, cwd=REPOSITORY, capture_output=True, text=True, timeout=30, check=False
    )
    assert duplicate.returncode != 0
    assert "already exists" in duplicate.stderr
    assert result_path.read_bytes() == original
    return record


def _check_scored_seed(row: dict[str, Any]) -> None:
    assert [(boundary["phase"], boundary["epoch"]) for boundary in row["boundaries"]] == [
        ("a", 1),
        ("a", 2),
        ("b", 1),
        ("b", 2),
    ]
    assert len(row["sleep_events_without_durations"]) == 4
    assert {name for name in row["role_hashes"] if name.endswith("final_test")} == {
        "phase_a_final_test",
        "phase_b_final_test",
    }
    assert set(row["metrics"]) == {
        "backprop",
        "predictive_coding",
        "circadian_predictive_coding",
        "replay_retention",
    }
    work = {item["method"]: item for item in row["applied_work"]}
    assert {name: item["inference_iterations"] for name, item in work.items()} == {
        "backprop": 0,
        "predictive_coding": 16,
        "circadian_predictive_coding": 24,
    }
    assert all(item["optimizer_updates"] == item["examples"] == 8 for item in work.values())
    assert all(item["sample_ids"] == work["backprop"]["sample_ids"] for item in work.values())
    for boundary in row["boundaries"]:
        assert len(boundary["selected_ids"]) == 2
        assert boundary["sleep_outcome"] == "accepted"
        assert all(
            item["sample_ids"] == boundary["selected_ids"] for item in boundary["applied_by_method"]
        )


def test_should_keep_v10_historical_and_wake_only_rows_matched(tmp_path: Path) -> None:
    v9 = _run_cli(tmp_path, "scripts.run_continual_matched_replay_outcomes")
    v10 = _run_cli(tmp_path, "scripts.run_continual_replay_side_effect_ablation")
    assert v10["protocol_id"] == "continual_replay_side_effect_ablation_v10"
    assert v10["manifest"]["matched"] == v9["manifest"]
    assert v10["manifest"]["side_effect_policies"] == ["historical", "wake_only_adaptive_v1"]
    assert len(v10["manifest_digest"]) == 64
    historical, wake_only = v10["outcomes"]
    assert [group["side_effect_policy"] for group in v10["outcomes"]] == [
        "historical",
        "wake_only_adaptive_v1",
    ]
    assert historical["retention"] == v9["policies"]
    assert [group["policy"]["name"] for group in wake_only["retention"]] == [
        "recent_fifo",
        "seeded_reservoir",
    ]
    for old_group, new_group in zip(historical["retention"], wake_only["retention"], strict=True):
        assert [row["seed"] for row in old_group["seeds"]] == [17, 19]
        assert [row["seed"] for row in new_group["seeds"]] == [17, 19]
        assert old_group["policy"] == new_group["policy"]
        assert old_group["aggregate"] == new_group["aggregate"]
        for old, new in zip(old_group["seeds"], new_group["seeds"], strict=True):
            _check_scored_seed(old)
            _check_scored_seed(new)
            assert old["boundaries"] == new["boundaries"]
            assert old["applied_work"] == new["applied_work"]
            assert old["role_ids"] == new["role_ids"]
            assert old["role_hashes"] == new["role_hashes"]
            assert old["metrics"]["backprop"] == new["metrics"]["backprop"]
            assert old["metrics"]["predictive_coding"] == new["metrics"]["predictive_coding"]
    for group in v10["outcomes"]:
        for seed_index in (0, 1):
            fifo = group["retention"][0]["seeds"][seed_index]
            reservoir = group["retention"][1]["seeds"][seed_index]
            assert fifo["role_ids"] == reservoir["role_ids"]
            assert fifo["role_hashes"] == reservoir["role_hashes"]
    assert "winner" not in v10
