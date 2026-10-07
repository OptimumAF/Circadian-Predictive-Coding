"""Read every fixed v11 modulation/control artifact cell without selection."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts import run_difficulty_matched_comparison as difficulty_cli


REPOSITORY = Path(__file__).resolve().parents[1]


def test_should_reject_occupied_v11_result_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_path = tmp_path / "occupied.json"
    result_path.write_bytes(b"prior result")
    monkeypatch.setattr(sys, "argv", ["difficulty", "--result", str(result_path)])
    monkeypatch.setattr(
        difficulty_cli,
        "run_difficulty_comparison",
        lambda *_args, **_kwargs: pytest.fail("trained before occupied result preflight"),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        difficulty_cli.main()
    assert result_path.read_bytes() == b"prior result"


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"nonfinite v11 value: {value}")


def _run_cli(tmp_path: Path) -> dict[str, Any]:
    result_path = tmp_path / "v11.json"
    command = [
        sys.executable,
        "-m",
        "scripts.run_difficulty_matched_comparison",
        "--result",
        str(result_path),
    ]
    completed = subprocess.run(
        command, cwd=REPOSITORY, capture_output=True, text=True, timeout=120, check=False
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads(result_path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    printed = json.loads(completed.stdout, parse_constant=_reject_nonfinite)
    assert isinstance(record, dict)
    original = result_path.read_bytes()
    assert printed == {"result": str(result_path), "sha256": sha256(original).hexdigest()}
    duplicate = subprocess.run(
        command, cwd=REPOSITORY, capture_output=True, text=True, timeout=30, check=False
    )
    assert duplicate.returncode != 0
    assert "already exists" in duplicate.stderr
    assert result_path.read_bytes() == original
    return record


def _check_work_and_roles(row: dict[str, Any]) -> None:
    assert row["work"] == {
        "optimizer_updates": 4,
        "examples": 96,
        "inference_iterations": 8,
        "inference_example_iterations": 192,
        "sleep_events": 0,
        "replay_updates": 0,
    }
    assert set(row["development_role_hashes"]) == {
        "a/train",
        "a/inner_guard",
        "a/outer_selection",
        "b/train",
        "b/inner_guard",
        "b/outer_selection",
    }
    assert set(row["final_role_hashes"]) == {"a", "b"}
    assert set(row["effective_train_hashes"]) == {"a", "b"}
    diagnostics = row["diagnostics"]
    assert [(step["phase"], step["epoch"]) for step in diagnostics] == [
        ("a", 0),
        ("a", 1),
        ("b", 0),
        ("b", 1),
    ]
    assert diagnostics[0]["prior_step_loss_improvement"] is None
    if not row["modulated"]:
        assert all(step["actual_reward_scale"] == 1.0 for step in diagnostics)


def test_should_read_all_v11_matched_difficulty_cells(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    record = _run_cli(tmp_path)
    assert record["protocol_id"] == "difficulty_matched_modulation_v11"
    manifest = record["manifest"]
    assert manifest["seeds"] == [17, 19]
    assert manifest["conditions"] == ["clean", "label_flip", "feature_outlier"]
    assert manifest["backends"] == ["numpy", "torch_cpu"]
    assert manifest["modulation_arms"] == [False, True]
    assert len(record["manifest_digest"]) == 64
    rows = record["outcomes"]
    assert len(rows) == 24
    indexed = {
        (row["seed"], row["condition"], row["backend"], row["modulated"]): row for row in rows
    }
    assert len(indexed) == 24
    for seed in (17, 19):
        seed_rows = [row for row in rows if row["seed"] == seed]
        assert len({tuple(row["development_role_hashes"].items()) for row in seed_rows}) == 1
        assert len({tuple(row["final_role_hashes"].items()) for row in seed_rows}) == 1
        assert len({row["effective_train_hashes"]["a"] for row in seed_rows}) == 1
        assert len({row["effective_train_hashes"]["b"] for row in seed_rows}) == 3
        for condition in manifest["conditions"]:
            for backend in manifest["backends"]:
                control = indexed[seed, condition, backend, False]
                modulated = indexed[seed, condition, backend, True]
                _check_work_and_roles(control)
                _check_work_and_roles(modulated)
                assert control["development_role_hashes"] == modulated["development_role_hashes"]
                assert control["final_role_hashes"] == modulated["final_role_hashes"]
                assert control["effective_train_hashes"] == modulated["effective_train_hashes"]
                assert control["work"] == modulated["work"]
                assert control["forgetting"] == pytest.approx(
                    control["a_after_a_accuracy"] - control["a_after_b_accuracy"]
                )
    assert "winner" not in record
