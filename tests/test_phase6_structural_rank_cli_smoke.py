"""Read fixed v12 train-only and scored factor artifacts without selection."""

from __future__ import annotations

from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts import run_structural_rank_comparison as structural_cli


REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("train_only", (True, False))
def test_should_reject_occupied_v12_result_before_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, train_only: bool
) -> None:
    result_path = tmp_path / "occupied.json"
    result_path.write_bytes(b"prior result")
    arguments = ["structural", "--result", str(result_path)]
    if train_only:
        arguments.append("--train-only")
    monkeypatch.setattr(sys, "argv", arguments)
    entrypoint = (
        "train_unscored_structural_rank_study" if train_only else "run_structural_rank_comparison"
    )
    monkeypatch.setattr(
        structural_cli,
        entrypoint,
        lambda *_args, **_kwargs: pytest.fail("worked before occupied result preflight"),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        structural_cli.main()
    assert result_path.read_bytes() == b"prior result"


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"nonfinite v12 value: {value}")


def _run_cli(tmp_path: Path, *, train_only: bool) -> dict[str, Any]:
    result_path = tmp_path / ("v12-train.json" if train_only else "v12-result.json")
    command = [
        sys.executable,
        "-m",
        "scripts.run_structural_rank_comparison",
        "--result",
        str(result_path),
    ]
    if train_only:
        command.append("--train-only")
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


def _check_train_fact(fact: dict[str, Any]) -> None:
    assert fact["a_work"] == {
        "wake_updates": 8,
        "wake_examples": 192,
        "inference_loops": 16,
        "example_inference_loops": 384,
        "replay_updates": 0,
        "sleep_events": 1,
    }
    assert fact["total_work"] == {
        "wake_updates": 16,
        "wake_examples": 384,
        "inference_loops": 32,
        "example_inference_loops": 768,
        "replay_updates": 0,
        "sleep_events": 1,
    }
    assert (fact["initial_width"], fact["post_sleep_width"], fact["final_width"]) == (8, 8, 8)
    assert len(fact["split_pairs"]) == len(fact["removed_prune_ids"]) == 1
    assert len(fact["reward_scales"]) == 16
    assert set(fact["role_hashes"]) == {
        "a/train",
        "a/inner_guard",
        "a/outer_selection",
        "b/train",
        "b/inner_guard",
        "b/outer_selection",
    }
    assert "final_role_hashes" not in fact


def test_should_bind_all_v12_unscored_facts_to_final_rows(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    train = _run_cli(tmp_path, train_only=True)
    scored = _run_cli(tmp_path, train_only=False)
    assert train["protocol_id"] == scored["protocol_id"] == "structural_rank_factors_v12"
    assert train["manifest"] == scored["manifest"]
    assert train["manifest_digest"] == scored["manifest_digest"]
    assert len(train["manifest_digest"]) == 64
    assert train["final_released"] is False
    assert "outcomes" not in train
    manifest = train["manifest"]
    assert manifest["seeds"] == [23, 29]
    assert manifest["backends"] == ["numpy", "torch_cpu"]
    assert [
        manifest[name] for name in ("wake_modulation", "reward_weight_history", "rank_importance")
    ] == [[False, True]] * 3
    facts = train["trials"]
    outcomes = scored["outcomes"]
    assert len(facts) == len(outcomes) == 32
    assert facts == [row["train_facts"] for row in outcomes]
    indexed = {
        (
            fact["seed"],
            fact["backend"],
            fact["factors"]["wake_modulation"],
            fact["factors"]["reward_weight_history"],
            fact["factors"]["rank_importance"],
        ): fact
        for fact in facts
    }
    assert set(indexed) == set(
        product((23, 29), ("numpy", "torch_cpu"), (False, True), (False, True), (False, True))
    )
    for fact in facts:
        _check_train_fact(fact)
    for seed in (23, 29):
        seed_rows = [fact for fact in facts if fact["seed"] == seed]
        assert len({tuple(row["role_hashes"].items()) for row in seed_rows}) == 1
        for backend in manifest["backends"]:
            backend_rows = [row for row in seed_rows if row["backend"] == backend]
            assert len(backend_rows) == 8
            assert len({row["initial_parameter_count"] for row in backend_rows}) == 1
            assert len({row["initial_parameter_digest"] for row in backend_rows}) == 1
    for row in outcomes:
        assert set(row["final_role_hashes"]) == {"a", "b"}
        assert all(
            0.0 <= row[name] <= 1.0
            for name in ("a_after_a_accuracy", "a_after_b_accuracy", "b_after_b_accuracy")
        )
        assert row["forgetting"] == pytest.approx(
            row["a_after_a_accuracy"] - row["a_after_b_accuracy"]
        )
    for seed in (23, 29):
        assert (
            len(
                {
                    tuple(row["final_role_hashes"].items())
                    for row in outcomes
                    if row["train_facts"]["seed"] == seed
                }
            )
            == 1
        )
    assert "winner" not in scored
