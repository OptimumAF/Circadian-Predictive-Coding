"""Read the fixed tiny matched-head selection and three-scope CPU artifacts."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from scripts import run_repeated_confirmation_smoke as cli  # noqa: E402
from src.app.matched_head_tuning import HeadTuningAttempt  # noqa: E402
from src.app import repeated_head_confirmation as repeated  # noqa: E402


REPOSITORY = Path(__file__).resolve().parents[1]
HEADS = {"backprop_mlp", "predictive_coding", "circadian_predictive_coding"}
SEEDS = (53, 59, 61)
FILES = {
    name: f"benchmark_repeated_{name}_smoke.json"
    for name in ("selection", "manifest", "result", "failure")
}


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"nonfinite repeated-confirmation value: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    assert isinstance(payload, dict)
    return payload


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_should_read_predeclared_three_scope_cpu_cli_artifacts(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "scripts.run_repeated_confirmation_smoke",
        "--output-dir",
        str(tmp_path),
    ]
    # Why this: the saved grid stays unchanged, while one CPU thread bounds
    # setup and per-head work on shared local and CI hosts.
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    completed = subprocess.run(
        command,
        cwd=REPOSITORY,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert {path.name for path in tmp_path.iterdir()} == {
        FILES[name] for name in ("selection", "manifest", "result")
    }
    paths = {name: tmp_path / filename for name, filename in FILES.items()}
    selection = _read_json(paths["selection"])
    manifest = _read_json(paths["manifest"])
    result = _read_json(paths["result"])
    printed = json.loads(completed.stdout, parse_constant=_reject_nonfinite)
    assert printed["artifacts"] == {
        name: str(paths[name]) for name in ("selection", "manifest", "result")
    }
    assert printed["manifest_digest"] == manifest["manifest_digest"]
    assert printed["selection_seeds"] == [47]
    assert printed["confirmation_seeds"] == list(SEEDS)

    assert selection["protocol_id"] == "vision_matched_head_validation_selection_v1"
    assert selection["seeds"] == [47]
    assert selection["candidates_per_head"] == 2
    assert selection["trials_per_head"] == 2
    assert selection["confirmations"] == []
    assert len(selection["attempts"]) == len(selection["trials"]) == 6
    assert all(row["status"] == "complete" for row in selection["attempts"])
    assert {(row["head_name"], row["candidate_id"]) for row in selection["trials"]} == {
        (head, candidate) for head in HEADS for candidate in ("a", "b")
    }
    assert len(selection["selections"]) == 3
    assert {row["head_name"] for row in selection["selections"]} == HEADS
    assert all("test_accuracy" not in row for row in selection["trials"])
    assert all(row["seed"] == 47 and row["seen_samples"] == 8 for row in selection["trials"])
    assert all(
        row["sleep_attempts"] == 1
        for row in selection["trials"]
        if row["head_name"] == "circadian_predictive_coding"
    )
    for field in ("backbone_hash", "initial_head_hash", "split_hashes", "feature_hashes"):
        assert len({json.dumps(row[field], sort_keys=True) for row in selection["trials"]}) == 1
    assert set(selection["trials"][0]["feature_hashes"]) == {"train", "guard", "validation"}

    restored = repeated.restore_confirmation_manifest(manifest)
    assert restored.manifest_digest == manifest["manifest_digest"]
    source_digest = sha256(
        json.dumps(selection, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert manifest["source_selection_digest"] == source_digest
    assert manifest["selection_seeds"] == [47]
    assert manifest["confirmation_seeds"] == list(SEEDS)
    assert manifest["metric_names"] == ["accuracy", "cross_entropy"]
    assert manifest["scopes"] == ["fixed_data_epoch", "fixed_wall_time", "isolated_capacity_memory"]
    assert manifest["wall_time_budget_seconds"] == 0.05
    assert manifest["wall_time_epoch_cap"] == 1000
    assert {head["head_name"] for head in manifest["selected_heads"]} == HEADS
    assert {(head["head_name"], head["candidate_id"]) for head in manifest["selected_heads"]} == {
        (row["head_name"], row["candidate_id"]) for row in selection["selections"]
    }

    assert result["protocol_id"] == "vision_matched_head_repeated_confirmation_v1"
    assert result["manifest"] == manifest
    fixed = result["fixed_data"]
    assert fixed["seeds"] == list(SEEDS)
    assert fixed["candidates_per_head"] == 1
    assert len(fixed["attempts"]) == len(fixed["trials"]) == len(fixed["confirmations"]) == 9
    assert {(row["head_name"], row["seed"]) for row in fixed["trials"]} == {
        (head, seed) for head in HEADS for seed in SEEDS
    }
    assert {(row["head_name"], row["seed"]) for row in fixed["confirmations"]} == {
        (head, seed) for head in HEADS for seed in SEEDS
    }
    assert len(result["wall_time"]) == len(result["capacity_memory"]) == 3
    for seed, wall, memory in zip(
        SEEDS, result["wall_time"], result["capacity_memory"], strict=True
    ):
        assert memory["config"]["seed"] == seed
        assert wall["wall_time_budget_seconds"] == 0.05
        assert set(memory["reports"]) == HEADS
        assert set(wall["feature_hashes"]) == {"train", "guard", "validation", "test"}
        assert len({report["pid"] for report in memory["reports"].values()}) == 3
        for head, field in (
            ("backprop_mlp", "backprop"),
            ("predictive_coding", "predictive_coding"),
            ("circadian_predictive_coding", "circadian"),
        ):
            trial = next(
                row for row in fixed["trials"] if (row["head_name"], row["seed"]) == (head, seed)
            )
            confirmed = next(
                row
                for row in fixed["confirmations"]
                if (row["head_name"], row["seed"]) == (head, seed)
            )
            observed = memory["reports"][head]
            assert trial["trained_head_hash"] == confirmed["trained_head_hash"]
            assert trial["backbone_hash"] == wall["backbone_hash"] == observed["backbone_hash"]
            assert (
                trial["initial_head_hash"]
                == wall["initial_head_hashes"][head]
                == observed["initial_head_hash"]
            )
            # Why this: trials and isolated memory are development-only;
            # the wall-time completed report also names the final role.
            assert (
                trial["split_hashes"]
                == observed["split_hashes"]
                == {role: wall["split_hashes"][role] for role in ("train", "guard", "validation")}
            )
            assert (
                trial["feature_hashes"]
                == observed["feature_hashes"]
                == {role: wall["feature_hashes"][role] for role in ("train", "guard", "validation")}
            )
            assert confirmed["test_feature_hash"] == wall["feature_hashes"]["test"]
            assert confirmed["test_split_hash"] == wall["split_hashes"]["test"]
            assert wall[field]["stop_reason"] == "deadline"
            assert observed["head_parameters"] == observed["final_head_parameters"]
            assert observed["cuda_allocated_peak_bytes"] is None
    for name in ("fixed_data_accuracy", "wall_time_accuracy", "observed_train_rss"):
        assert set(result[name]) == HEADS
        assert set(printed[name]) == HEADS
        assert all(
            row["seeds"] == list(SEEDS) and len(row["values"]) == 3 for row in result[name].values()
        )

    before = {name: _digest(paths[name]) for name in ("selection", "manifest", "result")}
    occupied = subprocess.run(
        command,
        cwd=REPOSITORY,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert occupied.returncode != 0
    assert "already exist" in occupied.stderr
    assert before == {name: _digest(paths[name]) for name in before}


@pytest.mark.parametrize("name", FILES)
def test_should_refuse_any_occupied_output_before_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    path = tmp_path / FILES[name]
    path.write_bytes(b"previous user content")
    monkeypatch.setattr(
        cli, "run_matched_head_tuning", lambda *args, **kwargs: pytest.fail("selection started")
    )
    with pytest.raises(FileExistsError, match="already exist"):
        cli.main(output_dir=tmp_path)
    assert path.read_bytes() == b"previous user content"


def test_should_preserve_selection_manifest_and_failure_after_confirmation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")

    def fail_confirmation(manifest: Any) -> Any:
        assert (tmp_path / FILES["selection"]).is_file()
        assert (tmp_path / FILES["manifest"]).is_file()
        assert (
            repeated.restore_confirmation_manifest(_read_json(tmp_path / FILES["manifest"]))
            == manifest
        )
        raise RuntimeError("injected confirmation interruption")

    monkeypatch.setattr(cli, "run_repeated_confirmation", fail_confirmation)
    with pytest.raises(RuntimeError, match="injected confirmation interruption"):
        cli.main(output_dir=tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        FILES[name] for name in ("selection", "manifest", "failure")
    }
    failure = _read_json(tmp_path / FILES["failure"])
    assert failure["schema_id"] == cli.FAILURE_SCHEMA
    assert failure["stage"] == "confirmation"
    assert failure["error_type"] == "RuntimeError"
    assert failure["error"] == "injected confirmation interruption"
    assert failure["existing_artifacts"] == {
        name: {
            "path": str(tmp_path / FILES[name]),
            "sha256": _digest(tmp_path / FILES[name]),
        }
        for name in ("selection", "manifest")
    }
    assert not (tmp_path / FILES["result"]).exists()


def test_should_save_incomplete_selection_attempt_without_a_false_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = HeadTuningAttempt(
        head_name="backprop_mlp",
        candidate_id="a",
        seed=47,
        config=cli._base_config(),
        status="started",
    )

    def fail_selection(*args: Any, **kwargs: Any) -> Any:
        raise cli.MatchedHeadTuningError("injected selection failure", (attempt,), ())

    monkeypatch.setattr(cli, "run_matched_head_tuning", fail_selection)
    with pytest.raises(cli.MatchedHeadTuningError, match="injected selection failure"):
        cli.main(output_dir=tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {FILES["failure"]}
    failure = _read_json(tmp_path / FILES["failure"])
    assert failure["schema_id"] == cli.FAILURE_SCHEMA
    assert failure["stage"] == "selection"
    assert failure["error_type"] == "MatchedHeadTuningError"
    assert failure["existing_artifacts"] == {}
    assert failure["attempts"] == [json.loads(json.dumps(asdict(attempt)))]
    assert failure["trials"] == failure["selections"] == []
