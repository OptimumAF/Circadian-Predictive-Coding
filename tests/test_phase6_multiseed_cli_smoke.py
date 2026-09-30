"""Read the public multiseed writer's complete tiny synthetic CPU output."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = {
    "BackpropResNet50",
    "PredictiveCodingResNet50",
    "CircadianPredictiveCodingResNet50",
}
METRICS = (
    "validation_accuracy",
    "test_accuracy",
    "final_cross_entropy",
    "train_seconds",
    "train_samples_per_second",
    "mean_train_step_ms",
    "inference_latency_mean_ms",
    "inference_latency_p95_ms",
    "inference_samples_per_second",
)


def run_tiny_multiseed(prefix: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_multiseed_resnet_benchmark",
            "--dataset-name",
            "synthetic",
            "--dataset-no-download",
            "--seeds",
            "5,11",
            "--train-samples",
            "8",
            "--validation-samples",
            "8",
            "--guard-samples",
            "8",
            "--test-samples",
            "8",
            "--classes",
            "3",
            "--image-size",
            "32",
            "--batch-size",
            "4",
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--backbone-weights",
            "none",
            "--backprop-freeze-backbone",
            "--target-accuracy",
            "-1",
            "--eval-batches",
            "1",
            "--inference-batches",
            "1",
            "--warmup-batches",
            "0",
            "--output-prefix",
            str(prefix),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def reject_nonfinite(token: str) -> Any:
    raise AssertionError(f"Nonfinite multiseed JSON output: {token}")


def test_public_multiseed_cli_writes_complete_synthetic_rows(tmp_path: Path) -> None:
    prefix = tmp_path / "tiny"
    process = run_tiny_multiseed(prefix)

    assert process.returncode == 0, process.stderr
    assert process.stderr == ""
    assert "[1/2] running seed=5" in process.stdout
    assert "[2/2] running seed=11" in process.stdout
    assert "Winners:" in process.stdout
    expected_paths = {
        prefix.with_suffix(".json"),
        tmp_path / "tiny_per_seed.csv",
        tmp_path / "tiny_summary.csv",
    }
    assert set(tmp_path.iterdir()) == expected_paths
    for path in expected_paths:
        assert f"Wrote {path}" in process.stdout

    payload = json.loads(
        prefix.with_suffix(".json").read_text(encoding="utf-8"),
        parse_constant=reject_nonfinite,
    )
    assert payload["protocol_id"] == "vision_guard_separated_unmatched_v2"
    assert payload["comparison_status"] == (
        "unmatched reference; validation winners are descriptive only"
    )
    assert payload["dataset"]["name"] == "synthetic"
    assert payload["dataset"]["download"] is False
    assert payload["runtime"]["seeds"] == [5, 11]
    assert payload["runtime"]["epochs"] == 1
    assert payload["runtime"]["device"] == "cpu"
    assert payload["runtime"]["backbone_weights"] == "none"
    resolved = payload["resolved_config"]
    assert resolved["schema_id"] == "resnet_multiseed_resolved_config_v1"
    assert resolved["preset"] == "historical-unmatched"
    assert resolved["seeds"] == [5, 11]
    assert resolved["base_config"]["dataset_name"] == "synthetic"
    assert resolved["base_config"]["seed"] == 7
    assert resolved["base_config"]["train_samples"] == 8
    assert [trial["seed"] for trial in resolved["trial_configs"]] == [5, 11]
    assert all(trial["train_samples"] == 8 for trial in resolved["trial_configs"])
    assert resolved["explicit_inputs"][-2:] == ["--output-prefix", str(prefix)]
    assert set(payload["split_hashes_by_seed"]) == {"5", "11"}
    for hashes in payload["split_hashes_by_seed"].values():
        assert set(hashes) == {"train", "guard", "validation", "test"}
        assert all(len(digest) == 64 for digest in hashes.values())

    per_seed = payload["per_seed"]
    assert len(per_seed) == 6
    assert {(row["seed"], row["model_name"]) for row in per_seed} == {
        (seed, name) for seed in (5, 11) for name in MODEL_NAMES
    }
    assert all(row["benchmark_track"] == "unmatched_reference" for row in per_seed)
    assert all(row["backbone_pretraining"] == "none" for row in per_seed)
    assert all(row["backbone_trainable"] is False for row in per_seed)
    assert all(row["epochs_ran"] == 1 for row in per_seed)
    for row in per_seed:
        assert all(math.isfinite(row[metric]) for metric in METRICS)
        assert 0 <= row["validation_accuracy"] <= 1
        assert 0 <= row["test_accuracy"] <= 1
        assert row["total_parameters"] > 0
        assert row["trainable_parameters"] > 0

    summary = payload["summary"]
    assert len(summary) == 3
    assert {row["model_name"] for row in summary} == MODEL_NAMES
    assert all(row["seed_count"] == 2 for row in summary)
    for row in summary:
        source = [item for item in per_seed if item["model_name"] == row["model_name"]]
        assert row["validation_accuracy_mean"] == pytest.approx(
            sum(item["validation_accuracy"] for item in source) / 2
        )
        assert math.isfinite(row["balanced_score"])
        assert all(math.isfinite(row[f"{metric}_mean"]) for metric in METRICS)
        assert all(math.isfinite(row[f"{metric}_std"]) for metric in METRICS)
    assert set(payload["winners"].values()) <= MODEL_NAMES

    per_seed_csv = read_csv(tmp_path / "tiny_per_seed.csv")
    summary_csv = read_csv(tmp_path / "tiny_summary.csv")
    assert len(per_seed_csv) == 6
    assert len(summary_csv) == 3
    assert {(int(row["seed"]), row["model_name"]) for row in per_seed_csv} == {
        (row["seed"], row["model_name"]) for row in per_seed
    }
    for csv_row in per_seed_csv:
        json_row = next(
            row
            for row in per_seed
            if row["seed"] == int(csv_row["seed"]) and row["model_name"] == csv_row["model_name"]
        )
        for metric in METRICS:
            assert float(csv_row[metric]) == json_row[metric]
    for csv_row in summary_csv:
        json_row = next(row for row in summary if row["model_name"] == csv_row["model_name"])
        assert int(csv_row["seed_count"]) == 2
        assert float(csv_row["validation_accuracy_mean"]) == json_row["validation_accuracy_mean"]


@pytest.mark.parametrize("suffix", (".json", "_per_seed.csv", "_summary.csv"))
def test_public_multiseed_cli_refuses_each_occupied_output_before_work(
    tmp_path: Path, suffix: str
) -> None:
    prefix = tmp_path / "occupied"
    occupied = tmp_path / f"occupied{suffix}"
    occupied.write_bytes(b"preserve existing output")

    process = run_tiny_multiseed(prefix)

    assert process.returncode != 0
    assert "Multi-seed output already exists" in process.stderr
    assert "running seed=" not in process.stdout
    assert occupied.read_bytes() == b"preserve existing output"
    assert list(tmp_path.iterdir()) == [occupied]
