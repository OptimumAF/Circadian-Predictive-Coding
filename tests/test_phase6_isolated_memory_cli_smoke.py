"""Read the public process-isolated matched-head memory diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")


REPO_ROOT = Path(__file__).resolve().parents[1]
HEAD_NAMES = {"backprop_mlp", "predictive_coding", "circadian_predictive_coding"}
DEVELOPMENT_ROLES = {"train", "guard", "validation"}


def reject_nonfinite(token: str) -> Any:
    raise AssertionError(f"Nonfinite isolated-memory JSON output: {token}")


def test_public_memory_cli_reports_three_matched_cpu_processes() -> None:
    process = subprocess.run(
        [sys.executable, "-m", "scripts.run_isolated_head_memory_smoke"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )

    assert process.returncode == 0, process.stderr
    assert process.stderr == ""
    payload = json.loads(process.stdout, parse_constant=reject_nonfinite)
    assert payload["protocol_id"] == "vision_three_head_fixed_width_process_memory_v1"
    assert payload["source_protocol_id"] == "vision_guard_separated_unmatched_v2"
    assert payload["benchmark_track"] == "frozen_shared_representation"
    assert payload["memory_scope"] == "setup_and_trainer_observed_process_rss"
    assert payload["seed"] == 47
    config = payload["config"]
    assert config["seed"] == 47
    assert config["dataset_name"] == "synthetic"
    assert config["device"] == "cpu"
    assert config["backbone_weights"] == "none"
    assert config["train_samples"] == 8
    assert config["guard_samples"] == 8
    assert config["validation_samples"] == 8
    assert config["test_samples"] == 8
    assert config["epochs"] == 1
    assert config["circadian_head_hidden_dim"] == config["circadian_min_hidden_dim"]
    assert config["circadian_head_hidden_dim"] == config["circadian_max_hidden_dim"]

    reports = payload["head_reports"]
    assert set(reports) == HEAD_NAMES
    assert len({report["pid"] for report in reports.values()}) == 3
    assert len({report["backbone_hash"] for report in reports.values()}) == 1
    assert len({report["head_parameters"] for report in reports.values()}) == 1
    reference_features = reports["backprop_mlp"]["feature_hashes"]
    reference_splits = reports["backprop_mlp"]["split_hashes"]
    assert set(reference_features) == DEVELOPMENT_ROLES
    assert set(reference_splits) == DEVELOPMENT_ROLES
    assert all(len(value) == 64 for value in reference_features.values())
    assert all(len(value) == 64 for value in reference_splits.values())
    for name, report in reports.items():
        assert report["pid"] > 0
        assert report["head_parameters"] > 0
        assert report["feature_bytes"] > 0
        assert report["feature_hashes"] == reference_features
        assert report["split_hashes"] == reference_splits
        assert report["guard_examples_scored"] >= 0
        assert report["sleep_attempts"] >= (1 if name == "circadian_predictive_coding" else 0)
        for start, peak in (
            ("setup_rss_start_bytes", "setup_rss_peak_observed_bytes"),
            ("train_rss_start_bytes", "train_rss_peak_observed_bytes"),
        ):
            assert report[start] is None or report[start] > 0
            assert report[peak] is None or report[peak] >= report[start]
