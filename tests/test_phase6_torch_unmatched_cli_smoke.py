"""Bounded real-process Torch CPU outputs for the unmatched synthetic vision CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")


REPOSITORY = Path(__file__).resolve().parents[1]
MODEL_NAMES = (
    "BackpropResNet50",
    "PredictiveCodingResNet50",
    "CircadianPredictiveCodingResNet50",
)
COMMON_ARGS = (
    "--protocol-id",
    "vision_guard_separated_unmatched_v2",
    "--dataset-name",
    "synthetic",
    "--dataset-no-download",
    "--train-samples",
    "4",
    "--guard-samples",
    "4",
    "--validation-samples",
    "4",
    "--test-samples",
    "4",
    "--classes",
    "3",
    "--image-size",
    "32",
    "--batch-size",
    "4",
    "--epochs",
    "1",
    "--seed",
    "47",
    "--device",
    "cpu",
    "--target-accuracy",
    "-1",
    "--backprop-freeze-backbone",
    "--backbone-weights",
    "none",
    "--pc-hidden-dim",
    "8",
    "--pc-steps",
    "1",
    "--circ-hidden-dim",
    "8",
    "--circ-min-hidden-dim",
    "8",
    "--circ-max-hidden-dim",
    "8",
    "--circ-steps",
    "1",
    "--circ-sleep-interval",
    "1",
    "--circ-disable-adaptive-sleep-trigger",
    "--circ-sleep-warmup-steps",
    "0",
    "--inference-batches",
    "1",
    "--warmup-batches",
    "0",
    "--eval-batches",
    "1",
)


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"nonfinite Torch CLI value: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    assert isinstance(payload, dict)
    return payload


def _run_cli(tmp_path: Path, arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = tmp_path / f"{arm}-result.json"
    config_path = tmp_path / f"{arm}-config.json"
    arm_args = ("--circ-force-sleep",) if arm == "forced" else ("--circ-sleep-mode", "disabled")
    command = [
        sys.executable,
        "resnet50_benchmark.py",
        *COMMON_ARGS,
        *arm_args,
        "--json-result",
        str(result_path),
        "--resolved-config",
        str(config_path),
    ]
    # Why this: one CPU thread keeps the tiny real ResNet smoke budgeted on shared hosts.
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    completed = subprocess.run(
        command,
        cwd=REPOSITORY,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Comparison status: legacy unmatched heads" in completed.stdout
    result = _read_json(result_path)
    resolved = _read_json(config_path)
    assert result["resolved_config"] == resolved

    before = (result_path.read_bytes(), config_path.read_bytes())
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
    assert "already exists" in occupied.stderr
    assert (result_path.read_bytes(), config_path.read_bytes()) == before
    return result, resolved


def test_should_read_forced_and_disabled_torch_cpu_vision_artifacts(tmp_path: Path) -> None:
    arms = {arm: _run_cli(tmp_path, arm) for arm in ("forced", "control")}
    forced, forced_config = arms["forced"]
    control, control_config = arms["control"]

    for result, resolved in arms.values():
        config = result["config"]
        assert result["device"] == config["device"] == "cpu"
        assert config["backbone_weights"] == "none"
        assert config["dataset_name"] == "synthetic"
        assert config["dataset_download"] is False
        assert config["target_accuracy"] is None
        assert config["train_samples"] == config["guard_samples"] == 4
        assert config["validation_samples"] == config["test_samples"] == 4
        assert result["training_order"] == ["backprop", "predictive", "circadian"]
        assert set(result["split_hashes"]) == {"train", "validation", "test", "guard"}
        assert all(len(digest) == 64 for digest in result["split_hashes"].values())
        assert resolved["schema_id"] == "resnet_single_resolved_config_v1"
        assert resolved["benchmark_track"] == "unmatched_reference"
        assert resolved["seed"] == config["seed"] == 47
        assert resolved["config"] == config
        reports = result["reports"]
        assert [report["model_name"] for report in reports] == list(MODEL_NAMES)
        assert all(report["epochs_ran"] == 1 for report in reports)
        assert all(report["benchmark_track"] == "unmatched_reference" for report in reports)
        assert all(report["backbone_pretraining"] == "none" for report in reports)
        assert all(report["trainable_parameters"] > 0 for report in reports)
        assert all(not report["sleep_events"] for report in reports[:2])
        assert all(report["circadian_sleep_attempts"] == 0 for report in reports[:2])

    assert forced["split_hashes"] == control["split_hashes"]
    assert forced_config["config"]["circadian_force_sleep"] is True
    assert control_config["config"]["circadian_sleep_mode"] == "disabled"
    forced_head = forced["reports"][2]
    control_head = control["reports"][2]
    assert forced_head["circadian_sleep_attempts"] == 1
    assert control_head["circadian_sleep_attempts"] == 0
    assert len(forced_head["sleep_events"]) == len(control_head["sleep_events"]) == 1
    forced_event = forced_head["sleep_events"][0]
    control_event = control_head["sleep_events"][0]
    assert (forced_event["trigger_reason"], forced_event["outcome"]) == ("periodic", "accepted")
    assert forced_event["guard"]["role"] == "inner_guard"
    assert forced_event["guard"]["role_hash"] == forced["split_hashes"]["guard"]
    assert (control_event["trigger_reason"], control_event["outcome"]) == ("disabled", "skipped")
    assert control_event["reason"] == "sleep_disabled"
    assert control_event["guard"] is None
    assert all(event["replay"]["applied_updates"] == 0 for event in (forced_event, control_event))
