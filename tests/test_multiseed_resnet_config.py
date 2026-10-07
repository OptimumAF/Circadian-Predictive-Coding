"""The descriptive ResNet CLI saves the exact typed settings it runs."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest

from scripts import run_multiseed_resnet_benchmark as script
from src.app.resnet50_benchmark import (
    ModelSpeedReport,
    ResNet50BenchmarkConfig,
    ResNet50BenchmarkResult,
)
from src.app.resnet_experiment_config import MULTISEED_RESNET_PRESET_ID


def _config_digest(config: ResNet50BenchmarkConfig) -> str:
    return sha256(json.dumps(asdict(config), sort_keys=True, allow_nan=False).encode()).hexdigest()


@pytest.mark.parametrize(
    ("arguments", "expected_digest"),
    [
        ([], "8639c5d9a43fde3f3b84d70921e364b43822d2ab22203213317682338565666f"),
        (
            [
                "--dataset-name",
                "synthetic",
                "--classes",
                "10",
                "--epochs",
                "2",
                "--seeds",
                "5",
                "--dataset-no-download",
                "--backprop-train-backbone",
                "--target-accuracy",
                "0.8",
            ],
            "ac82520c19c15f63c1589d17c72c4d9cc248147efc4dc7b3197f730934c8d029",
        ),
    ],
)
def test_should_preserve_current_default_and_flagged_config(
    arguments: list[str], expected_digest: str
) -> None:
    parsed = script.build_parser().parse_args(arguments)
    assert parsed.preset == MULTISEED_RESNET_PRESET_ID
    assert _config_digest(script.build_base_config(parsed)) == expected_digest


def test_should_keep_unused_cifar_synthetic_sample_flags_valid() -> None:
    parsed = script.build_parser().parse_args(["--train-samples", "0", "--test-samples", "0"])

    config = script.build_base_config(parsed)

    assert config.dataset_name == "cifar100"
    assert config.train_samples == 0
    assert config.test_samples == 0


def _report(model_name: str) -> ModelSpeedReport:
    return ModelSpeedReport(
        model_name=model_name,
        epochs_ran=1,
        final_metric_name="cross_entropy",
        final_metric_value=0.5,
        validation_accuracy=0.6,
        test_accuracy=0.5,
        train_seconds=1.0,
        train_samples_per_second=8.0,
        mean_train_step_ms=1.0,
        inference_latency_mean_ms=1.0,
        inference_latency_p95_ms=1.0,
        inference_samples_per_second=8.0,
        total_parameters=100,
        trainable_parameters=100,
        final_cross_entropy=0.5,
    )


def test_should_save_every_actual_trial_config_and_override_after_legacy_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "one-seed"
    arguments = [
        "--output-prefix",
        str(prefix),
        "--seeds",
        "5",
        "--epochs",
        "3",
        "--override",
        "epochs=2",
        "--override",
        "dataset_noise_std=0.1",
    ]
    seen: list[ResNet50BenchmarkConfig] = []

    def fake_runner(config: ResNet50BenchmarkConfig) -> ResNet50BenchmarkResult:
        seen.append(config)
        return ResNet50BenchmarkResult(
            device="cpu",
            config=config,
            reports=[
                _report("BackpropResNet50"),
                _report("PredictiveCodingResNet50"),
                _report("CircadianPredictiveCodingResNet50"),
            ],
            split_hashes={"train": "train-hash", "test": "test-hash"},
        )

    monkeypatch.setattr(script, "run_resnet50_benchmark", fake_runner)
    monkeypatch.setattr(sys, "argv", ["run_multiseed_resnet_benchmark", *arguments])
    script.main()

    saved = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    resolved = saved["resolved_config"]
    assert saved["comparison_status"] == (
        "unmatched reference; validation winners are descriptive only"
    )
    assert resolved["preset"] == MULTISEED_RESNET_PRESET_ID
    assert resolved["seeds"] == [5]
    assert resolved["explicit_inputs"] == arguments
    assert resolved["overrides"] == {"epochs": 2, "dataset_noise_std": 0.1}
    assert resolved["trial_configs"] == [asdict(config) for config in seen]
    assert resolved["base_config"] == asdict(
        script.build_base_config(
            script.build_parser().parse_args(["--seeds", "5", "--epochs", "3"])
        )
    ) | {"epochs": 2, "dataset_noise_std": 0.1}
    assert seen[0].seed == 5
    assert seen[0].epochs == 2
    assert seen[0].dataset_noise_std == 0.1


@pytest.mark.parametrize(
    "argument",
    [
        ["--override", "backprop_learning_rate=0.1"],
        ["--override", "unknown_field=1"],
        ["--override", "epochs=true"],
        ["--override", "epochs=0"],
        ["--override", "dataset_noise_std=NaN"],
        ["--override", "epochs=2", "--override", "epochs=3"],
        ["--dataset-noise-std", "nan"],
        ["--epochs", "-1"],
    ],
)
def test_should_reject_invalid_multiseed_settings_before_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argument: list[str]
) -> None:
    prefix = tmp_path / "invalid"
    monkeypatch.setattr(
        script,
        "run_resnet50_benchmark",
        lambda *_: pytest.fail("runner started before configuration rejection"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_multiseed_resnet_benchmark", "--output-prefix", str(prefix), *argument],
    )
    with pytest.raises((ValueError, SystemExit)):
        script.main()
    assert not prefix.with_suffix(".json").exists()


def test_should_reject_unknown_preset_before_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        script,
        "run_resnet50_benchmark",
        lambda *_: pytest.fail("runner started before preset rejection"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_multiseed_resnet_benchmark",
            "--output-prefix",
            str(tmp_path / "bad"),
            "--preset",
            "other",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        script.main()


def test_should_reject_runner_config_drift_before_writing_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "drift"

    def changed_config(config: ResNet50BenchmarkConfig) -> ResNet50BenchmarkResult:
        return ResNet50BenchmarkResult(
            device="cpu",
            config=replace(config, epochs=config.epochs + 1),
            reports=[],
            split_hashes={},
        )

    monkeypatch.setattr(script, "run_resnet50_benchmark", changed_config)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_multiseed_resnet_benchmark", "--output-prefix", str(prefix), "--seeds", "5"],
    )
    with pytest.raises(ValueError, match="configuration differs"):
        script.main()
    assert not prefix.with_suffix(".json").exists()
