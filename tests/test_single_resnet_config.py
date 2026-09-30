"""Parity and artifact gates for the documented single-run ResNet CLI."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest

from src.adapters import resnet_benchmark_cli as cli
from src.app.resnet50_benchmark import (
    ResNet50BenchmarkConfig,
    ResNet50BenchmarkResult,
    VISION_DEFAULT_MODEL_ORDER,
)
from src.app.single_resnet_experiment_config import get_single_resnet_preset


def _digest(config: ResNet50BenchmarkConfig) -> str:
    raw = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(raw.encode("utf-8")).hexdigest()


def _fake_result(config: ResNet50BenchmarkConfig) -> ResNet50BenchmarkResult:
    return ResNet50BenchmarkResult(
        device="cpu",
        config=config,
        reports=[],
        split_hashes={},
        training_order=VISION_DEFAULT_MODEL_ORDER,
    )


def test_should_preserve_historical_single_resnet_preset_identity() -> None:
    preset = get_single_resnet_preset()
    assert preset.preset_id == "historical-single-unmatched"
    assert _digest(preset.config) == (
        "f7a4b9664fc73e2260c60a7dcb02929d254f9390723e1093270807364b593775"
    )
    with pytest.raises(ValueError, match="unknown single-run ResNet preset"):
        get_single_resnet_preset("unknown")


@pytest.mark.parametrize(
    "arguments,expected_digest",
    [
        (
            [],
            "f7a4b9664fc73e2260c60a7dcb02929d254f9390723e1093270807364b593775",
        ),
        (
            ["--preset", "historical-single-unmatched"],
            "f7a4b9664fc73e2260c60a7dcb02929d254f9390723e1093270807364b593775",
        ),
        (
            [
                "--dataset-name",
                "cifar100",
                "--classes",
                "100",
                "--dataset-train-subset-size",
                "20000",
                "--dataset-test-subset-size",
                "5000",
                "--epochs",
                "12",
                "--device",
                "cuda",
                "--target-accuracy",
                "-1",
                "--backprop-freeze-backbone",
                "--backbone-weights",
                "imagenet",
            ],
            "2ef3ac40d8cbe27aee7f292cf26d2188e4fc4680608801f614058f608391f349",
        ),
    ],
)
def test_should_keep_default_and_readme_flag_configs_and_stdout(
    arguments: list[str],
    expected_digest: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[ResNet50BenchmarkConfig] = []
    monkeypatch.setattr(sys, "argv", ["resnet50_benchmark.py", *arguments])
    monkeypatch.setattr(cli, "run_resnet50_benchmark", lambda config: captured.append(config))
    monkeypatch.setattr(cli, "format_resnet50_benchmark_result", lambda result: "summary")
    cli.main()
    assert len(captured) == 1
    assert _digest(captured[0]) == expected_digest
    assert capsys.readouterr().out == "summary\n"


def test_should_save_complete_exact_config_and_descriptive_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_path = tmp_path / "result.json"
    config_path = tmp_path / "resolved.json"
    captured: list[ResNet50BenchmarkConfig] = []

    def fake_run(config: ResNet50BenchmarkConfig) -> ResNet50BenchmarkResult:
        captured.append(config)
        return _fake_result(config)

    monkeypatch.setattr(cli, "run_resnet50_benchmark", fake_run)
    monkeypatch.setattr(cli, "format_resnet50_benchmark_result", lambda result: "summary")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "resnet50_benchmark.py",
            "--dataset-name",
            "cifar100",
            "--classes",
            "100",
            "--train-samples",
            "0",
            "--test-samples",
            "0",
            "--epochs",
            "4",
            "--target-accuracy",
            "-1",
            "--override",
            "epochs=2",
            "--json-result",
            str(result_path),
            "--resolved-config",
            str(config_path),
        ],
    )
    cli.main()
    assert captured[0].epochs == 2
    assert captured[0].target_accuracy is None
    assert captured[0].train_samples == captured[0].test_samples == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    record = json.loads(config_path.read_text(encoding="utf-8"))
    assert record == result["resolved_config"]
    assert record["schema_id"] == "resnet_single_resolved_config_v1"
    assert record["preset"] == "historical-single-unmatched"
    assert record["benchmark_track"] == "unmatched_reference"
    assert record["config"] == result["config"] == json.loads(json.dumps(asdict(captured[0])))
    assert record["seed"] == captured[0].seed
    assert record["training_order"] == ["backprop", "predictive", "circadian"]
    assert record["overrides"] == {"epochs": 2}
    assert result["reports"] == []


@pytest.mark.parametrize(
    "arguments,match",
    [
        (["--preset", "unknown"], "invalid choice"),
        (["--override", "missing=1"], "unknown"),
        (["--override", "epochs=2", "--override", "epochs=3"], "duplicate"),
        (["--override", "epochs=true"], "integer"),
        (["--override", "backprop_learning_rate=NaN"], "finite JSON"),
        (["--backprop-lr", "nan"], "finite"),
        (["--epochs", "0"], "epochs"),
        (["--dataset-name", "cifar100", "--classes", "10"], "num_classes"),
        (["--circ-sleep-mode", "legacy", "--circ-disable-split"], "components"),
    ],
)
def test_should_reject_bad_settings_before_runner(
    arguments: list[str],
    match: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "result.json"
    monkeypatch.setattr(
        sys, "argv", ["resnet50_benchmark.py", "--json-result", str(path), *arguments]
    )
    monkeypatch.setattr(
        cli, "run_resnet50_benchmark", lambda config: pytest.fail("invalid input reached runner")
    )
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 2
    assert match in capsys.readouterr().err
    assert not path.exists()


def test_should_require_artifact_for_override_and_refuse_occupied_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "run_resnet50_benchmark", lambda config: pytest.fail("reached runner"))
    monkeypatch.setattr(sys, "argv", ["resnet50_benchmark.py", "--override", "epochs=2"])
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 2
    assert "--json-result or --resolved-config" in capsys.readouterr().err
    path = tmp_path / "existing.json"
    path.write_text("previous user content", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["resnet50_benchmark.py", "--resolved-config", str(path)])
    with pytest.raises(FileExistsError):
        cli.main()
    assert path.read_text(encoding="utf-8") == "previous user content"

    fresh = tmp_path / "same.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["resnet50_benchmark.py", "--json-result", str(fresh), "--resolved-config", str(fresh)],
    )
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 2
    assert not fresh.exists()


def test_should_reject_runner_config_drift_before_artifact_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["resnet50_benchmark.py", "--json-result", str(path)])
    monkeypatch.setattr(
        cli,
        "run_resnet50_benchmark",
        lambda config: _fake_result(replace(config, epochs=config.epochs + 1)),
    )
    with pytest.raises(ValueError, match="config differs"):
        cli.main()
    assert not path.exists()


def test_should_reject_runner_order_drift_before_artifact_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["resnet50_benchmark.py", "--json-result", str(path)])
    monkeypatch.setattr(
        cli,
        "run_resnet50_benchmark",
        lambda config: replace(
            _fake_result(config), training_order=("circadian", "predictive", "backprop")
        ),
    )
    with pytest.raises(ValueError, match="order differs"):
        cli.main()
    assert not path.exists()


def test_should_preserve_valid_ignored_cifar_flags_and_zero_backprop_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[ResNet50BenchmarkConfig] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "resnet50_benchmark.py",
            "--dataset-name",
            "cifar100",
            "--train-samples",
            "-1",
            "--test-samples",
            "-1",
            "--dataset-root",
            "",
            "--backprop-lr",
            "0",
        ],
    )
    monkeypatch.setattr(cli, "run_resnet50_benchmark", lambda config: captured.append(config))
    monkeypatch.setattr(cli, "format_resnet50_benchmark_result", lambda result: "summary")
    cli.main()
    assert captured[0].num_classes == 100
    assert captured[0].train_samples == captured[0].test_samples == -1
    assert captured[0].dataset_data_root == ""
    assert captured[0].backprop_learning_rate == 0.0
