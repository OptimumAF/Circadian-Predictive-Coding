"""Configuration identity and artifact gates for the documented toy CLI."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from src.adapters import cli
from src.app.experiment_runner import ExperimentConfig, run_experiment
from src.app.indepth_comparison import build_indepth_trial_config
from src.app.toy_experiment_config import get_toy_experiment_preset
from src.config.settings import Settings


def _config_hash(config: ExperimentConfig) -> str:
    raw = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(raw.encode("utf-8")).hexdigest()


def _json_config(config: ExperimentConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), allow_nan=False))


def _fixed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PC_BASE_SEED", "7")
    monkeypatch.setenv("PC_DATASET_SIZE", "400")
    monkeypatch.setenv("PC_EPOCHS", "160")


def test_should_preserve_complete_historical_toy_preset_identity() -> None:
    preset = get_toy_experiment_preset(Settings())
    assert preset.preset_id == "historical-toy"
    assert _config_hash(preset.config) == (
        "b9c678b3ded07dad2969412d8f595d75ac5dc25aa987cff86bca80e51ad42801"
    )
    assert preset.indepth_seeds == (3, 7, 11, 19, 23)
    assert preset.indepth_noise_levels == (0.6, 0.8, 1.0)
    with pytest.raises(ValueError, match="unknown toy preset"):
        get_toy_experiment_preset(Settings(), "unknown")


def test_should_preserve_complete_flagged_toy_config_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixed_environment(monkeypatch)
    captured: list[ExperimentConfig] = []
    monkeypatch.setattr(cli, "run_experiment", lambda config: captured.append(config))
    monkeypatch.setattr(cli, "format_experiment_result", lambda result: "summary")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "toy",
            "--samples",
            "80",
            "--epochs",
            "4",
            "--seed",
            "13",
            "--hidden-dims",
            "4,4",
            "--noise",
            "0.7",
            "--adaptive-sleep-trigger",
            "--reward-modulated-learning",
            "--reward-scale-min",
            "0.8",
            "--reward-scale-max",
            "1.4",
        ],
    )
    cli.main()
    assert _config_hash(captured[0]) == (
        "96941126d2990a7a16ce2e3e4084e0357844b48f195dac98e10f192e4db8a19c"
    )


def test_should_keep_default_and_explicit_preset_cli_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixed_environment(monkeypatch)
    captured: list[ExperimentConfig] = []
    monkeypatch.setattr(cli, "run_experiment", lambda config: captured.append(config))
    monkeypatch.setattr(cli, "format_experiment_result", lambda result: "summary")
    for args in ([], ["--preset", "historical-toy"]):
        monkeypatch.setattr(sys, "argv", ["toy", *args])
        cli.main()
    assert len(captured) == 2
    assert captured[0] == captured[1]
    assert _config_hash(captured[0]) == (
        "b9c678b3ded07dad2969412d8f595d75ac5dc25aa987cff86bca80e51ad42801"
    )


def test_should_preserve_historical_indepth_request_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixed_environment(monkeypatch)
    captured: list[tuple[ExperimentConfig, list[int], list[float]]] = []

    def capture(
        *, base_config: ExperimentConfig, seeds: list[int], noise_levels: list[float]
    ) -> Any:
        captured.append((base_config, seeds, noise_levels))
        return object()

    monkeypatch.setattr(cli, "run_indepth_comparison", capture)
    monkeypatch.setattr(cli, "format_indepth_comparison_result", lambda result: "summary")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "toy",
            "--mode",
            "indepth",
            "--samples",
            "80",
            "--epochs",
            "2",
            "--seed-list",
            "13,7",
            "--noise-levels",
            "0.7,1.1",
        ],
    )
    cli.main()
    config, seeds, noise_levels = captured[0]
    raw = json.dumps(
        {"base_config": asdict(config), "seeds": seeds, "noise_levels": noise_levels},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert sha256(raw.encode("utf-8")).hexdigest() == (
        "097f8551dc083593539903c3a47aa1369cfa3c91b7b81a9af44fc200a2b684f4"
    )


def test_should_save_full_baseline_config_used_by_bounded_actual_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixed_environment(monkeypatch)
    path = tmp_path / "toy.json"
    config_path = tmp_path / "resolved.json"
    captured: list[ExperimentConfig] = []

    def run_and_capture(config: ExperimentConfig) -> Any:
        captured.append(config)
        return run_experiment(config)

    monkeypatch.setattr(cli, "run_experiment", run_and_capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "toy",
            "--samples",
            "80",
            "--epochs",
            "1",
            "--json-result",
            str(path),
            "--resolved-config",
            str(config_path),
        ],
    )
    cli.main()
    payload = json.loads(path.read_text(encoding="utf-8"))
    resolved = payload["resolved_config"]
    assert json.loads(config_path.read_text(encoding="utf-8")) == resolved
    assert resolved["schema_id"] == "toy_resolved_config_v1"
    assert resolved["preset"] == "historical-toy"
    assert resolved["mode"] == "baseline"
    assert resolved["config"] == _json_config(captured[0])
    assert resolved["seeds"] == [7]
    assert resolved["noise_levels"] == [0.8]
    assert resolved["trial_configs"] == [_json_config(captured[0])]
    assert payload["protocol_id"] == captured[0].protocol_id
    assert payload["training_order"] == list(captured[0].model_order)


def test_should_save_ordered_indepth_grid_without_score_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixed_environment(monkeypatch)
    path = tmp_path / "indepth-config.json"
    captured: list[tuple[ExperimentConfig, list[int], list[float]]] = []

    def record_grid(
        *, base_config: ExperimentConfig, seeds: list[int], noise_levels: list[float]
    ) -> Any:
        captured.append((base_config, seeds, noise_levels))
        return object()

    monkeypatch.setattr(cli, "run_indepth_comparison", record_grid)
    monkeypatch.setattr(cli, "format_indepth_comparison_result", lambda result: "summary")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "toy",
            "--mode",
            "indepth",
            "--samples",
            "80",
            "--epochs",
            "1",
            "--seed-list",
            "13,7",
            "--noise-levels",
            "0.7,1.1",
            "--resolved-config",
            str(path),
        ],
    )
    cli.main()
    config, seeds, noise_levels = captured[0]
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["mode"] == "indepth"
    assert record["config"] == _json_config(config)
    assert record["seeds"] == seeds == [13, 7]
    assert record["noise_levels"] == noise_levels == [0.7, 1.1]
    assert record["trial_configs"] == [
        _json_config(build_indepth_trial_config(config, seed, noise))
        for noise in noise_levels
        for seed in seeds
    ]
    assert len(record["trial_configs"]) == 4
    assert "score" not in record and "winner" not in record


@pytest.mark.parametrize(
    "args,match",
    [
        (["--override", "backprop_learning_rate=0.9"], "unknown"),
        (["--override", "missing=1"], "unknown"),
        (["--override", "epoch_count=2", "--override", "epoch_count=3"], "duplicate"),
        (["--override", "epoch_count=true"], "integer"),
        (["--override", "noise_scale=NaN"], "finite JSON"),
        (["--epochs", "0"], "epoch_count"),
        (["--noise", "nan"], "finite"),
        (["--reward-scale-min", "2", "--reward-scale-max", "1"], "reward_scale_max"),
        (["--mode", "indepth", "--seed-list", "-1"], "non-negative"),
        (["--mode", "indepth", "--noise-levels", "nan"], "finite"),
    ],
)
def test_should_reject_bad_inputs_before_training(
    args: list[str],
    match: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _fixed_environment(monkeypatch)
    path = tmp_path / "resolved.json"
    monkeypatch.setattr(sys, "argv", ["toy", "--resolved-config", str(path), *args])
    monkeypatch.setattr(
        cli, "run_experiment", lambda config: pytest.fail("bad input reached training")
    )
    monkeypatch.setattr(
        cli, "run_indepth_comparison", lambda **kwargs: pytest.fail("bad grid reached training")
    )
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 2
    assert match in capsys.readouterr().err
    assert not path.exists()


def test_should_require_artifact_for_override_and_protect_existing_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _fixed_environment(monkeypatch)
    monkeypatch.setattr(cli, "run_experiment", lambda config: pytest.fail("reached training"))
    monkeypatch.setattr(sys, "argv", ["toy", "--override", "epoch_count=2"])
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 2
    assert "--json-result or --resolved-config" in capsys.readouterr().err
    path = tmp_path / "resolved.json"
    path.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["toy", "--resolved-config", str(path)])
    with pytest.raises(FileExistsError):
        cli.main()
    assert path.read_text(encoding="utf-8") == "existing"


def test_should_apply_typed_override_after_legacy_flags_and_save_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixed_environment(monkeypatch)
    path = tmp_path / "resolved.json"
    captured: list[ExperimentConfig] = []
    monkeypatch.setattr(cli, "run_experiment", lambda config: captured.append(config))
    monkeypatch.setattr(cli, "format_experiment_result", lambda result: "summary")
    monkeypatch.setattr(
        sys,
        "argv",
        ["toy", "--epochs", "4", "--override", "epoch_count=2", "--resolved-config", str(path)],
    )
    cli.main()
    record = json.loads(path.read_text(encoding="utf-8"))
    assert captured[0].epoch_count == 2
    assert record["config"] == _json_config(captured[0])
    assert record["overrides"] == {"epoch_count": 2}


def test_should_preserve_unused_legacy_validation_fraction_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixed_environment(monkeypatch)
    captured: list[ExperimentConfig] = []
    monkeypatch.setattr(cli, "run_experiment", lambda config: captured.append(config))
    monkeypatch.setattr(cli, "format_experiment_result", lambda result: "summary")
    monkeypatch.setattr(
        sys,
        "argv",
        ["toy", "--protocol-id", "toy_legacy_train_test_v0", "--validation-fraction", "0"],
    )
    cli.main()
    assert captured[0].validation_fraction == 0.0


def test_should_preserve_disabled_replay_flag_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixed_environment(monkeypatch)
    captured: list[ExperimentConfig] = []
    monkeypatch.setattr(cli, "run_experiment", lambda config: captured.append(config))
    monkeypatch.setattr(cli, "format_experiment_result", lambda result: "summary")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "toy",
            "--replay-steps",
            "0",
            "--replay-learning-rate",
            "0",
            "--replay-inference-steps",
            "0",
            "--replay-inference-learning-rate",
            "0",
        ],
    )
    cli.main()
    assert captured[0].circadian_config is not None
    assert captured[0].circadian_config.replay_steps == 0
    assert captured[0].circadian_config.replay_learning_rate == 0.0
