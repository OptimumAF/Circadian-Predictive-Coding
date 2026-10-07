"""P6.1 bounded real-process smokes for the public NumPy toy CLI.

The results establish artifact and branch function, not a model ranking.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPOSITORY / "predictive_coding_experiment.py"


def _reject_nonfinite(token: str) -> object:
    raise ValueError(f"nonfinite toy artifact value: {token}")


def _read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    assert isinstance(result, dict)
    return result


def _run_toy_cli(output: Path, *arguments: str) -> str:
    command = [sys.executable, str(ENTRYPOINT), *arguments]
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY,
            stdout=stream,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    assert completed.returncode == 0, completed.stderr
    text = output.read_text(encoding="utf-8")
    paths = [
        Path(arguments[index + 1])
        for index, value in enumerate(arguments)
        if value in {"--json-result", "--resolved-config"}
    ]
    original = {path: path.read_bytes() for path in paths}
    duplicate = subprocess.run(
        command,
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert duplicate.returncode != 0
    assert "already exists" in duplicate.stderr
    assert {path: path.read_bytes() for path in paths} == original
    return text


def _run_baseline(
    tmp_path: Path, name: str, interval: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = tmp_path / f"{name}.json"
    config_path = tmp_path / f"{name}-config.json"
    output = _run_toy_cli(
        tmp_path / f"{name}.txt",
        "--samples",
        "80",
        "--epochs",
        "2",
        "--seed",
        "13",
        "--hidden-dim",
        "4",
        "--noise",
        "0.8",
        "--sleep-interval",
        str(interval),
        "--split-threshold",
        "0",
        "--json-result",
        str(result_path),
        "--resolved-config",
        str(config_path),
    )
    assert "Backprop test accuracy:" in output
    assert "Circadian sleep:" in output
    result = _read_json(result_path)
    config = _read_json(config_path)
    assert result["resolved_config"] == config
    assert config["mode"] == "baseline"
    assert config["trial_configs"] == [config["config"]]
    assert config["seeds"] == [13]
    assert config["noise_levels"] == [0.8]
    assert result["protocol_id"] == "toy_validation_v1"
    assert set(result["split_hashes"]) == {"train", "validation", "test"}
    return result, config


def test_should_write_forced_and_no_sleep_toy_baseline_artifacts(tmp_path: Path) -> None:
    forced, forced_config = _run_baseline(tmp_path, "forced", 1)
    control, control_config = _run_baseline(tmp_path, "no-sleep", 0)
    assert forced["split_hashes"] == control["split_hashes"]
    assert forced["backprop"] == control["backprop"]
    assert forced["predictive_coding"] == control["predictive_coding"]
    assert forced_config["config"]["circadian_sleep_interval"] == 1
    assert control_config["config"]["circadian_sleep_interval"] == 0
    forced_events = forced["circadian_sleep"]["events"]
    control_events = control["circadian_sleep"]["events"]
    assert [event["completed_epoch"] for event in forced_events] == [1, 2]
    assert [event["completed_epoch"] for event in control_events] == [1, 2]
    assert all(event["trigger_reason"] == "periodic" for event in forced_events)
    assert any(event["outcome"] == "applied" for event in forced_events)
    assert all(event["outcome"] == "skipped" for event in control_events)


def test_should_write_actual_indepth_grid_text_and_config(tmp_path: Path) -> None:
    config_path = tmp_path / "indepth-config.json"
    output = _run_toy_cli(
        tmp_path / "indepth.txt",
        "--mode",
        "indepth",
        "--samples",
        "80",
        "--epochs",
        "2",
        "--seed",
        "13",
        "--seed-list",
        "13,17",
        "--noise-levels",
        "0.7,0.9",
        "--hidden-dim",
        "4",
        "--sleep-interval",
        "1",
        "--split-threshold",
        "0",
        "--resolved-config",
        str(config_path),
    )
    config = _read_json(config_path)
    assert config["schema_id"] == "toy_resolved_config_v1"
    assert config["mode"] == "indepth"
    assert config["seeds"] == [13, 17]
    assert config["noise_levels"] == [0.7, 0.9]
    assert [(trial["noise_scale"], trial["random_seed"]) for trial in config["trial_configs"]] == [
        (0.7, 13),
        (0.7, 17),
        (0.9, 13),
        (0.9, 17),
    ]
    assert output.startswith("In-Depth Model Comparison\n")
    assert output.count("Scenario noise=") == 2
    assert output.count("Split hashes by seed:") == 2
    assert "Scenario noise=0.70, runs=2" in output
    assert "Scenario noise=0.90, runs=2" in output
    assert "Seeds: [13, 17]" in output


def test_should_keep_explicit_legacy_toy_artifacts_separate(tmp_path: Path) -> None:
    result_path = tmp_path / "legacy.json"
    config_path = tmp_path / "legacy-config.json"
    output = _run_toy_cli(
        tmp_path / "legacy.txt",
        "--protocol-id",
        "toy_legacy_train_test_v0",
        "--samples",
        "80",
        "--epochs",
        "2",
        "--seed",
        "13",
        "--hidden-dim",
        "4",
        "--sleep-interval",
        "0",
        "--json-result",
        str(result_path),
        "--resolved-config",
        str(config_path),
    )
    result = _read_json(result_path)
    config = _read_json(config_path)
    assert result["protocol_id"] == config["config"]["protocol_id"] == "toy_legacy_train_test_v0"
    assert result["resolved_config"] == config
    assert result["split_hashes"] == {}
    assert "Protocol: toy_legacy_train_test_v0" in output
    for method in ("backprop", "predictive_coding", "circadian_predictive_coding"):
        assert result[method]["validation_accuracy"] is None
    assert len(result["circadian_sleep"]["events"]) == 2
    assert all(event["outcome"] == "skipped" for event in result["circadian_sleep"]["events"])


def test_should_write_adaptive_toy_decisions_without_forcing_sleep(tmp_path: Path) -> None:
    result_path = tmp_path / "adaptive.json"
    config_path = tmp_path / "adaptive-config.json"
    output = _run_toy_cli(
        tmp_path / "adaptive.txt",
        "--samples",
        "80",
        "--epochs",
        "2",
        "--seed",
        "13",
        "--hidden-dim",
        "4",
        "--noise",
        "0.8",
        "--sleep-interval",
        "1",
        "--adaptive-sleep-trigger",
        "--respect-adaptive-sleep-trigger",
        "--json-result",
        str(result_path),
        "--resolved-config",
        str(config_path),
    )
    result = _read_json(result_path)
    config = _read_json(config_path)
    assert "Circadian sleep:" in output
    assert result["resolved_config"] == config
    assert config["config"]["circadian_force_sleep"] is False
    assert config["config"]["circadian_config"]["use_adaptive_sleep_trigger"] is True
    events = result["circadian_sleep"]["events"]
    assert [event["completed_epoch"] for event in events] == [1, 2]
    assert all(event["trigger_reason"] == "periodic" for event in events)
    assert all(
        (event["outcome"], event["reason"]) == ("skipped", "adaptive_not_due") for event in events
    )
