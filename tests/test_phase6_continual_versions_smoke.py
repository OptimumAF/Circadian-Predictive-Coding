"""Bounded CLI artifact checks for the five pre-global-seal continual protocols.

These fixed cases test branch and output function, not comparative scores.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from src.app.continual_shift_benchmark import (
    CONTINUAL_BOUNDED_REPLAY_PROTOCOL,
    CONTINUAL_LEGACY_PROTOCOL,
    CONTINUAL_PHASE_ARRIVAL_PROTOCOL,
    CONTINUAL_PHASE_LOCAL_SCHEDULE_PROTOCOL,
    CONTINUAL_VALIDATION_PROTOCOL,
)


REPOSITORY = Path(__file__).resolve().parents[1]
PROTOCOLS = (
    CONTINUAL_LEGACY_PROTOCOL,
    CONTINUAL_VALIDATION_PROTOCOL,
    CONTINUAL_PHASE_ARRIVAL_PROTOCOL,
    CONTINUAL_PHASE_LOCAL_SCHEDULE_PROTOCOL,
    CONTINUAL_BOUNDED_REPLAY_PROTOCOL,
)
SPLIT_ROLES = {
    "phase_a_train",
    "phase_a_validation",
    "phase_a_test",
    "phase_b_train",
    "phase_b_validation",
    "phase_b_test",
}


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"nonfinite continual artifact value: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    assert isinstance(record, dict)
    return record


def _protocol_command(
    protocol: str, interval: int, report_path: Path, result_path: Path, config_path: Path
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.run_continual_shift_benchmark",
        "--protocol-id",
        protocol,
        "--profile",
        "strength-case",
        "--seeds",
        "17",
        "--sample-count-phase-a",
        "40",
        "--sample-count-phase-b",
        "40",
        "--phase-b-train-fraction",
        "0.5",
        "--phase-a-epochs",
        "2",
        "--phase-b-epochs",
        "2",
        "--hidden-dim",
        "4",
        "--phase-a-noise-scale",
        "0.8",
        "--phase-b-noise-scale",
        "0.8",
        "--phase-b-rotation-degrees",
        "0",
        "--phase-b-translation-x",
        "0",
        "--phase-b-translation-y",
        "0",
        "--sleep-interval-phase-a",
        str(interval),
        "--sleep-interval-phase-b",
        str(interval),
        "--output-file",
        str(report_path),
        "--json-result",
        str(result_path),
        "--resolved-config",
        str(config_path),
    ]
    if protocol == CONTINUAL_BOUNDED_REPLAY_PROTOCOL:
        command.extend(
            ["--replay-max-examples", "4", "--replay-max-bytes", "96", "--sleep-mode", "components"]
        )
    return command


def _run_protocol(
    directory: Path, protocol: str, interval: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    name = "forced" if interval else "no-sleep"
    report_path = directory / f"{name}.txt"
    result_path = directory / f"{name}.json"
    config_path = directory / f"{name}-config.json"
    command = _protocol_command(protocol, interval, report_path, result_path, config_path)
    completed = subprocess.run(
        command, cwd=REPOSITORY, capture_output=True, text=True, timeout=30, check=False
    )
    assert completed.returncode == 0, completed.stderr
    report = report_path.read_text(encoding="utf-8")
    assert report == completed.stdout
    assert report.startswith("Continual Shift Benchmark\n")
    assert f"Protocol: {protocol}" in report
    assert "Phase B uses configured noise and transform." in report
    result = _read_json(result_path)
    resolved = _read_json(config_path)
    assert result["config"] == resolved["config"]
    assert resolved["schema_id"] == "continual_resolved_config_v1"
    assert resolved["seeds"] == result["seeds"] == [17]
    assert result["aggregate"]["run_count"] == len(result["seed_results"]) == 1
    assert result["config"]["protocol_id"] == protocol
    original = {path: path.read_bytes() for path in (report_path, result_path, config_path)}
    duplicate = subprocess.run(
        command, cwd=REPOSITORY, capture_output=True, text=True, timeout=30, check=False
    )
    assert duplicate.returncode != 0
    assert "already exists" in duplicate.stderr
    assert {path: path.read_bytes() for path in original} == original
    return result, resolved


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_should_publish_forced_and_disabled_artifacts_for_each_continual_version(
    tmp_path: Path, protocol: str
) -> None:
    directory = tmp_path / protocol
    directory.mkdir()
    forced, forced_config = _run_protocol(directory, protocol, 1)
    control, control_config = _run_protocol(directory, protocol, 0)
    assert forced_config["config"]["circadian_sleep_interval_phase_a"] == 1
    assert control_config["config"]["circadian_sleep_interval_phase_a"] == 0
    forced_seed = forced["seed_results"][0]
    control_seed = control["seed_results"][0]
    assert forced_seed["seed"] == control_seed["seed"] == 17
    expected_roles = set() if protocol == CONTINUAL_LEGACY_PROTOCOL else SPLIT_ROLES
    assert set(forced_seed["split_hashes"]) == expected_roles
    assert forced_seed["split_hashes"] == control_seed["split_hashes"]
    assert forced_seed["backprop"] == control_seed["backprop"]
    assert forced_seed["predictive_coding"] == control_seed["predictive_coding"]
    forced_sleep = forced_seed["circadian_predictive_coding"]
    control_sleep = control_seed["circadian_predictive_coding"]
    assert [event["completed_epoch"] for event in forced_sleep["sleep_events"]] == [1, 2, 3, 4]
    assert [event["completed_epoch"] for event in control_sleep["sleep_events"]] == [1, 2, 3, 4]
    assert all(event["trigger_reason"] == "periodic" for event in forced_sleep["sleep_events"])
    # Why this: historical schedules can exhaust a structural budget even
    # though a periodic decision was made and earlier sleep events ran.
    assert any(event["outcome"] == "applied" for event in forced_sleep["sleep_events"])
    assert all(
        (event["outcome"], event["reason"])
        in {("applied", "core_executed"), ("skipped", "zero_structural_budget")}
        for event in forced_sleep["sleep_events"]
    )
    assert all(event["outcome"] == "skipped" for event in control_sleep["sleep_events"])
    if protocol == CONTINUAL_BOUNDED_REPLAY_PROTOCOL:
        for seed in (forced_seed, control_seed):
            retained = seed["replay_retention"]
            assert retained["budget_examples"] == 4
            assert retained["budget_bytes"] == 96
            for phase in ("phase_a", "phase_b"):
                assert retained[phase]["example_count"] <= 4
                assert retained[phase]["retained_bytes"] <= 96
    else:
        assert "replay_retention" not in forced_seed
        assert "replay_retention" not in control_seed
