"""Bounded end-to-end smokes for corrected public experiment artifacts.

The fixtures establish runnable branches and complete outputs, not model rank.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts import run_continual_shift_benchmark as continual_cli
from scripts import run_sleep_trigger_comparison as trigger_cli
from src.app.continual_shift_benchmark import CONTINUAL_GLOBAL_SEAL_PROTOCOL
from src.infra.trigger_streams import TriggerPhaseSource


def _reject_nonfinite(token: str) -> object:
    raise ValueError(f"nonfinite artifact value: {token}")


def _read_artifact(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    assert isinstance(record, dict)
    return record


def _identity_source_arguments(
    sleep_interval: int, report_path: Path, result_path: Path, config_path: Path
) -> list[str]:
    return [
        "continual",
        "--protocol-id",
        CONTINUAL_GLOBAL_SEAL_PROTOCOL,
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
        str(sleep_interval),
        "--sleep-interval-phase-b",
        str(sleep_interval),
        "--replay-max-examples",
        "4",
        "--replay-max-bytes",
        "96",
        "--sleep-mode",
        "components",
        "--output-file",
        str(report_path),
        "--json-result",
        str(result_path),
        "--resolved-config",
        str(config_path),
    ]


def _run_identity_source_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    name: str,
    sleep_interval: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = tmp_path / f"{name}.txt"
    result_path = tmp_path / f"{name}.json"
    config_path = tmp_path / f"{name}-config.json"
    monkeypatch.setattr(
        sys,
        "argv",
        _identity_source_arguments(sleep_interval, report_path, result_path, config_path),
    )
    continual_cli.main()
    printed = capsys.readouterr().out
    assert report_path.read_text(encoding="utf-8") == printed
    result = _read_artifact(result_path)
    resolved = _read_artifact(config_path)
    assert result["config"] == resolved["config"]
    assert resolved["schema_id"] == "continual_resolved_config_v1"
    assert resolved["seeds"] == result["seeds"] == [17]
    assert resolved["config"]["protocol_id"] == CONTINUAL_GLOBAL_SEAL_PROTOCOL
    assert len(result["seed_results"]) == result["aggregate"]["run_count"] == 1
    assert "Phase A uses the base source; Phase B uses configured noise and transform." in printed
    assert "shifted/rotated distribution" not in printed
    with pytest.raises(FileExistsError):
        continual_cli.main()
    assert capsys.readouterr().out == ""
    return result, resolved


def test_should_publish_identity_source_forced_sleep_and_no_sleep_control_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    forced, forced_config = _run_identity_source_control(
        tmp_path, monkeypatch, capsys, name="forced", sleep_interval=1
    )
    control, control_config = _run_identity_source_control(
        tmp_path, monkeypatch, capsys, name="no-sleep", sleep_interval=0
    )
    for resolved in (forced_config, control_config):
        config = resolved["config"]
        assert config["phase_a_noise_scale"] == config["phase_b_noise_scale"] == 0.8
        assert config["phase_b_rotation_degrees"] == 0.0
        assert config["phase_b_translation_x"] == config["phase_b_translation_y"] == 0.0

    forced_seed = forced["seed_results"][0]
    control_seed = control["seed_results"][0]
    assert forced_seed["split_hashes"] == control_seed["split_hashes"]
    assert forced_seed["backprop"] == control_seed["backprop"]
    assert forced_seed["predictive_coding"] == control_seed["predictive_coding"]
    forced_sleep = forced_seed["circadian_predictive_coding"]
    control_sleep = control_seed["circadian_predictive_coding"]
    assert [event["completed_epoch"] for event in forced_sleep["sleep_events"]] == [1, 2, 3, 4]
    assert [event["completed_epoch"] for event in control_sleep["sleep_events"]] == [1, 2, 3, 4]
    assert forced_sleep["sleep_event_count"] == 4
    assert all(event["trigger_reason"] == "periodic" for event in forced_sleep["sleep_events"])
    assert all(event["outcome"] == "applied" for event in forced_sleep["sleep_events"])
    assert control_sleep["sleep_event_count"] == 0
    assert all(event["outcome"] == "skipped" for event in control_sleep["sleep_events"])
    for seed in (forced_seed, control_seed):
        retention = seed["replay_retention"]
        assert retention["budget_examples"] == 4
        assert retention["budget_bytes"] == 96
        for phase in ("phase_a", "phase_b"):
            assert retention[phase]["example_count"] <= 4
            assert retention[phase]["retained_bytes"] <= 96


def _run_trigger_artifact(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    train_only: bool,
) -> dict[str, Any]:
    arguments = ["trigger", "--result", str(path)]
    if train_only:
        arguments.append("--train-only")
    monkeypatch.setattr(sys, "argv", arguments)
    trigger_cli.main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["result"] == str(path)
    assert printed["sha256"] == sha256(path.read_bytes()).hexdigest()
    return _read_artifact(path)


def test_should_publish_v13_independent_stationary_and_forced_trigger_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    a_source = TriggerPhaseSource(41, "a", "stationary_noise")
    b_source = TriggerPhaseSource(41, "b", "stationary_noise")
    assert not (a_source.train_input == b_source.train_input).all()
    assert (a_source.train_target == b_source.train_target).all()

    train_path = tmp_path / "trigger-train-only.json"
    train = _run_trigger_artifact(train_path, monkeypatch, capsys, train_only=True)
    assert train["final_released"] is False
    assert "outcomes" not in train
    assert len(train["trials"]) == 12
    cells = {(trial["condition"], trial["seed"], trial["arm"]): trial for trial in train["trials"]}
    assert set(cells) == {
        (condition, seed, arm)
        for condition in ("stationary_noise", "axis_shift")
        for seed in (41, 43)
        for arm in ("periodic", "adaptive", "no_sleep")
    }
    for (_, _, arm), trial in cells.items():
        assert len(trial["decisions"]) == 32
        expected_events = 4 if arm == "periodic" else 0
        assert sum(decision["performed"] for decision in trial["decisions"]) == expected_events
        assert trial["total_work"]["sleep_events"] == expected_events

    scored_path = tmp_path / "trigger-scored.json"
    scored = _run_trigger_artifact(scored_path, monkeypatch, capsys, train_only=False)
    assert scored["protocol_id"] == train["protocol_id"]
    assert scored["manifest_digest"] == train["manifest_digest"]
    assert scored["manifest"] == train["manifest"]
    assert len(scored["outcomes"]) == 12
    final_hashes: dict[tuple[str, int], set[tuple[tuple[str, str], ...]]] = {}
    for outcome in scored["outcomes"]:
        facts = outcome["train_facts"]
        assert facts == cells[(facts["condition"], facts["seed"], facts["arm"])]
        assert set(outcome["final_role_hashes"]) == {"a", "b"}
        key = (facts["condition"], facts["seed"])
        final_hashes.setdefault(key, set()).add(tuple(sorted(outcome["final_role_hashes"].items())))
    assert len(final_hashes) == 4
    assert all(len(hashes) == 1 for hashes in final_hashes.values())

    monkeypatch.setattr(
        trigger_cli,
        "run_trigger_comparison",
        lambda manifest: pytest.fail("occupied path reached final scoring"),
    )
    with pytest.raises(FileExistsError):
        trigger_cli.main()
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(sys, "argv", ["trigger", "--result", str(train_path), "--train-only"])
    monkeypatch.setattr(
        trigger_cli,
        "train_unscored_trigger_study",
        lambda manifest: pytest.fail("occupied path reached train-only work"),
    )
    with pytest.raises(FileExistsError):
        trigger_cli.main()
    assert capsys.readouterr().out == ""
