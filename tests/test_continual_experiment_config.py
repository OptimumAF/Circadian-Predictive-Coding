"""Typed overrides for the existing configurable continual-shift route."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts import run_continual_shift_benchmark as cli
from src.app.continual_experiment_config import resolve_continual_overrides
from src.app.continual_shift_benchmark import ContinualShiftConfig


def test_should_preserve_preset_when_overrides_are_empty() -> None:
    base = ContinualShiftConfig()
    assert resolve_continual_overrides(base, {}) == base


@pytest.mark.parametrize(
    "preset,expected",
    [
        ("baseline", (500, 110, 80, 12)),
        ("strength-case", (500, 110, 80, 12)),
        ("hardest-case", (700, 120, 180, 24)),
    ],
)
def test_should_keep_existing_profile_defaults_without_overrides(
    preset: str,
    expected: tuple[int, int, int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[ContinualShiftConfig] = []

    def record_run(*, config: ContinualShiftConfig, seeds: list[int]) -> Any:
        assert seeds == [13]
        captured.append(config)
        return object()

    monkeypatch.setattr(sys, "argv", ["continual", "--profile", preset, "--seeds", "13"])
    monkeypatch.setattr(cli, "run_continual_shift_benchmark", record_run)
    monkeypatch.setattr(cli, "format_continual_shift_benchmark", lambda result: "summary")
    cli.main()
    config = captured[0]
    assert (
        config.sample_count_phase_a,
        config.phase_a_epochs,
        config.phase_b_epochs,
        config.hidden_dim,
    ) == expected
    assert config.protocol_id == ContinualShiftConfig().protocol_id


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"unknown_parameter": 1}, "unknown"),
        ({"protocol_id": "changed"}, "unknown"),
        ({"model_order": ["circadian_predictive_coding"]}, "unknown"),
        ({"backprop_learning_rate": 0.5}, "unknown"),
        ({"circadian_config": {}}, "unknown"),
        ({"phase_a_epochs": True}, "integer"),
        ({"phase_a_epochs": 2.0}, "integer"),
        ({"phase_b_noise_scale": False}, "finite number"),
        ({"phase_b_noise_scale": float("inf")}, "finite number"),
        ({"hidden_dims": [4, True]}, "integer"),
        ({"phase_a_epochs": 0}, "phase epochs"),
    ],
)
def test_should_reject_unknown_or_invalid_override_before_training(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_continual_overrides(ContinualShiftConfig(), overrides)


def test_should_apply_typed_existing_fields_without_changing_protocol() -> None:
    base = ContinualShiftConfig()
    resolved = resolve_continual_overrides(
        base,
        {"phase_a_epochs": 2, "hidden_dim": 4, "hidden_dims": [4, 4], "phase_b_noise_scale": 1},
    )
    assert base == ContinualShiftConfig()
    assert resolved.phase_a_epochs == 2
    assert resolved.hidden_dim == 4 and resolved.hidden_dims == (4, 4)
    assert resolved.phase_b_noise_scale == 1.0
    assert resolved.protocol_id == base.protocol_id
    assert resolved.model_order == base.model_order


def test_should_save_exact_resolved_config_used_for_small_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_path = tmp_path / "result.json"
    config_path = tmp_path / "resolved-config.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "continual",
            "--profile",
            "baseline",
            "--seeds",
            "13",
            "--sample-count-phase-a",
            "80",
            "--sample-count-phase-b",
            "80",
            "--phase-b-train-fraction",
            "0.5",
            "--phase-a-epochs",
            "5",
            "--override",
            "phase_a_epochs=2",
            "--override",
            "phase_b_epochs=2",
            "--override",
            "hidden_dim=4",
            "--override",
            "circadian_sleep_interval_phase_a=2",
            "--override",
            "circadian_sleep_interval_phase_b=1",
            "--json-result",
            str(result_path),
            "--resolved-config",
            str(config_path),
        ],
    )
    cli.main()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["schema_id"] == "continual_resolved_config_v1"
    assert saved["preset"] == "baseline"
    assert saved["seeds"] == [13]
    assert saved["config"] == result["config"]
    assert saved["config"]["phase_a_epochs"] == 2
    assert saved["config"]["phase_b_epochs"] == 2
    assert saved["config"]["circadian_sleep_interval_phase_a"] == 2
    assert saved["overrides"]["hidden_dim"] == 4
    assert saved["config"] != asdict(ContinualShiftConfig())


def test_should_refuse_occupied_resolved_config_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "resolved-config.json"
    path.write_text("previous user content", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["continual", "--resolved-config", str(path)])
    monkeypatch.setattr(
        cli,
        "run_continual_shift_benchmark",
        lambda **kwargs: pytest.fail("occupied config path reached training"),
    )
    with pytest.raises(FileExistsError, match="resolved config already exists"):
        cli.main()
    assert path.read_text(encoding="utf-8") == "previous user content"


def test_should_reject_unknown_and_duplicate_cli_overrides_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "run_continual_shift_benchmark",
        lambda **kwargs: pytest.fail("invalid override reached training"),
    )
    for overrides, message in (
        (["unknown_parameter=1"], "unknown"),
        (["phase_a_epochs=2", "phase_a_epochs=3"], "duplicate"),
        (["phase_a_epochs=NaN"], "JSON"),
    ):
        args = ["continual", "--json-result", str(tmp_path / "result.json")]
        for item in overrides:
            args.extend(("--override", item))
        monkeypatch.setattr(sys, "argv", args)
        with pytest.raises(SystemExit) as raised:
            cli.main()
        assert raised.value.code == 2
        assert message in capsys.readouterr().err
        assert not (tmp_path / "result.json").exists()


def test_should_require_saved_artifact_for_explicit_overrides(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["continual", "--override", "phase_a_epochs=2"])
    monkeypatch.setattr(
        cli,
        "run_continual_shift_benchmark",
        lambda **kwargs: pytest.fail("unrecorded override reached training"),
    )
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 2
    assert "--json-result or --resolved-config" in capsys.readouterr().err


def test_should_reject_nonfinite_legacy_flag_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "continual",
            "--phase-b-rotation-degrees",
            "NaN",
            "--resolved-config",
            str(tmp_path / "resolved.json"),
        ],
    )
    monkeypatch.setattr(
        cli,
        "run_continual_shift_benchmark",
        lambda **kwargs: pytest.fail("nonfinite flag reached training"),
    )
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 2
    assert "finite JSON" in capsys.readouterr().err
    assert not (tmp_path / "resolved.json").exists()
