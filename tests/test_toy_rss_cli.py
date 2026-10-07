"""The toy CLI records one measured RSS segment per budgeted attempt."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Sequence

import pytest

from src.adapters import cli
from src.app import experiment_runner
from src.shared.process_memory import ProcessRssSampler


def _arguments(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "toy",
        "--samples",
        "80",
        "--epochs",
        "1",
        "--sleep-interval",
        "0",
        "--run-state",
        str(tmp_path / "state.json"),
        "--checkpoint",
        str(tmp_path / "checkpoint.bin"),
        "--json-result",
        str(tmp_path / "result.json"),
        "--resolved-config",
        str(tmp_path / "config.json"),
        *extra,
    ]


def _state(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))


def _sampler(readings: Sequence[int | None]) -> ProcessRssSampler:
    remaining = iter(readings)
    last = readings[-1]

    def read() -> int | None:
        nonlocal last
        last = next(remaining, last)
        return last

    return ProcessRssSampler(interval_seconds=60.0, read_rss_bytes=read)


def test_should_record_fresh_rss_segment_on_checked_cli_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = iter(
        (
            [100, 100, 100, 150],
            [200, 200, 200, 200, 200, 200],
        )
    )
    monkeypatch.setattr(experiment_runner, "ProcessRssSampler", lambda: _sampler(next(attempts)))
    first = _arguments(tmp_path, "--max-process-rss-bytes", "120")
    monkeypatch.setattr(sys, "argv", first)

    with pytest.raises(SystemExit) as stopped:
        cli.main()

    assert stopped.value.code == 3
    state = _state(tmp_path)
    assert (state["status"], state["reason"]) == ("incomplete", "max_process_rss_bytes")
    assert state["budget"]["max_process_rss_bytes"] == 120
    assert state["work"]["training_updates_observed"] == 1
    assert state["checkpoint"]["position"]["stage"] == "wake"
    assert state["work"]["process_rss"]["scope"] == "absolute_current_process_per_invocation"
    assert state["work"]["process_rss"]["start_bytes"] == 100
    assert state["work"]["process_rss"]["peak_bytes"] == 150
    assert state["work"]["process_rss"]["sample_count"] >= 4
    assert not (tmp_path / "result.json").exists()

    monkeypatch.setattr(
        sys,
        "argv",
        _arguments(tmp_path, "--resume", "--max-process-rss-bytes", "200"),
    )
    cli.main()

    completed = _state(tmp_path)
    assert completed["status"] == "completed"
    assert completed["work"]["process_rss"]["start_bytes"] == 200
    assert completed["work"]["process_rss"]["peak_bytes"] == 200
    assert completed["work"]["process_rss"]["sample_count"] >= 6
    assert completed["attempts"][0]["work"]["process_rss"]["peak_bytes"] == 150
    assert completed["attempts"][1]["work"]["process_rss"]["peak_bytes"] == 200
    assert completed["checkpoint"]["training_updates"] == 3
    assert (tmp_path / "result.json").exists()


def test_should_record_unsupported_host_as_error_before_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiment_runner, "ProcessRssSampler", lambda: _sampler([None]))
    monkeypatch.setattr(
        experiment_runner,
        "generate_two_cluster_dataset",
        lambda **_kwargs: pytest.fail("unsupported RSS reached dataset construction"),
    )
    monkeypatch.setattr(sys, "argv", _arguments(tmp_path, "--max-process-rss-bytes", "100"))

    with pytest.raises(RuntimeError, match="process RSS is unavailable"):
        cli.main()

    state = _state(tmp_path)
    assert (state["status"], state["reason"], state["error_type"]) == (
        "error",
        "process_rss_unavailable",
        "ToyProcessRssUnavailable",
    )
    assert state["work"]["process_rss"] is None
    assert not (tmp_path / "result.json").exists()


def test_should_withhold_result_when_final_sampler_sample_exceeds_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        experiment_runner,
        "ProcessRssSampler",
        lambda: _sampler([100, 100, 100, 100, 100, 100, 100, 140]),
    )
    monkeypatch.setattr(sys, "argv", _arguments(tmp_path, "--max-process-rss-bytes", "120"))

    with pytest.raises(SystemExit) as stopped:
        cli.main()

    assert stopped.value.code == 3
    state = _state(tmp_path)
    assert (state["status"], state["reason"]) == ("incomplete", "max_process_rss_bytes")
    assert state["checkpoint"]["position"]["stage"] == "after_sleep"
    assert state["work"]["process_rss"]["peak_bytes"] == 140
    assert not (tmp_path / "result.json").exists()


def test_should_reject_invalid_rss_limit_before_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", _arguments(tmp_path, "--max-process-rss-bytes", "0"))
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 2
    assert list(tmp_path.iterdir()) == []
