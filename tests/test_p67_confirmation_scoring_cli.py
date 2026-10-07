"""Public CLI routing and worker failures; no reserved scientific execution."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from test_continual_confirmation_scoring_execution import (
    reference_report as reference_report,
    request_bindings as request_bindings,
    scored_request as scored_request,
)
from test_continual_confirmation_scoring_bindings import seal_file_boundary as seal_file_boundary
from scripts import run_p67_confirmation_scoring as adapter
from src.infra import continual_confirmation_scoring_worker as worker
from src.infra.continual_confirmation_io import parse_json
from src.shared.process_memory import ProcessRssSampler


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--execute", "--read-only"],
        ["--worker"],
        ["--execute", "--request-file", "saved"],
        ["--worker", "--execute", "--request-file", "saved"],
        ["--worker", "--read-only", "--request-file", "saved"],
        ["--execute", "--seed", "101"],
        ["--execute", "--max-optimizer-updates", "16001"],
        ["--execute", "--partial"],
    ],
)
def test_should_refuse_ambiguous_modes_and_every_scientific_or_budget_override_before_work(
    monkeypatch: pytest.MonkeyPatch, flags: list[str], tmp_path: Path
) -> None:
    def forbid(*args: Any) -> Any:
        raise AssertionError("invalid CLI reached a reader/worker")

    monkeypatch.setattr(sys, "argv", ["scored", *flags])
    for name in ("_references", "_worker", "run_bounded_scoring", "read_completed_scored_bundle"):
        monkeypatch.setattr(adapter, name, forbid)
    with pytest.raises(SystemExit) as error:
        adapter.main()
    assert error.value.code == 2 and not list(tmp_path.iterdir())


def test_should_show_help_without_readers_worker_or_files(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["scored", "--help"])
    with pytest.raises(SystemExit) as error:
        adapter.main()
    output = capsys.readouterr().out
    assert error.value.code == 0 and "--execute" in output and "--read-only" in output
    assert "--worker" not in output and "--request-file" not in output


@pytest.mark.parametrize("mode", ["--execute", "--read-only"])
def test_should_route_explicit_public_mode_through_its_complete_reference_reader(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], mode: str, tmp_path: Path
) -> None:
    scope, output = tmp_path / "scope", tmp_path / "output"
    calls: list[Any] = []
    audit = {
        "status": "completed",
        "request_sha256": "1" * 64,
        "result_sha256": "2" * 64,
        "work": {"totals": {"executed_optimizer_updates": 15210}},
        "final_observation": {"prediction_attempts": 1680, "prediction_examples": 67200},
        "process_rss": {"peak_bytes": 100},
        "worker_elapsed_seconds": 1.0,
        "elapsed_seconds": 2.0,
    }
    monkeypatch.setattr(
        sys, "argv", ["scored", mode, "--scope-file", str(scope), "--output-dir", str(output)]
    )

    def reader_spy(path: Path) -> Any:
        calls.append(("reader", path))
        return {"spy": True}

    monkeypatch.setattr(adapter, "_references", reader_spy)

    def boundary(root: Path, directory: Path, scope_file: Path, reader: Any) -> Any:
        calls.append((mode, root, directory, scope_file))
        assert reader() == {"spy": True}
        return audit if mode == "--execute" else ({}, {}, audit)

    selected = "run_bounded_scoring" if mode == "--execute" else "read_completed_scored_bundle"
    monkeypatch.setattr(adapter, selected, boundary)
    adapter.main()
    assert calls == [(mode, adapter.REPO_ROOT, output, scope), ("reader", scope)]
    assert json.loads(capsys.readouterr().out)["final_prediction_examples"] == 67200
    assert not output.exists()


def test_should_delegate_both_complete_training_readers_without_a_partial_or_alternate_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[Any] = []
    scope = tmp_path / "scope"
    directories = [tmp_path / "train", tmp_path / "repeat"]

    def references(root: Any, manifest: Any, reader: Any) -> Any:
        assert root == adapter.REPO_ROOT and len(manifest.training_bundles) == 2
        assert [reader(directory) for directory in directories] == ["complete", "complete"]
        return {"fixture": True}

    monkeypatch.setattr(adapter, "read_training_references", references)

    def complete_reader_spy(directory: Path, path: Path) -> Any:
        calls.append((directory, path))
        return "complete"

    monkeypatch.setattr(adapter.training_adapter, "read_completed_bundle", complete_reader_spy)
    assert adapter._references(scope) == {"fixture": True}
    assert calls == [(directory, scope) for directory in directories]


def test_should_frame_already_serialized_science_without_changing_result_bytes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    scientific = '{"fabricated": [1, 2]}'
    monkeypatch.setattr(
        adapter, "worker_parts", lambda *args: (scientific, {"request_sha256": "1" * 64})
    )
    adapter._worker(Path("request"), Path("scope"))
    output = capsys.readouterr().out
    assert scientific in output and parse_json(output)["result"] == {"fabricated": [1, 2]}


@pytest.mark.parametrize("error", [ValueError, OSError, MemoryError])
def test_should_return_structured_stderr_and_no_score_on_worker_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: type[Exception],
) -> None:
    def fail(*args: Any) -> Any:
        raise error("fabricated worker failure")

    monkeypatch.setattr(adapter, "worker_parts", fail)
    with pytest.raises(SystemExit) as stopped:
        adapter._worker(Path("request"), Path("scope"))
    output = capsys.readouterr()
    assert stopped.value.code == 1 and output.out == ""
    assert parse_json(output.err)["error_type"] == error.__name__


def _fake_sampler(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    samplers: list[Any] = []

    def sample(**kwargs: Any) -> Any:
        assert kwargs == {"interval_seconds": 0.005}
        sampler = ProcessRssSampler(read_rss_bytes=lambda: 100)
        samplers.append(sampler)
        return sampler

    monkeypatch.setattr(worker, "ProcessRssSampler", sample)
    return samplers


def test_should_stop_public_child_on_current_request_failure_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samplers = _fake_sampler(monkeypatch)

    def fail(*args: Any) -> Any:
        raise ValueError("fabricated current request drift")

    monkeypatch.setattr(worker, "checked_scoring_request", fail)
    with pytest.raises(ValueError, match="request drift"):
        worker.worker_parts(Path("root"), Path("request"), Path("scope"))
    assert samplers[0]._stop.is_set() and not samplers[0]._thread.is_alive()


def test_should_require_actual_observed_optimizer_work_before_any_final_call(
    scored_request: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    samplers = _fake_sampler(monkeypatch)
    monkeypatch.setattr(worker, "checked_scoring_request", lambda *args: deepcopy(scored_request))
    monkeypatch.setattr(
        worker, "stream_file_identity", lambda path: {"sha256": "1" * 64, "byte_count": 1}
    )
    monkeypatch.setattr(
        worker, "train_confirmation", lambda manifest: SimpleNamespace(fabricated=True)
    )

    def forbid(*args: Any) -> Any:
        raise AssertionError("zero observed work reached final scoring")

    monkeypatch.setattr(worker, "_score_held", forbid)
    with pytest.raises(ValueError, match="updates|optimizer"):
        worker.worker_parts(Path("root"), Path("request"), Path("scope"))
    assert samplers[0]._stop.is_set() and not samplers[0]._thread.is_alive()
