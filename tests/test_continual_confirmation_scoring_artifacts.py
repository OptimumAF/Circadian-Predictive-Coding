"""Exclusive real IO and public whole validators with a fabricated child.

Current-source/reference dispatch is an explicit IO spy in this module. The
actual complete readers/current source map have independent preflight evidence.
No subprocess trains, opens final data or supplies scientific/resource evidence.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from test_continual_confirmation_scoring_execution import (
    reference_report as reference_report,
    request_bindings as request_bindings,
    scored_request as scored_request,
    worker_payload as worker_payload,
    fabricated_scored_json as fabricated_scored_json,
)
from test_continual_confirmation_scoring_bindings import seal_file_boundary as seal_file_boundary
from src.app import continual_confirmation_scoring_execution as execution
from src.infra import continual_confirmation_scoring_artifacts as artifacts
from src.infra.continual_confirmation_io import read_json, write_exclusive
from src.infra.continual_confirmation_training_references import stream_file_identity


@pytest.fixture
def io_boundary(
    tmp_path: Path,
    request_bindings: dict[str, Any],
    reference_report: dict[str, Any],
    worker_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    state = SimpleNamespace(
        root=tmp_path,
        output=tmp_path / "scored",
        scope=tmp_path / "scope.json",
        report=deepcopy(reference_report),
        bindings=deepcopy(request_bindings),
        child=deepcopy(worker_payload),
        reader_calls=0,
        checks=0,
        calls=[],
    )
    state.paths = artifacts.artifact_paths(state.output)

    def reader() -> dict[str, Any]:
        state.reader_calls += 1
        return deepcopy(state.report)

    def check(root: Path, path: Path, scope: Path, expected: Any = None) -> dict[str, Any]:
        state.checks += 1
        body = read_json(path)
        if expected is not None:
            assert stream_file_identity(path) == expected
        execution.verify_scoring_execution_request(body, **state.bindings)
        return body

    def child(command: Any, **kwargs: Any) -> Any:
        state.calls.append((command, kwargs))
        assert state.reader_calls == 1 and state.paths["claim"].is_file()
        assert state.paths["request"].is_file() and not state.paths["result"].exists()
        state.child["request_sha256"] = stream_file_identity(state.paths["request"])["sha256"]
        return SimpleNamespace(
            returncode=0, stderr="", stdout=json.dumps(state.child, allow_nan=False)
        )

    monkeypatch.setattr(artifacts, "scoring_bindings", lambda *args: deepcopy(state.bindings))
    monkeypatch.setattr(artifacts, "checked_scoring_request", check)
    monkeypatch.setattr(artifacts.subprocess, "run", child)
    state.reader, state.check = reader, check
    return state


def _run(state: Any) -> dict[str, Any]:
    return artifacts.run_bounded_scoring(state.root, state.output, state.scope, state.reader)


def _read(state: Any) -> Any:
    return artifacts.read_completed_scored_bundle(
        state.root, state.output, state.scope, state.reader
    )


def _assert_failed(state: Any, reason: str = "worker_or_audit") -> None:
    assert not state.paths["claim"].exists()
    failure = read_json(state.paths["failure"])
    assert failure["status"] == "failed" and failure["reason"] == reason
    if state.paths["request"].is_file():
        assert failure["request_identity"] == stream_file_identity(state.paths["request"])
    with pytest.raises(ValueError, match="complete successful"):
        _read(state)


def test_should_save_request_before_child_and_link_every_complete_result_audit_readback(
    io_boundary: Any,
) -> None:
    state = io_boundary
    audit = _run(state)
    assert audit == read_json(state.paths["audit"])
    assert state.checks == 3 and state.reader_calls == len(state.calls) == 1
    assert state.calls[0] == (
        state.bindings["command"],
        {"cwd": state.root, "capture_output": True, "text": True, "timeout": 600, "check": False},
    )
    before = {name: path.read_bytes() for name, path in state.paths.items() if path.exists()}
    saved, result, checked = _read(state)
    assert checked == audit and result == state.child["result"]
    assert execution.encoded_identity(saved)["sha256"] == audit["request_sha256"]
    assert checked["final_observation"]["prediction_examples"] == 67200
    assert checked["work"]["totals"]["executed_optimizer_updates"] == 15210
    assert state.checks == 5 and state.reader_calls == 2
    assert not state.paths["failure"].exists() and not state.paths["claim"].exists()
    assert {
        name: path.read_bytes() for name, path in state.paths.items() if path.exists()
    } == before


@pytest.mark.parametrize("name", ["request", "result", "audit", "failure", "claim"])
def test_should_refuse_every_occupied_output_before_readers_and_preserve_foreign_bytes(
    io_boundary: Any, name: str
) -> None:
    state = io_boundary
    state.output.mkdir()
    state.paths[name].write_bytes(b"foreign artifact")
    with pytest.raises(FileExistsError):
        _run(state)
    assert state.paths[name].read_bytes() == b"foreign artifact"
    assert state.reader_calls == 0 and not state.calls
    assert [path.name for path in state.output.iterdir()] == [state.paths[name].name]


@pytest.mark.parametrize("name", ["request", "claim"])
def test_should_refuse_a_competing_writer_that_arrives_after_initial_preflight(
    io_boundary: Any, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    state = io_boundary
    original = artifacts.scoring_bindings

    def racing(*args: Any) -> Any:
        state.output.mkdir()
        state.paths[name].write_bytes(b"competing writer")
        return original(*args)

    monkeypatch.setattr(artifacts, "scoring_bindings", racing)
    with pytest.raises(FileExistsError):
        _run(state)
    assert state.paths[name].read_bytes() == b"competing writer"
    assert not state.paths["failure"].exists() and not state.calls
    assert state.paths["claim"].exists() is (name == "claim")


def test_should_preserve_noncooperative_request_collision_without_marking_it_failed(
    io_boundary: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = io_boundary
    original = artifacts.write_exclusive

    def collision(path: Path, value: Any) -> None:
        if path == state.paths["request"]:
            path.write_bytes(b"foreign request")
        original(path, value)

    monkeypatch.setattr(artifacts, "write_exclusive", collision)
    with pytest.raises(FileExistsError):
        _run(state)
    assert state.paths["request"].read_bytes() == b"foreign request"
    assert not state.paths["failure"].exists() and not state.paths["claim"].exists()


@pytest.mark.parametrize("stage", ["references", "bindings"])
def test_should_not_publish_request_or_launch_child_after_preflight_failure(
    io_boundary: Any, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    state = io_boundary

    def fail(*args: Any) -> Any:
        raise ValueError("fabricated complete reference/source failure")

    if stage == "references":
        state.reader = fail
    else:
        monkeypatch.setattr(artifacts, "scoring_bindings", fail)
    with pytest.raises(ValueError):
        _run(state)
    assert not state.output.exists() and not state.calls


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_should_record_current_request_source_reference_failure_before_or_after_worker_publication(
    io_boundary: Any, monkeypatch: pytest.MonkeyPatch, stage: int
) -> None:
    state = io_boundary

    def fail(*args: Any) -> Any:
        if state.checks + 1 == stage:
            raise ValueError("fabricated current source/request/reference drift")
        return state.check(*args)

    monkeypatch.setattr(artifacts, "checked_scoring_request", fail)
    with pytest.raises(ValueError):
        _run(state)
    assert len(state.calls) == (0 if stage == 1 else 1)
    _assert_failed(state)


@pytest.mark.parametrize(
    "kind", ["exit", "malformed", "duplicate", "nonfinite", "counter", "rss", "wall", "endpoint"]
)
def test_should_reject_failed_or_invalid_child_without_success_publication(
    io_boundary: Any, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    state = io_boundary
    original = artifacts.subprocess.run

    def child(*args: Any, **kwargs: Any) -> Any:
        process = original(*args, **kwargs)
        if kind == "exit":
            process.returncode, process.stderr = 7, "fabricated child failure"
        elif kind in {"malformed", "duplicate", "nonfinite"}:
            process.stdout = {
                "malformed": "{",
                "duplicate": '{"x":1,"x":2}',
                "nonfinite": '{"x":NaN}',
            }[kind]
        else:
            payload = json.loads(process.stdout)
            if kind == "counter":
                payload["observed_updates"]["executed_updates"] -= 1
            elif kind == "rss":
                payload["process_rss"]["peak_bytes"] = 536870913
            elif kind == "wall":
                payload["worker_elapsed_seconds"] = 600
            else:
                payload["result"]["evaluations"][-1]["result"]["correct_count"] = 100
            process.stdout = json.dumps(payload)
        return process

    monkeypatch.setattr(artifacts.subprocess, "run", child)
    with pytest.raises((ValueError, RuntimeError)):
        _run(state)
    _assert_failed(state)
    assert not state.paths["result"].exists() and not state.paths["audit"].exists()


@pytest.mark.parametrize("kind", ["timeout", "interrupt", "exit", "os_error"])
def test_should_record_worker_cancellation_or_timeout_and_release_owned_claim(
    io_boundary: Any, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    state = io_boundary
    error: BaseException = {
        "timeout": subprocess.TimeoutExpired("fixture child", 600),
        "interrupt": KeyboardInterrupt("fixture cancel"),
        "exit": SystemExit("fixture exit"),
        "os_error": OSError("fixture child cannot start"),
    }[kind]

    def child(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(artifacts.subprocess, "run", child)
    with pytest.raises(type(error)):
        _run(state)
    _assert_failed(
        state,
        "wall_limit"
        if kind == "timeout"
        else "canceled"
        if kind in {"interrupt", "exit"}
        else "worker_or_audit",
    )


@pytest.mark.parametrize("name", ["request", "result", "audit"])
@pytest.mark.parametrize("kind", ["error", "changed_bytes"])
def test_should_reject_publication_errors_or_changed_published_bytes(
    io_boundary: Any, monkeypatch: pytest.MonkeyPatch, name: str, kind: str
) -> None:
    state = io_boundary
    original = artifacts.write_exclusive

    def damaged(path: Path, value: Any) -> None:
        if path == state.paths[name] and kind == "error":
            raise OSError("fabricated publication error")
        original(path, value)
        if path == state.paths[name]:
            path.write_bytes(path.read_bytes() + b" ")

    monkeypatch.setattr(artifacts, "write_exclusive", damaged)
    with pytest.raises((ValueError, OSError)):
        _run(state)
    _assert_failed(state)


@pytest.mark.parametrize("name", ["result", "audit"])
def test_should_preserve_uncooperative_result_or_audit_collisions_and_mark_own_request_failed(
    io_boundary: Any, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    state = io_boundary
    original = artifacts.write_exclusive

    def collision(path: Path, value: Any) -> None:
        if path == state.paths[name]:
            path.write_bytes(b"foreign publication")
        original(path, value)

    monkeypatch.setattr(artifacts, "write_exclusive", collision)
    with pytest.raises(FileExistsError):
        _run(state)
    assert state.paths[name].read_bytes() == b"foreign publication"
    _assert_failed(state)


@pytest.mark.parametrize("name", ["request", "result", "audit", "failure", "claim"])
def test_should_refuse_incomplete_failed_or_occupied_readback_before_reference_dispatch(
    io_boundary: Any, name: str
) -> None:
    state = io_boundary
    _run(state)
    if name in {"failure", "claim"}:
        state.paths[name].write_bytes(b"blocked readback")
    else:
        state.paths[name].unlink()
    with pytest.raises(ValueError, match="complete successful"):
        _read(state)
    assert state.reader_calls == 1


@pytest.mark.parametrize("name", ["request", "result", "audit"])
@pytest.mark.parametrize("kind", ["noncanonical", "semantic"])
def test_should_reject_changed_complete_scientific_or_audit_readback(
    io_boundary: Any, name: str, kind: str
) -> None:
    state = io_boundary
    _run(state)
    path = state.paths[name]
    if kind == "noncanonical":
        path.write_bytes(path.read_bytes() + b" ")
    else:
        body = read_json(path)
        body["unexpected"] = True
        path.unlink()
        write_exclusive(path, body)
    with pytest.raises((ValueError, AssertionError)):
        _read(state)


@pytest.mark.parametrize(
    "kind", ["report", "late_request", "late_result", "late_audit", "late_claim", "late_failure"]
)
def test_should_reject_complete_reference_drift_or_late_readback_file_and_marker_drift(
    io_boundary: Any, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    state = io_boundary
    _run(state)
    if kind == "report":
        state.report["unexpected"] = True
    else:

        def reader() -> dict[str, Any]:
            state.reader_calls += 1
            name = kind.removeprefix("late_")
            path = state.paths[name]
            path.write_bytes(path.read_bytes() + b" " if path.exists() else b"late marker")
            return deepcopy(state.report)

        state.reader = reader
    with pytest.raises((ValueError, AssertionError)):
        _read(state)


@pytest.mark.parametrize("kind", ["owned", "changed"])
def test_should_leave_no_readable_success_when_failure_marker_write_fails_after_audit_publication(
    io_boundary: Any, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    state = io_boundary
    original = artifacts.write_exclusive

    def fail_marker(path: Path, value: Any) -> None:
        if path == state.paths["failure"]:
            raise OSError("fabricated failure marker cannot be written")
        original(path, value)

    def late_error(*args: Any) -> Any:
        if state.checks + 1 == 3:
            if kind == "changed":
                state.paths["audit"].write_bytes(b"foreign changed audit")
            raise ValueError("fabricated late current binding failure")
        return state.check(*args)

    monkeypatch.setattr(artifacts, "write_exclusive", fail_marker)
    monkeypatch.setattr(artifacts, "checked_scoring_request", late_error)
    with pytest.raises(OSError, match="failure marker"):
        _run(state)
    monkeypatch.setattr(artifacts, "checked_scoring_request", state.check)
    assert state.paths["request"].is_file() and state.paths["result"].is_file()
    if kind == "changed":
        assert state.paths["audit"].read_bytes() == b"foreign changed audit"
    else:
        assert not state.paths["audit"].exists()
    with pytest.raises(ValueError):
        _read(state)
