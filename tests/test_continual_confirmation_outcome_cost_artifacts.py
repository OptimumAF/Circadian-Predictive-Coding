"""Small exclusive IO spies grant no actual scientific or original-reader authority."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.infra import continual_confirmation_outcome_cost_artifacts as artifacts
from src.infra.continual_confirmation_io import read_json, write_exclusive


@pytest.fixture
def boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    state = SimpleNamespace(
        root=tmp_path,
        directory=tmp_path / "presentation",
        scope=tmp_path / "scope",
        reads=0,
        checks=0,
    )
    state.paths = artifacts.artifact_paths(state.directory)
    state.reader = lambda: ({}, {}, {})
    state.request = {
        "fixture_only": True,
        "protocol_id": "fixture",
        "source_map_sha256": "a" * 64,
        "inputs": {"fixture_only": True},
    }
    state.body = {
        "fixture_only": True,
        "coverage": {"fixture_only": True},
        "stage_storage_totals": {"fixture_only": True},
        "provenance": {"complete_original_report_parts": {"fixture_only": True}},
    }

    def checked(*args: Any) -> Any:
        state.checks += 1
        return read_json(state.paths["request"])

    def body(root: Path, paths: Any, scope: Any, reader: Any, snapshot: Any) -> Any:
        assert reader is state.reader and snapshot == state.request
        state.reads += 1
        return deepcopy(state.body)

    state.check = checked
    monkeypatch.setattr(artifacts, "outcome_cost_request", lambda *args: deepcopy(state.request))
    monkeypatch.setattr(artifacts, "checked_outcome_cost_request", checked)
    monkeypatch.setattr(artifacts, "_read_bound_outcome_cost_inputs", body)
    monkeypatch.setattr(
        artifacts,
        "render_outcome_cost_presentation",
        lambda value: "# fixture\n" + canonical_body_identity(value)["sha256"] + "\n",
    )
    return state


def publish(state: Any) -> Any:
    return artifacts.publish_outcome_costs(state.root, state.directory, state.scope, state.reader)


def read(state: Any) -> Any:
    return artifacts.read_completed_outcome_costs(
        state.root,
        state.directory,
        state.scope,
        state.reader,
    )


def test_should_publish_and_independently_rebuild_every_part_without_writing(boundary: Any) -> None:
    state = boundary
    audit = publish(state)
    before = {name: path.read_bytes() for name, path in state.paths.items() if path.exists()}
    request, body, actual_audit = read(state)
    assert body == state.body and request == state.request and audit == actual_audit
    assert before == {
        name: path.read_bytes() for name, path in state.paths.items() if path.exists()
    }
    assert (
        state.reads == 2
        and not state.paths["failure"].exists()
        and not state.paths["claim"].exists()
    )
    assert audit["derivative_validation_elapsed_seconds"] < 240


@pytest.mark.parametrize("name", ["request", "result", "markdown", "audit", "failure", "claim"])
def test_should_preserve_occupied_outputs_before_any_reader(boundary: Any, name: str) -> None:
    state = boundary
    state.directory.mkdir()
    state.paths[name].write_bytes(b"user artifact\n")
    with pytest.raises(FileExistsError):
        publish(state)
    assert state.paths[name].read_bytes() == b"user artifact\n"
    assert state.reads == state.checks == 0


@pytest.mark.parametrize("name", ["request", "result", "markdown", "audit", "failure", "claim"])
def test_should_refuse_partial_failed_or_claimed_readback_before_readers(
    boundary: Any,
    name: str,
) -> None:
    state = boundary
    publish(state)
    if name in {"failure", "claim"}:
        state.paths[name].write_bytes(b"failed or occupied")
    else:
        state.paths[name].unlink()
    with pytest.raises(ValueError, match="complete"):
        read(state)
    assert state.reads == 1


@pytest.mark.parametrize("name", ["result", "markdown", "audit"])
@pytest.mark.parametrize("reseal", [False, True])
def test_should_refuse_changed_or_resealed_whole_parts(
    boundary: Any,
    name: str,
    reseal: bool,
) -> None:
    state = boundary
    publish(state)
    path = state.paths[name]
    if name == "markdown" or not reseal:
        with path.open("ab") as output:
            output.write(b" ")
    else:
        body = read_json(path)
        body["forged"] = True
        path.unlink()
        write_exclusive(path, body)
    with pytest.raises(ValueError):
        read(state)


def test_should_record_reader_failure_and_preserve_its_request(
    boundary: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = boundary

    def fail(*args: Any) -> Any:
        raise ValueError("complete reader failed")

    monkeypatch.setattr(artifacts, "_read_bound_outcome_cost_inputs", fail)
    with pytest.raises(ValueError, match="reader failed"):
        publish(state)
    assert state.paths["request"].is_file() and state.paths["failure"].is_file()
    assert not state.paths["claim"].exists()


@pytest.mark.parametrize("foreign_audit", [False, True])
def test_should_revoke_only_owned_audit_when_failure_publication_is_blocked(
    boundary: Any,
    monkeypatch: pytest.MonkeyPatch,
    foreign_audit: bool,
) -> None:
    state = boundary

    def late(*args: Any) -> Any:
        if state.paths["audit"].exists():
            if foreign_audit:
                state.paths["audit"].write_bytes(b"foreign audit")
            state.paths["failure"].write_bytes(b"foreign failure")
            raise ValueError("late binding failure")
        return state.check(*args)

    monkeypatch.setattr(artifacts, "checked_outcome_cost_request", late)
    with pytest.raises(FileExistsError):
        publish(state)
    assert state.paths["failure"].read_bytes() == b"foreign failure"
    assert not state.paths["claim"].exists()
    if foreign_audit:
        assert state.paths["audit"].read_bytes() == b"foreign audit"
    else:
        assert not state.paths["audit"].exists()


@pytest.mark.parametrize("change", ["failure", "claim", "foreign_claim", "bytes", "budget"])
def test_should_reject_late_markers_bytes_or_budget_after_completed_audit(
    boundary: Any,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    state = boundary

    def late(*args: Any) -> Any:
        if state.paths["audit"].exists():
            if change == "failure":
                state.paths["failure"].write_bytes(b"foreign marker")
            elif change == "claim":
                state.paths["claim"].unlink()
            elif change == "foreign_claim":
                state.paths["claim"].write_bytes(b"foreign claim")
            elif change == "bytes":
                state.paths["result"].write_bytes(b"late byte drift")
            else:
                monkeypatch.setattr(artifacts, "monotonic", lambda: 10**12)
        return state.check(*args)

    monkeypatch.setattr(artifacts, "checked_outcome_cost_request", late)
    with pytest.raises((ValueError, FileExistsError, FileNotFoundError)):
        publish(state)
    if change == "foreign_claim":
        assert state.paths["claim"].read_bytes() == b"foreign claim"
    else:
        assert not state.paths["claim"].exists()
    assert state.paths["failure"].exists()


@pytest.mark.parametrize("change", ["bytes", "failure", "claim", "budget", "request"])
def test_should_reject_late_readback_drift_without_writing_failure(
    boundary: Any,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    state = boundary
    publish(state)
    count = 0

    def late(*args: Any) -> Any:
        nonlocal count
        count += 1
        if count == 2:
            if change == "bytes":
                state.paths["markdown"].write_bytes(b"late readback byte drift")
            elif change in {"failure", "claim"}:
                state.paths[change].write_bytes(b"foreign marker")
            elif change == "request":
                value = read_json(state.paths["request"])
                value["forged"] = True
                state.paths["request"].unlink()
                write_exclusive(state.paths["request"], value)
            else:
                monkeypatch.setattr(artifacts, "monotonic", lambda: 10**12)
        return state.check(*args)

    monkeypatch.setattr(artifacts, "checked_outcome_cost_request", late)
    with pytest.raises(ValueError):
        read(state)
    if change != "failure":
        assert not state.paths["failure"].exists()


def test_should_preserve_foreign_request_created_after_claim(
    boundary: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = boundary

    def race(*args: Any) -> Any:
        state.paths["request"].write_bytes(b"foreign request")
        return deepcopy(state.request)

    monkeypatch.setattr(artifacts, "outcome_cost_request", race)
    with pytest.raises(FileExistsError):
        publish(state)
    assert state.paths["request"].read_bytes() == b"foreign request"
    assert not state.paths["failure"].exists() and not state.paths["audit"].exists()


@pytest.mark.parametrize("invalid", [None, -1, 240, True, float("inf")])
def test_should_refuse_resealed_invalid_recorded_elapsed_time(boundary: Any, invalid: Any) -> None:
    state = boundary
    publish(state)
    audit = read_json(state.paths["audit"])
    audit["derivative_validation_elapsed_seconds"] = invalid
    state.paths["audit"].unlink()
    if invalid == float("inf"):
        state.paths["audit"].write_text('{"invalid": Infinity}', encoding="utf-8")
    else:
        write_exclusive(state.paths["audit"], audit)
    with pytest.raises(ValueError):
        read(state)
