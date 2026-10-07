"""Exclusive resource IO spies test ownership/failures; no scientific readback authority."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.infra import continual_confirmation_resource_artifacts as artifacts
from src.infra.continual_confirmation_io import read_json, write_exclusive


def _forbid(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("publication IO spy entered a scientific input reader")


@pytest.fixture
def io_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    state = SimpleNamespace(
        root=tmp_path,
        directory=tmp_path / "report",
        scope=tmp_path / "scope.json",
        checks=0,
        reads=0,
    )
    state.paths = artifacts.artifact_paths(state.directory)
    state.request = {
        "fixture_only": True,
        "protocol_id": "fixture",
        "source_map_sha256": "a" * 64,
        "inputs": {"fixture_only": True},
    }
    state.body = {
        "fixture_only": True,
        "coverage": {"fixture_only": True},
        "report_identity": {"fixture_only": True},
    }

    def checked(*args: Any) -> Any:
        state.checks += 1
        return read_json(state.paths["request"])

    def body(*args: Any) -> Any:
        state.reads += 1
        return deepcopy(state.body)

    state.check = checked
    state.read = body
    monkeypatch.setattr(artifacts, "resource_request", lambda *args: deepcopy(state.request))
    monkeypatch.setattr(artifacts, "checked_resource_request", checked)
    monkeypatch.setattr(artifacts, "read_resource_inventory_inputs", body)
    monkeypatch.setattr(
        artifacts,
        "render_resource_inventory",
        lambda body: "# fixture\n" + canonical_body_identity(body)["sha256"] + "\n",
    )
    return state


def _publish(state: Any) -> Any:
    return artifacts.publish_resource_inventory(state.root, state.directory, state.scope)


def _read(state: Any) -> Any:
    return artifacts.read_completed_resource_inventory(state.root, state.directory, state.scope)


def test_should_publish_all_files_exclusively_and_rebuild_full_body_markdown_and_audit(
    io_boundary: Any,
) -> None:
    state = io_boundary
    audit = _publish(state)
    assert not state.paths["claim"].exists() and not state.paths["failure"].exists()
    before = {name: path.read_bytes() for name, path in state.paths.items() if path.exists()}
    request, body, read_audit = _read(state)
    assert body == state.body and request == state.request and read_audit == audit
    assert state.reads == 2
    assert {
        name: path.read_bytes() for name, path in state.paths.items() if path.exists()
    } == before


@pytest.mark.parametrize("name", ["request", "result", "markdown", "audit", "failure", "claim"])
def test_should_preserve_every_occupied_output_before_entering_any_reader(
    io_boundary: Any, name: str
) -> None:
    state = io_boundary
    state.directory.mkdir()
    state.paths[name].write_bytes(b"user artifact\n")
    with pytest.raises(FileExistsError):
        _publish(state)
    assert state.paths[name].read_bytes() == b"user artifact\n"
    assert state.reads == state.checks == 0


@pytest.mark.parametrize("name", ["request", "result", "markdown", "audit", "failure", "claim"])
def test_should_reject_missing_parts_or_failure_claim_before_complete_readback(
    io_boundary: Any, name: str
) -> None:
    state = io_boundary
    _publish(state)
    before_reads = state.reads
    if name in {"failure", "claim"}:
        state.paths[name].write_bytes(b"failed or occupied")
    else:
        state.paths[name].unlink()
    with pytest.raises(ValueError, match="complete"):
        _read(state)
    assert state.reads == before_reads


@pytest.mark.parametrize("name", ["result", "markdown", "audit"])
@pytest.mark.parametrize("kind", ["noncanonical", "resealed"])
def test_should_reject_changed_unknown_or_resealed_report_fields(
    io_boundary: Any, name: str, kind: str
) -> None:
    state = io_boundary
    _publish(state)
    path = state.paths[name]
    if kind == "noncanonical" or name == "markdown":
        with path.open("ab") as output:
            output.write(b" ")
    else:
        body = read_json(path)
        body["undeclared"] = True
        path.unlink()
        write_exclusive(path, body)
    with pytest.raises(ValueError):
        _read(state)


def test_should_record_reader_failure_and_preserve_request_evidence(
    io_boundary: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = io_boundary

    def fail(*args: Any) -> Any:
        raise ValueError("complete input reader failed")

    monkeypatch.setattr(artifacts, "read_resource_inventory_inputs", fail)
    with pytest.raises(ValueError, match="reader failed"):
        _publish(state)
    assert state.paths["request"].is_file() and state.paths["failure"].is_file()
    assert not state.paths["claim"].exists()
    with pytest.raises(ValueError):
        _read(state)


@pytest.mark.parametrize("point", ["result", "markdown", "audit"])
def test_should_record_publication_failure_without_a_readable_completion(
    io_boundary: Any, monkeypatch: pytest.MonkeyPatch, point: str
) -> None:
    state = io_boundary
    original = artifacts.write_exclusive
    if point == "markdown":
        monkeypatch.setattr(
            artifacts,
            "render_resource_inventory",
            lambda *args: (_ for _ in ()).throw(OSError("Markdown publication failure")),
        )
    else:

        def fail(path: Path, value: Any) -> None:
            if path == state.paths[point]:
                raise OSError("publication failure")
            original(path, value)

        monkeypatch.setattr(artifacts, "write_exclusive", fail)
    with pytest.raises(OSError, match="publication failure"):
        _publish(state)
    assert state.paths["failure"].is_file() and not state.paths["claim"].exists()
    with pytest.raises(ValueError):
        _read(state)


@pytest.mark.parametrize("kind", ["owned", "foreign"])
def test_should_revoke_only_unchanged_owned_audit_if_failure_marker_write_fails(
    io_boundary: Any, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    state = io_boundary
    original = artifacts.write_exclusive

    def late(*args: Any) -> Any:
        if state.checks == 1:
            if kind == "foreign":
                state.paths["audit"].write_bytes(b"foreign changed audit")
            raise ValueError("late binding failure")
        return state.check(*args)

    def fail(path: Path, value: Any) -> None:
        if path == state.paths["failure"]:
            raise OSError("failure marker cannot be written")
        original(path, value)

    monkeypatch.setattr(artifacts, "checked_resource_request", late)
    monkeypatch.setattr(artifacts, "write_exclusive", fail)
    with pytest.raises(OSError, match="failure marker"):
        _publish(state)
    if kind == "owned":
        assert not state.paths["audit"].exists()
    else:
        assert state.paths["audit"].read_bytes() == b"foreign changed audit"
    assert state.paths["request"].is_file() and state.paths["result"].is_file()
    with pytest.raises(ValueError):
        _read(state)


@pytest.mark.parametrize("name", ["request", "result", "markdown", "audit", "claim", "failure"])
def test_should_refuse_late_readback_byte_or_marker_changes(
    io_boundary: Any, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    state = io_boundary
    _publish(state)

    def late(*args: Any) -> Any:
        with state.paths[name].open("ab") as output:
            output.write(b"late drift")
        return deepcopy(state.body)

    monkeypatch.setattr(artifacts, "read_resource_inventory_inputs", late)
    with pytest.raises(ValueError):
        _read(state)


def test_should_refuse_exceeded_derivative_budget_without_changing_scientific_caps(
    io_boundary: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = io_boundary
    ticks = iter((0.0, 181.0, 182.0))
    monkeypatch.setattr(artifacts, "monotonic", lambda: next(ticks))
    with pytest.raises(ValueError, match="wall limit"):
        _publish(state)
    assert state.paths["failure"].is_file() and not state.paths["audit"].exists()


def test_should_preserve_foreign_request_collision_without_failing_its_owner(
    io_boundary: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = io_boundary

    def collision(*args: Any) -> Any:
        state.paths["request"].write_bytes(b"foreign request")
        return deepcopy(state.request)

    monkeypatch.setattr(artifacts, "resource_request", collision)
    with pytest.raises(FileExistsError):
        _publish(state)
    assert state.paths["request"].read_bytes() == b"foreign request"
    assert not state.paths["failure"].exists()
