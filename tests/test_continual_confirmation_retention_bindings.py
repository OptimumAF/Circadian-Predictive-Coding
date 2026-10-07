"""IO metadata spies grant no actual original-reader or scientific authority."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.app.continual_confirmation_execution import digest_json
from src.app.continual_confirmation_json import same_json
from src.infra import continual_confirmation_retention_bindings as bindings
from src.infra.continual_confirmation_io import read_json, verify_source_files, write_exclusive
from src.infra.continual_confirmation_resource_artifacts import artifact_paths as inventory_paths
from src.infra.continual_confirmation_retention_artifacts import artifact_paths
from src.infra.continual_confirmation_training_references import stream_file_identity


def replace_json(path: Path, body: Any) -> None:
    path.unlink()
    write_exclusive(path, body)


@pytest.fixture
def boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    state = SimpleNamespace(
        root=tmp_path, directory=tmp_path / "ledger", scope=tmp_path / "scope.json", reads=0
    )
    state.scope.write_bytes(b"fixture only")
    base = {}
    for index in range(114):
        name = f"base/{index}.py"
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"original IO fixture source")
        base[name] = stream_file_identity(path)["sha256"]
    monkeypatch.setattr(
        bindings, "current_resource_sources", lambda root: verify_source_files(root, base)
    )
    monkeypatch.setattr(bindings, "INVENTORY_SOURCE_MAP_SHA256", digest_json(base))
    apps = {}
    for name in (*bindings.APP_SOURCES, *bindings.OWN_SOURCES):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"new IO fixture source")
        if name in bindings.APP_SOURCES:
            apps[name] = stream_file_identity(path)["sha256"]
    monkeypatch.setattr(bindings, "APP_SOURCES", apps)
    state.input_file = tmp_path / "original-input.bin"
    state.input_file.write_bytes(b"original fixture input")
    inputs = {"original-input.bin": stream_file_identity(state.input_file)}
    state.originals = []
    files = {}
    for directory in (
        "artifacts/runs/p610-resource-inventory",
        "artifacts/runs/p610-resource-inventory-repeat",
    ):
        paths = inventory_paths(tmp_path / directory)
        paths["request"].parent.mkdir(parents=True)
        write_exclusive(paths["request"], {"source_sha256": base, "inputs": inputs})
        write_exclusive(paths["result"], {"fixture_inventory": True})
        write_exclusive(paths["audit"], {"fixture_audit": True})
        paths["markdown"].write_bytes(b"fixture Markdown\n")
        files[str(paths["request"].parent.resolve())] = {
            name: stream_file_identity(paths[name])
            for name in ("request", "result", "markdown", "audit")
        }
        state.originals.append(paths)
    monkeypatch.setattr(
        bindings, "INVENTORY_ID", stream_file_identity(state.originals[0]["result"])
    )
    handoff_path = tmp_path / bindings.HANDOFF_FILE
    write_exclusive(handoff_path, {"source_sha256": base, "inventory_files": files})
    monkeypatch.setattr(bindings, "HANDOFF_ID", stream_file_identity(handoff_path))

    def checked_inventory(root: Path, paths: dict[str, Path], scope: Path) -> Any:
        for name, identity in inputs.items():
            same_json(stream_file_identity(root / name), identity, "fixture upstream input bytes")
        return read_json(paths["request"])

    def references(*args: Any) -> Any:
        state.reads += 1
        return {
            "projection": {"coverage": {"fixture_only": True}},
            "training_references": {"fixture_only": True},
        }

    monkeypatch.setattr(bindings, "checked_resource_request", checked_inventory)
    monkeypatch.setattr(bindings, "read_retention_references", references)
    state.references = references
    state.directory.mkdir()
    state.paths = artifact_paths(state.directory)
    state.request = bindings.retention_request(
        tmp_path, state.directory, state.scope, "2026-10-01T22:00:00+00:00"
    )
    write_exclusive(state.paths["request"], state.request)
    return state


def test_should_bind_all_121_sources_and_complete_reference_inputs(boundary: Any) -> None:
    state = boundary
    result = bindings.read_retention_inputs(
        state.root, state.paths, state.scope, lambda path: ({}, {}, {})
    )
    assert len(state.request["source_sha256"]) == 121 and state.reads == 1
    assert result["provenance"]["complete_original_training_readers"] == 2
    assert result["provenance"]["source_sha256"] == state.request["source_sha256"]
    assert state.request["validation_budget_seconds"] == 180
    assert state.request["new_measurement_training_or_final_access_authorized"] is False


@pytest.mark.parametrize(
    "change", ["source", "budget", "scope", "inputs", "environment", "command", "time"]
)
def test_should_reject_every_changed_request_contract_before_reading(
    boundary: Any, change: str
) -> None:
    state = boundary
    request = deepcopy(state.request)
    if change == "source":
        request["source_sha256"].pop(next(iter(request["source_sha256"])))
    elif change == "budget":
        request["validation_budget_seconds"] += 1
    elif change == "scope":
        request["scope"]["cells"] -= 1
    elif change == "inputs":
        request["inputs"] = {}
    elif change == "environment":
        request["environment"]["numpy_version"] = "different"
    elif change == "command":
        request["command"][-1] = "different"
    else:
        request["started_utc"] = "2026-10-01"
    replace_json(state.paths["request"], request)
    with pytest.raises(ValueError):
        bindings.read_retention_inputs(
            state.root, state.paths, state.scope, lambda path: ({}, {}, {})
        )
    assert state.reads == 0


@pytest.mark.parametrize(
    "change",
    [
        "old_source",
        "new_app",
        "new_infra",
        "input",
        "inventory_bytes",
        "inventory_claim",
        "inventory_failure",
        "handoff",
    ],
)
def test_should_refuse_current_source_input_or_completion_drift(boundary: Any, change: str) -> None:
    state = boundary
    if change in {"old_source", "new_app", "new_infra"}:
        name = (
            "base/0.py"
            if change == "old_source"
            else next(iter(bindings.APP_SOURCES))
            if change == "new_app"
            else bindings.OWN_SOURCES[0]
        )
        (state.root / name).write_bytes(b"changed fixture source")
    elif change == "input":
        state.input_file.write_bytes(b"changed original input")
    elif change == "handoff":
        (state.root / bindings.HANDOFF_FILE).write_bytes(b"changed handoff")
    elif change == "inventory_bytes":
        state.originals[-1]["result"].write_bytes(b"changed inventory")
    else:
        state.originals[-1]["claim" if change == "inventory_claim" else "failure"].write_bytes(
            b"occupied or failed"
        )
    with pytest.raises(ValueError):
        bindings.read_retention_inputs(
            state.root, state.paths, state.scope, lambda path: ({}, {}, {})
        )
    assert state.reads == 0


@pytest.mark.parametrize("change", ["source", "input", "last_inventory_marker"])
def test_should_recheck_bindings_after_the_last_complete_reference(
    boundary: Any, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    state = boundary

    def late(*args: Any) -> Any:
        value = state.references(*args)
        if change == "source":
            (state.root / bindings.OWN_SOURCES[-1]).write_bytes(b"late source change")
        elif change == "input":
            state.input_file.write_bytes(b"late input change")
        else:
            state.originals[-1]["failure"].write_bytes(b"late failure")
        return value

    monkeypatch.setattr(bindings, "read_retention_references", late)
    with pytest.raises(ValueError):
        bindings.read_retention_inputs(
            state.root, state.paths, state.scope, lambda path: ({}, {}, {})
        )
    assert state.reads == 1
