"""IO/source metadata spies have no original scientific readback authority."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import test_continual_confirmation_training_references as reference_tests
from src.app.continual_confirmation_execution import digest_json
from src.app.continual_confirmation_json import same_json
from src.infra import continual_confirmation_resource_bindings as bindings
from src.infra.continual_confirmation_io import read_json, write_exclusive
from src.infra.continual_confirmation_resource_artifacts import artifact_paths
from src.infra.continual_confirmation_report_artifacts import artifact_paths as report_paths
from src.infra.continual_confirmation_training_references import stream_file_identity

seal_reference_data = reference_tests.seal_reference_data


def _replace(path: Path, value: Any) -> None:
    path.unlink()
    write_exclusive(path, value)


@pytest.fixture
def boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    state = SimpleNamespace(
        root=tmp_path, directory=tmp_path / "inventory", scope=tmp_path / "scope.json", reads=0
    )
    state.scope.write_bytes(b"fixture scope")
    base = {}
    for i in range(106):
        name = f"base/{i}.py"
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"original fixture source")
        base[name] = stream_file_identity(path)["sha256"]
    monkeypatch.setattr(bindings, "current_report_sources", lambda root: deepcopy(base))
    monkeypatch.setattr(bindings, "REPORT_SOURCE_MAP_SHA256", digest_json(base))
    apps = {}
    for name in (*bindings.APP_SOURCES, *bindings.OWN_SOURCES):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"new fixture source")
        if name in bindings.APP_SOURCES:
            apps[name] = stream_file_identity(path)["sha256"]
    monkeypatch.setattr(bindings, "APP_SOURCES", apps)
    inputs = {}
    for repeat in ("", "-repeat"):
        names = [f"artifacts/runs/p611-confirmation-costs{repeat}.json"]
        names.extend(
            f"artifacts/runs/p67-confirmation-{kind}{repeat}/confirmation-{kind}.audit.json"
            for kind in ("train", "scored")
        )
        for name in names:
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            write_exclusive(path, {"fixture_only": True})
            inputs[name] = stream_file_identity(path)
    references = []
    for repeat in ("", "-repeat"):
        directory = f"artifacts/runs/p611-confirmation-report{repeat}"
        paths = report_paths(tmp_path / directory)
        paths["request"].parent.mkdir(parents=True)
        request = {
            "inputs": inputs,
            "scope_identity": stream_file_identity(state.scope),
            "scoring_manifest_sha256": "a" * 64,
            "analysis_contract_sha256": "b" * 64,
        }
        for name in ("request", "result", "audit"):
            write_exclusive(paths[name], request if name == "request" else {"fixture_only": True})
        paths["markdown"].write_bytes(b"fixture Markdown\n")
        references.append(
            (
                directory,
                {
                    name: stream_file_identity(paths[name])
                    for name in ("request", "result", "markdown", "audit")
                },
            )
        )
    monkeypatch.setattr(bindings, "REPORT_FILES", tuple(references))
    validations = {}
    for name in bindings.RECORDED_VALIDATIONS:
        path = tmp_path / name
        write_exclusive(path, {"fixture_only": True})
        validations[name] = stream_file_identity(path)
    monkeypatch.setattr(bindings, "RECORDED_VALIDATIONS", validations)

    def checked_original(root: Path, paths: dict[str, Path], scope: Path) -> dict[str, Any]:
        for name, identity in inputs.items():
            same_json(stream_file_identity(root / name), identity, "fixture original input drift")
        return read_json(paths["request"])

    def builder(*args: Any) -> dict[str, Any]:
        state.reads += 1
        return {"fixture_only": True, "coverage": {"cells": 560}}

    monkeypatch.setattr(bindings, "checked_report_request", checked_original)
    monkeypatch.setattr(bindings, "build_resource_inventory", builder)
    state.builder = builder
    state.directory.mkdir()
    state.paths = artifact_paths(state.directory)
    state.request = bindings.resource_request(
        tmp_path, state.directory, state.scope, "2026-10-01T20:00:00+00:00"
    )
    write_exclusive(state.paths["request"], state.request)
    return state


def test_should_bind_all_114_sources_and_rederive_both_complete_saved_inspections(
    boundary: Any,
) -> None:
    state = boundary
    result = bindings.read_resource_inventory_inputs(state.root, state.paths, state.scope)
    assert state.reads == 2 and len(state.request["source_sha256"]) == 114
    assert result["provenance"]["complete_saved_cost_inspections_read"] == 2
    assert result["provenance"]["inputs"] == state.request["inputs"]
    assert "not_new_scientific_readback" in result["provenance"]["validation_scope"]


@pytest.mark.parametrize(
    "name",
    [
        "source_sha256",
        "source_map_sha256",
        "inputs",
        "inventory",
        "validation_budget_seconds",
        "command",
        "environment",
        "started_utc",
        "new_measurement_training_or_final_access_authorized",
        "new_complete_scientific_readback_claimed",
        "unknown",
    ],
)
def test_should_refuse_changed_or_unknown_request_before_any_inventory_reads(
    boundary: Any, name: str
) -> None:
    state = boundary
    body = deepcopy(state.request)
    body[name] = True
    _replace(state.paths["request"], body)
    with pytest.raises(ValueError):
        bindings.read_resource_inventory_inputs(state.root, state.paths, state.scope)
    assert state.reads == 0


@pytest.mark.parametrize("name", [*bindings.APP_SOURCES, *bindings.OWN_SOURCES])
def test_should_refuse_current_source_drift_before_inventory_reads(
    boundary: Any, name: str
) -> None:
    state = boundary
    with (state.root / name).open("ab") as stream:
        stream.write(b"source drift")
    with pytest.raises(ValueError):
        bindings.read_resource_inventory_inputs(state.root, state.paths, state.scope)
    assert state.reads == 0


@pytest.mark.parametrize("name", ["request", "result", "markdown", "audit", "claim", "failure"])
def test_should_refuse_changed_original_report_part_or_marker_before_reads(
    boundary: Any, name: str
) -> None:
    state = boundary
    paths = report_paths(state.root / bindings.REPORT_FILES[-1][0])
    with paths[name].open("ab") as stream:
        stream.write(b"input drift")
    with pytest.raises(ValueError):
        bindings.read_resource_inventory_inputs(state.root, state.paths, state.scope)
    assert state.reads == 0


@pytest.mark.parametrize(
    "kind", ["source", "cost", "audit", "marker", "request", "recorded_validation"]
)
def test_should_recheck_late_source_input_marker_request_and_original_authority(
    boundary: Any, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    state = boundary

    def late(*args: Any) -> dict[str, Any]:
        paths = report_paths(state.root / bindings.REPORT_FILES[0][0])
        path = {
            "source": state.root / bindings.OWN_SOURCES[0],
            "cost": state.root / "artifacts/runs/p611-confirmation-costs.json",
            "audit": state.root
            / "artifacts/runs/p67-confirmation-train/confirmation-train.audit.json",
            "marker": paths["claim"],
            "request": state.paths["request"],
            "recorded_validation": state.root / next(iter(bindings.RECORDED_VALIDATIONS)),
        }[kind]
        with path.open("ab") as stream:
            stream.write(b"late drift")
        return state.builder(*args)

    monkeypatch.setattr(bindings, "build_resource_inventory", late)
    with pytest.raises(ValueError):
        bindings.read_resource_inventory_inputs(state.root, state.paths, state.scope)


def test_should_refuse_different_repeated_inventory_without_authority_claim(
    boundary: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = boundary

    def changed(*args: Any) -> dict[str, Any]:
        return {**state.builder(*args), "changed": state.reads}

    monkeypatch.setattr(bindings, "build_resource_inventory", changed)
    with pytest.raises(ValueError, match="repetition"):
        bindings.read_resource_inventory_inputs(state.root, state.paths, state.scope)


@pytest.mark.parametrize(
    "time", [None, "invalid", "2026-10-01T20:00:00", "2026-10-01T20:00:00-07:00"]
)
def test_should_refuse_malformed_or_non_utc_request_time(boundary: Any, time: Any) -> None:
    with pytest.raises(ValueError):
        bindings.resource_request(boundary.root, boundary.directory, boundary.scope, time)
