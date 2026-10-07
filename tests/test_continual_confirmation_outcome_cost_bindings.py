"""Small IO/reader spies exercise bindings without original scientific authority."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.app.continual_confirmation_execution import digest_json
from src.app.continual_confirmation_json import same_json
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.infra import continual_confirmation_outcome_cost_bindings as bindings
from src.infra.continual_confirmation_io import read_json, verify_source_files, write_exclusive
from src.infra.continual_confirmation_outcome_cost_artifacts import artifact_paths
from src.infra.continual_confirmation_training_references import stream_file_identity


def replace_json(path: Path, body: Any) -> None:
    path.unlink()
    write_exclusive(path, body)


@pytest.fixture
def boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    state = SimpleNamespace(
        root=tmp_path,
        directory=tmp_path / "presentation",
        scope=tmp_path / "scope.json",
        reads=0,
        builds=0,
    )
    state.scope.write_bytes(b"fixture original scope")
    state.input_file = tmp_path / "original-input.bin"
    state.input_file.write_bytes(b"fixture original input")
    inputs = {"original-input.bin": stream_file_identity(state.input_file)}
    base = {}
    for index in range(121):
        name = f"base/{index}.py"
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"fixture old source")
        base[name] = stream_file_identity(path)["sha256"]
    monkeypatch.setattr(
        bindings, "current_retention_sources", lambda root: verify_source_files(root, base)
    )
    monkeypatch.setattr(bindings, "RETENTION_SOURCE_MAP_SHA256", digest_json(base))
    apps = {}
    for name in (*bindings.APP_SOURCES, *bindings.OWN_SOURCES):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture new source")
        if name in bindings.APP_SOURCES:
            apps[name] = stream_file_identity(path)["sha256"]
    monkeypatch.setattr(bindings, "APP_SOURCES", apps)
    pure_sources = {**base, **apps}
    monkeypatch.setattr(bindings, "PURE_SOURCE_MAP_SHA256", digest_json(pure_sources))
    state.originals, expected_requests, reports = {}, {}, []
    all_files: dict[str, Any] = {}
    bodies = {kind: {"fixture_kind": kind} for kind in ("report", "inventory", "retention")}
    input_files = {}
    for kind, prefix, builder in (
        ("report", "p611-confirmation-report", bindings.report_paths),
        ("inventory", "p610-resource-inventory", bindings.inventory_paths),
        ("retention", "p610-retention-costs", bindings.retention_paths),
    ):
        all_files[kind] = {}
        for suffix in ("", "-repeat"):
            directory = "artifacts/runs/" + prefix + suffix
            paths = builder(tmp_path / directory)
            paths["request"].parent.mkdir(parents=True)
            request = {
                "fixture_kind": kind,
                "source_sha256": base,
                "inputs": inputs,
                "scope_identity": stream_file_identity(state.scope),
            }
            write_exclusive(paths["request"], request)
            write_exclusive(paths["result"], bodies[kind])
            write_exclusive(paths["audit"], {"fixture_audit": kind})
            paths["markdown"].write_bytes(b"fixture Markdown\n")
            identities = {
                name: stream_file_identity(paths[name])
                for name in ("request", "result", "markdown", "audit")
            }
            all_files[kind][str(paths["request"].parent.resolve())] = identities
            expected_requests[kind + suffix] = request
            state.originals[kind + suffix] = paths
            if kind == "report":
                reports.append((directory, identities))
            if not suffix:
                input_files[kind] = str(paths["result"].relative_to(tmp_path))
    monkeypatch.setattr(bindings, "REPORT_FILES", tuple(reports))
    monkeypatch.setattr(bindings, "INPUT_FILES", input_files)
    monkeypatch.setattr(
        bindings,
        "INPUT_IDENTITIES",
        {kind: canonical_body_identity(body) for kind, body in bodies.items()},
    )
    state.body = {
        "fixture_only": True,
        "coverage": {"cells": 560},
        "stage_storage_totals": {"fixture_only": True},
    }
    monkeypatch.setattr(bindings, "PURE_RESULT_ID", canonical_body_identity(state.body))
    pure_files = {}
    for suffix in ("", "-repeat"):
        pure_directory = tmp_path / ("artifacts/runs/p610-outcome-cost-pure" + suffix)
        pure_directory.mkdir()
        write_exclusive(pure_directory / "outcome-costs.result.json", state.body)
        (pure_directory / "outcome-costs.md").write_bytes(b"fixture pure Markdown\n")
        pure_files[str(pure_directory.resolve())] = {
            "result": stream_file_identity(pure_directory / "outcome-costs.result.json"),
            "markdown": stream_file_identity(pure_directory / "outcome-costs.md"),
        }
    handoffs = deepcopy(bindings.HANDOFFS)
    for name, body in (
        ("retention", {"source_sha256": base, "retention_files": all_files["retention"]}),
        (
            "pure",
            {
                "source_sha256": pure_sources,
                "pure_output_files": pure_files,
                "current_complete_upstream_bindings": {
                    "previous_inventory_and_matrix_files": {
                        "inventory_files": all_files["inventory"]
                    },
                    "original_and_companion_requests": {
                        key: value
                        for key, value in expected_requests.items()
                        if not key.startswith("inventory")
                    },
                },
            },
        ),
    ):
        path = tmp_path / handoffs[name]["file"]
        write_exclusive(path, body)
        handoffs[name]["identity"] = stream_file_identity(path)
    monkeypatch.setattr(bindings, "HANDOFFS", handoffs)

    def checked(root: Path, paths: dict[str, Path], scope: Path) -> Any:
        for name, identity in inputs.items():
            same_json(stream_file_identity(root / name), identity, "fixture original input bytes")
        return read_json(paths["request"])

    for name in ("checked_report_request", "checked_resource_request", "checked_retention_request"):
        monkeypatch.setattr(bindings, name, checked)

    def build(report: Any, inventory: Any, retention: Any) -> Any:
        state.builds += 1
        for kind, body in (("report", report), ("inventory", inventory), ("retention", retention)):
            same_json(
                canonical_body_identity(body),
                bindings.INPUT_IDENTITIES[kind],
                "fixture whole " + kind,
            )
        return deepcopy(state.body)

    def reader() -> Any:
        state.reads += 1
        return tuple(
            read_json(state.originals["report"][name]) for name in ("request", "result", "audit")
        )

    monkeypatch.setattr(bindings, "build_outcome_cost_presentation", build)
    state.reader, state.build = reader, build
    state.directory.mkdir()
    state.paths = artifact_paths(state.directory)
    state.request = bindings.outcome_cost_request(
        tmp_path, state.directory, state.scope, "2026-10-01T22:00:00+00:00"
    )
    write_exclusive(state.paths["request"], state.request)
    return state


def read(state: Any, reader: Any = None) -> Any:
    return bindings.read_outcome_cost_inputs(
        state.root, state.paths, state.scope, reader or state.reader
    )


def test_should_bind_all_127_sources_six_complete_bundles_and_one_whole_reader(
    boundary: Any,
) -> None:
    state = boundary
    body = read(state)
    assert len(state.request["source_sha256"]) == 127
    assert len(state.request["inputs"]["current_upstream_requests"]) == 6
    assert state.reads == state.builds == 1
    assert body["provenance"]["complete_original_report_readers"] == 1
    assert body["provenance"]["inputs"] == state.request["inputs"]
    assert state.request["validation_budget_seconds"] == 240
    assert state.request["new_measurement_training_or_final_access_authorized"] is False


@pytest.mark.parametrize(
    "change",
    ["source", "budget", "scope", "inputs", "environment", "command", "time", "authorization"],
)
def test_should_reject_changed_request_contract_before_the_reader(
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
    elif change == "authorization":
        request["new_measurement_training_or_final_access_authorized"] = True
    else:
        request["started_utc"] = "2026-10-01"
    replace_json(state.paths["request"], request)
    with pytest.raises(ValueError):
        read(state)
    assert state.reads == state.builds == 0


@pytest.mark.parametrize(
    "kind",
    ["report", "report-repeat", "inventory", "inventory-repeat", "retention", "retention-repeat"],
)
@pytest.mark.parametrize("part", ["request", "result", "audit", "markdown", "claim", "failure"])
def test_should_reject_any_upstream_whole_part_or_marker_drift_before_reader(
    boundary: Any,
    kind: str,
    part: str,
) -> None:
    state = boundary
    state.originals[kind][part].write_bytes(b"corrupt or foreign fixture part")
    with pytest.raises(ValueError):
        read(state)
    assert state.reads == state.builds == 0


@pytest.mark.parametrize(
    "change",
    [
        "old_source",
        "pure_source",
        "own_source",
        "input",
        "scope",
        "retention_handoff",
        "pure_handoff",
        "pure_output",
    ],
)
def test_should_refuse_current_source_or_proof_drift(boundary: Any, change: str) -> None:
    state = boundary
    if change in {"old_source", "pure_source", "own_source"}:
        name = (
            "base/0.py"
            if change == "old_source"
            else next(iter(bindings.APP_SOURCES))
            if change == "pure_source"
            else bindings.OWN_SOURCES[0]
        )
        path = state.root / name
    elif change == "input":
        path = state.input_file
    elif change == "scope":
        path = state.scope
    elif change.endswith("handoff"):
        path = state.root / bindings.HANDOFFS[change.removesuffix("_handoff")]["file"]
    else:
        path = state.root / "artifacts/runs/p610-outcome-cost-pure-repeat/outcome-costs.result.json"
    path.write_bytes(b"changed fixture bytes")
    with pytest.raises(ValueError):
        read(state)
    assert state.reads == state.builds == 0


@pytest.mark.parametrize("change", ["arity", "list", "request", "result", "audit"])
def test_should_refuse_incomplete_or_changed_returned_whole_report_parts(
    boundary: Any, change: str
) -> None:
    state = boundary

    def bad() -> Any:
        parts = list(state.reader())
        if change == "arity":
            return tuple(parts[:2])
        if change == "list":
            return parts
        parts[("request", "result", "audit").index(change)]["forged"] = True
        return tuple(parts)

    with pytest.raises(ValueError):
        read(state, bad)
    assert state.reads == 1 and state.builds == 0


@pytest.mark.parametrize(
    "change", ["source", "input", "last_retention_failure", "pure_output", "request"]
)
def test_should_recheck_all_bindings_after_the_complete_reader(boundary: Any, change: str) -> None:
    state = boundary

    def late() -> Any:
        parts = state.reader()
        path = (
            state.paths["request"]
            if change == "request"
            else state.root / bindings.OWN_SOURCES[-1]
            if change == "source"
            else state.input_file
            if change == "input"
            else state.originals["retention-repeat"]["failure"]
            if change == "last_retention_failure"
            else state.root / "artifacts/runs/p610-outcome-cost-pure-repeat/outcome-costs.md"
        )
        path.write_bytes(b"late fixture drift")
        return parts

    with pytest.raises(ValueError):
        read(state, late)
    assert state.reads == state.builds == 1


@pytest.mark.parametrize("change", ["snapshot", "saved_request"])
def test_should_refuse_changed_snapshot_or_request_before_reusing_current_bindings(
    boundary: Any,
    change: str,
) -> None:
    state = boundary
    snapshot = deepcopy(state.request)
    if change == "snapshot":
        snapshot["forged"] = True
    else:
        changed = deepcopy(state.request)
        changed["forged"] = True
        replace_json(state.paths["request"], changed)
    with pytest.raises(ValueError, match="snapshot"):
        bindings._read_bound_outcome_cost_inputs(
            state.root,
            state.paths,
            state.scope,
            state.reader,
            snapshot,
        )
    assert state.reads == state.builds == 0


def test_should_refuse_any_changed_derivation_even_if_all_inputs_match(
    boundary: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = boundary

    def changed(*args: Any) -> Any:
        body = state.build(*args)
        body["forged"] = True
        return body

    monkeypatch.setattr(bindings, "build_outcome_cost_presentation", changed)
    with pytest.raises(ValueError, match="pure result"):
        read(state)
    assert state.reads == state.builds == 1
