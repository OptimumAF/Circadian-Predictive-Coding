"""Source/IO/readback port spies are metadata tests, not scientific proof."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import test_continual_confirmation_matrix as matrix_tests
import test_continual_confirmation_training_references as reference_tests
from src.app.continual_confirmation_execution import digest_json
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.infra import continual_confirmation_matrix_bindings as bindings
from src.infra.continual_confirmation_io import read_json, write_exclusive
from src.infra.continual_confirmation_matrix_artifacts import artifact_paths
from src.infra.continual_confirmation_report_artifacts import artifact_paths as report_paths
from src.infra.continual_confirmation_training_references import stream_file_identity

fabricated_scored_json = matrix_tests.fabricated_scored_json
original_cost_metadata = matrix_tests.original_cost_metadata
fabricated_report = matrix_tests.fabricated_report
seal_reference_data = reference_tests.seal_reference_data


def _replace(path: Path, value: Any) -> None:
    path.unlink()
    write_exclusive(path, value)


@pytest.fixture
def boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fabricated_report: dict[str, Any]
) -> Any:
    state = SimpleNamespace(
        root=tmp_path, directory=tmp_path / "matrix", scope=tmp_path / "scope.json", reads=0
    )
    state.scope.write_bytes(b"fixture scope")
    base = {}
    for index in range(106):
        name = f"base/{index}.py"
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"fixture old source")
        base[name] = stream_file_identity(path)["sha256"]
    monkeypatch.setattr(bindings, "current_report_sources", lambda root: deepcopy(base))
    monkeypatch.setattr(bindings, "REPORT_SOURCE_MAP_SHA256", digest_json(base))
    apps = {}
    for name in (*bindings.MATRIX_APP_SHA256, *bindings.OWN_SOURCES):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture new source")
        if name in bindings.MATRIX_APP_SHA256:
            apps[name] = stream_file_identity(path)["sha256"]
    monkeypatch.setattr(bindings, "MATRIX_APP_SHA256", apps)
    state.parts = []
    references = []
    for index in range(2):
        directory = f"input-report-{index}"
        paths = report_paths(tmp_path / directory)
        paths["request"].parent.mkdir()
        request = {
            "fixture_only": True,
            "inputs": {"fixture_only": True},
            "scope_identity": stream_file_identity(state.scope),
            "scoring_manifest_sha256": "a" * 64,
            "analysis_contract_sha256": "b" * 64,
        }
        result, audit = deepcopy(fabricated_report), {"fixture_only": True}
        for name, body in (("request", request), ("result", result), ("audit", audit)):
            write_exclusive(paths[name], body)
        paths["markdown"].write_bytes(b"fixture original Markdown\n")
        references.append(
            (
                directory,
                {
                    n: stream_file_identity(paths[n])
                    for n in ("request", "result", "markdown", "audit")
                },
            )
        )
        state.parts.append((request, result, audit))
    monkeypatch.setattr(bindings, "REPORT_FILES", tuple(references))
    monkeypatch.setattr(
        bindings, "checked_report_request", lambda root, paths, scope: read_json(paths["request"])
    )
    state.directory.mkdir()
    state.paths = artifact_paths(state.directory)
    state.request = bindings.matrix_request(
        tmp_path, state.directory, state.scope, "2026-10-01T20:00:00+00:00"
    )
    write_exclusive(state.paths["request"], state.request)

    def reader() -> Any:
        state.reads += 1
        return deepcopy(state.parts[0])

    state.reader = reader
    return state


def test_should_bind_all_112_sources_and_both_report_bundles_before_complete_port(
    boundary: Any,
) -> None:
    state = boundary
    body = bindings.verified_matrix_body(state.root, state.paths, state.scope, state.reader)
    assert state.reads == 1 and len(state.request["source_sha256"]) == 112
    assert len(state.request["inputs"]["report_files"]) == 2
    assert body["provenance"]["inputs"] == state.request["inputs"]
    assert body["coverage"]["matrix_slots"] == 2240
    assert body["report_identity"] == canonical_body_identity(state.parts[0][1])


@pytest.mark.parametrize(
    "name",
    [
        "source_map_sha256",
        "source_sha256",
        "inputs",
        "inventory",
        "validation_budget_seconds",
        "command",
        "environment",
        "started_utc",
        "new_training_or_final_source_authorized",
        "unknown",
    ],
)
def test_should_reject_changed_or_unknown_request_before_complete_report_port(
    boundary: Any, name: str
) -> None:
    state = boundary
    value = deepcopy(state.request)
    value[name] = True
    _replace(state.paths["request"], value)
    with pytest.raises(ValueError):
        bindings.verified_matrix_body(state.root, state.paths, state.scope, state.reader)
    assert state.reads == 0


@pytest.mark.parametrize("name", ["request", "result", "markdown", "audit", "claim", "failure"])
def test_should_refuse_changed_input_bytes_or_marker_before_reader(
    boundary: Any, name: str
) -> None:
    state = boundary
    paths = report_paths(state.root / bindings.REPORT_FILES[-1][0])
    with paths[name].open("ab") as output:
        output.write(b"late or occupied input")
    with pytest.raises(ValueError):
        bindings.verified_matrix_body(state.root, state.paths, state.scope, state.reader)
    assert state.reads == 0


@pytest.mark.parametrize("name", [*bindings.MATRIX_APP_SHA256, *bindings.OWN_SOURCES])
def test_should_refuse_current_source_drift_before_reader(boundary: Any, name: str) -> None:
    state = boundary
    with (state.root / name).open("ab") as output:
        output.write(b"source drift")
    with pytest.raises(ValueError):
        bindings.verified_matrix_body(state.root, state.paths, state.scope, state.reader)
    assert state.reads == 0


@pytest.mark.parametrize("part", [0, 1, 2])
def test_should_refuse_detached_complete_reader_result_before_pure_matrix(
    boundary: Any, monkeypatch: pytest.MonkeyPatch, part: int
) -> None:
    state = boundary
    parts = list(deepcopy(state.parts[0]))
    parts[part]["detached"] = True

    def forbid(*args: Any) -> Any:
        raise AssertionError("detached input reached pure matrix")

    monkeypatch.setattr(bindings, "build_confirmation_matrix", forbid)
    with pytest.raises(ValueError):
        bindings.verified_matrix_body(state.root, state.paths, state.scope, lambda: tuple(parts))


@pytest.mark.parametrize("kind", ["source", "input", "marker", "request"])
def test_should_recheck_source_input_markers_and_request_after_full_reader(
    boundary: Any, kind: str
) -> None:
    state = boundary

    def reader() -> Any:
        if kind == "source":
            path = state.root / bindings.OWN_SOURCES[0]
        elif kind == "request":
            path = state.paths["request"]
        else:
            paths = report_paths(state.root / bindings.REPORT_FILES[0][0])
            path = paths["result" if kind == "input" else "claim"]
        with path.open("ab") as output:
            output.write(b"late drift")
        return deepcopy(state.parts[0])

    with pytest.raises(ValueError):
        bindings.verified_matrix_body(state.root, state.paths, state.scope, reader)


@pytest.mark.parametrize("value", [None, [], ({},), [{}, {}, {}]])
def test_should_refuse_partial_reader_dispatch(boundary: Any, value: Any) -> None:
    state = boundary
    with pytest.raises(ValueError):
        bindings.verified_matrix_body(state.root, state.paths, state.scope, lambda: value)


@pytest.mark.parametrize(
    "time", [None, "invalid", "2026-10-01T20:00:00", "2026-10-01T20:00:00-07:00"]
)
def test_should_refuse_invalid_non_utc_or_naive_request_time_before_readers(
    boundary: Any, time: Any
) -> None:
    with pytest.raises(ValueError):
        bindings.matrix_request(boundary.root, boundary.directory, boundary.scope, time)


def test_should_refuse_noncanonical_request_bytes(boundary: Any) -> None:
    with boundary.paths["request"].open("ab") as output:
        output.write(b" ")
    with pytest.raises(ValueError, match="canonical"):
        bindings.verified_matrix_body(
            boundary.root, boundary.paths, boundary.scope, boundary.reader
        )
    assert boundary.reads == 0
