"""Real whole IO with private reader spies; no historical/source authority."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from test_scored_release_witnesses import (
    fabricated_bundle as fabricated_bundle,
    fabricated_scored_json as fabricated_scored_json,
    reference_report as reference_report,
    request_bindings as request_bindings,
    scored_request as scored_request,
    seal_execution as seal_execution,
    worker_payload as worker_payload,
)
from src.app.continual_confirmation_scoring_execution import encoded_identity
from src.infra import original_release_witnesses as boundary
from src.infra.continual_confirmation_io import write_exclusive


@pytest.fixture
def fixture_boundary(tmp_path: Path, fabricated_bundle: tuple[dict[str, Any], ...]) -> Any:
    directory = tmp_path / "scored"
    directory.mkdir()
    expected = {}
    for name, body in zip(("request", "result", "audit"), fabricated_bundle, strict=True):
        write_exclusive(directory / f"confirmation-scored.{name}.json", body)
        expected[name] = encoded_identity(body)
    state = SimpleNamespace(
        root=tmp_path,
        directory="scored",
        expected=expected,
        parts=deepcopy(fabricated_bundle),
        reader_calls=0,
        source_checks=0,
    )

    def reader(path: Path) -> tuple[dict[str, Any], ...]:
        assert path == directory
        state.reader_calls += 1
        return deepcopy(state.parts)

    def sources() -> dict[str, str]:
        state.source_checks += 1
        return deepcopy(state.parts[0]["source_sha256"])

    state.reader, state.sources = reader, sources
    return state


def _read(state: Any) -> Any:
    return boundary._read_scored_witness(
        state.root, state.directory, state.expected, state.reader, state.sources
    )


def test_should_bind_all_actual_whole_files_before_and_after_the_complete_reader(
    fixture_boundary: Any,
) -> None:
    state = fixture_boundary
    before = deepcopy(state.parts)
    witness = _read(state)
    assert state.reader_calls == 1 and state.source_checks == 2
    assert witness.file_identities == state.expected
    assert witness.chronology["coverage"]["causal_nodes"] == 2043
    assert witness.complete_original_reader_verified is False
    assert witness.chronology["complete_original_reader_verified"] is False
    assert witness.chronology["fresh_roles_authorized"] is False
    witness.file_identities["audit"]["sha256"] = "a" * 64
    witness.chronology["historical_resource_observation"]["pid"] = 0
    assert state.parts == before
    assert state.expected["audit"]["sha256"] != "a" * 64


@pytest.mark.parametrize("part", ["request", "result", "audit"])
def test_should_reject_changed_whole_bytes_before_any_reader(
    fixture_boundary: Any, part: str
) -> None:
    state = fixture_boundary
    (state.root / state.directory / f"confirmation-scored.{part}.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        _read(state)
    assert state.reader_calls == 0


@pytest.mark.parametrize("marker", ["claim", "failure.json"])
def test_should_refuse_pending_or_failed_runs_before_any_reader(
    fixture_boundary: Any, marker: str
) -> None:
    state = fixture_boundary
    (state.root / state.directory / f"confirmation-scored.{marker}").write_bytes(
        b"preserved marker"
    )
    with pytest.raises(ValueError):
        _read(state)
    assert state.reader_calls == 0


def test_should_reject_late_whole_file_drift_after_complete_decoded_validation(
    fixture_boundary: Any,
) -> None:
    state = fixture_boundary
    original = state.reader

    def mutate(path: Path) -> Any:
        parts = original(path)
        (path / "confirmation-scored.audit.json").write_bytes(b"changed late")
        return parts

    state.reader = mutate
    with pytest.raises(ValueError):
        _read(state)
    assert state.reader_calls == 1


def test_should_reject_late_source_drift_and_retain_all_input_files(fixture_boundary: Any) -> None:
    state = fixture_boundary

    def snapshot() -> dict[str, bytes]:
        files = {}
        for path in (state.root / state.directory).iterdir():
            with path.open("rb") as stream:
                files[path.name] = stream.read()
        return files

    before = snapshot()
    original = state.sources

    def changed() -> dict[str, str]:
        sources = original()
        if state.source_checks == 2:
            sources["scripts/run_p67_confirmation_scoring.py"] = "a" * 64
        return sources

    state.sources = changed
    with pytest.raises(ValueError):
        _read(state)
    assert state.reader_calls == 1
    assert snapshot() == before


@pytest.mark.parametrize("kind", ["list", "missing_part", "extra_part", "changed_payload"])
def test_should_reject_partial_or_detached_reader_returns(fixture_boundary: Any, kind: str) -> None:
    state = fixture_boundary
    if kind == "list":
        state.reader = lambda path: list(state.parts)
    elif kind == "missing_part":
        state.reader = lambda path: state.parts[:2]
    elif kind == "extra_part":
        state.reader = lambda path: (*state.parts, {})
    else:
        state.parts[1]["evaluations"][-1]["example_count"] = 39
    with pytest.raises(ValueError):
        _read(state)


@pytest.mark.parametrize(
    "directory", ["../scored", "/scored", "C:/scored", "scored/../scored", "scored\\nested"]
)
def test_should_reject_external_or_noncanonical_paths_before_readers(
    fixture_boundary: Any, directory: str
) -> None:
    state = fixture_boundary
    state.directory = directory
    with pytest.raises(ValueError):
        _read(state)
    assert state.reader_calls == 0


def test_should_require_the_actual_original_reader_root_for_the_public_boundary(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="original reader root"):
        boundary.read_original_scoring_release(tmp_path, "canonical")


def test_should_refuse_unknown_original_bundle_names_before_sources_or_readers(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="bundle name"):
        boundary.read_original_scoring_release(tmp_path, "caller_selected_seeds")
