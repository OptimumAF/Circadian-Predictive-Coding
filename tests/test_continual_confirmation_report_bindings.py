"""Complete-reader port/source/input failures; IO spies carry no scientific proof."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import test_continual_confirmation_report as report_tests
import test_continual_confirmation_training_references as reference_tests
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.infra import continual_confirmation_report_bindings as bindings
from src.infra.continual_confirmation_io import write_exclusive

fabricated_scored_json = report_tests.fabricated_scored_json
original_cost_metadata = report_tests.original_cost_metadata
seal_reference_data = reference_tests.seal_reference_data


@pytest.fixture
def reader_spy(
    fabricated_scored_json: dict[str, Any],
    original_cost_metadata: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    request = {
        "command": ["fixture"],
        "environment": {"fixture_only": True},
        "started_utc": "2026-10-01T00:00:00+00:00",
    }
    audit: dict[str, Any] = {
        key: {"fixture_only": True}
        for key in ("work", "observed_updates", "final_observation", "process_rss")
    }
    audit.update(
        worker_elapsed_seconds=1.0,
        elapsed_seconds=2.0,
        validation_scope="fixture_only",
        source_map_sha256="a" * 64,
    )
    parts = (request, deepcopy(fabricated_scored_json), audit)
    ids = {
        key: canonical_body_identity(body)
        for key, body in zip(("request", "result", "audit"), parts, strict=True)
    }
    monkeypatch.setattr(bindings, "SCORED_FILES", (("canonical", ids), ("repeat", deepcopy(ids))))
    monkeypatch.setattr(
        bindings, "compact_report_costs", lambda inspection: deepcopy(original_cost_metadata)
    )
    calls = []

    def costs() -> Any:
        calls.append("costs")
        return {"cost_references": {"training_references": {"fixture_only": True}}}

    def scored(directory: Path, references: Any) -> Any:
        calls.append(directory.name)
        assert references() == {"fixture_only": True}
        return deepcopy(parts)

    return parts, scored, costs, calls


def test_should_read_both_scored_bundles_with_two_fresh_complete_cost_dispatches(
    reader_spy: Any, tmp_path: Path
) -> None:
    _, scored, costs, calls = reader_spy
    payloads, cost, facts = bindings._collect_inputs(tmp_path, scored, costs)
    assert calls == ["canonical", "costs", "repeat", "costs"]
    assert len(payloads) == 2 and len(facts) == 2
    assert len(cost["cells"]) == 560
    assert facts[0]["process_rss"] == {"fixture_only": True}
    assert facts[1]["worker_elapsed_seconds"] == 1.0


@pytest.mark.parametrize("kind", ["request", "result", "audit"])
def test_should_reject_detached_returned_reader_bodies(
    reader_spy: Any, tmp_path: Path, kind: str
) -> None:
    parts, _, costs, _ = reader_spy

    def detached(directory: Path, references: Any) -> Any:
        references()
        rows = list(deepcopy(parts))
        rows[("request", "result", "audit").index(kind)]["extra"] = True
        return tuple(rows)

    with pytest.raises(ValueError, match="returned scored"):
        bindings._collect_inputs(tmp_path, detached, costs)


@pytest.mark.parametrize("count", [0, 2])
def test_should_require_exactly_one_fresh_complete_cost_reference_read_per_scored_bundle(
    reader_spy: Any, tmp_path: Path, count: int
) -> None:
    parts, _, costs, _ = reader_spy

    def wrong(directory: Path, references: Any) -> Any:
        for _ in range(count):
            references()
        return deepcopy(parts)

    with pytest.raises(ValueError, match="dispatch"):
        bindings._collect_inputs(tmp_path, wrong, costs)


def test_should_reject_last_cost_change_even_when_both_scored_bodies_repeat(
    reader_spy: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    original_cost_metadata: dict[str, Any],
) -> None:
    _, scored, costs, _ = reader_spy
    calls = 0

    def drift(inspection: Any) -> Any:
        nonlocal calls
        calls += 1
        body = deepcopy(original_cost_metadata)
        if calls == 2:
            body["cells"][-1]["executed_optimizer_updates"] += 1
        return body

    monkeypatch.setattr(bindings, "compact_report_costs", drift)
    with pytest.raises(ValueError, match="repeated original cost"):
        bindings._collect_inputs(tmp_path, scored, costs)


def test_should_preserve_every_original_source_and_add_exactly_six_report_modules() -> None:
    root = Path(__file__).parents[1]
    sources = bindings.current_report_sources(root)
    assert len(sources) == 106
    assert {n: sources[n] for n in bindings.B1_SOURCE_SHA256} == bindings.B1_SOURCE_SHA256


@pytest.mark.parametrize("name", list(bindings.REPORT_APP_SHA256) + list(bindings.B1_SOURCE_SHA256))
def test_should_refuse_current_source_drift_before_inputs(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).parents[1]
    for relative in bindings.current_report_sources(root):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((root / relative).read_bytes())
    path = tmp_path / name
    with path.open("a", encoding="utf-8") as output:
        output.write("\n# drift fixture\n")
    with pytest.raises(ValueError):
        bindings.current_report_sources(tmp_path)


def test_should_reject_missing_inputs_before_any_reader(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        bindings._input_files(tmp_path)


@pytest.fixture
def request_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    root = tmp_path
    scope = root / "scope.json"
    scope_id = {"sha256": bindings.fixed_scoring_manifest().scope_record_sha256, "byte_count": 123}
    monkeypatch.setattr(
        bindings, "current_report_sources", lambda root: {"fixture_only.py": "a" * 64}
    )
    original_identity = bindings.stream_file_identity
    monkeypatch.setattr(
        bindings,
        "stream_file_identity",
        lambda path: deepcopy(scope_id) if path == scope else original_identity(path),
    )
    monkeypatch.setattr(bindings, "_input_files", lambda root: {"fixture_only": True})
    request = bindings.report_request(root, root / "output", scope, "2026-10-01T00:00:00+00:00")
    path = root / "output/confirmation-report.request.json"
    path.parent.mkdir()
    write_exclusive(path, request)
    return root, scope, {"request": path}, request


def test_should_require_exact_current_request_and_keep_its_bytes_on_readback(
    request_boundary: Any,
) -> None:
    root, scope, paths, request = request_boundary
    before = paths["request"].read_bytes()
    assert bindings.checked_report_request(root, paths, scope) == request
    assert paths["request"].read_bytes() == before


@pytest.mark.parametrize(
    "key",
    [
        "source_map_sha256",
        "inputs",
        "environment",
        "command",
        "inventory",
        "cost_vector_identity",
        "validation_budget_seconds",
        "analysis_contract",
        "new_training_or_final_source_authorized",
        "extra",
    ],
)
def test_should_reject_resealed_request_or_scientific_scope_changes(
    request_boundary: Any, key: str
) -> None:
    root, scope, paths, request = request_boundary
    bad = deepcopy(request)
    bad[key] = "changed"
    paths["request"].unlink()
    write_exclusive(paths["request"], bad)
    with pytest.raises(ValueError):
        bindings.checked_report_request(root, paths, scope)


@pytest.mark.parametrize(
    "time", [None, "invalid", "2026-10-01T00:00:00", "2026-10-01T00:00:00-07:00"]
)
def test_should_reject_invalid_or_non_utc_request_time(request_boundary: Any, time: Any) -> None:
    root, scope, paths, request = request_boundary
    bad = deepcopy(request)
    bad["started_utc"] = time
    paths["request"].unlink()
    write_exclusive(paths["request"], bad)
    with pytest.raises(ValueError):
        bindings.checked_report_request(root, paths, scope)


def test_should_reject_late_request_change_during_source_input_checks(
    request_boundary: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, scope, paths, request = request_boundary

    def drift(root: Path) -> Any:
        with paths["request"].open("ab") as output:
            output.write(b" ")
        return {"fixture_only": True}

    monkeypatch.setattr(bindings, "_input_files", drift)
    with pytest.raises(ValueError, match="changed during bindings"):
        bindings.checked_report_request(root, paths, scope)


@pytest.mark.parametrize("kind", ["cost", "request", "result", "audit", "claim", "failure"])
def test_should_reject_changed_or_failed_input_identity_before_any_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    expected = {
        tmp_path / path: deepcopy(bindings.COST_INSPECTION_ID) for path in bindings.COST_FILES
    }
    for directory, identities in bindings.SCORED_FILES:
        for name, identity in identities.items():
            expected[tmp_path / directory / f"confirmation-scored.{name}.json"] = deepcopy(identity)
    monkeypatch.setattr(
        bindings, "verify_training_reference_bytes", lambda *a: {"fixture_only": True}
    )
    monkeypatch.setattr(bindings, "stream_file_identity", lambda path: deepcopy(expected[path]))
    if kind in {"claim", "failure"}:
        marker_directory = tmp_path / bindings.SCORED_FILES[-1][0]
        marker_directory.mkdir(parents=True)
        path = marker_directory / (
            "confirmation-scored.claim" if kind == "claim" else "confirmation-scored.failure.json"
        )
        path.write_bytes(b"late marker")
    else:
        if kind == "cost":
            path = tmp_path / bindings.COST_FILES[-1]
        else:
            path = tmp_path / bindings.SCORED_FILES[-1][0] / f"confirmation-scored.{kind}.json"
        expected[path]["sha256"] = "b" * 64
    with pytest.raises(ValueError):
        bindings._input_files(tmp_path)
