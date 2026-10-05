"""Complete small IO spies test current bindings without scientific authority."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import test_continual_findings_development as fixtures
from src.app.continual_confirmation_execution import digest_json
from src.infra import continual_findings_development_bindings as bindings
from src.infra.continual_confirmation_io import write_exclusive
from src.infra.continual_confirmation_training_references import stream_file_identity


@pytest.fixture
def boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    state = SimpleNamespace(root=tmp_path, development_calls=[], preflight_calls=[])
    inputs = fixtures.fabricated_inputs()
    names = (
        [
            f"scripts/run_p63_{family}_factor_preflight.py"
            for family in ("sleep", "schedule", "combined", "parent")
        ]
        + ["fixture_source.py"]
        + [f"prior/{i}.py" for i in range(124)]
    )
    sources = {}
    for name in (*names, *bindings.OWN_SOURCES[:-1]):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture source\n")
        if name in names:
            sources[name] = stream_file_identity(path)["sha256"]
    selected_sources = {"fixture_source.py": sources["fixture_source.py"]}
    for bundle in inputs["development_bundles"] + inputs["preflight_bundles"]:
        request, audit = bundle["parts"]["request"], bundle["parts"]["audit"]
        request["source_sha256"] = selected_sources
        request["max_process_rss_bytes"] = 256 * 1024 * 1024
        request["adapter_sha256"] = sources.get(
            f"scripts/run_p63_{bundle['family']}_factor_preflight.py", "3" * 64
        )
        audit["process_rss"] = {
            "pid": 123,
            "sample_count": 2,
            "start_bytes": 100,
            "peak_bytes": 200,
            "interval_seconds": 0.005,
        }
        fixtures.seal_bundle(bundle)
        directory = tmp_path / bundle["directory"]
        directory.mkdir(parents=True)
        prefix = (
            fixtures.development.PREFIXES[bundle["family"]]
            if bundle in inputs["development_bundles"]
            else bundle["family"] + "-factor-preflight"
        )
        for name, part in bundle["parts"].items():
            write_exclusive(directory / f"{prefix}.{name}.json", part)
    for index, reference in enumerate(inputs["original_scope"]["development_references"]):
        for suffix_index, bundle in enumerate(
            inputs["development_bundles"][2 * index : 2 * index + 2]
        ):
            reference["bundles"][suffix_index] = {
                "directory": bundle["directory"],
                "file_sha256": {k: v["sha256"] for k, v in bundle["files"].items()},
                "result_bytes": bundle["files"]["result"]["byte_count"],
                "source_sha256": selected_sources,
            }
    catalog: dict[str, Any] = {
        "validation_seconds": 120,
        "preflight_bundles": [
            {
                "family": b["family"],
                "directory": b["directory"],
                "prefix": b["family"] + "-factor-preflight",
                "files": b["files"],
            }
            for b in inputs["preflight_bundles"]
        ],
        "scope_files": {},
        "prior_sources": {"file": "artifacts/runs/prior.json"},
    }
    for name in ("artifacts/runs/scope.json", "artifacts/runs/scope-repeat.json"):
        path = tmp_path / name
        write_exclusive(path, inputs["original_scope"])
        catalog["scope_files"][name] = stream_file_identity(path)
    prior = tmp_path / catalog["prior_sources"]["file"]
    write_exclusive(prior, {"source_sha256": sources, "source_map_sha256": digest_json(sources)})
    catalog["prior_sources"]["identity"] = stream_file_identity(prior)
    catalog_path = tmp_path / bindings.CATALOG_FILE
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    write_exclusive(catalog_path, catalog)
    monkeypatch.setattr(bindings, "CATALOG_ID", stream_file_identity(catalog_path))
    state.inputs, state.catalog = inputs, catalog

    def verifier(directory: Path, family: Any) -> dict[str, Any]:
        state.development_calls.append((family.name, directory.name))
        reference = next(
            ref
            for ref in inputs["original_scope"]["development_references"]
            if ref["family"] == family.name
        )
        return deepcopy(
            next(
                ref
                for ref in reference["bundles"]
                if ref["directory"] == directory.relative_to(tmp_path).as_posix()
            )
        )

    def preflight(family: str, result: Any) -> None:
        state.preflight_calls.append(family)
        assert result["outer_selection_scored"] is False
        assert result["final_released"] is False

    state.verifiers = {name: verifier for name in fixtures.development.FAMILIES}
    state.preflight = preflight
    return state


def read(state: Any) -> dict[str, Any]:
    return bindings.read_development_inputs(state.root, state.verifiers, state.preflight)


def part_path(state: Any, kind: str, part: str) -> Path:
    bundle = state.inputs[kind + "_bundles"][-1]
    prefix = (
        fixtures.development.PREFIXES[bundle["family"]]
        if kind == "development"
        else bundle["family"] + "-factor-preflight"
    )
    return (
        state.root
        / bundle["directory"]
        / (prefix + "." + part + ("" if part == "claim" else ".json"))
    )


def test_should_read_every_whole_part_and_dispatch_all_original_ports(boundary: Any) -> None:
    body = read(boundary)
    assert len(body["source_sha256"]) == 133
    assert len(boundary.development_calls) == 12
    assert boundary.preflight_calls == [
        "sleep",
        "sleep",
        "schedule",
        "schedule",
        "combined",
        "combined",
        "parent",
        "parent",
    ]
    assert len(body["development_bundles"]) == 12 and len(body["preflight_bundles"]) == 8
    assert body["original_scope"] == boundary.inputs["original_scope"]
    assert body["development_bundles"] == boundary.inputs["development_bundles"]
    assert body["preflight_bundles"] == boundary.inputs["preflight_bundles"]
    assert body["validation_facts"]["fresh_confirmation_reader_authority"] is False
    assert read(boundary) == body
    assert len(boundary.development_calls) == 24


@pytest.mark.parametrize("kind", ["development", "preflight"])
@pytest.mark.parametrize("part", ["request", "result", "audit", "failure", "claim"])
def test_should_reject_any_whole_part_or_completion_marker_drift(
    boundary: Any, kind: str, part: str
) -> None:
    part_path(boundary, kind, part).write_bytes(b"corrupted complete part or foreign marker")
    with pytest.raises(ValueError):
        read(boundary)


@pytest.mark.parametrize(
    "change", ["prior_source", "own_source", "catalog", "prior_handoff", "scope_repeat"]
)
def test_should_reject_current_source_or_handoff_drift(boundary: Any, change: str) -> None:
    paths = {
        "prior_source": "prior/123.py",
        "own_source": bindings.OWN_SOURCES[0],
        "catalog": bindings.CATALOG_FILE,
        "prior_handoff": boundary.catalog["prior_sources"]["file"],
        "scope_repeat": "artifacts/runs/scope-repeat.json",
    }
    if change == "own_source":
        # New presentation sources are captured at operation start; prior
        # accepted sources have permanent pins. Test drift against that snapshot.
        inputs = read(boundary)
        (boundary.root / paths[change]).write_bytes(b"changed after captured source snapshot")
        with pytest.raises(ValueError):
            bindings.verify_development_input_bindings(boundary.root, inputs)
        return
    (boundary.root / paths[change]).write_bytes(b"late changed binding")
    with pytest.raises(ValueError):
        read(boundary)


@pytest.mark.parametrize(
    "change",
    [
        "development_part",
        "preflight_failure",
        "prior_source",
        "own_source",
        "scope_repeat",
        "catalog",
        "prior_handoff",
    ],
)
def test_should_recheck_all_bindings_after_the_last_original_validator(
    boundary: Any, change: str
) -> None:
    original = boundary.preflight

    def late(family: str, result: Any) -> None:
        original(family, result)
        if len(boundary.preflight_calls) == 8:
            paths = {
                "prior_source": boundary.root / "prior/123.py",
                "own_source": boundary.root / bindings.OWN_SOURCES[1],
                "scope_repeat": boundary.root / "artifacts/runs/scope-repeat.json",
                "catalog": boundary.root / bindings.CATALOG_FILE,
                "prior_handoff": boundary.root / boundary.catalog["prior_sources"]["file"],
                "development_part": part_path(boundary, "development", "result"),
                "preflight_failure": part_path(boundary, "preflight", "failure"),
            }
            paths[change].write_bytes(b"late fixture drift")

    boundary.preflight = late
    with pytest.raises(ValueError):
        read(boundary)
    assert len(boundary.development_calls) == 12 and len(boundary.preflight_calls) == 8


def test_should_propagate_an_original_validator_rejection(boundary: Any) -> None:
    def reject(*args: Any) -> Any:
        raise ValueError("original development gate rejected")

    boundary.verifiers["parent"] = reject
    with pytest.raises(ValueError, match="original development gate rejected"):
        read(boundary)


def test_should_refuse_detached_or_missing_validator_ports(boundary: Any) -> None:
    boundary.verifiers.pop("parent")
    with pytest.raises(ValueError, match="validator ports"):
        read(boundary)
    assert boundary.development_calls == []


def test_should_refuse_an_original_validator_returning_a_different_whole_binding(
    boundary: Any,
) -> None:
    def wrong(*args: Any) -> Any:
        return {"summary_only": True}

    boundary.verifiers["gating"] = wrong
    with pytest.raises(ValueError, match="whole development validator result"):
        read(boundary)
