"""Cost-reference IO spies establish neither confirmation nor source authority."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

import test_continual_confirmation_training_references as reference_tests
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.infra import continual_confirmation_report_cost_references as costs
from src.infra import continual_confirmation_training_references as references

fabricated_bundles = reference_tests.fabricated_bundles
seal_reference_data = reference_tests.seal_reference_data


def _collect(root: Any, manifest: Any, reader: Any, projector: Any) -> dict[str, Any]:
    return costs._collect_cost_references(
        root, manifest, reader, projector, reference_reader=references._read_reference_bundles
    )


def _fixture_projection(body: dict[str, Any]) -> dict[str, Any]:
    assert body["fixture_only"] is True
    return {
        "fixture_only": True,
        "cells": [{"family": "fixture", "seed": 41, "arm": "fixture"}],
        "work": {"by_seed": [{"fixture_only": True}], "totals": {"cells": 560}},
        "rejected_executed_replay_updates": 2,
    }


def test_should_compare_complete_projected_costs_and_only_retain_small_metadata(
    fabricated_bundles: tuple[Any, Any, list[str]], tmp_path: Any
) -> None:
    manifest, reader, calls = fabricated_bundles
    report = _collect(tmp_path, manifest, reader, _fixture_projection)
    assert calls == ["p67-confirmation-train", "p67-confirmation-train-repeat"]
    assert report["projection"]["rejected_executed_replay_updates"] == 2
    assert report["projection_identity"] == canonical_body_identity(report["projection"])
    assert report["deterministic_cost_repetition"] is True
    assert report["statistical_seed_report_complete"] is False
    assert "unscored" not in str(report)


def test_should_refuse_even_one_different_projected_cost_from_the_repeat(
    fabricated_bundles: tuple[Any, Any, list[str]], tmp_path: Any
) -> None:
    manifest, reader, calls = fabricated_bundles
    projections: list[dict[str, Any]] = []

    def drift(body: dict[str, Any]) -> dict[str, Any]:
        projected = _fixture_projection(body)
        if projections:
            projected["rejected_executed_replay_updates"] = 0
        projections.append(projected)
        return projected

    with pytest.raises(ValueError, match="repeated original cost"):
        _collect(tmp_path, manifest, reader, drift)
    assert len(calls) == 2


def test_should_compare_projected_work_with_independent_complete_reader_audit(
    fabricated_bundles: tuple[Any, Any, list[str]], tmp_path: Any
) -> None:
    manifest, reader, calls = fabricated_bundles

    def drift(body: dict[str, Any]) -> dict[str, Any]:
        projected = _fixture_projection(body)
        projected["work"]["totals"]["cells"] = 559
        return projected

    with pytest.raises(ValueError, match="cost projected work"):
        _collect(tmp_path, manifest, reader, drift)
    assert len(calls) == 1


def test_should_reject_late_training_file_drift_even_when_projected_costs_match(
    fabricated_bundles: tuple[Any, Any, list[str]], tmp_path: Any
) -> None:
    manifest, reader, calls = fabricated_bundles

    def drift(directory: Any) -> Any:
        parts = reader(directory)
        if len(calls) == 2:
            path = (
                tmp_path / manifest.training_bundles[0].directory / "confirmation-train.result.json"
            )
            with path.open("a", encoding="utf-8") as output:
                output.write(" ")
        return parts

    with pytest.raises(ValueError, match="bytes"):
        _collect(tmp_path, manifest, drift, _fixture_projection)


def test_should_reject_detached_complete_reader_body_before_publishing_metadata(
    fabricated_bundles: tuple[Any, Any, list[str]], tmp_path: Any
) -> None:
    manifest, reader, calls = fabricated_bundles

    def detached(directory: Any) -> Any:
        parts = list(reader(directory))
        parts[1] = deepcopy(parts[1])
        parts[1]["late_drift"] = True
        return tuple(parts)

    with pytest.raises(ValueError, match="decoded"):
        _collect(tmp_path, manifest, detached, _fixture_projection)


def test_should_not_expose_partial_fixture_scope_through_public_gate(
    fabricated_bundles: tuple[Any, Any, list[str]], tmp_path: Any
) -> None:
    _, reader, calls = fabricated_bundles
    with pytest.raises(ValueError):
        costs.read_confirmation_cost_references(tmp_path, reader)
    assert not calls
