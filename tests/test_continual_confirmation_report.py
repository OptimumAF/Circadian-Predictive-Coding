"""Complete report arithmetic/pairing using fabricated scores and original cost metadata."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pytest

import test_continual_confirmation_scoring_validation as scored_tests
from src.app import continual_confirmation_report as reporting
from src.app import continual_confirmation_report_cost_binding as costs
from src.app.continual_confirmation_analysis import analyze_confirmation
from src.app.continual_confirmation_scoring_manifest import fixed_scoring_manifest
from src.app.continual_confirmation_scoring_validation import verify_scored_payload

fabricated_scored_json = scored_tests.fabricated_scored_json
seal_validation = scored_tests.seal_validation


@pytest.fixture(scope="module")
def original_cost_metadata() -> dict[str, Any]:
    return json.loads(
        (Path(__file__).parent / "fixtures/p611_confirmation_report_costs.json").read_text(
            encoding="utf-8"
        )
    )


def test_should_publish_every_cell_metric_vector_pair_and_raw_cost_without_additional_replications(
    fabricated_scored_json: dict[str, Any], original_cost_metadata: dict[str, Any]
) -> None:
    payload = deepcopy(fabricated_scored_json)
    before = deepcopy(payload)
    report = reporting.build_confirmation_report(
        (payload, deepcopy(payload)), original_cost_metadata
    )
    expected = analyze_confirmation(
        verify_scored_payload(payload, fixed_scoring_manifest()).cells,
        fixed_scoring_manifest().analysis_contract,
    )
    assert report["analysis"] == json.loads(json.dumps(asdict(expected)))
    assert report["coverage"]["metric_vectors"] == 626
    assert report["coverage"]["seed_observations"] == 6260
    assert report["coverage"]["primary_contrast_statements"] == 116
    assert report["coverage"]["cells"] == 560
    assert report["replication"]["planned_seeds_per_vector"] == 10
    assert report["replication"]["distinct_source_seeds"] == 50
    assert report["replication"]["deterministic_repeat_adds_replications"] is False
    assert (
        report["analysis_repetition"]["identities"][0]
        == report["analysis_repetition"]["identities"][1]
    )
    assert len(report["endpoint_evaluations"]) == 1680
    assert len(report["final_role_facts"]) == 60
    assert len(report["joined_cells"]) == 560
    for joined, outcome, cost in zip(
        report["joined_cells"],
        report["analysis"]["cells"],
        original_cost_metadata["cells"],
        strict=True,
    ):
        assert joined["outcome"] == outcome
        assert joined["cost"] == cost
    assert report["joined_cells"][0]["metrics"]["final_mean_task_accuracy"]["value"] == 0.0625
    assert report["joined_cells"][0]["metrics"]["signed_forgetting_a"]["value"] == -0.025
    assert report["joined_cells"][0]["metrics"]["retention_ratio_a"]["value"] == 2.0
    assert (
        report["cost_reference"]["complete_context_files"]
        == original_cost_metadata["complete_context_files"]
    )
    assert "seed_contexts" not in report["cost_reference"]
    assert (
        report["validation_scope"]
        == "declared_full_scored_analysis_and_original_cost_join_only_not_external_proof"
    )
    assert payload == before
    assert report == reporting.build_confirmation_report(
        (payload, deepcopy(payload)), original_cost_metadata
    )


@pytest.mark.parametrize("indices", [[0], [1679], list(range(1680))])
def test_should_publish_all_failure_nulls_and_suppress_intervals_without_dropping_any_seed(
    fabricated_scored_json: dict[str, Any],
    original_cost_metadata: dict[str, Any],
    indices: list[int],
) -> None:
    payload = deepcopy(fabricated_scored_json)
    scored_tests._fail_endpoints(payload, indices, "nonfinite_predictions")
    report = reporting.build_confirmation_report(
        (payload, deepcopy(payload)), original_cost_metadata
    )
    assert report["coverage"]["metric_vectors"] == 626
    assert report["coverage"]["seed_observations"] == 6260
    assert report["analysis"]["failed_cells"] == len({i // 3 for i in indices})
    assert len(report["endpoint_evaluations"]) == 1680
    for index in {i // 3 for i in indices}:
        row = report["joined_cells"][index]
        assert all(v["value"] is None and v["reason"] for v in row["metrics"].values())
    if len(indices) == 1680:
        for family in report["analysis"]["families"]:
            for group in family["arms"] + family["contrasts"]:
                for metric in group["metrics"]:
                    assert metric["summary"]["mean"] is None
                    assert metric["summary"]["simultaneous_interval"] is None


@pytest.mark.parametrize(
    "change", ["late_metric", "late_proof", "missing_cell", "extra_field", "wrong_role"]
)
def test_should_reject_changed_or_incomplete_scored_repetition_before_reporting(
    fabricated_scored_json: dict[str, Any], original_cost_metadata: dict[str, Any], change: str
) -> None:
    other = deepcopy(fabricated_scored_json)
    if change == "late_metric":
        other["cells"][-1]["accuracy"]["a_after_b"] = 0.5
    elif change == "late_proof":
        other["training_after_evaluation"]["rows"] = 1
    elif change == "missing_cell":
        other["cells"].pop()
    elif change == "extra_field":
        other["extra"] = True
    else:
        other["final_roles"][-1]["b"]["sha256"] = "a" * 64
    with pytest.raises(ValueError):
        reporting.build_confirmation_report((fabricated_scored_json, other), original_cost_metadata)


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "rejected", "capacity", "context", "source", "extra"]
)
def test_should_reject_detached_or_incomplete_original_cost_metadata(
    original_cost_metadata: dict[str, Any], change: str
) -> None:
    value = deepcopy(original_cost_metadata)
    if change == "missing":
        value["cells"].pop()
    elif change == "duplicate":
        value["cells"][-1] = deepcopy(value["cells"][0])
    elif change == "rejected":
        value["cells"][-1]["rejected_executed_replay_updates"] += 1
    elif change == "capacity":
        value["cells"][-1]["checkpoints"][-1]["parameter_count"] += 1
    elif change == "context":
        value["context_references"][-1]["json_pointer"] = "/missing"
    elif change == "source":
        value["base_source_map_sha256"] = "a" * 64
    else:
        value["extra"] = True
    with pytest.raises(ValueError):
        costs.validate_report_costs(value)


def test_should_reject_cost_drift_before_any_statistical_summary(
    fabricated_scored_json: dict[str, Any],
    original_cost_metadata: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = deepcopy(original_cost_metadata)
    bad["cells"].pop()

    def forbid(*args: Any) -> Any:
        raise AssertionError("unbound costs reached analysis")

    monkeypatch.setattr(reporting, "analyze_confirmation", forbid)
    with pytest.raises(ValueError):
        reporting.build_confirmation_report((fabricated_scored_json, fabricated_scored_json), bad)


@pytest.mark.parametrize("value", [{}, {"cost_references": {}}, {"nonfinite": float("nan")}])
def test_should_refuse_partial_or_unbound_complete_cost_inspection(value: Any) -> None:
    with pytest.raises(ValueError):
        costs.compact_report_costs(value)
