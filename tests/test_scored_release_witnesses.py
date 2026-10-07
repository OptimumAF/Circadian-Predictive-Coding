"""Whole fixed fabricated scoring/audit traces; never historical execution proof."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from test_continual_confirmation_scoring_execution import (
    fabricated_scored_json as fabricated_scored_json,
    reference_report as reference_report,
    request_bindings as request_bindings,
    scored_request as scored_request,
    seal_execution as seal_execution,
    worker_payload as worker_payload,
)
from src.app.continual_confirmation_scoring_execution import encoded_identity, scoring_audit
from src.app.scored_release_witnesses import build_scored_release_witnesses
from test_continual_confirmation_scoring_validation import _fail_endpoints
from test_continual_confirmation_final_observation import _observation


@pytest.fixture
def fabricated_bundle(
    scored_request: dict[str, Any], worker_payload: dict[str, Any]
) -> tuple[dict[str, Any], ...]:
    result = deepcopy(worker_payload["result"])
    audit = scoring_audit(
        scored_request,
        encoded_identity(scored_request)["sha256"],
        encoded_identity(result)["sha256"],
        worker_payload,
        2.0,
    )
    return deepcopy(scored_request), result, audit


def test_should_bind_every_fixed_release_source_read_prediction_failure_and_stage(
    fabricated_bundle: tuple[dict[str, Any], ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    before = deepcopy(fabricated_bundle)

    def forbid(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("decoded chronology performed IO")

    monkeypatch.setattr(Path, "open", forbid)
    witness = build_scored_release_witnesses(*fabricated_bundle)
    assert witness["coverage"] == {
        "family_seed_rows": 60,
        "cells": 560,
        "release_events": 120,
        "source_reads": 240,
        "prediction_events": 1680,
        "prediction_examples": 67200,
        "causal_nodes": 2043,
    }
    nodes = witness["causal_nodes"]
    assert nodes[0]["input_pointer"] == "/result/training_before_release"
    assert nodes[2]["input_pointer"] == "/audit/final_observation/source_events/1"
    assert nodes[360]["input_pointer"] == "/audit/final_observation/release_events/119"
    assert nodes[-2]["input_pointer"] == "/audit/final_observation/prediction_events/1679"
    assert nodes[-1]["input_pointer"] == "/result/training_after_evaluation"
    assert all(len(row["whole_record_sha256"]) == 64 for row in nodes)
    assert fabricated_bundle == before


def test_should_keep_actual_release_time_cross_run_order_and_independence_unproved(
    fabricated_bundle: tuple[dict[str, Any], ...],
) -> None:
    witness = build_scored_release_witnesses(*fabricated_bundle)
    assert witness["documentary_request_started_utc"] == "2026-10-01T00:00:00+00:00"
    assert witness["exact_actual_release_utc"] is None
    assert witness["cross_run_release_order_verified"] is False
    assert witness["independent_source_replications"] is None
    assert witness["distinct_recorded_base_seeds"] == 50
    assert witness["complete_original_reader_verified"] is False
    assert witness["fresh_roles_authorized"] is False
    assert witness["complete_prior_usage_acceptance"] is False
    assert witness["original_p67_acceptance_complete"] is False
    assert witness == build_scored_release_witnesses(*deepcopy(fabricated_bundle))


def test_should_keep_returned_metadata_detached_from_complete_input_evidence(
    fabricated_bundle: tuple[dict[str, Any], ...],
) -> None:
    before = deepcopy(fabricated_bundle)
    witness = build_scored_release_witnesses(*fabricated_bundle)
    witness["historical_executed_updates"]["executed_updates"] += 1
    witness["historical_resource_observation"]["pid"] = 12345
    witness["complete_training_references"][0]["directory"] = "foreign"
    witness["causal_nodes"][-1]["ordinal"] = 0
    assert fabricated_bundle == before


@pytest.mark.parametrize("indices", [[1679], list(range(1680))])
@pytest.mark.parametrize("kind", ["nonfinite_predictions", "numerical_prediction_error"])
def test_should_retain_all_failed_endpoints_cells_and_late_trace_positions(
    fabricated_bundle: tuple[dict[str, Any], ...], indices: list[int], kind: str
) -> None:
    request, result, old_audit = deepcopy(fabricated_bundle)
    _fail_endpoints(result, indices, kind)
    payload = {
        key: old_audit[key]
        for key in (
            "reference_report_sha256",
            "source_map_sha256",
            "observed_updates",
            "process_rss",
            "worker_elapsed_seconds",
        )
    }
    payload.update(
        result=result,
        request_sha256=encoded_identity(request)["sha256"],
        final_observation=_observation(result),
    )
    audit = scoring_audit(
        request, payload["request_sha256"], encoded_identity(result)["sha256"], payload, 2.0
    )
    witness = build_scored_release_witnesses(request, result, audit)
    assert witness["coverage"]["prediction_events"] == 1680
    assert witness["coverage"]["cells"] == 560
    assert witness["historical_outcome_counts"]["failed_endpoints"] == len(indices)
    assert witness["historical_outcome_counts"]["failed_cells"] == len({i // 3 for i in indices})
    assert witness["causal_nodes"][-2]["input_pointer"].endswith("/1679")
    assert not witness["fresh_roles_authorized"]


@pytest.mark.parametrize(
    "mutation",
    [
        "last_target",
        "omit_last_release",
        "reorder_releases",
        "late_prediction",
        "missing_endpoint",
        "training_after",
        "request_digest",
        "result_digest",
        "source_digest",
        "work",
        "status",
        "extra_audit",
        "narrow_manifest",
        "bad_timestamp",
    ],
)
def test_should_reject_late_missing_reordered_or_detached_original_trace_links(
    fabricated_bundle: tuple[dict[str, Any], ...], mutation: str
) -> None:
    request, result, audit = deepcopy(fabricated_bundle)
    if mutation == "last_target":
        audit["final_observation"]["source_events"][-1]["field"] = "test_input"
    elif mutation == "omit_last_release":
        audit["final_observation"]["release_events"].pop()
    elif mutation == "reorder_releases":
        audit["final_observation"]["release_events"].reverse()
    elif mutation == "late_prediction":
        audit["final_observation"]["prediction_events"][-1]["role_sha256"] = "a" * 64
    elif mutation == "missing_endpoint":
        result["evaluations"].pop()
    elif mutation == "training_after":
        result["training_after_evaluation"]["rows"] = 59
    elif mutation == "request_digest":
        audit["request_sha256"] = "a" * 64
    elif mutation == "result_digest":
        audit["result_sha256"] = "a" * 64
    elif mutation == "source_digest":
        audit["source_map_sha256"] = "a" * 64
    elif mutation == "work":
        audit["work"]["by_seed"][-1]["wake_updates"] += 1
    elif mutation == "status":
        audit["status"] = "failed"
    elif mutation == "extra_audit":
        audit["verified_fresh_roles"] = True
    elif mutation == "narrow_manifest":
        request["manifest"]["train_manifest"]["families"].pop()
    else:
        request["started_utc"] = "request_is_not_a_release_time"
    with pytest.raises(ValueError):
        build_scored_release_witnesses(request, result, audit)
