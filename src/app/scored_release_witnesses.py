"""Bind the entire original scored payload/audit to supported release trace order.

Inputs are complete decoded original request/result/audit dictionaries. Outputs
retain every causal trace pointer and whole record digest, raw declared identities,
execution/resource/failure links and explicit chronology limits. Existing complete
scoring validators remain unchanged. This pure consumer establishes no actual
file readback, source provenance, independent replication or fresh admission.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from typing import Any

from src.app.continual_confirmation_json import same_json
from src.app.continual_confirmation_scoring_execution import encoded_identity, scoring_audit
from src.core.seed_release_chronology import RecordedChronologyNode, recorded_release_order


def _whole_record_digest(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _check_complete_bundle(
    request: dict[str, Any], result: dict[str, Any], audit: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    if any(type(value) is not dict for value in (request, result, audit)):
        raise ValueError("release witnesses require complete request/result/audit objects")
    identities = {
        name: encoded_identity(value)
        for name, value in (("request", request), ("result", result), ("audit", audit))
    }
    fields = (
        "reference_report_sha256",
        "source_map_sha256",
        "observed_updates",
        "final_observation",
        "process_rss",
        "worker_elapsed_seconds",
    )
    try:
        payload = {key: audit[key] for key in fields}
        payload.update(result=result, request_sha256=identities["request"]["sha256"])
        expected = scoring_audit(
            request,
            identities["request"]["sha256"],
            identities["result"]["sha256"],
            payload,
            audit["elapsed_seconds"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("release witness original audit links are missing/malformed") from error
    same_json(audit, expected, "release witness complete original scoring audit")
    return identities


def _node_record(
    node: RecordedChronologyNode, result: dict[str, Any], observation: dict[str, Any]
) -> tuple[str, Any]:
    sections = {
        "source_input_read": "source_events",
        "source_target_read": "source_events",
        "role_release": "release_events",
        "prediction": "prediction_events",
    }
    if node.kind in sections:
        section = sections[node.kind]
        assert node.record_index is not None
        return f"/audit/final_observation/{section}/{node.record_index}", observation[section][
            node.record_index
        ]
    return f"/result/{node.kind}", result[node.kind]


def _causal_records(result: dict[str, Any], observation: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = recorded_release_order(
        len(observation["release_events"]), len(observation["prediction_events"])
    )
    rows = []
    for node in nodes:
        pointer, record = _node_record(node, result, observation)
        rows.append(
            {
                **asdict(node),
                "input_pointer": pointer,
                "whole_record_sha256": _whole_record_digest(record),
            }
        )
    return rows


def build_scored_release_witnesses(
    request: dict[str, Any], result: dict[str, Any], audit: dict[str, Any]
) -> dict[str, Any]:
    """Validate every fixed original field before interpreting an observer record."""
    identities = _check_complete_bundle(request, result, audit)
    observation = audit["final_observation"]
    nodes = _causal_records(result, observation)
    summary = request["summary"]
    # Why this: different family views, callbacks, copies and repeated runs may
    # share a numeric source. Preserve events without converting them to trials.
    seeds = {row["seed"] for row in observation["release_events"]}
    return {
        "schema_id": "complete_original_scored_release_witnesses_v1",
        "validation_scope": "complete_decoded_original_scoring_audit_and_observer_links_only",
        "declared_decoded_file_identities": identities,
        "source_map_sha256": request["source_map_sha256"],
        "manifest_sha256": request["manifest_sha256"],
        "training_reference_report_sha256": request["reference_report_sha256"],
        "complete_training_references": deepcopy(request["manifest"]["training_bundles"]),
        "coverage": {
            "family_seed_rows": summary["family_seed_rows"],
            "cells": summary["cells"],
            "release_events": len(observation["release_events"]),
            "source_reads": len(observation["source_events"]),
            "prediction_events": len(observation["prediction_events"]),
            "prediction_examples": observation["prediction_examples"],
            "causal_nodes": len(nodes),
        },
        "causal_nodes": nodes,
        "whole_observer_trace_sha256": _whole_record_digest(observation),
        "historical_outcome_counts": result["totals"].copy(),
        "historical_executed_updates": deepcopy(audit["observed_updates"]),
        "historical_full_work_sha256": _whole_record_digest(audit["work"]),
        "historical_resource_observation": deepcopy(audit["process_rss"]),
        "historical_worker_elapsed_seconds": audit["worker_elapsed_seconds"],
        "historical_parent_elapsed_seconds": audit["elapsed_seconds"],
        "documentary_request_started_utc": request["started_utc"],
        "exact_actual_release_utc": None,
        "cross_run_release_order_verified": False,
        "distinct_recorded_base_seeds": len(seeds),
        "independent_source_replications": None,
        "all_original_causal_records_bound_by_pointer_and_whole_digest": True,
        "complete_original_reader_verified": False,
        "prospective_real_role_seeds_selected": False,
        "fresh_roles_authorized": False,
        "complete_prior_usage_acceptance": False,
        "original_p67_acceptance_complete": False,
    }
