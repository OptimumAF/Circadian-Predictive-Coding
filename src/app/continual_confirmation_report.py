"""Compose every frozen seed/contrast summary with original per-arm costs.

Inputs are two complete declared scored payloads and the pinned cost vector.
Output retains all outcomes, endpoints, roles, vectors, intervals and null
reasons. This pure module proves pairing/arithmetic/repetition only. Actual
artifact/source/resource/readback proof belongs to infrastructure; no IO,
model/data, scoring, selection, pooling or new interval rule belongs here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

from src.app.continual_confirmation_analysis import OutcomeCell, analyze_confirmation
from src.app.continual_confirmation_analysis_contract import fixed_analysis_contract
from src.app.continual_confirmation_execution import json_value
from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_report_cost_binding import REPORT_COST_ID, validate_report_costs
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_confirmation_scoring_manifest import fixed_scoring_manifest
from src.app.continual_confirmation_scoring_validation import verify_scored_payload


REPORT_SCHEMA = "p611_complete_confirmation_seed_cost_report_v1"
METRICS = (
    "a_after_a",
    "a_after_b",
    "b_after_b",
    "final_mean_task_accuracy",
    "signed_forgetting_a",
    "retention_ratio_a",
)


def _cell_metrics(cell: OutcomeCell) -> dict[str, Any]:
    values = {}
    for name in METRICS:
        value = getattr(cell.accuracy, name) if cell.accuracy is not None else None
        reason = (
            cell.failure if cell.accuracy is None else ("zero_a_after_a" if value is None else None)
        )
        values[name] = {"value": value, "reason": reason}
    return values


def _joined_cells(cells: tuple[OutcomeCell, ...], costs: dict[str, Any]) -> list[dict[str, Any]]:
    require(len(cells) == len(costs["cells"]) == 560, "report complete cost/outcome join differs")
    result = []
    for cell, cost in zip(cells, costs["cells"], strict=True):
        same_json(
            {"family": cell.family, "seed": cell.seed, "arm": cell.arm},
            {key: cost[key] for key in ("family", "seed", "arm")},
            "report cost/outcome identity",
        )
        result.append(
            {
                "outcome": json_value(asdict(cell)),
                "cost": deepcopy(cost),
                "metrics": _cell_metrics(cell),
            }
        )
    return result


def metric_coverage(analysis: dict[str, Any]) -> dict[str, Any]:
    """Count retained vectors and eligibility without selecting or ranking any."""
    metrics = [
        metric
        for family in analysis["families"]
        for group in family["arms"] + family["contrasts"]
        for metric in group["metrics"]
    ]
    statements = [
        metric
        for family in analysis["families"]
        for pair in family["contrasts"]
        for metric in pair["metrics"]
        if metric["primary_endpoint"]
    ]
    observations = [row for metric in metrics for row in metric["summary"]["observations"]]
    statuses: dict[str, int] = {}
    for metric in metrics:
        status = metric["summary"]["interval_status"]
        statuses[status] = statuses.get(status, 0) + 1
    require(
        len(metrics) == 626 and len(observations) == 6260 and len(statements) == 116,
        "report fixed metric scope differs",
    )
    return {
        "cells": len(analysis["cells"]),
        "arm_vectors": 336,
        "paired_vectors": 290,
        "metric_vectors": len(metrics),
        "seed_observations": len(observations),
        "primary_contrast_statements": len(statements),
        "null_seed_observations": sum(row["value"] is None for row in observations),
        "negative_seed_observations": sum(
            row["value"] is not None and row["value"] < 0 for row in observations
        ),
        "interval_status_counts": statuses,
        "available_simultaneous_primary_intervals": sum(
            m["summary"]["simultaneous_interval"] is not None for m in statements
        ),
        "successful_cells": analysis["successful_cells"],
        "failed_cells": analysis["failed_cells"],
    }


def build_confirmation_report(
    scored_payloads: tuple[dict[str, Any], dict[str, Any]], costs: dict[str, Any]
) -> dict[str, Any]:
    """Verify complete inputs before exposing any statistics or partial report."""
    validate_report_costs(costs)
    require(
        type(scored_payloads) is tuple and len(scored_payloads) == 2,
        "report requires both full scored runs",
    )
    manifest = fixed_scoring_manifest()
    scored = tuple(verify_scored_payload(payload, manifest) for payload in scored_payloads)
    same_json(scored_payloads[0], scored_payloads[1], "report complete scored repetition")
    analyses = tuple(analyze_confirmation(row.cells, manifest.analysis_contract) for row in scored)
    bodies = tuple(json_value(asdict(analysis)) for analysis in analyses)
    same_json(bodies[0], bodies[1], "report complete analysis repetition")
    joined = _joined_cells(analyses[0].cells, costs)
    return {
        "schema_id": REPORT_SCHEMA,
        "analysis_contract": json_value(asdict(fixed_analysis_contract())),
        "analysis": bodies[0],
        "analysis_repetition": {
            "identities": [canonical_body_identity(body) for body in bodies],
            "complete_payloads_equal": True,
            "complete_analysis_equal": True,
        },
        "replication": {
            "unit": "source_seed_within_declared_family",
            "planned_seeds_per_vector": 10,
            "distinct_source_seeds": 50,
            "deterministic_repeat_adds_replications": False,
            "families_pooled": False,
        },
        "joined_cells": joined,
        "endpoint_evaluations": json_value(asdict(scored[0]))["evaluations"],
        "final_role_facts": json_value(asdict(scored[0]))["final_roles"],
        "evaluation_totals": json_value(asdict(scored[0].totals)),
        "cost_reference": {key: deepcopy(value) for key, value in costs.items() if key != "cells"},
        "cost_vector_identity": deepcopy(REPORT_COST_ID),
        "coverage": metric_coverage(bodies[0]),
        "validation_scope": "declared_full_scored_analysis_and_original_cost_join_only_not_external_proof",
        "outer_selection_scored": False,
    }
