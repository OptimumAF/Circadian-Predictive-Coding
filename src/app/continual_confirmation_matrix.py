"""Present every stored confirmation stage/task value, null and role link.

Input is the complete declared report. Output is every 560 individual 2x2
matrix, three original endpoint records and original derived metrics. B-after-A
and forward transfer remain unmeasured. No IO/source/execution proof, scoring,
learning, inference, selection or new primary/interval family belongs here.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.app.continual_confirmation_matrix_inputs import verify_matrix_report_input
from src.app.continual_confirmation_report_costs import canonical_body_identity


MATRIX_SCHEMA = "p69_complete_stored_confirmation_matrix_v1"
UNKNOWN_B_REASON = "not_declared_before_task_b_arrival"


def _slot(record: dict[str, Any], pointer: str) -> dict[str, Any]:
    result = record["result"]
    failure = result["failure"]
    return {
        "value": result["correct_count"] / record["example_count"] if failure is None else None,
        "status": "measured" if failure is None else "failed",
        "reason": None
        if failure is None
        else f"{failure['code']}:{failure['error_type'] or 'none'}",
        "endpoint_pointer": pointer,
    }


def _row(joined: dict[str, Any], records: list[dict[str, Any]], index: int) -> dict[str, Any]:
    outcome = joined["outcome"]
    endpoints: list[dict[str, Any]] = [
        {"pointer": f"/endpoint_evaluations/{3 * index + offset}", "record": deepcopy(record)}
        for offset, record in enumerate(records)
    ]
    aa, ab, bb = (_slot(item["record"], item["pointer"]) for item in endpoints)
    unknown = {
        "value": None,
        "status": "unmeasured",
        "reason": UNKNOWN_B_REASON,
        "endpoint_pointer": None,
    }
    forgetting = joined["metrics"]["signed_forgetting_a"]
    transfer = -forgetting["value"] if forgetting["value"] is not None else None
    direction = (
        "undefined"
        if transfer is None
        else ("positive" if transfer > 0 else ("negative" if transfer < 0 else "zero"))
    )
    return {
        **{name: outcome[name] for name in ("family", "seed", "arm")},
        "score_role": outcome["roles"]["score_role"],
        "accuracy_matrix": [[aa, unknown], [ab, bb]],
        "original_cell_failure": outcome["failure"],
        "metrics": deepcopy(joined["metrics"]),
        "backward_transfer_a": {
            "value": transfer,
            "reason": forgetting["reason"],
            "direction": direction,
            "interpretation": "descriptive_a_after_b_minus_a_after_a_same_stored_roles",
        },
        "forward_transfer_b": {
            "value": None,
            "reason": "b_after_a_and_untrained_reference_not_declared",
        },
        "endpoints": endpoints,
    }


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    slots = [slot for row in rows for stage in row["accuracy_matrix"] for slot in stage]
    transfers = [row["backward_transfer_a"]["direction"] for row in rows]
    return {
        "rows": len(rows),
        "matrix_slots": len(slots),
        "successful_endpoints": sum(slot["status"] == "measured" for slot in slots),
        "failed_endpoints": sum(slot["status"] == "failed" for slot in slots),
        "unmeasured_slots": sum(slot["status"] == "unmeasured" for slot in slots),
        "failed_cells": sum(row["original_cell_failure"] is not None for row in rows),
        "undefined_retention_zero_denominator": sum(
            row["metrics"]["retention_ratio_a"]["reason"] == "zero_a_after_a" for row in rows
        ),
        "backward_transfer_directions": {
            name: transfers.count(name) for name in ("positive", "negative", "zero", "undefined")
        },
    }


def build_confirmation_matrix(report: dict[str, Any]) -> dict[str, Any]:
    """Validate the whole original scope before exposing any individual matrix."""
    rebuilt = verify_matrix_report_input(report)
    endpoints = rebuilt["endpoint_evaluations"]
    rows = [
        _row(joined, endpoints[3 * index : 3 * index + 3], index)
        for index, joined in enumerate(rebuilt["joined_cells"])
    ]
    return {
        "schema_id": MATRIX_SCHEMA,
        "stage_axis": ["after_a", "after_b"],
        "task_axis": ["a", "b"],
        "measured_endpoint_order": ["a_after_a", "a_after_b", "b_after_b"],
        "report_identity": canonical_body_identity(report),
        "analysis_contract_sha256": rebuilt["analysis"]["contract_sha256"],
        "replication": deepcopy(rebuilt["replication"]),
        "rows": rows,
        "coverage": _coverage(rows),
        "missing_cell_policy": UNKNOWN_B_REASON,
        "derived_metric_policy": "original_whole_cell_failure_and_zero_denominator_rules_unchanged",
        "transfer_scope": "descriptive_restated_backward_transfer_only_forward_transfer_unmeasured",
        "validation_scope": "complete_declared_stored_matrix_and_report_links_only_not_source_or_execution_proof",
        "new_training_or_final_source_access": False,
        "original_fully_measured_matrix_acceptance_complete": False,
    }
