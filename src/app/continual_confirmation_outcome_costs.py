"""Join every complete original outcome to separately scoped accepted costs.

Public inputs require all three whole original canonical byte identities.
Output retains every metric, proof, history and original analysis declaration.
Pure arithmetic/body binding only; no IO, official-reader authority, scientific
access, new measurement, resource allocation, ranking or parent completion.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.app.continual_confirmation_json import same_json
from src.app.continual_confirmation_outcome_cost_inputs import validate_outcome_cost_inputs
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_confirmation_resources import REPORT_ID
from src.app.continual_confirmation_retention_costs import INVENTORY_ID


SCHEMA_ID = "p610_complete_original_outcomes_against_scoped_costs_v1"
RETENTION_ID = {
    "byte_count": 17804510,
    "sha256": "3338c308a36425a82716e831bd276c46f6d2a2936d27afaec4930c239d04b970",
}
INPUT_IDENTITIES = {"report": REPORT_ID, "inventory": INVENTORY_ID, "retention": RETENTION_ID}
INPUT_FILES = {
    "report": "artifacts/runs/p611-confirmation-report/confirmation-report.result.json",
    "inventory": "artifacts/runs/p610-resource-inventory/resource-inventory.result.json",
    "retention": "artifacts/runs/p610-retention-costs/retention-costs.result.json",
}
REPORT_RECORDS = (
    "analysis_contract",
    "analysis",
    "analysis_repetition",
    "replication",
    "endpoint_evaluations",
    "final_role_facts",
    "evaluation_totals",
    "coverage",
)


def _source(name: str, pointer: str) -> dict[str, Any]:
    return {
        "artifact": INPUT_FILES[name],
        "whole_identity": dict(INPUT_IDENTITIES[name]),
        "json_pointer": pointer,
    }


def _derive_outcome_costs(
    report: dict[str, Any], inventory: dict[str, Any], retention: dict[str, Any]
) -> dict[str, Any]:
    """Private complete development seam; no original artifact/reader authority."""
    validate_outcome_cost_inputs(report, inventory, retention)
    rows = []
    for i, (joined, resource, owned) in enumerate(
        zip(report["joined_cells"], inventory["rows"], retention["rows"], strict=True)
    ):
        rows.append(
            {
                **{name: resource[name] for name in ("family", "seed", "arm", "context_index")},
                "outcome": deepcopy(joined["outcome"]),
                "metrics": deepcopy(joined["metrics"]),
                "original_cost": deepcopy(joined["cost"]),
                "resource_fields": deepcopy(resource["fields"]),
                "owned_retention": deepcopy(owned),
                "source_references": {
                    name: _source(name, pointer)
                    for name, pointer in (
                        ("report", f"/joined_cells/{i}"),
                        ("inventory", f"/rows/{i}"),
                        ("retention", f"/rows/{i}"),
                    )
                },
            }
        )
    contexts = [
        {
            "family": original["family"],
            "seed": original["seed"],
            "inventory": deepcopy(original),
            "retention": deepcopy(owned),
            "source_references": {
                name: _source(name, f"/contexts/{i}") for name in ("inventory", "retention")
            },
        }
        for i, (original, owned) in enumerate(
            zip(inventory["contexts"], retention["contexts"], strict=True)
        )
    ]
    metrics = [value for row in rows for value in row["metrics"].values()]
    fields = [value for row in rows for value in row["resource_fields"].values()]
    return {
        "schema_id": SCHEMA_ID,
        "rows": rows,
        "contexts": contexts,
        "original_report_records": {name: deepcopy(report[name]) for name in REPORT_RECORDS},
        "historical_process_segments": deepcopy(inventory["run_resources"]),
        "stage_storage_totals": deepcopy(retention["stage_totals"]),
        "work": deepcopy(inventory["work"]),
        "original_resource_contract": deepcopy(inventory["field_contract"]),
        "original_inventory_gaps": deepcopy(inventory["measurement_gaps"]),
        "original_retention_scope": deepcopy(retention["scope"]),
        "coverage": {
            "cells": len(rows),
            "contexts": len(contexts),
            "owned_checkpoints": 3 * len(rows),
            "cell_metrics": len(metrics),
            "null_cell_metrics": sum(m["value"] is None for m in metrics),
            "negative_cell_metrics": sum(
                m["value"] is not None and m["value"] < 0 for m in metrics
            ),
            "above_one_retention_values": sum(
                r["metrics"]["retention_ratio_a"]["value"] is not None
                and r["metrics"]["retention_ratio_a"]["value"] > 1
                for r in rows
            ),
            "failed_cells": report["coverage"]["failed_cells"],
            "resource_fields": len(fields),
            "parameter_history_points": sum(
                len(r["resource_fields"]["parameter_history"]["value"]) for r in rows
            ),
            "historical_process_segments": 4,
            "projection_gaps_resolved": retention["coverage"]["projection_gaps_resolved"],
            "metric_vectors": report["coverage"]["metric_vectors"],
            "primary_contrast_statements": report["coverage"]["primary_contrast_statements"],
        },
        "presentation_scope": {
            "all_original_cells_and_analysis_records_preserved": True,
            "per_arm_compute": "original_recorded_and_formula_derived_work_including_executed_rollback_not_wall_time_CPU_or_FLOPs",
            "per_arm_memory": "owned_input_target_arrays_at_separate_named_checkpoints_and_recorded_capacity_history_not_process_RSS",
            "shared_memory": "one_shared_FIFO_per_context_kept_separate_no_per_arm_allocation",
            "historical_wall_and_sampled_RSS": "four_distinct_original_whole_process_segments_no_per_arm_attribution_no_summing_repeats",
            "null_failed_negative_above_one_rejected_inactive": "preserve_original_values_reasons_outcome_cost_and_endpoint_records",
            "repeated_runs_add_seed_replications_or_original_work": False,
            "new_metric_or_composite_winner": False,
        },
        "input_identities": deepcopy(INPUT_IDENTITIES),
        "validation_scope": "pure_complete_whole_saved_input_body_binding_and_joint_proof_only_not_current_file_source_or_fresh_official_reader_execution",
        "new_measurement_training_or_final_access": False,
        "original_P6_10_acceptance_complete": False,
    }


def build_outcome_cost_presentation(
    report: dict[str, Any], inventory: dict[str, Any], retention: dict[str, Any]
) -> dict[str, Any]:
    """Bind every whole input before validating/presenting any original outcome."""
    for name, body in (("report", report), ("inventory", inventory), ("retention", retention)):
        same_json(
            canonical_body_identity(body),
            INPUT_IDENTITIES[name],
            "complete original outcome-cost " + name,
        )
    return _derive_outcome_costs(report, inventory, retention)
