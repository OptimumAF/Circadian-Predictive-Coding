"""Inventory all original confirmation resources without new measurements.

Public inputs are the entire pinned cost inspection/report and four audit
objects. Output is every resource row, context, run scope and explicit gap.
Pure validation binds declared input bytes; current file/source/execution
authority belongs to infrastructure. No IO, model, scoring or ranking.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_manifest import ConfirmationFamily, fixed_confirmation_manifest
from src.app.continual_confirmation_matrix_inputs import verify_matrix_report_input
from src.app.continual_confirmation_report_cost_binding import COST_INSPECTION_ID
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_confirmation_resource_contexts import count, summarize_context
from src.app.continual_confirmation_resource_fields import cell_fields


RESOURCE_SCHEMA = "p610_complete_original_resource_inventory_v1"
REPORT_ID = {
    "byte_count": 7537678,
    "sha256": "363ed97dba808281d13224526b4b36614ae452188cade256992b530b6ab03088",
}
RESOURCE_FIELDS = {
    "wall_time": "individual_arm_wall_time_unmeasured_four_distinct_original_whole_process_audits_recorded",
    "wake_updates": "original_per_arm_counter_and_live_aggregate_including_rejected_replay",
    "latent_iterations": "formula_based_wake_applied_and_rejected_replay_counts_not_CPU_or_FLOP_profiling",
    "replay_exposure": "original_applied_presentations_distinct_committed_ids_and_separate_shared_supply",
    "sleep_guard_overhead": "attempts_prediction_calls_and_examples_recorded_isolated_duration_unmeasured",
    "peak_memory": "four_original_sampled_process_RSS_segments_not_per_arm_or_continuous_peak",
    "replay_bytes": "individual_owned_fields_when_available_group_FIFO_and_owned_arrays_before_copies_not_RSS",
    "initial_final_peak_parameters": "actual_three_checkpoint_capacities_and_derived_transient_peak_geometry",
    "parameter_history": "every_available_checkpoint_epoch_and_transaction_point_not_continuous_time_series",
    "outcome_against_compute_memory": "separate_P6_10b_unfinished_no_composite_winner",
}


def _scope(projection: dict[str, Any], families: tuple[ConfirmationFamily, ...]) -> None:
    expected = [(f.name, seed, arm) for f in families for seed in f.seeds for arm in f.arms]
    same_json(
        [(c["family"], c["seed"], c["arm"]) for c in projection["cells"]],
        expected,
        "resource entire ordered cell scope",
    )
    contexts = [(f.name, seed) for f in families for seed in f.seeds]
    same_json(
        [(c["family"], c["seed"]) for c in projection["seed_contexts"]],
        contexts,
        "resource entire ordered context scope",
    )
    same_json(
        [(c["family"], c["seed"]) for c in projection["work"]["by_seed"]],
        contexts,
        "resource entire work scope",
    )


def _contexts(
    projection: dict[str, Any], families: tuple[ConfirmationFamily, ...]
) -> list[dict[str, Any]]:
    results = []
    family_map = {family.name: family for family in families}
    for index, (context, work) in enumerate(
        zip(projection["seed_contexts"], projection["work"]["by_seed"], strict=True)
    ):
        cells = [
            cell
            for cell in projection["cells"]
            if (cell["family"], cell["seed"]) == (context["family"], context["seed"])
        ]
        same_json(len(cells), work["cells"], "resource context cell count")
        same_json(
            work["retained_array_bytes_before_copies"],
            family_map[context["family"]].retained_array_bytes_per_seed_before_copies,
            "resource original declared group storage",
        )
        results.append(summarize_context(context, index, work, cells))
    totals = projection["work"]["totals"]
    for name, value in totals.items():
        same_json(
            count(value, name),
            sum(count(work[name], name) for work in projection["work"]["by_seed"]),
            f"resource complete work total {name}",
        )
    same_json(
        projection["work"]["maximum_transient_width"],
        max(row["maximum_transient_width"] for row in results),
        "resource complete transient width",
    )
    return results


def _coverage(
    rows: list[dict[str, Any]], contexts: list[dict[str, Any]], runs: list[dict[str, Any]]
) -> dict[str, Any]:
    fields = [field for row in rows for field in row["fields"].values()]
    return {
        "cells": len(rows),
        "checkpoint_capacities": 3 * len(rows),
        "shared_contexts": len(contexts),
        "resource_fields": len(fields),
        "run_resource_segments": len(runs),
        "status_counts": {
            status: sum(field["status"] == status for field in fields)
            for status in ("measured", "derived", "unmeasured")
        },
        "parameter_history_points": sum(
            len(row["fields"]["parameter_history"]["value"]) for row in rows
        ),
        "shared_FIFO_bytes_across_distinct_contexts": sum(
            row["shared_fifo_bytes"] for row in contexts
        ),
        "owned_array_bytes_across_distinct_contexts_before_copies": sum(
            row["owned_retained_array_bytes_before_copies"] for row in contexts
        ),
        "per_arm_timing_or_RSS_measured": False,
    }


def _derive_inventory(
    projection: dict[str, Any], families: tuple[ConfirmationFamily, ...], runs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Private development seam; does not bind inputs or establish authority."""
    try:
        _scope(projection, families)
        contexts = _contexts(projection, families)
        indexed = {
            (context["family"], context["seed"]): i
            for i, context in enumerate(projection["seed_contexts"])
        }
        rows = []
        for index, cell in enumerate(projection["cells"]):
            context_index = indexed[(cell["family"], cell["seed"])]
            fields = cell_fields(
                cell, index, projection["seed_contexts"][context_index], context_index
            )
            rows.append(
                {
                    "family": cell["family"],
                    "seed": cell["seed"],
                    "arm": cell["arm"],
                    "context_index": context_index,
                    "fields": fields,
                }
            )
        for index, context in enumerate(contexts):
            peak = max(
                max(point["width"] for point in row["fields"]["parameter_history"]["value"])
                for row in rows
                if row["context_index"] == index
            )
            same_json(
                peak,
                context["maximum_transient_width"],
                "resource independently reconstructed context capacity peak",
            )
        return {
            "schema_id": RESOURCE_SCHEMA,
            "field_contract": deepcopy(RESOURCE_FIELDS),
            "rows": rows,
            "contexts": contexts,
            "run_resources": deepcopy(runs),
            "work": deepcopy(projection["work"]),
            "coverage": _coverage(rows, contexts, runs),
            "measurement_gaps": {
                "per_arm_wall_RSS_and_isolated_sleep_guard_duration": "unmeasured_original_whole_process_scope_cannot_be_divided_by_arm_count_any_new_measurement_requires_prospective_protocol",
                "continuous_parameter_history": "only_named_original_checkpoint_epoch_transaction_points_available_do_not_infer_unrecorded_time_samples",
                "per_arm_owned_retention_when_absent_from_raw_method": "use_exact_group_storage_and_original_checkpoint_proof_do_not_allocate_group_bytes_by_arm_count",
                "accuracy_forgetting_against_cost": "P6.10b_must_present_all_original_outcomes_against_this_scoped_inventory_without_a_composite_winner",
            },
            "replication": "distinct_original_contexts_once_repeats_do_not_add_work_or_seed_replications",
            "validation_scope": "pure_original_declared_resource_inventory_only_not_current_file_source_or_execution_proof",
            "new_measurement_training_or_final_access": False,
            "original_P6_10_acceptance_complete": False,
        }
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("resource inventory has malformed original cost/context facts") from error


def _run_rows(
    inspection: dict[str, Any], report: dict[str, Any], audits: tuple[dict[str, Any], ...]
) -> list[dict[str, Any]]:
    references = inspection["cost_references"]["training_references"]["bundles"]
    recorded = references + report["scored_run_facts"]
    require(
        len(audits) == len(recorded) == 4, "resource requires both original train and scored audits"
    )
    rows = []
    for index, (audit, reference) in enumerate(zip(audits, recorded, strict=True)):
        kind, repeat = ("train" if index < 2 else "scored"), bool(index % 2)
        path = f"artifacts/runs/p67-confirmation-{kind}{'-repeat' if repeat else ''}/confirmation-{kind}.audit.json"
        same_json(
            canonical_body_identity(audit),
            reference["files"]["audit"],
            "resource complete original audit body",
        )
        for name in (
            "elapsed_seconds",
            "worker_elapsed_seconds",
            "process_rss",
            "observed_updates",
            "work",
        ):
            same_json(audit[name], reference[name], f"resource recorded run {name}")
        same_json(
            audit["work"],
            inspection["cost_references"]["projection"]["work"],
            "resource audit original work",
        )
        rows.append(
            {
                "kind": kind,
                "repeat": repeat,
                "audit_path": path,
                "audit_identity": deepcopy(reference["files"]["audit"]),
                "scope": "original_whole_process_segment_including_training_validation_and_for_scored_run_final_evaluation_not_per_arm",
                "wall_time": {
                    "value": audit["elapsed_seconds"],
                    "unit": "seconds",
                    "status": "measured",
                    "source_pointer": "/elapsed_seconds",
                },
                "worker_time": {
                    "value": audit["worker_elapsed_seconds"],
                    "unit": "seconds",
                    "status": "measured",
                    "source_pointer": "/worker_elapsed_seconds",
                },
                "sampled_RSS": {
                    "value": deepcopy(audit["process_rss"]),
                    "unit": "RSS_bytes_and_seconds",
                    "status": "measured",
                    "source_pointer": "/process_rss",
                    "reason": "sampled_absolute_process_segment_not_continuous_peak_or_isolated_model_storage",
                },
                "observed_optimizer_updates": deepcopy(audit["observed_updates"]),
            }
        )
    return rows


def build_resource_inventory(
    inspection: dict[str, Any], report: dict[str, Any], audits: tuple[dict[str, Any], ...]
) -> dict[str, Any]:
    """Bind every original complete input before deriving the full inventory."""
    same_json(
        canonical_body_identity(inspection),
        COST_INSPECTION_ID,
        "resource complete original cost inspection",
    )
    same_json(canonical_body_identity(report), REPORT_ID, "resource complete original report")
    verify_matrix_report_input(report)
    projection = inspection["cost_references"]["projection"]
    same_json(
        projection["cells"],
        [cell["cost"] for cell in report["joined_cells"]],
        "resource all original cost/report cells",
    )
    same_json(
        projection["work"], report["cost_reference"]["work"], "resource original cost/report work"
    )
    runs = _run_rows(inspection, report, audits)
    result = _derive_inventory(projection, fixed_confirmation_manifest().families, runs)
    result["input_identities"] = {
        "cost_inspection": deepcopy(COST_INSPECTION_ID),
        "confirmation_report": deepcopy(REPORT_ID),
        "complete_cost_projection": deepcopy(inspection["cost_references"]["projection_identity"]),
    }
    return result
