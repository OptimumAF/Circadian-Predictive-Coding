"""Complete original pilot precision and work-budget feasibility consumer.

Input validation reuses the whole original pilot/development boundary. Output
retains every contrast and conditional assumption, including constant pilots.
No IO, confirmation source, scoring, winner or fresh-seed authority is granted.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_pilot_variability import build_pilot_variability_report
from src.app.continual_precision_contract import (
    ConfirmationPrecisionContract,
    validate_precision_contract,
)
from src.core.seed_precision_budget import PrecisionBudgetSpec, plan_seed_precision_budget
from src.core.seed_statistics import SeedObservation


def _work_budget(pilot: dict[str, Any], contract: ConfirmationPrecisionContract) -> dict[str, Any]:
    manifest = pilot["original_manifest"]
    per_seed, families = 0, []
    for family in manifest["families"]:
        count = len(family["seeds"])
        maximum = family["maximum_optimizer_updates"]
        require(
            count == 10 and type(maximum) is int and maximum % count == 0,
            "exact original additive family work",
        )
        ceiling = maximum // count
        per_seed += ceiling
        families.append(
            {
                "name": family["name"],
                "cells_per_replication": len(family["arms"]),
                "maximum_updates_per_replication": ceiling,
            }
        )
    require(per_seed == 1562, "complete six-family optimizer work ceiling")
    return {
        "families": families,
        "maximum_optimizer_updates": manifest["max_optimizer_updates"],
        "maximum_complete_replications_per_family": manifest["max_optimizer_updates"] // per_seed,
        "candidate_replications_per_family": contract.candidate_seed_count,
        "candidate_maximum_optimizer_updates": per_seed * contract.candidate_seed_count,
        "wall_limit_seconds": manifest["wall_limit_seconds"],
        "max_process_rss_bytes": manifest["max_process_rss_bytes"],
        "rss_interval_seconds": manifest["rss_interval_seconds"],
        "resource_status": "prospective_ceiling_not_new_time_or_memory_fit_measurement",
    }


def _vector(
    row: dict[str, Any], pilot: dict[str, Any], contract: ConfirmationPrecisionContract
) -> dict[str, Any]:
    metric = row["primary_metric"]
    bounds = (
        contract.mean_accuracy_difference_range
        if metric == "final_mean_task_accuracy"
        else contract.forgetting_difference_range
    )
    spec = PrecisionBudgetSpec(
        contract.target_half_width,
        contract.mean_family_alpha,
        contract.pilot_variance_family_alpha,
        contract.statement_count,
        *bounds,
        contract.candidate_seed_count,
        pilot["original_analysis_contract"]["simultaneous_critical"],
        pilot["original_analysis_contract"]["zero_deviation_tolerance"],
    )
    original = row["projection"]["pilot_summary"]["observations"]
    observations = tuple(SeedObservation(r["seed"], r["value"], r["reason"]) for r in original)
    family = next(f for f in pilot["original_manifest"]["families"] if f["name"] == row["family"])
    result = plan_seed_precision_budget(observations, tuple(family["development_seeds"]), spec)
    same_json(asdict(result.pilot), row["projection"], "unchanged whole original pilot projection")
    return {
        "family": row["family"],
        "left": row["left"],
        "right": row["right"],
        "primary_metric": metric,
        "original_metric_field": row["original_metric_field"],
        "original_pair_provenance": row["original_pair_provenance"],
        "precision": asdict(result),
    }


def _project_precision_feasibility(
    pilot: dict[str, Any], contract: ConfirmationPrecisionContract
) -> dict[str, Any]:
    """Private fabricated seam, with no original/current input or execution authority."""
    validate_precision_contract(contract)
    try:
        require(type(pilot) is dict, "precision requires the complete pilot object")
        same_json(
            pilot["coverage"]["primary_vectors"],
            contract.statement_count,
            "all original primary vectors",
        )
        expected = [
            (f["name"], left, right, metric)
            for f in pilot["original_manifest"]["families"]
            for left, right in f["contrasts"]
            for metric in pilot["original_analysis_contract"]["primary_metrics"]
        ]
        same_json(
            [(r["family"], r["left"], r["right"], r["primary_metric"]) for r in pilot["vectors"]],
            expected,
            "complete ordered precision scope",
        )
        vectors = [_vector(row, pilot, contract) for row in pilot["vectors"]]
        budget = _work_budget(pilot, contract)
        require(
            budget["maximum_complete_replications_per_family"] == contract.candidate_seed_count,
            "candidate count must retain the fixed complete budget",
        )
        result = {
            "schema_id": "p67_complete_prospective_precision_feasibility_v1",
            "contract": asdict(contract),
            "complete_original_pilot_identity": canonical_body_identity(pilot),
            "original_input_bindings": pilot["input_bindings"],
            "coverage": pilot["coverage"],
            "vectors": vectors,
            "work_budget": budget,
            "precision_objective_supported_by_all_bounds": all(
                r["precision"]["bounded_mean_target_met"] for r in vectors
            ),
            "conditional_normal_sensitivity_target_met_count": sum(
                r["precision"]["conditional_normal_target_met"] is True for r in vectors
            ),
            "conditional_normal_sensitivity_unresolved_count": sum(
                r["precision"]["conditional_normal_target_met"] is None for r in vectors
            ),
            "interpretation": "model_sensitivity_and_conservative_bounded_feasibility_only_not_necessary_counts_measured_intervals_power_or_prospective_execution_authority",
            "new_confirmation_authorized": False,
            "untouched_role_seed_binding_complete": False,
            "original_p67_acceptance_complete": False,
        }
        canonical_body_identity(result)
        return deepcopy(result)
    except (KeyError, TypeError, IndexError, AttributeError, StopIteration) as error:
        raise ValueError("precision pilot input is malformed or incomplete") from error


def build_precision_feasibility(
    original_development_inputs: dict[str, Any], contract: ConfirmationPrecisionContract
) -> dict[str, Any]:
    """Reject a changed objective before verifying any original declaration."""
    validate_precision_contract(contract)
    return _project_precision_feasibility(
        build_pilot_variability_report(original_development_inputs), contract
    )
