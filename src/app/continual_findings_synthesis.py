"""Synthesize all fixed development and independent confirmation findings.

Inputs are complete pinned ledgers, raw contexts, costs, matrix and failures.
Output retains them all and joins every original cell and hypothesis context.
No new statistic, selection, pooling, IO or scientific operation belongs here.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.app.continual_confirmation_findings import HYPOTHESES
from src.app.continual_confirmation_json import require, same_json
from src.app.continual_findings_failures import build_failure_findings
from src.app.continual_findings_synthesis_inputs import (
    verify_synthesis_declarations,
    verify_synthesis_identities,
)


CONTEXT_FAMILIES = {
    "H1": ("gating", "sleep", "combined"),
    "H2": ("sleep", "combined", "parent"),
    "H3": ("replay", "sleep", "combined"),
    "H4": ("schedule", "combined"),
}
LIMITS = {
    "H1": "Both A endpoints matter: weaker A-after-A can reduce signed forgetting without improving retention. Full-system ablations do not isolate chemical gating; the original simultaneous family resolves no directional benefit.",
    "H2": "Named initial/after-A/after-B and transient capacities, fixed/wider/scheduled/random controls and rollback work are retained. Recorded work is not FLOPs or isolated runtime; no resolved accuracy/capacity or compute advantage is established at these settings.",
    "H3": "Original offered/applied/rejected replay identities, updates, presentations and homeostasis proposals remain. Comparators share their declared replay schedule within their own family; unequal exposure/work cannot be silently treated as a matched cost or an independent consolidation benefit.",
    "H4": "Original adaptive/periodic/no-sleep attempts, skips, commits, reasons and work remain. Shared neutral controllers have three matched appliers. Zero attempts or unchanged proposals supply no positive evidence of a scheduling mechanism; no resolved benefit at the declared budget is established.",
}


def _cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[name] for name in ("family", "seed", "arm"))


def _join_cells(bodies: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    columns = (
        bodies["outcome_costs"]["rows"],
        bodies["matrix"]["rows"],
        bodies["activity"]["cells"],
    )
    for index, (outcome, matrix, activity) in enumerate(zip(*columns, strict=True)):
        same_json(_cell_key(outcome), _cell_key(matrix), "complete matrix/cost cell identity")
        same_json(_cell_key(outcome), _cell_key(activity), "complete activity/cost cell identity")
        same_json(outcome["metrics"], matrix["metrics"], "original matrix/cost metrics")
        same_json(
            outcome["outcome"]["failure"], matrix["original_cell_failure"], "original cell failure"
        )
        same_json(
            outcome["resource_fields"],
            activity["original_work_and_capacity_fields"],
            "original cell work/capacity",
        )
        rows.append(
            {
                "family": outcome["family"],
                "seed": outcome["seed"],
                "arm": outcome["arm"],
                "metrics": deepcopy(outcome["metrics"]),
                "failure": deepcopy(matrix["original_cell_failure"]),
                "recorded_transaction_outcomes": deepcopy(
                    activity["recorded_transaction_outcomes"]
                ),
                "transaction_scope": activity["transaction_scope"],
                "zero_recorded_sleep_attempts": outcome["resource_fields"]["sleep_attempts"][
                    "value"
                ]
                == 0,
                "original_evidence_pointers": {
                    "outcome_costs": f"/bodies/outcome_costs/rows/{index}",
                    "matrix": f"/bodies/matrix/rows/{index}",
                    "activity": f"/bodies/activity/cells/{index}",
                },
            }
        )
    return rows


def _hypothesis_findings(
    primary: dict[str, Any], cells: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    statements = primary["primary_statements"]
    require(
        all(
            s["classification"] in ("unresolved_zero_included", "interval_ineligible")
            for s in statements
        ),
        "fixed synthesis requires the original unresolved primary evidence",
    )
    return [
        {
            "hypothesis_id": identifier,
            "title": title,
            "question": question,
            "status": "unresolved",
            "decision_scope": "complete_fixed_six_family_development_and_independent_confirmation_measured_settings",
            "primary_statement_ids": [
                s["statement_id"] for s in statements if identifier in s["hypothesis_context"]
            ],
            "complete_context_cell_indices": [
                index
                for index, cell in enumerate(cells)
                if cell["family"] in CONTEXT_FAMILIES[identifier]
            ],
            "context_families": list(CONTEXT_FAMILIES[identifier]),
            "remaining_uncertainty": LIMITS[identifier],
            "interpretation": "not_equivalence_or_broad_rejection; all_contexts_retained_without_a_hypothesis_vote_or_mechanism_inference_from_full_system_comparisons",
        }
        for identifier, title, question in HYPOTHESES
    ]


def _derive_synthesis(inputs: dict[str, Any]) -> dict[str, Any]:
    """Complete pure declaration seam; no current source or execution authority."""
    bodies = inputs["bodies"]
    verify_synthesis_declarations(bodies, inputs["preserved_records"])
    cells = _join_cells(bodies)
    same_json(len(cells), 560, "all original confirmation cells")
    failures = build_failure_findings(
        inputs["preserved_records"], inputs["catalog"]["failure_groups"]
    )
    return {
        "schema_id": "p612_complete_fixed_development_confirmation_findings_v1",
        "original_inputs": deepcopy(inputs),
        "confirmation_cells": cells,
        "hypothesis_findings": _hypothesis_findings(bodies["primary"], cells),
        "operational_failures": failures,
        "coverage": {
            "confirmation_cells": 560,
            "primary_statements": len(bodies["primary"]["primary_statements"]),
            "development_cells": len(bodies["development"]["cells"]),
            "development_pairs": len(bodies["development"]["paired_differences"]),
            "activity": deepcopy(bodies["activity"]["coverage"]),
            "outcome_costs": deepcopy(bodies["outcome_costs"]["coverage"]),
            "matrix": deepcopy(bodies["matrix"]["coverage"]),
            "preserved_whole_records": len(inputs["preserved_records"]),
        },
        "interpretation_rules": {
            "development": deepcopy(bodies["development"]["interpretation"]),
            "independent_confirmation": "original_final_role_ten_source_seeds_per_vector_fifty_distinct_sources_no_family_pooling_or_repeat_replication",
            "primary": "original_116_statement_simultaneous_family_unchanged_no_marginal_fallback_or_selected_interval",
            "costs": deepcopy(bodies["outcome_costs"]["presentation_scope"]),
            "activity": deepcopy(bodies["activity"]["scope"]),
            "negative_and_null": "retain_raw_values_reasons_failed_records_zero_denominators_and_ineligible_intervals; negative_signed_values_are_not_all_regressions",
            "transfer": bodies["matrix"]["transfer_scope"],
            "hypotheses": "all_four_unresolved_within_complete_fixed_measured_settings_no_equivalence_global_rejection_or_composite_winner",
        },
        "validation_scope": "complete_pinned_stored_declarations_only_not_current_file_source_or_fresh_official_reader_proof",
        "fresh_official_reader_authority": False,
        "new_measurement_training_scoring_or_final_access": False,
        "original_p612_acceptance_complete": False,
        "pending_acceptance": [
            "P6.12b3 complete current publication and independent readbacks",
            "unchanged original P6.12b/P6.12 audit",
        ],
    }


def build_complete_findings(inputs: dict[str, Any]) -> dict[str, Any]:
    """Bind all complete canonical inputs before interpreting any finding."""
    try:
        require(type(inputs) is dict, "findings inputs must be a complete object")
        verify_synthesis_identities(inputs)
        return _derive_synthesis(inputs)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("complete findings inputs are malformed or incomplete") from error
