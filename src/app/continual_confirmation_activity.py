"""Retain original guarded decisions, skipped policies and replay offers.

Inputs are the whole original raw cost inspection and outcome/cost presentation.
Outputs are complete ordered transaction/offer records and scoped cell counters.
No IO, source execution proof, new measurement, hypothesis decision or model
access belongs here. A no-op or skipped control remains an observed result.
"""

from __future__ import annotations

from copy import deepcopy
from math import isclose, isfinite
from typing import Any

from src.app.continual_confirmation_json import integer, require, same_json
from src.app.continual_confirmation_report_costs import canonical_body_identity


COST_INSPECTION_ID = {
    "byte_count": 112635395,
    "sha256": "214ce7ad00f61ca1aac8f85ad1381f50e9fc4235baeec59d7daf6043cc7d81dd",
}
OUTCOME_COST_ID = {
    "byte_count": 39956601,
    "sha256": "5ba4d44772838113a244807b2033e8e24ef3a77ec8fe74bb967b795f7f1d4bfd",
}
OUTCOMES = ("accepted", "rolled_back", "skipped")


def _identity(row: dict[str, Any]) -> tuple[str, int, str]:
    return row["family"], row["seed"], row["arm"]


def _check_transaction(event: dict[str, Any]) -> None:
    outcome = event["outcome"]
    require(outcome in OUTCOMES, "activity original transaction outcome")
    replay, changes, guard = event["replay"], event["changes"], event["guard"]
    proposed = integer(replay["proposed_updates"], "activity proposed updates")
    applied = integer(replay["applied_updates"], "activity applied updates")
    require(applied <= proposed, "activity applied replay exceeds proposal")
    if outcome == "skipped":
        require(
            guard is None and proposed == applied == 0, "skipped activity has executed guarded work"
        )
    else:
        require(type(guard) is dict, "attempted activity lacks its original guard")
        pre, post, tolerance = guard["pre_accuracy"], guard["post_accuracy"], guard["tolerance"]
        require(
            all(
                type(value) in (int, float) and isfinite(value)
                for value in (pre, post, tolerance, guard["delta"])
            ),
            "activity guard finite numeric fields",
        )
        require(0 <= pre <= 1 and 0 <= post <= 1 and tolerance >= 0, "activity guard fractions")
        # The original guard delta is pre minus post. Preserve its sign.
        require(
            isclose(guard["delta"], pre - post, rel_tol=0, abs_tol=1e-12),
            "activity original guard delta",
        )
        require((outcome == "accepted") == (post >= pre - tolerance), "activity guard acceptance")
    if outcome != "accepted":
        require(applied == 0, "uncommitted activity retained replay updates")
        for field in (
            "applied_split_pairs",
            "applied_removed_prune_ids",
            "applied_scheduled_prune_ids",
        ):
            require(not changes[field], "uncommitted activity retained topology changes")
        same_json(event["final_width"], event["before_width"], "activity rollback/skipped width")


def _transaction_rows(context: dict[str, Any], index: int) -> list[dict[str, Any]]:
    rows = []
    family = context["family"]
    for opportunity_index, opportunity in enumerate(
        context["legacy_train_context"]["opportunities"]
    ):
        for decision_index, decision in enumerate(opportunity["decisions"]):
            _check_transaction(decision["event"])
            rows.append(
                {
                    "family": family,
                    "seed": context["seed"],
                    "owner": decision["name"],
                    "owner_scope": "individual_guarded_transaction",
                    "phase": opportunity["phase"],
                    "epoch": opportunity["epoch"],
                    "outcome": decision["event"]["outcome"],
                    "reason": decision["event"]["reason"],
                    "trigger_reason": decision["event"]["trigger_reason"],
                    "raw_record": deepcopy(decision),
                    "source_pointer": f"/cost_references/projection/seed_contexts/{index}/legacy_train_context/opportunities/{opportunity_index}/decisions/{decision_index}",
                }
            )
    return rows


def _schedule_rows(context: dict[str, Any], index: int) -> list[dict[str, Any]]:
    rows = []
    for opportunity_index, opportunity in enumerate(
        context["legacy_train_context"]["opportunities"]
    ):
        for decision_index, decision in enumerate(opportunity["decisions"]):
            outcome, attempted = decision["outcome"], decision["attempted"]
            require(
                outcome in OUTCOMES and type(attempted) is bool, "schedule activity outcome/attempt"
            )
            require(
                attempted == (outcome != "skipped"), "schedule activity attempt and outcome differ"
            )
            if attempted:
                pre, post = decision["guard_pre_accuracy"], decision["guard_post_accuracy"]
                require(
                    all(type(value) in (int, float) and isfinite(value) for value in (pre, post)),
                    "schedule activity finite guard fields",
                )
                require(0 <= pre <= 1 and 0 <= post <= 1, "schedule activity guard fractions")
                require(
                    (outcome == "accepted") == (post >= pre), "schedule activity original guard"
                )
            else:
                require(
                    decision["guard_pre_accuracy"] is None
                    and decision["guard_post_accuracy"] is None,
                    "skipped schedule has measured guard",
                )
            if outcome != "accepted":
                require(
                    all(not ids for ids in decision["applied_ids_by_method"].values()),
                    "uncommitted schedule retained replay identities",
                )
            rows.append(
                {
                    "family": "schedule",
                    "seed": context["seed"],
                    "owner": decision["policy"],
                    "owner_scope": "neutral_controller_with_three_matched_appliers_not_three_independent_transactions",
                    "phase": opportunity["phase"],
                    "epoch": opportunity["epoch"],
                    "outcome": outcome,
                    "reason": decision["reason"],
                    "trigger_reason": decision["trigger_reason"],
                    "raw_record": deepcopy(decision),
                    "source_pointer": f"/cost_references/projection/seed_contexts/{index}/legacy_train_context/opportunities/{opportunity_index}/decisions/{decision_index}",
                }
            )
    return rows


def _sleep_rows(context: dict[str, Any], cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, cell in enumerate(cells):
        if (cell["family"], cell["seed"]) != (context["family"], context["seed"]):
            continue
        event = cell["method_facts"]["sleep"]
        if event is None:
            continue
        require(event["outcome"] in ("accepted", "rolled_back"), "sleep activity outcome")
        require(
            (event["outcome"] == "accepted")
            == (event["guard_post_accuracy"] >= event["guard_pre_accuracy"]),
            "sleep activity original guard",
        )
        if event["outcome"] == "rolled_back":
            require(
                not event["applied_split_pairs"] and not event["applied_removed_prune_ids"],
                "rolled-back sleep retained topology",
            )
        rows.append(
            {
                "family": "sleep",
                "seed": context["seed"],
                "owner": cell["arm"],
                "owner_scope": "individual_a_boundary_guarded_transaction",
                "phase": "a",
                "epoch": 12,
                "outcome": event["outcome"],
                "reason": "original_outcome_no_separate_reason_field",
                "trigger_reason": "fixed_original_a_boundary",
                "raw_record": deepcopy(event),
                "source_pointer": f"/cost_references/projection/cells/{index}/method_facts/sleep",
            }
        )
    return rows


def _context_activity(
    context: dict[str, Any], cells: list[dict[str, Any]], index: int
) -> list[dict[str, Any]]:
    if context["family"] == "sleep":
        return _sleep_rows(context, cells)
    if context["family"] == "schedule":
        return _schedule_rows(context, index)
    if context["family"] in ("combined", "parent"):
        return _transaction_rows(context, index)
    # These pilots retain attempt counters/offered identities without a
    # per-sleep commit record. An absent record is not an observed zero commit.
    return []


def _cell_activity(row: dict[str, Any], decisions: list[dict[str, Any]]) -> dict[str, Any]:
    family, seed, arm = _identity(row)
    relevant = [
        d
        for d in decisions
        if (d["family"], d["seed"]) == (family, seed)
        and (d["owner"] == arm or family == "schedule" and arm == "neutral_" + d["owner"])
    ]
    fields = row["resource_fields"]
    recorded = (
        family in ("sleep", "combined", "parent")
        or family == "schedule"
        and arm.startswith("neutral_")
    )
    counts = {
        outcome: sum(d["outcome"] == outcome for d in relevant) if recorded else None
        for outcome in OUTCOMES
    }
    if recorded:
        same_json(
            integer(counts["accepted"], "activity accepted transaction count")
            + integer(counts["rolled_back"], "activity rolled-back transaction count"),
            fields["sleep_attempts"]["value"],
            "activity transactions and original attempt counter",
        )
    return {
        "family": family,
        "seed": seed,
        "arm": arm,
        "recorded_transaction_outcomes": counts,
        "transaction_scope": "recorded_own_transactions"
        if recorded
        else "no_individual_transaction_record_not_zero_activity",
        "original_method_facts": deepcopy(row["original_cost"]["method_facts"]),
        "original_work_and_capacity_fields": deepcopy(fields),
        "activity_references": [d["source_pointer"] for d in relevant],
    }


def _derive_activity(
    cost_inspection: dict[str, Any], outcome_costs: dict[str, Any]
) -> dict[str, Any]:
    """Complete declaration projection, without current source/execution authority."""
    projection = cost_inspection["cost_references"]["projection"]
    cells, contexts = projection["cells"], projection["seed_contexts"]
    require(len(cells) == 560 and len(contexts) == 60, "activity complete original scope")
    same_json(
        cells,
        [row["original_cost"] for row in outcome_costs["rows"]],
        "activity all original outcome-cost cell facts",
    )
    same_json(projection["work"], outcome_costs["work"], "activity complete original work")
    decisions = [
        record
        for index, context in enumerate(contexts)
        for record in _context_activity(context, cells, index)
    ]
    offers = [
        {
            "family": "replay",
            "seed": context["seed"],
            "raw_record": deepcopy(boundary),
            "source_pointer": f"/cost_references/projection/seed_contexts/{index}/legacy_train_context/boundaries/{boundary_index}",
        }
        for index, context in enumerate(contexts)
        if context["family"] == "replay"
        for boundary_index, boundary in enumerate(context["legacy_train_context"]["boundaries"])
    ]
    require(
        len(decisions) == 3650 and len(offers) == 60, "activity all original decision/offer records"
    )
    attempted = sum(record["outcome"] != "skipped" for record in decisions)
    same_json(
        attempted,
        projection["work"]["totals"]["guarded_attempts"],
        "activity all original guarded attempts",
    )
    return {
        "schema_id": "p612_complete_original_confirmation_activity_v1",
        "decisions": decisions,
        "replay_offers": offers,
        "cells": [_cell_activity(row, decisions) for row in outcome_costs["rows"]],
        "coverage": {
            "cells": 560,
            "contexts": 60,
            "decisions": len(decisions),
            "replay_offers": len(offers),
            "guarded_attempts": attempted,
            "outcomes": {
                outcome: sum(record["outcome"] == outcome for record in decisions)
                for outcome in OUTCOMES
            },
        },
        "scope": {
            "all_original_skipped_and_rejected_reasons_retained": True,
            "replay_counter_only_commit_status": "unmeasured_not_inferred_from_attempts",
            "schedule_transactions": "one_neutral_controller_per_policy_with_three_matched_appliers",
            "inactive_controls": "preserve_zero_attempts_empty_proposals_and_same_state_no_benefit_inference",
            "parameter_and_work_fields": "original_scope_units_status_and_nulls_unchanged_no_per_arm_wall_or_RSS",
            "validation": "pure_whole_stored_input_binding_and_activity_projection_not_current_source_execution_proof",
        },
    }


def build_confirmation_activity(
    cost_inspection: dict[str, Any], outcome_costs: dict[str, Any]
) -> dict[str, Any]:
    """Bind both whole original bodies before interpreting any activity record."""
    for body, expected, label in (
        (cost_inspection, COST_INSPECTION_ID, "cost inspection"),
        (outcome_costs, OUTCOME_COST_ID, "outcomes/costs"),
    ):
        same_json(canonical_body_identity(body), expected, "complete original activity " + label)
    try:
        return _derive_activity(cost_inspection, outcome_costs)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("confirmation activity is malformed or incomplete") from error
