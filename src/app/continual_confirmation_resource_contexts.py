"""Derive scoped context work, replay supply and recorded capacity histories.

Inputs are original decoded cost contexts/cells. Outputs are small summaries
and exact source pointers; complete proofs stay in their bound input artifact.
No IO, model, profiling, outcomes, execution proof or cost attribution.
"""

from __future__ import annotations

from typing import Any

from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_report_costs import canonical_body_identity


def count(value: Any, name: str) -> int:
    require(type(value) is int and value >= 0, f"resource {name} must be a nonnegative integer")
    return int(value)


def context_pointer(index: int) -> str:
    return f"/cost_references/projection/seed_contexts/{index}"


def _guard_work(context: dict[str, Any]) -> tuple[int, int]:
    raw, family = context["legacy_train_context"], context["family"]
    if family == "sleep":
        attempts = len(context["supplemental_guards"])
        return attempts, 48 * attempts
    guarded = [
        opportunity["phase"]
        for opportunity in raw.get("opportunities", [])
        for decision in opportunity["decisions"]
        if (
            decision["attempted"]
            if family == "schedule"
            else decision["event"]["guard"] is not None
        )
    ]
    require(all(phase in {"a", "b"} for phase in guarded), "resource guard phase differs")
    return len(guarded), sum(2 * (24 if phase == "a" else 12) for phase in guarded)


def summarize_context(
    context: dict[str, Any], index: int, work: dict[str, Any], cells: list[dict[str, Any]]
) -> dict[str, Any]:
    raw, family = context["legacy_train_context"], context["family"]
    supply = raw.get("boundaries", raw.get("opportunities", []))
    shared = max((count(item["retained_bytes"], "FIFO bytes") for item in supply), default=0)
    retained = count(work["retained_array_bytes_before_copies"], "group retained bytes")
    require(shared <= retained, "resource shared FIFO exceeds group retained storage")
    attempts, examples = _guard_work(context)
    same_json(attempts, work["guarded_attempts"], "resource context guarded attempts")
    same_json(2 * attempts, work["guard_evaluations"], "resource context guard predictions")
    same_json(examples, work["guard_examples"], "resource context guard examples")
    for key in (
        "wake_updates",
        "applied_replay_updates",
        "rejected_executed_replay_updates",
        "executed_optimizer_updates",
    ):
        same_json(
            sum(count(cell[key], key) for cell in cells), work[key], f"resource context {key}"
        )
    pointer = context_pointer(index)
    return {
        "family": family,
        "seed": context["seed"],
        "json_pointer": pointer,
        "complete_context_identity": canonical_body_identity(context),
        "cells": len(cells),
        "shared_fifo_bytes": shared,
        "shared_supply_observations": len(supply),
        "shared_selected_presentations": sum(len(item.get("selected_ids", [])) for item in supply),
        "shared_distinct_selected_examples": len(
            {sample for item in supply for sample in item.get("selected_ids", [])}
        ),
        "retained_array_bytes_before_copies": retained,
        "owned_retained_array_bytes_before_copies": retained - shared,
        "storage_scope": "one_group_FIFO_plus_original_owned_model_arrays_before_checkpoint_copies_not_RSS",
        "guarded_attempts": attempts,
        "guard_prediction_calls": 2 * attempts,
        "guard_examples": examples,
        "maximum_transient_width": work["maximum_transient_width"],
        "supplemental_proof_count": len(context["supplemental_guards"]),
        "history_scope": "original_checkpoints_and_recorded_epoch_transaction_points_only_not_continuous_profiling",
    }


def _point(width: Any, pointer: str, stage: str, *, parameters: Any = None) -> dict[str, Any]:
    width = count(width, "history width")
    require(width > 0, "resource history width is zero")
    derived = 4 * width + 1
    if parameters is not None:
        same_json(
            count(parameters, "history parameters"),
            derived,
            "resource two-input one-output parameter geometry",
        )
    return {
        "stage": stage,
        "width": width,
        "parameters": derived,
        "status": "derived" if parameters is None else "measured",
        "source_pointer": pointer,
        "reason": "parameters=4*recorded_width+1_for_original_two_input_one_output_head"
        if parameters is None
        else "original_recorded_width_and_parameter_count",
    }


def capacity_history(
    cell: dict[str, Any], cell_index: int, context: dict[str, Any], context_index: int
) -> list[dict[str, Any]]:
    base = f"/cost_references/projection/cells/{cell_index}/checkpoints"
    points = [
        _point(item["width"], f"{base}/{i}", item["stage"], parameters=item["parameter_count"])
        for i, item in enumerate(cell["checkpoints"])
    ]
    raw, arm = context["legacy_train_context"], cell["arm"]
    base = context_pointer(context_index)
    for i, opportunity in enumerate(raw.get("opportunities", [])):
        prefix = f"{base}/legacy_train_context/opportunities/{i}"
        for j, wake in enumerate(opportunity.get("wake", [])):
            if wake["name"] == arm:
                points.append(
                    _point(
                        wake["width"],
                        f"{prefix}/wake/{j}",
                        "after_wake",
                        parameters=wake["parameters"],
                    )
                )
        if arm in opportunity.get("after_epoch_widths", {}):
            points.append(
                _point(
                    opportunity["after_epoch_widths"][arm],
                    f"{prefix}/after_epoch_widths/{arm}",
                    "after_epoch",
                )
            )
        for j, decision in enumerate(opportunity["decisions"]):
            if decision.get("name") != arm:
                continue
            event = decision["event"]
            event_pointer = f"{prefix}/decisions/{j}/event"
            for field in ("before_width", "proposed_width", "final_width"):
                points.append(_point(event[field], f"{event_pointer}/{field}", field))
            transient = event["before_width"] + len(event["changes"]["proposed_split_pairs"])
            points.append(_point(transient, event_pointer, "transient_before_prune_or_rollback"))
    sleep = cell["method_facts"].get("sleep")
    if sleep is not None:
        prefix = f"/cost_references/projection/cells/{cell_index}/method_facts/sleep"
        for field in ("width_before", "proposed_width", "width_after", "transient_peak_width"):
            points.append(_point(sleep[field], f"{prefix}/{field}", field))
    return points


def applied_replay_ids(cell: dict[str, Any], context: dict[str, Any]) -> set[str]:
    family, arm, raw = cell["family"], cell["arm"], context["legacy_train_context"]
    samples: set[str] = set()
    for item in raw.get("boundaries", raw.get("opportunities", [])):
        if family == "replay":
            samples.update(item["applied_ids_by_method"][arm])
        elif family == "schedule":
            for decision in item["decisions"]:
                for kind, ids in decision["applied_ids_by_method"].items():
                    if arm == f"{kind}_{decision['policy']}":
                        samples.update(ids)
        elif family == "combined":
            samples.update(item["applied_control_replay_ids"].get(arm, []))
            for decision in item["decisions"]:
                if decision["name"] == arm:
                    samples.update(decision["applied_replay_ids"])
    return samples
