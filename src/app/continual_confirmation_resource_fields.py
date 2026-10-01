"""Give every original arm resource field a unit, scope, status and pointer.

Inputs are decoded bound cost rows/contexts. Output is a resource ledger row;
raw proofs stay referenced. No outcome ranking, IO, runtime measurement,
source/execution proof or attribution of shared time/RSS to arms.
"""

from __future__ import annotations

from typing import Any

from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_resource_contexts import (
    applied_replay_ids,
    capacity_history,
    context_pointer,
    count,
)


def _field(
    value: Any,
    unit: str,
    pointers: list[str],
    *,
    status: str = "derived",
    scope: str = "individual_arm_original_training",
    reason: str = "derived_from_original_recorded_cost_facts",
) -> dict[str, Any]:
    return {
        "value": value,
        "unit": unit,
        "scope": scope,
        "status": status,
        "reason": reason,
        "source_pointers": pointers,
    }


def _missing(unit: str, pointers: list[str], reason: str) -> dict[str, Any]:
    return _field(None, unit, pointers, status="unmeasured", reason=reason)


def _raw(method: dict[str, Any], names: tuple[str, ...], base: str) -> tuple[int, list[str]]:
    present = [name for name in names if name in method]
    require(len(present) == 1, "resource raw counter missing or ambiguous: " + "/".join(names))
    name = present[0]
    return count(method[name], name), [f"{base}/{name}"]


def _work_fields(cell: dict[str, Any], index: int) -> dict[str, Any]:
    method = cell["method_facts"]
    root = f"/cost_references/projection/cells/{index}"
    base = root + "/method_facts"
    wake, wake_ptr = _raw(
        method, ("latent_iterations", "latent_inference_loops", "wake_inference_loops"), base
    )
    examples, example_ptr = _raw(
        method, ("example_inference_iterations", "wake_example_inference_iterations"), base
    )
    presentations, presentation_ptr = _raw(
        method, ("train_presentations", "wake_presentations"), base
    )
    applied, applied_ptr = _raw(method, ("replay_updates", "applied_replay_updates"), base)
    same_json(applied, cell["applied_replay_updates"], "resource raw applied replay")
    rejected = count(cell["rejected_executed_replay_updates"], "rejected replay")
    replay_loops = count(
        method.get("replay_inference_loops", method.get("applied_replay_inference_loops", 0)),
        "applied replay loops",
    )
    rejected_loops = count(
        method.get("rejected_replay_inference_loops", 0), "rejected replay loops"
    )
    replay_rows = count(
        method.get("replay_presentations", method.get("applied_replay_presentations", 0)),
        "applied replay presentations",
    )
    fields = {
        "wake_updates": _field(
            count(cell["wake_updates"], "wake updates"),
            "optimizer_updates",
            [root + "/wake_updates", base + "/wake_updates"],
        ),
        "wake_presentations": _field(presentations, "example_presentations", presentation_ptr),
        "wake_latent_iterations": _field(wake, "latent_iterations_per_batch", wake_ptr),
        "wake_example_iterations": _field(examples, "example_latent_iterations", example_ptr),
        "applied_replay_updates": _field(applied, "optimizer_updates", applied_ptr),
        "rejected_replay_updates": _field(
            rejected,
            "optimizer_updates_executed_then_rolled_back",
            [root + "/rejected_executed_replay_updates"],
        ),
        "executed_optimizer_updates": _field(
            count(cell["executed_optimizer_updates"], "executed updates"),
            "optimizer_updates_including_rollback",
            [
                root + "/wake_updates",
                root + "/applied_replay_updates",
                root + "/rejected_executed_replay_updates",
            ],
        ),
        "applied_replay_presentations": _field(
            replay_rows,
            "example_presentations",
            [
                base
                + (
                    "/replay_presentations"
                    if "replay_presentations" in method
                    else "/applied_replay_presentations"
                )
            ]
            if "replay_presentations" in method or "applied_replay_presentations" in method
            else applied_ptr,
            reason="recorded_presentations_or_declared_zero_replay_protocol",
        ),
        "applied_replay_latent_iterations": _field(
            replay_loops,
            "latent_iterations_per_batch",
            [base],
            reason="original_replay_loop_counter_or_declared_zero_replay_protocol",
        ),
        "rejected_replay_latent_iterations": _field(
            rejected_loops,
            "latent_iterations_per_batch",
            [base],
            reason="original_rejected_loop_counter_or_declared_no_rejection_protocol",
        ),
        "optimizer_latent_iterations": _field(
            wake + replay_loops + rejected_loops,
            "latent_iterations_per_batch",
            [base],
            reason="wake_plus_applied_plus_rejected_replay_loops_excludes_prediction_guard_and_selection_work",
        ),
    }
    same_json(
        fields["executed_optimizer_updates"]["value"],
        fields["wake_updates"]["value"] + applied + rejected,
        "resource executed work sum",
    )
    return fields


def _capacity_fields(
    cell: dict[str, Any], index: int, context: dict[str, Any], context_index: int
) -> dict[str, Any]:
    history = capacity_history(cell, index, context, context_index)
    method, checkpoints = cell["method_facts"], cell["checkpoints"]
    require(
        [item["stage"] for item in checkpoints] == ["initial", "after_a", "after_b"],
        "resource checkpoint order differs",
    )
    peak = count(
        method.get("parameters_peak", max(point["parameters"] for point in history)),
        "peak parameters",
    )
    same_json(
        peak,
        max(point["parameters"] for point in history),
        "resource complete recorded capacity peak",
    )
    fields = {
        f"parameters_{item['stage']}": _field(
            count(item["parameter_count"], "checkpoint parameters"),
            "trainable_scalar_parameters",
            [f"/cost_references/projection/cells/{index}/checkpoints/{i}/parameter_count"],
            status="measured",
            reason="original_actual_checkpoint_capacity_bound_to_complete_state",
        )
        for i, item in enumerate(checkpoints)
    }
    fields["parameters_peak"] = _field(
        peak,
        "trainable_scalar_parameters",
        [f"/cost_references/projection/cells/{index}/method_facts", context_pointer(context_index)],
        reason="maximum_recorded_checkpoint_epoch_and_transient_split_before_prune_or_rollback_capacity",
    )
    fields["parameter_history"] = _field(
        history,
        "trainable_scalar_parameters_at_named_points",
        [context_pointer(context_index), f"/cost_references/projection/cells/{index}/checkpoints"],
        scope="individual_arm_recorded_capacity_points",
        reason="recorded_checkpoint_epoch_transaction_points_not_continuous_time_or_every_optimizer_update",
    )
    return fields


def cell_fields(
    cell: dict[str, Any], index: int, context: dict[str, Any], context_index: int
) -> dict[str, Any]:
    method = cell["method_facts"]
    base = f"/cost_references/projection/cells/{index}/method_facts"
    fields = {**_work_fields(cell, index), **_capacity_fields(cell, index, context, context_index)}
    attempts = method.get(
        "own_sleep_attempts", method.get("sleep_attempts", int(method.get("sleep") is not None))
    )
    guards = method.get(
        "guard_evaluations",
        method.get("sleep", {}).get("guard_evaluations", 0)
        if method.get("sleep") is not None
        else 0,
    )
    fields["sleep_attempts"] = _field(count(attempts, "sleep attempts"), "sleep_attempts", [base])
    fields["guard_prediction_calls"] = _field(
        count(guards, "guard calls"),
        "prediction_calls",
        [base, context_pointer(context_index)],
        reason="original_guard_counter_or_declared_no_guard_protocol",
    )
    fields["distinct_applied_replay_examples"] = _field(
        len(applied_replay_ids(cell, context)),
        "distinct_example_ids",
        [context_pointer(context_index)],
        reason="union_of_original_committed_replay_ids_per_arm_not_available_FIFO_rows",
    )
    retained = method.get("retained_array_bytes")
    if retained is None and method.get("retention") is not None:
        retained = method["retention"]["retained_bytes"]
    fields["owned_replay_array_bytes"] = (
        _missing(
            "array_bytes",
            [base],
            "per_arm_owned_retention_not_present_in_this_raw_method_projection_use_exact_group_work_and_original_checkpoint_proof",
        )
        if retained is None
        else _field(
            count(retained, "owned replay bytes"),
            "array_bytes",
            [base],
            status="measured",
            scope="individual_arm_owned_retention_before_copies_excludes_shared_FIFO",
            reason="original_owned_retention_field_not_process_RSS",
        )
    )
    for name, unit in (
        ("wall_time_seconds", "seconds"),
        ("peak_process_rss_bytes", "RSS_bytes"),
        ("sleep_guard_duration_seconds", "seconds"),
    ):
        fields[name] = _missing(
            unit,
            [base],
            "no_original_isolated_per_arm_measurement_whole_process_audits_have_a_different_scope",
        )
    return fields
