"""Extract owned retention proofs and separately scoped shared FIFO facts.

Inputs are independently validated original checkpoint/context JSON. Outputs
preserve original nullable views, state fields, array fingerprints and exact
source pointers. No IO, arrays of sample values, models, data or profiling.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.app.continual_confirmation_json import (
    canonical_array,
    canonical_dataclass,
    canonical_mapping,
    hash_value,
    integer,
    require,
    same_json,
)
from src.app.continual_confirmation_report_costs import canonical_body_identity


STAGES = ("initial", "after_a", "after_b")
SHARED_FAMILIES = {"replay", "schedule", "combined", "parent"}
_SNAPSHOT = "src.core.circadian_predictive_coding.CircadianNetworkSnapshot"
_REPLAY_ROW = "src.core.circadian_predictive_coding.ReplaySnapshot"


def _owned_fields(point: dict[str, Any], pointer: str) -> dict[str, Any]:
    state = point["state"]
    state_pointer = pointer + "/state"
    if point["clocks"] is not None:
        state = canonical_dataclass(state, _SNAPSHOT)["state"]
        state_pointer += "/fields/state"
    owner = canonical_mapping(state)
    return {
        key: {
            "value": deepcopy(value),
            "json_pointer": f"{state_pointer}/dict/{index}/1",
        }
        for index, (key, value) in enumerate(owner.items())
        if key.startswith("_replay_")
    }


def _array_proofs(fields: dict[str, Any]) -> list[dict[str, Any]]:
    if "_replay_memory" not in fields:
        return []
    memory = fields["_replay_memory"]
    proofs = []
    for index, item in enumerate(memory["value"]["deque"]):
        data = canonical_dataclass(item, _REPLAY_ROW)
        canonical_array(data["input_batch"], (1, 2))
        canonical_array(data["target_batch"], (1, 1))
        base = f"{memory['json_pointer']}/deque/{index}/fields"
        # Why this: exact <f8 geometry is independently checked above. These
        # 16+8 bytes describe stored input/target arrays, excluding overhead.
        proofs.append(
            {
                "deque_index": index,
                "input_array": deepcopy(data["input_batch"]),
                "target_array": deepcopy(data["target_batch"]),
                "input_array_bytes": 16,
                "target_array_bytes": 8,
                "input_json_pointer": base + "/input_batch",
                "target_json_pointer": base + "/target_batch",
            }
        )
    return proofs


def retention_checkpoint(point: dict[str, Any], pointer: str) -> dict[str, Any]:
    fields = _owned_fields(point, pointer)
    arrays = _array_proofs(fields)
    owned = sum(row["input_array_bytes"] + row["target_array_bytes"] for row in arrays)
    view = point["retention"]
    if point["clocks"] is None:
        require(view is None and not fields, "baseline owned retention is not absent")
        status, reason = "not_applicable", "closed_baseline_state_has_no_owned_replay_buffer"
    elif view is None:
        require(
            fields["_replay_memory"]["value"] == {"deque": [], "maxlen": 0},
            "disabled retention has owned memory",
        )
        status, reason = "disabled", "original_circadian_zero_memory_state_no_retention_budget"
    else:
        require(
            integer(view["retained_bytes"], "owned retained bytes") == owned
            and integer(view["example_count"], "owned retained examples") == len(arrays),
            "owned array geometry differs from original retention view",
        )
        status = "configured_empty" if not arrays else "retained"
        reason = "original_configured_empty_view" if not arrays else "original_owned_array_proof"
    return {
        "checkpoint_json_pointer": pointer,
        "checkpoint_identity": canonical_body_identity(point),
        "model_type": point["model_type"],
        "state_sha256": point["state_sha256"],
        "parameter_sha256": point["parameter_sha256"],
        "retention_json_pointer": pointer + "/retention",
        "retention": deepcopy(view),
        "retention_status": status,
        "retention_reason": reason,
        "owned_array_bytes": owned,
        "owned_array_bytes_status": "derived",
        "owned_array_bytes_unit": "array_bytes",
        "owned_array_bytes_scope": "individual_owned_input_target_arrays_at_named_checkpoint_before_copies_excludes_shared_FIFO_and_process_RSS",
        "owned_state_fields": fields,
        "owned_array_fingerprints": arrays,
        "sample_identity_scope": "original_sorted_content_ID_view_and_ordered_array_fingerprints_no_new_sample_values_or_content_ID_rehash",
    }


def shared_fifo(row: dict[str, Any], stage: str, index: int) -> dict[str, Any]:
    family = row["family"]
    if family not in SHARED_FAMILIES or stage == "initial":
        return {
            "sample_order_ids": [],
            "example_count": 0,
            "array_bytes": 0,
            "measurement_status": "derived",
            "reason": "no_shared_FIFO_in_original_family"
            if family not in SHARED_FAMILIES
            else "original_shared_FIFO_constructor_empty_before_first_wake",
            "json_pointer": f"/seed_results/{index}/{stage}",
            "protocol_source_references": []
            if family not in SHARED_FAMILIES
            else [
                "src/core/shared_replay_schedule.py:SharedReplayBuffer.__init__",
                "src/app/continual_confirmation_training.py:_train_a",
            ],
        }
    raw = row["legacy_train_facts"]
    phase = "a" if stage == "after_a" else "b"
    if family == "replay":
        position = max(i for i, item in enumerate(raw["boundaries"]) if item["phase"] == phase)
        collection = "boundaries"
    else:
        position, collection = (11 if phase == "a" else 23), "opportunities"
    source = raw[collection][position]
    require(source["phase"] == phase, "retention shared stage differs")
    ids = source["retained_order_ids"]
    require(type(ids) is list and len(set(ids)) == len(ids) == 8, "shared retained IDs differ")
    for sample in ids:
        hash_value(sample, "shared retained ID")
    require(
        integer(source["retained_examples"], "shared examples") == len(ids)
        and integer(source["retained_bytes"], "shared bytes") == 24 * len(ids),
        "shared FIFO count/array bytes differ",
    )
    return {
        "sample_order_ids": deepcopy(ids),
        "example_count": len(ids),
        "array_bytes": source["retained_bytes"],
        "measurement_status": "measured",
        "reason": "original_stage_last_shared_FIFO_record",
        "json_pointer": f"/seed_results/{index}/legacy_train_facts/{collection}/{position}",
        "protocol_source_references": [],
    }


def checkpoint_group(
    row: dict[str, Any],
    index: int,
    arms: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoints = {
        arm: {
            stage: retention_checkpoint(row[stage][arm], f"/seed_results/{index}/{stage}/{arm}")
            for stage in STAGES
        }
        for arm in arms
    }
    stages = {}
    for stage in STAGES:
        shared = shared_fifo(row, stage, index)
        for arm in arms:
            view = checkpoints[arm][stage]["retention"]
            if view is not None:
                same_json(
                    view["sample_ids"],
                    sorted(shared["sample_order_ids"]),
                    "owned/shared stage FIFO IDs",
                )
        owned = sum(checkpoints[arm][stage]["owned_array_bytes"] for arm in arms)
        stages[stage] = {
            "owned_array_bytes": owned,
            "shared_fifo_array_bytes": shared["array_bytes"],
            "owned_plus_shared_array_bytes_before_copies": owned + shared["array_bytes"],
            "shared_fifo": shared,
        }
    return checkpoints, {
        "family": row["family"],
        "seed": row["seed"],
        "training_context_json_pointer": f"/seed_results/{index}",
        "roles_json_pointer": f"/seed_results/{index}/roles",
        "roles": deepcopy(row["roles"]),
        "stages": stages,
    }
