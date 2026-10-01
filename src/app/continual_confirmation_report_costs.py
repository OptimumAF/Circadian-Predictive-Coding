"""Project original confirmation training costs without training or scoring.

The public input is the entire pinned training result. Output retains every
family/seed/arm's raw method facts, three checkpoint capacities, shared seed
context and independently verified work totals. File/source/process proof
belongs to the complete reader boundary; no data/model/IO or cost estimation
belongs here. Private seed projection is for development correctness only.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any

from src.app.continual_confirmation_execution import work_summary
from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_manifest import ConfirmationFamily, fixed_confirmation_manifest
from src.app.continual_confirmation_scoring_manifest import fixed_scoring_manifest
from src.app.continual_confirmation_work_validation import verify_confirmation_payload


SCHEMA_ID = "p611_confirmation_original_cost_projection_v1"
CHECKPOINT_STAGES = ("initial", "after_a", "after_b")


def canonical_body_identity(value: Any) -> dict[str, Any]:
    """Match the original pretty ASCII JSON plus LF without another large string."""
    digest = sha256()
    size = 0
    encoder = json.JSONEncoder(indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
    try:
        for fragment in encoder.iterencode(value):
            encoded = fragment.encode("utf-8")
            digest.update(encoded)
            size += len(encoded)
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError("confirmation cost body is malformed/nonfinite") from error
    digest.update(b"\n")
    return {"sha256": digest.hexdigest(), "byte_count": size + 1}


def _count(value: Any, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"confirmation cost {name} must be an exact nonnegative integer")
    return value


def _checkpoint_cost(checkpoint: dict[str, Any], stage: str) -> dict[str, Any]:
    result: dict[str, Any] = {"stage": stage}
    for name in ("width", "parameter_count"):
        result[name] = _count(checkpoint[name], name, positive=True)
    model_type = checkpoint["model_type"]
    require(type(model_type) is str and bool(model_type.strip()), "cost model type is malformed")
    result["model_type"] = model_type
    for name in ("state_sha256", "parameter_sha256"):
        value = checkpoint[name]
        require(
            type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            f"cost checkpoint {name} is malformed",
        )
        result[name] = value
    return result


def _method_cost(
    row: dict[str, Any], family: ConfirmationFamily, method: dict[str, Any], arm: str
) -> dict[str, Any]:
    simple_replay = family.name in {"gating", "replay", "sleep"}
    wake = _count(method["wake_updates"], "wake updates")
    applied = _count(
        method["replay_updates" if simple_replay else "applied_replay_updates"],
        "applied replay updates",
    )
    rejected = _count(method.get("rejected_executed_replay_updates", 0), "rejected executed replay")
    return {
        "family": family.name,
        "seed": row["seed"],
        "arm": arm,
        "wake_updates": wake,
        "applied_replay_updates": applied,
        "rejected_executed_replay_updates": rejected,
        "executed_optimizer_updates": wake + applied + rejected,
        "checkpoints": [_checkpoint_cost(row[stage][arm], stage) for stage in CHECKPOINT_STAGES],
        # Why raw facts: inference, presentations, guards, memory and inactive
        # events have different meanings across families; do not normalize away.
        "method_facts": deepcopy(method),
    }


def _project_seed_costs(row: dict[str, Any], family: ConfirmationFamily) -> dict[str, Any]:
    """Development seam; this projection alone establishes no source provenance."""
    try:
        require(row["family"] == family.name, "cost family differs")
        require(type(row["seed"]) is int and row["seed"] in family.seeds, "cost seed differs")
        raw = row["legacy_train_facts"]
        collection = "arms" if family.name == "sleep" else "methods"
        name_field = "method" if family.name in {"gating", "replay"} else "name"
        methods = raw[collection]
        names = [method[name_field] for method in methods]
        require(
            len(names) == len(family.arms) and set(names) == set(family.arms),
            "cost methods are incomplete, duplicate or unknown",
        )
        for stage in CHECKPOINT_STAGES:
            require(set(row[stage]) == set(family.arms), "cost checkpoint arm scope differs")
        indexed = {method[name_field]: method for method in methods}
        cells = [_method_cost(row, family, indexed[arm], arm) for arm in family.arms]
        context = {
            "family": family.name,
            "seed": row["seed"],
            "legacy_train_context": deepcopy({k: v for k, v in raw.items() if k != collection}),
            "supplemental_guards": deepcopy(row["supplemental_guards"]),
            "training_roles": deepcopy(row["roles"]),
        }
        # Finite JSON and no mutable aliases, including nested raw event facts.
        canonical_body_identity({"cells": cells, "seed_context": context})
        return {"cells": cells, "seed_context": context}
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("confirmation cost projection has malformed raw facts") from error


def project_confirmation_costs(payload: dict[str, Any]) -> dict[str, Any]:
    """Bind the whole original result before deriving every scheduled raw cost."""
    manifest = fixed_confirmation_manifest()
    reference = fixed_scoring_manifest().training_bundles[0]
    identity = {"sha256": reference.result_sha256, "byte_count": reference.result_bytes}
    same_json(canonical_body_identity(payload), identity, "cost complete original train body")
    work = verify_confirmation_payload(payload, manifest)
    families = {family.name: family for family in manifest.families}
    projected = [
        _project_seed_costs(row, families[row["family"]]) for row in payload["seed_results"]
    ]
    cells = [cell for seed in projected for cell in seed["cells"]]
    totals = work_summary(work)
    same_json(len(cells), totals["totals"]["cells"], "cost complete cell join")
    for name in ("wake_updates", "applied_replay_updates", "rejected_executed_replay_updates"):
        same_json(sum(cell[name] for cell in cells), totals["totals"][name], f"cost {name} total")
    same_json(
        canonical_body_identity(payload), identity, "cost train body changed during projection"
    )
    return {
        "schema_id": SCHEMA_ID,
        "training_result": identity,
        "validation_scope": "complete_bound_training_facts_only_not_file_or_execution_proof",
        "cells": cells,
        "seed_contexts": [seed["seed_context"] for seed in projected],
        "work": totals,
        "cost_interpretation": {
            "optimizer_updates": "wake_plus_applied_plus_rejected_executed_replay",
            "raw_method_facts": "original_fields_preserved_without_cross_family_unit_conversion",
            "checkpoint_capacity": "initial_after_a_after_b_actual_width_and_parameter_count",
            "shared_retained_storage": "seed_group_array_bytes_before_copies_not_per_arm_rss",
            "wall_and_rss": "whole_run_observations_only_no_invented_per_arm_allocation",
            "deterministic_repeat": "reproducibility_check_not_an_additional_seed_replication",
        },
        "final_release_authorized": False,
        "outer_selection_scored": False,
    }
