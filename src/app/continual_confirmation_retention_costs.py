"""Recover complete original per-arm retention costs without resource estimates.

Inputs are the entire pinned training result and completed resource inventory.
Output includes every owned checkpoint proof, shared context, original gap and
independently reconciled stage/group totals. No IO, model/data construction,
training, scoring, resource measurement, selection or original parent claim.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

from src.app.continual_confirmation_execution import work_summary
from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_manifest import ConfirmationFamily, fixed_confirmation_manifest
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_confirmation_retention_checkpoints import checkpoint_group, STAGES
from src.app.continual_confirmation_validation import _verify_seed_envelope
from src.app.continual_confirmation_work_validation import _verify_seed_work


TRAINING_ID = {
    "byte_count": 134554378,
    "sha256": "3d85c60627de63769d0f0fc0bf5ec781c77d50468673dab466d8bbe28089e547",
}
INVENTORY_ID = {
    "byte_count": 15490091,
    "sha256": "46b923ad046a2c980f60a6d2d6fc16f9415c7cfe1787893c2078c2263091821f",
}


def _scope(
    rows: list[dict[str, Any]], families: tuple[ConfirmationFamily, ...], inventory: dict[str, Any]
) -> None:
    expected = [(f.name, seed) for f in families for seed in f.seeds]
    same_json(
        [(r["family"], r["seed"]) for r in rows], expected, "retention complete training scope"
    )
    same_json(
        [(r["family"], r["seed"]) for r in inventory["contexts"]],
        expected,
        "retention complete inventory contexts",
    )
    cells = [(f.name, seed, arm) for f in families for seed in f.seeds for arm in f.arms]
    same_json(
        [(r["family"], r["seed"], r["arm"]) for r in inventory["rows"]],
        cells,
        "retention complete inventory arm scope",
    )


def _join_owned(
    rows: list[dict[str, Any]], contexts: list[dict[str, Any]], inventory: dict[str, Any]
) -> None:
    for point, original in zip(rows, inventory["rows"], strict=True):
        field = original["fields"]["owned_replay_array_bytes"]
        require(
            field["status"] in {"measured", "derived", "unmeasured"}, "owned field status differs"
        )
        if field["value"] is not None:
            same_json(
                field["value"],
                point["checkpoints"]["after_b"]["owned_array_bytes"],
                "original measured owned field",
            )
        else:
            require(field["status"] == "unmeasured", "missing owned field is not explicit")
        point["original_inventory_field"] = deepcopy(field)
    for context, original in zip(contexts, inventory["contexts"], strict=True):
        final = context["stages"]["after_b"]
        for derived, field in (
            ("owned_array_bytes", "owned_retained_array_bytes_before_copies"),
            ("shared_fifo_array_bytes", "shared_fifo_bytes"),
            ("owned_plus_shared_array_bytes_before_copies", "retained_array_bytes_before_copies"),
        ):
            same_json(
                final[derived], original[field], "original inventory group owned/shared storage"
            )


def _derive_retention_costs(
    source_rows: list[dict[str, Any]],
    families: tuple[ConfirmationFamily, ...],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    """Private complete development seam; grants no original reader authority."""
    try:
        _scope(source_rows, families, inventory)
        family_map = {family.name: family for family in families}
        rows: list[dict[str, Any]] = []
        contexts: list[dict[str, Any]] = []
        work = []
        for index, source in enumerate(source_rows):
            family = family_map[source["family"]]
            _verify_seed_envelope(source, family)
            observed = _verify_seed_work(source, family)
            work.append(observed)
            checkpoints, context = checkpoint_group(source, index, family.arms)
            same_json(
                context["stages"]["after_b"]["owned_plus_shared_array_bytes_before_copies"],
                observed.retained_array_bytes_before_copies,
                "original independently derived group work",
            )
            contexts.append(context)
            for arm in family.arms:
                rows.append(
                    {
                        "family": family.name,
                        "seed": source["seed"],
                        "arm": arm,
                        "roles_json_pointer": context["roles_json_pointer"],
                        "checkpoints": checkpoints[arm],
                    }
                )
        same_json(
            work_summary(tuple(work)), inventory["work"], "complete original retention/work ledger"
        )
        _join_owned(rows, contexts, inventory)
        totals = {
            stage: {
                field: sum(context["stages"][stage][field] for context in contexts)
                for field in (
                    "owned_array_bytes",
                    "shared_fifo_array_bytes",
                    "owned_plus_shared_array_bytes_before_copies",
                )
            }
            for stage in STAGES
        }
        statuses = Counter(
            point["retention_status"] for row in rows for point in row["checkpoints"].values()
        )
        return {
            "schema_id": "p610_complete_original_retention_costs_v1",
            "scope": {
                "storage": "owned_model_arrays_plus_one_shared_FIFO_per_context_at_each_named_stage_before_copies_not_process_RSS",
                "stages_are_separate_checkpoints_not_additive_live_memory": True,
                "checkpoint_copy_bytes_measured": False,
                "per_arm_RSS_measured": False,
                "null_view_policy": "preserve_null_derive_zero_only_from_closed_original_baseline_or_explicit_empty_disabled_state",
                "sample_values_reopened": False,
            },
            "coverage": {
                "cells": len(rows),
                "checkpoints": 3 * len(rows),
                "contexts": len(contexts),
                "stage_contexts": 3 * len(contexts),
                "projection_gaps_resolved": sum(
                    row["original_inventory_field"]["value"] is None for row in rows
                ),
                "retention_status_counts": dict(sorted(statuses.items())),
                "owned_array_pairs": sum(
                    len(point["owned_array_fingerprints"])
                    for row in rows
                    for point in row["checkpoints"].values()
                ),
            },
            "rows": rows,
            "contexts": contexts,
            "stage_totals": totals,
            "work": deepcopy(inventory["work"]),
            "original_P6_10_acceptance_complete": False,
            "new_measurement_training_or_final_access": False,
        }
    except (KeyError, TypeError, IndexError, AttributeError, StopIteration) as error:
        raise ValueError(
            "retention original checkpoint/inventory proof is malformed or incomplete"
        ) from error


def project_retention_costs(training: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    """Require entire original bodies before exposing their complete projection."""
    same_json(
        canonical_body_identity(training), TRAINING_ID, "whole original retention training body"
    )
    same_json(
        canonical_body_identity(inventory), INVENTORY_ID, "whole completed resource inventory body"
    )
    result = _derive_retention_costs(
        training["seed_results"], fixed_confirmation_manifest().families, inventory
    )
    result["original_training_result_identity"] = dict(TRAINING_ID)
    result["original_resource_inventory_identity"] = dict(INVENTORY_ID)
    require(
        result["coverage"]["cells"] == 560
        and result["coverage"]["checkpoints"] == 1680
        and result["coverage"]["contexts"] == 60
        and result["coverage"]["projection_gaps_resolved"] == 300,
        "retention complete original coverage differs",
    )
    return result
