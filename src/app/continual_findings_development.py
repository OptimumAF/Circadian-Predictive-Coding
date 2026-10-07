"""Project every fixed development cell and declared pair without selection.

Inputs are complete stored scope/development/preflight records. Outputs retain
all inputs plus ordered outer-selection cells and raw left-minus-right pairs.
This pure layer grants no current-source, official-reader or execution proof;
IO, experiment selection, new statistics and confirmation synthesis are external.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from math import isclose, isfinite
from typing import Any

from src.app.continual_confirmation_execution import digest_json
from src.app.continual_confirmation_json import hash_value, require, same_json
from src.app.continual_confirmation_manifest import fixed_confirmation_manifest
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.core.continual_metrics import TwoTaskAccuracy


SCOPE_ID = {
    "byte_count": 119067,
    "sha256": "622feead54f155521928341151c8496c23c02a54b355b1b3b1e0f68b76c5772f",
}
CATALOG_ID = {
    "byte_count": 5714,
    "sha256": "ac4758249585f877e0bb1a2f2c7394a4cf0d4b7b6157e6aab5618e1d60fa35dc",
}
ENDPOINTS = ("a_after_a", "a_after_b", "b_after_b")
PAIR_FIELDS = ENDPOINTS + ("final_mean_task_accuracy", "signed_forgetting_a")
FAMILIES = ("gating", "replay", "sleep", "schedule", "combined", "parent")
PREFIXES = {"gating": "gating-pilot", "replay": "replay-factor-pilot"} | {
    name: name + "-factor-development" for name in FAMILIES[2:]
}


def _score_fields(family: str) -> tuple[str, ...]:
    return (
        ENDPOINTS
        + ("final_mean_task_accuracy",)
        + (
            ("signed_forgetting",)
            if family == "gating"
            else ("signed_forgetting_a", "retention_ratio_a")
        )
    )


def _check_scores(scores: dict[str, Any], family: str) -> None:
    same_json(set(scores), set(_score_fields(family)), "development original metric fields")
    for name in ENDPOINTS:
        value = scores[name]
        require(
            type(value) in (int, float) and isfinite(value) and 0 <= value <= 1,
            "development endpoint accuracy fraction",
        )
    values = TwoTaskAccuracy(*(scores[name] for name in ENDPOINTS))
    expected = {
        "final_mean_task_accuracy": values.final_mean_task_accuracy,
        "signed_forgetting"
        if family == "gating"
        else "signed_forgetting_a": values.signed_forgetting_a,
    }
    for name, value in expected.items():
        require(type(scores[name]) in (int, float), "development derived score type")
        # Why this: retain the gating pilot's original 1e-12 arithmetic gate;
        # stored numbers are never rounded, clipped or replaced by this check.
        if family == "gating":
            require(isclose(scores[name], value, rel_tol=0, abs_tol=1e-12), "gating arithmetic")
        else:
            same_json(scores[name], value, "development original derived metric " + name)
    if family != "gating":
        same_json(
            scores["retention_ratio_a"], values.retention_ratio_a, "development original ratio"
        )


def _bundle_parts(bundle: dict[str, Any], reference: dict[str, Any] | None) -> dict[str, Any]:
    same_json(set(bundle["parts"]), {"request", "result", "audit"}, "development complete parts")
    same_json(set(bundle["files"]), {"request", "result", "audit"}, "development file identities")
    for name, part in bundle["parts"].items():
        same_json(canonical_body_identity(part), bundle["files"][name], "development whole " + name)
        if reference is not None:
            same_json(
                bundle["files"][name]["sha256"],
                reference["file_sha256"][name],
                "original scope part",
            )
    parts = bundle["parts"]
    require(parts["result"]["final_released"] is False, "development final evaluation seal")
    require(parts["audit"]["status"] == "completed", "development completed audit")
    for name in ("request", "result"):
        same_json(
            parts["audit"][name + "_sha256"],
            bundle["files"][name]["sha256"],
            "development audit binding",
        )
    if reference is not None:
        same_json(
            bundle["files"]["result"]["byte_count"],
            reference["result_bytes"],
            "whole development bytes",
        )
        same_json(
            parts["request"]["source_sha256"],
            reference["source_sha256"],
            "development declared source scope",
        )
    return parts


def _scores(result: dict[str, Any], family: str) -> list[dict[str, Any]]:
    if family in ("gating", "replay"):
        return [
            {
                "seed": seed["seed"],
                "arms": [{"name": arm["method"], **arm["development"]} for arm in seed["methods"]],
            }
            for seed in result["seed_results"]
        ]
    require(result["outer_selection_scored"] is True, "development outer-selection seal")
    return result["scored_seeds"]


def _check_roles(result: dict[str, Any], family: dict[str, Any]) -> None:
    train = result if family["name"] in ("gating", "replay") else result["train_facts"]
    same_json(
        [row["seed"] for row in train["seed_results"]],
        family["development_seeds"],
        "development original training seeds",
    )
    counts = {
        "a_train": 72,
        "a_inner_guard": 24,
        "a_outer_selection": 24,
        "b_train": 36,
        "b_inner_guard": 12,
        "b_outer_selection": 12,
    }
    for row in train["seed_results"]:
        prefix = "source_role_" if family["name"] in ("gating", "replay") else "role_"
        same_json(row[prefix + "counts"], counts, "development six original role counts")
        same_json(
            set(row[prefix + "hashes"]), set(counts), "development six original role identities"
        )
        for value in row[prefix + "hashes"].values():
            hash_value(value, "development original role identity")


def _project_seed(
    family: dict[str, Any], seed: dict[str, Any], index: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    name = family["name"]
    same_json([arm["name"] for arm in seed["arms"]], family["arms"], "development ordered arms")
    cells, by_name = [], {}
    for arm_index, arm in enumerate(seed["arms"]):
        scores = {field: arm[field] for field in _score_fields(name)}
        _check_scores(scores, name)
        by_name[arm["name"]] = scores
        pointer = (
            f"/seed_results/{index}/methods/{arm_index}/development"
            if name in ("gating", "replay")
            else f"/scored_seeds/{index}/arms/{arm_index}"
        )
        cells.append(
            {
                "family": name,
                "seed": seed["seed"],
                "arm": arm["name"],
                "scores": deepcopy(scores),
                "result_pointer": pointer,
            }
        )
    pairs = []
    for pair_index, (left, right) in enumerate(family["contrasts"]):
        fields = ENDPOINTS + (
            "final_mean_task_accuracy",
            "signed_forgetting" if name == "gating" else "signed_forgetting_a",
        )
        differences = {field: by_name[left][field] - by_name[right][field] for field in fields}
        stored = name not in ("gating", "replay")
        if stored:
            original = seed["contrasts"][pair_index]
            same_json(
                original,
                {"left": left, "right": right, **differences},
                "development complete stored pair",
            )
        pairs.append(
            {
                "family": name,
                "seed": seed["seed"],
                "left": left,
                "right": right,
                "differences": differences,
                "origin": "original_stored_contrast"
                if stored
                else "projection_of_original_cells_and_prospective_pair",
                "result_pointer": f"/scored_seeds/{index}/contrasts/{pair_index}"
                if stored
                else None,
            }
        )
    if name not in ("gating", "replay"):
        same_json(
            len(seed["contrasts"]), len(family["contrasts"]), "development complete pair count"
        )
    return cells, pairs


def _project_development_ledger(inputs: dict[str, Any]) -> dict[str, Any]:
    same_json(
        set(inputs),
        {
            "schema_id",
            "input_catalog",
            "original_scope",
            "scope_files",
            "source_sha256",
            "source_map_sha256",
            "development_bundles",
            "preflight_bundles",
            "validation_facts",
        },
        "complete development input fields",
    )
    same_json(
        inputs["schema_id"],
        "p612_complete_current_development_inputs_v1",
        "development input schema",
    )
    same_json(
        digest_json(inputs["source_sha256"]),
        inputs["source_map_sha256"],
        "complete development declared source map",
    )
    same_json(
        inputs["validation_facts"],
        {
            "unchanged_development_bundle_validations": 12,
            "unchanged_preflight_result_validations": 8,
            "complete_bundle_files": 60,
            "current_before_and_after_byte_source_bindings": True,
            "new_training_scoring_or_final_access": False,
            "fresh_confirmation_reader_authority": False,
        },
        "development validation declaration",
    )
    scope = inputs["original_scope"]
    manifest = json.loads(json.dumps(asdict(fixed_confirmation_manifest()), allow_nan=False))
    same_json(scope["manifest"], manifest, "complete original prospective manifest")
    development, preflights = inputs["development_bundles"], inputs["preflight_bundles"]
    same_json(
        [(b["family"], b["directory"]) for b in development],
        [
            (name, f"artifacts/runs/p63-{PREFIXES[name]}{suffix}")
            for name in FAMILIES
            for suffix in ("", "-repeat")
        ],
        "all ordered development bundles",
    )
    same_json(
        [(b["family"], b["directory"]) for b in preflights],
        [
            (name, f"artifacts/runs/p63-{name}-factor-preflight{suffix}")
            for name in FAMILIES[2:]
            for suffix in ("", "-repeat")
        ],
        "all ordered preflight bundles",
    )
    for bundle in preflights:
        parts = _bundle_parts(bundle, None)
        reference = next(
            row
            for row in inputs["input_catalog"]["preflight_bundles"]
            if row["directory"] == bundle["directory"]
        )
        same_json(
            bundle["files"],
            reference["files"],
            "development complete declared preflight identities",
        )
        require(parts["result"]["outer_selection_scored"] is False, "preflight unscored seal")
    cells, pairs = [], []
    for family_index, family in enumerate(manifest["families"]):
        references = scope["development_references"][family_index]
        same_json(references["family"], family["name"], "development original reference order")
        first, repeat = development[2 * family_index : 2 * family_index + 2]
        parts = _bundle_parts(first, references["bundles"][0])
        repeated = _bundle_parts(repeat, references["bundles"][1])
        same_json(parts["result"], repeated["result"], "whole deterministic development repetition")
        same_json(
            parts["request"]["manifest"],
            json.loads(family["development_manifest_json"]),
            "development original settings",
        )
        _check_roles(parts["result"], family)
        if family_index >= 2:
            for preflight in preflights[2 * (family_index - 2) : 2 * (family_index - 2) + 2]:
                same_json(
                    parts["result"]["train_facts"],
                    preflight["parts"]["result"],
                    "development complete train reference",
                )
                same_json(
                    parts["result"]["reference_sha256"],
                    preflight["files"]["result"]["sha256"],
                    "development train reference identity",
                )
        scored = _scores(parts["result"], family["name"])
        same_json(
            [seed["seed"] for seed in scored],
            family["development_seeds"],
            "development original ordered seeds",
        )
        for seed_index, seed in enumerate(scored):
            new_cells, new_pairs = _project_seed(family, seed, seed_index)
            cells.extend(new_cells)
            pairs.extend(new_pairs)
    same_json((len(cells), len(pairs)), (168, 174), "development complete cell/pair coverage")
    return {
        "schema_id": "p612_complete_fixed_development_ledger_v1",
        "original_inputs": deepcopy(inputs),
        "cells": cells,
        "paired_differences": pairs,
        "coverage": {
            "families": 6,
            "development_bundles": 12,
            "preflight_bundles": 8,
            "complete_bundle_files": 60,
            "cells": 168,
            "ordered_seed_pairs": 174,
            "original_stored_pairs": 162,
            "projected_pairs": 12,
            "family_seed_instances": 18,
            "distinct_source_seeds": 15,
        },
        "interpretation": {
            "evaluation_role": "original_outer_selection_development_not_independent_final_confirmation",
            "selection": "fixed_settings_all_arms_retained_no_score_based_candidate_selection_in_these_pilots",
            "metric_name": "final_mean_task_accuracy_describes_the_two_task_endpoint_not_the_data_role",
            "pairs": "original_left_minus_right_accuracy_fractions_no_new_metric_interval_or_hypothesis_vote",
            "replication": "three_original_seeds_per_family_gating_and_replay_share_sources_no_pooling_repeat_adds_no_replications",
            "history_scope": "six_complete_fixed_pilots_bound_by_original_confirmation_manifest_not_all_historical_tuning_campaigns",
        },
        "validation_scope": "pure_complete_stored_development_declarations_only",
        "fresh_confirmation_reader_authority": False,
        "new_training_scoring_or_final_access": False,
        "original_p612_acceptance_complete": False,
    }


def _derive_development_ledger(inputs: dict[str, Any]) -> dict[str, Any]:
    """Fixture seam for complete declarations; no current IO or execution authority."""
    try:
        require(type(inputs) is dict, "development inputs must be an object")
        return _project_development_ledger(inputs)
    except (KeyError, TypeError, IndexError, AttributeError, StopIteration) as error:
        raise ValueError("development inputs are malformed or incomplete") from error


def build_development_ledger(inputs: dict[str, Any]) -> dict[str, Any]:
    """Project the entire original pinned scope, retaining all raw development inputs."""
    require(
        type(inputs) is dict and "original_scope" in inputs and "input_catalog" in inputs,
        "development inputs require whole scope and catalog",
    )
    same_json(
        canonical_body_identity(inputs["original_scope"]),
        SCOPE_ID,
        "whole original prospective scope",
    )
    same_json(
        canonical_body_identity(inputs["input_catalog"]),
        CATALOG_ID,
        "whole declared development input catalog",
    )
    return _derive_development_ledger(inputs)
