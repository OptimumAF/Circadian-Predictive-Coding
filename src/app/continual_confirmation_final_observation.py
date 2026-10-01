"""Compare external final observations with the whole fixed scored declaration.

Inputs are decoded observations, scored JSON and the frozen scoring manifest.
Outputs are independently derived expected traces/counters. No live source,
model, arrays, prediction, resource enforcement or file provenance belongs here.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.app.continual_confirmation_checkpoints import declarations
from src.app.continual_confirmation_execution import json_value
from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_manifest import ConfirmationManifest
from src.app.continual_confirmation_scoring import ScoredConfirmation
from src.app.continual_confirmation_scoring_manifest import ConfirmationScoringManifest
from src.app.continual_confirmation_scoring_validation import verify_scored_payload


_KINDS = {
    "src.core.backprop_mlp.BackpropMLP": "backprop",
    "src.core.predictive_coding.PredictiveCodingNetwork": "pc",
    "src.core.circadian_predictive_coding.CircadianPredictiveCodingNetwork": "circadian",
    "src.core.controlled_parent_selection.ParentControlledCircadianNetwork": "circadian",
}


def _release_events(scored: ScoredConfirmation) -> list[dict[str, Any]]:
    return [
        {
            "family": row.family,
            "seed": row.seed,
            "phase": role.phase,
            "role_sha256": role.sha256,
            "example_count": role.count,
            "sample_ids": list(role.sample_ids),
        }
        for row in scored.final_roles
        for role in (row.a, row.b)
    ]


def _prediction_events(
    scored: ScoredConfirmation, manifest: ConfirmationManifest
) -> list[dict[str, Any]]:
    kinds = {
        (family.name, seed, arm): _KINDS[spec.model_type]
        for family in manifest.families
        for seed in family.seeds
        for arm, spec in declarations(family, seed).items()
    }
    return [
        {
            **json_value(asdict(row)),
            "model_kind": kinds[row.family, row.seed, row.arm],
            "returned": row.result.failure is None
            or row.result.failure.code != "numerical_prediction_error",
        }
        for row in scored.evaluations
    ]


def _expected_observation(
    scored: ScoredConfirmation, manifest: ConfirmationManifest
) -> dict[str, Any]:
    """Private development seam; public verification requires complete scoring JSON."""
    releases = _release_events(scored)
    predictions = _prediction_events(scored, manifest)
    source_events = [
        {key: event[key] for key in ("family", "seed", "phase", "example_count")} | {"field": field}
        for event in releases
        for field in ("test_input", "test_target")
    ]
    expected_cells = sum(len(f.seeds) * len(f.arms) for f in manifest.families)
    expected_releases = 2 * sum(len(f.seeds) for f in manifest.families)
    require(
        len(releases) == expected_releases and len(predictions) == 3 * expected_cells,
        "final observation complete supplied inventory differs",
    )
    failed_endpoints = sum(row.result.failure is not None for row in scored.evaluations)
    failed_cells = sum(
        any(row.result.failure is not None for row in scored.evaluations[index : index + 3])
        for index in range(0, len(scored.evaluations), 3)
    )
    same_json(
        asdict(scored.totals),
        {
            "release_calls": expected_releases,
            "final_role_views": expected_releases,
            "endpoint_calls": len(predictions),
            "example_count": sum(row["example_count"] for row in predictions),
            "successful_endpoints": len(predictions) - failed_endpoints,
            "failed_endpoints": failed_endpoints,
            "successful_cells": expected_cells - failed_cells,
            "failed_cells": failed_cells,
        },
        "external final observation/app totals",
    )
    returns = sum(row["returned"] for row in predictions)
    return {
        "schema_id": "p67_confirmation_final_observation_v1",
        "validation_scope": "actual_final_calls_for_supplied_held_inventory_only",
        "source_provenance_verified": False,
        "release_attempts": len(releases),
        "release_successes": len(releases),
        "input_attempts": len(releases),
        "input_reads": len(releases),
        "target_attempts": len(releases),
        "target_reads": len(releases),
        "prediction_attempts": len(predictions),
        "prediction_returns": returns,
        "prediction_numerical_errors": len(predictions) - returns,
        "prediction_examples": sum(row["example_count"] for row in predictions),
        "unexpected_prediction_errors": 0,
        "blocked_calls": 0,
        "by_model_kind": {
            kind: sum(row["model_kind"] == kind for row in predictions)
            for kind in ("backprop", "pc", "circadian")
        },
        "release_events": releases,
        "source_events": source_events,
        "prediction_events": predictions,
    }


def verify_final_observation(
    observed: Any, scored_json: Any, manifest: ConfirmationScoringManifest
) -> dict[str, Any]:
    """Require all fixed final calls/links; file/resource authority remains separate."""
    scored = verify_scored_payload(scored_json, manifest)
    expected = _expected_observation(scored, manifest.train_manifest)
    same_json(observed, expected, "complete external final observation")
    return expected
