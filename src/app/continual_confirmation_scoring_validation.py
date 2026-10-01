"""Independently read the complete fixed scored JSON representation.

Inputs are decoded JSON and the frozen scientific scoring manifest. Outputs
are typed roles, endpoints, derived cells/totals and the three bound proofs.
This checks declared links only; no arrays/models/source, IO, prediction,
live-state proof, observed work/resource or publication authority belongs here.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.app.continual_confirmation_analysis import FinalRoleIdentity, OutcomeCell
from src.app.continual_confirmation_execution import json_value
from src.app.continual_confirmation_json import hash_value, object_fields, require, same_json
from src.app.continual_confirmation_scoring import (
    EndpointEvaluation,
    EvaluationTotals,
    ReleasedSeedFacts,
    ScoredConfirmation,
)
from src.app.continual_confirmation_scoring_manifest import (
    ConfirmationScoringManifest,
    scoring_manifest_digest,
    scoring_summary,
    validate_scoring_manifest,
)
from src.app.continual_confirmation_scoring_state import (
    TrainingArtifactIdentity,
    TrainingStateProof,
)
from src.core.confirmation_final_roles import EndpointFailure, EndpointResult, FinalRoleFacts
from src.core.confirmation_final_roles import validate_endpoint_result
from src.core.continual_metrics import TwoTaskAccuracy


_FIELDS = {
    "training_before_release",
    "training_after_release",
    "training_after_evaluation",
    "final_roles",
    "evaluations",
    "cells",
    "totals",
    "scoring_manifest_sha256",
    "validation_scope",
    "external_execution_verified",
    "outer_selection_scored",
}
_ENDPOINT_FIELDS = {
    "family",
    "seed",
    "arm",
    "endpoint",
    "checkpoint",
    "phase",
    "role_sha256",
    "example_count",
    "result",
}


def _array(value: Any, size: int, context: str) -> list[Any]:
    require(type(value) is list and len(value) == size, f"scored {context} array/count differs")
    return value


def _final_facts(value: Any, phase: str, seed: int, count: int) -> FinalRoleFacts:
    row = object_fields(value, {"phase", "seed", "sample_ids", "sha256", "count"}, "scored final")
    expected = {
        "phase": phase,
        "seed": seed,
        "sample_ids": [f"phase_{phase}/seed_{seed}/final/{index}" for index in range(count)],
        "count": count,
    }
    same_json({name: row[name] for name in expected}, expected, "scored final identity/IDs/count")
    digest = hash_value(row["sha256"], "scored final")
    return FinalRoleFacts(phase, seed, tuple(row["sample_ids"]), digest, count)


def _released_rows(
    values: Any, manifest: ConfirmationScoringManifest
) -> tuple[ReleasedSeedFacts, ...]:
    rows = _array(values, int(scoring_summary(manifest)["family_seed_rows"]), "role rows")
    index = 0
    result = []
    shared: dict[tuple[str, int], FinalRoleFacts] = {}
    for family in manifest.train_manifest.families:
        for seed in family.seeds:
            row = object_fields(rows[index], {"family", "seed", "a", "b"}, "scored released seed")
            same_json(
                {"family": row["family"], "seed": row["seed"]},
                {"family": family.name, "seed": seed},
                "scored family/seed order",
            )
            phases = tuple(
                _final_facts(row[phase], phase, seed, count)
                for phase, count in zip(("a", "b"), family.expected_final_counts, strict=True)
            )
            for facts in phases:
                key = (facts.phase, facts.seed)
                require(
                    key not in shared or shared[key] == facts,
                    "scored shared final signature differs",
                )
                shared[key] = facts
            result.append(ReleasedSeedFacts(family.name, seed, phases[0], phases[1]))
            index += 1
    return tuple(result)


def _endpoint_result(value: Any, count: int) -> EndpointResult:
    row = object_fields(value, {"correct_count", "failure"}, "scored endpoint result")
    failure = None
    if row["failure"] is not None:
        fields = object_fields(row["failure"], {"code", "error_type"}, "scored endpoint failure")
        failure = EndpointFailure(fields["code"], fields["error_type"])
    result = EndpointResult(row["correct_count"], failure)
    validate_endpoint_result(result, count)
    return result


def _endpoint(value: Any, row: ReleasedSeedFacts, arm: str, endpoint: str) -> EndpointEvaluation:
    raw = object_fields(value, _ENDPOINT_FIELDS, "scored endpoint")
    role = row.b if endpoint == "b_after_b" else row.a
    checkpoint = "a" if endpoint == "a_after_a" else "b"
    expected = {
        "family": row.family,
        "seed": row.seed,
        "arm": arm,
        "endpoint": endpoint,
        "checkpoint": checkpoint,
        "phase": role.phase,
        "role_sha256": role.sha256,
        "example_count": role.count,
    }
    same_json(
        {name: raw[name] for name in expected}, expected, "scored endpoint identity/order/role"
    )
    result = _endpoint_result(raw["result"], role.count)
    return EndpointEvaluation(
        row.family, row.seed, arm, endpoint, checkpoint, role.phase, role.sha256, role.count, result
    )


def _cell(row: ReleasedSeedFacts, arm: str, records: tuple[EndpointEvaluation, ...]) -> OutcomeCell:
    # Why independent: derive from decoded endpoint counts rather than trusting
    # saved floats/failures or calling the producer's cell-construction helper.
    values = tuple(validate_endpoint_result(item.result, item.example_count) for item in records)
    failures = [
        f"{item.endpoint}:{item.result.failure.code}:{item.result.failure.error_type or 'none'}"
        for item in records
        if item.result.failure is not None
    ]
    accuracy = None
    if not failures:
        a_after_a, a_after_b, b_after_b = values
        assert a_after_a is not None and a_after_b is not None and b_after_b is not None
        accuracy = TwoTaskAccuracy(a_after_a, a_after_b, b_after_b)
    roles = FinalRoleIdentity(row.a.sha256, row.b.sha256, row.a.count, row.b.count)
    return OutcomeCell(
        row.family, row.seed, arm, roles, accuracy, ";".join(failures) if failures else None
    )


def _outcomes(
    payload: dict[str, Any],
    rows: tuple[ReleasedSeedFacts, ...],
    manifest: ConfirmationScoringManifest,
) -> tuple[tuple[EndpointEvaluation, ...], tuple[OutcomeCell, ...]]:
    summary = scoring_summary(manifest)
    endpoints = _array(payload["evaluations"], int(summary["final_calls"]), "endpoints")
    cells = _array(payload["cells"], int(summary["cells"]), "cells")
    families = {family.name: family for family in manifest.train_manifest.families}
    decoded_endpoints: list[EndpointEvaluation] = []
    decoded_cells = []
    index = 0
    for row in rows:
        for arm in families[row.family].arms:
            records = tuple(
                _endpoint(endpoints[3 * index + offset], row, arm, endpoint)
                for offset, endpoint in enumerate(manifest.evaluation_endpoints)
            )
            cell = _cell(row, arm, records)
            same_json(
                cells[index], json_value(asdict(cell)), "scored endpoint/cell/accuracy/failure link"
            )
            decoded_endpoints.extend(records)
            decoded_cells.append(cell)
            index += 1
    return tuple(decoded_endpoints), tuple(decoded_cells)


def _proof(manifest: ConfirmationScoringManifest) -> TrainingStateProof:
    summary = scoring_summary(manifest)
    reference = manifest.training_bundles[0]
    return TrainingStateProof(
        TrainingArtifactIdentity(reference.result_sha256, reference.result_bytes),
        int(summary["family_seed_rows"]),
        int(summary["cells"]),
        2 * int(summary["cells"]),
        scoring_manifest_digest(manifest),
    )


def _totals(
    rows: tuple[ReleasedSeedFacts, ...],
    records: tuple[EndpointEvaluation, ...],
    cells: tuple[OutcomeCell, ...],
) -> EvaluationTotals:
    successful_endpoints = sum(item.result.correct_count is not None for item in records)
    successful_cells = sum(item.accuracy is not None for item in cells)
    return EvaluationTotals(
        2 * len(rows),
        2 * len(rows),
        len(records),
        sum(item.example_count for item in records),
        successful_endpoints,
        len(records) - successful_endpoints,
        successful_cells,
        len(cells) - successful_cells,
    )


def verify_scored_payload(
    payload: Any, manifest: ConfirmationScoringManifest
) -> ScoredConfirmation:
    """Require every planned row and link; preserve explicit app-only authority."""
    validate_scoring_manifest(manifest)
    row = object_fields(payload, _FIELDS, "scored envelope")
    proof = _proof(manifest)
    for stage in ("training_before_release", "training_after_release", "training_after_evaluation"):
        same_json(row[stage], json_value(asdict(proof)), f"scored {stage}")
    roles = _released_rows(row["final_roles"], manifest)
    endpoints, cells = _outcomes(row, roles, manifest)
    result = ScoredConfirmation(
        proof,
        proof,
        proof,
        roles,
        endpoints,
        cells,
        _totals(roles, endpoints, cells),
        scoring_manifest_digest(manifest),
    )
    # Closed recursive type/value equality also rejects authority flags, extra
    # fields, nonfinite floats and bool/int/float aliases throughout the body.
    same_json(row, json_value(asdict(result)), "complete scored payload")
    return result
