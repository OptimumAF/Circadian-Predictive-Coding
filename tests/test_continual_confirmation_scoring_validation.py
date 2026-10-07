"""Fabricated whole scored JSON checks declarations, never reserved execution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from typing import Any

import pytest

from test_continual_confirmation_scoring import (
    _fabricated_role,
    _forbid,
    _seal_scoring,
    _whole_metadata,
)
from src.app import continual_confirmation_scoring as scoring
from src.app import continual_confirmation_scoring_validation as validator
from src.app.continual_confirmation_analysis import analyze_confirmation
from src.app.continual_confirmation_scoring_manifest import (
    fixed_scoring_manifest,
    scoring_manifest_digest,
)
from src.app.continual_confirmation_scoring_state import (
    TrainingArtifactIdentity,
    TrainingStateProof,
)
from src.core.confirmation_final_roles import EndpointResult
from src.infra import continual_confirmation_final as final_adapter


@pytest.fixture(scope="module")
def fabricated_scored_json() -> dict[str, Any]:
    # Explicit state spy: these are metadata model tokens/fabricated arrays,
    # not a reserved training result or a resource/provenance proof.
    manifest = fixed_scoring_manifest()
    reference = manifest.training_bundles[0]
    proof = TrainingStateProof(
        TrainingArtifactIdentity(reference.result_sha256, reference.result_bytes),
        60,
        560,
        1120,
        scoring_manifest_digest(manifest),
    )
    calls = 0

    def release(item: Any, phase: str) -> Any:
        facts = item.facts.roles[0 if phase == "a" else 1]
        return _fabricated_role(phase, item.facts.seed, facts.sample_ids["final_test"])

    def evaluate(model: Any, role: Any) -> EndpointResult:
        nonlocal calls
        calls += 1
        return EndpointResult(calls % 41)

    with pytest.MonkeyPatch.context() as patcher:
        _seal_scoring(patcher)
        patcher.setattr(scoring, "verify_scoring_training_state", lambda *args: proof)
        result = scoring.evaluate_confirmation(
            _whole_metadata(), manifest, release, evaluate, lambda stage: None
        )
    assert calls == 1680
    return json.loads(json.dumps(asdict(result), allow_nan=False))


@pytest.fixture(autouse=True)
def seal_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    _seal_scoring(monkeypatch)
    monkeypatch.setattr(final_adapter, "release_confirmation_final", _forbid)
    monkeypatch.setattr(final_adapter, "evaluate_confirmation_final", _forbid)
    monkeypatch.setattr(scoring, "_cell_from_records", _forbid)


def test_should_independently_read_all_roles_endpoints_cells_and_analysis_pairs(
    fabricated_scored_json: dict[str, Any],
) -> None:
    payload = deepcopy(fabricated_scored_json)
    before = deepcopy(payload)
    manifest = fixed_scoring_manifest()
    result = validator.verify_scored_payload(payload, manifest)
    assert json.loads(json.dumps(asdict(result))) == before == payload
    assert len(result.final_roles) == 60 and len(result.evaluations) == 1680
    assert result.totals.endpoint_calls == 1680 and result.totals.example_count == 67200
    assert result.totals.release_calls == result.totals.final_role_views == 120
    assert result.training_before_release == result.training_after_release
    assert result.training_before_release == result.training_after_evaluation
    assert result.external_execution_verified is False and result.outer_selection_scored is False
    analyzed = analyze_confirmation(result.cells, manifest.analysis_contract)
    assert analyzed.successful_cells == 560 and analyzed.failed_cells == 0
    assert analyzed.primary_statement_count == 116
    assert (
        sum(
            len(pair.metrics[0].summary.observations)
            for family in analyzed.families
            for pair in family.contrasts
        )
        == 580
    )


def _fail_endpoints(payload: dict[str, Any], indices: list[int], kind: str) -> None:
    failure = {
        "code": kind,
        "error_type": "FloatingPointError" if kind == "numerical_prediction_error" else None,
    }
    for index in indices:
        payload["evaluations"][index]["result"] = {"correct_count": None, "failure": failure.copy()}
    affected = {index // 3 for index in indices}
    for index in affected:
        endpoints = payload["evaluations"][3 * index : 3 * index + 3]
        reasons = [
            f"{row['endpoint']}:{row['result']['failure']['code']}:"
            f"{row['result']['failure']['error_type'] or 'none'}"
            for row in endpoints
            if row["result"]["failure"] is not None
        ]
        payload["cells"][index]["accuracy"] = None
        payload["cells"][index]["failure"] = ";".join(reasons)
    payload["totals"].update(
        failed_endpoints=len(indices),
        successful_endpoints=1680 - len(indices),
        failed_cells=len(affected),
        successful_cells=560 - len(affected),
    )


@pytest.mark.parametrize("indices", [[0], [1679], [0, 1, 2], list(range(1680))])
@pytest.mark.parametrize("kind", ["nonfinite_predictions", "numerical_prediction_error"])
def test_should_keep_every_call_and_null_cell_for_each_declared_numerical_failure(
    fabricated_scored_json: dict[str, Any], indices: list[int], kind: str
) -> None:
    payload = deepcopy(fabricated_scored_json)
    _fail_endpoints(payload, indices, kind)
    result = validator.verify_scored_payload(payload, fixed_scoring_manifest())
    assert json.loads(json.dumps(asdict(result))) == payload
    assert len(result.cells) == 560 and len(result.evaluations) == 1680
    assert result.totals.failed_endpoints == len(indices)
    assert result.totals.failed_cells == len({index // 3 for index in indices})
    analyzed = analyze_confirmation(result.cells, fixed_scoring_manifest().analysis_contract)
    assert analyzed.failed_cells == result.totals.failed_cells
    assert analyzed.primary_statement_count == 116


_DRIFT = [
    (("scoring_manifest_sha256",), "0" * 64),
    (("validation_scope",), "source_proof"),
    (("external_execution_verified",), True),
    (("outer_selection_scored",), True),
    (("outer_selection_scored",), 0),
    (("final_roles", -1, "family"), "combined"),
    (("final_roles", -1, "seed"), 419.0),
    (("final_roles", -1, "a", "phase"), "b"),
    (("final_roles", -1, "a", "seed"), 359),
    (("final_roles", -1, "a", "count"), 39),
    (("final_roles", -1, "a", "count"), 40.0),
    (("final_roles", -1, "a", "sha256"), "ABC"),
    (("final_roles", -1, "a", "sample_ids", -1), "phase_a/seed_419/development/39"),
    (("evaluations", -1, "family"), "combined"),
    (("evaluations", -1, "seed"), 419.0),
    (("evaluations", -1, "arm"), "unknown"),
    (("evaluations", -1, "endpoint"), "a_after_b"),
    (("evaluations", -1, "checkpoint"), "a"),
    (("evaluations", -1, "phase"), "a"),
    (("evaluations", -1, "role_sha256"), "0" * 64),
    (("evaluations", -1, "example_count"), 39),
    (("evaluations", -1, "example_count"), 40.0),
    (("evaluations", -1, "result", "correct_count"), True),
    (("evaluations", -1, "result", "correct_count"), 40.0),
    (("evaluations", -1, "result", "correct_count"), 41),
    (("evaluations", -1, "result", "correct_count"), -1),
    (("evaluations", -1, "result", "correct_count"), None),
    (
        ("evaluations", -1, "result", "failure"),
        {"code": "nonfinite_predictions", "error_type": None},
    ),
    (("cells", -1, "family"), "combined"),
    (("cells", -1, "seed"), 419.0),
    (("cells", -1, "arm"), "unknown"),
    (("cells", -1, "roles"), None),
    (("cells", -1, "roles", "score_role"), "outer_selection"),
    (("cells", -1, "roles", "a_sha256"), "0" * 64),
    (("cells", -1, "roles", "b_count"), 39),
    (("cells", -1, "accuracy", "a_after_a"), float("nan")),
    (("cells", -1, "accuracy", "b_after_b"), float("inf")),
    (("cells", -1, "accuracy", "a_after_b"), 0.001),
    (("cells", -1, "accuracy"), None),
    (("cells", -1, "failure"), "numerical_failure"),
]


def _replace_path(payload: Any, path: tuple[Any, ...], value: Any) -> None:
    owner = payload
    for name in path[:-1]:
        owner = owner[name]
    owner[path[-1]] = deepcopy(value)


@pytest.mark.parametrize("path,value", _DRIFT, ids=[str(path) for path, value in _DRIFT])
def test_should_reject_late_endpoint_role_cell_and_authority_drift(
    fabricated_scored_json: dict[str, Any], path: tuple[Any, ...], value: Any
) -> None:
    payload = deepcopy(fabricated_scored_json)
    _replace_path(payload, path, value)
    with pytest.raises(ValueError):
        validator.verify_scored_payload(payload, fixed_scoring_manifest())


@pytest.mark.parametrize(
    "stage", ["training_before_release", "training_after_release", "training_after_evaluation"]
)
@pytest.mark.parametrize(
    "path,value",
    [
        (("training_artifact", "sha256"), "0" * 64),
        (("training_artifact", "byte_count"), 134554377),
        (("training_artifact", "byte_count"), 134554378.0),
        (("family_seed_rows",), 59),
        (("cells",), 559),
        (("held_checkpoints",), 1119),
        (("scoring_manifest_sha256",), None),
        (("validation_scope",), "source_proof"),
        (("source_provenance_verified",), True),
        (("final_release_authorized",), True),
    ],
)
def test_should_reject_every_global_training_proof_link_failure(
    fabricated_scored_json: dict[str, Any], stage: str, path: tuple[Any, ...], value: Any
) -> None:
    payload = deepcopy(fabricated_scored_json)
    _replace_path(payload[stage], path, value)
    with pytest.raises(ValueError):
        validator.verify_scored_payload(payload, fixed_scoring_manifest())


@pytest.mark.parametrize(
    "name",
    [
        "release_calls",
        "final_role_views",
        "endpoint_calls",
        "example_count",
        "successful_endpoints",
        "failed_endpoints",
        "successful_cells",
        "failed_cells",
    ],
)
@pytest.mark.parametrize("alias", [False, True])
def test_should_derive_each_total_instead_of_trusting_reported_counts(
    fabricated_scored_json: dict[str, Any], name: str, alias: bool
) -> None:
    payload = deepcopy(fabricated_scored_json)
    payload["totals"][name] = (
        float(payload["totals"][name]) if alias else payload["totals"][name] + 1
    )
    with pytest.raises(ValueError):
        validator.verify_scored_payload(payload, fixed_scoring_manifest())


@pytest.mark.parametrize("name", ["final_roles", "evaluations", "cells"])
@pytest.mark.parametrize("drift", ["missing", "duplicate", "reverse", "tuple", "mapping"])
def test_should_require_complete_ordered_json_arrays(
    fabricated_scored_json: dict[str, Any], name: str, drift: str
) -> None:
    payload = deepcopy(fabricated_scored_json)
    rows = payload[name]
    payload[name] = {
        "missing": rows[:-1],
        "duplicate": rows + [rows[-1]],
        "reverse": rows[::-1],
        "tuple": tuple(rows),
        "mapping": {},
    }[drift]
    with pytest.raises(ValueError):
        validator.verify_scored_payload(payload, fixed_scoring_manifest())


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("training_after_evaluation",),
        ("training_after_evaluation", "training_artifact"),
        ("final_roles", -1),
        ("final_roles", -1, "b"),
        ("evaluations", -1),
        ("evaluations", -1, "result"),
        ("cells", -1),
        ("cells", -1, "roles"),
        ("cells", -1, "accuracy"),
        ("totals",),
    ],
)
@pytest.mark.parametrize("kind", ["extra", "missing"])
def test_should_require_closed_fields_at_every_saved_object(
    fabricated_scored_json: dict[str, Any], path: tuple[Any, ...], kind: str
) -> None:
    payload = deepcopy(fabricated_scored_json)
    owner = payload
    for key in path:
        owner = owner[key]
    if kind == "extra":
        owner["unknown"] = "extra"
    else:
        del owner[next(iter(owner))]
    with pytest.raises(ValueError):
        validator.verify_scored_payload(payload, fixed_scoring_manifest())


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "failure",
        {},
        {"code": "unknown", "error_type": None},
        {"code": "numerical_prediction_error", "error_type": None},
        {"code": "nonfinite_predictions", "error_type": "FloatingPointError"},
        {"code": "nonfinite_predictions", "error_type": None, "unknown": 1},
    ],
)
def test_should_reject_unplanned_or_missing_numerical_failure_policies(
    fabricated_scored_json: dict[str, Any], failure: Any
) -> None:
    payload = deepcopy(fabricated_scored_json)
    _fail_endpoints(payload, [1679], "nonfinite_predictions")
    payload["evaluations"][-1]["result"]["failure"] = failure
    with pytest.raises(ValueError):
        validator.verify_scored_payload(payload, fixed_scoring_manifest())


def test_should_reject_resealed_shared_source_signature_drift(
    fabricated_scored_json: dict[str, Any],
) -> None:
    payload = deepcopy(fabricated_scored_json)
    # Reseal all replay endpoints/cells for this shared gating/replay seed.
    row = payload["final_roles"][10]
    seed = row["seed"]
    row["a"]["sha256"] = "0" * 64
    for record in payload["evaluations"]:
        if record["family"] == "replay" and record["seed"] == seed and record["phase"] == "a":
            record["role_sha256"] = "0" * 64
    for cell in payload["cells"]:
        if cell["family"] == "replay" and cell["seed"] == seed:
            cell["roles"]["a_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="shared"):
        validator.verify_scored_payload(payload, fixed_scoring_manifest())


def test_should_refuse_partial_manifest_before_touching_any_payload() -> None:
    manifest = fixed_scoring_manifest()
    partial = replace(
        manifest,
        train_manifest=replace(
            manifest.train_manifest, families=manifest.train_manifest.families[:1]
        ),
    )
    with pytest.raises(ValueError, match="frozen complete"):
        validator.verify_scored_payload(None, partial)


@pytest.mark.parametrize("payload", [None, [], 1, True, "json"])
def test_should_raise_useful_value_errors_for_non_object_payloads(payload: Any) -> None:
    with pytest.raises(ValueError):
        validator.verify_scored_payload(payload, fixed_scoring_manifest())
