"""Read fabricated final observations without live/source/resource authority."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from test_continual_confirmation_scoring import _forbid, _seal_scoring
from test_continual_confirmation_scoring_validation import (
    _fail_endpoints,
    fabricated_scored_json as fabricated_scored_json,
)
from src.app.continual_confirmation_final_observation import verify_final_observation
from src.app.continual_confirmation_scoring_manifest import fixed_scoring_manifest
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.controlled_parent_selection import ParentControlledCircadianNetwork
from src.core.predictive_coding import PredictiveCodingNetwork


@pytest.fixture(autouse=True)
def seal_live_work(monkeypatch: pytest.MonkeyPatch) -> None:
    _seal_scoring(monkeypatch)
    monkeypatch.setattr(Path, "open", _forbid)
    for model in (
        BackpropMLP,
        PredictiveCodingNetwork,
        CircadianPredictiveCodingNetwork,
        ParentControlledCircadianNetwork,
    ):
        monkeypatch.setattr(model, "predict_proba", _forbid)
        monkeypatch.setattr(model, "compute_accuracy", _forbid)
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "_run_training_step", _forbid)


def _observation(payload: dict[str, Any]) -> dict[str, Any]:
    # Independent fixture oracle: known arm names and the complete fixed
    # inventory, without production observation helpers/checkpoint factories.
    releases = []
    reads = []
    for row in payload["final_roles"]:
        for phase in ("a", "b"):
            role = row[phase]
            base = {"family": row["family"], "seed": row["seed"], "phase": phase}
            releases.append(
                base
                | {
                    "role_sha256": role["sha256"],
                    "example_count": 40,
                    "sample_ids": list(role["sample_ids"]),
                }
            )
            for field in ("test_input", "test_target"):
                reads.append(base | {"field": field, "example_count": 40})
    predictions = []
    for row in payload["evaluations"]:
        arm = row["arm"]
        kind = (
            "backprop"
            if arm.startswith("backprop_")
            else "pc"
            if arm.startswith("pc_") or arm == "ordinary_pc"
            else "circadian"
        )
        failure = row["result"]["failure"]
        predictions.append(
            deepcopy(row)
            | {
                "model_kind": kind,
                "returned": failure is None or failure["code"] == "nonfinite_predictions",
            }
        )
    numerical_errors = sum(not row["returned"] for row in predictions)
    return {
        "schema_id": "p67_confirmation_final_observation_v1",
        "validation_scope": "actual_final_calls_for_supplied_held_inventory_only",
        "source_provenance_verified": False,
        "release_attempts": 120,
        "release_successes": 120,
        "input_attempts": 120,
        "input_reads": 120,
        "target_attempts": 120,
        "target_reads": 120,
        "prediction_attempts": 1680,
        "prediction_returns": 1680 - numerical_errors,
        "prediction_numerical_errors": numerical_errors,
        "prediction_examples": 67200,
        "unexpected_prediction_errors": 0,
        "blocked_calls": 0,
        "by_model_kind": {"backprop": 420, "pc": 450, "circadian": 810},
        "release_events": releases,
        "source_events": reads,
        "prediction_events": predictions,
    }


def test_should_read_every_whole_observation_without_mutating_or_granting_authority(
    fabricated_scored_json: dict[str, Any],
) -> None:
    payload = deepcopy(fabricated_scored_json)
    observed = _observation(payload)
    before = deepcopy((observed, payload))
    actual = verify_final_observation(observed, payload, fixed_scoring_manifest())
    assert actual == observed and (observed, payload) == before
    assert len(actual["release_events"]) == 120 and len(actual["source_events"]) == 240
    assert len(actual["prediction_events"]) == 1680 and actual["prediction_examples"] == 67200
    assert actual["by_model_kind"] == {"backprop": 420, "pc": 450, "circadian": 810}
    assert actual["source_provenance_verified"] is False
    assert payload["external_execution_verified"] is False


@pytest.mark.parametrize("kind", ["nonfinite_predictions", "numerical_prediction_error"])
@pytest.mark.parametrize("indices", [[0], [1679], [0, 1, 2], list(range(1680))])
def test_should_keep_all_observed_calls_and_distinguish_returns_from_numerical_exceptions(
    fabricated_scored_json: dict[str, Any], kind: str, indices: list[int]
) -> None:
    payload = deepcopy(fabricated_scored_json)
    _fail_endpoints(payload, indices, kind)
    observed = _observation(payload)
    assert verify_final_observation(observed, payload, fixed_scoring_manifest()) == observed
    errors = len(indices) if kind == "numerical_prediction_error" else 0
    assert observed["prediction_returns"] == 1680 - errors
    assert observed["prediction_numerical_errors"] == errors
    assert len(observed["prediction_events"]) == 1680
    observed["prediction_events"][indices[-1]]["returned"] = kind == "numerical_prediction_error"
    with pytest.raises(ValueError):
        verify_final_observation(observed, payload, fixed_scoring_manifest())


_DRIFT = [
    (("schema_id",), "other"),
    (("validation_scope",), "full_source_proof"),
    (("source_provenance_verified",), True),
    (("release_attempts",), 121),
    (("release_successes",), 119),
    (("input_attempts",), 121),
    (("input_reads",), 119),
    (("target_attempts",), 121),
    (("target_reads",), 119),
    (("prediction_attempts",), 1679),
    (("prediction_returns",), 1679),
    (("prediction_numerical_errors",), 1),
    (("prediction_examples",), 67160),
    (("unexpected_prediction_errors",), 1),
    (("blocked_calls",), 1),
    (("by_model_kind", "backprop"), 419),
    (("by_model_kind", "pc"), 451),
    (("by_model_kind", "circadian"), 809),
    (("release_events", -1, "family"), "combined"),
    (("release_events", -1, "seed"), 419.0),
    (("release_events", -1, "phase"), "a"),
    (("release_events", -1, "role_sha256"), "0" * 64),
    (("release_events", -1, "example_count"), 39),
    (("release_events", -1, "sample_ids", -1), "phase_b/seed_419/final/0"),
    (("source_events", -1, "family"), "combined"),
    (("source_events", -1, "seed"), 419.0),
    (("source_events", -1, "phase"), "a"),
    (("source_events", -1, "field"), "test_input"),
    (("source_events", -1, "example_count"), 39),
    (("prediction_events", -1, "family"), "combined"),
    (("prediction_events", -1, "seed"), 419.0),
    (("prediction_events", -1, "arm"), "neutral_off"),
    (("prediction_events", -1, "endpoint"), "a_after_b"),
    (("prediction_events", -1, "checkpoint"), "a"),
    (("prediction_events", -1, "phase"), "a"),
    (("prediction_events", -1, "role_sha256"), "0" * 64),
    (("prediction_events", -1, "example_count"), 39),
    (("prediction_events", -1, "result", "correct_count"), 41),
    (("prediction_events", -1, "model_kind"), "circadian"),
    (("prediction_events", -1, "returned"), False),
]


@pytest.mark.parametrize(("path", "value"), _DRIFT)
def test_should_reject_changed_last_links_and_forged_counts_even_with_valid_app_json(
    fabricated_scored_json: dict[str, Any], path: tuple[Any, ...], value: Any
) -> None:
    observed = _observation(fabricated_scored_json)
    row = observed
    for key in path[:-1]:
        row = row[key]
    row[path[-1]] = value
    with pytest.raises(ValueError):
        verify_final_observation(observed, fabricated_scored_json, fixed_scoring_manifest())


@pytest.mark.parametrize(
    "field",
    [
        "release_attempts",
        "release_successes",
        "input_attempts",
        "input_reads",
        "target_attempts",
        "target_reads",
        "prediction_attempts",
        "prediction_returns",
        "prediction_numerical_errors",
        "prediction_examples",
        "unexpected_prediction_errors",
        "blocked_calls",
    ],
)
@pytest.mark.parametrize("kind", ["float", "bool", "null"])
def test_should_refuse_coerced_count_types_including_zero_equal_to_false(
    fabricated_scored_json: dict[str, Any], field: str, kind: str
) -> None:
    observed = _observation(fabricated_scored_json)
    observed[field] = (
        float(observed[field]) if kind == "float" else False if kind == "bool" else None
    )
    with pytest.raises(ValueError):
        verify_final_observation(observed, fabricated_scored_json, fixed_scoring_manifest())


@pytest.mark.parametrize("field", ["release_events", "source_events", "prediction_events"])
@pytest.mark.parametrize("kind", ["missing", "extra", "reorder", "tuple", "unknown_field"])
def test_should_require_exact_ordered_closed_trace_inventories(
    fabricated_scored_json: dict[str, Any], field: str, kind: str
) -> None:
    observed = _observation(fabricated_scored_json)
    rows = observed[field]
    if kind == "missing":
        rows.pop()
    elif kind == "extra":
        rows.append(deepcopy(rows[-1]))
    elif kind == "reorder":
        rows[0], rows[-1] = rows[-1], rows[0]
    elif kind == "tuple":
        observed[field] = tuple(rows)
    else:
        rows[-1]["unexpected"] = False
    with pytest.raises(ValueError):
        verify_final_observation(observed, fabricated_scored_json, fixed_scoring_manifest())


@pytest.mark.parametrize("kind", ["unknown", "missing", "list", "float_flag", "kind_extra"])
def test_should_reject_open_or_malformed_observation_objects(
    fabricated_scored_json: dict[str, Any], kind: str
) -> None:
    observed: Any = _observation(fabricated_scored_json)
    if kind == "unknown":
        observed["unexpected"] = False
    elif kind == "missing":
        del observed["blocked_calls"]
    elif kind == "list":
        observed = [observed]
    elif kind == "float_flag":
        observed["source_provenance_verified"] = 0.0
    else:
        observed["by_model_kind"]["parent"] = 0
    with pytest.raises(ValueError):
        verify_final_observation(observed, fabricated_scored_json, fixed_scoring_manifest())


@pytest.mark.parametrize("kind", ["partial_manifest", "partial_scored", "unbound", "late_proof"])
def test_should_verify_the_whole_scientific_manifest_and_scored_body_before_trace_links(
    fabricated_scored_json: dict[str, Any], kind: str
) -> None:
    payload = deepcopy(fabricated_scored_json)
    observed = _observation(payload)
    manifest = fixed_scoring_manifest()
    if kind == "partial_manifest":
        manifest = replace(manifest, train_manifest=replace(manifest.train_manifest, families=()))
    elif kind == "partial_scored":
        payload["cells"].pop()
    elif kind == "unbound":
        payload["scoring_manifest_sha256"] = None
    else:
        payload["training_after_evaluation"]["training_artifact"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        verify_final_observation(observed, payload, manifest)
