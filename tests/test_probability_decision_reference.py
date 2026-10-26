"""Native-probability scoring oracles and paid scalar-callback boundaries.

All replies are analytic fixtures. No model, tokenizer, upstream import, network,
training, RNG or native inference occurs in this module.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
import json
import hashlib
from math import log
from typing import Any

import pytest

from src.adapters.open_jev_choice import decode_open_jev_probabilities
from src.app.decision_reference_evaluation import (
    DecisionCase,
    DecisionEvaluationFailed,
    evaluate_probability_reference,
)
from src.core.typed_decision import (
    DecisionCandidate,
    DecisionRequest,
    ProbabilityDecisionResult,
    validate_probability_decision_result,
)


def request(identifier: str = "case-1") -> DecisionRequest:
    return DecisionRequest(
        identifier,
        "Route this invoice",
        (DecisionCandidate("z", "Billing"), DecisionCandidate("a", "Account access")),
    )


def case(identifier: str = "case-1", target: str = "z") -> DecisionCase:
    return DecisionCase(request(identifier), target, "development", "outer_selection")


def result(query: DecisionRequest, **overrides: Any) -> ProbabilityDecisionResult:
    values: dict[str, Any] = {
        "request_id": query.request_id,
        "selected_id": "z",
        "probabilities": (("z", 0.8), ("a", 0.2)),
        "reported_confidence": 0.6,
    }
    values.update(overrides)
    return ProbabilityDecisionResult(**values)


def forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("inadmissible cases must be refused before callbacks")


def test_should_preserve_native_mass_and_allow_the_provider_to_choose_any_exact_tie():
    query = request()
    rounded = result(query, probabilities=(("z", 0.8000003), ("a", 0.2000003)))
    validate_probability_decision_result(query, rounded)
    assert rounded.probabilities == (("z", 0.8000003), ("a", 0.2000003))
    tied = result(query, selected_id="a", probabilities=(("z", 0.5), ("a", 0.5)))
    validate_probability_decision_result(query, tied)
    validate_probability_decision_result(
        replace(query, allowed_candidates=tuple(reversed(query.allowed_candidates))), tied
    )
    assert tied.selected_id == "a"
    with pytest.raises(FrozenInstanceError):
        setattr(tied, "selected_id", "z")


@pytest.mark.parametrize(
    "fault",
    [
        "identity",
        "missing",
        "extra",
        "duplicate",
        "mass",
        "nonmax",
        "nan",
        "bool_probability",
        "confidence",
        "bool_tokens",
        "negative_tokens",
        "fractional_tokens",
    ],
)
def test_should_refuse_corrupted_probability_records_without_relaxing_native_scalar_types(fault):
    query = request()
    value = result(query)
    changes: dict[str, tuple[str, Any]] = {
        "identity": ("request_id", "different"),
        "missing": ("probabilities", (("z", 1.0),)),
        "extra": ("probabilities", (("z", 0.8), ("a", 0.2), ("other", 0.0))),
        "duplicate": ("probabilities", (("z", 0.8), ("z", 0.2))),
        "mass": ("probabilities", (("z", 0.8), ("a", 0.21))),
        "nonmax": ("selected_id", "a"),
        "nan": ("probabilities", (("z", float("nan")), ("a", 0.2))),
        "bool_probability": ("probabilities", (("z", True), ("a", 0.0))),
        "confidence": ("reported_confidence", float("inf")),
        "bool_tokens": ("reported_input_tokens", True),
        "negative_tokens": ("reported_output_tokens", -1),
        "fractional_tokens": ("reported_input_tokens", 1.0),
    }
    name, replacement = changes[fault]
    object.__setattr__(value, name, replacement)
    with pytest.raises(ValueError):
        validate_probability_decision_result(query, value)


def test_should_score_present_and_missing_targets_with_an_independent_probability_oracle():
    supplied: list[DecisionRequest] = []

    def predict(query):
        assert type(query) is DecisionRequest
        assert {field.name for field in fields(query)} == {
            "request_id",
            "query",
            "allowed_candidates",
        }
        supplied.append(deepcopy(query))
        return result(query)

    report = evaluate_probability_reference(
        (case(), case("case-2", "not-recalled")), predict=predict
    )
    assert report["complete"] is True and len(supplied) == 4
    rows = report["cases"]
    assert [row["brier"] for row in rows] == pytest.approx([0.08, 1.68])
    assert [row["target_probability"] for row in rows] == [0.8, 0.0]
    assert [row["candidate_recalled"] for row in rows] == [True, False]
    assert all(row["confidence"] == 0.8 for row in rows)
    assert all(row["result"]["reported_confidence"] == 0.6 for row in rows)
    assert all("raw_score" not in json.dumps(row["result"]) for row in rows)
    metrics = report["metrics"]
    assert metrics["accuracy"] == metrics["candidate_recall"] == 0.5
    assert metrics["brier_mean"] == pytest.approx(0.88)
    assert metrics["nll_mean"] is None
    assert metrics["nll_status"] == "infinite_due_to_zero_target_probability"
    assert metrics["zero_target_probability_count"] == 1
    assert metrics["ece_5bin"] == pytest.approx(0.3)
    assert sum(item["support"] for item in metrics["calibration_bins"]) == 2
    assert report["learning_active"] is None and report["training_updates"] is None
    assert report["external_actions"] is None
    assert report["predictor_behavior"] == "UNVERIFIED_CALLBACK_BEHAVIOR"


def test_should_compute_finite_nll_without_treating_native_margin_as_ece_confidence():
    report = evaluate_probability_reference((case(),), predict=result)
    assert report["metrics"]["nll_mean"] == pytest.approx(-log(0.8))
    assert report["metrics"]["nll_status"] == "finite"
    assert report["metrics"]["ece_5bin"] == pytest.approx(0.2)


def test_should_distinguish_unknown_reported_token_costs_from_observed_zero():
    calls: list[None] = []

    def predict(query):
        calls.append(None)
        if len(calls) == 1:
            return result(query, reported_input_tokens=0, reported_output_tokens=0)
        return result(query)

    report = evaluate_probability_reference((case(),), predict=predict)
    assert len(calls) == 2
    reported = report["work"]["reported"]
    assert reported["original"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "input_tokens_unknown_calls": 0,
        "output_tokens_unknown_calls": 0,
    }
    assert reported["reversed"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "input_tokens_unknown_calls": 1,
        "output_tokens_unknown_calls": 1,
    }
    assert report["cases"][0]["result"]["reported_input_tokens"] == 0
    assert report["cases"][0]["reversed_result"]["reported_input_tokens"] is None


@pytest.mark.parametrize("role", ["final", "probe_train", "probe_test", "inner_guard"])
def test_should_refuse_protected_roles_before_any_callback(role):
    record = case()
    object.__setattr__(record, "role", role)
    with pytest.raises(ValueError):
        evaluate_probability_reference((record,), predict=forbidden)


def test_should_refuse_train_and_development_source_group_overlap_before_callbacks():
    first = replace(case(), role="train")
    with pytest.raises(ValueError):
        evaluate_probability_reference((first, case("case-2")), predict=forbidden)


def test_should_retain_provider_tie_disagreement_as_a_valid_adverse_order_control():
    calls: list[None] = []

    def predict(query):
        calls.append(None)
        return result(
            query,
            selected_id=query.allowed_candidates[0].candidate_id,
            probabilities=(("z", 0.5), ("a", 0.5)),
            reported_confidence=0.0,
        )

    report = evaluate_probability_reference((case(),), predict=predict)
    assert report["complete"] is True and len(calls) == 2
    assert report["candidate_order_control"] == {
        "type": "reverse_order_only",
        "cases": 1,
        "all_equal": False,
    }
    control = report["cases"][0]["order_control"]
    assert control == {
        "probabilities_agree": True,
        "choice_agrees": False,
        "agreement": False,
        "max_absolute_probability_difference": 0.0,
    }
    assert report["cases"][0]["result"]["selected_id"] == "z"
    assert report["cases"][0]["reversed_result"]["selected_id"] == "a"


@pytest.mark.parametrize("delta,agrees", [(0.0000005, True), (0.000002, False)])
def test_should_compare_probability_order_controls_with_absolute_native_tolerance(delta, agrees):
    calls: list[None] = []

    def predict(query):
        calls.append(None)
        difference = 0.0 if len(calls) == 1 else delta
        return result(query, probabilities=(("a", 0.2 - difference), ("z", 0.8 + difference)))

    report = evaluate_probability_reference((case(),), predict=predict)
    assert report["complete"] is True and len(calls) == 2
    control = report["cases"][0]["order_control"]
    assert control["probabilities_agree"] is agrees
    assert control["choice_agrees"] is True and control["agreement"] is agrees
    assert control["max_absolute_probability_difference"] == pytest.approx(delta)


def test_should_score_a_private_case_snapshot_when_callback_closure_mutates_the_caller():
    original = case()
    original_request = deepcopy(original.request)
    calls: list[None] = []

    def predict(query):
        calls.append(None)
        object.__setattr__(original, "target_id", "a")
        object.__setattr__(original.request, "query", "caller changed its own query")
        return result(query)

    report = evaluate_probability_reference((original,), predict=predict)
    assert len(calls) == 2 and report["complete"] is True
    assert report["cases"][0]["request"]["query"] == original_request.query
    assert report["cases"][0]["target_id"] == "z"
    assert report["metrics"]["accuracy"] == 1.0
    assert original.target_id == "a" and original.request.query == "caller changed its own query"


def test_should_refuse_mutation_of_supplied_request_after_retaining_the_return():
    calls: list[None] = []

    def predict(query):
        calls.append(None)
        object.__setattr__(query, "query", "changed supplied request")
        return result(query)

    with pytest.raises(DecisionEvaluationFailed) as captured:
        evaluate_probability_reference((case(),), predict=predict)
    partial = captured.value.report
    assert calls == [None] and partial["complete"] is False
    assert partial["work"]["prediction_calls_attempted"] == 1
    assert partial["work"]["prediction_calls_returned"] == 1
    assert partial["work"]["prediction_calls_validated"] == 0
    assert partial["work"]["unvalidated_return_work_unknown"] is True
    assert partial["pending_case"]["original_returned"]["validated"] is False
    json.dumps(partial, allow_nan=False)


def test_should_retain_complete_case_and_pending_original_when_later_reverse_raises():
    calls: list[None] = []

    def predict(query):
        calls.append(None)
        if len(calls) == 4:
            raise RuntimeError("controlled paid reverse failure")
        return result(query, reported_input_tokens=5, reported_output_tokens=0)

    with pytest.raises(DecisionEvaluationFailed) as captured:
        evaluate_probability_reference((case(), case("case-2")), predict=predict)
    error = captured.value
    partial = error.report
    assert len(calls) == 4 and len(partial["cases"]) == 1
    assert "controlled paid reverse failure" in partial["error"]
    assert partial["cases"][0]["request"]["request_id"] == "case-1"
    assert partial["pending_case"]["request_id"] == "case-2"
    assert partial["pending_case"]["original_returned"]["validated"] is True
    assert "metrics" not in partial
    assert partial["work"]["prediction_calls_attempted"] == 4
    assert partial["work"]["prediction_calls_returned"] == 3
    assert partial["work"]["prediction_calls_validated"] == 3
    assert partial["work"]["unreturned_prediction_work_unknown"] is True
    assert partial["work"]["unvalidated_return_work_unknown"] is False
    assert partial["work"]["reported"]["original"]["input_tokens"] == 10
    assert partial["work"]["reported"]["reversed"]["input_tokens"] == 5
    changed_copy = error.work
    changed_copy["prediction_calls_attempted"] = 0
    assert error.work["prediction_calls_attempted"] == 4
    json.dumps(partial, allow_nan=False)


@pytest.mark.parametrize("fault", ["nonfinite", "missing", "unsupported"])
def test_should_capture_rejected_returns_as_bounded_json_without_inventing_reported_costs(fault):
    calls: list[None] = []

    def predict(query):
        calls.append(None)
        value = result(query)
        if fault == "nonfinite":
            object.__setattr__(value, "probabilities", (("z", float("nan")), ("a", 0.2)))
        elif fault == "missing":
            object.__delattr__(value, "probabilities")
        else:
            return object()
        return value

    with pytest.raises(DecisionEvaluationFailed) as captured:
        evaluate_probability_reference((case(),), predict=predict)
    partial = captured.value.report
    assert calls == [None] and partial["complete"] is False
    assert partial["work"]["prediction_calls_attempted"] == 1
    assert partial["work"]["prediction_calls_returned"] == 1
    assert partial["work"]["prediction_calls_validated"] == 0
    assert partial["work"]["unreturned_prediction_work_unknown"] is False
    assert partial["work"]["unvalidated_return_work_unknown"] is True
    assert partial["work"]["reported"]["original"]["input_tokens"] == 0
    assert partial["work"]["reported"]["original"]["input_tokens_unknown_calls"] == 0
    observed = partial["pending_case"]["original_returned"]
    if fault == "unsupported":
        assert observed == {"record_type": "unsupported", "payload_unavailable": True}
    else:
        assert observed["record_type"] == "ProbabilityDecisionResult"
        assert observed["observed"]["request_id"] == "case-1"
        assert observed["observed"]["selected_id"] == "z"
        if fault == "nonfinite":
            assert "unavailable" in observed["observed"]["probabilities"][0][1]["capture"]
        else:
            assert "unavailable" in observed["observed"]["probabilities"]["capture"]
    json.dumps(partial, allow_nan=False)


def test_should_detach_a_validated_return_before_the_next_callback_changes_its_object():
    first = result(request(), reported_input_tokens=3, reported_output_tokens=0)
    calls: list[None] = []

    def predict(query):
        calls.append(None)
        if len(calls) == 1:
            return first
        object.__setattr__(first, "reported_input_tokens", 999)
        return result(query, reported_input_tokens=7, reported_output_tokens=0)

    report = evaluate_probability_reference((case(),), predict=predict)
    assert len(calls) == 2 and report["complete"] is True
    assert report["cases"][0]["result"]["reported_input_tokens"] == 3
    assert report["cases"][0]["reversed_result"]["reported_input_tokens"] == 7
    assert report["work"]["reported"]["original"]["input_tokens"] == 3


def native_body(usage: Any = None, *, include_usage: bool = True) -> bytes:
    probabilities = {"z": 0.8000003, "a": 0.2000003}
    margin = (probabilities["z"] / sum(probabilities.values()) - 0.5) / 0.5
    payload: dict[str, Any] = {
        "answers": {
            "decision": {
                "type": "choice",
                "choice": "z",
                "probabilities": probabilities,
                "confidence": margin,
            }
        }
    }
    if include_usage:
        payload["usage"] = usage
    return json.dumps(payload, allow_nan=False).encode()


@pytest.mark.parametrize(
    "usage",
    [None, {}, {"input_tokens": 0, "output_tokens": 0}, {"input_tokens": 17, "output_tokens": 0}],
    ids=["absent", "empty", "zero", "reported"],
)
def test_should_decode_native_probabilities_and_optional_reported_costs_without_invention(usage):
    body = native_body(usage, include_usage=usage is not None)
    value = decode_open_jev_probabilities(request(), body)
    assert value.request_id == "case-1" and value.selected_id == "z"
    assert value.probabilities == (("z", 0.8000003), ("a", 0.2000003))
    assert sum(probability for _, probability in value.probabilities) != 1.0
    assert value.reported_confidence == json.loads(body)["answers"]["decision"]["confidence"]
    assert value.reported_input_tokens == (None if usage is None else usage.get("input_tokens"))
    assert value.reported_output_tokens == (None if usage is None else usage.get("output_tokens"))
    assert {field.name for field in fields(value)} == {
        "request_id",
        "selected_id",
        "probabilities",
        "reported_confidence",
        "reported_input_tokens",
        "reported_output_tokens",
    }


@pytest.mark.parametrize(
    "usage",
    [{"input_tokens": True}, {"output_tokens": -1}, {"input_tokens": 1.0}, None],
    ids=["bool", "negative", "fractional", "null-container"],
)
def test_should_refuse_malformed_reported_usage_in_the_native_probability_bridge(usage):
    with pytest.raises(ValueError):
        decode_open_jev_probabilities(request(), native_body(usage))


def test_should_route_observed_native_probabilities_into_existing_metrics_with_reported_costs_only():
    calls: list[None] = []
    body = native_body({"input_tokens": 17, "output_tokens": 0})

    def predict(query):
        calls.append(None)
        return decode_open_jev_probabilities(query, body)

    report = evaluate_probability_reference((case(),), predict=predict)
    assert report["complete"] is True and len(calls) == 2
    assert report["cases"][0]["target_probability"] == 0.8000003
    assert report["work"]["reported"]["original"]["input_tokens"] == 17
    assert report["work"]["reported"]["reversed"]["input_tokens"] == 17
    assert report["work"]["prediction_calls_validated"] == 2
    assert "token_visits" not in json.dumps(report)
    assert "raw_score" not in json.dumps(report)


def saved_smoke() -> dict[str, Any]:
    criteria = {
        "password": "Password resets and login problems",
        "billing": "Charges, invoices and refunds",
    }
    reply = {
        "answers": {
            "decision": {
                "type": "choice",
                "choice": "password",
                "probabilities": {"password": 0.8, "billing": 0.2},
                "confidence": 0.6,
            }
        },
        "usage": {"input_tokens": 116, "output_tokens": 0},
    }
    rows: list[dict[str, Any]] = []
    for allowed in (criteria, dict(reversed(tuple(criteria.items())))):
        rows.append(
            {
                "request": {
                    "model": "open-jev",
                    "state": "I forgot my password and cannot sign in.",
                    "questions": {
                        "decision": {
                            "type": "choice",
                            "instructions": "Choose the customer's support intent.",
                            "criteria": deepcopy(allowed),
                        }
                    },
                },
                "response": deepcopy(reply),
                "response_json": json.dumps(reply),
            }
        )
    return {"definition": "open-jev-native-choice-smoke-v1", "complete": True, "requests": rows}


def test_should_reprocess_one_owned_saved_pair_without_new_native_compute(tmp_path):
    from scripts.score_open_jev_smoke import score_saved_smoke

    path = tmp_path / "synthetic-smoke.json"
    body = json.dumps(saved_smoke()).encode()
    path.write_bytes(body)
    report = score_saved_smoke(path)
    assert report["complete"] is True
    assert report["metrics"]["brier_mean"] == pytest.approx(0.08)
    assert report["cases"][0]["target_id"] == "password"
    assert report["cases"][0]["role"] == "outer_selection"
    assert report["source_artifact"] == {
        "path": str(path.resolve()),
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    assert report["execution"]["mode"] == "saved_reply_reprocessing"
    assert report["execution"]["callback_observations"] == 2
    assert report["execution"]["new_model_loads"] == report["execution"]["new_model_forwards"] == 0
    assert report["execution"]["new_training_updates"] == 0
    assert "inherited" in report["execution"]["reported_tokens"]
    assert report["work"]["reported"]["original"]["input_tokens"] == 116
    assert report["work"]["reported"]["reversed"]["input_tokens"] == 116
    assert path.read_bytes() == body


@pytest.mark.parametrize("fault", ["first-query", "second-order", "raw-owned"])
def test_should_reject_saved_request_or_raw_response_mismatch_before_evaluation(
    monkeypatch, tmp_path, fault
):
    from scripts import score_open_jev_smoke as command

    payload = saved_smoke()
    if fault == "first-query":
        payload["requests"][0]["request"]["state"] = "different query"
    elif fault == "second-order":
        payload["requests"][1]["request"]["questions"]["decision"]["criteria"] = deepcopy(
            payload["requests"][0]["request"]["questions"]["decision"]["criteria"]
        )
    else:
        payload["requests"][1]["response"]["usage"]["input_tokens"] = 115
    path = tmp_path / "mismatch.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(command, "evaluate_probability_reference", forbidden)
    with pytest.raises(ValueError):
        command.score_saved_smoke(path)


def test_should_preserve_existing_cli_output_and_publish_a_fresh_failure_without_loading_models(
    monkeypatch, tmp_path
):
    from scripts import score_open_jev_smoke as command

    existing = tmp_path / "existing.json"
    existing.write_text("original")
    monkeypatch.setattr(command, "score_saved_smoke", forbidden)
    with pytest.raises(SystemExit) as captured:
        command.main(["--smoke-report", "never-read.json", "--output", str(existing)])
    assert captured.value.code == 2 and existing.read_text() == "original"

    calls: list[None] = []

    def fail(path):
        calls.append(None)
        raise ValueError("controlled saved-input refusal")

    monkeypatch.setattr(command, "score_saved_smoke", fail)
    fresh = tmp_path / "failed.json"
    assert command.main(["--smoke-report", "never-read.json", "--output", str(fresh)]) == 1
    report = json.loads(fresh.read_text())
    assert calls == [None] and report["complete"] is False
    assert "controlled saved-input refusal" in report["error"]
    assert report["execution"]["new_model_loads"] == report["execution"]["new_model_forwards"] == 0
