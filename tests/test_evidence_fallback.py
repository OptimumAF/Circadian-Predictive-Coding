"""Typed memory/frozen-callback availability policy; no models or fitting."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from src.app.decision_evidence import EvidenceDecisionPredictor
from src.core.decision_evidence import DecisionEvidenceStore
from test_decision_evidence import evidence, request, store_with_fact, typed_result


def fallback(store, predict, **kwargs):
    return EvidenceDecisionPredictor(
        store, cutoff=3, policy_version="v2", mode="fallback", predict=predict, **kwargs
    )


def test_should_answer_eligible_memory_without_calling_the_frozen_predictor():
    def forbidden(_):
        pytest.fail("known memory must not buy a callback")

    predictor = fallback(store_with_fact(), forbidden)
    result = predictor(request())
    assert result.selected_id == "inspect"
    assert predictor.report["calls"][0]["choice_source"] == "memory"
    assert predictor.report["work"] == dict(
        attempted=1, returned=1, validated=1, callback_attempted=0, callback_returned=0
    )


@pytest.mark.parametrize("scored", [False, True])
@pytest.mark.parametrize(
    "absence", ["empty", "future", "other-policy", "no-overlap", "inactive-target"]
)
def test_should_call_once_with_original_recipe_when_no_eligible_memory_exists(scored, absence):
    store = DecisionEvidenceStore()
    query = request()
    if absence == "future":
        store.add(evidence(recorded=4))
    elif absence == "other-policy":
        store.add(evidence(policy="v1"))
    elif absence == "no-overlap":
        store.add(evidence(query="password"))
    elif absence == "inactive-target":
        store.add(evidence())
        query = replace(
            query,
            allowed_candidates=tuple(
                c for c in query.allowed_candidates if c.candidate_id != "inspect"
            ),
        )
    received = []

    def predict(supplied):
        received.append(deepcopy(supplied))
        assert supplied == query and supplied is not query
        assert supplied.allowed_candidates[0] is not query.allowed_candidates[0]
        return typed_result(supplied, "route", probability_only=not scored)

    predictor = fallback(store, predict)
    original = deepcopy(query)
    result = predictor(query)
    assert result.selected_id == "route" and query == original and received == [query]
    row = predictor.report["calls"][0]
    assert row["choice_source"] == "fallback" and row["selection"]["items"] == ()
    assert row["supplied"] == row["request"]
    assert (
        predictor.report["work"]["callback_attempted"]
        == predictor.report["work"]["callback_returned"]
        == 1
    )
    assert store.work["retrieval_attempts"] == 1


def test_should_keep_a_valid_adverse_callback_choice_without_correcting_or_retrying_it():
    calls = []

    def wrong(supplied):
        calls.append(supplied.request_id)
        return typed_result(supplied, "route", probability_only=True)

    predictor = fallback(DecisionEvidenceStore(), wrong, max_calls=1)
    result = predictor(request("unseen invoice"))
    assert result.selected_id == "route" and calls == ["query"]
    with pytest.raises(ValueError, match="exhausted"):
        predictor(request("unseen invoice"))
    assert predictor.report["work"]["attempted"] == 1


@pytest.mark.parametrize(
    "fault", ["unreturned", "unsupported", "nonfinite", "request", "candidates", "store-write"]
)
def test_should_preserve_paid_callback_failures_and_stop_without_more_retrieval(fault):
    store = DecisionEvidenceStore()
    calls = []

    def predict(supplied):
        calls.append(supplied.request_id)
        if fault == "unreturned":
            raise RuntimeError("controlled unavailable-head return")
        if fault == "unsupported":
            return object()
        result = typed_result(supplied, probability_only=True)
        if fault == "nonfinite":
            object.__setattr__(result, "probabilities", (("route", float("nan")),))
        elif fault == "request":
            object.__setattr__(supplied, "query", "changed recipe")
        elif fault == "candidates":
            object.__setattr__(supplied.allowed_candidates[0], "description", "changed candidate")
        elif fault == "store-write":
            store.add(evidence(recorded=4))
        return result

    original = request()
    predictor = fallback(store, predict)
    with pytest.raises((ValueError, RuntimeError)):
        predictor(original)
    report = predictor.report
    assert report["work"]["callback_attempted"] == 1
    assert report["work"]["callback_returned"] == int(fault != "unreturned")
    assert report["work"]["validated"] == 0
    assert ("returned" in report["calls"][0]) is (fault != "unreturned")
    assert original == request()
    json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError, match="stopped"):
        predictor(original)
    assert store.work["retrieval_attempts"] == 1 and calls == ["query"]
    assert store.work["writes_completed"] == int(fault == "store-write")


def test_should_mix_memory_and_unknown_requests_with_one_shared_finite_allowance():
    calls = []

    def predict(supplied):
        calls.append(supplied.query)
        return typed_result(supplied, probability_only=True)

    store = store_with_fact()
    predictor = fallback(store, predict, max_calls=2)
    assert predictor(request()).selected_id == "inspect"
    assert predictor(request("unseen password")).selected_id == "route"
    before = deepcopy(predictor.report)
    before["calls"][0]["choice_source"] = "caller changed report"
    assert predictor.report["calls"][0]["choice_source"] == "memory"
    with pytest.raises(ValueError, match="exhausted"):
        predictor(request())
    assert calls == ["unseen password"]
    assert store.work["retrieval_attempts"] == 2
    assert predictor.report["work"] == dict(
        attempted=2, returned=2, validated=2, callback_attempted=1, callback_returned=1
    )


@pytest.mark.parametrize("predict", [None, 1, "not callable"])
def test_should_require_a_callback_before_buying_any_retrieval(predict):
    store = DecisionEvidenceStore()
    with pytest.raises(ValueError, match="require"):
        fallback(store, predict)
    assert store.work["retrieval_attempts"] == 0
