"""Analytic allowed-candidate retrieval and label-isolated scalar evaluation.

No neural model, RNG, training, simulator, final-role data or side effect is used.
Reference call count is bounded prospectively below the512-call fixture allowance.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from itertools import permutations
import json
import math
from typing import Any

import pytest

from src.core.typed_decision import (
    CandidateScore,
    DecisionCandidate,
    DecisionRequest,
    DecisionResult,
    validate_decision_result,
)
from src.core.decision_retrieval import retrieve_decision
from src.app.decision_reference_evaluation import (
    DecisionCase,
    DecisionEvaluationFailed,
    evaluate_decision_reference,
)
from scripts import run_decision_reference as cli


def request(identity="q", *, candidates=None):
    if candidates is None:
        candidates = (
            DecisionCandidate("a", "red blue"),
            DecisionCandidate("b", "red"),
            DecisionCandidate("c", "green"),
        )
    return DecisionRequest(identity, "RED blue", candidates)


def result_for(value, probabilities=(0.8, 0.2)):
    ordered = sorted(value.allowed_candidates, key=lambda item: item.candidate_id)
    scores = tuple(
        CandidateScore(item.candidate_id, 0.0, probability)
        for item, probability in zip(ordered, probabilities, strict=True)
    )
    selected = min(scores, key=lambda item: (-item.probability, item.candidate_id)).candidate_id
    return DecisionResult(value.request_id, selected, scores, 0, len(ordered))


def test_should_match_an_independent_jaccard_and_temperature_one_probability_oracle():
    value = request()
    result = retrieve_decision(value)
    denominator = math.fsum((math.exp(1.0), math.exp(0.5), 1.0))
    expected = (
        ("a", 1.0, math.exp(1.0) / denominator),
        ("b", 0.5, math.exp(0.5) / denominator),
        ("c", 0.0, 1.0 / denominator),
    )
    assert result.request_id == "q" and result.selected_id == "a"
    assert tuple(item.candidate_id for item in result.scores) == ("a", "b", "c")
    for actual, (identity, raw, probability) in zip(result.scores, expected, strict=True):
        assert actual.candidate_id == identity and actual.raw_score == raw
        assert actual.probability == pytest.approx(probability, rel=1e-14, abs=1e-15)
    assert result.token_visits == 6 and result.candidate_comparisons == 3
    validate_decision_result(value, result)


def test_should_return_the_same_canonical_scores_for_every_three_candidate_order():
    value = request()
    expected = retrieve_decision(value)
    for ordered in permutations(value.allowed_candidates):
        assert retrieve_decision(replace(value, allowed_candidates=ordered)) == expected


def test_should_tie_by_candidate_id_and_preserve_unicode_word_token_semantics():
    value = DecisionRequest(
        "unicode", "CAFÉ café", (DecisionCandidate("z", "café"), DecisionCandidate("a", "CAFÉ"))
    )
    result = retrieve_decision(value)
    assert result.selected_id == "a"
    assert [(item.candidate_id, item.raw_score, item.probability) for item in result.scores] == [
        ("a", 1.0, 0.5),
        ("z", 1.0, 0.5),
    ]
    assert result.token_visits == 4 and result.candidate_comparisons == 2


def test_should_freeze_native_records_and_select_from_probability_not_raw_score_order():
    value = request(candidates=(DecisionCandidate("a", "red"), DecisionCandidate("b", "blue")))
    result = DecisionResult(
        "q", "a", (CandidateScore("a", 0.0, 0.5), CandidateScore("b", 9.0, 0.5)), 0, 2
    )
    validate_decision_result(value, result)
    for owner, field, changed in (
        (value, "query", "changed"),
        (value.allowed_candidates[0], "description", "changed"),
        (result, "selected_id", "b"),
        (result.scores[0], "probability", 0.2),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(owner, field, changed)


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate",
        "empty",
        "nine",
        "list",
        "id-bool",
        "id-untrimmed",
        "id-bytes",
        "query-bytes",
        "description-bytes",
    ],
)
def test_should_refuse_malformed_native_requests_before_retrieval(fault):
    candidates = (DecisionCandidate("a", "red"),)
    with pytest.raises(ValueError):
        if fault == "duplicate":
            request(candidates=candidates * 2)
        elif fault == "empty":
            request(candidates=())
        elif fault == "nine":
            request(candidates=tuple(DecisionCandidate(str(i), "red") for i in range(9)))
        elif fault == "list":
            request(candidates=list(candidates))
        elif fault == "id-bool":
            bad: Any = True
            DecisionCandidate(bad, "red")
        elif fault == "id-untrimmed":
            DecisionCandidate(" a ", "red")
        elif fault == "id-bytes":
            DecisionCandidate("é" * 33, "red")
        elif fault == "query-bytes":
            DecisionRequest("q", "é" * 1025, candidates)
        else:
            DecisionCandidate("a", "é" * 513)


@pytest.mark.parametrize(
    "fault",
    [
        "raw-nan",
        "probability-bool",
        "probability-negative",
        "mass",
        "wrong-request",
        "outside-id",
        "duplicate-id",
        "missing-score",
        "wrong-selection",
        "negative-visits",
        "boolean-comparisons",
    ],
)
def test_should_revalidate_corrupted_frozen_result_fields_at_the_boundary(fault):
    value = request(candidates=(DecisionCandidate("a", "red"), DecisionCandidate("b", "blue")))
    result = result_for(value)
    if fault == "raw-nan":
        object.__setattr__(result.scores[0], "raw_score", float("nan"))
    elif fault == "probability-bool":
        object.__setattr__(result.scores[0], "probability", True)
    elif fault == "probability-negative":
        object.__setattr__(result.scores[1], "probability", -0.1)
    elif fault == "mass":
        object.__setattr__(result.scores[0], "probability", 0.7)
    elif fault == "wrong-request":
        object.__setattr__(result, "request_id", "other")
    elif fault == "outside-id":
        object.__setattr__(result.scores[1], "candidate_id", "outside")
    elif fault == "duplicate-id":
        object.__setattr__(result.scores[1], "candidate_id", "a")
    elif fault == "missing-score":
        object.__setattr__(result, "scores", result.scores[:1])
    elif fault == "wrong-selection":
        object.__setattr__(result, "selected_id", "b")
    elif fault == "negative-visits":
        object.__setattr__(result, "token_visits", -1)
    else:
        object.__setattr__(result, "candidate_comparisons", True)
    with pytest.raises(ValueError):
        validate_decision_result(value, result)


def forbidden(*args, **options):
    pytest.fail("invalid decision cases reached prediction")


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test", "inner_guard"])
def test_should_refuse_nontraining_selection_roles_before_predicting(role):
    with pytest.raises(ValueError):
        evaluate_decision_reference(
            (DecisionCase(request(), "a", "group", role),), predict=forbidden
        )


@pytest.mark.parametrize(
    "fault", ["empty", "list", "sixty-five", "duplicate-request", "cross-role-group"]
)
def test_should_refuse_ambiguous_or_unbounded_cases_before_predicting(fault):
    first = DecisionCase(request("q1"), "a", "same-group", "train")
    cases: Any
    if fault == "empty":
        cases = ()
    elif fault == "list":
        cases = [first]
    elif fault == "sixty-five":
        cases = tuple(DecisionCase(request(f"q{i}"), "a", "group") for i in range(65))
    elif fault == "duplicate-request":
        cases = (first, first)
    else:
        cases = (first, DecisionCase(request("q2"), "a", "same-group", "outer_selection"))
    with pytest.raises(ValueError):
        evaluate_decision_reference(cases, predict=forbidden)


def test_should_keep_labels_out_of_detached_predictor_inputs_and_include_missing_targets_in_all_metrics():
    allowed = (DecisionCandidate("a", "red"), DecisionCandidate("b", "blue"))
    cases = (
        DecisionCase(request("present", candidates=allowed), "a", "training-source", "train"),
        DecisionCase(request("missing", candidates=allowed), "not-recalled", "evaluation-source"),
    )
    originals = {case.request.request_id: case.request for case in cases}
    seen: list[DecisionRequest] = []

    def predict(value):
        assert type(value) is DecisionRequest and not hasattr(value, "target_id")
        original = originals[value.request_id]
        assert value is not original and value.allowed_candidates is not original.allowed_candidates
        assert all(
            item is not owner
            for item in value.allowed_candidates
            for owner in original.allowed_candidates
        )
        seen.append(value)
        return result_for(value)

    before = deepcopy(cases)
    report = evaluate_decision_reference(cases, predict=predict)
    assert cases == before and len(seen) == 4
    assert report["complete"] is True and report["learning_active"] is None
    assert report["training_updates"] is report["external_actions"] is None
    assert report["predictor_behavior"] == "UNVERIFIED_CALLBACK_BEHAVIOR"
    assert report["method"] == "caller_supplied_reference"
    assert report["candidate_order_control"] == dict(
        type="reverse_order_only", cases=2, all_equal=True
    )
    metrics = report["metrics"]
    assert metrics["accuracy"] == metrics["candidate_recall"] == 0.5
    assert [row["brier"] for row in report["cases"]] == pytest.approx([0.08, 1.68])
    assert metrics["brier_mean"] == pytest.approx(0.88)
    assert (
        metrics["nll_mean"] is None
        and metrics["nll_status"] == "infinite_due_to_zero_target_probability"
    )
    assert metrics["zero_target_probability_count"] == 1
    assert metrics["ece_5bin"] == pytest.approx(0.3)
    assert [row["support"] for row in metrics["calibration_bins"]] == [0, 0, 0, 0, 2]
    assert (
        report["work"]["prediction_calls_attempted"]
        == report["work"]["prediction_calls_returned"]
        == 4
    )
    assert "unverified" in report["work"]["counter_scope"]


def test_should_report_finite_nll_when_every_target_has_positive_mass():
    value = request(candidates=(DecisionCandidate("a", "red"), DecisionCandidate("b", "blue")))
    report = evaluate_decision_reference((DecisionCase(value, "a", "group"),), predict=result_for)
    assert report["metrics"]["nll_status"] == "finite"
    assert report["metrics"]["nll_mean"] == pytest.approx(-math.log(0.8))
    assert report["metrics"]["zero_target_probability_count"] == 0


def test_should_keep_a_recalled_zero_probability_target_as_an_infinite_nll_error():
    value = request(candidates=(DecisionCandidate("a", "red"), DecisionCandidate("b", "blue")))
    report = evaluate_decision_reference(
        (DecisionCase(value, "b", "group"),), predict=lambda item: result_for(item, (1.0, 0.0))
    )
    assert report["metrics"]["candidate_recall"] == 1.0
    assert report["metrics"]["accuracy"] == 0.0 and report["metrics"]["brier_mean"] == 2.0
    assert (
        report["metrics"]["nll_mean"] is None
        and report["metrics"]["zero_target_probability_count"] == 1
    )
    assert report["metrics"]["ece_5bin"] == 1.0


def test_should_use_all_five_fixed_calibration_bins_and_keep_misses_in_support():
    probabilities = ((1 / 6,) * 6, (0.2,) * 5, (0.4, 0.3, 0.3), (0.6, 0.4), (0.8, 0.2), (1.0,))
    targets = ("a", "a", "missing", "a", "missing", "a")
    cases = tuple(
        DecisionCase(
            DecisionRequest(
                f"bin{i}",
                "unmatched query",
                tuple(DecisionCandidate(chr(97 + j), "description") for j in range(len(vector))),
            ),
            target,
            f"group-{i}",
        )
        for i, (vector, target) in enumerate(zip(probabilities, targets, strict=True))
    )
    report = evaluate_decision_reference(
        cases, predict=lambda value: result_for(value, probabilities[int(value.request_id[3:])])
    )
    bins = report["metrics"]["calibration_bins"]
    assert [(row["lower"], row["upper"]) for row in bins] == [
        (0.0, 0.2),
        (0.2, 0.4),
        (0.4, 0.6),
        (0.6, 0.8),
        (0.8, 1.0),
    ]
    assert [row["support"] for row in bins] == [1, 1, 1, 1, 2]
    assert sum(row["support"] for row in bins) == 6
    assert bins[-1]["mean_confidence"] == pytest.approx(0.9)
    assert bins[-1]["accuracy"] == 0.5
    assert report["metrics"]["ece_5bin"] == pytest.approx(97 / 180)


def test_should_refuse_a_predictor_that_changes_the_supplied_request_before_returning():
    case = DecisionCase(request(), "a", "group")
    calls: list[DecisionRequest] = []

    def predict(value):
        calls.append(value)
        result = retrieve_decision(value)
        object.__setattr__(value, "query", "changed but still valid")
        return result

    with pytest.raises(ValueError, match="changed request"):
        evaluate_decision_reference((case,), predict=predict)
    assert len(calls) == 1 and case.request.query == "RED blue"


def test_should_refuse_a_valid_but_candidate_order_dependent_predictor():
    case = DecisionCase(
        request(candidates=(DecisionCandidate("a", "red"), DecisionCandidate("b", "blue"))),
        "a",
        "group",
    )
    calls: list[DecisionRequest] = []

    def predict(value):
        calls.append(value)
        first = value.allowed_candidates[0].candidate_id
        scores = tuple(
            CandidateScore(item.candidate_id, 0.0, 0.9 if item.candidate_id == first else 0.1)
            for item in sorted(value.allowed_candidates, key=lambda item: item.candidate_id)
        )
        return DecisionResult(value.request_id, first, scores, 0, 2)

    with pytest.raises(ValueError, match="order"):
        evaluate_decision_reference((case,), predict=predict)
    assert len(calls) == 2


def test_should_score_the_admitted_snapshot_when_a_callback_changes_original_caller_records():
    original = DecisionCase(
        request(candidates=(DecisionCandidate("a", "red"), DecisionCandidate("b", "blue"))),
        "a",
        "group",
    )
    calls: list[DecisionRequest] = []

    def predict(value):
        calls.append(value)
        object.__setattr__(original, "target_id", "not-recalled")
        object.__setattr__(original.request, "query", "caller changed after admission")
        return result_for(value)

    report = evaluate_decision_reference((original,), predict=predict)
    assert len(calls) == 2 and original.target_id == "not-recalled"
    assert report["cases"][0]["target_id"] == "a"
    assert report["cases"][0]["request"]["query"] == "RED blue"
    assert report["cases"][0]["correct"] is True
    assert all(value.query == "RED blue" for value in calls)


def two_cases():
    allowed = (DecisionCandidate("a", "red"), DecisionCandidate("b", "blue"))
    return (
        DecisionCase(request("first", candidates=allowed), "a", "group-1"),
        DecisionCase(request("second", candidates=allowed), "a", "group-2"),
    )


@pytest.fixture(scope="module")
def interrupted_evaluation():
    calls: list[DecisionRequest] = []

    def predict(value):
        calls.append(value)
        if len(calls) == 4:
            raise RuntimeError("reversed callback did not return")
        return result_for(value)

    with pytest.raises(DecisionEvaluationFailed) as failure:
        evaluate_decision_reference(two_cases(), predict=predict)
    return failure.value, calls


def test_should_preserve_completed_and_pending_cases_with_exact_unreturned_callback_charges(
    interrupted_evaluation,
):
    error, calls = interrupted_evaluation
    report, work = error.report, error.work
    assert len(calls) == 4 and report["complete"] is False
    assert "reversed callback did not return" in report["error"]
    assert len(report["cases"]) == 1 and report["cases"][0]["request"]["request_id"] == "first"
    assert report["pending_case"]["request_id"] == "second"
    pending = report["pending_case"]["original_returned"]
    assert pending["record_type"] == "DecisionResult" and pending["validated"] is True
    assert pending["observed"]["request_id"] == "second"
    assert work["prediction_calls_attempted"] == 4
    assert work["prediction_calls_returned"] == work["prediction_calls_validated"] == 3
    assert work["unreturned_prediction_work_unknown"] is True
    assert work["unvalidated_return_work_unknown"] is False
    assert work["reported"]["original"]["candidate_comparisons"] == 4
    assert work["reported"]["reversed"]["candidate_comparisons"] == 2
    work["prediction_calls_attempted"] = 0
    assert (
        error.work["prediction_calls_attempted"]
        == error.report["work"]["prediction_calls_attempted"]
        == 4
    )


def test_should_charge_a_malformed_paid_return_and_stop_before_the_next_control():
    calls: list[DecisionRequest] = []

    def predict(value):
        calls.append(value)
        return None if len(calls) == 3 else result_for(value)

    with pytest.raises(DecisionEvaluationFailed) as failure:
        evaluate_decision_reference(two_cases(), predict=predict)
    report, work = failure.value.report, failure.value.work
    assert len(calls) == 3 and report["complete"] is False
    assert len(report["cases"]) == 1 and report["pending_case"]["request_id"] == "second"
    assert work["prediction_calls_attempted"] == work["prediction_calls_returned"] == 3
    assert work["prediction_calls_validated"] == 2
    assert work["unreturned_prediction_work_unknown"] is False
    assert work["unvalidated_return_work_unknown"] is True
    assert sum(value["candidate_comparisons"] for value in work["reported"].values()) == 4


def test_should_preserve_other_paid_fields_when_a_returned_frozen_counter_was_deleted():
    case = DecisionCase(
        request(candidates=(DecisionCandidate("a", "red"), DecisionCandidate("b", "blue"))),
        "a",
        "group",
    )

    def predict(value):
        result = result_for(value)
        object.__delattr__(result, "token_visits")
        return result

    with pytest.raises(DecisionEvaluationFailed) as failure:
        evaluate_decision_reference((case,), predict=predict)
    report, work = failure.value.report, failure.value.work
    assert report["complete"] is False and report["cases"] == []
    assert work["prediction_calls_attempted"] == work["prediction_calls_returned"] == 1
    assert work["prediction_calls_validated"] == 0
    assert work["unvalidated_return_work_unknown"] is True
    returned = report["pending_case"]["original_returned"]
    assert returned["validated"] is False
    assert returned["observed"]["token_visits"] == {"capture": "unavailable: missing field"}
    assert returned["observed"]["request_id"] == "q"
    assert returned["observed"]["candidate_comparisons"] == 2


def test_should_publish_the_partial_cli_report_once_without_retrying_prediction(
    monkeypatch, tmp_path, interrupted_evaluation
):
    error, _ = interrupted_evaluation
    calls: list[tuple[DecisionCase, ...]] = []

    def evaluate(cases):
        calls.append(cases)
        raise error

    output = tmp_path / "partial.json"
    monkeypatch.setattr(cli, "evaluate_decision_reference", evaluate)
    with pytest.raises(SystemExit) as exit_status:
        cli.main(["--output", str(output)])
    assert exit_status.value.code == 1 and len(calls) == 1
    saved = json.loads(output.read_text(encoding="utf8"))
    # JSON arrays preserve the report's tuple contents, not Python container types.
    assert saved == json.loads(json.dumps(error.report, allow_nan=False))
    assert saved["complete"] is False
    assert saved["work"]["prediction_calls_attempted"] == 4


@pytest.fixture(scope="module")
def example_report(tmp_path_factory):
    path = tmp_path_factory.mktemp("decision-reference") / "six-cases.json"
    cli.main(["--output", str(path)])
    return path, json.loads(path.read_text(encoding="utf8"))


def test_should_preserve_the_frozen_six_case_example_and_all_three_errors(example_report):
    path, report = example_report
    expected = (
        ("invoice refund", "billing", ("billing", "password", "shipping")),
        ("password reset", "password", ("billing", "password", "shipping")),
        ("delivery tracking", "shipping", ("billing", "password", "shipping")),
        ("forgot credentials", "password", ("billing", "password", "shipping")),
        ("password reset", "password", ("billing", "shipping")),
        ("refund payment", "billing", ("password", "shipping")),
    )
    assert (
        tuple(
            (
                case.request.query,
                case.target_id,
                tuple(c.candidate_id for c in case.request.allowed_candidates),
            )
            for case in cli.example_cases()
        )
        == expected
    )
    assert report["complete"] is True and len(report["cases"]) == 6
    assert [row["correct"] for row in report["cases"]] == [True, True, True, False, False, False]
    assert report["metrics"]["accuracy"] == 0.5
    assert report["metrics"]["candidate_recall"] == pytest.approx(4 / 6)
    assert report["metrics"]["zero_target_probability_count"] == 2
    assert (
        report["work"]["prediction_calls_attempted"]
        == report["work"]["prediction_calls_returned"]
        == 12
    )
    assert sum(row["candidate_comparisons"] for row in report["work"]["reported"].values()) == 32
    assert sum(row["token_visits"] for row in report["work"]["reported"].values()) == 152
    assert path.is_file() and "not trained" in report["qualification"]


def test_should_refuse_existing_cli_output_before_prediction(monkeypatch, example_report):
    path, _ = example_report
    original = path.read_bytes()
    monkeypatch.setattr(cli, "evaluate_decision_reference", forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main(["--output", str(path)])
    assert error.value.code == 2 and path.read_bytes() == original


def test_should_print_the_public_cli_report_without_an_output_file(monkeypatch, capsys):
    report = dict(complete=True, training_updates=0, external_actions=0)
    calls: list[tuple[DecisionCase, ...]] = []

    def evaluate(cases):
        calls.append(cases)
        return deepcopy(report)

    monkeypatch.setattr(cli, "evaluate_decision_reference", evaluate)
    cli.main([])
    assert len(calls) == 1 and len(calls[0]) == 6
    assert json.loads(capsys.readouterr().out) == report
