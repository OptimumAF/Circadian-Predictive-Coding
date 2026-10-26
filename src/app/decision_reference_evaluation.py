"""Bounded typed-routing evaluation with labels separate from prediction.

Reports ordinary retrieval errors, candidate recall and descriptive calibration.
No training, selection, final evaluation, environment actions or IO occurs here.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from math import fsum, isclose, isfinite, log
from typing import Any

from src.core.decision_retrieval import retrieve_decision
from src.core.experience import ExperienceRole
from src.core.typed_decision import (
    CandidateScore,
    DecisionRequest,
    DecisionResult,
    ProbabilityDecisionResult,
    _text,
    validate_decision_result,
    validate_probability_decision_result,
)

_ABSENT = object()


@dataclass(frozen=True)
class DecisionCase:
    request: DecisionRequest
    target_id: str
    source_group: str
    role: ExperienceRole = "outer_selection"

    def __post_init__(self) -> None:
        if type(self.request) is not DecisionRequest:
            raise ValueError("expected exact DecisionRequest")
        self.request.__post_init__()
        _text(self.target_id, "target_id", 64)
        _text(self.source_group, "source_group", 64)
        if type(self.role) is not str or self.role not in ("train", "outer_selection"):
            raise ValueError(
                "reference scoring accepts train/outer_selection only; final and probe roles refused"
            )


def _admit(cases: tuple[DecisionCase, ...]) -> None:
    if type(cases) is not tuple or not 1 <= len(cases) <= 64:
        raise ValueError("cases must be an immutable tuple of1..64 DecisionCase records")
    groups: dict[str, str] = {}
    ids = []
    for case in cases:
        if type(case) is not DecisionCase:
            raise ValueError("expected exact DecisionCase")
        case.__post_init__()
        ids.append(case.request.request_id)
        if case.source_group in groups and groups[case.source_group] != case.role:
            raise ValueError("source groups must be disjoint across train/evaluation roles")
        groups[case.source_group] = case.role
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate request ID")


class DecisionEvaluationFailed(ValueError):
    """Detached paid prefix plus at most one pending case; not a resumable cursor."""

    def __init__(self, report: dict[str, Any]):
        self.report = deepcopy(report)
        super().__init__(self.report["error"])

    @property
    def work(self) -> dict[str, Any]:
        return deepcopy(self.report["work"])


def _observed_scalar(value: Any) -> Any:
    # Why this: rejected returns must not execute deepcopy/repr on arbitrary objects.
    if value is _ABSENT:
        return {"capture": "unavailable: missing field"}
    if type(value) is bool or value is None:
        return value
    if type(value) is int and abs(value) <= 2**53 - 1:
        return value
    if type(value) is float and isfinite(value):
        return value
    if type(value) is str and len(value) <= 2048:
        return value
    return {"capture": "unavailable: unsupported, nonfinite or outside snapshot bound"}


def _observed_return(result: Any) -> dict[str, Any]:
    if type(result) is not DecisionResult:
        return dict(record_type="unsupported", payload_unavailable=True)
    raw = {
        key: _observed_scalar(getattr(result, key, _ABSENT))
        for key in ("request_id", "selected_id", "token_visits", "candidate_comparisons")
    }
    scores = getattr(result, "scores", _ABSENT)
    if (
        type(scores) is tuple
        and len(scores) <= 8
        and all(type(item) is CandidateScore for item in scores)
    ):
        raw["scores"] = [
            {
                key: _observed_scalar(getattr(item, key, _ABSENT))
                for key in ("candidate_id", "raw_score", "probability")
            }
            for item in scores
        ]
    else:
        raw["scores"] = dict(capture="unavailable: invalid score records")
    return dict(record_type="DecisionResult", observed=raw, validated=False)


def _predict(
    predict: Callable[[DecisionRequest], DecisionResult],
    request: DecisionRequest,
    report: dict[str, Any],
    phase: str,
) -> DecisionResult:
    supplied = deepcopy(request)
    work = report["work"]
    work["prediction_calls_attempted"] += 1
    result = predict(supplied)
    work["prediction_calls_returned"] += 1
    report["pending_case"][phase + "_returned"] = _observed_return(result)
    if type(supplied) is not DecisionRequest:
        raise ValueError("predictor changed request type")
    supplied.__post_init__()
    if supplied != request:
        raise ValueError("predictor changed request")
    validate_decision_result(request, result)
    owned = deepcopy(result)
    work["prediction_calls_validated"] += 1
    for key in ("token_visits", "candidate_comparisons"):
        work["reported"][phase][key] += getattr(owned, key)
    report["pending_case"][phase + "_returned"]["validated"] = True
    return owned


def _canonical(result: DecisionResult) -> tuple:
    return result.selected_id, tuple(
        sorted((item.candidate_id, item.raw_score, item.probability) for item in result.scores)
    )


def _score_case(
    case: DecisionCase, result: DecisionResult, control: DecisionResult
) -> dict[str, Any]:
    return _probability_row(
        case,
        result.selected_id,
        {item.candidate_id: item.probability for item in result.scores},
        asdict(result),
        asdict(control),
    )


def _probability_row(
    case: DecisionCase,
    selected_id: str,
    probabilities: dict[str, float],
    result: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, Any]:
    recalled = case.target_id in probabilities
    target_probability = probabilities.get(case.target_id, 0.0)
    # Why this: missing candidate labels remain wrong and contribute to Brier/NLL.
    brier = fsum(
        (value - (identity == case.target_id)) ** 2 for identity, value in probabilities.items()
    ) + (0 if recalled else 1)
    return dict(
        request=asdict(case.request),
        target_id=case.target_id,
        source_group=case.source_group,
        role=case.role,
        result=result,
        reversed_result=control,
        candidate_recalled=recalled,
        correct=selected_id == case.target_id,
        target_probability=target_probability,
        brier=brier,
        confidence=max(probabilities.values()),
    )


def _calibration(rows: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    bins = []
    gaps = []
    for index in range(5):
        selected = [row for row in rows if min(4, int(row["confidence"] * 5)) == index]
        support = len(selected)
        confidence: float | None = None
        accuracy: float | None = None
        gap: float | None = None
        if support:
            confidence = fsum(row["confidence"] for row in selected) / support
            accuracy = sum(1 for row in selected if row["correct"]) / support
            gap = abs(confidence - accuracy)
            gaps.append(support / len(rows) * gap)
        bins.append(
            dict(
                lower=index / 5,
                upper=(index + 1) / 5,
                support=support,
                mean_confidence=confidence,
                accuracy=accuracy,
                absolute_gap=gap,
            )
        )
    return fsum(gaps), bins


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    zero = sum(row["target_probability"] == 0 for row in rows)
    ece, bins = _calibration(rows)
    return dict(
        accuracy=sum(row["correct"] for row in rows) / count,
        candidate_recall=sum(row["candidate_recalled"] for row in rows) / count,
        brier_mean=fsum(row["brier"] for row in rows) / count,
        nll_mean=None if zero else fsum(-log(row["target_probability"]) for row in rows) / count,
        nll_status="infinite_due_to_zero_target_probability" if zero else "finite",
        zero_target_probability_count=zero,
        ece_5bin=ece,
        calibration_bins=bins,
    )


def _new_report(
    predict: Callable[[DecisionRequest], DecisionResult | ProbabilityDecisionResult],
) -> dict[str, Any]:
    known = predict is retrieve_decision
    report: dict[str, Any] = dict(
        definition="typed_decision_retrieval_reference_v1",
        complete=False,
        method="lexical_jaccard_softmax_v1" if known else "caller_supplied_reference",
        learning_active=False if known else None,
        training_updates=0 if known else None,
        external_actions=0 if known else None,
        predictor_behavior="stateless_lexical_reference"
        if known
        else "UNVERIFIED_CALLBACK_BEHAVIOR",
        evaluator_owned_work=dict(training_updates=0, external_actions=0, model_initializations=0),
        cases=[],
        pending_case=None,
        work=dict(
            prediction_calls_attempted=0,
            prediction_calls_returned=0,
            prediction_calls_validated=0,
            unreturned_prediction_work_unknown=False,
            unvalidated_return_work_unknown=False,
            reported={
                name: dict(token_visits=0, candidate_comparisons=0)
                for name in ("original", "reversed")
            },
            counter_scope="validated predictor-reported logical work only; injected callback behavior unverified; excludes copies/validation/metrics/overhead",
        ),
        qualification="small descriptive lexical control, not trained or calibrated modern-model/agent/adaptation/CPC evidence; no final evaluation",
    )
    return report


def _observed_probability_return(result: Any) -> dict[str, Any]:
    if type(result) is not ProbabilityDecisionResult:
        return dict(record_type="unsupported", payload_unavailable=True)
    raw = {
        name: _observed_scalar(getattr(result, name, _ABSENT))
        for name in (
            "request_id",
            "selected_id",
            "reported_confidence",
            "reported_input_tokens",
            "reported_output_tokens",
        )
    }
    values = getattr(result, "probabilities", _ABSENT)
    if type(values) is tuple and len(values) <= 8:
        raw["probabilities"] = [
            [_observed_scalar(item[0]), _observed_scalar(item[1])]
            if type(item) is tuple and len(item) == 2
            else dict(capture="unavailable: invalid probability pair")
            for item in values
        ]
    else:
        raw["probabilities"] = dict(capture="unavailable: invalid probabilities")
    return dict(record_type="ProbabilityDecisionResult", observed=raw, validated=False)


def _predict_probabilities(predict, request, report, phase):
    supplied = deepcopy(request)
    work = report["work"]
    work["prediction_calls_attempted"] += 1
    result = predict(supplied)
    work["prediction_calls_returned"] += 1
    report["pending_case"][phase + "_returned"] = _observed_probability_return(result)
    if type(supplied) is not DecisionRequest:
        raise ValueError("predictor changed request type")
    supplied.__post_init__()
    if supplied != request:
        raise ValueError("predictor changed request")
    validate_probability_decision_result(request, result)
    owned = deepcopy(result)
    work["prediction_calls_validated"] += 1
    reported = work["reported"][phase]
    for name in ("input_tokens", "output_tokens"):
        value = getattr(owned, "reported_" + name)
        if value is None:
            reported[name + "_unknown_calls"] += 1
        else:
            reported[name] += value
    report["pending_case"][phase + "_returned"]["validated"] = True
    return owned


def _order_probabilities(first, second):
    a, b = dict(first.probabilities), dict(second.probabilities)
    difference = max(abs(a[key] - b[key]) for key in a)
    probabilities_agree = all(isclose(a[key], b[key], abs_tol=1e-6, rel_tol=0) for key in a)
    choice_agrees = first.selected_id == second.selected_id
    return dict(
        probabilities_agree=probabilities_agree,
        choice_agrees=choice_agrees,
        agreement=probabilities_agree and choice_agrees,
        max_absolute_probability_difference=difference,
    )


def evaluate_probability_reference(
    cases: tuple[DecisionCase, ...],
    *,
    predict: Callable[[DecisionRequest], ProbabilityDecisionResult],
) -> dict[str, Any]:
    """Evaluate observed probabilities; no raw-score or work fabrication.

    Native margin confidence is descriptive provider data only. ECE uses max p,
    sharing the existing bins/metrics. Callback behavior remains unsandboxed.
    Valid candidate-order sensitivity is reported rather than discarded.
    """
    _admit(cases)
    if not callable(predict):
        raise ValueError("predict must be callable")
    owned = deepcopy(cases)
    report = _new_report(predict)
    report.update(
        definition="probability_decision_reference_v1",
        method="caller_supplied_probability_reference",
        qualification="descriptive probability scoring; not calibration, adaptation or CPC benefit; no final evaluation",
    )
    report["work"]["reported"] = {
        name: dict(
            input_tokens=0,
            output_tokens=0,
            input_tokens_unknown_calls=0,
            output_tokens_unknown_calls=0,
        )
        for name in ("original", "reversed")
    }
    report["work"]["counter_scope"] = (
        "validated predictor-reported tokens only; callback behavior and physical work unverified; unknown calls separate from zero"
    )
    try:
        for case in owned:
            report["pending_case"] = dict(
                request_id=case.request.request_id, role=case.role, source_group=case.source_group
            )
            result = _predict_probabilities(predict, case.request, report, "original")
            reverse = DecisionRequest(
                case.request.request_id,
                case.request.query,
                tuple(reversed(case.request.allowed_candidates)),
            )
            control = _predict_probabilities(predict, reverse, report, "reversed")
            row = _probability_row(
                case,
                result.selected_id,
                dict(result.probabilities),
                asdict(result),
                asdict(control),
            )
            row["order_control"] = _order_probabilities(result, control)
            report["cases"].append(row)
            report["pending_case"] = None
        report["candidate_order_control"] = dict(
            type="reverse_order_only",
            cases=len(owned),
            all_equal=all(row["order_control"]["agreement"] for row in report["cases"]),
        )
        report["metrics"] = _metrics(report["cases"])
        report["complete"] = True
    except Exception as error:
        work = report["work"]
        work["unreturned_prediction_work_unknown"] = (
            work["prediction_calls_attempted"] > work["prediction_calls_returned"]
        )
        work["unvalidated_return_work_unknown"] = (
            work["prediction_calls_returned"] > work["prediction_calls_validated"]
        )
        report["error"] = type(error).__name__ + ": " + str(error)[:512]
        raise DecisionEvaluationFailed(report) from error
    return report


def evaluate_decision_reference(
    cases: tuple[DecisionCase, ...],
    *,
    predict: Callable[[DecisionRequest], DecisionResult] = retrieve_decision,
) -> dict[str, Any]:
    """Score a private admitted snapshot; callbacks are unsandboxed prediction ports."""
    _admit(cases)
    if not callable(predict):
        raise ValueError("predict must be callable")
    owned = deepcopy(cases)
    report = _new_report(predict)
    try:
        for case in owned:
            report["pending_case"] = dict(
                request_id=case.request.request_id, role=case.role, source_group=case.source_group
            )
            result = _predict(predict, case.request, report, "original")
            reverse = DecisionRequest(
                case.request.request_id,
                case.request.query,
                tuple(reversed(case.request.allowed_candidates)),
            )
            control = _predict(predict, reverse, report, "reversed")
            if _canonical(result) != _canonical(control):
                raise ValueError("candidate-order control changed canonical result")
            report["cases"].append(_score_case(case, result, control))
            report["pending_case"] = None
        report["candidate_order_control"] = dict(
            type="reverse_order_only", cases=len(owned), all_equal=True
        )
        report["metrics"] = _metrics(report["cases"])
        report["complete"] = True
    except Exception as error:
        work = report["work"]
        work["unreturned_prediction_work_unknown"] = (
            work["prediction_calls_attempted"] > work["prediction_calls_returned"]
        )
        work["unvalidated_return_work_unknown"] = (
            work["prediction_calls_returned"] > work["prediction_calls_validated"]
        )
        report["error"] = type(error).__name__ + ": " + str(error)[:512]
        raise DecisionEvaluationFailed(report) from error
    return report
