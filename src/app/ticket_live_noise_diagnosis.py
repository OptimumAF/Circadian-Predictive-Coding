"""Compare already-paid private task decisions from a bounded live pair.

Inputs are saved JSON records; outputs are detached scalar margins and common
observation differences. No IO, model reconstruction, teacher, prediction,
training, task rollout, evidence authentication or controller feedback.
"""

from copy import deepcopy
import json
from math import isfinite
from typing import Any

from src.core.typed_decision import (
    CandidateScore,
    DecisionCandidate,
    DecisionRequest,
    DecisionResult,
    validate_decision_result,
)

CONCEPTS = ("invoice", "parcel", "legacy", "password")
ARMS = ("clean", "retained_corruption")
LIMITS = dict(saved_decisions=48, saved_scores=384, paired_decisions=24, margin_subtractions=144)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any) -> dict[str, Any]:
    _require(type(value) is dict and len(value) <= 128, "expected bounded saved object")
    return value


def _records(value: Any, maximum: int) -> list[dict[str, Any]]:
    _require(type(value) is list and 1 <= len(value) <= maximum, "invalid saved record count")
    return [_mapping(row) for row in value]


def _same(left: Any, right: Any) -> bool:
    # JSON scalars retain bool/int/float distinctions without creating models.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def _detached_metadata(value):
    # Do not carry nonfinite/unserializable metadata into a successful report.
    json.dumps(value, allow_nan=False)
    return deepcopy(value)


def _charge(work: dict[str, int], key: str, amount: int = 1) -> None:
    work[key] += amount
    _require(work[key] <= LIMITS[key], "saved-score work exceeds finite bound")


def _difference(left: float, right: float, work: dict[str, int]) -> float:
    _charge(work, "margin_subtractions")
    value = left - right
    _require(isfinite(value), "saved-score margin is not finite")
    return value


def _decision(step: dict[str, Any], index: int, work: dict[str, int]) -> dict[str, Any]:
    _charge(work, "saved_decisions")
    _require(type(step["attempt"]) is int and step["attempt"] == index + 1, "invalid saved attempt")
    record = _mapping(step["result"])
    scores = _records(record["scores"], 8)
    _charge(work, "saved_scores", len(scores))
    data = _mapping(step["request"])
    request = DecisionRequest(
        data["request_id"],
        data["query"],
        tuple(DecisionCandidate(**c) for c in _records(data["allowed_candidates"], 8)),
    )
    result = DecisionResult(**{**record, "scores": tuple(CandidateScore(**c) for c in scores)})
    validate_decision_result(request, result)
    returned = _mapping(step["returned"])
    _require(
        returned["record_type"] == "DecisionResult" and _same(returned["observed"], record),
        "saved raw return and validated result differ",
    )
    ranked = sorted(result.scores, key=lambda s: (-s.raw_score, s.candidate_id))
    _require(
        ranked[0].candidate_id == result.selected_id, "saved selection differs from raw maximum"
    )
    competitors = [s for s in result.scores if s.candidate_id != result.selected_id]
    selected = next(s for s in result.scores if s.candidate_id == result.selected_id)
    return dict(
        attempt=index + 1,
        selected_id=result.selected_id,
        scores=deepcopy(scores),
        selected_raw_margin=None
        if not competitors
        else _difference(selected.raw_score, max(s.raw_score for s in competitors), work),
        selected_probability_margin=None
        if not competitors
        else _difference(selected.probability, max(s.probability for s in competitors), work),
    )


def _trace(row: dict[str, Any], concept: str, work: dict[str, int]):
    _require(
        row["concept"] == concept and row["role"] == "outer_selection", "wrong saved concept/role"
    )
    episode = _mapping(row["episode"])
    _require(
        episode["complete"] is True and type(episode["success"]) is bool,
        "saved task is incomplete or unknown",
    )
    _require(
        episode["scenario"] == "ordinary"
        and type(episode["max_steps"]) is int
        and episode["max_steps"] == 6,
        "wrong saved task scenario/bound",
    )
    _require(
        episode["unreturned_prediction_work_unknown"] is False
        and episode["unvalidated_return_work_unknown"] is False,
        "saved paid work is unknown",
    )
    steps = _records(episode["steps"], 6)
    paid = _mapping(episode["work"])
    _require(
        all(
            type(paid[k]) is int and paid[k] == len(steps)
            for k in (
                "prediction_calls_attempted",
                "prediction_calls_returned",
                "prediction_calls_validated",
            )
        ),
        "saved decision work differs from complete trace",
    )
    decisions = [_decision(step, i, work) for i, step in enumerate(steps)]
    inputs = [
        deepcopy(
            {
                key: _mapping(step[key])
                for key in ("request", "observation_state", "capabilities_observed")
            }
        )
        for step in steps
    ]
    return dict(concept=concept, success=episode["success"], decisions=decisions), inputs


def _arm(arm: dict[str, Any], index: int, work: dict[str, int]):
    _require(arm["name"] == ARMS[index], "wrong saved arm order")
    case = _mapping(arm["case"])
    _require(
        case["complete"] is True and case["definition"] == "live_catalog_bp_ticket_private_v1",
        "wrong or incomplete saved live case",
    )
    recipe = _mapping(arm["recipe"])
    _require(recipe["retained_invoice_noise"] is bool(index), "wrong saved corruption flag")
    _require(_same(recipe, case["live_training_recipe"]), "saved recipe records differ")
    corruption = _mapping(case["training_corruption"])
    _require(
        corruption["role"] == "train"
        and corruption["rows_per_batch"] == ([7] if index else [])
        and corruption["definition"] == "explicit_experimental_label_noise_not_verified_teacher",
        "saved corruption is not the marked TRAIN fixture",
    )
    recorded = _mapping(case["training_work"])
    updates = recorded["returned_updates"]
    _require(
        type(updates) is int
        and 0 <= updates <= 5
        and type(recorded["attempted_updates"]) is int
        and recorded["attempted_updates"] == updates
        and type(recorded["attempted_rows"]) is int
        and recorded["attempted_rows"] == 12 * updates
        and recorded["unreturned_work_unknown"] is False,
        "unknown or invalid saved exposure",
    )
    rows = _records(case["candidate_episodes"], 4)
    _require(len(rows) == 4, "all four saved private tasks required")
    parsed = [_trace(row, concept, work) for row, concept in zip(rows, CONCEPTS)]
    outcomes = _mapping(case["reference_outcomes"])
    for key in ("new_before", "new_after", "old_before", "old_after"):
        _require(type(outcomes[key]) is int and 0 <= outcomes[key] <= 2, "invalid saved task count")
    _require(
        outcomes["new_after"] == sum(parsed[i][0]["success"] for i in (2, 3))
        and outcomes["old_after"] == sum(parsed[i][0]["success"] for i in (0, 1)),
        "saved task counts disagree with recorded outcomes",
    )
    return dict(
        name=arm["name"],
        recipe=_detached_metadata(recipe),
        corruption=_detached_metadata(corruption),
        outcomes=_detached_metadata(outcomes),
        recorded_training=deepcopy(recorded),
        traces=[p[0] for p in parsed],
    ), [p[1] for p in parsed]


def _compare(concept, clean, noisy, clean_inputs, noisy_inputs, work):
    paired, first = 0, None
    for a, b, a_input, b_input in zip(
        clean["decisions"], noisy["decisions"], clean_inputs, noisy_inputs
    ):
        _charge(work, "paired_decisions")
        _require(_same(a_input, b_input), "saved observations differ before action divergence")
        paired += 1
        if a["selected_id"] != b["selected_id"]:
            left, right = a["selected_id"], b["selected_id"]
            a_scores = {s["candidate_id"]: s["raw_score"] for s in a["scores"]}
            b_scores = {s["candidate_id"]: s["raw_score"] for s in b["scores"]}
            first = dict(
                attempt=a["attempt"],
                input=deepcopy(a_input),
                clean_selected=left,
                retained_corruption_selected=right,
                noise_action_minus_clean_raw=dict(
                    clean=_difference(a_scores[right], a_scores[left], work),
                    retained_corruption=_difference(b_scores[right], b_scores[left], work),
                ),
            )
            # Subsequent states can differ because different actions took effect.
            break
    return dict(
        concept=concept,
        paired_decisions=paired,
        first_divergence=first,
        unpaired_steps=dict(
            clean=len(clean["decisions"]) - paired,
            retained_corruption=len(noisy["decisions"]) - paired,
        ),
    )


def diagnose_live_noise_pair(source: dict[str, Any], *, work: dict[str, int] | None = None):
    """Pure saved-score analysis; failed work remains in caller's counters."""
    if work is None:
        work = {}
    for key in LIMITS:
        work.setdefault(key, 0)
        _require(
            type(work[key]) is int and 0 <= work[key] <= LIMITS[key], "invalid saved work counter"
        )
    try:
        source = _mapping(source)
        _require(
            source["definition"] == "native_catalog_live_noise_pair_v1"
            and source["case_complete"] is True,
            "unsupported/incomplete saved pair",
        )
        arms = _records(source["arms"], 2)
        _require(len(arms) == 2, "two saved arms required")
        expected = dict(
            recipe_id="live_new_retained_noise",
            contexts=["new_contexts"] * 8,
            label_delay=2,
            noisy_invoice=False,
        )
        _require(
            all(_same(_mapping(a["recipe"])["recipe"], expected) for a in arms),
            "saved pair differs from original mixed/delay2 recipe",
        )
        parsed = [_arm(a, i, work) for i, a in enumerate(arms)]
        concepts = [
            _compare(
                c,
                parsed[0][0]["traces"][i],
                parsed[1][0]["traces"][i],
                parsed[0][1][i],
                parsed[1][1][i],
                work,
            )
            for i, c in enumerate(CONCEPTS)
        ]
        return dict(
            definition="ticket_live_saved_score_diagnosis_v1",
            complete=True,
            role="saved_candidate_scores_only",
            arms=[p[0] for p in parsed],
            concepts=concepts,
            work=deepcopy(work),
            qualification="recorded decisions and scalar subtractions only; no model parity, execution authentication, causal mechanism, tuning or corruption benefit",
        )
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("malformed saved live pair: " + str(error)) from error
