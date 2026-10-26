"""Bounded retrieval, context or evidence-first fallback typed decision calls.

Uses a causal TRAIN evidence store and an optional trusted frozen callback.
Preserves paid returns; never fits, promotes, retries or repairs a model choice.
"""

from copy import deepcopy
from dataclasses import asdict, fields, replace
from typing import Any

from src.app.decision_reference_evaluation import _observed_probability_return, _observed_return
from src.core.decision_evidence import (
    DecisionEvidence,
    DecisionEvidenceStore,
    EvidenceRevision,
    EvidenceSelection,
    _exact,
)
from src.core.experience import Experience, ExperiencePermissions, LabelArrival
from src.core.typed_decision import (
    DecisionCandidate,
    DecisionRequest,
    DecisionResult,
    ProbabilityDecisionResult,
    _count,
    _text,
    validate_decision_result,
    validate_probability_decision_result,
)


def apply_decision_feedback(
    store: DecisionEvidenceStore,
    experience: Experience[DecisionRequest],
    label: LabelArrival[str],
    *,
    policy_version: str,
    source_group: str,
    recorded_at: int,
) -> EvidenceRevision:
    """Add separately consented raw feedback without changing source permissions.

    Why this: projected training inbox consent cannot authorize memory reuse.
    The caller supplies the original raw observation and its released label.
    """
    if type(store) is not DecisionEvidenceStore:
        raise ValueError("expected exact DecisionEvidenceStore")
    _exact(experience, Experience)
    _exact(label, LabelArrival)
    _exact(experience.permissions, ExperiencePermissions)
    _exact(experience.features, DecisionRequest)
    for candidate in experience.features.allowed_candidates:
        _exact(candidate, DecisionCandidate)
    if experience.role != "train" or label.role != "train":
        raise ValueError("decision feedback requires TRAIN observation and label")
    if not experience.permissions.training or not experience.permissions.replay:
        raise ValueError("decision feedback requires original TRAIN and reuse consent")
    if experience.key != label.key or experience.model_version != label.model_version:
        raise ValueError("feedback key and model version must match the observation")
    if experience.features.request_id != experience.sample_id:
        raise ValueError("raw request ID must match the observed sample ID")
    if experience.candidate_ids != tuple(
        candidate.candidate_id for candidate in experience.features.allowed_candidates
    ):
        raise ValueError("observed candidates must match the original allowed IDs")
    evidence = DecisionEvidence(
        evidence_id=experience.sample_id,
        observed_at=experience.observed_at,
        available_at=label.arrived_at,
        recorded_at=recorded_at,
        policy_version=policy_version,
        source_group=source_group,
        request=experience.features,
        target_id=label.targets,
        role=experience.role,
        permissions=experience.permissions,
    )
    return store.add(evidence)


def decision_from_evidence(
    request: DecisionRequest, selection: EvidenceSelection
) -> ProbabilityDecisionResult | None:
    """Decode a store selection; no eligible item is explicitly unavailable."""
    _exact(request, DecisionRequest)
    for candidate in request.allowed_candidates:
        _exact(candidate, DecisionCandidate)
    if type(selection) is not EvidenceSelection or set(vars(selection)) != {
        field.name for field in fields(EvidenceSelection)
    }:
        raise ValueError("expected exact complete EvidenceSelection")
    if type(selection.items) is not tuple or len(selection.items) > 8:
        raise ValueError("evidence selection requires at most eight immutable items")
    if type(selection.context) is not str or len(selection.context.encode("utf8")) > 1536:
        raise ValueError("evidence selection context exceeds its native text bound")
    if type(selection.work) is not dict or set(selection.work) != {
        "metadata_scans",
        "candidate_comparisons",
        "token_visits",
        "rendered_bytes",
    }:
        raise ValueError("evidence selection requires original store work fields")
    for name, value in selection.work.items():
        _count(value, name)
    for item in selection.items:
        if type(item) is not EvidenceRevision or set(vars(item)) != {
            field.name for field in fields(EvidenceRevision)
        }:
            raise ValueError("expected exact complete EvidenceRevision")
        _count(item.revision, "revision")
        _exact(item.evidence, DecisionEvidence)
    if not selection.items:
        return None
    selected = selection.items[0].evidence.target_id
    result = ProbabilityDecisionResult(
        request.request_id,
        selected,
        tuple(
            (candidate.candidate_id, float(candidate.candidate_id == selected))
            for candidate in request.allowed_candidates
        ),
    )
    validate_probability_decision_result(request, result)
    return result


class EvidenceDecisionPredictor:
    def __init__(
        self, store, *, cutoff, policy_version, mode="retrieval", predict=None, max_calls=64
    ):
        if type(store) is not DecisionEvidenceStore:
            raise ValueError("expected exact DecisionEvidenceStore")
        _count(cutoff, "cutoff")
        _text(policy_version, "policy_version", 64)
        if mode not in ("retrieval", "context", "fallback") or type(mode) is not str:
            raise ValueError("mode must be retrieval, context or fallback")
        if (mode in ("context", "fallback") and not callable(predict)) or (
            mode == "retrieval" and predict is not None
        ):
            raise ValueError("context and fallback require a frozen predictor")
        if type(max_calls) is not int or not 1 <= max_calls <= 128:
            raise ValueError("max_calls must be an integer in1..128")
        self._store, self._cutoff, self._version = store, cutoff, policy_version
        self._mode, self._predict, self._max_calls = mode, predict, max_calls
        self._failed = False
        self._report: dict[str, Any] = dict(
            mode=mode,
            cutoff=cutoff,
            policy_version=policy_version,
            calls=[],
            work=dict(
                attempted=0, returned=0, validated=0, callback_attempted=0, callback_returned=0
            ),
        )

    @property
    def report(self):
        return deepcopy(self._report)

    def __call__(self, request):
        if self._failed or self._report["work"]["attempted"] >= self._max_calls:
            raise ValueError("evidence predictor stopped or call allowance exhausted")
        if type(request) is not DecisionRequest:
            raise ValueError("expected exact DecisionRequest")
        request.__post_init__()
        row: dict[str, Any] = dict(request=asdict(request))
        self._report["calls"].append(row)
        work = self._report["work"]
        work["attempted"] += 1
        try:
            selection = self._store.retrieve(
                request, cutoff=self._cutoff, policy_version=self._version
            )
            row["selection"] = asdict(selection)
            result: Any = (
                None if self._mode == "context" else decision_from_evidence(request, selection)
            )
            if self._mode == "retrieval" and result is None:
                raise ValueError("no temporally eligible overlapping decision evidence")
            if self._mode == "fallback":
                row["choice_source"] = "memory" if result is not None else "fallback"
            if self._mode == "context" or (self._mode == "fallback" and result is None):
                # The original candidate schema and request identity remain fixed.
                # Why: unavailable memory supplies no new information; fallback
                # uses the predictor's original input recipe, without an empty context.
                supplied = (
                    replace(
                        deepcopy(request), query=request.query + " evidence " + selection.context
                    )
                    if self._mode == "context"
                    else deepcopy(request)
                )
                row["supplied"] = asdict(supplied)
                expected = deepcopy(supplied)
                version = (
                    len(self._store.events),
                    self._store.storage_bytes,
                    self._store.work["writes_attempted"],
                )
                work["callback_attempted"] += 1
                result = self._predict(supplied)
                work["callback_returned"] += 1
                row["returned"] = (
                    _observed_return(result)
                    if type(result) is DecisionResult
                    else _observed_probability_return(result)
                )
                if supplied != expected or version != (
                    len(self._store.events),
                    self._store.storage_bytes,
                    self._store.work["writes_attempted"],
                ):
                    raise ValueError("callback changed protected request or evidence state")
            work["returned"] += 1
            if "returned" not in row:
                row["returned"] = _observed_probability_return(result)
            if type(result) is DecisionResult:
                validate_decision_result(request, result)
            else:
                validate_probability_decision_result(request, result)
            row["result"] = asdict(result)
            work["validated"] += 1
            return result
        except Exception as error:
            self._failed = True
            row["error"] = type(error).__name__ + ": " + str(error)[:512]
            raise
        finally:
            row["store_work_after"] = self._store.work
