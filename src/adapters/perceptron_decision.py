"""Bridge typed decisions to the existing perceptron and attribution stream.

Builds TRAIN packets and a detached read-only affine readout. Training, budgets,
replay, checkpoint selection, model fitting and IO remain in existing components.
"""

from copy import deepcopy
from dataclasses import dataclass
from math import exp, fsum

import numpy as np

from src.app.coupled_attribution import AttributionBatch
from src.app.decision_reference_evaluation import DecisionCase
from src.core.categorical_perceptron import PerceptronSnapshot, copy_perceptron_snapshot
from src.core.decision_features import DecisionFeatureSpec, encode_decision
from src.core.experience import require_identifier
from src.core.typed_decision import (
    CandidateScore,
    DecisionRequest,
    DecisionResult,
    validate_decision_result,
)


def build_decision_training_batch(
    cases: tuple[DecisionCase, ...],
    spec: DecisionFeatureSpec,
    context: str,
    *,
    allow_subsets: bool = False,
) -> AttributionBatch:
    """Project TRAIN labels; explicit subsets keep fixed union class indices."""
    if type(allow_subsets) is not bool:
        raise ValueError("allow_subsets must be an explicit boolean")
    require_identifier(context, "training context")
    if type(spec) is not DecisionFeatureSpec:
        raise ValueError("expected exact DecisionFeatureSpec")
    spec.__post_init__()
    if type(cases) is not tuple or not 1 <= len(cases) <= 128:
        raise ValueError("training cases must be an immutable tuple of 1..128 records")
    ids = []
    class_ids = tuple(item.candidate_id for item in spec.catalog)
    for case in cases:
        if type(case) is not DecisionCase:
            raise ValueError("expected exact DecisionCase")
        case.__post_init__()
        if case.role != "train":
            raise ValueError("decision training packets require TRAIN cases")
        allowed = {item.candidate_id for item in case.request.allowed_candidates}
        if not allow_subsets and allowed != set(class_ids):
            raise ValueError("training requests require the complete fixed catalog")
        if case.target_id not in allowed or not allowed <= set(class_ids):
            raise ValueError("training target must be an allowed catalog candidate")
        if any(item not in spec.catalog for item in case.request.allowed_candidates):
            raise ValueError("training candidates must match the fixed catalog descriptions")
        ids.append(case.request.request_id)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate training request ID")
    features = np.vstack([encode_decision(case.request, spec)[0] for case in cases])
    targets = np.array([[float(class_ids.index(case.target_id))] for case in cases])
    return AttributionBatch(
        "train", context, tuple(ids), features, targets, num_classes=len(class_ids)
    )


@dataclass(frozen=True)
class PerceptronDecisionPredictor:
    spec: DecisionFeatureSpec
    state: PerceptronSnapshot

    def __post_init__(self) -> None:
        if type(self.spec) is not DecisionFeatureSpec:
            raise ValueError("expected exact DecisionFeatureSpec")
        self.spec.__post_init__()
        owned = copy_perceptron_snapshot(self.state)
        if owned.policy.feature_dim != len(self.spec.vocabulary) or owned.policy.num_classes != len(
            self.spec.catalog
        ):
            raise ValueError("perceptron policy must match vocabulary and catalog dimensions")
        object.__setattr__(self, "spec", deepcopy(self.spec))
        object.__setattr__(self, "state", owned)

    def __call__(self, request: DecisionRequest) -> DecisionResult:
        features, token_visits = encode_decision(request, self.spec)
        state = copy_perceptron_snapshot(self.state)
        if state.policy.feature_dim != features.shape[1] or state.policy.num_classes != len(
            self.spec.catalog
        ):
            raise ValueError("perceptron policy differs from fixed feature schema")
        allowed = sorted(request.allowed_candidates, key=lambda item: item.candidate_id)
        class_ids = tuple(item.candidate_id for item in self.spec.catalog)
        indices = [class_ids.index(item.candidate_id) for item in allowed]
        coefficients = np.array(state.coefficients, dtype=np.float64)[:, indices]
        # Why this: use native affine coefficients before softmax. Slicing already
        # normalized global probabilities can lose all allowed mass to underflow.
        logits = np.append(features[0], 1.0) @ coefficients
        weights = [exp(float(value - max(logits))) for value in logits]
        mass = fsum(weights)
        scores = tuple(
            CandidateScore(item.candidate_id, float(raw), weight / mass)
            for item, raw, weight in zip(allowed, logits, weights, strict=True)
        )
        winner = min(scores, key=lambda item: (-item.probability, item.candidate_id))
        result = DecisionResult(
            request.request_id, winner.candidate_id, scores, token_visits, len(scores)
        )
        validate_decision_result(request, result)
        return result
