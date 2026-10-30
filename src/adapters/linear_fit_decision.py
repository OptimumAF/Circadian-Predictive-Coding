"""Frozen one-hot LS scores at the existing typed decision boundary.

Owns model/spec copies, stable allowed-candidate normalization and paid reads.
No fitting, role admission, task loop, training effects, RNG or IO lives here.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
from math import exp, fsum

import numpy as np

from src.core.categorical_linear_fit import (
    CategoricalLinearFit,
    _features,
    predict_categorical_linear_logits,
)
from src.core.coupled_energy import Array
from src.core.decision_features import DecisionFeatureSpec, encode_decision
from src.core.typed_decision import (
    CandidateScore,
    DecisionRequest,
    DecisionResult,
    validate_decision_result,
)


@dataclass(frozen=True)
class LinearDecisionWork:
    decision_attempts: int = 0
    decision_returns: int = 0
    affine_attempts: int = 0
    affine_returns: int = 0
    rows_attempted: int = 0
    rows_returned: int = 0
    nominal_affine_terms: int = 0
    token_visits: int = 0
    unfitted_attempts: int = 0
    unfitted_returns: int = 0
    unfitted_rows_attempted: int = 0
    unfitted_rows_returned: int = 0


class LinearFitDecisionPredictor:
    def __init__(self, spec: DecisionFeatureSpec, model: CategoricalLinearFit | None):
        if type(spec) is not DecisionFeatureSpec or (
            model is not None and type(model) is not CategoricalLinearFit
        ):
            raise ValueError("linear readout requires exact feature spec and fitted model or None")
        spec.__post_init__()
        if model is not None:
            model.__post_init__()
            if model.feature_dim != len(spec.vocabulary) or model.num_classes != len(spec.catalog):
                raise ValueError("linear readout model must match the fixed feature catalog")
        self.spec, self.model = deepcopy((spec, model))
        self._owners = (id(self.spec), id(self.model))
        self._pin = deepcopy((self.spec, self.model))
        self._work = LinearDecisionWork()
        self._work_owner = self._work
        self._work_pin = deepcopy(self._work)

    @property
    def work(self) -> LinearDecisionWork:
        return deepcopy(self._work_pin)

    def assert_unchanged(self) -> None:
        if (
            self._owners != (id(self.spec), id(self.model))
            or (self.spec, self.model) != self._pin
            or self._work is not self._work_owner
            or self._work != self._work_pin
        ):
            raise ValueError("frozen linear readout model/spec/work changed")

    def _charge(self, **increments: int) -> None:
        self.assert_unchanged()
        self._work = replace(
            self._work,
            **{key: getattr(self._work, key) + value for key, value in increments.items()},
        )
        self._work_owner = self._work
        self._work_pin = deepcopy(self._work)

    def predict_logits(self, features: Array) -> Array:
        if self.model is None:
            self._charge(unfitted_attempts=1, unfitted_rows_attempted=len(features))
            _features(features, len(self.spec.vocabulary))
            logits = np.zeros((len(features), len(self.spec.catalog)), dtype=np.float64)
            self._charge(unfitted_returns=1, unfitted_rows_returned=len(features))
            return logits
        self._charge(
            affine_attempts=1,
            rows_attempted=len(features),
            nominal_affine_terms=len(features)
            * (self.model.feature_dim + 1)
            * self.model.num_classes,
        )
        logits = predict_categorical_linear_logits(self.model, features)
        self._charge(affine_returns=1, rows_returned=len(logits))
        return logits

    def __call__(self, request: DecisionRequest) -> DecisionResult:
        self._charge(decision_attempts=1)
        features, visits = encode_decision(request, self.spec)
        self._charge(token_visits=visits)
        logits = self.predict_logits(features)[0]
        ids = tuple(item.candidate_id for item in self.spec.catalog)
        allowed = sorted(request.allowed_candidates, key=lambda item: item.candidate_id)
        raw = [float(logits[ids.index(item.candidate_id)]) for item in allowed]
        weights = [exp(value - max(raw)) for value in raw]
        mass = fsum(weights)
        scores = tuple(
            CandidateScore(item.candidate_id, value, weight / mass)
            for item, value, weight in zip(allowed, raw, weights, strict=True)
        )
        winner = min(scores, key=lambda item: (-item.probability, item.candidate_id))
        result = DecisionResult(
            request.request_id, winner.candidate_id, scores, visits, len(scores)
        )
        validate_decision_result(request, result)
        self._charge(decision_returns=1)
        return result
