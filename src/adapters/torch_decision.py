"""Frozen native categorical heads behind the existing typed decision port.

Owns one full copied checkpoint, fixed query/catalog binding and inference work
counters. No training, promotion, replay, IO or final/probe selection authority.
Caller owns admission and budgets. Private Python objects are not a sandbox.
"""

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
from math import exp, fsum
from typing import Any

from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeatureSnapshot
from src.core.decision_features import DecisionFeatureSpec, encode_decision
from src.core.typed_decision import (
    CandidateScore,
    DecisionRequest,
    DecisionResult,
    validate_decision_result,
)
from src.shared.torch_runtime import require_torch


def _same_values(left: Any, right: Any) -> bool:
    """Exact complete values, including tensor dtype/shape and RNG bytes."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_same_values(left[k], right[k]) for k in left)
    if type(left) in (tuple, list):
        return len(left) == len(right) and all(
            _same_values(a, b) for a, b in zip(left, right, strict=True)
        )
    if is_dataclass(left):
        return all(
            _same_values(getattr(left, f.name), getattr(right, f.name)) for f in fields(left)
        )
    torch = require_torch()
    if type(left) is torch.Tensor:
        return (
            left.dtype == right.dtype
            and left.device == right.device
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    return bool(left == right)


@dataclass(frozen=True)
class TorchDecisionWork:
    decision_attempts: int = 0
    native_forward_attempts: int = 0
    native_forward_returns: int = 0
    decision_returns: int = 0


class TorchDecisionPredictor:
    """Own a detached native readout; latch any fault, keep attempted work paid.

    Construction restores the COMPLETE snapshot, including BP optimizer/gradients
    or PC/CPC adaptive clocks and generator. The existing restore port initializes
    twice before replacing tensors; this cost is additional to prediction work.
    Storage is constant in call count: one learner, one pin, four counters.
    """

    def __init__(self, spec: DecisionFeatureSpec, state: TorchFeatureSnapshot) -> None:
        if type(spec) is not DecisionFeatureSpec or type(state) is not TorchFeatureSnapshot:
            raise ValueError("expected exact feature spec and complete Torch feature snapshot")
        spec.__post_init__()
        if set(vars(state)) != {item.name for item in fields(TorchFeatureSnapshot)}:
            raise ValueError("expected complete Torch feature snapshot fields")
        owned = replace(state)
        if owned.policy.feature_dim != len(spec.vocabulary) or owned.policy.num_classes != len(
            spec.catalog
        ):
            raise ValueError("Torch policy must match vocabulary and catalog dimensions")
        self._spec = deepcopy(spec)
        self._initial_spec = deepcopy(spec)
        self._learner = TorchFeatureLearner(owned.policy)
        self._learner.restore_state(owned)
        self._initial = self._learner.snapshot_state()
        self._work = TorchDecisionWork()
        self._stopped = False
        self.assert_unchanged()

    @property
    def work(self) -> TorchDecisionWork:
        return replace(self._work)

    def assert_unchanged(self) -> None:
        try:
            if self._stopped:
                raise ValueError("frozen Torch decision predictor stopped after a previous fault")
            if self._spec != self._initial_spec or not _same_values(
                self._initial, self._learner.snapshot_state()
            ):
                raise ValueError("frozen Torch decision complete local state changed")
        except Exception:
            self._stopped = True
            raise

    def __call__(self, request: DecisionRequest) -> DecisionResult:
        self._work = replace(self._work, decision_attempts=self._work.decision_attempts + 1)
        try:
            self.assert_unchanged()
            features, visits = encode_decision(request, self._spec)
            self._work = replace(
                self._work, native_forward_attempts=self._work.native_forward_attempts + 1
            )
            logits = self._learner.predict_logits(features)[0]
            self._work = replace(
                self._work, native_forward_returns=self._work.native_forward_returns + 1
            )
            self.assert_unchanged()
            allowed = sorted(request.allowed_candidates, key=lambda item: item.candidate_id)
            class_ids = tuple(item.candidate_id for item in self._spec.catalog)
            raw = [float(logits[class_ids.index(item.candidate_id)]) for item in allowed]
            # All classes are projected natively, but only allowed logits normalize.
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
            self._work = replace(self._work, decision_returns=self._work.decision_returns + 1)
            return result
        except Exception:
            self._stopped = True
            raise
