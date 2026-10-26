"""Context-head bindings to the existing atomic serving and guard ports.

Complete validated JSON bytes include policy, coefficients and all counters.
This metric excludes Python heap/temporary copies/process RSS. No authority,
training, final evaluation, probe selection, readout cache or IO belongs here.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from threading import Lock
from typing import Any

import numpy as np

from src.adapters.context_perceptron import (
    ContextPerceptronLearner,
    ContextPerceptronDecisionPredictor,
    _head_spec,
)
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.app.serving_promotion import PromotableActor
from src.core.context_perceptron import ContextPerceptronSnapshot, copy_context_snapshot
from src.core.decision_features import DecisionFeatureSpec
from src.core.typed_decision import DecisionRequest, DecisionResult


def snapshot_bytes(state: ContextPerceptronSnapshot) -> bytes:
    """Canonical complete validated record, with explicit UTF-8 and compact JSON."""
    return json.dumps(
        asdict(copy_context_snapshot(state)), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def snapshot_digest(state: ContextPerceptronSnapshot) -> str:
    return sha256(snapshot_bytes(state)).hexdigest()


def snapshot_record_bytes(state: ContextPerceptronSnapshot) -> int:
    return len(snapshot_bytes(state))


def build_context(state: ContextPerceptronSnapshot) -> ContextPerceptronLearner:
    # Existing evaluator/controller restore the supplied complete state.
    return ContextPerceptronLearner(copy_context_snapshot(state).policy)


def categorical_guard(spec: DecisionFeatureSpec, clock: Any) -> Any:
    if type(spec) is not DecisionFeatureSpec:
        raise ValueError("context guard requires fixed feature spec")
    spec.__post_init__()
    ids = tuple(c.candidate_id for c in spec.catalog)

    def state_valid(state):
        try:
            _head_spec(spec, copy_context_snapshot(state))
        except (ValueError, TypeError):
            return False
        return True

    def valid(p):
        return bool(
            type(p) is np.ndarray
            and p.ndim == 2
            and p.shape[1] == len(ids)
            and len(p) > 0
            and np.isfinite(p).all()
            and (p >= 0).all()
            and (p <= 1).all()
            and np.allclose(p.sum(axis=1), 1, atol=1e-6)
        )

    def utility(p, targets):
        if (
            type(targets) is not np.ndarray
            or targets.shape != (len(p), 1)
            or not np.isfinite(targets).all()
            or not (targets == np.floor(targets)).all()
            or not ((targets >= 0) & (targets < len(ids))).all()
        ):
            raise ValueError("categorical guard requires Bx1 class-index targets")
        return float(np.mean(np.argmax(p, axis=1) == targets[:, 0]))

    return PromotionGuardEvaluator(
        build_learner=build_context,
        utility=utility,
        actions=lambda p: tuple(ids[int(i)] for i in np.argmax(p, axis=1)),
        state_valid=state_valid,
        prediction_valid=valid,
        resource_bytes=snapshot_record_bytes,
        state_digest=snapshot_digest,
        clock=clock,
    )


@dataclass(frozen=True)
class ContextServedDecision:
    generation: int
    actor_version: str
    completed_updates: int
    result: DecisionResult

    @property
    def native_updates(self) -> int:
        """Existing timeline port name: complete learner passes, not Torch-only work."""
        return self.completed_updates


class ContextServingPredictor:
    """One detached complete actor slot per typed request; faults permanently latch."""

    def __init__(self, actor: PromotableActor[Any, Any, Any, Any], spec: DecisionFeatureSpec):
        if not isinstance(actor, PromotableActor) or type(spec) is not DecisionFeatureSpec:
            raise ValueError("typed serving requires a promotable actor and fixed feature spec")
        spec.__post_init__()
        self._actor, self._spec = actor, spec
        self._gate, self._stopped = Lock(), False
        self.work = dict(attempts=0, returns=0, readouts=0, typed_attempts=0)

    def serve(self, request: DecisionRequest) -> ContextServedDecision:
        if not self._gate.acquire(blocking=False):
            raise ValueError("typed serving predictor is busy")
        try:
            if self._stopped:
                raise ValueError("typed serving stopped after previous fault")
            self.work["attempts"] += 1
            publication = self._actor.serving_snapshot()
            self.work["readouts"] += 1
            predictor = ContextPerceptronDecisionPredictor(self._spec, publication.model_state)
            self.work["typed_attempts"] += 1
            result = predictor(request)
            self.work["returns"] += 1
            return ContextServedDecision(
                publication.generation,
                publication.actor_version,
                publication.model_state.completed_updates,
                result,
            )
        except Exception:
            self._stopped = True
            raise
        finally:
            self._gate.release()

    def __call__(self, request: DecisionRequest) -> DecisionResult:
        return self.serve(request).result
