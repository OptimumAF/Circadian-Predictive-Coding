"""Bind native categorical heads to existing serving and promotion ports.

Each typed request reads one atomic complete actor snapshot, then uses the
existing frozen raw-logit decoder. No readout cache or success history survives
the request. Guard callbacks measure categorical development-template accuracy;
they grant no training, final-evaluation or promotion authority themselves.
"""

from dataclasses import asdict, dataclass, fields, is_dataclass
from hashlib import sha256
import json
from threading import Lock
from typing import Any

import numpy as np

from src.adapters.torch_decision import TorchDecisionPredictor
from src.adapters.torch_feature_learner import (
    TorchFeatureLearner,
    TorchFeatureSnapshot,
    _snapshot,
)
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.app.serving_promotion import PromotableActor
from src.core.decision_features import DecisionFeatureSpec
from src.core.typed_decision import DecisionRequest, DecisionResult
from src.shared.torch_runtime import require_torch


def _values(value: Any) -> Any:
    if type(value) is require_torch().Tensor:
        return dict(dtype=str(value.dtype), shape=list(value.shape), values=value.tolist())
    if is_dataclass(value):
        return {f.name: _values(getattr(value, f.name)) for f in fields(value)}
    if type(value) is dict:
        return {str(k): _values(v) for k, v in value.items()}
    if type(value) in (tuple, list):
        return [_values(v) for v in value]
    return value


def snapshot_digest(state: TorchFeatureSnapshot) -> str:
    """Required controller identity callback, including optimizer and gradients."""
    return sha256(json.dumps(_values(state), sort_keys=True, allow_nan=False).encode()).hexdigest()


def snapshot_tensor_bytes(value: Any) -> int:
    """Retained native tensors only, not Python objects or process memory."""
    if type(value) is require_torch().Tensor:
        return int(value.numel() * value.element_size())
    if is_dataclass(value):
        return sum(snapshot_tensor_bytes(getattr(value, f.name)) for f in fields(value))
    if type(value) is dict:
        return sum(snapshot_tensor_bytes(v) for v in value.values())
    if type(value) in (tuple, list):
        return sum(snapshot_tensor_bytes(v) for v in value)
    return 0


def build_native(state: TorchFeatureSnapshot) -> TorchFeatureLearner:
    # The controller/evaluator restore the complete supplied state themselves.
    return TorchFeatureLearner(state.policy)


def _valid_state(state: TorchFeatureSnapshot) -> bool:
    try:
        _snapshot(state)
    except (ValueError, TypeError):
        return False
    return True


def categorical_guard(spec: DecisionFeatureSpec, clock: Any) -> Any:
    """Existing evaluator, global categorical accuracy on declared inner guards.

    Typed serving additionally masks raw logits to the request's allowed set.
    This guard never claims that global action IDs certify effect-time permission.
    """
    spec.__post_init__()
    ids = tuple(c.candidate_id for c in spec.catalog)

    def valid(p: Any) -> bool:
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

    def utility(p: Any, targets: Any) -> float:
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
        build_learner=build_native,
        utility=utility,
        actions=lambda p: tuple(ids[int(i)] for i in np.argmax(p, axis=1)),
        state_valid=_valid_state,
        prediction_valid=valid,
        resource_bytes=snapshot_tensor_bytes,
        state_digest=snapshot_digest,
        clock=clock,
    )


@dataclass(frozen=True)
class ServedDecision:
    generation: int
    actor_version: str
    native_updates: int
    result: DecisionResult


class TorchServingPredictor:
    """One snapshot per request; failures latch and all attempted work stays paid.

    A concurrent swap may follow the snapshot read. That request still reports
    its actual generation/version; it cannot combine two publications. Native
    reconstruction is intentionally counted, not hidden behind an unsafe cache.
    """

    def __init__(self, actor: PromotableActor[Any, Any, Any, Any], spec: DecisionFeatureSpec):
        if not isinstance(actor, PromotableActor) or type(spec) is not DecisionFeatureSpec:
            raise ValueError("typed serving requires a promotable actor and fixed feature spec")
        spec.__post_init__()
        self._actor, self._spec = actor, spec
        self._gate, self._stopped = Lock(), False
        self.work = dict(attempts=0, returns=0, readouts=0, native_forwards=0)

    def serve(self, request: DecisionRequest) -> ServedDecision:
        if not self._gate.acquire(blocking=False):
            raise ValueError("typed serving predictor is busy")
        try:
            if self._stopped:
                raise ValueError("typed serving stopped after previous fault")
            self.work["attempts"] += 1
            publication = self._actor.serving_snapshot()
            self.work["readouts"] += 1
            predictor = TorchDecisionPredictor(self._spec, publication.model_state)
            try:
                result = predictor(request)
            finally:
                self.work["native_forwards"] += asdict(predictor.work)["native_forward_attempts"]
            self.work["returns"] += 1
            return ServedDecision(
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
