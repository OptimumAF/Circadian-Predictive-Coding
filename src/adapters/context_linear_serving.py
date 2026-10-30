"""Explicit ContextLS bindings to the existing atomic serving/guard ports.

Complete record bytes include LS coefficients, fit evidence, policy and counters;
not Python heap, copies or RSS. No fitting, publication authority or data IO.
"""

from dataclasses import asdict, fields, replace
from hashlib import sha256
import json

from src.adapters.context_linear_fit_decision import (
    ContextLinearFitDecisionPredictor,
    ContextLinearDecisionWork,
)
from src.adapters.context_linear_fit_learner import ContextLinearFitLearner
from src.adapters.context_serving import ContextServingPredictor, _categorical_guard
from src.core.context_linear_state import copy_context_linear_state


def snapshot_bytes(state):
    return json.dumps(
        asdict(copy_context_linear_state(state)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def snapshot_digest(state):
    return sha256(snapshot_bytes(state)).hexdigest()


def snapshot_record_bytes(state):
    return len(snapshot_bytes(state))


def build_context(state):
    return ContextLinearFitLearner(copy_context_linear_state(state).policy)


def categorical_guard(spec, clock):
    return _categorical_guard(
        spec,
        clock,
        build_learner=build_context,
        copy_state=copy_context_linear_state,
        resource_bytes=snapshot_record_bytes,
        state_digest=snapshot_digest,
    )


class ContextLinearServingPredictor(ContextServingPredictor):
    """Same detached atomic slot/fault latch; bounded aggregate native read costs."""

    def __init__(self, actor, spec):
        super().__init__(actor, spec)
        self._native_work = ContextLinearDecisionWork()

    @property
    def native_work(self):
        return replace(self._native_work)

    def _predict(self, state, request):
        reader = ContextLinearFitDecisionPredictor(self._spec, state)
        self.work["typed_attempts"] += 1
        try:
            result = reader(request)
            reader.assert_unchanged()
            return result
        finally:
            # Why: a lost return still pays routing, tokenization and affine work.
            observed = reader.work
            self._native_work = ContextLinearDecisionWork(
                **{
                    f.name: getattr(self._native_work, f.name) + getattr(observed, f.name)
                    for f in fields(ContextLinearDecisionWork)
                }
            )
