"""Shared LS state bindings for existing atomic serving and promotion guards.

Inputs: exact complete shared snapshots, fixed feature schema, existing actor.
Outputs: validated state records and versioned typed reads with paid native work.
No fitting, authority renewal, ordinary workflow wiring, promotion decision,
guard-data selection, task scoring or IO lives here. Record bytes are not heap/RSS.
"""

from copy import deepcopy
from dataclasses import asdict, fields, replace
from hashlib import sha256
import json

from src.adapters.context_serving import ContextServingPredictor, _categorical_guard
from src.adapters.linear_fit_decision import LinearFitDecisionPredictor, LinearDecisionWork
from src.adapters.linear_fit_learner import LinearFitLearner
from src.core.categorical_linear_state import copy_linear_snapshot
from src.core.decision_features import DecisionFeatureSpec


def snapshot_bytes(state):
    return json.dumps(
        asdict(copy_linear_snapshot(state)), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf8")


def snapshot_digest(state):
    return sha256(snapshot_bytes(state)).hexdigest()


def snapshot_record_bytes(state):
    return len(snapshot_bytes(state))


def build_linear(state):
    # The existing evaluator/controller restores the complete supplied snapshot.
    return LinearFitLearner(copy_linear_snapshot(state).policy)


def _validate_schema(spec, state):
    if type(spec) is not DecisionFeatureSpec:
        raise ValueError("shared serving requires exact feature spec")
    spec.__post_init__()
    state = copy_linear_snapshot(state)
    if (state.policy.feature_dim, state.policy.num_classes) != (
        len(spec.vocabulary),
        len(spec.catalog),
    ):
        raise ValueError("shared serving snapshot geometry differs from fixed schema")


def categorical_guard(spec, clock):
    """Existing disjoint inner_guard measurement; caller owns data and costs."""
    return _categorical_guard(
        deepcopy(spec),
        clock,
        build_learner=build_linear,
        copy_state=copy_linear_snapshot,
        resource_bytes=snapshot_record_bytes,
        state_digest=snapshot_digest,
        validate_state=_validate_schema,
    )


class LinearFitServingPredictor(ContextServingPredictor):
    """Reuse the atomic slot/fault latch, with owned shared schema and read costs."""

    def __init__(self, actor, spec):
        super().__init__(actor, deepcopy(spec))
        self._spec_owner = self._spec
        self._spec_pin = deepcopy(self._spec)
        self._native_work = LinearDecisionWork()

    @property
    def native_work(self):
        return replace(self._native_work)

    def _predict(self, state, request):
        if self._spec is not self._spec_owner or self._spec != self._spec_pin:
            raise ValueError("shared serving schema changed")
        _validate_schema(self._spec, state)
        reader = LinearFitDecisionPredictor(self._spec, state.model)
        self.work["typed_attempts"] += 1
        try:
            result = reader(request)
            reader.assert_unchanged()
            return result
        finally:
            # A lost typed return still consumes the delegated native read.
            observed = reader.work
            self._native_work = LinearDecisionWork(
                **{
                    f.name: getattr(self._native_work, f.name) + getattr(observed, f.name)
                    for f in fields(LinearDecisionWork)
                }
            )
