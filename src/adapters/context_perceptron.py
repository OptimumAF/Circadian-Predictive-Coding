"""Conventional public-context learner port and detached typed affine readout.

Owns full state/forks and validates observed policy tokens. No context inference,
teacher targets, permission changes, data access, scheduling, probes or IO.
"""

from copy import deepcopy
from dataclasses import dataclass
import re

from src.adapters.perceptron_decision import PerceptronDecisionPredictor
from src.core.categorical_perceptron import PERCEPTRON_MISTAKES_DIAGNOSTIC
from src.core.context_perceptron import (
    ContextPerceptronPolicy,
    ContextPerceptronSnapshot,
    copy_context_policy,
    copy_context_snapshot,
    create_context_snapshot,
    predict_context_perceptron,
    route_context_features,
    update_context_perceptron,
)
from src.core.decision_features import DecisionFeatureSpec, encode_decision
from src.core.learner_ports import TrainingDiagnostic


class ContextPerceptronLearner:
    def __init__(self, policy: ContextPerceptronPolicy):
        self._policy = copy_context_policy(policy)
        self._state = create_context_snapshot(self._policy)

    @property
    def policy(self):
        return copy_context_policy(self._policy)

    @property
    def completed_updates(self):
        return self.snapshot_state().completed_updates

    @property
    def training_examples(self):
        return self.snapshot_state().training_examples

    def snapshot_state(self):
        owned = copy_context_snapshot(self._state)
        if owned.policy != self.policy:
            raise ValueError("context state policy differs from its bound learner policy")
        return owned

    def restore_state(self, snapshot):
        owned = copy_context_snapshot(snapshot)
        if owned.policy != self.policy:
            raise ValueError("context restore policy differs from its bound learner policy")
        self._state = owned

    def fork(self):
        child = ContextPerceptronLearner(self.policy)
        child.restore_state(self.snapshot_state())
        return child

    def train_batch(self, features, targets):
        before = self.snapshot_state()
        after = update_context_perceptron(before, features, targets)
        diagnostic = TrainingDiagnostic(
            PERCEPTRON_MISTAKES_DIAGNOSTIC, float(after.mistake_updates - before.mistake_updates)
        )
        self._state = after
        return diagnostic

    def predict(self, features):
        return predict_context_perceptron(self.snapshot_state(), features)


def _head_spec(spec, state):
    if type(spec) is not DecisionFeatureSpec:
        raise ValueError("context readout requires exact DecisionFeatureSpec")
    spec.__post_init__()
    columns = state.policy.context_columns
    if (
        len(spec.vocabulary) != state.policy.feature_dim
        or len(spec.catalog) != state.policy.num_classes
        or tuple(spec.vocabulary[i] for i in columns) != ("policy_v1", "policy_v2")
    ):
        raise ValueError("readout schema must bind observed policy_v1/v2 to original head order")
    return DecisionFeatureSpec(
        tuple(w for i, w in enumerate(spec.vocabulary) if i not in columns), spec.catalog
    )


@dataclass(frozen=True)
class ContextPerceptronDecisionPredictor:
    spec: DecisionFeatureSpec
    state: ContextPerceptronSnapshot

    def __post_init__(self):
        state = copy_context_snapshot(self.state)
        _head_spec(self.spec, state)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "spec", deepcopy(self.spec))

    def __call__(self, request):
        state = copy_context_snapshot(self.state)
        head_spec = _head_spec(self.spec, state)
        features, _ = encode_decision(request, self.spec)
        tokens = [
            w for w in re.findall(r"\w+", request.query.casefold()) if w.startswith("policy_")
        ]
        if len(tokens) != 1 or tokens[0] not in ("policy_v1", "policy_v2"):
            raise ValueError("typed context must be one unambiguous known public policy token")
        index, _ = route_context_features(features, state.policy)
        # Original adapter retains stable permitted-candidate logits and request
        # validation; its D7 projection ignores the already validated context tag.
        return PerceptronDecisionPredictor(head_spec, state.heads[index])(request)
