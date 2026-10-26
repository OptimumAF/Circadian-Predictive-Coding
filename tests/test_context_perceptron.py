"""Owned conventional head state, strict routing and original affine parity."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from src.core.context_perceptron import (
    ContextPerceptronPolicy,
    copy_context_policy,
    copy_context_snapshot,
    create_context_snapshot,
    route_context_features,
    update_context_perceptron,
    predict_context_perceptron,
)
from src.core.categorical_perceptron import update_perceptron, predict_perceptron
from src.adapters.context_perceptron import (
    ContextPerceptronLearner,
    ContextPerceptronDecisionPredictor,
)
from src.adapters.perceptron_decision import PerceptronDecisionPredictor
from src.core.decision_features import DecisionFeatureSpec
from src.core.ticket_workflow import (
    WorkflowState,
    WorkflowCapabilities,
    ACTION_IDS,
    dispatch_workflow,
    workflow_request,
)
from scripts import recurring_ticket_workflow_example as study


@pytest.fixture(scope="module")
def zero():
    return create_context_snapshot(ContextPerceptronPolicy(4, 3, (1, 2)))


def data(index=0):
    x = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    if index:
        x[:, 1:3] = (0.0, 1.0)
    return x, np.array([[1.0], [2.0]])


@pytest.fixture(scope="module")
def learned(zero):
    xa, ya = data()
    xb, yb = data(1)
    a = update_context_perceptron(zero, xa, ya)
    b = update_context_perceptron(a, xb, yb)
    return a, b


def test_should_apply_exact_original_rule_and_preserve_inactive_complete_head(zero, learned):
    a, b = learned
    xa, ya = data()
    xb, yb = data(1)
    assert a.heads[1] == zero.heads[1] and b.heads[0] == a.heads[0]
    assert a.heads[0] == update_perceptron(zero.heads[0], xa[:, (0, 3)], ya)
    assert b.heads[1] == update_perceptron(zero.heads[1], xb[:, (0, 3)], yb)
    assert b.completed_updates == sum(h.completed_updates for h in b.heads) == 2
    assert b.training_examples == sum(h.training_examples for h in b.heads) == 4
    assert b.mistake_updates == sum(h.mistake_updates for h in b.heads)
    assert np.array_equal(
        predict_context_perceptron(b, xa), predict_perceptron(a.heads[0], xa[:, (0, 3)])
    )


@pytest.mark.parametrize("columns", [(0, 0), (2, 1), (True, 2), (0, 4), [1, 2], (1,)])
def test_should_refuse_invalid_fixed_context_columns(columns):
    with pytest.raises(ValueError):
        ContextPerceptronPolicy(4, 3, columns)


@pytest.mark.parametrize("fault", ["sum", "heads", "policy", "extra", "head_extra"])
def test_should_refuse_corrupt_complete_snapshot_before_use(zero, fault):
    state = deepcopy(zero)
    if fault == "sum":
        state = replace(state, completed_updates=1)
    elif fault == "heads":
        object.__setattr__(state, "heads", (state.heads[0],))
    elif fault == "policy":
        state = replace(state, policy=ContextPerceptronPolicy(5, 3, (1, 2)))
    elif fault == "extra":
        object.__setattr__(state, "untracked", 1)
    else:
        object.__setattr__(state.heads[0], "untracked", 1)
    with pytest.raises(ValueError):
        copy_context_snapshot(state)


def test_should_refuse_tampered_policy_record(zero):
    policy = deepcopy(zero.policy)
    object.__setattr__(policy, "extra", 1)
    with pytest.raises(ValueError):
        copy_context_policy(policy)


@pytest.mark.parametrize(
    "fault", ["missing", "ambiguous", "mixed", "fraction", "nan", "dtype", "target"]
)
def test_should_refuse_bad_training_atomically(zero, fault):
    learner = ContextPerceptronLearner(zero.policy)
    x, y = data()
    if fault == "missing":
        x[:, 1:3] = 0
    elif fault == "ambiguous":
        x[:, 1:3] = 1
    elif fault == "mixed":
        x[1, 1:3] = (0, 1)
    elif fault == "fraction":
        x[:, 1:3] = 0.5
    elif fault == "nan":
        x[0, 0] = np.nan
    elif fault == "dtype":
        x = x.astype(np.float32)
    else:
        y[0, 0] = 3
    before = learner.snapshot_state()
    with pytest.raises(ValueError):
        learner.train_batch(x, y)
    assert learner.snapshot_state() == before


def test_should_own_fork_restore_and_fail_foreign_policy_without_mutation(learned):
    learner = ContextPerceptronLearner(learned[0].policy)
    learner.restore_state(learned[0])
    child = learner.fork()
    child.train_batch(*data(1))
    assert learner.snapshot_state() == learned[0] and child.snapshot_state() == learned[1]
    assert learner.snapshot_state() is not learner.snapshot_state()
    foreign = create_context_snapshot(ContextPerceptronPolicy(5, 3, (1, 2)))
    with pytest.raises(ValueError):
        learner.restore_state(foreign)
    assert learner.snapshot_state() == learned[0]


@pytest.fixture(scope="module")
def typed():
    spec, batches = study.prepare_training(context_return=True)
    policy = ContextPerceptronPolicy(
        9, 6, (spec.vocabulary.index("policy_v1"), spec.vocabulary.index("policy_v2"))
    )
    learner = ContextPerceptronLearner(policy)
    learner.train_batch(batches[0].features, batches[0].targets)
    learner.train_batch(batches[4].features, batches[4].targets)
    return spec, learner.snapshot_state()


def request(version="v1"):
    caps = WorkflowCapabilities(ACTION_IDS)
    ready = dispatch_workflow(
        WorkflowState("head-read", "invoice", policy_version=version), "open", 0, caps
    ).state
    return workflow_request(ready, caps, "read", include_policy_context=True)


@pytest.mark.parametrize("index,version", [(0, "v1"), (1, "v2")])
def test_should_delegate_identical_allowed_affine_decision_to_selected_head(typed, index, version):
    spec, state = typed
    predict = ContextPerceptronDecisionPredictor(spec, state)
    head_spec = DecisionFeatureSpec(study.FEATURES, study.CATALOG)
    q = request(version)
    assert predict(q) == PerceptronDecisionPredictor(head_spec, state.heads[index])(q)
    assert predict.state == state and predict.state is not state


@pytest.mark.parametrize(
    "token", ["", "policy_v3", "policy_v1 policy_v2", "policy_v1 policy_v1", "policy_v1 policy_v3"]
)
def test_should_refuse_missing_unknown_or_ambiguous_public_context_before_affine(
    typed, token, monkeypatch
):
    spec, state = typed
    predict = ContextPerceptronDecisionPredictor(spec, state)
    q = request()
    monkeypatch.setattr(
        PerceptronDecisionPredictor, "__call__", lambda *a: pytest.fail("affine reached")
    )
    with pytest.raises(ValueError):
        predict(replace(q, query=q.query.replace("policy_v1", token)))


def test_should_refuse_unbound_readout_context_order(typed):
    spec, state = typed
    wrong = replace(state, policy=ContextPerceptronPolicy(9, 6, (0, 1)))
    with pytest.raises(ValueError):
        ContextPerceptronDecisionPredictor(spec, wrong)


def test_should_route_owned_features_without_aliasing(zero):
    x, _ = data()
    before = x.copy()
    index, routed = route_context_features(x, zero.policy)
    assert index == 0
    routed[0, 0] += 1
    assert np.array_equal(x, before)


def test_should_score_interleaved_contexts_in_original_order_without_mutating(learned):
    state = learned[1]
    xa, _ = data()
    xb, _ = data(1)
    mixed = np.vstack((xb[1], xa[0], xb[0], xa[1]))
    before = mixed.copy()
    expected = np.vstack(tuple(predict_context_perceptron(state, x[None, :]) for x in mixed))
    learner = ContextPerceptronLearner(state.policy)
    learner.restore_state(state)
    assert np.array_equal(learner.predict(mixed), expected)
    assert learner.snapshot_state() == state and np.array_equal(mixed, before)


@pytest.mark.parametrize("context", [(0, 0), (1, 1), (0.5, 0.5)])
def test_should_refuse_invalid_context_even_in_mixed_readonly_batch(learned, context):
    x, _ = data()
    x[1, 1:3] = context
    with pytest.raises(ValueError, match="binary public context"):
        predict_context_perceptron(learned[1], x)
