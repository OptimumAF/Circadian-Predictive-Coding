"""Native learner adapters preserve full state and their separate diagnostics."""

from dataclasses import replace
import pickle

import numpy as np
import pytest

from src.adapters.numpy_learners import BackpropLearner, CircadianLearner
from src.app.learner_step import update_learner
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.backprop_mlp import BackpropMLP, NUMPY_BACKPROP_LOSS_ID
from src.core.circadian_predictive_coding import (
    CircadianPredictiveCodingNetwork,
    NUMPY_CIRCADIAN_ENERGY_ID,
)


FEATURES = np.array([[0.3, -0.2], [-0.5, 0.4], [0.1, 0.6], [-0.7, -0.1]])
TARGETS = np.array([[1.0], [0.0], [1.0], [0.0]])


def make_pair(kind):
    if kind == "backprop":
        model = BackpropMLP(2, 4, seed=23, hidden_dims=(3, 4))
        return model, BackpropLearner(model, learning_rate=0.03)
    circadian = CircadianPredictiveCodingNetwork(
        2, 4, seed=23, hidden_dims=(3, 4), min_hidden_dim=4, max_hidden_dim=4
    )
    return circadian, CircadianLearner(
        circadian, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.2
    )


def session():
    return ToyBudgetSession(ToyExecutionBudget(max_training_updates=4), lambda: 0.0)


@pytest.mark.parametrize("kind", ["backprop", "circadian"])
def test_should_match_native_update_prediction_and_entire_model_state(kind):
    native, learner = make_pair(kind)
    if kind == "backprop":
        expected = native.train_epoch(FEATURES, TARGETS, 0.03)
        definition, value = NUMPY_BACKPROP_LOSS_ID, expected.loss
    else:
        expected = native.train_epoch(FEATURES, TARGETS, 0.03, 2, 0.2)
        definition, value = NUMPY_CIRCADIAN_ENERGY_ID, expected.energy
    budget = session()

    result = update_learner(learner, FEATURES, TARGETS, budget)

    assert (result.definition, result.value) == (definition, value)
    assert budget.updates_completed == 1
    np.testing.assert_array_equal(learner.predict(FEATURES), native.predict_proba(FEATURES))
    assert pickle.dumps(learner.snapshot_state().state) == pickle.dumps(native.__dict__)


@pytest.mark.parametrize("kind", ["backprop", "circadian"])
def test_should_copy_input_model_and_snapshot_and_restore_complete_continued_state(kind):
    original, learner = make_pair(kind)
    original_state = pickle.dumps(original.__dict__)
    update_learner(learner, FEATURES, TARGETS, session())
    saved = learner.snapshot_state()
    saved_bytes = pickle.dumps(saved.state)
    first = update_learner(learner, FEATURES, TARGETS, session())
    expected_state = pickle.dumps(learner.snapshot_state().state)
    expected_prediction = learner.predict(FEATURES)

    learner.restore_state(saved)
    second = update_learner(learner, FEATURES, TARGETS, session())

    assert first == second
    assert pickle.dumps(learner.snapshot_state().state) == expected_state
    np.testing.assert_array_equal(learner.predict(FEATURES), expected_prediction)
    assert pickle.dumps(saved.state) == saved_bytes
    assert pickle.dumps(original.__dict__) == original_state
    learner.restore_state(saved)
    restored = pickle.dumps(learner.snapshot_state().state)
    saved.state["weight_hidden_output"][:] = 99.0
    assert pickle.dumps(learner.snapshot_state().state) == restored


@pytest.mark.parametrize("kind", ["backprop", "circadian"])
def test_should_refuse_foreign_or_incompatible_snapshot_before_mutation(kind):
    _, learner = make_pair(kind)
    _, foreign = make_pair("circadian" if kind == "backprop" else "backprop")
    saved = pickle.dumps(learner.snapshot_state().state)
    with pytest.raises(ValueError):
        learner.restore_state(foreign.snapshot_state())
    own = learner.snapshot_state()
    with pytest.raises(ValueError):
        learner.restore_state(replace(own, input_dim=3))
    assert pickle.dumps(learner.snapshot_state().state) == saved


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_field",
        "shape",
        "nonfinite",
        "alias",
        "traffic",
        "container",
        "dtype",
        "negative_traffic",
    ],
)
def test_should_refuse_corrupted_backprop_state_before_mutation(corruption):
    _, learner = make_pair("backprop")
    saved = learner.snapshot_state()
    original = pickle.dumps(saved.state)
    if corruption == "missing_field":
        saved.state.pop("_traffic_steps")
    elif corruption == "shape":
        saved.state["bias_output"] = np.zeros((2, 1))
    elif corruption == "nonfinite":
        saved.state["weight_hidden_output"][0, 0] = np.nan
    elif corruption == "alias":
        saved.state["weight_input_hidden"] = saved.state["weight_input_hidden"].copy()
    elif corruption == "traffic":
        saved.state["_traffic_steps"] = True
    elif corruption == "container":
        saved.state["_traffic_sums"] = tuple(saved.state["_traffic_sums"])
    elif corruption == "dtype":
        saved.state["bias_output"] = saved.state["bias_output"].astype(np.float32)
    else:
        saved.state["_traffic_sums"][0][0] = -1.0
    with pytest.raises(ValueError, match="snapshot"):
        learner.restore_state(saved)
    assert pickle.dumps(learner.snapshot_state().state) == original


def test_should_restore_backprop_with_arbitrary_supported_input_width():
    model = BackpropMLP(3, 4, seed=23)
    learner = BackpropLearner(model, learning_rate=0.03)
    features = np.column_stack((FEATURES, np.zeros(4)))
    saved = learner.snapshot_state()
    first = update_learner(learner, features, TARGETS, session())
    prediction = learner.predict(features)
    learner.restore_state(saved)
    second = update_learner(learner, features, TARGETS, session())
    assert first == second
    np.testing.assert_array_equal(learner.predict(features), prediction)


@pytest.mark.parametrize("kind", ["backprop", "circadian"])
def test_should_predict_without_mutating_model_state(kind):
    _, learner = make_pair(kind)
    saved = pickle.dumps(learner.snapshot_state().state)
    learner.predict(FEATURES)
    assert pickle.dumps(learner.snapshot_state().state) == saved


def test_should_preserve_backprop_multilayer_aliases_and_traffic_after_restore():
    _, learner = make_pair("backprop")
    update_learner(learner, FEATURES, TARGETS, session())
    saved = learner.snapshot_state()
    learner.restore_state(saved)
    state = learner.snapshot_state().state
    assert state["weight_input_hidden"] is state["_hidden_weights"][0]
    assert state["bias_hidden"] is state["_hidden_biases"][0]
    assert state["_traffic_steps"] == 1
    assert all(np.any(traffic > 0) for traffic in state["_traffic_sums"])


@pytest.mark.parametrize("kind", ["backprop", "circadian"])
def test_should_refuse_malformed_training_without_recording_a_completed_update(kind):
    _, learner = make_pair(kind)
    original = pickle.dumps(learner.snapshot_state().state)
    budget = session()
    with pytest.raises(ValueError):
        update_learner(learner, FEATURES, np.zeros((4, 2)), budget)
    assert budget.updates_completed == 0
    assert pickle.dumps(learner.snapshot_state().state) == original
