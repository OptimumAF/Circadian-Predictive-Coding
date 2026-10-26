"""Native port behavior, ownership and atomic refusal for the perceptron adapter."""

import pickle
import random

import numpy as np
import pytest

from src.adapters.perceptron_learner import PerceptronLearner
from src.core.categorical_perceptron import PerceptronPolicy


def _data():
    return np.array([[-1.0], [1.0], [0.0]]), np.array([[0.0], [1.0], [2.0]])


def test_should_report_actual_pass_mistakes_and_native_counters():
    learner = PerceptronLearner(PerceptronPolicy(1, 3))
    features, targets = _data()
    # Scalar arithmetic: first zero-score class0 is correct; x=1 chooses0
    # and updates class1/class0; x=0 then chooses1 and updates class2/class1.
    diagnostic = learner.train_batch(features, targets)
    assert diagnostic.definition == "categorical_perceptron_mistakes_v1"
    assert diagnostic.value == 2.0
    state = learner.snapshot_state()
    assert state.coefficients == ((-1.0, 1.0, 0.0), (-1.0, 0.0, 1.0))
    assert state.completed_updates == learner.completed_updates == 1
    assert state.training_examples == learner.training_examples == 3
    assert state.mistake_updates == 2


def test_should_fork_full_state_and_restore_without_aliasing_or_rng_changes():
    learner = PerceptronLearner(PerceptronPolicy(1, 3))
    features, targets = _data()
    random_before = random.getstate(), pickle.dumps(np.random.get_state())
    learner.train_batch(features, targets)
    saved = learner.snapshot_state()
    saved_bytes = pickle.dumps(saved)
    forked = learner.fork()
    assert forked is not learner and forked.snapshot_state() == saved
    forked.train_batch(np.zeros((1, 1)), np.array([[0.0]]))
    assert pickle.dumps(learner.snapshot_state()) == saved_bytes
    assert forked.snapshot_state() != saved
    learner.train_batch(np.zeros((1, 1)), np.array([[0.0]]))
    learner.restore_state(saved)
    assert learner.snapshot_state() == saved
    assert learner.snapshot_state() is not saved
    object.__setattr__(saved, "training_examples", 999)
    assert pickle.dumps(learner.snapshot_state()) == saved_bytes
    assert random.getstate() == random_before[0]
    assert pickle.dumps(np.random.get_state()) == random_before[1]


def test_should_return_owned_probabilities_without_advancing_state():
    learner = PerceptronLearner(PerceptronPolicy(1, 3))
    features, targets = _data()
    learner.train_batch(features, targets)
    before = pickle.dumps(learner.snapshot_state())
    predictions = learner.predict(features)
    assert predictions.shape == (3, 3) and predictions.dtype == np.float64
    assert np.isfinite(predictions).all()
    np.testing.assert_allclose(predictions.sum(axis=1), 1.0)
    expected = predictions.copy()
    predictions.fill(0.0)
    np.testing.assert_array_equal(learner.predict(features), expected)
    assert pickle.dumps(learner.snapshot_state()) == before


@pytest.mark.parametrize("fault", ["batch", "snapshot", "policy", "rogue"])
def test_should_refuse_invalid_training_or_restore_atomically(fault):
    learner = PerceptronLearner(PerceptronPolicy(1, 3))
    before = pickle.dumps(learner.snapshot_state())
    with pytest.raises(ValueError):
        if fault == "batch":
            learner.train_batch(np.array([[float("nan")]]), np.array([[1.0]]))
        elif fault == "policy":
            other = PerceptronLearner(PerceptronPolicy(2, 3)).snapshot_state()
            learner.restore_state(other)
        else:
            corrupt = learner.snapshot_state()
            object.__setattr__(corrupt, "completed_updates" if fault == "snapshot" else "rogue", 7)
            learner.restore_state(corrupt)
    assert pickle.dumps(learner.snapshot_state()) == before


def test_should_own_policy_and_refuse_rogue_policy_at_admission():
    policy = PerceptronPolicy(1, 3)
    learner = PerceptronLearner(policy)
    before = learner.snapshot_state()
    object.__setattr__(policy, "feature_dim", 2)
    assert learner.snapshot_state() == before
    rogue = PerceptronPolicy(1, 3)
    object.__setattr__(rogue, "unexpected", True)
    with pytest.raises(ValueError):
        PerceptronLearner(rogue)
