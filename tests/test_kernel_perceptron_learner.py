"""Native port ownership, complete forks and failed-pass accounting."""

import pickle
import random

import numpy as np
import pytest

from src.adapters.kernel_perceptron_learner import KernelPerceptronLearner
from src.core.categorical_kernel_perceptron import (
    KernelPerceptronCapacityExceeded,
    KernelPerceptronPolicy,
)


def _data():
    return np.array([[-1.0], [1.0], [0.0]]), np.array([[0.0], [1.0], [2.0]])


def test_should_report_actual_mistakes_dictionary_and_kernel_pair_cost():
    learner = KernelPerceptronLearner(KernelPerceptronPolicy(1, 3))
    features, targets = _data()
    diagnostic = learner.train_batch(features, targets)
    assert diagnostic.definition == "categorical_polynomial_kernel_perceptron_mistakes_v1"
    assert diagnostic.value == 2.0
    state = learner.snapshot_state()
    assert state.prototypes == ((1.0,), (0.0,))
    assert state.coefficients == ((-1, 1, 0), (0, -1, 1))
    assert state.completed_updates == learner.completed_updates == 1
    assert state.training_examples == learner.training_examples == 3
    assert (state.mistake_updates, state.training_kernel_evaluations) == (2, 1)


def test_should_fork_restore_full_dictionary_without_aliases_or_rng_changes():
    learner = KernelPerceptronLearner(KernelPerceptronPolicy(1, 3))
    features, targets = _data()
    rng_before = random.getstate(), pickle.dumps(np.random.get_state())
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
    assert learner.snapshot_state() == saved and learner.snapshot_state() is not saved
    object.__setattr__(saved, "prototypes", ((0.5,),))
    assert pickle.dumps(learner.snapshot_state()) == saved_bytes
    assert random.getstate() == rng_before[0]
    assert pickle.dumps(np.random.get_state()) == rng_before[1]


def test_should_return_owned_stable_probabilities_without_advancing_native_work():
    learner = KernelPerceptronLearner(KernelPerceptronPolicy(1, 3))
    features, targets = _data()
    learner.train_batch(features, targets)
    before = pickle.dumps(learner.snapshot_state())
    result = learner.predict(features)
    assert result.shape == (3, 3) and result.dtype == np.float64
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result.sum(axis=1), 1.0)
    expected = result.copy()
    result.fill(0.0)
    np.testing.assert_array_equal(learner.predict(features), expected)
    assert pickle.dumps(learner.snapshot_state()) == before


def test_should_keep_published_state_after_capacity_failure_and_expose_paid_prefix():
    learner = KernelPerceptronLearner(KernelPerceptronPolicy(1, 2, 1))
    before = pickle.dumps(learner.snapshot_state())
    with pytest.raises(KernelPerceptronCapacityExceeded) as caught:
        learner.train_batch(np.array([[0.0], [1.0]]), np.array([[1.0], [0.0]]))
    assert caught.value.work["training_rows_evaluated"] == 2
    assert caught.value.work["kernel_evaluations_completed"] == 1
    assert caught.value.work["mistake_updates_uncommitted"] == 1
    assert caught.value.work["native_pass_committed"] is False
    assert pickle.dumps(learner.snapshot_state()) == before
    learner.train_batch(np.zeros((1, 1)), np.array([[1.0]]))
    learner.train_batch(np.zeros((1, 1)), np.array([[0.0]]))
    assert learner.completed_updates == 2 and learner.training_examples == 2
    assert learner.snapshot_state().prototypes == ((0.0,),)


@pytest.mark.parametrize("fault", ["batch", "snapshot", "policy", "capacity", "rogue"])
def test_should_refuse_bad_batches_and_incompatible_restores_atomically(fault):
    learner = KernelPerceptronLearner(KernelPerceptronPolicy(1, 3, 4))
    before = pickle.dumps(learner.snapshot_state())
    with pytest.raises(ValueError):
        if fault == "batch":
            learner.train_batch(np.array([[float("nan")]]), np.array([[1.0]]))
        elif fault in {"policy", "capacity"}:
            other = KernelPerceptronLearner(
                KernelPerceptronPolicy(2 if fault == "policy" else 1, 3, 5)
            )
            learner.restore_state(other.snapshot_state())
        else:
            corrupt = learner.snapshot_state()
            object.__setattr__(corrupt, "completed_updates" if fault == "snapshot" else "rogue", 7)
            learner.restore_state(corrupt)
    assert pickle.dumps(learner.snapshot_state()) == before


def test_should_own_constructor_policy_and_validate_rogue_policy():
    policy = KernelPerceptronPolicy(1, 3, 4)
    learner = KernelPerceptronLearner(policy)
    before = learner.snapshot_state()
    object.__setattr__(policy, "max_prototypes", 1)
    assert learner.snapshot_state() == before
    rogue = KernelPerceptronPolicy(1, 3, 4)
    object.__setattr__(rogue, "unexpected", True)
    with pytest.raises(ValueError):
        KernelPerceptronLearner(rogue)
