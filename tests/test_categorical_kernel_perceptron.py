"""Independent scalar arithmetic and bounded-state behavior for the kernel rule.

The fixed XOR dose tests nonlinear representational capacity, not MiniGrid
performance. The oracle uses Python lists and scalar pair sums independently of
the production update; it never chooses an endpoint or changes the kernel.
"""

from dataclasses import FrozenInstanceError
import math
import pickle
import random

import numpy as np
import pytest

from src.core import categorical_kernel_perceptron as kernel
from src.core.learner_ports import TrainingUpdateFailed


def _initial(dim=2, classes=3, capacity=128):
    return kernel.create_kernel_perceptron_snapshot(
        kernel.KernelPerceptronPolicy(dim, classes, capacity)
    )


def _pair(left, right):
    return (1.0 + sum(x * y for x, y in zip(left, right, strict=True)) / len(left)) ** 2


def _oracle(prototypes, coefficients, rows, labels, classes):
    prototypes = [list(row) for row in prototypes]
    coefficients = [list(row) for row in coefficients]
    mistakes = pairs = 0
    for features, target in zip(rows, labels, strict=True):
        scores = [0.0] * classes
        for prototype, alpha in zip(prototypes, coefficients, strict=True):
            value = _pair(features, prototype)
            pairs += 1
            for action in range(classes):
                scores[action] += value * alpha[action]
        predicted = max(range(classes), key=scores.__getitem__)
        if predicted != target:
            mistakes += 1
            if list(features) not in prototypes:
                prototypes.append(list(features))
                coefficients.append([0] * classes)
            index = prototypes.index(list(features))
            coefficients[index][target] += 1
            coefficients[index][predicted] -= 1
    return tuple(map(tuple, prototypes)), tuple(map(tuple, coefficients)), mistakes, pairs


def _oracle_probabilities(state, rows):
    probabilities = []
    for row in rows:
        scores = [
            sum(
                _pair(row, prototype) * alpha[action]
                for prototype, alpha in zip(state.prototypes, state.coefficients, strict=True)
            )
            for action in range(state.policy.num_classes)
        ]
        exponentials = [math.exp(value - max(scores)) for value in scores]
        probabilities.append([value / sum(exponentials) for value in exponentials])
    return probabilities


def test_should_match_scalar_kernel_order_coefficients_and_cost_after_every_row():
    rows = [[0.0, 0.0], [1.0, -1.0], [-1.0, 0.5], [1.0, 0.0], [0.0, 0.0]]
    labels = [2, 1, 0, 2, 1]
    state = _initial()
    prototypes = coefficients = ()
    mistakes = pairs = 0
    for index, (row, label) in enumerate(zip(rows, labels, strict=True), start=1):
        prototypes, coefficients, count, cost = _oracle(prototypes, coefficients, [row], [label], 3)
        mistakes += count
        pairs += cost
        state = kernel.update_kernel_perceptron(state, np.array([row]), np.array([[float(label)]]))
        assert state.prototypes == prototypes
        assert state.coefficients == coefficients
        assert (state.completed_updates, state.training_examples) == (index, index)
        assert (state.mistake_updates, state.training_kernel_evaluations) == (mistakes, pairs)
        np.testing.assert_allclose(
            kernel.predict_kernel_perceptron(state, np.array(rows)),
            _oracle_probabilities(state, rows),
            rtol=1e-14,
            atol=1e-15,
        )
    batch = kernel.update_kernel_perceptron(
        _initial(), np.array(rows), np.array(labels, dtype=np.float64).reshape(-1, 1)
    )
    assert batch.prototypes == prototypes and batch.coefficients == coefficients
    assert (batch.completed_updates, batch.training_examples) == (1, 5)
    assert (batch.mistake_updates, batch.training_kernel_evaluations) == (mistakes, pairs)


def test_should_keep_first_class_ties_supplied_order_and_cancelled_prototypes():
    zero = np.zeros((1, 1))
    state = kernel.update_kernel_perceptron(_initial(1), zero, np.array([[0.0]]))
    assert state.prototypes == state.coefficients == ()
    np.testing.assert_allclose(kernel.predict_kernel_perceptron(state, zero), [[1 / 3] * 3])
    state = kernel.update_kernel_perceptron(state, zero, np.array([[1.0]]))
    state = kernel.update_kernel_perceptron(state, zero, np.array([[0.0]]))
    assert state.prototypes == ((0.0,),)
    assert state.coefficients == ((0, 0, 0),)
    assert state.mistake_updates == 2 and state.training_kernel_evaluations == 1
    forward = kernel.update_kernel_perceptron(
        _initial(1), np.zeros((2, 1)), np.array([[1.0], [2.0]])
    )
    reverse = kernel.update_kernel_perceptron(
        _initial(1), np.zeros((2, 1)), np.array([[2.0], [1.0]])
    )
    assert forward.coefficients == ((-1, 0, 1),)
    assert reverse.coefficients == ((-1, 1, 0),)


def test_should_learn_fixed_xor_in_exactly_sixteen_ordered_passes():
    features = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    targets = np.array([[0.0], [1.0], [1.0], [0.0]])
    state = _initial(2, 2, 4)
    prototypes = coefficients = ()
    mistakes = pairs = 0
    for _ in range(16):
        prototypes, coefficients, count, cost = _oracle(
            prototypes, coefficients, features.tolist(), [0, 1, 1, 0], 2
        )
        mistakes += count
        pairs += cost
        state = kernel.update_kernel_perceptron(state, features, targets)
        assert state.prototypes == prototypes and state.coefficients == coefficients
    assert (state.completed_updates, state.training_examples) == (16, 64)
    assert (state.mistake_updates, state.training_kernel_evaluations) == (mistakes, pairs)
    assert kernel.predict_kernel_perceptron(state, features).argmax(axis=1).tolist() == [0, 1, 1, 0]


def test_should_softmax_large_legal_scores_stably_without_advancing_state():
    state = kernel.KernelPerceptronSnapshot(
        kernel.KernelPerceptronPolicy(1, 3),
        ((1.0,),),
        ((1000, 0, -1000),),
        1000,
        1000,
        1000,
        999,
    )
    before = pickle.dumps(state)
    probabilities = kernel.predict_kernel_perceptron(state, np.array([[-1.0], [0.0], [1.0]]))
    assert probabilities.dtype == np.float64 and np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(probabilities[0], [1 / 3] * 3)
    np.testing.assert_array_equal(probabilities[2], [1.0, 0.0, 0.0])
    assert pickle.dumps(state) == before


def test_should_own_bounded_storage_preserve_inputs_and_both_rng_states():
    state = _initial(296, 16)
    features, targets = np.zeros((128, 296)), np.full((128, 1), 15.0)
    before = pickle.dumps(state)
    rng_before = random.getstate(), pickle.dumps(np.random.get_state())
    result = kernel.update_kernel_perceptron(state, features, targets)
    copied = kernel.copy_kernel_perceptron_snapshot(result)
    assert (result.completed_updates, result.training_examples, result.mistake_updates) == (
        1,
        128,
        1,
    )
    assert result.training_kernel_evaluations == 127
    assert copied == result and copied is not result and copied.policy is not result.policy
    assert copied.prototypes is not result.prototypes
    assert copied.coefficients is not result.coefficients
    assert len(copied.prototypes) == 1 and len(copied.prototypes[0]) == 296
    assert type(copied.prototypes[0][0]) is float
    assert all(type(value) is int for row in copied.coefficients for value in row)
    assert pickle.dumps(state) == before
    assert not features.any() and np.all(targets == 15.0)
    predictions = kernel.predict_kernel_perceptron(result, features[:1])
    predictions.fill(0.0)
    features.fill(1.0)
    targets.fill(0.0)
    assert copied == result
    assert random.getstate() == rng_before[0]
    assert pickle.dumps(np.random.get_state()) == rng_before[1]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        setattr(result, "completed_updates", 2)


def test_should_allow_existing_key_mistakes_at_capacity_without_eviction():
    state = kernel.update_kernel_perceptron(_initial(1, 2, 1), np.zeros((1, 1)), np.array([[1.0]]))
    state = kernel.update_kernel_perceptron(state, np.zeros((1, 1)), np.array([[0.0]]))
    assert state.prototypes == ((0.0,),) and state.coefficients == ((0, 0),)
    state = kernel.update_kernel_perceptron(state, np.array([[-0.0]]), np.array([[1.0]]))
    assert state.prototypes == ((0.0,),) and state.coefficients == ((-1, 1),)
    assert (state.mistake_updates, state.training_kernel_evaluations) == (3, 2)


def test_should_refuse_full_pass_atomically_and_retain_observed_capacity_work():
    state = _initial(1, 2, 1)
    before = pickle.dumps(state)
    with pytest.raises(kernel.KernelPerceptronCapacityExceeded) as caught:
        kernel.update_kernel_perceptron(state, np.array([[0.0], [1.0]]), np.array([[1.0], [0.0]]))
    error = caught.value
    assert isinstance(error, TrainingUpdateFailed)
    assert error.definition == "categorical_polynomial_kernel_perceptron_mistakes_v1"
    assert error.work == {
        "attempted_passes": 1,
        "attempted_training_rows": 2,
        "training_rows_evaluated": 2,
        "kernel_evaluations_completed": 1,
        "mistake_updates_uncommitted": 1,
        "prototype_count_before": 0,
        "prototype_count_proposed": 2,
        "native_pass_committed": False,
    }
    detached = error.work
    detached["attempted_training_rows"] = 0
    assert error.work["attempted_training_rows"] == 2
    assert pickle.dumps(state) == before


@pytest.mark.parametrize(
    "dim,classes,capacity",
    [
        (True, 3, 4),
        (0, 3, 4),
        (297, 3, 4),
        (1.0, 3, 4),
        (1, 1, 4),
        (1, 17, 4),
        (1, True, 4),
        (1, 3, True),
        (1, 3, 0),
        (1, 3, 129),
    ],
)
def test_should_refuse_invalid_policy(dim, classes, capacity):
    with pytest.raises(ValueError):
        kernel.create_kernel_perceptron_snapshot(
            kernel.KernelPerceptronPolicy(dim, classes, capacity)
        )


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "float32",
        "subclass",
        "flat",
        "empty",
        "too_many",
        "width",
        "nan",
        "inf",
        "above",
        "below",
        "target_list",
        "target_dtype",
        "target_flat",
        "target_rows",
        "target_nan",
        "target_fraction",
        "target_negative",
        "target_outside",
    ],
)
def test_should_refuse_invalid_batch_before_changing_source(fault):
    features, targets = np.zeros((2, 2)), np.zeros((2, 1))
    if fault == "list":
        features = features.tolist()
    elif fault == "float32":
        features = features.astype(np.float32)
    elif fault == "subclass":
        features = features.view(type("ArraySubclass", (np.ndarray,), {}))
    elif fault == "flat":
        features = features.ravel()
    elif fault == "empty":
        features, targets = np.zeros((0, 2)), np.zeros((0, 1))
    elif fault == "too_many":
        features, targets = np.zeros((129, 2)), np.zeros((129, 1))
    elif fault == "width":
        features = np.zeros((2, 3))
    elif fault in {"nan", "inf", "above", "below"}:
        features[0, 0] = {"nan": float("nan"), "inf": float("inf"), "above": 1.01, "below": -1.01}[
            fault
        ]
    elif fault == "target_list":
        targets = targets.tolist()
    elif fault == "target_dtype":
        targets = targets.astype(np.int64)
    elif fault == "target_flat":
        targets = targets.ravel()
    elif fault == "target_rows":
        targets = np.zeros((1, 1))
    else:
        targets[0, 0] = {
            "target_nan": float("nan"),
            "target_fraction": 0.5,
            "target_negative": -1.0,
            "target_outside": 3.0,
        }[fault]
    state = _initial()
    before = pickle.dumps(state)
    with pytest.raises(ValueError):
        kernel.update_kernel_perceptron(state, features, targets)
    if not fault.startswith("target_"):
        with pytest.raises(ValueError):
            kernel.predict_kernel_perceptron(state, features)
    assert pickle.dumps(state) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("completed_updates", True),
        ("completed_updates", -1),
        ("completed_updates", 2**53),
        ("training_examples", 1),
        ("mistake_updates", 1),
        ("training_kernel_evaluations", True),
        ("training_kernel_evaluations", -1),
        ("training_kernel_evaluations", 1),
        ("prototypes", [[0.0, 0.0]]),
        ("coefficients", [[0, 0, 0]]),
        ("rogue", True),
    ],
)
def test_should_refuse_corrupt_initial_state_on_all_boundaries(field, value):
    state = _initial()
    object.__setattr__(state, field, value)
    for operation in (
        lambda: kernel.copy_kernel_perceptron_snapshot(state),
        lambda: kernel.predict_kernel_perceptron(state, np.zeros((1, 2))),
        lambda: kernel.update_kernel_perceptron(state, np.zeros((1, 2)), np.zeros((1, 1))),
    ):
        with pytest.raises(ValueError):
            operation()


@pytest.mark.parametrize(
    "field,value",
    [
        ("prototypes", ((float("nan"),),)),
        ("prototypes", ((float("inf"),),)),
        ("prototypes", ((1.01,),)),
        ("prototypes", ((0,),)),
        ("prototypes", ((0.0,), (0.0,))),
        ("coefficients", ((-1.0, 1.0),)),
        ("coefficients", ((True, -1),)),
        ("coefficients", ((-1, 0),)),
        ("coefficients", ((-2, 2),)),
        ("coefficients", ((-1, 1, 0),)),
        ("coefficients", ()),
    ],
)
def test_should_refuse_nonfinite_duplicate_or_impossible_trained_state(field, value):
    state = kernel.KernelPerceptronSnapshot(
        kernel.KernelPerceptronPolicy(1, 2), ((0.0,),), ((-1, 1),), 1, 1, 1, 0
    )
    object.__setattr__(state, field, value)
    with pytest.raises(ValueError):
        kernel.copy_kernel_perceptron_snapshot(state)


@pytest.mark.parametrize(
    "updates,rows,mistakes,pairs",
    [(2, 1, 0, 0), (1, 129, 0, 0), (1, 1, 2, 0), (1, 0, 0, 0), (1, 1, 0, 129)],
)
def test_should_refuse_impossible_counter_relationships(updates, rows, mistakes, pairs):
    state = _initial()
    for field, value in zip(
        (
            "completed_updates",
            "training_examples",
            "mistake_updates",
            "training_kernel_evaluations",
        ),
        (updates, rows, mistakes, pairs),
        strict=True,
    ):
        object.__setattr__(state, field, value)
    with pytest.raises(ValueError):
        kernel.copy_kernel_perceptron_snapshot(state)


def test_should_refuse_counter_overflow_before_any_publication():
    maximum = 2**53 - 1
    state = kernel.KernelPerceptronSnapshot(
        kernel.KernelPerceptronPolicy(1, 2), (), (), maximum, maximum, 0, 0
    )
    before = pickle.dumps(state)
    with pytest.raises(ValueError):
        kernel.update_kernel_perceptron(state, np.zeros((1, 1)), np.zeros((1, 1)))
    assert pickle.dumps(state) == before


@pytest.mark.parametrize("fault", ["duplicates", "total_magnitude", "capacity"])
def test_should_check_dictionary_uniqueness_total_alpha_bound_and_declared_capacity(fault):
    state = kernel.KernelPerceptronSnapshot(
        kernel.KernelPerceptronPolicy(1, 2, 2),
        ((0.0,), (1.0,)),
        ((-1, 1), (-1, 1)),
        2,
        2,
        2,
        1,
    )
    if fault == "duplicates":
        object.__setattr__(state, "prototypes", ((0.0,), (-0.0,)))
    elif fault == "total_magnitude":
        object.__setattr__(state, "coefficients", ((-2, 2), (-2, 2)))
    else:
        object.__setattr__(state.policy, "max_prototypes", 1)
    with pytest.raises(ValueError):
        kernel.copy_kernel_perceptron_snapshot(state)


@pytest.mark.parametrize(
    "field,value",
    [("feature_dim", True), ("num_classes", 1), ("max_prototypes", 129), ("rogue", True)],
)
def test_should_revalidate_tampered_owned_policy(field, value):
    state = _initial()
    object.__setattr__(state.policy, field, value)
    with pytest.raises(ValueError):
        kernel.copy_kernel_perceptron_snapshot(state)
