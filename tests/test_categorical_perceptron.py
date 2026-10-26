"""Independent list arithmetic for the fixed bounded multiclass perceptron.

The oracle makes one ordered mistake-driven pass; it does not call the production
update or use a numerical optimizer. Tiny fixed examples are engineering checks,
not evidence about the MiniGrid task or comparisons with other learners.
"""

from dataclasses import FrozenInstanceError
import pickle
import random

import numpy as np
import pytest

from src.core import categorical_perceptron as perceptron


def _oracle(coefficients, rows, labels):
    result = [list(row) for row in coefficients]
    mistakes = 0
    for features, target in zip(rows, labels, strict=True):
        augmented = list(features) + [1.0]
        scores = [
            sum(value * result[index][action] for index, value in enumerate(augmented))
            for action in range(len(result[0]))
        ]
        predicted = max(range(len(scores)), key=scores.__getitem__)
        if predicted != target:
            mistakes += 1
            for index, value in enumerate(augmented):
                result[index][target] += value
                result[index][predicted] -= value
    return tuple(tuple(row) for row in result), mistakes


def _initial(dim=2, classes=3):
    return perceptron.create_perceptron_snapshot(perceptron.PerceptronPolicy(dim, classes))


def test_should_match_independent_coefficients_after_every_ordered_row():
    rows = [[0.0, 0.0], [1.0, -1.0], [-1.0, 0.5], [1.0, 0.0], [0.0, 0.0]]
    labels = [2, 1, 0, 2, 1]
    state = _initial()
    expected = state.coefficients
    total_mistakes = 0
    for index, (row, label) in enumerate(zip(rows, labels, strict=True), start=1):
        expected, mistakes = _oracle(expected, [row], [label])
        total_mistakes += mistakes
        state = perceptron.update_perceptron(
            state, np.array([row], dtype=np.float64), np.array([[float(label)]])
        )
        assert state.coefficients == expected
        assert state.completed_updates == index
        assert state.training_examples == index
        assert state.mistake_updates == total_mistakes
    batch = perceptron.update_perceptron(
        _initial(), np.array(rows), np.array(labels, dtype=np.float64).reshape(-1, 1)
    )
    assert batch.coefficients == expected
    assert batch.completed_updates == 1 and batch.training_examples == 5
    assert batch.mistake_updates == total_mistakes


def test_should_use_first_score_argmax_and_intercept_and_preserve_supplied_order():
    # All-zero inputs isolate intercept updates. The reversed order must produce
    # a different state rather than an implicitly shuffled or averaged batch.
    features = np.zeros((2, 1), dtype=np.float64)
    forward = perceptron.update_perceptron(_initial(1), features, np.array([[1.0], [2.0]]))
    reverse = perceptron.update_perceptron(_initial(1), features, np.array([[2.0], [1.0]]))
    assert forward.coefficients == ((0.0, 0.0, 0.0), (-1.0, 0.0, 1.0))
    assert reverse.coefficients == ((0.0, 0.0, 0.0), (-1.0, 1.0, 0.0))
    assert forward.mistake_updates == reverse.mistake_updates == 2
    correct_tie = perceptron.update_perceptron(_initial(1), np.zeros((1, 1)), np.array([[0.0]]))
    assert correct_tie.coefficients == ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert correct_tie.mistake_updates == 0
    probabilities = perceptron.predict_perceptron(_initial(1), np.zeros((1, 1)))
    np.testing.assert_allclose(probabilities, [[1 / 3] * 3])
    assert probabilities.argmax(axis=1).tolist() == [0]


def test_should_learn_fixed_three_class_separable_rows_in_eight_ordered_passes():
    features = np.array([[-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]])
    labels = np.array([[0.0], [1.0], [2.0]])
    state = _initial()
    expected = state.coefficients
    mistakes = 0
    for _ in range(8):
        expected, count = _oracle(expected, features.tolist(), [0, 1, 2])
        mistakes += count
        state = perceptron.update_perceptron(state, features, labels)
        assert state.coefficients == expected
    assert state.completed_updates == 8 and state.training_examples == 24
    assert state.mistake_updates == mistakes
    assert perceptron.predict_perceptron(state, features).argmax(axis=1).tolist() == [0, 1, 2]


def test_should_softmax_large_legal_scores_stably_and_keep_first_class_ties():
    state = perceptron.PerceptronSnapshot(
        policy=perceptron.PerceptronPolicy(1, 3),
        coefficients=((1000.0, 0.0, -1000.0), (0.0, 0.0, 0.0)),
        completed_updates=1000,
        training_examples=1000,
        mistake_updates=1000,
    )
    result = perceptron.predict_perceptron(state, np.array([[-1.0], [0.0], [1.0]]))
    assert result.dtype == np.float64 and np.isfinite(result).all()
    np.testing.assert_allclose(result.sum(axis=1), 1.0)
    np.testing.assert_array_equal(result[0], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(result[2], [1.0, 0.0, 0.0])
    assert result.argmax(axis=1).tolist() == [2, 0, 0]


def test_should_admit_upper_geometry_single_pass_and_bounded_immutable_storage():
    state = _initial(296, 16)
    features = np.zeros((128, 296), dtype=np.float64)
    targets = np.full((128, 1), 15.0)
    result = perceptron.update_perceptron(state, features, targets)
    assert result.completed_updates == 1 and result.training_examples == 128
    assert result.mistake_updates == 1
    assert len(result.coefficients) == 297
    assert all(type(row) is tuple and len(row) == 16 for row in result.coefficients)
    assert all(type(value) is float for row in result.coefficients for value in row)
    assert perceptron.predict_perceptron(result, features).argmax(axis=1).tolist() == [15] * 128
    assert set(vars(result)) == {
        "policy",
        "coefficients",
        "completed_updates",
        "training_examples",
        "mistake_updates",
    }
    with pytest.raises((FrozenInstanceError, AttributeError)):
        setattr(result, "completed_updates", 2)


def test_should_preserve_inputs_rng_and_earlier_state_and_own_copies():
    features = np.array([[1.0, -1.0]])
    targets = np.array([[2.0]])
    before_x, before_y = features.copy(), targets.copy()
    rng_before = (random.getstate(), pickle.dumps(np.random.get_state()))
    original = _initial()
    before_state = pickle.dumps(original)
    result = perceptron.update_perceptron(original, features, targets)
    copied = perceptron.copy_perceptron_snapshot(result)
    assert copied == result and copied is not result and copied.policy is not result.policy
    assert copied.coefficients is not result.coefficients
    assert pickle.dumps(original) == before_state
    np.testing.assert_array_equal(features, before_x)
    np.testing.assert_array_equal(targets, before_y)
    probabilities = perceptron.predict_perceptron(result, features)
    probabilities.fill(0.0)
    features.fill(0.0)
    targets.fill(0.0)
    assert copied == result
    assert random.getstate() == rng_before[0]
    assert pickle.dumps(np.random.get_state()) == rng_before[1]


@pytest.mark.parametrize(
    "dim,classes", [(True, 3), (0, 3), (297, 3), (1.0, 3), (1, 1), (1, 17), (1, True)]
)
def test_should_refuse_invalid_policy(dim, classes):
    with pytest.raises(ValueError):
        perceptron.create_perceptron_snapshot(perceptron.PerceptronPolicy(dim, classes))


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
def test_should_refuse_invalid_batch_without_changing_original(fault):
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
        perceptron.update_perceptron(state, features, targets)
    if not fault.startswith("target_"):
        with pytest.raises(ValueError):
            perceptron.predict_perceptron(state, features)
    assert pickle.dumps(state) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("completed_updates", True),
        ("completed_updates", -1),
        ("completed_updates", 2**53),
        ("training_examples", 1),
        ("mistake_updates", 1),
        ("coefficients", ((0.0, 0.0, 0.0),) * 2),
        ("coefficients", ((float("nan"), 0.0, 0.0),) * 3),
        ("coefficients", ((float("inf"), 0.0, 0.0),) * 3),
        ("coefficients", ((0, 0, 0),) * 3),
        ("coefficients", [[0.0, 0.0, 0.0]] * 3),
        ("coefficients", ((1.0, 0.0, 0.0),) * 3),
        ("rogue", "unexpected"),
    ],
)
def test_should_refuse_corrupt_or_extended_snapshot_on_every_boundary(field, value):
    state = _initial()
    object.__setattr__(state, field, value)
    for operation in (
        lambda: perceptron.copy_perceptron_snapshot(state),
        lambda: perceptron.predict_perceptron(state, np.zeros((1, 2))),
        lambda: perceptron.update_perceptron(state, np.zeros((1, 2)), np.zeros((1, 1))),
    ):
        with pytest.raises(ValueError):
            operation()


def test_should_refuse_counter_overflow_before_publishing_any_update():
    maximum = 2**53 - 1
    state = perceptron.PerceptronSnapshot(
        perceptron.PerceptronPolicy(1, 2), ((0.0, 0.0), (0.0, 0.0)), maximum, maximum, 0
    )
    before = pickle.dumps(state)
    with pytest.raises(ValueError):
        perceptron.update_perceptron(state, np.zeros((1, 1)), np.zeros((1, 1)))
    assert pickle.dumps(state) == before


@pytest.mark.parametrize(
    "field,value", [("feature_dim", True), ("num_classes", 1), ("rogue", True)]
)
def test_should_revalidate_tampered_policy_inside_snapshot(field, value):
    state = _initial()
    object.__setattr__(state.policy, field, value)
    with pytest.raises(ValueError):
        perceptron.copy_perceptron_snapshot(state)


@pytest.mark.parametrize("updates,rows,mistakes", [(2, 1, 0), (1, 129, 0), (1, 1, 2), (1, 0, 0)])
def test_should_refuse_impossible_counter_relationships(updates, rows, mistakes):
    state = _initial()
    object.__setattr__(state, "completed_updates", updates)
    object.__setattr__(state, "training_examples", rows)
    object.__setattr__(state, "mistake_updates", mistakes)
    with pytest.raises(ValueError):
        perceptron.copy_perceptron_snapshot(state)
