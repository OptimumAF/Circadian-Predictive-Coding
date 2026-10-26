"""Analytic exact-input frequency bounds without any learner or fitting work."""

from dataclasses import FrozenInstanceError, asdict
from math import log
import pickle
import random
from typing import Any

import numpy as np
import pytest

from src.core.categorical_input_bounds import CategoricalInputBounds, categorical_input_bounds
from src.core.categorical_probability_scores import probability_scores


def example():
    return (
        np.array([[0, 1]] * 3 + [[1, 0]] * 2 + [[-1, -1]] * 3, dtype=np.float64),
        np.array([[0], [0], [1], [1], [2], [2], [2], [2]], dtype=np.float64),
    )


def test_should_use_modal_counts_and_conditional_entropy_with_equal_row_frequencies():
    features, targets = example()
    result = categorical_input_bounds(features, targets, 4)
    assert type(result) is CategoricalInputBounds
    assert asdict(result) == dict(
        rows=8,
        feature_dim=2,
        num_classes=4,
        unique_input_keys=3,
        conflicting_input_keys=2,
        conflicting_rows=5,
        max_correct_rows=6,
        min_incorrect_rows=2,
        row_accuracy_upper_bound=0.75,
        normalized_ce_lower_bound=pytest.approx((-2 * log(2 / 3) - log(1 / 3) + 2 * log(2)) / 8),
    )
    assert all(
        type(value) is int
        for key, value in asdict(result).items()
        if key not in {"row_accuracy_upper_bound", "normalized_ce_lower_bound"}
    )
    assert type(result.row_accuracy_upper_bound) is type(result.normalized_ce_lower_bound) is float


def test_should_attain_bounds_on_unrestricted_normalized_simplex_without_absent_class_mass():
    features, targets = example()
    probabilities = np.array(
        [[2 / 3, 1 / 3, 0, 0]] * 3 + [[0, 0.5, 0.5, 0]] * 2 + [[0, 0, 1, 0]] * 3, dtype=np.float64
    )
    bound = categorical_input_bounds(features, targets, 4)
    score = probability_scores(probabilities, targets, 4)
    assert score["ce"] == pytest.approx(bound.normalized_ce_lower_bound)
    assert score["accuracy"] == bound.row_accuracy_upper_bound


@pytest.mark.parametrize("classes", (2, 3, 7, 16))
def test_should_allow_unique_rows_to_fit_any_declared_labels_without_capacity_claim(classes):
    features = np.array([[-1], [0], [1]], dtype=np.float64)
    targets = np.array([[0], [classes - 1], [0]], dtype=np.float64)
    bound = categorical_input_bounds(features, targets, classes)
    assert bound.unique_input_keys == bound.max_correct_rows == 3
    assert bound.conflicting_rows == bound.conflicting_input_keys == bound.min_incorrect_rows == 0
    assert bound.row_accuracy_upper_bound == 1.0 and bound.normalized_ce_lower_bound == 0.0


def test_should_count_three_way_tie_and_not_depend_on_class_number_order():
    features = np.zeros((3, 2), dtype=np.float64)
    for labels in ([0, 1, 2], [2, 1, 0]):
        bound = categorical_input_bounds(features, np.array(labels, dtype=np.float64)[:, None], 7)
        assert bound.max_correct_rows == 1 and bound.min_incorrect_rows == 2
        assert bound.conflicting_input_keys == 1 and bound.conflicting_rows == 3
        assert bound.row_accuracy_upper_bound == 1 / 3
        assert bound.normalized_ce_lower_bound == pytest.approx(log(3))


def test_should_change_bounds_with_label_frequency_instead_of_only_unique_labels():
    features = np.zeros((4, 1), dtype=np.float64)
    skewed = categorical_input_bounds(features, np.array([[0], [0], [0], [1]], dtype=np.float64), 2)
    balanced = categorical_input_bounds(
        features, np.array([[0], [0], [1], [1]], dtype=np.float64), 2
    )
    assert skewed.row_accuracy_upper_bound == 0.75 and balanced.row_accuracy_upper_bound == 0.5
    assert skewed.normalized_ce_lower_bound == pytest.approx(-0.75 * log(0.75) - 0.25 * log(0.25))
    assert balanced.normalized_ce_lower_bound == pytest.approx(log(2))


def test_should_preserve_bounds_under_joint_row_permutation_and_uniform_duplication():
    features, targets = example()
    original = categorical_input_bounds(features, targets, 4)
    order = [7, 4, 2, 0, 5, 1, 6, 3]
    assert categorical_input_bounds(features[order], targets[order], 4) == original
    doubled = categorical_input_bounds(
        np.repeat(features, 2, axis=0), np.repeat(targets, 2, axis=0), 4
    )
    for field in ("rows", "conflicting_rows", "max_correct_rows", "min_incorrect_rows"):
        assert getattr(doubled, field) == 2 * getattr(original, field)
    for field in (
        "unique_input_keys",
        "conflicting_input_keys",
        "row_accuracy_upper_bound",
        "normalized_ce_lower_bound",
    ):
        assert getattr(doubled, field) == pytest.approx(getattr(original, field))


def test_should_merge_signed_zero_but_keep_adjacent_finite_values_distinct():
    features = np.array([[0.0, -0.0], [-0.0, 0.0], [np.nextafter(0.0, 1.0), 0.0]], dtype=np.float64)
    targets = np.array([[0], [1], [1]], dtype=np.float64)
    bound = categorical_input_bounds(features, targets, 2)
    assert bound.unique_input_keys == 2 and bound.conflicting_input_keys == 1
    assert bound.conflicting_rows == 2 and bound.max_correct_rows == 2
    assert bound.normalized_ce_lower_bound == pytest.approx(2 * log(2) / 3)


def test_should_keep_minimum_positive_frequency_above_existing_score_clip():
    features = np.zeros((128, 296), dtype=np.float64)
    targets = np.zeros((128, 1), dtype=np.float64)
    targets[-1] = 15
    bound = categorical_input_bounds(features, targets, 16)
    assert bound.rows == 128 and bound.feature_dim == 296
    assert bound.max_correct_rows == 127 and bound.min_incorrect_rows == 1
    assert bound.normalized_ce_lower_bound == pytest.approx(
        -(127 * log(127 / 128) + log(1 / 128)) / 128
    )


def test_should_accept_readonly_noncontiguous_views_without_mutation_or_rng_use():
    storage = np.array([[0, 0.8, 1, -0.8], [0, 0.7, 1, -0.7], [1, 0.6, 0, -0.6]], dtype=np.float64)
    features = storage[:, ::2]
    target_storage = np.array([[0, 9], [1, 9], [1, 9]], dtype=np.float64)
    targets = target_storage[:, :1]
    features.flags.writeable = targets.flags.writeable = False
    before = pickle.dumps((storage, target_storage, np.random.get_state(), random.getstate()))
    bound = categorical_input_bounds(features, targets, 2)
    assert bound.max_correct_rows == 2
    assert (
        pickle.dumps((storage, target_storage, np.random.get_state(), random.getstate())) == before
    )
    assert not features.flags.writeable and not targets.flags.writeable
    with pytest.raises(FrozenInstanceError):
        setattr(bound, "rows", 99)


@pytest.mark.parametrize(
    "features",
    (
        [[0.0]],
        np.array([[0]], dtype=np.int64),
        np.array([[0]], dtype=np.float32),
        np.array([[0.0]], dtype=object),
        np.zeros((0, 1)),
        np.zeros((129, 1)),
        np.zeros((1, 0)),
        np.zeros((1, 297)),
        np.zeros((1, 1, 1)),
        np.array([0.0]),
        np.array([[float("nan")]]),
        np.array([[float("inf")]]),
        np.array([[-float("inf")]]),
        np.array([[1.0000000001]]),
        np.array([[-1.0000000001]]),
    ),
)
def test_should_reject_invalid_feature_types_geometry_and_values(features):
    with pytest.raises(ValueError):
        categorical_input_bounds(features, np.zeros((1, 1)), 2)


def test_should_refuse_ndarray_subclass_without_coercion():
    class ForeignArray(np.ndarray):
        pass

    for field in ("features", "targets"):
        features, targets = np.zeros((1, 1)), np.zeros((1, 1))
        if field == "features":
            features = features.view(ForeignArray)
        else:
            targets = targets.view(ForeignArray)
        with pytest.raises(ValueError):
            categorical_input_bounds(features, targets, 2)


@pytest.mark.parametrize(
    "targets",
    (
        [[0.0]],
        np.array([[0]], dtype=np.int64),
        np.array([[0]], dtype=np.float32),
        np.array([[True]]),
        np.array([0.0]),
        np.zeros((0, 1)),
        np.zeros((2, 1)),
        np.zeros((1, 2)),
        np.array([[0.5]]),
        np.array([[-1.0]]),
        np.array([[2.0]]),
        np.array([[float("nan")]]),
        np.array([[float("inf")]]),
    ),
)
def test_should_reject_non_hard_or_invalid_target_arrays(targets):
    with pytest.raises(ValueError):
        categorical_input_bounds(np.zeros((1, 1)), targets, 2)


@pytest.mark.parametrize("classes", (True, False, 1, 17, 2.0, np.int64(2), "2", None))
def test_should_reject_non_native_or_unsupported_class_counts(classes: Any):
    with pytest.raises(ValueError):
        categorical_input_bounds(np.zeros((1, 1)), np.zeros((1, 1)), classes)
