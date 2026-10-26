"""Scalar oracles and bounded categorical diagnostics, without learner work."""

from math import log
from typing import Any

import numpy as np
import pytest

from src.core.shift_signals import summarize_categorical_shift_batch, summarize_shift_batch


@pytest.mark.parametrize("classes", [2, 3, 6, 16])
def test_should_match_scalar_entropy_residual_and_means_without_modifying_arrays(classes):
    features = np.array([[-1.0, 0.5], [1.0, -0.25]])
    probabilities = np.full((2, classes), 0.5 / (classes - 1))
    probabilities[0, 0] = probabilities[1, -1] = 0.5
    labels = np.array([[0.0], [0.0]])
    original = [array.copy() for array in (features, probabilities, labels)]
    metrics = summarize_categorical_shift_batch(features, probabilities, labels, classes)
    entropy = (
        sum(-p * log(p) for row in probabilities.tolist() for p in row if p > 0) / 2 / log(classes)
    )
    assert metrics.uncertainty == pytest.approx(entropy)
    assert metrics.residual == pytest.approx(
        (1 - probabilities[0, 0] + 1 - probabilities[1, 0]) / 2
    )
    assert metrics.feature_mean == (0.0, 0.125) and metrics.rows == 2
    for actual, expected in zip((features, probabilities, labels), original, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("classes", [2, 3, 6, 16])
def test_should_bound_uniform_and_exact_predictions_and_handle_zero_probabilities(classes):
    features = np.zeros((2, 9))
    labels = np.array([[0.0], [float(classes - 1)]])
    uniform = summarize_categorical_shift_batch(
        features, np.full((2, classes), 1 / classes), labels, classes
    )
    assert uniform.uncertainty == pytest.approx(1.0) and 0 <= uniform.uncertainty <= 1
    assert uniform.residual == pytest.approx(1 - 1 / classes)
    exact = np.zeros((2, classes))
    exact[0, 0] = exact[1, -1] = 1
    correct = summarize_categorical_shift_batch(features, exact, labels, classes)
    assert correct.uncertainty == correct.residual == 0
    assert (
        summarize_categorical_shift_batch(features, exact[::-1].copy(), labels, classes).residual
        == 1
    )


def test_should_agree_with_binary_hard_labels_without_changing_binary_soft_label_semantics():
    features = np.array([[-0.5], [0.5]])
    positive = np.array([[0.2], [0.9]])
    labels = np.array([[0.0], [1.0]])
    categorical = summarize_categorical_shift_batch(
        features, np.hstack((1 - positive, positive)), labels, 2
    )
    binary = summarize_shift_batch(features, positive, labels)
    assert categorical.rows == binary.rows and categorical.feature_mean == binary.feature_mean
    assert categorical.uncertainty == pytest.approx(binary.uncertainty)
    assert categorical.residual == pytest.approx(binary.residual)
    soft = np.array([[0.25], [0.75]])
    assert summarize_shift_batch(features, positive, soft).residual == pytest.approx(0.1)
    with pytest.raises(ValueError, match="hard class"):
        summarize_categorical_shift_batch(features, np.hstack((1 - positive, positive)), soft, 2)


def test_should_preserve_probability_tolerance_without_normalizing_values():
    probabilities = np.array([[0.2 + 5e-7, 0.3, 0.5]])
    original = probabilities.copy()
    metric = summarize_categorical_shift_batch(np.zeros((1, 1)), probabilities, np.zeros((1, 1)), 3)
    assert metric.residual == 1 - probabilities[0, 0]
    np.testing.assert_array_equal(probabilities, original)


@pytest.mark.parametrize("invalid", [True, 1, 17, 6.0, np.int64(6), None])
def test_should_refuse_undeclared_class_counts(invalid):
    with pytest.raises(ValueError):
        summarize_categorical_shift_batch(
            np.zeros((1, 9)), np.full((1, 6), 1 / 6), np.zeros((1, 1)), invalid
        )


@pytest.mark.parametrize(
    "fault",
    [
        "feature_list",
        "feature_float32",
        "feature_nan",
        "feature_range",
        "feature_width",
        "zero_rows",
        "129rows",
        "probability_list",
        "probability_float32",
        "probability_width",
        "probability_rows",
        "probability_nan",
        "probability_range",
        "probability_sum",
        "target_float32",
        "target_shape",
        "target_nan",
        "target_fraction",
        "target_negative",
        "target_class",
    ],
)
def test_should_refuse_invalid_bounded_metadata_and_values_without_coercion(fault):
    features: Any = np.zeros((2, 9))
    probabilities: Any = np.full((2, 6), 1 / 6)
    labels: Any = np.zeros((2, 1))
    if fault == "feature_list":
        features = features.tolist()
    elif fault == "feature_float32":
        features = features.astype(np.float32)
    elif fault == "feature_nan":
        features[-1, -1] = np.nan
    elif fault == "feature_range":
        features[-1, -1] = 1.01
    elif fault == "feature_width":
        features = np.zeros((2, 17))
    elif fault in ("zero_rows", "129rows"):
        rows = 0 if fault == "zero_rows" else 129
        features, probabilities, labels = (
            np.zeros((rows, 9)),
            np.full((rows, 6), 1 / 6),
            np.zeros((rows, 1)),
        )
    elif fault == "probability_list":
        probabilities = probabilities.tolist()
    elif fault == "probability_float32":
        probabilities = probabilities.astype(np.float32)
    elif fault == "probability_width":
        probabilities = np.full((2, 5), 0.2)
    elif fault == "probability_rows":
        probabilities = probabilities[:1]
    elif fault == "probability_nan":
        probabilities[-1, -1] = np.nan
    elif fault == "probability_range":
        probabilities[-1, -1] = -0.1
    elif fault == "probability_sum":
        probabilities[-1, -1] = 0.5
    elif fault == "target_float32":
        labels = labels.astype(np.float32)
    elif fault == "target_shape":
        labels = labels[:, 0]
    else:
        labels[-1, 0] = {
            "target_nan": np.nan,
            "target_fraction": 0.5,
            "target_negative": -1,
            "target_class": 6,
        }[fault]
    with pytest.raises(ValueError):
        summarize_categorical_shift_batch(features, probabilities, labels, 6)
