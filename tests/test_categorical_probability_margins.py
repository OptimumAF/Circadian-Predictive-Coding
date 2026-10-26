"""Target probability margins retain every row, tie, and absent target class."""

import json

import numpy as np
import pytest

from src.core.categorical_probability_margins import DEFINITION, probability_margins


def test_should_report_signed_margins_and_first_argmax_ties_separately():
    predictions = np.array(
        [[0.6, 0.3, 0.1, 0.0], [0.4, 0.4, 0.1, 0.1], [0.4, 0.4, 0.1, 0.1], [0.1, 0.2, 0.6, 0.1]],
        dtype=np.float64,
    )
    targets = np.array([[0], [0], [1], [1]], dtype=np.float64)

    result = probability_margins(predictions, targets, 4)

    assert DEFINITION == result["definition"] == "categorical_target_probability_margin_v1"
    assert result["num_classes"] == 4 and result["rows"] == 4
    assert result["target_margins"] == pytest.approx([0.3, 0.0, 0.0, -0.4])
    assert result["mean_margin"] == pytest.approx(-0.025)
    assert result["min_margin"] == pytest.approx(-0.4)
    assert result["max_margin"] == pytest.approx(0.3)
    assert [result[key] for key in ("positive_rows", "zero_rows", "negative_rows")] == [1, 2, 1]
    assert result["argmax_correct_rows"] == 2
    classes = result["per_class"]
    assert [row["class_index"] for row in classes] == [0, 1, 2, 3]
    assert classes[0] == dict(
        class_index=0,
        rows=2,
        mean_margin=pytest.approx(0.15),
        min_margin=0.0,
        max_margin=pytest.approx(0.3),
        positive_rows=1,
        zero_rows=1,
        negative_rows=0,
        argmax_correct_rows=2,
    )
    assert classes[1] == dict(
        class_index=1,
        rows=2,
        mean_margin=pytest.approx(-0.2),
        min_margin=pytest.approx(-0.4),
        max_margin=0.0,
        positive_rows=0,
        zero_rows=1,
        negative_rows=1,
        argmax_correct_rows=0,
    )
    for index in (2, 3):
        assert classes[index] == dict(
            class_index=index,
            rows=0,
            mean_margin=None,
            min_margin=None,
            max_margin=None,
            positive_rows=0,
            zero_rows=0,
            negative_rows=0,
            argmax_correct_rows=0,
        )


def test_should_use_full_two_class_probabilities_including_endpoints():
    predictions = np.array([[0.5, 0.5], [0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    targets = np.array([[1], [1], [0]], dtype=np.float64)
    result = probability_margins(predictions, targets, 2)
    assert result["target_margins"] == [0.0, 1.0, 1.0]
    assert result["mean_margin"] == pytest.approx(2 / 3)
    assert result["argmax_correct_rows"] == 2
    assert result["zero_rows"] == 1


def test_should_include_every_class_at_the_declared_upper_bound():
    predictions = np.full((16, 16), 1 / 16, dtype=np.float64)
    targets = np.arange(16, dtype=np.float64).reshape(16, 1)
    result = probability_margins(predictions, targets, 16)
    assert result["target_margins"] == [0.0] * 16
    assert result["zero_rows"] == 16 and result["argmax_correct_rows"] == 1
    assert [row["rows"] for row in result["per_class"]] == [1] * 16
    assert [row["argmax_correct_rows"] for row in result["per_class"]] == [1] + [0] * 15


def test_should_preserve_native_roundoff_without_normalizing_or_aliasing_inputs():
    predictions = np.array([[0.5000004, 0.5]], dtype=np.float64)
    targets = np.array([[1.0]], dtype=np.float64)
    original_predictions, original_targets = predictions.copy(), targets.copy()
    predictions.setflags(write=False)
    targets.setflags(write=False)
    result = probability_margins(predictions, targets, 2)
    assert result["target_margins"] == [0.5 - 0.5000004]
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    result["target_margins"][0] = 99.0
    result["per_class"][1]["mean_margin"] = 99.0
    np.testing.assert_array_equal(predictions, original_predictions)
    np.testing.assert_array_equal(targets, original_targets)
    assert probability_margins(predictions, targets, 2)["target_margins"] == [0.5 - 0.5000004]


@pytest.mark.parametrize("count", [True, False, 2.0, np.int64(2), 1, 17, None])
def test_should_refuse_non_native_or_out_of_range_class_counts(count):
    with pytest.raises(ValueError):
        probability_margins(np.array([[0.5, 0.5]]), np.array([[0.0]]), count)


@pytest.mark.parametrize(
    "predictions",
    [
        [[0.5, 0.5]],
        np.array([[0.5, 0.5]], dtype=np.float32),
        np.array([[0, 1]], dtype=np.int64),
        np.empty((0, 2), dtype=np.float64),
        np.array([0.5, 0.5]),
        np.array([[0.5]]),
        np.array([[0.3, 0.3, 0.4]]),
        np.array([[np.nan, 0.5]]),
        np.array([[np.inf, 0.5]]),
        np.array([[-0.1, 1.1]]),
        np.array([[0.4, 0.5]]),
    ],
)
def test_should_refuse_invalid_probabilities_without_repair(predictions):
    with pytest.raises(ValueError):
        probability_margins(predictions, np.array([[0.0]]), 2)


@pytest.mark.parametrize(
    "targets",
    [
        [[0.0]],
        np.array([[0.0]], dtype=np.float32),
        np.array([[0]], dtype=np.int64),
        np.array([0.0]),
        np.array([[0.0], [1.0]]),
        np.array([[0.5]]),
        np.array([[-1.0]]),
        np.array([[2.0]]),
        np.array([[np.inf]]),
        np.array([[np.nan]]),
    ],
)
def test_should_refuse_invalid_targets_without_coercion(targets):
    with pytest.raises(ValueError):
        probability_margins(np.array([[0.5, 0.5]]), targets, 2)


@pytest.mark.parametrize("which", ["predictions", "targets"])
def test_should_refuse_array_subclasses(which):
    class Subclass(np.ndarray):
        pass

    predictions = np.array([[0.5, 0.5]])
    targets = np.array([[0.0]])
    if which == "predictions":
        predictions = predictions.view(Subclass)
    else:
        targets = targets.view(Subclass)
    with pytest.raises(ValueError):
        probability_margins(predictions, targets, 2)
