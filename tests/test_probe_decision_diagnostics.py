"""Decision and class details preserve metric conventions and bounded ownership."""

from math import log, log1p
from typing import Any

import numpy as np
import pytest

from src.core.binary_probability_scores import probability_scores as binary_scores
from src.core.categorical_probability_scores import probability_scores as categorical_scores
from src.core.probe_decision_diagnostics import decision_class_diagnostics


def arrays(probabilities=((0.8, 0.1, 0.1), (0.2, 0.3, 0.5)), labels=(0.0, 2.0)):
    return np.array(probabilities, dtype=np.float64), np.array(labels, dtype=np.float64).reshape(
        -1, 1
    )


def test_should_report_balanced_binary_decisions_confusion_and_exact_schema():
    predictions, targets = arrays(((0.2,), (0.6,), (0.8,), (0.1,)), (0, 0, 1, 1))
    result = decision_class_diagnostics(predictions, targets, 2)
    assert set(result) == {
        "num_classes",
        "rows",
        "loss_definition",
        "target_class_rule",
        "class_counts",
        "predicted_class_counts",
        "confusion",
        "per_class",
        "decisions",
    }
    assert result["class_counts"] == result["predicted_class_counts"] == [2, 2]
    assert result["confusion"] == [[1, 1], [1, 1]]
    assert result["rows"] == 4 and result["num_classes"] == 2
    assert result["loss_definition"] == "binary_cross_entropy"
    assert result["target_class_rule"] == "threshold_ge_half"
    assert result["per_class"][0] == {
        "class_index": 0,
        "support": 2,
        "predicted_count": 2,
        "correct": 1,
        "accuracy": 0.5,
        "mean_loss": pytest.approx((-log(0.8) - log(0.4)) / 2),
    }
    assert result["decisions"][1] == {
        "index": 1,
        "target": 0.0,
        "target_class": 0,
        "predicted_class": 1,
        "loss": pytest.approx(-log(0.4)),
        "correct": False,
    }


def test_should_keep_soft_binary_labels_and_inclusive_half_ties():
    predictions, targets = arrays(((0.5,), (0.49,), (0.5,), (0.9,)), (0.5, 0.49, 0.2, 0.7))
    result = decision_class_diagnostics(predictions, targets, 2)
    assert [row["target_class"] for row in result["decisions"]] == [1, 0, 0, 1]
    assert [row["predicted_class"] for row in result["decisions"]] == [1, 0, 1, 1]
    assert [row["target"] for row in result["decisions"]] == [0.5, 0.49, 0.2, 0.7]
    assert result["decisions"][-1]["loss"] == pytest.approx(-0.7 * log(0.9) - 0.3 * log1p(-0.9))
    assert (
        np.mean([row["loss"] for row in result["decisions"]])
        == binary_scores(predictions, targets)["bce"]
    )


def test_should_reveal_imbalanced_categorical_action_collapse_and_absent_support():
    predictions, targets = arrays(((0.1, 0.8, 0.1),) * 4, (0, 0, 0, 1))
    result = decision_class_diagnostics(predictions, targets, 3)
    assert result["class_counts"] == [3, 1, 0]
    assert result["predicted_class_counts"] == [0, 4, 0]
    assert result["confusion"] == [[0, 3, 0], [0, 1, 0], [0, 0, 0]]
    assert result["per_class"][0]["accuracy"] == 0.0
    assert result["per_class"][1]["accuracy"] == 1.0
    assert result["per_class"][2] == {
        "class_index": 2,
        "support": 0,
        "predicted_count": 0,
        "correct": 0,
        "accuracy": None,
        "mean_loss": None,
    }
    assert result["loss_definition"] == "categorical_cross_entropy"
    assert result["target_class_rule"] == "hard_class_index"
    assert (
        np.mean([row["loss"] for row in result["decisions"]])
        == categorical_scores(predictions, targets, 3)["ce"]
    )


def test_should_use_first_argmax_tie_and_count_predictions_of_absent_classes():
    predictions, targets = arrays(((0.4, 0.4, 0.2), (0.0, 0.0, 1.0)), (1, 1))
    result = decision_class_diagnostics(predictions, targets, 3)
    assert result["predicted_class_counts"] == [1, 0, 1]
    assert result["class_counts"] == [0, 2, 0]
    assert result["per_class"][0]["accuracy"] is None
    assert result["per_class"][2]["mean_loss"] is None
    assert [row["predicted_class"] for row in result["decisions"]] == [0, 2]


@pytest.mark.parametrize("classes", [2, 3])
def test_should_preserve_existing_probability_endpoint_clipping(classes):
    predictions, targets = (
        arrays(((0.0,), (1.0,)), (1, 0)) if classes == 2 else arrays(((0, 1, 0), (0, 0, 1)), (0, 2))
    )
    result = decision_class_diagnostics(predictions, targets, classes)
    expected = (
        binary_scores(predictions, targets)["bce"]
        if classes == 2
        else categorical_scores(predictions, targets, 3)["ce"]
    )
    assert np.mean([row["loss"] for row in result["decisions"]]) == expected
    assert result["decisions"][0]["loss"] == -log(1e-12)


@pytest.mark.parametrize("classes", [2, 16])
def test_should_accept_maximum_rows_and_class_count(classes):
    predictions = np.full(
        (128, 1 if classes == 2 else classes), 0.5 if classes == 2 else 1 / classes
    )
    targets = np.zeros((128, 1))
    result = decision_class_diagnostics(predictions, targets, classes)
    assert len(result["decisions"]) == 128 and len(result["confusion"]) == classes


def test_should_return_only_detached_python_primitives_and_preserve_inputs():
    predictions, targets = arrays()
    before_predictions, before_targets = predictions.copy(), targets.copy()
    result = decision_class_diagnostics(predictions, targets, 3)

    def check(value: Any):
        assert type(value) in (dict, list, int, float, str, bool, type(None))
        if type(value) is dict:
            for key, item in value.items():
                assert type(key) is str
                check(item)
        elif type(value) is list:
            for item in value:
                check(item)

    check(result)
    np.testing.assert_array_equal(predictions, before_predictions)
    np.testing.assert_array_equal(targets, before_targets)
    predictions[:] = 0
    targets[:] = 1
    assert result["decisions"][0]["target"] == 0.0
    assert result["decisions"][0]["predicted_class"] == 0
    result["confusion"][0][0] = 99
    fresh = decision_class_diagnostics(before_predictions, before_targets, 3)
    assert fresh["confusion"][0][0] == 1


@pytest.mark.parametrize("classes", [True, np.int64(3), 1, 17, 3.0, "3"])
def test_should_refuse_invalid_class_counts(classes):
    predictions, targets = arrays()
    with pytest.raises(ValueError):
        decision_class_diagnostics(predictions, targets, classes)


@pytest.mark.parametrize(
    "predictions,targets,classes",
    [
        (np.zeros((0, 1)), np.zeros((0, 1)), 2),
        (np.zeros((129, 1)), np.zeros((129, 1)), 2),
        (np.zeros((129, 3)), np.zeros((129, 1)), 3),
        (np.zeros((1, 2)), np.zeros((1, 1)), 2),
        (np.zeros((1, 1)), np.zeros((1, 1)), 3),
        (np.zeros((1, 3)), np.zeros((2, 1)), 3),
        (np.zeros((1, 3)), np.zeros((1,)), 3),
        (np.zeros((1, 3), dtype=np.float32), np.zeros((1, 1)), 3),
        (np.zeros((1, 3)), np.zeros((1, 1), dtype=np.float32), 3),
        ([[0.5]], np.zeros((1, 1)), 2),
        (np.zeros((1, 1)), [[0.0]], 2),
    ],
)
def test_should_refuse_invalid_envelopes_before_value_array_checks(predictions, targets, classes):
    with pytest.raises(ValueError):
        decision_class_diagnostics(predictions, targets, classes)


def test_should_refuse_oversized_envelope_before_numpy_value_allocations(monkeypatch):
    predictions, targets = np.zeros((129, 3)), np.zeros((129, 1))

    def forbidden(*args, **kwargs):
        raise AssertionError("value check allocation reached before envelope refusal")

    monkeypatch.setattr(np, "isfinite", forbidden)
    with pytest.raises(ValueError, match="B1..128"):
        decision_class_diagnostics(predictions, targets, 3)


@pytest.mark.parametrize(
    "predictions,targets,classes",
    [
        (((float("nan"),),), (0,), 2),
        (((float("inf"),),), (0,), 2),
        (((-0.1,),), (0,), 2),
        (((1.1,),), (0,), 2),
        (((0.5,),), (float("nan"),), 2),
        (((0.5,),), (-0.1,), 2),
        (((0.5,),), (1.1,), 2),
        (((0.2, 0.3, 0.4),), (0,), 3),
        (((-0.1, 0.6, 0.5),), (0,), 3),
        (((0.2, 0.3, float("inf")),), (0,), 3),
        (((0.2, 0.3, 0.5),), (0.5,), 3),
        (((0.2, 0.3, 0.5),), (3,), 3),
        (((0.2, 0.3, 0.5),), (-1,), 3),
    ],
)
def test_should_refuse_invalid_values_without_probability_repair(predictions, targets, classes):
    values, labels = arrays(predictions, targets)
    with pytest.raises(ValueError):
        decision_class_diagnostics(values, labels, classes)
