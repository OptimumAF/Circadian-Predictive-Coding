"""Hand-computed four-phase train-fit gains/losses and strict score admission."""

from copy import deepcopy
import json
from typing import Any

import numpy as np
import pytest

from src.core.context_return_metrics import summarize_four_phase_return


def points():
    ce = {"A": [8, 2, 4, 6, 3], "B": [7, 6, 1, 3, 5], "C": [8, 7, 6, 2, 4]}
    accuracy = {
        "A": [0, 0.75, 0.5, 0.25, 0.625],
        "B": [0, 0.125, 0.875, 0.5, 0.25],
        "C": [0, 0.125, 0.25, 0.75, 0.5],
    }
    exact = {"A": [0, 0.5, 0, 0, 0.5], "B": [0, 0, 1, 0.5, 0], "C": [0, 0, 0, 1, 0.5]}
    return tuple(
        {
            "updates": update,
            **{
                context: {
                    "ce": ce[context][index],
                    "accuracy": accuracy[context][index],
                    "exact_sequence_accuracy": exact[context][index],
                }
                for context in ("A", "B", "C")
            },
        }
        for index, update in enumerate((0, 64, 128, 192, 256))
    )


def test_should_preserve_all_hand_calculated_ce_comparison_directions():
    summary = summarize_four_phase_return(points())
    assert summary["metrics"]["ce"] == {
        "initial_A_gain": 6,
        "new_B_gain": 5,
        "new_C_gain": 4,
        "old_A_loss_after_B": 2,
        "old_A_loss_before_return": 4,
        "old_B_loss_after_C": 2,
        "old_B_loss_after_return": 4,
        "old_C_loss_after_return": 2,
        "return_A_gain": 3,
        "return_gap": 1,
    }


def test_should_preserve_all_hand_calculated_accuracy_and_sequence_directions():
    summary = summarize_four_phase_return(points())
    assert summary["metrics"]["accuracy"] == {
        "initial_A_gain": 0.75,
        "new_B_gain": 0.75,
        "new_C_gain": 0.5,
        "old_A_loss_after_B": 0.25,
        "old_A_loss_before_return": 0.5,
        "old_B_loss_after_C": 0.375,
        "old_B_loss_after_return": 0.625,
        "old_C_loss_after_return": 0.25,
        "return_A_gain": 0.375,
        "return_gap": 0.125,
    }
    assert summary["metrics"]["exact_sequence_accuracy"] == {
        "initial_A_gain": 0.5,
        "new_B_gain": 1,
        "new_C_gain": 1,
        "old_A_loss_after_B": 0.5,
        "old_A_loss_before_return": 0.5,
        "old_B_loss_after_C": 0.5,
        "old_B_loss_after_return": 1,
        "old_C_loss_after_return": 0.5,
        "return_A_gain": 0.5,
        "return_gap": 0,
    }
    assert summary["incomplete_initial_A_fit"] is True
    assert summary["useful_skill_established"] is False


def test_should_retain_negative_acquisition_return_and_loss_results():
    original = points()
    reverse = tuple(
        {"updates": point["updates"], **{context: other[context] for context in ("A", "B", "C")}}
        for point, other in zip(original, reversed(original), strict=True)
    )
    summary = summarize_four_phase_return(reverse)
    assert summary["metrics"]["ce"] == {
        "initial_A_gain": -3,
        "new_B_gain": 2,
        "new_C_gain": -1,
        "old_A_loss_after_B": -2,
        "old_A_loss_before_return": -4,
        "old_B_loss_after_C": 5,
        "old_B_loss_after_return": 6,
        "old_C_loss_after_return": 1,
        "return_A_gain": -6,
        "return_gap": 2,
    }
    assert summary["metrics"]["accuracy"]["initial_A_gain"] == -0.375
    assert summary["metrics"]["accuracy"]["return_A_gain"] == -0.75
    assert summary["metrics"]["exact_sequence_accuracy"]["return_A_gain"] == -0.5


def test_should_keep_detached_json_endpoints_and_incomplete_fit_caveats():
    source = points()
    before = deepcopy(source)
    summary = summarize_four_phase_return(source)
    assert source == before
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary
    summary["endpoint_values"][0]["C"]["ce"] = 99
    source[1]["B"]["ce"] = 88
    assert source[0]["C"]["ce"] == 8
    assert summary["endpoint_values"][1]["B"]["ce"] == 6
    assert (
        summary["interpretation"]
        == "incomplete_initial_acquisition_and_four_phase_return_refitting"
    )
    complete = points()
    complete[1]["A"]["exact_sequence_accuracy"] = 1
    result = summarize_four_phase_return(complete)
    assert result["incomplete_initial_A_fit"] is False
    assert result["useful_skill_established"] is False
    assert (
        result["interpretation"] == "train_only_four_phase_return_refitting_not_skill_restoration"
    )


def test_should_preserve_zero_differences_and_negative_return_gap_without_thresholding():
    source = points()
    for point in source:
        for context in ("A", "B", "C"):
            point[context] = {"ce": 1, "accuracy": 0.5, "exact_sequence_accuracy": 0}
    result = summarize_four_phase_return(source)
    assert all(value == 0 for metric in result["metrics"].values() for value in metric.values())
    source[-1]["A"] = {"ce": 0, "accuracy": 1, "exact_sequence_accuracy": 1}
    result = summarize_four_phase_return(source)
    assert result["metrics"]["ce"]["return_gap"] == -1
    assert result["metrics"]["accuracy"]["return_gap"] == -0.5
    assert result["metrics"]["exact_sequence_accuracy"]["return_gap"] == -1


class TupleSubclass(tuple):
    pass


class DictSubclass(dict):
    pass


@pytest.mark.parametrize(
    "kind",
    [
        "list",
        "tuple_subclass",
        "short",
        "long",
        "row_subclass",
        "score_subclass",
        "missing_C",
        "wrong_C_alias",
        "extra_row",
        "extra_metric",
        "missing_metric",
    ],
)
def test_should_require_exact_five_point_three_context_score_schema(kind):
    source: Any = points()
    if kind == "list":
        source = list(source)
    elif kind == "tuple_subclass":
        source = TupleSubclass(source)
    elif kind == "short":
        source = source[:-1]
    elif kind == "long":
        source = (*source, source[-1])
    elif kind == "row_subclass":
        source = (DictSubclass(source[0]), *source[1:])
    elif kind == "score_subclass":
        source[0]["C"] = DictSubclass(source[0]["C"])
    elif kind == "missing_C":
        source[0].pop("C")
    elif kind == "wrong_C_alias":
        source[0]["C_train"] = source[0].pop("C")
    elif kind == "extra_row":
        source[0]["picked"] = True
    elif kind == "extra_metric":
        source[0]["C"]["return"] = 1
    else:
        source[0]["C"].pop("ce")
    with pytest.raises(ValueError):
        summarize_four_phase_return(source)


@pytest.mark.parametrize("value", [True, 256.0, np.int64(256), 255, -1, None, "256"])
def test_should_require_original_native_integer_endpoint_counts(value):
    source: Any = points()
    source[-1]["updates"] = value
    with pytest.raises(ValueError):
        summarize_four_phase_return(source)


@pytest.mark.parametrize("metric", ["ce", "accuracy", "exact_sequence_accuracy"])
@pytest.mark.parametrize(
    "value",
    [True, np.float64(0.5), "0.5", None, float("nan"), float("inf"), -float("inf"), -0.1, 10**400],
)
def test_should_preserve_strict_native_finite_metric_admission(metric, value):
    source: Any = points()
    source[3]["C"][metric] = value
    with pytest.raises(ValueError):
        summarize_four_phase_return(source)
    assert source[3]["C"][metric] is value


@pytest.mark.parametrize("metric", ["accuracy", "exact_sequence_accuracy"])
def test_should_refuse_accuracy_above_one(metric):
    source: Any = points()
    source[3]["B"][metric] = 1.01
    with pytest.raises(ValueError):
        summarize_four_phase_return(source)
