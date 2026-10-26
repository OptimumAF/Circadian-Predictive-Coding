"""Hand-calculated fixed-protocol gains, negatives and malformed score refusal."""

from copy import deepcopy
import json
from typing import Any

import numpy as np
import pytest

from src.core.context_return_metrics import summarize_context_return


def points():
    a = [(4, 0.125, 0), (1, 0.75, 0.5), (3, 0.25, 0), (1.5, 0.625, 0.5)]
    b = [(5, 0.25, 0), (4, 0.125, 0), (1, 0.875, 1), (2, 0.5, 0.5)]
    metrics = ("ce", "accuracy", "exact_sequence_accuracy")
    return tuple(
        {"updates": count, "A": dict(zip(metrics, av)), "B": dict(zip(metrics, bv))}
        for count, av, bv in zip((0, 64, 128, 192), a, b, strict=True)
    )


def test_should_preserve_hand_calculated_ce_and_accuracy_directions():
    summary = summarize_context_return(points())
    assert summary["metrics"] == {
        "ce": {
            "initial_A_gain": 3,
            "new_B_gain": 3,
            "old_A_loss": 2,
            "return_A_gain": 1.5,
            "return_gap": 0.5,
        },
        "accuracy": {
            "initial_A_gain": 0.625,
            "new_B_gain": 0.75,
            "old_A_loss": 0.5,
            "return_A_gain": 0.375,
            "return_gap": 0.125,
        },
        "exact_sequence_accuracy": {
            "initial_A_gain": 0.5,
            "new_B_gain": 1,
            "old_A_loss": 0.5,
            "return_A_gain": 0.5,
            "return_gap": 0,
        },
    }
    assert summary["incomplete_initial_A_fit"] is True
    assert summary["useful_skill_established"] is False
    assert summary["interpretation"] == "incomplete_initial_acquisition_and_return_refitting"


def test_should_retain_negative_gains_and_better_than_initial_fit_return():
    original = points()
    reversed_scores = tuple(
        {"updates": row["updates"], "A": reverse["A"], "B": reverse["B"]}
        for row, reverse in zip(original, reversed(original), strict=True)
    )
    result = summarize_context_return(reversed_scores)
    assert result["metrics"]["ce"] == {
        "initial_A_gain": -1.5,
        "new_B_gain": -3,
        "old_A_loss": -2,
        "return_A_gain": -3,
        "return_gap": 1,
    }
    assert result["metrics"]["accuracy"]["initial_A_gain"] == -0.375
    assert result["metrics"]["accuracy"]["new_B_gain"] == -0.75
    assert result["metrics"]["exact_sequence_accuracy"]["new_B_gain"] == -1
    improved = points()
    improved[3]["A"] = {"ce": 0.5, "accuracy": 1, "exact_sequence_accuracy": 1}
    result = summarize_context_return(improved)
    assert result["metrics"]["ce"]["return_gap"] == -0.5
    assert result["metrics"]["accuracy"]["return_gap"] == -0.25
    assert result["metrics"]["exact_sequence_accuracy"]["return_gap"] == -0.5


def test_should_return_detached_json_values_without_changing_inputs_or_claiming_skill():
    source = points()
    before = deepcopy(source)
    summary = summarize_context_return(source)
    assert source == before
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary
    summary["endpoint_values"][0]["A"]["ce"] = 99
    assert source == before
    source[1]["A"]["ce"] = 88
    assert summary["endpoint_values"][1]["A"]["ce"] == 1
    complete = points()
    complete[1]["A"]["exact_sequence_accuracy"] = 1
    result = summarize_context_return(complete)
    assert result["incomplete_initial_A_fit"] is False
    assert result["useful_skill_established"] is False
    assert result["interpretation"] == "train_only_context_return_refitting_not_skill_restoration"


def test_should_preserve_all_zero_differences_and_boundary_scores():
    source = tuple(
        {
            "updates": update,
            "A": {"ce": 0, "accuracy": 1, "exact_sequence_accuracy": 0},
            "B": {"ce": 0, "accuracy": 1, "exact_sequence_accuracy": 0},
        }
        for update in (0, 64, 128, 192)
    )
    result = summarize_context_return(source)
    assert all(value == 0 for metric in result["metrics"].values() for value in metric.values())


class TupleSubclass(tuple):
    pass


class DictSubclass(dict):
    pass


@pytest.mark.parametrize("kind", ["list", "tuple_subclass", "short", "long", "row_subclass"])
def test_should_require_exact_four_endpoint_container_and_rows(kind):
    source: Any = points()
    if kind == "list":
        source = list(source)
    elif kind == "tuple_subclass":
        source = TupleSubclass(source)
    elif kind == "short":
        source = source[:3]
    elif kind == "long":
        source = (*source, source[-1])
    else:
        source = (DictSubclass(source[0]), *source[1:])
    with pytest.raises(ValueError):
        summarize_context_return(source)


@pytest.mark.parametrize("update", [True, 0.0, np.int64(0), -1, 64, None, "0"])
def test_should_refuse_wrong_type_or_wrong_fixed_update_count(update):
    source: Any = points()
    source[0]["updates"] = update
    with pytest.raises(ValueError, match="exactly 0, 64, 128 and 192"):
        summarize_context_return(source)


@pytest.mark.parametrize(
    "kind",
    ["row_missing", "row_extra", "score_missing", "score_extra", "score_subclass", "score_none"],
)
def test_should_require_exact_score_fieldsets(kind):
    source: Any = points()
    if kind == "row_missing":
        source[0].pop("B")
    elif kind == "row_extra":
        source[0]["selected"] = True
    elif kind == "score_missing":
        source[0]["A"].pop("ce")
    elif kind == "score_extra":
        source[0]["A"]["reward"] = 0
    elif kind == "score_subclass":
        source[0]["A"] = DictSubclass(source[0]["A"])
    else:
        source[0]["A"] = None
    with pytest.raises(ValueError):
        summarize_context_return(source)


@pytest.mark.parametrize("metric", ["ce", "accuracy", "exact_sequence_accuracy"])
@pytest.mark.parametrize(
    "value",
    [True, np.float64(0.5), "0.5", None, float("nan"), float("inf"), -float("inf"), -0.1, 10**400],
)
def test_should_refuse_non_native_nonfinite_negative_or_unrepresentable_scores(metric, value):
    source: Any = points()
    source[2]["B"][metric] = value
    before = deepcopy(source)
    with pytest.raises(ValueError):
        summarize_context_return(source)
    # Failed validation does not partially normalize or rewrite caller data.
    assert source[2]["B"][metric] is value
    assert source[0] == before[0]


@pytest.mark.parametrize("metric", ["accuracy", "exact_sequence_accuracy"])
def test_should_refuse_accuracy_above_one(metric):
    source: Any = points()
    source[3]["A"][metric] = 1.001
    with pytest.raises(ValueError):
        summarize_context_return(source)
