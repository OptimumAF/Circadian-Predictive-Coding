"""Observed recovery semantics and strictly prearrival phase label costs."""

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from src.core.sensor_recovery import RecoverySummary, summarize_phase_recovery


def _summarize(accuracies, **overrides: Any):
    options: dict[str, Any] = {"start_tick": 4, "label_delay": 2, "rows_per_batch": 4}
    options.update(overrides)
    return summarize_phase_recovery(accuracies, **options)


def test_should_report_no_deficit_without_inventing_zero_label_recovery():
    result = _summarize((1.0, 0.75, 1.0, 0.75))

    assert result == RecoverySummary("no_deficit", None, None, None, 4)
    with pytest.raises(FrozenInstanceError):
        setattr(result, "status", "recovered")


def test_should_censor_a_deficit_without_enough_following_adequate_predictions():
    result = _summarize((1.0, 0.5, 0.25, 0.75))

    assert result == RecoverySummary("censored", 5, None, None, 4)


def test_should_confirm_at_the_second_adequate_prediction_after_the_first_deficit():
    result = _summarize((1.0, 0.5, 0.75, 1.0, 1.0))

    assert result == RecoverySummary("recovered", 5, 7, 4, 8)


def test_should_reset_the_adequate_streak_at_each_new_deficit():
    result = _summarize((0.5, 0.75, 0.5, 0.75, 0.75))

    assert result == RecoverySummary("recovered", 4, 8, 8, 8)


def test_should_keep_the_first_recovery_when_a_later_deficit_occurs():
    result = _summarize((0.5, 1.0, 1.0, 0.0, 1.0, 1.0))

    assert result == RecoverySummary("recovered", 4, 6, 0, 12)


@pytest.mark.parametrize("delay,labels,last_labels", [(0, 8, 12), (2, 0, 4), (512, 0, 0)])
def test_should_exclude_same_tick_and_future_label_arrivals(delay, labels, last_labels):
    result = _summarize((0.5, 0.75, 0.75, 1.0), label_delay=delay)

    assert result.confirmation_tick == 6
    assert result.labels_to_recovery == labels
    assert result.labels_available_before_last_prediction == last_labels


def test_should_count_phase_labels_before_a_late_deficit_without_counting_earlier_phases():
    result = _summarize((1.0, 1.0, 0.5, 1.0, 1.0), start_tick=100, label_delay=0)

    assert result == RecoverySummary("recovered", 102, 104, 16, 16)


def test_should_allow_one_batch_confirmation_but_never_count_the_deficit_itself():
    assert _summarize((0.0,), recovery_batches=1).status == "censored"
    assert _summarize((0.0, 0.75), recovery_batches=1) == RecoverySummary("recovered", 4, 5, 0, 0)


def test_should_accept_native_integer_accuracies_and_the_last_permitted_tick():
    assert _summarize((1,), start_tick=511).status == "no_deficit"
    assert _summarize((0,) * 512, start_tick=0).first_deficit_tick == 0


@pytest.mark.parametrize(
    "accuracies",
    [
        [],
        [0.5],
        (),
        (0.5,) * 513,
        (True,),
        (float("nan"),),
        (float("inf"),),
        (-0.01,),
        (1.01,),
        ("1",),
    ],
)
def test_should_refuse_invalid_accuracy_sequences(accuracies):
    with pytest.raises(ValueError, match="accuracies"):
        _summarize(accuracies)


@pytest.mark.parametrize(
    "name,value",
    [
        ("start_tick", -1),
        ("start_tick", 512),
        ("start_tick", True),
        ("start_tick", 1.0),
        ("label_delay", -1),
        ("label_delay", 513),
        ("label_delay", False),
        ("rows_per_batch", 0),
        ("rows_per_batch", 129),
        ("rows_per_batch", 4.0),
        ("recovery_batches", 0),
        ("recovery_batches", 33),
        ("recovery_batches", True),
        ("accuracy_threshold", 0),
        ("accuracy_threshold", 1.01),
        ("accuracy_threshold", float("nan")),
        ("accuracy_threshold", float("inf")),
        ("accuracy_threshold", True),
    ],
)
def test_should_refuse_invalid_timing_and_recovery_configuration(name, value):
    with pytest.raises(ValueError, match=name):
        _summarize((1.0,), **{name: value})


def test_should_refuse_a_phase_extending_beyond_tick_511():
    with pytest.raises(ValueError, match="phase"):
        _summarize((1.0, 1.0), start_tick=511)


def test_should_validate_the_entire_sequence_before_returning_an_early_recovery():
    with pytest.raises(ValueError, match="accuracies"):
        _summarize((0.0, 1.0, 1.0, float("nan")))
