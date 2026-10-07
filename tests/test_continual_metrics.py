"""Primary two-task outcomes keep negative transfer and zero bases explicit."""

from __future__ import annotations

import math

import pytest

from src.core.continual_metrics import TwoTaskAccuracy


def test_should_report_final_mean_and_signed_forgetting_from_explicit_tasks() -> None:
    observed = TwoTaskAccuracy(0.75, 0.875, 0.625)

    assert observed.final_mean_task_accuracy == 0.75
    assert observed.signed_forgetting_a == -0.125
    assert observed.retention_ratio_a == pytest.approx(0.875 / 0.75)


def test_should_leave_zero_base_retention_ratio_undefined() -> None:
    observed = TwoTaskAccuracy(0.0, 0.25, 0.5)

    assert observed.signed_forgetting_a == -0.25
    assert observed.final_mean_task_accuracy == 0.375
    assert observed.retention_ratio_a is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.01, 1.01, True])
def test_should_reject_invalid_task_accuracy(bad: float) -> None:
    with pytest.raises(ValueError, match="finite accuracy"):
        TwoTaskAccuracy(0.5, bad, 0.5)


def test_should_not_collapse_weaker_initial_learning_into_retention_gain() -> None:
    neutral = TwoTaskAccuracy(0.75, 0.875, 0.625)
    gated = TwoTaskAccuracy(0.625, 0.875, 0.625)

    assert gated.final_mean_task_accuracy == neutral.final_mean_task_accuracy
    assert gated.signed_forgetting_a < neutral.signed_forgetting_a
    assert gated.a_after_a < neutral.a_after_a
    assert math.isfinite(gated.signed_forgetting_a)
