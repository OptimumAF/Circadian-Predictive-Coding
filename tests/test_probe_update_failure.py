"""Native paid work survives a failed probe without inventing app charges."""

from typing import Any

import numpy as np
import pytest

from src.core.learner_ports import TrainingUpdateFailed
from test_fresh_task_probe import FakeLearner, run, session


def failure():
    return TrainingUpdateFailed("paid step refused", "test_native", {"optimizer_steps": 1})


def test_should_own_failure_work_and_keep_runtime_error_compatibility():
    work = {"optimizer_steps": 1, "uncertain": True}
    error = TrainingUpdateFailed("paid", "test_native", work)
    work["optimizer_steps"] = 3
    returned = error.work
    returned["optimizer_steps"] = 9
    assert error.work == {"optimizer_steps": 1, "uncertain": True}
    assert isinstance(error, RuntimeError)
    with pytest.raises(AttributeError):
        error.definition = "replacement"  # type: ignore[misc]


@pytest.mark.parametrize(
    "work",
    [
        None,
        [],
        {"": 0},
        {" x": 0},
        {"x" * 65: 0},
        {"x": -1},
        {"x": 2**53},
        {"x": 1.0},
        {"x": np.int64(1)},
        {"x": object()},
        {str(i): 0 for i in range(33)},
    ],
)
def test_should_refuse_unbounded_or_foreign_failure_counters(work: Any):
    with pytest.raises(ValueError):
        TrainingUpdateFailed("paid", "test_native", work)


@pytest.mark.parametrize("reason,definition", [("", "x"), ("x", ""), (None, "x"), ("x", 1)])
def test_should_require_native_nonempty_failure_identity(reason, definition):
    with pytest.raises(ValueError):
        TrainingUpdateFailed(reason, definition, {})


def test_should_preserve_paid_failure_without_claiming_an_app_update():
    def stop(*args):
        raise failure()

    source, budget = FakeLearner(hooks={"train": stop}), session()
    report = run(source, budget=budget)
    assert report["status"] == "failed"
    assert report["attempted_updates"] == 1
    assert report["completed_updates"] == budget.updates_completed == 0
    assert report["completed_training_examples"] == 0
    assert [point["updates"] for point in report["checkpoints"]] == [0]
    assert report["failed_native_work"] == {
        "definition": "test_native",
        "work": {"optimizer_steps": 1},
    }
    assert source.value[0] == 99.0
    assert source.child.value[0] == 1.0  # Unknown paid work is not repaired.
    assert "_failed_native_work" not in report


@pytest.mark.parametrize("boundary", ["fork", "restore", "predict", "timer", "guard"])
def test_should_not_report_callback_failure_as_native_training_work(boundary):
    def stop(*args):
        raise failure()

    hooks = {boundary: stop} if boundary in {"fork", "restore", "predict"} else {}
    kwargs: dict[str, Any] = {}
    if boundary == "timer":
        kwargs["timer"] = stop
    elif boundary == "guard":
        kwargs["check_boundary"] = stop
    report = run(FakeLearner(hooks=hooks), **kwargs)
    assert report["status"] == "failed"
    assert report["attempted_updates"] == report["completed_updates"] == 0
    assert "failed_native_work" not in report
    assert "_failed_native_work" not in report


def test_should_leave_success_schema_unchanged():
    report = run(FakeLearner())
    assert report["status"] == "complete"
    assert "failed_native_work" not in report and "_failed_native_work" not in report
