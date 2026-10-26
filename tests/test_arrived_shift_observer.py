"""Arrived-only, retrospective signal observations using ordinary fake predictors."""

from dataclasses import replace
import json
from typing import Any

import numpy as np
import pytest

from src.app import arrived_shift_observer as module
from src.app.arrived_shift_observer import ArrivedShiftObserver
from src.app.coupled_attribution import AttributionBatch
from src.core.shift_signals import ShiftDetector, ShiftPolicy


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setattr(module, "monotonic", lambda: 0.0)


def batch(index=1, *, role="train"):
    return AttributionBatch(
        role,
        "C",
        (f"row-{index}",),
        np.array([[0.2, -0.1]], dtype=np.float64),
        np.array([[0.25]], dtype=np.float64),
    )


class PoisonState:
    def __init__(self):
        self.copies = 0

    def __deepcopy__(self, memo):
        self.copies += 1
        pytest.fail("refused input reached state copying")


class BrokenState:
    def __init__(self, error):
        self.error = error

    def __deepcopy__(self, memo):
        raise self.error


def recorder():
    calls: list[tuple[Any, Any]] = []

    def predict(previous, features):
        calls.append((previous, features.copy()))
        return np.full((len(features), 1), 0.5, dtype=np.float64)

    return calls, predict


@pytest.mark.parametrize("fault", ["callback", "zero", "boolean", "cap", "policy", "tampered"])
def test_should_refuse_invalid_constructor_controls_without_copying_or_predicting(fault):
    _, predict = recorder()
    options: dict[str, Any] = {}
    callback: Any = predict
    if fault == "callback":
        callback = object()
    elif fault == "policy":
        options["policy"] = object()
    elif fault == "tampered":
        policy = ShiftPolicy()
        object.__setattr__(policy, "feature_threshold", 0)
        options["policy"] = policy
    else:
        options["max_updates"] = {"zero": 0, "boolean": True, "cap": 32}[fault]
    with pytest.raises(ValueError):
        ArrivedShiftObserver(callback, **options)


def test_should_prime_without_prediction_and_retain_all_planned_not_run_slots():
    calls, predict = recorder()
    observer = ArrivedShiftObserver(predict)
    before = observer.report()
    assert set(before) == {
        "primed",
        "stopped",
        "reason",
        "initial_seconds",
        "total_seconds",
        "prediction_calls_attempted",
        "prediction_calls_completed",
        "rows",
    }
    assert before["primed"] is False and before["initial_seconds"] is None
    observer(0, {"version": 0}, None)
    report = observer.report()
    assert report["primed"] is True and report["stopped"] is False
    assert report["reason"] is None and report["initial_seconds"] == 0.0
    assert report["prediction_calls_attempted"] == report["prediction_calls_completed"] == 0
    assert report["total_seconds"] == 0.0 and calls == []
    assert len(report["rows"]) == 31
    for number, row in enumerate(report["rows"], 1):
        assert row == {
            "updates": number,
            "prediction_state_updates": number - 1,
            "status": "not_run",
            "reason": "observation not reached",
            "metrics": None,
            "decision": None,
            "elapsed_seconds": None,
        }
    json.dumps(report, allow_nan=False)


def test_should_score_previous_parameters_and_isolate_delegate_features_and_next_state():
    states: list[dict[str, Any]] = [
        {"version": number, "weights": np.array([float(number)])} for number in range(3)
    ]
    calls: list[tuple[int, float, np.ndarray[Any, Any]]] = []

    def predict(previous, features):
        calls.append((previous["version"], float(previous["weights"][0]), features.copy()))
        previous["version"] = 999
        previous["weights"].fill(999)
        features.fill(999)
        states[len(calls)]["version"] = 777
        states[len(calls)]["weights"].fill(777)
        return np.array([[0.5]])

    observer = ArrivedShiftObserver(predict, max_updates=2)
    observer(0, states[0], None)
    states[0]["version"] = 444
    states[0]["weights"].fill(444)
    first, second = batch(1), batch(2)
    original_inputs = (first.features.copy(), second.features.copy())
    observer(1, states[1], first)
    observer(2, states[2], second)
    assert [(version, weight) for version, weight, _ in calls] == [(0, 0.0), (1, 1.0)]
    for (_, _, received), original in zip(calls, original_inputs):
        np.testing.assert_array_equal(received, original)
    np.testing.assert_array_equal(first.features, original_inputs[0])
    np.testing.assert_array_equal(second.features, original_inputs[1])
    for number, row in enumerate(observer.report()["rows"]):
        assert row["prediction_state_updates"] == number
        assert row["status"] == "complete" and row["reason"] is None
        assert row["metrics"] == {
            "rows": 1,
            "uncertainty": 1.0,
            "feature_mean": (0.2, -0.1),
            "residual": 0.25,
        }
        assert row["decision"]["batch_index"] == number + 1
    assert observer.report()["prediction_calls_completed"] == 2


@pytest.mark.parametrize(
    "fault", ["unprimed", "zero_with_batch", "boolean", "skip", "duplicate", "beyond"]
)
def test_should_refuse_wrong_update_sequences_before_copy_or_prediction_and_preserve_report(fault):
    calls, predict = recorder()
    observer = ArrivedShiftObserver(predict, max_updates=1 if fault == "beyond" else 3)
    if fault not in ("unprimed", "zero_with_batch"):
        observer(0, {}, None)
    if fault in ("duplicate", "beyond"):
        observer(1, {}, batch(1))
    count = {
        "unprimed": 1,
        "zero_with_batch": 0,
        "boolean": True,
        "skip": 2,
        "duplicate": 1,
        "beyond": 2,
    }[fault]
    before, call_count = observer.report(), len(calls)
    poison = PoisonState()
    with pytest.raises(ValueError):
        observer(count, poison, batch(2))
    assert poison.copies == 0 and len(calls) == call_count
    assert observer.report() == before


@pytest.mark.parametrize("role", ["final_test", "probe_test", "inner_guard"])
def test_should_refuse_nontraining_roles_before_payload_access_or_state_copy(role):
    calls, predict = recorder()
    observer = ArrivedShiftObserver(predict, max_updates=2)
    observer(0, {}, None)
    poison = PoisonState()
    payload: Any = poison
    arrived = AttributionBatch(role, "C", ("heldout",), payload, payload)
    before = observer.report()
    with pytest.raises(ValueError, match="train"):
        observer(1, poison, arrived)
    assert calls == [] and poison.copies == 0 and observer.report() == before


@pytest.mark.parametrize(
    "fault",
    [
        "record",
        "dtype",
        "rank",
        "rows",
        "width",
        "nan",
        "row_ids",
        "duplicate_id",
        "changed_width",
    ],
)
def test_should_reject_invalid_arrived_batches_atomically_before_state_copy(fault, monkeypatch):
    calls, predict = recorder()
    observer = ArrivedShiftObserver(predict, max_updates=3)
    observer(0, {}, None)
    number = 1
    if fault in ("duplicate_id", "changed_width"):
        observer(1, {}, batch(1))
        number = 2
    arrived: Any = batch(number)
    if fault == "record":
        arrived = object()
    elif fault == "dtype":
        arrived = replace(arrived, features=arrived.features.astype(np.float32))
    elif fault == "rank":
        arrived = replace(arrived, features=np.array([0.0, 0.0]))
    elif fault == "rows":
        arrived = replace(
            arrived,
            features=np.zeros((129, 2)),
            targets=np.zeros((129, 1)),
            row_ids=tuple(f"r-{index}" for index in range(129)),
        )
        monkeypatch.setattr(
            module,
            "_validate_batch",
            lambda *args: pytest.fail("oversize reached value validation"),
        )
    elif fault == "width":
        arrived = replace(arrived, features=np.zeros((1, 17)))
    elif fault == "nan":
        arrived = replace(arrived, features=np.array([[np.nan, 0.0]]))
    elif fault == "row_ids":
        arrived = replace(arrived, row_ids=())
    elif fault == "duplicate_id":
        arrived = replace(arrived, row_ids=("row-1",))
    elif fault == "changed_width":
        arrived = replace(arrived, features=np.zeros((1, 3)))
    poison = PoisonState()
    before, call_count = observer.report(), len(calls)
    with pytest.raises(ValueError):
        observer(number, poison, arrived)
    assert poison.copies == 0 and len(calls) == call_count
    assert observer.report() == before


@pytest.mark.parametrize(
    "fault", ["prime_copy", "current_copy", "predict", "nan", "shape", "range", "detector"]
)
def test_should_stop_after_admitted_failure_keep_exact_attempt_counts_and_preserve_original_error(
    fault, monkeypatch
):
    error = RuntimeError(f"original-{fault}")
    calls: list[int] = []

    def predict(previous, features):
        calls.append(1)
        if fault == "predict":
            raise error
        return {
            "nan": np.array([[np.nan]]),
            "shape": np.zeros((1, 2)),
            "range": np.array([[1.01]]),
        }.get(fault, np.array([[0.5]]))

    observer = ArrivedShiftObserver(predict, max_updates=3)
    if fault != "prime_copy":
        observer(0, {}, None)
    if fault == "detector":

        def fail_detector(self, metrics):
            raise error

        monkeypatch.setattr(ShiftDetector, "observe", fail_detector)
    state: Any = BrokenState(error) if fault in ("prime_copy", "current_copy") else {}
    with pytest.raises((RuntimeError, ValueError)) as caught:
        observer(
            0 if fault == "prime_copy" else 1, state, None if fault == "prime_copy" else batch(1)
        )
    if fault in ("prime_copy", "current_copy", "predict", "detector"):
        assert caught.value is error
    else:
        assert type(caught.value) is ValueError
    report = observer.report()
    assert report["stopped"] is True and report["reason"].startswith(
        type(caught.value).__name__ + ":"
    )
    attempted = 0 if fault in ("prime_copy", "current_copy") else 1
    completed = 0 if fault in ("prime_copy", "current_copy", "predict") else 1
    assert len(calls) == report["prediction_calls_attempted"] == attempted
    assert report["prediction_calls_completed"] == completed
    if fault == "prime_copy":
        assert report["primed"] is False and report["initial_seconds"] == 0.0
        assert all(row["status"] == "not_run" for row in report["rows"])
    else:
        failed = report["rows"][0]
        assert failed["status"] == "failed" and failed["reason"] == report["reason"]
        assert failed["elapsed_seconds"] == 0.0 and failed["decision"] is None
        assert (failed["metrics"] is not None) is (fault == "detector")
        assert all(row["status"] == "not_run" for row in report["rows"][1:])
    poison = PoisonState()
    with pytest.raises(ValueError, match="stopped"):
        observer(1, poison, batch(2))
    assert poison.copies == 0 and len(calls) == attempted
    assert observer.report() == report


def test_should_return_detached_json_data_without_exposing_retained_states_or_detector():
    _, predict = recorder()
    observer = ArrivedShiftObserver(predict, max_updates=2)
    observer(0, {"private": np.array([1.0])}, None)
    observer(1, {"private": np.array([2.0])}, batch(1))
    expected = observer.report()
    returned = observer.report()
    returned["rows"][0]["metrics"]["feature_mean"] = (999.0,)
    returned["rows"][0]["decision"]["alarms"] = (True, True, True)
    returned["rows"][0]["status"] = "forged"
    returned["rows"].clear()
    returned["reason"] = "caller mutation"
    assert observer.report() == expected
    assert "private" not in json.dumps(expected, allow_nan=False)


def test_should_account_predictor_time_without_assuming_a_clock_read_count(monkeypatch):
    seconds = [0.0]
    monkeypatch.setattr(module, "monotonic", lambda: seconds[0])

    def predict(previous, features):
        seconds[0] += 0.25
        return np.array([[0.5]])

    observer = ArrivedShiftObserver(predict, max_updates=1)
    observer(0, {}, None)
    observer(1, {}, batch(1))
    report = observer.report()
    assert report["initial_seconds"] == 0.0
    assert report["total_seconds"] == report["rows"][0]["elapsed_seconds"] == 0.25
