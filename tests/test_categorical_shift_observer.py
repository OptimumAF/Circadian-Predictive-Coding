"""Categorical arrived-only diagnostics with scalar/scripted state delegates."""

from dataclasses import replace
import json

import numpy as np
import pytest

from src.app import arrived_shift_observer as module
from src.app.arrived_shift_observer import ArrivedShiftObserver
from src.app.coupled_attribution import AttributionBatch, run_attribution_stream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.learner_ports import TrainingDiagnostic
from src.core.shift_calibration import ShiftCalibration
from src.core.shift_signals import ShiftPolicy


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setattr(module, "monotonic", lambda: 0.0)


def batch(index=1, *, classes=6, role="train"):
    return AttributionBatch(
        role,
        "known-task",
        (f"row-{index}",),
        np.zeros((1, 9)),
        np.array([[float((index - 1) % classes)]]),
        num_classes=classes,
    )


def predict(previous, features):
    return np.full((len(features), 6), 1 / 6)


class Poison:
    def __deepcopy__(self, memo):
        pytest.fail("refused input reached state copy")


@pytest.mark.parametrize("classes", [True, 1, 17, 6.0, np.int64(6), None])
def test_should_refuse_invalid_class_controls_before_state_or_prediction(classes):
    with pytest.raises(ValueError):
        ArrivedShiftObserver(lambda *a: pytest.fail("predict"), num_classes=classes)


@pytest.mark.parametrize(
    "fault",
    [
        "final",
        "probe_train",
        "probe_test",
        "controller_train",
        "foreign_classes",
        "duplicate",
        "fractional",
        "early",
    ],
)
def test_should_refuse_foreign_unarrived_or_malformed_batch_before_copy_or_prediction(fault):
    observer = ArrivedShiftObserver(lambda *a: pytest.fail("predict"), num_classes=6)
    observer(0, {}, None)
    value = batch()
    if fault in ("final", "probe_train", "probe_test", "controller_train"):
        value = replace(value, role="final_test" if fault == "final" else fault)
    elif fault == "foreign_classes":
        value = batch(classes=3)
    elif fault == "duplicate":
        value = replace(
            value, row_ids=("same", "same"), features=np.zeros((2, 9)), targets=np.zeros((2, 1))
        )
    elif fault == "fractional":
        value.targets.fill(0.25)
    before = observer.report()
    with pytest.raises(ValueError):
        observer(0 if fault == "early" else 1, Poison(), value)
    assert observer.report() == before


def test_should_isolate_previous_state_arrays_and_report_with_explicit_categorical_costs():
    states = [{"version": n, "private": np.array([n], dtype=float)} for n in range(3)]
    versions = []
    original = batch()

    def mutating(previous, features):
        versions.append(previous["version"])
        previous["private"].fill(99)
        previous["version"] = 99
        features.fill(99)
        original.targets.fill(5)
        states[len(versions)]["version"] = 999
        return np.array([[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]])

    observer = ArrivedShiftObserver(mutating, num_classes=6, max_updates=2)
    observer(0, states[0], None)
    states[0]["version"] = 555
    observer(1, states[1], original)
    observer(2, states[2], batch(2))
    assert versions == [0, 1]
    np.testing.assert_array_equal(original.features, np.zeros((1, 9)))
    report = observer.report()
    assert report["rows"][0]["metrics"]["residual"] == 0.5
    assert report["rows"][1]["metrics"]["residual"] == 0.9
    assert report["prediction_schema"] == dict(
        num_classes=6,
        uncertainty="mean_entropy_divided_by_log_num_classes",
        residual="mean_one_minus_true_class_probability",
        prediction_rows_attempted=2,
        prediction_rows_completed=2,
    )
    assert report["prediction_calls_completed"] == 2
    assert "private" not in json.dumps(report) and "targets" not in json.dumps(report)
    report["prediction_schema"]["num_classes"] = 2
    report["rows"].clear()
    assert (
        len(observer.report()["rows"]) == 2
        and observer.report()["prediction_schema"]["num_classes"] == 6
    )


@pytest.mark.parametrize("failure", ["raise", "wrong_shape", "invalid_simplex"])
def test_should_retain_paid_prediction_failure_and_stop_further_diagnostics(failure):
    def broken(previous, features):
        if failure == "raise":
            raise RuntimeError("lost categorical return")
        return np.zeros((1, 1 if failure == "wrong_shape" else 6))

    observer = ArrivedShiftObserver(broken, num_classes=6, max_updates=2)
    observer(0, {}, None)
    with pytest.raises((ValueError, RuntimeError)):
        observer(1, {}, batch())
    report = observer.report()
    assert report["stopped"] and report["rows"][0]["status"] == "failed"
    assert report["rows"][1]["status"] == "not_run"
    assert report["prediction_schema"]["prediction_rows_attempted"] == 1
    assert report["prediction_schema"]["prediction_rows_completed"] == (
        0 if failure == "raise" else 1
    )
    with pytest.raises(ValueError, match="stopped"):
        observer(2, Poison(), batch(2))


def test_should_keep_binary_defaults_and_report_exact_with_explicit_two_classes():
    delegate = lambda state, features: np.full((len(features), 1), 0.5)
    binary = ArrivedShiftObserver(delegate, max_updates=1)
    explicit = ArrivedShiftObserver(delegate, max_updates=1, num_classes=2)
    value = batch(classes=2)
    value.targets.fill(0.25)
    for observer in (binary, explicit):
        observer(0, {}, None)
        observer(1, {}, value)
    assert binary.report() == explicit.report()
    assert "prediction_schema" not in binary.report()
    assert binary.report()["rows"][0]["metrics"]["residual"] == 0.25


def test_should_feed_existing_calibration_and_incidents_without_refitting_from_later_rows():
    observer = ArrivedShiftObserver(
        predict,
        num_classes=6,
        max_updates=4,
        incidents=True,
        policy=ShiftPolicy(reference_batches=1, window_batches=1, persistence_batches=1),
        calibration=ShiftCalibration(prefix_batches=2),
        calibration_model_id="scripted-categorical",
    )
    observer(0, {}, None)
    for number in (1, 2):
        observer(number, {}, batch(number))
    fitted = observer.report()["calibration"]["result"]
    for number in (3, 4):
        value = batch(number)
        value.features.fill(1)
        observer(number, {}, value)
    report = observer.report()
    assert report["calibration"]["result"] == fitted
    assert report["rows"][2]["incident"]["states"][1] == "onset"
    assert report["rows"][3]["incident"]["states"][1] == "continued"
    assert report["incident_semantics"]["feedback"] is False


class ScriptedLearner:
    """Finite counter changes are fixtures, not trained model functionality."""

    def __init__(self, value=0):
        self.value = value

    def fork(self):
        return ScriptedLearner(self.value)

    def snapshot_state(self):
        return np.array([float(self.value)])

    def restore_state(self, state):
        self.value = int(state[0])

    def train_batch(self, features, targets):
        self.value += 1
        return TrainingDiagnostic("scripted_shift_fixture", float(self.value))

    def predict(self, features):
        pytest.fail("training-only stream must not score candidate/final data")


def test_should_observe_only_arrived_labels_through_existing_delayed_categorical_stream():
    results = []
    for enabled in (False, True):
        source = ScriptedLearner()
        budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=3), lambda: 0.0)
        calls = []
        chronology = []

        def score(previous, features):
            calls.append(int(previous[0]))
            return np.full((len(features), 6), 1 / 6)

        observer = ArrivedShiftObserver(score, num_classes=6, max_updates=3)

        def observe(updates, state, arrived):
            chronology.append(("arrived", updates, None if arrived is None else arrived.row_ids))
            observer(updates, state, arrived)

        def prearrival(value, state):
            assert not hasattr(value, "targets") and not hasattr(value, "context")
            chronology.append(("prearrival", value.tick, None))

        result = run_attribution_stream(
            source,
            tuple(batch(n) for n in (1, 2, 3)),
            (),
            checkpoint_updates=(0, 1, 2, 3),
            label_delay=2,
            budget=budget,
            training_only=True,
            observe_checkpoint=observe if enabled else None,
            observe_features=prearrival,
        )
        assert source.value == 0 and budget.updates_completed == 3
        assert calls == ([0, 1, 2] if enabled else [])
        applied = result["applied_updates"]
        assert [item["observed_at"] for item in applied] == [0, 1, 2]
        assert [item["arrived_at"] for item in applied] == [2, 3, 4]
        assert [item["applied_at"] for item in applied] == [2, 3, 4]
        if enabled:
            first_arrival = chronology.index(("arrived", 1, ("row-1",)))
            assert ("prearrival", 0, None) in chronology[:first_arrival]
            assert ("prearrival", 1, None) in chronology[:first_arrival]
            assert observer.report()["prediction_schema"]["prediction_rows_completed"] == 3
        results.append(result)
    assert results[0] == results[1]
