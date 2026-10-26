"""Opt-in prefix reports bind arrived IDs to retrospective parameter indices."""

from dataclasses import replace
import json
from typing import Any

import numpy as np
import pytest

from src.app import arrived_shift_observer as module
from src.app.arrived_shift_observer import ArrivedShiftObserver
from src.app.coupled_attribution import AttributionBatch
from src.core.shift_calibration import CalibratedShiftDetector, ShiftCalibration
from src.core.shift_signals import ShiftPolicy


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setattr(module, "monotonic", lambda: 0.0)


def arrived(number, feature=0.0):
    return AttributionBatch(
        "train",
        "declared-C",
        (f"arrived-{number}",),
        np.array([[feature]], dtype=np.float64),
        np.zeros((1, 1)),
    )


def make_observer(predict, **options):
    controls: dict[str, Any] = dict(
        policy=ShiftPolicy(reference_batches=1, window_batches=1, persistence_batches=2),
        calibration=ShiftCalibration(prefix_batches=2, margin=0.05),
        calibration_model_id="declared-run",
        max_updates=4,
    )
    controls.update(options)
    return ArrivedShiftObserver(predict, **controls)


class Poison:
    def __init__(self):
        self.copies = 0

    def __deepcopy__(self, memo):
        self.copies += 1
        pytest.fail("refused calibration input was copied")


@pytest.mark.parametrize(
    "fault",
    [
        "missing_id",
        "blank_id",
        "bytes_id",
        "boolean_id",
        "long_utf8_id",
        "config_type",
        "config_tamper",
        "prefix_equal",
        "prefix_larger",
        "id_without_config",
        "policy_type",
    ],
)
def test_should_refuse_invalid_opt_in_controls_before_any_callback(fault):
    options: dict[str, Any] = {}
    if fault.endswith("_id"):
        options["calibration_model_id"] = {
            "missing_id": None,
            "blank_id": " \t",
            "bytes_id": b"run",
            "boolean_id": True,
            "long_utf8_id": "\U0001f600" * 33,
        }[fault]
    elif fault == "config_type":
        options["calibration"] = object()
    elif fault == "config_tamper":
        value = ShiftCalibration(prefix_batches=2)
        object.__setattr__(value, "margin", 0)
        options["calibration"] = value
    elif fault in ("prefix_equal", "prefix_larger"):
        options["calibration"] = ShiftCalibration(
            prefix_batches=4 if fault == "prefix_equal" else 5
        )
    elif fault == "id_without_config":
        options["calibration"] = None
    else:
        options["policy"] = object()
    with pytest.raises(ValueError):
        make_observer(lambda *args: pytest.fail("invalid controls predicted"), **options)


def test_should_report_prefix_fit_identity_and_arrived_ids_without_leaking_labels_or_future_state():
    calls: list[tuple[int, float]] = []

    def predict(previous, features):
        calls.append((previous["updates"], previous["p"]))
        return np.full((len(features), 1), previous["p"])

    identity = "\U0001f600" * 32  # The bound is UTF8 bytes, not character count.
    observer = make_observer(predict, calibration_model_id=identity)
    initial = observer.report()
    assert initial["calibration"] == {
        "configuration": {"prefix_batches": 2, "margin": 0.05},
        "model_id": identity,
        "identity_kind": "caller_declared_run_and_parameter_update_indices",
        "result": None,
    }
    assert all(row["arrived_row_ids"] is None for row in initial["rows"])
    observer(0, {"updates": 0, "p": 0.5}, None)
    for number in range(1, 5):
        observer(
            number,
            {"updates": number, "p": 0.5 if number == 1 else 0.0},
            arrived(number, 0.0 if number <= 2 else 1.0),
        )
        report = observer.report()
        if number == 1:
            assert report["calibration"]["result"] is None
        elif number == 2:
            fitted = report["calibration"]["result"]
    assert calls == [(0, 0.5), (1, 0.5), (2, 0.0), (3, 0.0)]
    report = observer.report()
    assert report["calibration"]["result"] == fitted
    assert fitted == {
        "policy": {
            "reference_batches": 1,
            "window_batches": 1,
            "persistence_batches": 2,
            "uncertainty_threshold": 0.05,
            "feature_threshold": 0.05,
            "residual_threshold": 0.05,
        },
        "reference_uncertainty": 1.0,
        "reference_feature_mean": (0.0,),
        "reference_residual": 0.5,
        "score_ceilings": (0.0, 1.0, 0.5),
        "reachable": (False, True, True),
        "calibration_batches": 2,
        "calibration_rows": 2,
        "calibration_windows": 1,
        "replayed_observations": 2,
    }
    for number, row in enumerate(report["rows"], 1):
        assert row["arrived_row_ids"] == (f"arrived-{number}",)
        assert row["prediction_state_updates"] == number - 1
        assert row["status"] == "complete" and row["reason"] is None
        assert row["decision"]["batch_index"] == number
        assert row["decision"]["phase"] == ("calibration" if number <= 2 else "monitor")
    assert report["rows"][2]["decision"]["alarms"] == (False, False, False)
    assert report["rows"][3]["decision"]["alarms"] == (False, True, False)
    serialized = json.dumps(report, allow_nan=False)
    assert "targets" not in serialized and '"p"' not in serialized


def test_should_keep_default_report_and_results_exact_when_calibration_is_disabled():
    predict = lambda previous, features: np.full((len(features), 1), 0.5)
    old = ArrivedShiftObserver(predict, max_updates=2)
    explicit = ArrivedShiftObserver(
        predict, max_updates=2, calibration=None, calibration_model_id=None
    )
    for observer in (old, explicit):
        observer(0, {}, None)
        observer(1, {}, arrived(1))
    assert old.report() == explicit.report()
    assert "calibration" not in old.report()
    assert all("arrived_row_ids" not in row for row in old.report()["rows"])


def test_should_fit_pristine_metrics_and_keep_previous_state_when_delegate_and_caller_mutate():
    states: list[dict[str, Any]] = [{"version": value, "p": 0.5} for value in range(3)]
    versions: list[int] = []

    def predict(previous, features):
        versions.append(previous["version"])
        previous["version"] = 999
        features.fill(1.0)
        states[len(versions)]["version"] = 777
        return np.array([[0.5]])

    observer = make_observer(predict)
    observer(0, states[0], None)
    states[0]["version"] = 555
    first, second = arrived(1, 0.25), arrived(2, 0.25)
    observer(1, states[1], first)
    observer(2, states[2], second)
    assert versions == [0, 1]
    assert first.features[0, 0] == second.features[0, 0] == 0.25
    report = observer.report()
    assert report["calibration"]["result"]["reference_feature_mean"] == (0.25,)
    assert report["calibration"]["result"]["policy"]["feature_threshold"] == 0.05
    assert all(row["metrics"]["feature_mean"] == (0.25,) for row in report["rows"][:2])


def test_should_detach_caller_configuration_and_nested_report_policy_from_future_monitoring():
    config = ShiftCalibration(prefix_batches=2, margin=0.125)
    observer = make_observer(lambda previous, features: np.array([[0.5]]), calibration=config)
    object.__setattr__(config, "margin", 1.0)
    object.__setattr__(config, "prefix_batches", 32)
    observer(0, {}, None)
    observer(1, {}, arrived(1))
    observer(2, {}, arrived(2))
    report = observer.report()
    expected = observer.report()
    report["calibration"]["configuration"]["prefix_batches"] = 32
    report["calibration"]["result"]["policy"]["feature_threshold"] = 2.0
    report["rows"][0]["arrived_row_ids"] = ("forged",)
    assert observer.report() == expected
    observer(3, {}, arrived(3, 1.0))
    observer(4, {}, arrived(4, 1.0))
    assert observer.report()["rows"][3]["decision"]["alarms"][1] is True
    assert observer.report()["calibration"]["result"] == expected["calibration"]["result"]


@pytest.mark.parametrize("role", ["final_test", "probe_test", "inner_guard"])
def test_should_exclude_nontraining_roles_before_payload_copy_prediction_or_prefix_fit(role):
    observer = make_observer(lambda *args: pytest.fail("heldout data reached prediction"))
    observer(0, {}, None)
    poison = Poison()
    payload: Any = poison
    batch = AttributionBatch(role, "heldout", ("heldout-row",), payload, payload)
    before = observer.report()
    with pytest.raises(ValueError):
        observer(1, poison, batch)
    assert observer.report() == before and poison.copies == 0


def test_should_refuse_reused_row_ids_before_extending_calibration_and_accept_the_next_fresh_id():
    calls: list[int] = []

    def predict(previous, features):
        calls.append(1)
        return np.array([[0.5]])

    observer = make_observer(predict)
    observer(0, {}, None)
    observer(1, {}, arrived(1))
    poison = Poison()
    before = observer.report()
    with pytest.raises(ValueError):
        observer(2, poison, replace(arrived(2), row_ids=("arrived-1",)))
    assert poison.copies == 0 and len(calls) == 1 and observer.report() == before
    observer(2, {}, arrived(2))
    assert observer.report()["calibration"]["result"]["calibration_batches"] == 2
    assert observer.report()["calibration"]["result"]["calibration_rows"] == 2


@pytest.mark.parametrize(
    "fault", ["prefix_predict", "prefix_detector", "monitor_predict", "monitor_copy"]
)
def test_should_preserve_fit_and_all_planned_slots_when_an_admitted_calibrated_observation_fails(
    fault, monkeypatch
):
    error = RuntimeError(f"original-{fault}")
    calls: list[int] = []
    failed_update = 1 if fault == "prefix_predict" else 2 if fault == "prefix_detector" else 3

    def predict(previous, features):
        calls.append(1)
        if fault.endswith("predict") and len(calls) == failed_update:
            raise error
        return np.array([[0.5]])

    if fault == "prefix_detector":
        original = CalibratedShiftDetector.observe
        observations = [0]

        def fail_at_prefix_end(self, metrics):
            observations[0] += 1
            if observations[0] == 2:
                raise error
            return original(self, metrics)

        monkeypatch.setattr(CalibratedShiftDetector, "observe", fail_at_prefix_end)

    class Broken:
        def __deepcopy__(self, memo):
            raise error

    observer = make_observer(predict)
    observer(0, {}, None)
    for number in range(1, failed_update):
        observer(number, {}, arrived(number))
    fitted = observer.report()["calibration"]["result"]
    with pytest.raises(RuntimeError) as caught:
        observer(failed_update, Broken() if fault == "monitor_copy" else {}, arrived(failed_update))
    assert caught.value is error
    report = observer.report()
    assert report["stopped"] is True and report["reason"] == f"RuntimeError: original-{fault}"
    assert report["calibration"]["result"] == fitted
    assert (fitted is None) is fault.startswith("prefix")
    failed = report["rows"][failed_update - 1]
    assert failed["status"] == "failed" and failed["arrived_row_ids"] == (
        f"arrived-{failed_update}",
    )
    for row in report["rows"][failed_update:]:
        assert row["status"] == "not_run" and row["arrived_row_ids"] is None
    attempted = failed_update - (fault == "monitor_copy")
    completed = attempted - fault.endswith("predict")
    assert report["prediction_calls_attempted"] == len(calls) == attempted
    assert report["prediction_calls_completed"] == completed
    poison = Poison()
    with pytest.raises(ValueError, match="stopped"):
        observer(failed_update + 1, poison, arrived(99))
    assert poison.copies == 0 and observer.report() == report


def test_should_include_calibration_fit_and_replay_time_in_the_observation_duration(monkeypatch):
    seconds = [0.0]
    monkeypatch.setattr(module, "monotonic", lambda: seconds[0])
    original = CalibratedShiftDetector.observe

    def observe(self, metrics):
        result = original(self, metrics)
        if self.calibration_result is not None:
            seconds[0] += 0.125
        return result

    monkeypatch.setattr(CalibratedShiftDetector, "observe", observe)
    observer = make_observer(lambda previous, features: np.array([[0.5]]))
    observer(0, {}, None)
    observer(1, {}, arrived(1))
    observer(2, {}, arrived(2))
    report = observer.report()
    assert report["rows"][0]["elapsed_seconds"] == 0.0
    assert report["rows"][1]["elapsed_seconds"] == report["total_seconds"] == 0.125
