"""Arrived-label monitoring of a fixed sensor recipe, without learner execution."""

from copy import deepcopy
import json

import numpy as np
import pytest

from scripts.run_sensor_drift_demo import main
from src.adapters.sensor_drift_demo import describe_protocol
from src.app.sensor_monitoring import SensorMonitoring
from src.core.sensor_stream import make_sensor_stream
from src.core.shift_calibration import CalibratedShiftDetector


def observe_recipe(monitor, probabilities=None):
    """Use caller-specified probabilities, never execute or tune a learner."""
    fixture = make_sensor_stream()
    for tick, batch in enumerate(fixture):
        predicted = batch.targets.copy() if probabilities is None else probabilities[tick]
        monitor.observe(tick, batch.features, predicted, batch.targets)
    return monitor.report()


def test_should_exclude_all_stationary_calibration_batches_from_false_alarm_denominators():
    report = observe_recipe(SensorMonitoring())
    false_alarms = report["false_alarms"]
    assert false_alarms["pre_shift_A"] == dict(
        monitoring_batches=0,
        alarm_batches=[0, 0, 0],
        incident_onsets=[0, 0, 0],
        rates=[None, None, None],
    )
    assert false_alarms["post_return_A"] == dict(
        monitoring_batches=4,
        alarm_batches=[0, 1, 0],
        incident_onsets=[0, 0, 0],
        rates=[0.0, 0.25, 0.0],
    )


def test_should_count_contaminated_return_window_alarm_without_inventing_a_new_incident():
    report = observe_recipe(SensorMonitoring())
    rows = report["rows"]
    assert [row["feature_tick"] for row in rows] == list(range(12))
    assert [row["label_tick"] for row in rows] == list(range(2, 14))
    assert [row["accuracy"] for row in rows] == [1.0] * 12
    assert [row["decision"]["phase"] for row in rows[:4]] == ["calibration"] * 4
    assert all(not any(row["decision"]["alarms"]) for row in rows[:4])
    assert [i for i, row in enumerate(rows) if row["decision"]["alarms"][1]] == [5, 6, 7, 8]
    assert rows[5]["incident"]["states"][1] == "onset"
    assert rows[8]["incident"]["states"][1] == "continued"
    assert rows[9]["incident"]["states"][1] == "cleared"
    assert rows[8]["decision"]["feature_drift"] == pytest.approx(0.125)
    assert rows[9]["decision"]["feature_drift"] == pytest.approx(0.0)


def test_should_fit_only_the_first_four_batches_and_freeze_thresholds_after_the_shift():
    monitor = SensorMonitoring()
    fixture = make_sensor_stream()
    for tick, batch in enumerate(fixture[:4]):
        probabilities = 0.1 + 0.8 * batch.targets
        monitor.observe(tick, batch.features, probabilities, batch.targets)
        if tick < 3:
            assert monitor.report()["calibration"] is None
    fitted = deepcopy(monitor.report()["calibration"])
    assert fitted["calibration_batches"] == 4
    assert fitted["calibration_rows"] == 16
    assert fitted["calibration_windows"] == 1
    assert fitted["replayed_observations"] == 4
    assert fitted["policy"]["reference_batches"] == 2
    assert fitted["policy"]["window_batches"] == 2
    assert fitted["policy"]["persistence_batches"] == 2
    for channel in ("uncertainty", "feature", "residual"):
        assert fitted["policy"][f"{channel}_threshold"] == pytest.approx(1e-6)
    for tick, batch in enumerate(fixture[4:], 4):
        monitor.observe(tick, batch.features, np.full((4, 1), 0.5), batch.targets)
        assert monitor.report()["calibration"] == fitted
    fitted["policy"]["feature_threshold"] = 2.0
    assert monitor.report()["calibration"]["policy"]["feature_threshold"] == pytest.approx(1e-6)


def test_should_report_no_deficit_with_null_recovery_labels_in_both_observed_phases():
    report = observe_recipe(SensorMonitoring())
    assert report["complete"] is True and report["stopped"] is False
    assert report["metrics_batches_completed"] == 12 and report["native_forward_calls"] == 0
    for phase in ("B", "A_return"):
        summary = report["recovery"][phase]
        assert summary == dict(
            status="no_deficit",
            first_deficit_tick=None,
            confirmation_tick=None,
            labels_to_recovery=None,
            labels_available_before_last_prediction=4,
            observed_batches=4,
            planned_batches=4,
            horizon_complete=True,
        )


def test_should_distinguish_recovered_and_censored_phases_with_prediction_time_label_counts():
    probabilities = [batch.targets.copy() for batch in make_sensor_stream()]
    for tick in (4, 8, 10):
        probabilities[tick] = 1.0 - probabilities[tick]
    report = observe_recipe(SensorMonitoring(), probabilities)
    # At prediction tick6, labels from B's tick4 arrive later in that same tick.
    assert report["recovery"]["B"] == dict(
        status="recovered",
        first_deficit_tick=4,
        confirmation_tick=6,
        labels_to_recovery=0,
        labels_available_before_last_prediction=4,
        observed_batches=4,
        planned_batches=4,
        horizon_complete=True,
    )
    # A good final prediction alone cannot supply two consecutive good batches.
    assert report["recovery"]["A_return"] == dict(
        status="censored",
        first_deficit_tick=8,
        confirmation_tick=None,
        labels_to_recovery=None,
        labels_available_before_last_prediction=4,
        observed_batches=4,
        planned_batches=4,
        horizon_complete=True,
    )


def test_should_report_partial_observation_horizon_without_claiming_missing_batches_were_good():
    monitor = SensorMonitoring()
    fixture = make_sensor_stream()
    initial = monitor.report()
    assert initial["complete"] is False and initial["calibration"] is None
    for summary in initial["recovery"].values():
        assert summary["status"] == "not_run" and summary["observed_batches"] == 0
        assert summary["labels_to_recovery"] is None and summary["horizon_complete"] is False
    for tick, batch in enumerate(fixture[:5]):
        probabilities = batch.targets if tick < 4 else 1.0 - batch.targets
        monitor.observe(tick, batch.features, probabilities, batch.targets)
    partial = monitor.report()
    assert partial["complete"] is False
    assert partial["recovery"]["B"]["status"] == "censored"
    assert partial["recovery"]["B"]["observed_batches"] == 1
    assert partial["recovery"]["B"]["horizon_complete"] is False
    assert partial["recovery"]["A_return"]["status"] == "not_run"


def test_should_stop_after_uncertain_detector_failure_and_keep_completed_calibration(monkeypatch):
    monitor = SensorMonitoring()
    fixture = make_sensor_stream()
    for tick, batch in enumerate(fixture[:4]):
        monitor.observe(tick, batch.features, batch.targets, batch.targets)
    before = monitor.report()
    error = RuntimeError("uncertain detector failure")

    def fail(self, metrics):
        raise error

    monkeypatch.setattr(CalibratedShiftDetector, "observe", fail)
    with pytest.raises(RuntimeError) as caught:
        monitor.observe(4, fixture[4].features, fixture[4].targets, fixture[4].targets)
    assert caught.value is error
    report = monitor.report()
    assert report["stopped"] is True and report["complete"] is False
    assert report["rows"] == before["rows"] and report["calibration"] == before["calibration"]
    with pytest.raises(ValueError, match="stopped"):
        monitor.observe(4, fixture[4].features, fixture[4].targets, fixture[4].targets)


@pytest.mark.parametrize("invalid_tick", [-1, True, 0.0, 1, 12])
def test_should_refuse_non_native_or_out_of_sequence_ticks_without_consuming_the_next_batch(
    invalid_tick,
):
    monitor = SensorMonitoring()
    batch = make_sensor_stream()[0]
    before = monitor.report()
    with pytest.raises(ValueError):
        monitor.observe(invalid_tick, batch.features, batch.targets, batch.targets)
    assert monitor.report() == before
    assert monitor.observe(0, batch.features, batch.targets, batch.targets)["feature_tick"] == 0


def test_should_refuse_more_than_twelve_observations_without_losing_completed_rows():
    monitor = SensorMonitoring()
    before = observe_recipe(monitor)
    batch = make_sensor_stream()[0]
    with pytest.raises(ValueError):
        monitor.observe(12, batch.features, batch.targets, batch.targets)
    assert monitor.report() == before


@pytest.mark.parametrize("fault", ["feature_dtype", "feature_width", "prob_shape", "nan", "range"])
def test_should_refuse_malformed_batch_before_calibration_or_sequence_progress(fault):
    monitor = SensorMonitoring()
    batch = make_sensor_stream()[0]
    features, probabilities, targets = batch.features, batch.targets.copy(), batch.targets
    if fault == "feature_dtype":
        features = features.astype(np.float32)
    elif fault == "feature_width":
        features = features[:, :-1]
    elif fault == "prob_shape":
        probabilities = probabilities[:, 0]
    elif fault == "nan":
        probabilities[0, 0] = np.nan
    else:
        probabilities[0, 0] = 1.01
    before = monitor.report()
    with pytest.raises(ValueError):
        monitor.observe(0, features, probabilities, targets)
    assert monitor.report() == before
    monitor.observe(0, batch.features, batch.targets, batch.targets)


def test_should_detach_each_observation_from_caller_arrays_and_returned_rows():
    monitor = SensorMonitoring()
    batch = make_sensor_stream()[0]
    features, probabilities, targets = (
        batch.features.copy(),
        batch.targets.copy(),
        batch.targets.copy(),
    )
    row = monitor.observe(0, features, probabilities, targets)
    expected = deepcopy(monitor.report())
    features.fill(-1.0)
    probabilities.fill(0.5)
    targets.fill(0.5)
    row["accuracy"] = 0.0
    row["decision"]["phase"] = "forged"
    row["metrics"]["feature_mean"] = [999.0] * 16
    assert monitor.report() == expected


def test_should_detach_nested_reports_and_serialize_without_nan_or_live_arrays():
    monitor = SensorMonitoring()
    report = observe_recipe(monitor)
    expected = deepcopy(report)
    report["rows"][0]["accuracy"] = 0.0
    report["false_alarms"]["post_return_A"]["alarm_batches"][1] = 100
    report["recovery"]["B"]["status"] = "forged"
    assert monitor.report() == expected
    encoded = json.dumps(monitor.report(), allow_nan=False)
    assert "NaN" not in encoded


def test_should_keep_default_recipe_unchanged_when_monitoring_is_explicitly_disabled(capsys):
    default = describe_protocol()
    assert default == describe_protocol(monitoring=False)
    assert default["protocol"] == "sensor_brightness_prearrival_bp_v1"
    assert main(["--describe"]) == 0
    assert json.loads(capsys.readouterr().out) == default


def test_should_describe_monitoring_without_changing_the_training_recipe(capsys):
    default = describe_protocol()
    assert main(["--describe", "--monitoring"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == describe_protocol(monitoring=True)
    assert report["protocol"] == "sensor_brightness_prearrival_monitoring_bp_v1"
    for field in (
        "seed",
        "dimensions",
        "learning_rate",
        "phases",
        "batches_per_phase",
        "rows_per_batch",
        "label_delay",
        "training_updates",
        "training_rows",
        "diagnostic_forward_calls",
        "diagnostic_forward_rows",
        "training_forward_backward_evaluations",
        "training_evaluated_rows",
        "body_seconds",
        "process_rss_bytes",
        "comparison",
        "final_evaluation",
    ):
        assert report[field] == default[field]


@pytest.mark.parametrize("mode", [0, 1, None, "yes", np.bool_(True)])
def test_should_refuse_non_native_monitoring_flags_before_any_learner_execution(mode):
    with pytest.raises(ValueError):
        describe_protocol(monitoring=mode)
