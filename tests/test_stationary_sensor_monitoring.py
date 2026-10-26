"""Stationary data ownership and observed false alarms without model execution."""

from copy import deepcopy
import json

import numpy as np
import pytest

import src.app.sensor_monitoring as monitoring
import src.core.sensor_stream as sensor_stream
from src.app.sensor_monitoring import SensorMonitoring, monitoring_configuration
from src.core.sensor_stream import make_sensor_stream


EMPTY_ALARMS = dict(
    monitoring_batches=0,
    alarm_batches=[0, 0, 0],
    incident_onsets=[0, 0, 0],
    rates=[None, None, None],
)
ABSENT_RECOVERY = dict(
    status="not_applicable",
    first_deficit_tick=None,
    confirmation_tick=None,
    labels_to_recovery=None,
    labels_available_before_last_prediction=0,
    observed_batches=0,
    planned_batches=0,
    horizon_complete=False,
)


def test_should_use_exact_existing_a_pixels_and_labels_in_all_stationary_batches():
    old = make_sensor_stream()
    stationary = make_sensor_stream(stationary=True)
    repeated = make_sensor_stream(stationary=True)
    assert len(stationary) == 12
    for batch in (*stationary, *repeated):
        np.testing.assert_array_equal(batch.features, old[0].features)
        np.testing.assert_array_equal(batch.targets, old[0].targets)
        assert batch.features.shape == (4, 16) and batch.targets.shape == (4, 1)
        assert batch.features.dtype == batch.targets.dtype == np.float64
    arrays = [
        values
        for batch in (*stationary, *repeated, *old)
        for values in (batch.features, batch.targets)
    ]
    for index, left in enumerate(arrays):
        for right in arrays[index + 1 :]:
            assert not np.shares_memory(left, right)
    stationary[0].features.fill(99)
    stationary[0].targets.fill(99)
    for batch in stationary[1:]:
        np.testing.assert_array_equal(batch.features, old[0].features)
        np.testing.assert_array_equal(batch.targets, old[0].targets)
    np.testing.assert_array_equal(repeated[0].features, old[0].features)


def test_should_preserve_original_default_and_explicit_false_pixels_and_descriptor():
    default = make_sensor_stream()
    explicit = make_sensor_stream(stationary=False)
    pattern = np.array([0.6, 0.6, 0.1, 0.1] * 4)
    expected = np.stack((pattern, 0.7 - pattern, pattern, 0.7 - pattern))
    for index, (left, right) in enumerate(zip(default, explicit)):
        offset = 0.25 if 4 <= index < 8 else 0.0
        np.testing.assert_allclose(left.features, expected + offset, rtol=0, atol=1e-16)
        np.testing.assert_array_equal(left.features, right.features)
        np.testing.assert_array_equal(left.targets, [[0.0], [1.0], [0.0], [1.0]])
        np.testing.assert_array_equal(left.targets, right.targets)
    assert (
        monitoring_configuration()
        == monitoring_configuration(stationary=False)
        == dict(
            policy=dict(
                reference_batches=2,
                window_batches=2,
                persistence_batches=2,
                uncertainty_threshold=0.05,
                feature_threshold=0.25,
                residual_threshold=0.05,
            ),
            calibration=dict(prefix_batches=4, margin=1e-6),
            calibration_scope="first four stationary-sensor A batches only; model may learn",
            observation_time="arrived labels; predictions and pixels retained from prearrival",
            channels=["uncertainty", "feature", "residual"],
            false_alarm_scope="exclude calibration; count return-A rolling-window carryover",
            recovery=dict(
                accuracy_threshold=0.75,
                recovery_batches=2,
                label_delay=2,
                rows_per_batch=4,
                labels="phase labels arrived strictly before confirmation prediction",
            ),
            feedback=False,
        )
    )
    assert SensorMonitoring().report() == SensorMonitoring(stationary=False).report()


def observe_prefix(monitor, count, *, synthetic_alarms=False):
    for tick, batch in enumerate(make_sensor_stream(stationary=True)[:count]):
        features = batch.features
        probabilities = batch.targets.copy()
        if synthetic_alarms and tick >= 4:
            # Inject known descriptive changes to test accounting in every
            # channel; this does not claim these occur in the all-A fixture.
            features = features + 0.25
            probabilities.fill(0.5)
        monitor.observe(tick, features, probabilities, batch.targets)
    return monitor.report()


@pytest.mark.parametrize("count", [0, 3, 4, 5, 6, 9, 12])
def test_should_count_actual_postprefix_stationary_alarms_and_not_invent_recovery(count):
    report = observe_prefix(SensorMonitoring(stationary=True), count, synthetic_alarms=True)
    observed = max(0, count - 4)
    alarms = max(0, count - 5)
    onsets = int(count >= 6)
    assert report["false_alarms"] == dict(
        pre_shift_A=EMPTY_ALARMS,
        post_return_A=EMPTY_ALARMS,
        stationary_A=dict(
            monitoring_batches=observed,
            alarm_batches=[alarms] * 3,
            incident_onsets=[onsets] * 3,
            rates=[alarms / observed if observed else None] * 3,
        ),
    )
    assert report["recovery"] == dict(B=ABSENT_RECOVERY, A_return=ABSENT_RECOVERY)
    assert report["complete"] is (count == 12)
    assert report["metrics_batches_completed"] == count
    assert report["native_forward_calls"] == 0
    assert [row["feature_tick"] for row in report["rows"]] == list(range(count))
    if count >= 4:
        fitted = report["calibration"]
        assert fitted["calibration_batches"] == 4 and fitted["calibration_rows"] == 16
        for channel in monitoring.CHANNELS:
            assert fitted["policy"][f"{channel}_threshold"] == pytest.approx(1e-6)


def test_should_report_all_eight_stationary_monitoring_batches_with_real_unchanged_fixture():
    report = observe_prefix(SensorMonitoring(stationary=True), 12)
    assert report["false_alarms"]["stationary_A"] == dict(
        monitoring_batches=8,
        alarm_batches=[0, 0, 0],
        incident_onsets=[0, 0, 0],
        rates=[0.0, 0.0, 0.0],
    )
    assert report["recovery"] == dict(B=ABSENT_RECOVERY, A_return=ABSENT_RECOVERY)
    assert [row["label_tick"] for row in report["rows"]] == list(range(2, 14))


def test_should_declare_stationary_truth_without_changing_detector_configuration():
    old = monitoring_configuration()
    stationary = monitoring_configuration(stationary=True)
    assert stationary["stationary"] is True and stationary["phase_schedule"] == ["A"] * 12
    assert "ticks4..11" in stationary["false_alarm_scope"]
    for key in ("policy", "calibration", "recovery", "feedback", "channels"):
        assert stationary[key] == old[key]
    ordinary = SensorMonitoring()
    control = SensorMonitoring(stationary=True)
    # Schedule truth affects only reporting; identical observations have exact
    # detector decisions and incidents under the original calibration policy.
    for tick, batch in enumerate(make_sensor_stream(stationary=True)):
        left = ordinary.observe(tick, batch.features, batch.targets, batch.targets)
        right = control.observe(tick, batch.features, batch.targets, batch.targets)
        assert left == right


def test_should_detach_stationary_configuration_rows_and_summaries():
    monitor = SensorMonitoring(stationary=True)
    report = observe_prefix(monitor, 12, synthetic_alarms=True)
    expected = deepcopy(report)
    report["configuration"]["phase_schedule"][0] = "B"
    report["false_alarms"]["stationary_A"]["alarm_batches"][0] = 100
    report["recovery"]["B"]["status"] = "no_deficit"
    report["rows"][0]["accuracy"] = 0
    assert monitor.report() == expected
    json.dumps(monitor.report(), allow_nan=False)
    first = monitoring_configuration(stationary=True)
    first["phase_schedule"][0] = "B"
    assert monitoring_configuration(stationary=True)["phase_schedule"] == ["A"] * 12


def test_should_preserve_actual_stationary_denominator_after_uncertain_observation(monkeypatch):
    monitor = SensorMonitoring(stationary=True)
    before = observe_prefix(monitor, 6, synthetic_alarms=True)

    def fail(self, metrics):
        raise RuntimeError("uncertain stationary observation")

    monkeypatch.setattr(monitoring.CalibratedShiftDetector, "observe", fail)
    batch = make_sensor_stream(stationary=True)[6]
    with pytest.raises(RuntimeError, match="uncertain stationary observation"):
        monitor.observe(6, batch.features, batch.targets, batch.targets)
    after = monitor.report()
    assert after["stopped"] is True and after["complete"] is False
    assert after["rows"] == before["rows"]
    assert after["false_alarms"] == before["false_alarms"]
    assert after["false_alarms"]["stationary_A"]["monitoring_batches"] == 2
    assert after["recovery"] == dict(B=ABSENT_RECOVERY, A_return=ABSENT_RECOVERY)


@pytest.mark.parametrize("invalid", [None, 0, 1, "yes", np.bool_(True), [], {}])
def test_should_reject_non_native_stationary_flags_before_array_or_detector_construction(
    monkeypatch, invalid
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid stationary flag reached array or detector construction")

    monkeypatch.setattr(sensor_stream.np, "array", forbidden)
    monkeypatch.setattr(monitoring, "CalibratedShiftDetector", forbidden)
    monkeypatch.setattr(monitoring, "ShiftIncidentTracker", forbidden)
    monkeypatch.setattr(monitoring, "ShiftPolicy", forbidden)
    for function in (make_sensor_stream, SensorMonitoring, monitoring_configuration):
        with pytest.raises(ValueError, match="exact boolean"):
            function(stationary=invalid)
