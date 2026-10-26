"""Scheduled sensor CLI and real delayed-stream behavior with fake native work.

The fake preserves the complete scheduled snapshot and makes no learning claim.
These tests cover orchestration; native BP mathematics is checked separately.
"""

from copy import deepcopy
import json

import numpy as np
import pytest

from scripts.run_sensor_drift_demo import main
from src.adapters import sensor_drift_demo
from src.app.sensor_monitoring import SensorMonitoring
from src.core import coupled_energy
from src.core.coupled_energy import CoupledParameters
from src.core.learner_ports import TrainingDiagnostic
from src.core.matched_backprop import MatchedBackpropSnapshot
from src.core.sensor_stream import make_sensor_stream


@pytest.mark.parametrize("monitoring", [False, True])
def test_should_describe_fixed_scheduled_dose_without_running_a_learner(
    monkeypatch, capsys, monitoring
):
    def refuse(**kwargs):
        raise AssertionError("describe must not execute the learner")

    monkeypatch.setattr(sensor_drift_demo, "run_demo", refuse)
    arguments = ["--describe", "--scheduled"] + (["--monitoring"] if monitoring else [])
    assert main(arguments) == 0
    protocol = json.loads(capsys.readouterr().out)
    assert protocol == sensor_drift_demo.describe_protocol(scheduled=True, monitoring=monitoring)
    assert protocol["protocol"] == (
        "sensor_brightness_scheduled_monitoring_bp_v1"
        if monitoring
        else "sensor_brightness_scheduled_bp_v1"
    )
    assert protocol["method"] == "scheduled_matched_bp"
    assert protocol["training_updates"] == 3 and protocol["wake_receipts"] == 12
    assert protocol["training_rows"] == 48
    assert protocol["training_forward_backward_evaluations"] == 6
    assert protocol["training_evaluated_rows"] == 96
    assert protocol["diagnostic_forward_calls"] == 12
    assert protocol["diagnostic_forward_rows"] == 48
    assert protocol["seed"] == 17 and protocol["learning_rate"] == 0.1
    assert protocol["label_delay"] == 2 and protocol["rows_per_batch"] == 4
    assert protocol["phases"] == ["A", "brightness_plus_0.25", "A_return"]
    assert protocol["comparison"] is protocol["final_evaluation"] is False


@pytest.mark.parametrize("monitoring", [False, True])
def test_should_leave_existing_every_arrival_descriptors_unchanged(monitoring):
    implicit = sensor_drift_demo.describe_protocol(monitoring=monitoring)
    explicit = sensor_drift_demo.describe_protocol(monitoring=monitoring, scheduled=False)
    assert implicit == explicit
    assert implicit["method"] == "matched_bp" and implicit["training_updates"] == 12
    assert implicit["training_forward_backward_evaluations"] == 24
    assert implicit["training_rows"] == 48 and "wake_receipts" not in implicit


@pytest.mark.parametrize("entry", [sensor_drift_demo.describe_protocol, sensor_drift_demo.run_demo])
@pytest.mark.parametrize("invalid", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_boolean_schedule_before_model_work(entry, invalid, monkeypatch):
    from src.core import coupled_predictive_coding

    def refuse(*args, **kwargs):
        raise AssertionError("invalid schedule must be rejected before model construction")

    monkeypatch.setattr(coupled_predictive_coding, "CoupledPredictiveCodingNetwork", refuse)
    with pytest.raises(ValueError, match="bool"):
        entry(scheduled=invalid)


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], {}),
        (["--monitoring"], {"monitoring": True}),
        (["--scheduled"], {"scheduled": True}),
        (["--scheduled", "--monitoring"], {"scheduled": True, "monitoring": True}),
    ],
)
def test_should_forward_only_explicit_cli_modes(arguments, expected, monkeypatch, capsys):
    observed = []

    def run(**kwargs):
        observed.append(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(sensor_drift_demo, "run_demo", run)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "7")
    assert main(arguments) == 0
    assert observed == [expected]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


class FakeScheduledLearner:
    """Exercise ordinary stream leases, delayed labels and complete snapshot copies."""

    trace: list = []

    def __init__(self, model=None, *, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.arrived = 0
        self.steps = 0
        self.buffer = []
        self.peak = 0

    def fork(self):
        return deepcopy(self)

    def snapshot_state(self):
        from src.adapters.scheduled_backprop_learner import ScheduledBackpropSnapshot

        parameters = CoupledParameters(
            (np.zeros((16, 8)),),
            (np.zeros((1, 8)),),
            np.zeros((8, 1)),
            np.array([[float(self.steps)]]),
        )
        return ScheduledBackpropSnapshot(
            format_version=1,
            native=MatchedBackpropSnapshot(1, parameters),
            learning_rate=self.learning_rate,
            arrived_batches=self.arrived,
            optimizer_steps_attempted=self.steps,
            optimizer_steps_completed=self.steps,
            buffered_batches=tuple((x.copy(), y.copy()) for x, y in self.buffer),
            peak_buffer_bytes=self.peak,
            stopped=False,
        )

    def restore_state(self, snapshot):
        self.learning_rate = snapshot.learning_rate
        self.arrived = snapshot.arrived_batches
        self.steps = snapshot.optimizer_steps_completed
        self.buffer = [(x.copy(), y.copy()) for x, y in snapshot.buffered_batches]
        self.peak = snapshot.peak_buffer_bytes

    def train_batch(self, features, targets):
        self.trace.append(("wake", self.arrived, features.copy(), targets.copy()))
        self.arrived += 1
        self.buffer.append((features.copy(), targets.copy()))
        self.peak = max(self.peak, sum(x.nbytes + y.nbytes for x, y in self.buffer))
        if len(self.buffer) == 4:
            features = np.concatenate([x for x, _ in self.buffer])
            targets = np.concatenate([y for _, y in self.buffer])
            self.trace.append(("optimizer", self.arrived, features, targets))
            self.steps += 1
            self.buffer = []
        return TrainingDiagnostic("fake_scheduled_sensor_boundary_v1", 0.0)

    def predict(self, features):
        raise AssertionError("prearrival prediction must use its detached native snapshot")

    def report(self):
        return dict(
            arrived_batches=self.arrived,
            optimizer_steps_attempted=self.steps,
            optimizer_steps_completed=self.steps,
            training_rows_attempted=self.steps * 16,
            training_rows_completed=self.steps * 16,
            buffered_batches=len(self.buffer),
            buffer_bytes=sum(x.nbytes + y.nbytes for x, y in self.buffer),
            peak_buffer_bytes=self.peak,
            stopped=False,
        )


@pytest.fixture
def fake_scheduled_ports(monkeypatch):
    from src.adapters import scheduled_backprop_learner

    FakeScheduledLearner.trace = []
    monkeypatch.setattr(
        scheduled_backprop_learner, "ScheduledBackpropLearner", FakeScheduledLearner
    )

    def predict(parameters, features):
        steps = int(parameters.output_bias[0, 0])
        FakeScheduledLearner.trace.append(("predict", steps, features.copy()))
        # Encode the visible native state without executing or claiming learning.
        return np.full((len(features), 1), 0.2 + 0.1 * steps)

    monkeypatch.setattr(coupled_energy, "predict_coupled", predict)


def test_should_predict_before_labels_and_separate_twelve_wakes_from_three_steps(
    fake_scheduled_ports,
):
    report = sensor_drift_demo.run_demo(scheduled=True)
    assert report["status"] == "complete", report["reason"]
    rows = report["rows"]
    assert [r["feature_tick"] for r in rows] == list(range(12))
    assert [r["label_tick"] for r in rows] == list(range(2, 14))
    assert [r["arrival_update"] for r in rows] == list(range(1, 13))
    assert [r["prediction_state_updates"] for r in rows] == [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    steps = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    assert [r["prediction_optimizer_steps"] for r in rows] == steps
    for row, completed in zip(rows, steps, strict=True):
        np.testing.assert_allclose(row["probabilities"], np.full((4, 1), 0.2 + 0.1 * completed))
    assert [r[1] for r in FakeScheduledLearner.trace if r[0] == "optimizer"] == [4, 8, 12]
    assert report["charged_training_updates"] == report["stream"]["completed_updates"] == 12
    assert len(report["stream"]["applied_updates"]) == 12
    assert report["stream"]["training_examples"] == 48
    assert report["stream"]["score_forward_calls"] == 0
    assert report["stream"]["checkpoints"] == []
    assert report["stream"]["evaluation_scope"] == "training_only"
    assert report["source_unchanged"] is report["actor_unchanged"] is True
    state = report["scheduled_training"]
    assert state["arrived_batches"] == 12
    assert state["optimizer_steps_attempted"] == state["optimizer_steps_completed"] == 3
    assert state["training_rows_attempted"] == state["training_rows_completed"] == 48
    assert state["buffered_batches"] == state["buffer_bytes"] == 0
    assert state["peak_buffer_bytes"] == 2176 and state["stopped"] is False
    assert report["diagnostics"]["prediction_calls_completed"] == 12


def test_should_train_each_original_arrived_row_once_in_four_batch_windows(fake_scheduled_ports):
    report = sensor_drift_demo.run_demo(scheduled=True)
    assert report["status"] == "complete", report["reason"]
    fixture = make_sensor_stream()
    wakes = [r for r in FakeScheduledLearner.trace if r[0] == "wake"]
    optimizer = [r for r in FakeScheduledLearner.trace if r[0] == "optimizer"]
    assert len(wakes) == 12 and len(optimizer) == 3
    for tick, wake in enumerate(wakes):
        assert wake[1] == tick
        np.testing.assert_array_equal(wake[2], fixture[tick].features)
        np.testing.assert_array_equal(wake[3], fixture[tick].targets)
    for window, step in enumerate(optimizer):
        batches = fixture[window * 4 : (window + 1) * 4]
        np.testing.assert_array_equal(step[2], np.concatenate([b.features for b in batches]))
        np.testing.assert_array_equal(step[3], np.concatenate([b.targets for b in batches]))


def test_should_add_monitoring_without_extra_work_or_schedule_feedback(fake_scheduled_ports):
    baseline = sensor_drift_demo.run_demo(scheduled=True)
    original_trace = FakeScheduledLearner.trace.copy()
    FakeScheduledLearner.trace = []
    monitored = sensor_drift_demo.run_demo(scheduled=True, monitoring=True)
    assert baseline["status"] == monitored["status"] == "complete"
    assert baseline["rows"] == monitored["rows"]
    assert baseline["stream"] == monitored["stream"]
    assert baseline["scheduled_training"] == monitored["scheduled_training"]
    assert len(original_trace) == len(FakeScheduledLearner.trace) == 27
    for left, right in zip(original_trace, FakeScheduledLearner.trace, strict=True):
        assert left[:2] == right[:2]
        for a, b in zip(left[2:], right[2:], strict=True):
            np.testing.assert_array_equal(a, b)
    monitoring = monitored["sensor_monitoring"]
    assert monitoring["complete"] is True and monitoring["native_forward_calls"] == 0
    assert monitoring["false_alarms"]["pre_shift_A"]["monitoring_batches"] == 0
    assert monitoring["false_alarms"]["post_return_A"]["alarm_batches"][1] == 1
    assert monitored["diagnostics"]["pending_feature_peak_bytes"] == 3 * 4 * 16 * 8


def test_should_isolate_monitor_observer_copies_from_paid_predictions_and_training(
    fake_scheduled_ports, monkeypatch
):
    original = SensorMonitoring.observe

    def observe(self, tick, features, probabilities, targets):
        assert len([r for r in FakeScheduledLearner.trace if r[0] == "wake"]) == tick + 1
        result = original(self, tick, features, probabilities, targets)
        features.fill(-999)
        probabilities.fill(1)
        targets.fill(1)
        return result

    monkeypatch.setattr(SensorMonitoring, "observe", observe)
    report = sensor_drift_demo.run_demo(scheduled=True, monitoring=True)
    assert report["status"] == "complete", report["reason"]
    assert all(row["targets"] == [[0.0], [1.0], [0.0], [1.0]] for row in report["rows"])
    assert report["scheduled_training"]["training_rows_completed"] == 48
    assert report["rows"][0]["probabilities"] == [[0.2]] * 4
    assert report["rows"][10]["probabilities"] == [[0.4]] * 4


def test_should_retain_paid_wakes_native_step_and_forecasts_after_observer_failure(
    fake_scheduled_ports, monkeypatch
):
    original = SensorMonitoring.observe

    def observe(self, tick, features, probabilities, targets):
        if tick == 4:
            raise RuntimeError("scheduled monitoring failed after paid wake")
        return original(self, tick, features, probabilities, targets)

    monkeypatch.setattr(SensorMonitoring, "observe", observe)
    report = sensor_drift_demo.run_demo(scheduled=True, monitoring=True)
    assert report["status"] == "incomplete" and "after paid wake" in report["reason"]
    assert report["charged_training_updates"] == 5
    assert report["scheduled_training"]["arrived_batches"] == 5
    assert report["scheduled_training"]["optimizer_steps_completed"] == 1
    assert report["scheduled_training"]["training_rows_completed"] == 16
    assert report["scheduled_training"]["buffered_batches"] == 1
    assert report["diagnostics"]["prediction_calls_completed"] == 7
    assert report["sensor_monitoring"]["metrics_batches_completed"] == 4
    assert report["sensor_monitoring"]["complete"] is False
    assert len(report["rows"]) == 5
    assert [r[1] for r in FakeScheduledLearner.trace if r[0] == "optimizer"] == [4]
