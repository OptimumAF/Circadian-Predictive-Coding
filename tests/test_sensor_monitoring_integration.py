"""Execute the ordinary demo/stream with fake native ports, without learning claims."""

import numpy as np
import pytest

from src.adapters import matched_backprop_learner, sensor_drift_demo
from src.app.sensor_monitoring import SensorMonitoring
from src.core import coupled_energy
from src.core.coupled_energy import CoupledParameters
from src.core.learner_ports import TrainingDiagnostic
from src.core.matched_backprop import MatchedBackpropSnapshot
from src.core.sensor_stream import make_sensor_stream


class FakeLearner:
    trace: list = []

    def __init__(self, model=None, *, learning_rate=0.1):
        self.updates = 0

    def fork(self):
        learner = FakeLearner()
        learner.updates = self.updates
        return learner

    def snapshot_state(self):
        parameters = CoupledParameters(
            (np.zeros((16, 8)),),
            (np.zeros((1, 8)),),
            np.zeros((8, 1)),
            np.array([[float(self.updates)]]),
        )
        return MatchedBackpropSnapshot(1, parameters)

    def restore_state(self, snapshot):
        self.updates = int(snapshot.parameters.output_bias[0, 0])

    def train_batch(self, features, targets):
        self.trace.append(("train", self.updates, features.copy(), targets.copy()))
        self.updates += 1
        return TrainingDiagnostic("fake_sensor_boundary_v1", 0.0)

    def predict(self, features):
        raise AssertionError("sensor shadow observation must use its detached snapshot")


@pytest.fixture
def fake_ports(monkeypatch):
    FakeLearner.trace = []
    monkeypatch.setattr(matched_backprop_learner, "MatchedBackpropLearner", FakeLearner)

    def predict(parameters, features):
        FakeLearner.trace.append(("predict", int(parameters.output_bias[0, 0]), features.copy()))
        return np.full((len(features), 1), 0.5)

    monkeypatch.setattr(coupled_energy, "predict_coupled", predict)


def test_should_add_monitoring_without_extra_predictions_or_changing_native_training(fake_ports):
    baseline = sensor_drift_demo.run_demo()
    original_trace = FakeLearner.trace.copy()
    FakeLearner.trace = []
    monitored = sensor_drift_demo.run_demo(monitoring=True)

    assert baseline["status"] == monitored["status"] == "complete"
    assert "sensor_monitoring" not in baseline
    assert baseline["rows"] == monitored["rows"]
    assert baseline["stream"] == monitored["stream"]
    assert monitored["source_unchanged"] is monitored["actor_unchanged"] is True
    assert monitored["sensor_monitoring"]["complete"] is True
    assert monitored["diagnostics"]["pending_feature_peak_bytes"] == 3 * 4 * 16 * 8
    assert monitored["sensor_monitoring"]["native_forward_calls"] == 0
    assert len(original_trace) == len(FakeLearner.trace) == 24
    for left, right in zip(original_trace, FakeLearner.trace, strict=True):
        assert left[:2] == right[:2]
        for a, b in zip(left[2:], right[2:], strict=True):
            np.testing.assert_array_equal(a, b)


def test_should_join_original_prearrival_pixels_only_after_the_actual_labelled_wake(
    fake_ports, monkeypatch
):
    observed = []
    original = SensorMonitoring.observe
    fixture = make_sensor_stream()

    def observe(self, tick, features, probabilities, targets):
        assert len([e for e in FakeLearner.trace if e[0] == "train"]) == tick + 1
        np.testing.assert_array_equal(features, fixture[tick].features)
        np.testing.assert_array_equal(targets, fixture[tick].targets)
        observed.append(tick)
        result = original(self, tick, features, probabilities, targets)
        features.fill(-999)
        probabilities.fill(1)
        targets.fill(1)
        return result

    monkeypatch.setattr(SensorMonitoring, "observe", observe)
    report = sensor_drift_demo.run_demo(monitoring=True)
    assert report["status"] == "complete" and observed == list(range(12))
    # Observer mutations must not change the retained prediction or arrived score.
    assert all(row["probabilities"] == [[0.5]] * 4 for row in report["rows"])
    assert all(row["targets"] == [[0.0], [1.0], [0.0], [1.0]] for row in report["rows"])


def test_should_keep_paid_wake_and_diagnostic_counts_when_monitoring_fails(fake_ports, monkeypatch):
    original = SensorMonitoring.observe

    def fail(self, tick, features, probabilities, targets):
        if tick == 2:
            raise RuntimeError("sensor monitoring failed after paid wake")
        return original(self, tick, features, probabilities, targets)

    monkeypatch.setattr(SensorMonitoring, "observe", fail)
    report = sensor_drift_demo.run_demo(monitoring=True)
    assert report["status"] == "incomplete" and "after paid wake" in report["reason"]
    assert report["charged_training_updates"] == 3
    assert report["diagnostics"]["prediction_calls_completed"] == 5
    assert report["sensor_monitoring"]["metrics_batches_completed"] == 2
    assert len(report["rows"]) == 3
    assert report["sensor_monitoring"]["complete"] is False
    assert len([e for e in FakeLearner.trace if e[0] == "train"]) == 3
