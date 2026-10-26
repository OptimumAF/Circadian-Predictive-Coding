"""Exercise the real delayed stream and recent rehearsal with fake native work.

No learning benefit is asserted. The shape-sensitive native trace distinguishes
four-row arrived wakes from the helper's existing one-row replay calls.
"""

from copy import deepcopy
import json

import numpy as np
import pytest

from scripts.run_sensor_drift_demo import main
from src.adapters import matched_backprop_learner, sensor_drift_demo
from src.app.sensor_monitoring import SensorMonitoring
from src.app.toy_execution_budget import ToyBudgetSession
from src.core import coupled_energy
from src.core.coupled_energy import CoupledParameters
from src.core.learner_ports import TrainingDiagnostic
from src.core.matched_backprop import MatchedBackpropSnapshot
from src.core.sensor_stream import make_sensor_stream


class FakeReplayLearner:
    trace: list = []
    fail_replay = False

    def __init__(self, model=None, *, learning_rate=0.1):
        self.updates = 0

    def fork(self):
        return deepcopy(self)

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
        if len(features) == 1 and self.fail_replay:
            raise RuntimeError("injected paid replay failure")
        self.updates += 1
        return TrainingDiagnostic("fake_sensor_replay_boundary_v1", 0.0)

    def predict(self, features):
        raise AssertionError("prearrival prediction requires the detached shadow snapshot")


@pytest.fixture
def fake_replay_ports(monkeypatch):
    FakeReplayLearner.trace = []
    FakeReplayLearner.fail_replay = False
    monkeypatch.setattr(matched_backprop_learner, "MatchedBackpropLearner", FakeReplayLearner)

    def predict(parameters, features):
        count = int(parameters.output_bias[0, 0])
        FakeReplayLearner.trace.append(("predict", count, features.copy()))
        return np.full((len(features), 1), 0.5 + 0.01 * count)

    monkeypatch.setattr(coupled_energy, "predict_coupled", predict)


@pytest.mark.parametrize("monitoring", [False, True])
def test_should_describe_the_fixed_recent_replay_dose_without_model_work(
    monitoring, monkeypatch, capsys
):
    def refuse(**kwargs):
        raise AssertionError("describe must not run a learner")

    monkeypatch.setattr(sensor_drift_demo, "run_demo", refuse)
    arguments = ["--describe", "--replay"] + (["--monitoring"] if monitoring else [])
    assert main(arguments) == 0
    protocol = json.loads(capsys.readouterr().out)
    assert protocol == sensor_drift_demo.describe_protocol(replay=True, monitoring=monitoring)
    assert protocol["protocol"] == (
        "sensor_brightness_recent_replay_monitoring_bp_v1"
        if monitoring
        else "sensor_brightness_recent_replay_bp_v1"
    )
    assert protocol["training_updates"] == 22 and protocol["training_rows"] == 58
    assert protocol["wake_receipts"] == protocol["wake_optimizer_steps"] == 12
    assert protocol["wake_training_rows"] == 48
    assert protocol["replay_training_updates"] == protocol["replay_training_rows"] == 10
    assert protocol["replay_allowance"] == 12
    assert protocol["training_forward_backward_evaluations"] == 44
    assert protocol["training_evaluated_rows"] == 116
    assert protocol["diagnostic_forward_calls"] == 12
    assert protocol["diagnostic_forward_rows"] == 48
    assert protocol["seed"] == 17 and protocol["learning_rate"] == 0.1
    assert protocol["comparison"] is protocol["final_evaluation"] is False


@pytest.mark.parametrize("monitoring", [False, True])
@pytest.mark.parametrize("scheduled", [False, True])
def test_should_preserve_all_existing_descriptors_when_replay_is_disabled(monitoring, scheduled):
    assert sensor_drift_demo.describe_protocol(
        monitoring=monitoring, scheduled=scheduled
    ) == sensor_drift_demo.describe_protocol(
        monitoring=monitoring, scheduled=scheduled, replay=False
    )


@pytest.mark.parametrize("entry", [sensor_drift_demo.describe_protocol, sensor_drift_demo.run_demo])
@pytest.mark.parametrize("invalid", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_boolean_replay_before_model_work(entry, invalid, monkeypatch):
    from src.core import coupled_predictive_coding

    def refuse(*args, **kwargs):
        raise AssertionError("invalid replay must fail before model construction")

    monkeypatch.setattr(coupled_predictive_coding, "CoupledPredictiveCodingNetwork", refuse)
    with pytest.raises(ValueError, match="bool"):
        entry(replay=invalid)


@pytest.mark.parametrize("entry", [sensor_drift_demo.describe_protocol, sensor_drift_demo.run_demo])
def test_should_refuse_combined_scheduled_replay_before_model_work(entry, monkeypatch):
    from src.core import coupled_predictive_coding

    def refuse(*args, **kwargs):
        raise AssertionError("ambiguous combined algorithm must fail before model construction")

    monkeypatch.setattr(coupled_predictive_coding, "CoupledPredictiveCodingNetwork", refuse)
    with pytest.raises(ValueError, match="scheduled|replay"):
        entry(scheduled=True, replay=True)


def test_should_refuse_combined_cli_modes_before_running_or_describing(monkeypatch):
    def refuse(**kwargs):
        raise AssertionError("incompatible CLI flags must fail before dispatch")

    monkeypatch.setattr(sensor_drift_demo, "run_demo", refuse)
    monkeypatch.setattr(sensor_drift_demo, "describe_protocol", refuse)
    with pytest.raises(SystemExit):
        main(["--scheduled", "--replay"])


@pytest.mark.parametrize("monitoring", [False, True])
def test_should_forward_only_explicit_replay_cli_mode(monitoring, monkeypatch, capsys):
    observed = []

    def run(**kwargs):
        observed.append(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(sensor_drift_demo, "run_demo", run)
    arguments = ["--replay"] + (["--monitoring"] if monitoring else [])
    assert main(arguments) == 0
    assert observed == [{"replay": True, **({"monitoring": True} if monitoring else {})}]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


def test_should_account_separate_wakes_and_per_row_replay_with_original_budgets(
    fake_replay_ports, monkeypatch
):
    sessions = []
    original = ToyBudgetSession.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        sessions.append(self)

    monkeypatch.setattr(ToyBudgetSession, "__init__", initialize)
    report = sensor_drift_demo.run_demo(replay=True)
    assert report["status"] == "complete", report["reason"]
    assert report["source_unchanged"] is report["actor_unchanged"] is True
    assert len(sessions) == 2
    wake, replay = sessions
    assert wake.clock is replay.clock
    assert wake.process_rss_sampler is replay.process_rss_sampler
    assert wake.budget.max_training_updates == 12
    assert wake.budget.max_replay_examples is None
    assert replay.budget.max_replay_examples == 12
    assert wake.budget.max_wall_seconds == replay.budget.max_wall_seconds == 30
    assert wake.budget.max_process_rss_bytes == replay.budget.max_process_rss_bytes == 256 * 1024**2
    assert wake.updates_completed == report["charged_training_updates"] == 12
    assert replay.replay_examples_completed == report["replay_budget_charged_exposures"] == 10
    stream = report["stream"]
    assert stream["completed_updates"] == len(stream["applied_updates"]) == 12
    assert stream["training_examples"] == 48 and stream["score_forward_calls"] == 0
    assert stream["checkpoints"] == [] and stream["evaluation_scope"] == "training_only"
    rehearsal = report["rehearsal"]
    assert rehearsal == stream["rehearsal"]
    assert rehearsal["control"] == "recent_fifo_v1"
    assert rehearsal["policy"] == {
        "every_updates": 4,
        "rows_per_event": 4,
        "retention": {"max_examples": 8, "max_bytes": 1088},
    }
    assert rehearsal["charged_exposures"] == rehearsal["attempted_updates"] == 10
    assert rehearsal["returned_updates"] == rehearsal["committed_updates"] == 10
    events = rehearsal["events"]
    assert [event["updates"] for event in events] == [4, 8, 12]
    assert [event["charged_exposures"] for event in events] == [2, 4, 4]
    assert [event["retention"]["example_count"] for event in events] == [2, 4, 4]
    assert [event["retention"]["retained_bytes"] for event in events] == [272, 544, 544]
    assert all(event["status"] == "complete" and event["receipt"] is not None for event in events)


def test_should_replay_only_content_distinct_arrived_rows_in_the_existing_helper_order(
    fake_replay_ports,
):
    report = sensor_drift_demo.run_demo(replay=True)
    assert report["status"] == "complete", report["reason"]
    fixture = make_sensor_stream()
    trains = [event for event in FakeReplayLearner.trace if event[0] == "train"]
    assert len(trains) == 22
    assert [len(event[2]) for event in trains] == (
        [4] * 4 + [1] * 2 + [4] * 4 + [1] * 4 + [4] * 4 + [1] * 4
    )
    wakes = [event for event in trains if len(event[2]) == 4]
    for tick, event in enumerate(wakes):
        np.testing.assert_array_equal(event[2], fixture[tick].features)
        np.testing.assert_array_equal(event[3], fixture[tick].targets)
    # Each event keeps both A classes; B appears only after its labels arrived.
    replay_rows = [event for event in trains if len(event[2]) == 1]
    expected = [
        (0, 0),
        (0, 1),
        (0, 0),
        (0, 1),
        (4, 0),
        (4, 1),
        (4, 0),
        (4, 1),
        (0, 0),
        (0, 1),
    ]
    for event, (batch, row) in zip(replay_rows, expected, strict=True):
        np.testing.assert_array_equal(event[2], fixture[batch].features[row : row + 1])
        np.testing.assert_array_equal(event[3], fixture[batch].targets[row : row + 1])
    assert sum(len(event[2]) for event in trains) == 58


def test_should_predict_before_same_tick_wake_and_observe_only_already_committed_replay(
    fake_replay_ports,
):
    report = sensor_drift_demo.run_demo(replay=True)
    assert report["status"] == "complete", report["reason"]
    rows = report["rows"]
    wakes = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    replay = [0] * 6 + [2] * 4 + [6] * 2
    assert [row["feature_tick"] for row in rows] == list(range(12))
    assert [row["label_tick"] for row in rows] == list(range(2, 14))
    assert [row["arrival_update"] for row in rows] == list(range(1, 13))
    assert [row["prediction_state_updates"] for row in rows] == wakes
    assert [row["prediction_replay_updates"] for row in rows] == replay
    for row, wake_count, replay_count in zip(rows, wakes, replay, strict=True):
        np.testing.assert_allclose(
            row["probabilities"], np.full((4, 1), 0.5 + 0.01 * (wake_count + replay_count))
        )
    trace = FakeReplayLearner.trace
    assert sum(event[0] == "predict" for event in trace) == 12
    assert report["diagnostics"]["prediction_calls_completed"] == 12
    for tick, index in enumerate(i for i, event in enumerate(trace) if event[0] == "predict"):
        completed = [event for event in trace[:index] if event[0] == "train" and len(event[2]) == 4]
        assert len(completed) == max(0, tick - 2)


def test_should_keep_monitoring_out_of_prediction_replay_and_native_training(fake_replay_ports):
    baseline = sensor_drift_demo.run_demo(replay=True)
    original_trace = FakeReplayLearner.trace.copy()
    FakeReplayLearner.trace = []
    monitored = sensor_drift_demo.run_demo(replay=True, monitoring=True)
    assert baseline["status"] == monitored["status"] == "complete"
    assert baseline["rows"] == monitored["rows"]
    assert baseline["stream"] == monitored["stream"]
    assert baseline["rehearsal"] == monitored["rehearsal"]
    assert monitored["sensor_monitoring"]["complete"] is True
    assert monitored["sensor_monitoring"]["native_forward_calls"] == 0
    assert len(original_trace) == len(FakeReplayLearner.trace) == 34
    for left, right in zip(original_trace, FakeReplayLearner.trace, strict=True):
        assert left[:2] == right[:2]
        for a, b in zip(left[2:], right[2:], strict=True):
            np.testing.assert_array_equal(a, b)


def test_should_preserve_paid_replay_attempt_and_original_wakes_on_native_failure(
    fake_replay_ports,
):
    FakeReplayLearner.fail_replay = True
    report = sensor_drift_demo.run_demo(replay=True)
    assert report["status"] == "incomplete" and "paid replay failure" in report["reason"]
    assert report["charged_training_updates"] == 4
    assert report["replay_budget_charged_exposures"] == 1
    rehearsal = report["rehearsal"]
    assert rehearsal["stopped"] is True
    assert rehearsal["charged_exposures"] == rehearsal["attempted_updates"] == 1
    assert rehearsal["returned_updates"] == rehearsal["committed_updates"] == 0
    event = rehearsal["events"][0]
    assert event["status"] == "failed" and event["receipt"] is None
    assert event["charged_exposures"] == event["attempted_updates"] == 1
    assert event["returned_updates"] == event["committed_updates"] == 0
    assert all(event["status"] == "not_run" for event in rehearsal["events"][1:])
    assert report["diagnostics"]["prediction_calls_completed"] == 6
    assert len(report["rows"]) == 3
    assert [row["feature_tick"] for row in report["pending_forecasts"]] == [3, 4, 5]
    assert len(report["rows"]) + len(report["pending_forecasts"]) == 6
    assert all(
        "targets" not in row and "features" not in row for row in report["pending_forecasts"]
    )
    assert all(len(row["probabilities"]) == 4 for row in report["pending_forecasts"])
    assert len([event for event in FakeReplayLearner.trace if event[0] == "train"]) == 5


def test_should_isolate_monitor_copies_from_predictions_labels_and_replay(
    fake_replay_ports, monkeypatch
):
    original = SensorMonitoring.observe

    def observe(self, tick, features, probabilities, targets):
        result = original(self, tick, features, probabilities, targets)
        features.fill(-999)
        probabilities.fill(1)
        targets.fill(1)
        return result

    monkeypatch.setattr(SensorMonitoring, "observe", observe)
    report = sensor_drift_demo.run_demo(replay=True, monitoring=True)
    assert report["status"] == "complete", report["reason"]
    assert report["source_unchanged"] is report["actor_unchanged"] is True
    assert report["rehearsal"]["committed_updates"] == 10
    assert all(row["targets"] == [[0.0], [1.0], [0.0], [1.0]] for row in report["rows"])
    assert report["rows"][0]["probabilities"] == [[0.5]] * 4
    fixture = make_sensor_stream()
    for event in FakeReplayLearner.trace:
        if event[0] == "train":
            for row, target in zip(event[2], event[3], strict=True):
                assert any(
                    np.array_equal(row, batch.features[index])
                    and np.array_equal(target, batch.targets[index])
                    for batch in fixture
                    for index in range(4)
                )


def test_should_isolate_predictor_snapshot_and_feature_mutation_from_native_work(
    fake_replay_ports, monkeypatch
):
    def mutate(parameters, features):
        count = int(parameters.output_bias[0, 0])
        FakeReplayLearner.trace.append(("predict", count, features.copy()))
        parameters.output_bias.fill(999)
        parameters.hidden_weights[0].fill(-999)
        features.fill(-999)
        return np.full((len(features), 1), 0.5 + 0.01 * count)

    monkeypatch.setattr(coupled_energy, "predict_coupled", mutate)
    report = sensor_drift_demo.run_demo(replay=True)
    assert report["status"] == "complete", report["reason"]
    assert report["source_unchanged"] is report["actor_unchanged"] is True
    assert report["rehearsal"]["committed_updates"] == 10
    predictions = [event for event in FakeReplayLearner.trace if event[0] == "predict"]
    assert [event[1] for event in predictions] == [0, 0, 0, 1, 2, 3, 6, 7, 8, 9, 14, 15]
    fixture = make_sensor_stream()
    for tick, event in enumerate(predictions):
        np.testing.assert_array_equal(event[2], fixture[tick].features)
    wakes = [
        event for event in FakeReplayLearner.trace if event[0] == "train" and len(event[2]) == 4
    ]
    for tick, event in enumerate(wakes):
        np.testing.assert_array_equal(event[2], fixture[tick].features)
        np.testing.assert_array_equal(event[3], fixture[tick].targets)
