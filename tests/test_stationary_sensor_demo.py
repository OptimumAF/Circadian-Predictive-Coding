"""Stationary sensor controls through existing streams, with fake native work."""

import json

import numpy as np
import pytest

from scripts.run_sensor_drift_demo import main
from src.adapters import sensor_drift_demo
from src.app.coupled_attribution import AttributionStream
from src.app.sensor_monitoring import SensorMonitoring
from src.app.toy_execution_budget import ToyBudgetSession
import src.app.sensor_monitoring as monitoring_module
import src.core.sensor_stream as fixture_module
from src.core.sensor_stream import make_sensor_stream
from test_sensor_monitoring_integration import FakeLearner
from test_sensor_monitoring_integration import fake_ports as fake_ports
from test_sensor_methods import fake_coupled_math as fake_coupled_math
from test_sensor_replay import FakeReplayLearner
from test_sensor_replay import fake_replay_ports as fake_replay_ports
from test_scheduled_sensor_demo import FakeScheduledLearner
from test_scheduled_sensor_demo import fake_scheduled_ports as fake_scheduled_ports


def forbidden(*args, **kwargs):
    pytest.fail("unselected or native work reached")


@pytest.fixture
def captured_training(monkeypatch):
    captured = []
    original = AttributionStream.__init__

    def initialize(self, learner, training, *args, **kwargs):
        captured.extend(training)
        original(self, learner, training, *args, **kwargs)

    monkeypatch.setattr(AttributionStream, "__init__", initialize)
    return captured


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc"])
def test_should_run_all_ordinary_methods_on_true_prearrival_all_A_data(
    method, request, captured_training
):
    if method == "bp":
        request.getfixturevalue("fake_ports")
        trace = FakeLearner.trace
    else:
        trace = request.getfixturevalue("fake_coupled_math")
    report = sensor_drift_demo.run_demo(method=method, monitoring=True, stationary=True)
    assert report["status"] == "complete", report["reason"]
    assert report["source_unchanged"] is report["actor_unchanged"] is True
    assert len(captured_training) == 12 and {batch.context for batch in captured_training} == {"A"}
    fixture = make_sensor_stream(stationary=True)
    rows = report["rows"]
    predictions = [event for event in trace if event[0] == "predict"]
    trains = [event for event in trace if event[0] == "train"]
    assert len(rows) == len(predictions) == len(trains) == 12
    assert [row["feature_tick"] for row in rows] == list(range(12))
    assert [row["label_tick"] for row in rows] == list(range(2, 14))
    assert [row["arrival_update"] for row in rows] == list(range(1, 13))
    assert [row["prediction_state_updates"] for row in rows] == [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for tick, (prediction, train, batch) in enumerate(
        zip(predictions, trains, fixture, strict=True)
    ):
        np.testing.assert_array_equal(prediction[2], batch.features)
        np.testing.assert_array_equal(train[2 if method == "bp" else 1], batch.features)
        np.testing.assert_array_equal(train[3 if method == "bp" else 2], batch.targets)
        np.testing.assert_array_equal(captured_training[tick].features, batch.features)
    monitor = report["sensor_monitoring"]
    assert monitor["complete"] is True and monitor["native_forward_calls"] == 0
    assert monitor["false_alarms"]["stationary_A"]["monitoring_batches"] == 8
    assert monitor["false_alarms"]["stationary_A"]["alarm_batches"][1] == 0
    assert monitor["false_alarms"]["stationary_A"]["incident_onsets"][1] == 0
    for phase in ("B", "A_return"):
        assert monitor["recovery"][phase]["status"] == "not_applicable"
        assert monitor["recovery"][phase]["labels_to_recovery"] is None
        assert monitor["recovery"][phase]["observed_batches"] == 0
    assert report["diagnostics"]["prediction_calls_completed"] == 12
    assert report["stream"]["evaluation_scope"] == "training_only"
    assert report["stream"]["score_forward_calls"] == 0


def test_should_join_only_retained_prearrival_pixels_after_their_labelled_wake(
    fake_ports, monkeypatch
):
    fixture = make_sensor_stream(stationary=True)
    observed = []
    original = SensorMonitoring.observe

    def observe(self, tick, features, probabilities, targets):
        assert sum(event[0] == "train" for event in FakeLearner.trace) == tick + 1
        np.testing.assert_array_equal(features, fixture[tick].features)
        np.testing.assert_array_equal(targets, fixture[tick].targets)
        observed.append(tick)
        result = original(self, tick, features, probabilities, targets)
        features.fill(-999)
        targets.fill(1)
        probabilities.fill(1)
        return result

    monkeypatch.setattr(SensorMonitoring, "observe", observe)
    report = sensor_drift_demo.run_demo(monitoring=True, stationary=True)
    assert report["status"] == "complete", report["reason"]
    assert observed == list(range(12))
    assert all(row["probabilities"] == [[0.5]] * 4 for row in report["rows"])
    assert all(row["targets"] == [[0.0], [1.0], [0.0], [1.0]] for row in report["rows"])
    assert report["source_unchanged"] is report["actor_unchanged"] is True


def test_should_keep_the_actual_stationary_denominator_and_paid_work_on_monitor_failure(
    fake_ports, monkeypatch
):
    original = SensorMonitoring.observe

    def fail(self, tick, *args):
        if tick == 6:
            raise RuntimeError("injected stationary monitor failure after paid wake")
        return original(self, tick, *args)

    monkeypatch.setattr(SensorMonitoring, "observe", fail)
    report = sensor_drift_demo.run_demo(monitoring=True, stationary=True)
    assert report["status"] == "incomplete" and "after paid wake" in report["reason"]
    assert report["charged_training_updates"] == 7
    assert report["diagnostics"]["prediction_calls_completed"] == 9
    assert len(report["rows"]) == 7
    monitor = report["sensor_monitoring"]
    assert monitor["complete"] is False and monitor["metrics_batches_completed"] == 6
    assert monitor["false_alarms"]["stationary_A"]["monitoring_batches"] == 2
    assert monitor["false_alarms"]["pre_shift_A"]["monitoring_batches"] == 0
    assert monitor["false_alarms"]["post_return_A"]["monitoring_batches"] == 0
    assert len([event for event in FakeLearner.trace if event[0] == "train"]) == 7
    assert report["pending_prediction_batches"] == 2
    assert len(report["rows"]) + report["pending_prediction_batches"] == 9
    assert monitor["recovery"]["B"]["status"] == "not_applicable"


def test_should_keep_twelve_stationary_wake_receipts_and_three_scheduled_updates(
    fake_scheduled_ports, captured_training
):
    report = sensor_drift_demo.run_demo(monitoring=True, scheduled=True, stationary=True)
    assert report["status"] == "complete", report["reason"]
    assert report["source_unchanged"] is report["actor_unchanged"] is True
    assert report["charged_training_updates"] == len(report["stream"]["applied_updates"]) == 12
    assert {batch.context for batch in captured_training} == {"A"}
    state = report["scheduled_training"]
    assert state["optimizer_steps_attempted"] == state["optimizer_steps_completed"] == 3
    assert state["training_rows_attempted"] == state["training_rows_completed"] == 48
    assert state["buffer_bytes"] == state["buffered_batches"] == 0
    optimizer = [event for event in FakeScheduledLearner.trace if event[0] == "optimizer"]
    assert [event[1] for event in optimizer] == [4, 8, 12]
    assert all(event[2].shape == (16, 16) for event in optimizer)
    assert sum(event[0] == "predict" for event in FakeScheduledLearner.trace) == 12
    assert report["sensor_monitoring"]["native_forward_calls"] == 0
    assert report["sensor_monitoring"]["false_alarms"]["stationary_A"]["monitoring_batches"] == 8


def test_should_charge_only_two_distinct_stationary_replay_rows_at_each_existing_boundary(
    fake_replay_ports,
):
    report = sensor_drift_demo.run_demo(monitoring=True, replay=True, stationary=True)
    assert report["status"] == "complete", report["reason"]
    assert report["source_unchanged"] is report["actor_unchanged"] is True
    assert report["charged_training_updates"] == 12
    assert report["replay_budget_charged_exposures"] == 6
    helper = report["rehearsal"]
    assert [event["updates"] for event in helper["events"]] == [4, 8, 12]
    assert [event["charged_exposures"] for event in helper["events"]] == [2, 2, 2]
    assert [event["retention"]["example_count"] for event in helper["events"]] == [2, 2, 2]
    assert (
        helper["attempted_updates"]
        == helper["returned_updates"]
        == helper["committed_updates"]
        == 6
    )
    trains = [event for event in FakeReplayLearner.trace if event[0] == "train"]
    assert [len(event[2]) for event in trains] == [4] * 4 + [1] * 2 + [4] * 4 + [1] * 2 + [
        4
    ] * 4 + [1] * 2
    assert sum(len(event[2]) for event in trains) == 54
    assert sum(event[0] == "predict" for event in FakeReplayLearner.trace) == 12
    assert [row["prediction_replay_updates"] for row in report["rows"]] == [0] * 6 + [2] * 4 + [
        4
    ] * 2
    assert report["protocol"]["replay_allowance"] == 12
    assert report["protocol"]["training_updates"] == 18
    assert report["protocol"]["training_rows"] == 54
    assert report["protocol"]["training_forward_backward_evaluations"] == 36
    assert report["protocol"]["training_evaluated_rows"] == 108


def test_should_skip_every_incident_replay_boundary_without_stationary_feature_onsets(
    fake_replay_ports,
):
    report = sensor_drift_demo.run_demo(
        monitoring=True, replay=True, incident_replay=True, stationary=True
    )
    assert report["status"] == "complete", report["reason"]
    assert report["source_unchanged"] is report["actor_unchanged"] is True
    assert report["charged_training_updates"] == 12
    assert report["replay_budget_charged_exposures"] == 0
    helper = report["rehearsal"]
    assert [event["status"] for event in helper["events"]] == ["skipped"] * 3
    assert (
        helper["charged_exposures"]
        == helper["attempted_updates"]
        == helper["committed_updates"]
        == 0
    )
    assert helper["incident_admission"]["observed_onsets"] == 0
    assert helper["incident_admission"]["admitted_events"] == 0
    assert sum(event[0] == "train" for event in FakeReplayLearner.trace) == 12
    assert sum(event[0] == "predict" for event in FakeReplayLearner.trace) == 12
    assert [row["prediction_replay_updates"] for row in report["rows"]] == [0] * 12


def test_should_keep_the_original_replay_allowance_and_shared_clock_in_stationary_mode(
    fake_replay_ports, monkeypatch
):
    sessions = []
    original = ToyBudgetSession.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        sessions.append(self)

    monkeypatch.setattr(ToyBudgetSession, "__init__", initialize)
    report = sensor_drift_demo.run_demo(monitoring=True, replay=True, stationary=True)
    assert report["status"] == "complete", report["reason"]
    assert len(sessions) == 2
    wake, replay = sessions
    assert wake is not replay and wake.clock is replay.clock
    assert wake.process_rss_sampler is replay.process_rss_sampler
    assert wake.budget.max_wall_seconds == replay.budget.max_wall_seconds == 30
    assert wake.budget.max_training_updates == wake.updates_completed == 12
    assert replay.budget.max_replay_examples == 12 and replay.replay_examples_completed == 6


def test_should_preserve_the_failed_stationary_replay_charge_without_a_retry(fake_replay_ports):
    FakeReplayLearner.fail_replay = True
    report = sensor_drift_demo.run_demo(monitoring=True, replay=True, stationary=True)
    assert report["status"] == "incomplete" and "paid replay failure" in report["reason"]
    assert report["charged_training_updates"] == 4
    assert report["replay_budget_charged_exposures"] == 1
    helper = report["rehearsal"]
    assert helper["stopped"] and helper["attempted_updates"] == helper["charged_exposures"] == 1
    assert helper["returned_updates"] == helper["committed_updates"] == 0
    assert [event["status"] for event in helper["events"]] == ["failed", "not_run", "not_run"]
    assert report["diagnostics"]["prediction_calls_completed"] == 6
    assert len(report["rows"]) == len(report["pending_forecasts"]) == 3
    assert all(
        "features" not in row and "targets" not in row for row in report["pending_forecasts"]
    )
    assert report["sensor_monitoring"]["false_alarms"]["stationary_A"]["monitoring_batches"] == 0


def test_should_preserve_original_monitor_and_fixture_no_keyword_calls_when_false(
    fake_ports, monkeypatch
):
    calls = []
    original_fixture = fixture_module.make_sensor_stream
    original_monitor = monitoring_module.SensorMonitoring

    def fixture(*args, **kwargs):
        calls.append(("fixture", args, kwargs))
        return original_fixture(*args, **kwargs)

    def monitor(*args, **kwargs):
        calls.append(("monitor", args, kwargs))
        return original_monitor(*args, **kwargs)

    monkeypatch.setattr(fixture_module, "make_sensor_stream", fixture)
    monkeypatch.setattr(monitoring_module, "SensorMonitoring", monitor)
    baseline = sensor_drift_demo.run_demo(monitoring=True)
    explicit = sensor_drift_demo.run_demo(monitoring=True, stationary=False)
    for key in ("protocol", "rows", "stream", "sensor_monitoring", "diagnostics"):
        assert baseline[key] == explicit[key]
    assert len(calls) == 4
    assert all(args == () and kwargs == {} for _, args, kwargs in calls)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"monitoring": True},
        {"scheduled": True},
        {"scheduled": True, "monitoring": True},
        {"replay": True},
        {"replay": True, "monitoring": True},
        {"replay": True, "monitoring": True, "incident_replay": True},
        {"method": "pc"},
        {"method": "neutral_cpc", "monitoring": True},
        {"method": "gated_cpc", "monitoring": True, "learnability": True, "heldout": True},
    ],
)
def test_should_keep_previous_descriptors_identical_with_explicit_false(options):
    assert sensor_drift_demo.describe_protocol(**options) == sensor_drift_demo.describe_protocol(
        **options, stationary=False
    )


@pytest.mark.parametrize("entry", [sensor_drift_demo.describe_protocol, sensor_drift_demo.run_demo])
@pytest.mark.parametrize("invalid", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_boolean_stationary_before_model_work(entry, invalid, monkeypatch):
    from src.core import coupled_predictive_coding

    monkeypatch.setattr(coupled_predictive_coding, "CoupledPredictiveCodingNetwork", forbidden)
    with pytest.raises(ValueError, match="bool"):
        entry(monitoring=True, stationary=invalid)


@pytest.mark.parametrize("entry", [sensor_drift_demo.describe_protocol, sensor_drift_demo.run_demo])
@pytest.mark.parametrize(
    "options",
    [{}, {"monitoring": True, "learnability": True}, {"monitoring": True, "heldout": True}],
)
def test_should_refuse_unsupported_stationary_modes_before_data_or_model_work(
    entry, options, monkeypatch
):
    monkeypatch.setattr(fixture_module, "make_sensor_stream", forbidden)
    with pytest.raises(ValueError):
        entry(stationary=True, **options)


@pytest.mark.parametrize(
    "extra,expected",
    [
        ([], {}),
        (["--method", "pc"], {"method": "pc"}),
        (["--method", "neutral_cpc"], {"method": "neutral_cpc"}),
        (["--method", "gated_cpc"], {"method": "gated_cpc"}),
        (["--scheduled"], {"scheduled": True}),
        (["--replay"], {"replay": True}),
        (["--replay", "--incident-replay"], {"replay": True, "incident_replay": True}),
    ],
)
def test_should_forward_only_selected_stationary_cli_modes(extra, expected, monkeypatch, capsys):
    calls = []

    def selected(**options):
        calls.append(options)
        return {"status": "complete"}

    monkeypatch.setattr(sensor_drift_demo, "run_demo", selected)
    assert main(["--stationary", "--monitoring", *extra]) == 0
    assert calls == [dict(monitoring=True, stationary=True, **expected)]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--scheduled"],
        ["--replay"],
        ["--replay", "--incident-replay"],
        ["--method", "pc"],
        ["--method", "neutral_cpc"],
        ["--method", "gated_cpc"],
    ],
)
def test_should_preview_each_stationary_control_without_running_a_model(extra, monkeypatch, capsys):
    monkeypatch.setattr(sensor_drift_demo, "run_demo", forbidden)
    assert main(["--describe", "--stationary", "--monitoring", *extra]) == 0
    descriptor = json.loads(capsys.readouterr().out)
    assert descriptor["diagnostic_forward_calls"] == 12
    assert descriptor["diagnostic_forward_rows"] == 48
    assert descriptor["final_evaluation"] is descriptor["comparison"] is False
    assert descriptor["monitoring"]["stationary"] is True


@pytest.mark.parametrize(
    "flags", [[], ["--monitoring", "--learnability"], ["--monitoring", "--heldout"]]
)
def test_should_refuse_unsupported_stationary_cli_before_dispatch(flags, monkeypatch):
    monkeypatch.setattr(sensor_drift_demo, "describe_protocol", forbidden)
    monkeypatch.setattr(sensor_drift_demo, "run_demo", forbidden)
    with pytest.raises(SystemExit) as error:
        main(["--describe", "--stationary", *flags])
    assert error.value.code == 2
