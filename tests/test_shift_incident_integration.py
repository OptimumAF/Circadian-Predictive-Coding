"""Opt-in incident reporting across observer, existing runner and shift CLI."""

from copy import deepcopy
import json
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_shift_signal_pilot as cli
import src.adapters.shift_signal_pilot as pilot
import src.app.arrived_shift_observer as observer_module
import src.app.toy_execution_budget as budget_module
from src.app.arrived_shift_observer import ArrivedShiftObserver
from src.app.coupled_attribution import AttributionBatch, run_attribution_stream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.learner_ports import TrainingDiagnostic
from src.core.shift_calibration import ShiftCalibration
from src.core.shift_signals import ShiftPolicy
from src.shared.process_memory import ProcessRssSegment


def forbidden(*args, **kwargs):
    pytest.fail("native work or unselected route reached")


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setattr(observer_module, "monotonic", lambda: 0.0)


def batch(index, *, x=0.0, y=0.0, role="train"):
    return AttributionBatch(role, "A", (f"{role}-{index}",), np.array([[x, x]]), np.array([[y]]))


def without_incidents(report):
    result = deepcopy(report)
    result.pop("incident_semantics", None)
    for row in result["rows"]:
        row.pop("incident", None)
    return result


@pytest.mark.parametrize("calibrated", [False, True])
def test_should_preserve_all_old_observations_and_predictions_with_optional_incidents(calibrated):
    reports = []
    traces = []
    for incidents in (False, True):
        calls = []

        def predict(state, features):
            calls.append((state["p"], features.tolist()))
            return np.full((len(features), 1), state["p"])

        options: dict[str, Any] = {}
        if calibrated:
            options.update(calibration=ShiftCalibration(2), calibration_model_id="fixed-fixture")
        observer = ArrivedShiftObserver(
            predict,
            policy=ShiftPolicy(reference_batches=1, window_batches=1, persistence_batches=1),
            max_updates=6,
            incidents=incidents,
            **options,
        )
        observer(0, {"p": 0.1}, None)
        for index, (current, x) in enumerate(
            [(0.1, 0.0), (0.9, 0.0), (0.9, 1.0), (0.1, 1.0), (0.1, 0.0)], 1
        ):
            observer(index, {"p": current}, batch(index, x=x))
        reports.append(observer.report())
        traces.append(calls)
    assert traces[0] == traces[1] and len(traces[0]) == 5
    assert reports[0] == without_incidents(reports[1])
    rows = reports[1]["rows"]
    assert [r["incident"]["states"][2] for r in rows[:5]] == [
        "inactive",
        "inactive",
        "onset",
        "continued",
        "cleared",
    ]
    assert [r["incident"]["onset_batches"][2] for r in rows[:5]] == [None, None, 3, 3, 3]
    assert rows[5]["incident"] is None and rows[5]["status"] == "not_run"
    rows[2]["incident"]["states"] = ("corrupted",) * 3
    assert observer.report()["rows"][2]["incident"]["states"][2] == "onset"


@pytest.mark.parametrize("invalid", [None, 1, "true"])
def test_should_refuse_nonboolean_observer_mode_before_prediction(invalid):
    with pytest.raises(ValueError, match="boolean"):
        ArrivedShiftObserver(forbidden, incidents=invalid)


def test_should_refuse_final_rows_before_incident_or_prediction_work():
    observer = ArrivedShiftObserver(forbidden, incidents=True)
    observer(0, {}, None)
    before = observer.report()
    with pytest.raises(ValueError, match="training"):
        observer(1, {}, batch(1, role="final_test"))
    assert observer.report() == before


def test_should_preserve_partial_incident_and_stop_after_a_postdecision_clock_failure(monkeypatch):
    observer = ArrivedShiftObserver(
        lambda state, x: np.full((len(x), 1), 0.1),
        policy=ShiftPolicy(reference_batches=1, window_batches=1, persistence_batches=1),
        incidents=True,
        max_updates=3,
    )
    observer(0, {}, None)
    observer(1, {}, batch(1))
    times = iter((2.0, 1.0, 3.0))
    monkeypatch.setattr(observer_module, "monotonic", lambda: next(times))
    with pytest.raises(ValueError, match="nondecreasing"):
        observer(2, {}, batch(2, y=1.0))
    report = observer.report()
    row = report["rows"][1]
    assert report["stopped"] and row["status"] == "failed"
    assert row["decision"]["alarms"][2] is True
    assert row["incident"]["states"][2] == "onset"
    assert row["incident"]["onset_batches"][2] == 2
    assert report["rows"][2]["incident"] is None
    with pytest.raises(ValueError, match="stopped"):
        observer(3, {}, batch(3))


@pytest.mark.parametrize("calibrated", [False, True])
def test_should_extend_only_selected_protocol_without_changing_scientific_controls(
    monkeypatch, calibrated
):
    monkeypatch.setattr(pilot, "build_learner", forbidden)
    monkeypatch.setattr(pilot, "build_stream", forbidden)
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", forbidden)
    original = pilot.describe_protocol(calibrated=calibrated)
    selected = pilot.describe_protocol(calibrated=calibrated, incidents=True)
    assert selected["protocol_id"] == original["protocol_id"] + "_incidents_v1"
    for name in original.keys() - {"protocol_id", "planned_work"}:
        assert selected[name] == original[name]
    assert selected["planned_work"] == dict(
        original["planned_work"], incident_steps=144, incident_channel_transitions=432
    )


@pytest.mark.parametrize("invalid", [None, 1, "true"])
def test_should_refuse_nonboolean_pilot_mode_before_sampler_or_native_work(monkeypatch, invalid):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError, match="boolean"):
        pilot.run_pilot(incidents=invalid)


@pytest.mark.parametrize("action", ["describe", "run"])
@pytest.mark.parametrize("calibrated", [False, True])
def test_should_route_explicit_incident_cli_and_only_pass_selected_modes(
    monkeypatch, capsys, action, calibrated
):
    calls = []

    def selected(**options):
        calls.append(options)
        return dict(status="complete", selected="incidents")

    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    monkeypatch.setattr(cli, "describe_protocol" if action == "describe" else "run_pilot", selected)
    args = ["shift", f"--{action}", "--incidents"]
    if calibrated:
        args.append("--calibrated")
    monkeypatch.setattr(sys, "argv", args)
    assert cli.main() == 0
    assert calls == [dict(calibrated=calibrated, incidents=True)]
    assert json.loads(capsys.readouterr().out)["selected"] == "incidents"


def test_should_preserve_all_optional_slots_when_pilot_setup_fails(monkeypatch):
    def fail():
        raise RuntimeError("declared setup failure")

    monkeypatch.setattr(pilot, "ProcessRssSampler", fail)
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", forbidden)
    report = pilot.run_pilot(calibrated=True, incidents=True)
    assert report["status"] == "incomplete" and report["benchmark_completed_updates"] == 0
    assert len(report["runs"]) == 8
    for row in report["runs"]:
        assert row["incident_outcomes"] is None
        assert all(o["incident"] is None for o in row["observations"]["rows"])
        assert row["observations"]["incident_semantics"]["feedback"] is False


class FakeLearner:
    def __init__(self, trace=None, value=0.0):
        self.trace = [] if trace is None else trace
        self.value = value

    def fork(self):
        return FakeLearner(self.trace, self.value)

    def snapshot_state(self):
        return np.array([self.value])

    def restore_state(self, state):
        self.value = float(state[0])

    def train_batch(self, x, y):
        self.value += 1
        self.trace.append(("train", self.value))
        return TrainingDiagnostic("incident_fixture", self.value)

    def predict(self, x):
        self.trace.append(("final", self.value))
        return np.full((len(x), 1), 0.1)


def test_should_keep_existing_runner_benchmark_clocks_and_source_exact_with_incidents():
    training = tuple(batch(n, y=float(7 <= n <= 12)) for n in range(1, 19))
    final = (batch(99, role="final_test"),)
    reports, observations, traces = [], [], []
    for incidents in (False, True):
        source = FakeLearner()

        def predict(state, features):
            source.trace.append(("observe", float(state[0])))
            return np.full((len(features), 1), 0.1)

        observer = ArrivedShiftObserver(
            predict,
            max_updates=18,
            calibration=ShiftCalibration(),
            calibration_model_id="fake-existing-runner",
            incidents=incidents,
        )
        report = run_attribution_stream(
            source,
            training,
            final,
            checkpoint_updates=tuple(range(19)),
            label_delay=2,
            budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=18), lambda: 0.0),
            observe_checkpoint=observer,
        )
        assert source.value == 0.0
        reports.append(report)
        observations.append(observer.report())
        traces.append(source.trace)
    assert reports[0] == reports[1] and traces[0] == traces[1]
    assert observations[0] == without_incidents(observations[1])
    events = [kind for kind, _ in traces[1]]
    assert events[-19:] == ["final"] * 19 and events.count("observe") == 18
    outcomes = pilot._incident_outcomes(observations[1], "recurring")
    assert outcomes["residual"]["onset_updates"] == [8]
    assert outcomes["residual"]["continued_updates"] == [9, 10, 11, 12, 13]
    assert outcomes["residual"]["cleared_updates"] == [14]
    assert outcomes["residual"]["first_return_onset_update"] is None
    assert (
        pilot._signal_outcomes(observations[1], "recurring")["residual"][
            "first_return_alert_update"
        ]
        == 13
    )
    stationary = pilot._incident_outcomes(observations[1], "stationary")
    assert stationary["residual"]["stationary_false_incidents"] == [8]
    observations[1]["rows"][-1]["status"] = "failed"
    assert pilot._incident_outcomes(observations[1], "recurring") is None


def test_should_run_selected_adapter_arm_through_existing_runner_without_native_work(monkeypatch):
    class Sampler:
        start_bytes = 1

        def sample(self):
            return 1

        def snapshot(self):
            return ProcessRssSegment(1, 1, 1, 1, 0.005)

    training = tuple(batch(n, y=float(7 <= n <= 12)) for n in range(1, 19))
    final = (
        batch(99, role="final_test"),
        AttributionBatch("final_test", "B", ("final-B",), np.zeros((1, 2)), np.zeros((1, 1))),
    )
    source = FakeLearner()
    monkeypatch.setattr(pilot, "monotonic", lambda: 0.0)
    monkeypatch.setattr(budget_module, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "build_stream", lambda stream: (training, final))
    monkeypatch.setattr(pilot, "build_learner", lambda method, initial: source)
    monkeypatch.setattr(pilot, "_snapshot_array_bytes", lambda state: state.nbytes)
    monkeypatch.setattr(pilot, "_predict_snapshot", lambda state, x: np.full((len(x), 1), 0.1))
    row = pilot._new_row("bp", "recurring", calibrated=True, incidents=True)
    pilot._run_one(row, None, 60.0, Sampler(), calibrated=True, incidents=True)
    assert row["status"] == "complete", row["reason"]
    assert row["benchmark_completed_updates"] == 18
    assert row["source_learner_unchanged"] and source.value == 0.0
    assert row["incident_outcomes"]["residual"]["onset_updates"] == [8]
    assert row["incident_outcomes"]["residual"]["first_return_onset_update"] is None
    assert row["signal_outcomes"]["residual"]["first_return_alert_update"] == 13
    assert row["observations"]["rows"][12]["incident"]["onset_batches"][2] == 8
