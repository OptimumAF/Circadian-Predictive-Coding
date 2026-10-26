"""Incident feedback through the existing shift runner, with fake native work.

The real frozen data, observer, detector, delayed stream and rehearsal compose
here. A shared trace distinguishes paid wakes/replay from finish-only scoring.
"""

from dataclasses import replace
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.run_shift_signal_pilot as cli
import src.adapters.shift_signal_pilot as pilot
import src.app.arrived_shift_observer as observer_module
import src.app.toy_execution_budget as budget_module
from src.app.arrived_shift_observer import ArrivedShiftObserver
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.coupled_energy import CoupledParameters
from src.core.learner_ports import TrainingDiagnostic
from src.shared.process_memory import ProcessRssSegment


def forbidden(*args, **kwargs):
    pytest.fail("native work or unselected route reached")


class FakeLearner:
    def __init__(self, trace=None, value=0, *, fail_replay=False):
        self.trace = [] if trace is None else trace
        self.value = value
        self.fail_replay = fail_replay

    def fork(self):
        return FakeLearner(self.trace, self.value, fail_replay=self.fail_replay)

    def snapshot_state(self):
        return np.array([float(self.value)])

    def restore_state(self, state):
        self.value = int(state[0])

    def train_batch(self, features, targets):
        self.trace.append(("train", self.value, features.copy(), targets.copy()))
        if len(features) == 1 and self.fail_replay:
            raise RuntimeError("injected paid replay failure")
        self.value += 1
        return TrainingDiagnostic("fake_shift_replay_v1", 0.0)

    def predict(self, features):
        self.trace.append(("final", self.value, features.copy()))
        return np.full((len(features), 1), 0.25)


class FakeSampler:
    start_bytes = 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def sample(self):
        return 1

    def snapshot(self):
        return ProcessRssSegment(1, 1, 1, 1, 0.005)


@pytest.fixture
def fake_ports(monkeypatch):
    sources = []

    def build(method, initial):
        source = FakeLearner()
        sources.append(source)
        return source

    def predict(state, features):
        sources[-1].trace.append(("observe", int(state[0]), features.copy()))
        return np.full((len(features), 1), 0.25)

    monkeypatch.setattr(pilot, "build_learner", build)
    monkeypatch.setattr(pilot, "_predict_snapshot", predict)
    monkeypatch.setattr(pilot, "_snapshot_array_bytes", lambda state: state.nbytes)
    monkeypatch.setattr(pilot, "monotonic", lambda: 0.0)
    monkeypatch.setattr(observer_module, "monotonic", lambda: 0.0)
    monkeypatch.setattr(budget_module, "ProcessRssSampler", FakeSampler)
    monkeypatch.setattr(pilot, "ProcessRssSampler", FakeSampler)
    return sources


def run_one(method="bp", stream="recurring", *, incident_replay=True):
    row = pilot._new_row(method, stream, calibrated=True, incidents=True)
    pilot._run_one(
        row,
        None,
        60.0,
        FakeSampler(),
        calibrated=True,
        incidents=True,
        incident_replay=incident_replay,
    )
    return row


@pytest.mark.parametrize("stream", ["stationary", "recurring"])
@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc"])
def test_should_admit_only_the_future_boundary_after_real_feature_onsets(
    fake_ports, stream, method
):
    row = run_one(method, stream)
    assert row["status"] == "complete", row["reason"]
    assert row["source_learner_unchanged"] and fake_ports[0].value == 0
    observations = row["observations"]
    assert observations["incident_semantics"]["feedback"] is False
    assert row["replay_policy_feedback"] is True
    assert row["incident_outcomes"]["feature"]["onset_updates"] == [8, 11]
    if stream == "stationary":
        assert row["incident_outcomes"]["feature"]["stationary_false_incidents"] == [8, 11]
    else:
        assert row["incident_outcomes"]["feature"]["first_return_onset_update"] is None
    rows = observations["rows"]
    assert rows[8]["incident"]["states"][1] == "cleared"
    assert all(r["incident"]["states"][1] == "continued" for r in rows[11:15])
    helper = row["rehearsal"]
    assert [e["updates"] for e in helper["events"]] == [4, 8, 12, 16]
    assert [e["status"] for e in helper["events"]] == ["skipped", "skipped", "complete", "skipped"]
    assert [e["charged_exposures"] for e in helper["events"]] == [0, 0, 4, 0]
    assert helper["events"][2]["incident_admission"] == dict(
        admitted=True,
        channel="feature",
        onset_batch=11,
        observed_updates=11,
        boundary_updates=12,
        reason="active_prior_onset",
    )
    assert helper["incident_admission"]["expired_onsets"] == 1
    assert helper["incident_admission"]["admitted_events"] == 1
    assert helper["incident_admission"]["last_consumed_onset"] == 11
    assert helper["policy"]["retention"] == {"max_examples": 8, "max_bytes": 192}
    assert helper["charged_exposures"] == helper["attempted_updates"] == 4
    assert helper["returned_updates"] == helper["committed_updates"] == 4
    assert row["benchmark_completed_updates"] == 18
    assert row["replay_budget_charged_exposures"] == 4
    assert row["stream_result"]["training_examples"] == 144
    assert row["stream_result"]["score_forward_calls"] == 38
    assert [r["prediction_state_updates"] for r in rows] == list(range(18))
    assert [r["prediction_replay_updates"] for r in rows] == [0] * 12 + [4] * 6
    trace = fake_ports[0].trace
    assert [e[1] for e in trace if e[0] == "observe"] == list(range(12)) + list(range(16, 22))
    trains = [e for e in trace if e[0] == "train"]
    assert [len(e[2]) for e in trains] == [8] * 12 + [1] * 4 + [8] * 6
    original, _ = pilot.build_stream(stream)
    wakes = [e for e in trains if len(e[2]) == 8]
    for event, batch in zip(wakes, original, strict=True):
        np.testing.assert_array_equal(event[2], batch.features)
        np.testing.assert_array_equal(event[3], batch.targets)
    arrived_pairs = {
        (tuple(x), float(y[0]))
        for batch in original[:12]
        for x, y in zip(batch.features, batch.targets, strict=True)
    }
    assert all(
        (tuple(e[2][0]), float(e[3][0, 0])) in arrived_pairs for e in trains if len(e[2]) == 1
    )
    assert [e[0] for e in trace[-38:]] == ["final"] * 38
    assert all(len(e[2]) == 64 for e in trace[-38:])
    before_final = trace[:-38]
    assert sum(e[0] == "observe" for e in before_final) == 18
    assert sum(e[0] == "train" for e in before_final) == 22


def test_should_keep_original_distinct_budget_sessions_and_shared_clock(fake_ports, monkeypatch):
    sessions = []
    initialize = ToyBudgetSession.__init__

    def record(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        sessions.append(self)

    monkeypatch.setattr(ToyBudgetSession, "__init__", record)
    row = run_one()
    assert row["status"] == "complete", row["reason"]
    assert len(sessions) == 2
    wake, replay = sessions
    assert wake.clock is replay.clock
    assert wake.process_rss_sampler is replay.process_rss_sampler
    assert wake.started_at == replay.started_at == 0.0
    assert wake.budget.max_wall_seconds == replay.budget.max_wall_seconds == 60.0
    assert wake.budget.max_training_updates == wake.updates_completed == 18
    assert wake.budget.max_replay_examples is None
    assert replay.budget.max_replay_examples == replay.replay_examples_completed == 4
    assert wake.budget.max_process_rss_bytes == replay.budget.max_process_rss_bytes


def test_should_skip_replay_on_a_genuinely_flat_stationary_feature_control(fake_ports, monkeypatch):
    training, final = pilot.build_stream("stationary")
    flat = tuple(replace(batch, features=np.zeros((8, 2))) for batch in training)
    monkeypatch.setattr(pilot, "build_stream", lambda stream: (flat, final))
    row = run_one(stream="stationary")
    assert row["status"] == "complete", row["reason"]
    assert row["incident_outcomes"]["feature"]["onset_updates"] == []
    assert all(e["status"] == "skipped" for e in row["rehearsal"]["events"])
    assert row["replay_budget_charged_exposures"] == 0
    assert row["rehearsal"]["attempted_updates"] == 0
    assert len([e for e in fake_ports[0].trace if e[0] == "train"]) == 18


def test_should_preserve_a_failed_paid_replay_without_final_scoring(fake_ports, monkeypatch):
    source = FakeLearner(fail_replay=True)
    monkeypatch.setattr(pilot, "build_learner", lambda method, initial: source)
    monkeypatch.setattr(pilot, "_predict_snapshot", lambda state, x: np.full((len(x), 1), 0.25))
    row = run_one()
    assert row["status"] == "failed" and "paid replay failure" in row["reason"]
    assert source.value == 0 and row["benchmark_completed_updates"] == 12
    assert row["replay_budget_charged_exposures"] == 1
    helper = row["rehearsal"]
    assert helper["stopped"] and helper["charged_exposures"] == helper["attempted_updates"] == 1
    assert helper["returned_updates"] == helper["committed_updates"] == 0
    assert helper["events"][2]["status"] == "failed"
    assert helper["incident_admission"]["last_consumed_onset"] == 11
    assert helper["events"][3]["status"] == "not_run"
    assert row["observations"]["prediction_calls_completed"] == 11
    assert all(e[0] != "final" for e in source.trace)
    assert len([e for e in source.trace if e[0] == "train"]) == 13


def test_should_preserve_committed_replay_when_a_later_observation_fails(fake_ports, monkeypatch):
    count = [0]

    def fail(state, features):
        count[0] += 1
        if count[0] == 13:
            raise RuntimeError("injected arrived prediction failure")
        return np.full((len(features), 1), 0.25)

    monkeypatch.setattr(pilot, "_predict_snapshot", fail)
    row = run_one()
    assert row["status"] == "failed" and "arrived prediction failure" in row["reason"]
    assert row["benchmark_completed_updates"] == 13
    assert row["replay_budget_charged_exposures"] == 4
    assert row["rehearsal"]["committed_updates"] == 4
    observation = row["observations"]
    assert observation["prediction_calls_attempted"] == 13
    assert observation["prediction_calls_completed"] == 12
    assert observation["rows"][12]["status"] == "failed"
    assert observation["rows"][12]["prediction_replay_updates"] == 4
    assert all(e[0] != "final" for e in fake_ports[0].trace)
    assert fake_ports[0].value == 0


def test_should_refuse_an_incomplete_observer_before_gate_input_or_more_work(
    fake_ports, monkeypatch
):
    original = ArrivedShiftObserver.__call__

    def omit(self, updates, state, arrived):
        if updates != 9:
            original(self, updates, state, arrived)

    monkeypatch.setattr(ArrivedShiftObserver, "__call__", omit)
    row = run_one()
    assert row["status"] == "failed"
    assert row["benchmark_completed_updates"] == 9
    assert row["replay_budget_charged_exposures"] == 0
    assert row["observations"]["prediction_calls_completed"] == 8
    assert row["observations"]["rows"][8]["status"] == "not_run"
    assert all(e[0] != "final" for e in fake_ports[0].trace)


def test_should_include_actual_replay_work_and_pc_neutral_parity_in_the_pilot(
    fake_ports, monkeypatch
):
    parameters = CoupledParameters(
        (np.zeros((2, 8)), np.zeros((8, 8))),
        (np.zeros((1, 8)), np.zeros((1, 8))),
        np.zeros((8, 1)),
        np.zeros((1, 1)),
    )
    monkeypatch.setattr(
        pilot,
        "CoupledPredictiveCodingNetwork",
        lambda *args: SimpleNamespace(snapshot_parameters=lambda: parameters),
    )
    report = pilot.run_pilot(calibrated=True, incidents=True, incident_replay=True)
    assert report["status"] == "complete", report["global_errors"]
    assert len(report["runs"]) == 8
    assert report["benchmark_completed_updates"] == report["completed_observations"] == 144
    assert report["native_training_updates"] == 176
    assert report["training_row_exposures"] == 1184
    assert report["pc_neutral_exact"] == {"stationary": True, "recurring": True}
    assert report["policy_feedback"] is True
    assert report["scientific_advantage_verified"] is report["full_R27_complete"] is False
    for stream in ("stationary", "recurring"):
        pc, neutral = [
            r
            for r in report["runs"]
            if r["stream"] == stream and r["method"] in ("pc", "neutral_cpc")
        ]
        assert pc["stream_result"] == neutral["stream_result"]
        assert pc["rehearsal"] == neutral["rehearsal"]
        # The declared calibration model ID intentionally names the method.
        assert pc["observations"]["rows"] == neutral["observations"]["rows"]


@pytest.mark.parametrize("calibrated", [False, True])
@pytest.mark.parametrize("incidents", [False, True])
def test_should_preserve_all_four_old_descriptors_when_feedback_is_disabled(calibrated, incidents):
    assert pilot.describe_protocol(
        calibrated=calibrated, incidents=incidents
    ) == pilot.describe_protocol(
        calibrated=calibrated,
        incidents=incidents,
        incident_replay=False,
    )


@pytest.mark.parametrize("entry", [pilot.describe_protocol, pilot.run_pilot])
@pytest.mark.parametrize("invalid", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_boolean_feedback_before_native_work(entry, invalid, monkeypatch):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(pilot, "build_stream", forbidden)
    with pytest.raises(ValueError, match="bool"):
        entry(calibrated=True, incidents=True, incident_replay=invalid)


@pytest.mark.parametrize("entry", [pilot.describe_protocol, pilot.run_pilot])
@pytest.mark.parametrize("calibrated,incidents", [(False, False), (False, True), (True, False)])
def test_should_require_both_calibration_and_incidents_before_feedback(
    entry, calibrated, incidents, monkeypatch
):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError, match="calibrat|incident"):
        entry(calibrated=calibrated, incidents=incidents, incident_replay=True)


@pytest.mark.parametrize("action", ["describe", "run"])
def test_should_forward_explicit_cli_feedback_once(action, monkeypatch, capsys):
    calls = []

    def selected(**options):
        calls.append(options)
        return {"status": "complete"}

    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    monkeypatch.setattr(cli, "describe_protocol" if action == "describe" else "run_pilot", selected)
    monkeypatch.setattr(
        sys, "argv", ["shift", f"--{action}", "--calibrated", "--incidents", "--incident-replay"]
    )
    assert cli.main() == 0
    assert calls == [dict(calibrated=True, incidents=True, incident_replay=True)]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


@pytest.mark.parametrize("flags", [[], ["--incidents"], ["--calibrated"]])
def test_should_refuse_incomplete_cli_feedback_mode_before_dispatch(flags, monkeypatch):
    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["shift", "--describe", "--incident-replay", *flags])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_preserve_paid_partial_final_scores_and_all_replay(fake_ports, monkeypatch):
    original = FakeLearner.predict
    calls = 0

    def fail(self, features):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second final readout failure")
        return original(self, features)

    monkeypatch.setattr(FakeLearner, "predict", fail)
    row = run_one()
    assert row["status"] == "failed" and "final readout failure" in row["reason"]
    assert row["benchmark_completed_updates"] == 18
    assert row["replay_budget_charged_exposures"] == row["rehearsal"]["committed_updates"] == 4
    assert row["observations"]["prediction_calls_completed"] == 18
    progress = row["final_score_progress"]
    assert progress["status"] == "failed"
    assert progress["prediction_calls_attempted"] == 2
    assert progress["prediction_calls_completed"] == 1
    assert progress["prediction_rows_attempted"] == 128
    assert progress["prediction_rows_completed"] == 64
    assert set(progress["checkpoints"][0]["scores"]) == {"A"}
    assert progress["checkpoints"][0]["status"] == "incomplete"
    assert row["outcomes"] is None and fake_ports[0].value == 0


def test_should_refuse_a_silently_incomplete_last_observation_before_final(fake_ports, monkeypatch):
    original = ArrivedShiftObserver.__call__

    def omit(self, updates, state, arrived):
        if updates != 18:
            original(self, updates, state, arrived)

    monkeypatch.setattr(ArrivedShiftObserver, "__call__", omit)
    row = run_one()
    assert row["status"] == "failed"
    assert row["benchmark_completed_updates"] == 18
    assert row["replay_budget_charged_exposures"] == 4
    assert row["observations"]["rows"][-1]["status"] == "not_run"
    assert row["final_score_progress"] == {}
    assert all(e[0] != "final" for e in fake_ports[0].trace)
