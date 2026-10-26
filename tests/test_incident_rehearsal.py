"""Future-boundary incident replay through the real stream with fake native work."""

from copy import deepcopy
import json

import numpy as np
import pytest

from scripts.run_sensor_drift_demo import main
from src.adapters import sensor_drift_demo
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.stream_rehearsal import StreamRehearsal
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.shift_incidents import ShiftIncidentDecision
from src.core.sensor_stream import make_sensor_stream
from test_sensor_replay import FakeReplayLearner, fake_replay_ports as fake_replay_ports


def _stream(*, omit=None, onset=6):
    now = [0.0]
    clock = lambda: now[0]
    wake = ToyBudgetSession(ToyExecutionBudget(max_training_updates=12, max_wall_seconds=30), clock)
    replay = ToyBudgetSession(
        ToyExecutionBudget(max_replay_examples=12, max_wall_seconds=30), clock
    )
    source = FakeReplayLearner()
    helper = StreamRehearsal(
        source,
        wake,
        sensor_drift_demo._replay_policy(),
        replay_budget=replay,
        total_updates=12,
        input_dim=16,
        incident_channel="feature",
    )
    training = tuple(
        AttributionBatch(
            "train",
            "A" if n < 4 or n >= 8 else "B",
            tuple(f"row-{n}-{r}" for r in range(4)),
            b.features,
            b.targets,
        )
        for n, b in enumerate(make_sensor_stream())
    )

    def observe(n, state, batch):
        if not n or n == omit:
            return
        state = (
            "onset"
            if n == onset
            else "continued"
            if onset < n < 10
            else "cleared"
            if n == 10
            else "inactive"
        )
        helper.observe_incident(
            ShiftIncidentDecision(
                n,
                ("inactive", state, "inactive"),
                (None, onset if state != "inactive" else None, None),
            ),
            completed_updates=n,
        )

    stream = AttributionStream(
        source,
        training,
        (),
        checkpoint_updates=tuple(range(13)),
        label_delay=2,
        budget=wake,
        observe_checkpoint=observe,
        training_only=True,
        rehearsal=helper,
    )
    return stream, helper, wake, replay, source, now


def test_should_skip_admit_skip_using_only_prior_arrived_incident(fake_replay_ports):
    report = sensor_drift_demo.run_demo(replay=True, monitoring=True, incident_replay=True)
    assert report["status"] == "complete", report["reason"]
    helper = report["rehearsal"]
    assert [e["status"] for e in helper["events"]] == ["skipped", "complete", "skipped"]
    assert [e["charged_exposures"] for e in helper["events"]] == [0, 4, 0]
    decision = helper["events"][1]["incident_admission"]
    assert (
        decision["onset_batch"] == 6
        and decision["observed_updates"] == 7
        and decision["boundary_updates"] == 8
    )
    assert helper["incident_admission"]["last_consumed_onset"] == 6
    assert helper["incident_admission"]["admitted_events"] == 1
    assert report["charged_training_updates"] == 12
    assert report["replay_budget_charged_exposures"] == 4
    assert report["source_unchanged"] and report["actor_unchanged"]
    trains = [e for e in FakeReplayLearner.trace if e[0] == "train"]
    assert [len(e[2]) for e in trains] == [4] * 8 + [1] * 4 + [4] * 4
    # Replay rows may include current arrived wake8; its incident cannot admit this event.
    fixture = make_sensor_stream()
    arrived = {tuple(row) for batch in fixture[:8] for row in batch.features}
    assert all(tuple(e[2][0]) in arrived for e in trains if len(e[2]) == 1)
    assert helper["events"][0]["receipt"] is helper["events"][2]["receipt"] is None


def test_should_keep_paid_failure_and_consumed_onset_without_more_training(fake_replay_ports):
    FakeReplayLearner.fail_replay = True
    report = sensor_drift_demo.run_demo(replay=True, monitoring=True, incident_replay=True)
    assert report["status"] == "incomplete"
    assert report["charged_training_updates"] == 8
    helper = report["rehearsal"]
    assert helper["stopped"] and helper["charged_exposures"] == 1
    assert (
        helper["attempted_updates"] == 1
        and helper["returned_updates"] == helper["committed_updates"] == 0
    )
    assert helper["events"][1]["status"] == "failed"
    assert helper["incident_admission"]["pending_onset"] is None
    assert helper["incident_admission"]["last_consumed_onset"] == 6
    assert len(report["rows"]) == 7 and len(report["pending_forecasts"]) == 3
    assert len([e for e in FakeReplayLearner.trace if e[0] == "train"]) == 9


def test_should_preserve_pending_gate_and_original_sessions_through_pause(fake_replay_ports):
    stream, helper, wake, replay, source, now = _stream()
    original = (id(wake), id(replay), wake.started_at, replay.started_at, id(wake.clock))
    stream.run_until(6)
    assert helper.report["incident_admission"]["pending_onset"] == 6
    pause = stream.pause()
    before = helper.report
    now[0] = 1.0
    stream.resume(pause)
    assert helper.report == before
    assert original == (id(wake), id(replay), wake.started_at, replay.started_at, id(wake.clock))
    stream.run_until(12)
    report = stream.finish()
    assert report["completed_updates"] == 12 and replay.replay_examples_completed == 4
    assert helper.report["incident_admission"]["admitted_events"] == 1
    assert source.updates == 0


@pytest.mark.parametrize("replace_gate", [False, True])
def test_should_refuse_changed_pending_gate_after_pause(fake_replay_ports, replace_gate):
    stream, helper, *_ = _stream()
    stream.run_until(6)
    pause = stream.pause()
    if replace_gate:
        helper._incident_gate = deepcopy(helper._incident_gate)
    else:
        helper._incident_gate._pending_onset = None
    with pytest.raises(ValueError):
        stream.resume(pause)


def test_should_refuse_corrupted_gate_in_pause_evidence(fake_replay_ports):
    stream, helper, wake, replay, *_ = _stream()
    stream.run_until(6)
    pause = stream.pause()
    pause._snapshot["rehearsal_state"].incident_gate._pending_onset = None
    with pytest.raises(ValueError):
        stream.resume(pause)
    assert wake.updates_completed == 6 and replay.replay_examples_completed == 0


@pytest.mark.parametrize("omitted", [1, 5])
def test_should_stop_before_next_paid_wake_when_incident_observation_missing(
    fake_replay_ports, omitted
):
    stream, helper, wake, replay, *_ = _stream(omit=omitted)
    with pytest.raises(ValueError, match="preceding arrived observation"):
        stream.run_until(12)
    assert wake.updates_completed == omitted
    assert replay.replay_examples_completed == 0


def test_should_not_admit_an_onset_observed_at_the_current_boundary(fake_replay_ports):
    stream, helper, *_ = _stream(onset=4)
    stream.run_until(4)
    assert helper.report["events"][0]["status"] == "skipped"
    assert helper.report["incident_admission"]["pending_onset"] == 4
    stream.run_until(8)
    assert helper.report["events"][1]["incident_admission"]["onset_batch"] == 4
    assert helper.report["committed_updates"] == 4


@pytest.mark.parametrize(
    "options",
    [
        {"incident_replay": True},
        {"replay": True, "incident_replay": True},
        {"monitoring": True, "incident_replay": True},
    ],
)
def test_should_refuse_missing_explicit_modes_before_model_work(options):
    with pytest.raises(ValueError, match="requires"):
        sensor_drift_demo.describe_protocol(**options)
    with pytest.raises(ValueError, match="requires"):
        sensor_drift_demo.run_demo(**options)


@pytest.mark.parametrize("invalid", [1, None, "true", np.bool_(True)])
def test_should_refuse_non_native_incident_flag(invalid):
    with pytest.raises(ValueError, match="bool"):
        sensor_drift_demo.describe_protocol(replay=True, monitoring=True, incident_replay=invalid)


def test_should_preview_and_forward_only_explicit_incident_cli_mode(monkeypatch, capsys):
    seen = []

    def run(**options):
        seen.append(options)
        return {"status": "complete"}

    monkeypatch.setattr(sensor_drift_demo, "run_demo", run)
    assert main(["--replay", "--monitoring", "--incident-replay"]) == 0
    assert seen == [{"replay": True, "monitoring": True, "incident_replay": True}]
    capsys.readouterr()
    assert main(["--describe", "--replay", "--monitoring", "--incident-replay", "--heldout"]) == 0
    protocol = json.loads(capsys.readouterr().out)
    assert protocol["incident_replay"]["channel"] == "feature"
    assert protocol["final_evaluation"] and protocol["monitoring"]["feedback"]
    with pytest.raises(SystemExit):
        main(["--incident-replay"])


def test_should_score_final_sets_only_after_incident_training_finishes(
    fake_replay_ports, monkeypatch
):
    final_calls = []

    def predict(self, features):
        trains = [e for e in FakeReplayLearner.trace if e[0] == "train"]
        assert len(trains) == 16
        assert features.shape == (32, 16)
        final_calls.append(self.updates)
        return np.full((32, 1), 0.5)

    monkeypatch.setattr(FakeReplayLearner, "predict", predict)
    report = sensor_drift_demo.run_demo(
        replay=True, monitoring=True, incident_replay=True, heldout=True
    )
    assert report["status"] == "complete", report["reason"]
    assert len(final_calls) == 26
    assert report["heldout_evaluation"]["progress"]["prediction_calls_completed"] == 26
    assert len(report["heldout_evaluation"]["training_receipts"]) == 12
    assert report["rehearsal"]["committed_updates"] == 4


def test_should_leave_final_unavailable_after_failed_incident_replay(
    fake_replay_ports, monkeypatch
):
    calls = []

    def predict(self, features):
        calls.append(len(features))
        return np.full((len(features), 1), 0.5)

    monkeypatch.setattr(FakeReplayLearner, "predict", predict)
    FakeReplayLearner.fail_replay = True
    report = sensor_drift_demo.run_demo(
        replay=True, monitoring=True, incident_replay=True, heldout=True
    )
    assert report["status"] == "incomplete"
    assert calls == []
    assert report["heldout_evaluation"]["progress"]["status"] == "not_run"
    assert report["heldout_evaluation"]["progress"]["prediction_calls_attempted"] == 0
    assert len(report["heldout_evaluation"]["training_receipts"]) == 8
    assert report["charged_training_updates"] == 8
    assert report["replay_budget_charged_exposures"] == 1
