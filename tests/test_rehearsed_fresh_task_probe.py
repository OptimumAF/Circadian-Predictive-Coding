"""Fixed TRAIN replay belongs only to the disposable role-preserving F stream."""

from dataclasses import replace
import pickle

import numpy as np
import pytest

from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.stream_rehearsal import RecentStreamReplayPolicy, StreamRehearsal
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.demonstration_episode import EpisodeReplaySchema
from src.core.learner_ports import TrainingDiagnostic
from test_probe_training_readouts import cohorts
from test_probe_return_phase import return_templates


class Learner:
    def __init__(self, control=None, state=None):
        self.control = dict(calls=[], scores=[], children=[]) if control is None else control
        self.state = np.array([99.0]) if state is None else state.copy()

    def fork(self):
        child = Learner(self.control, self.state)
        self.control["children"].append(child)
        return child

    def snapshot_state(self):
        return self.state.copy()

    def restore_state(self, state):
        self.state = state.copy()

    def train_batch(self, features, targets):
        self.control["calls"].append((features.copy(), targets.copy()))
        if self.control.get("fail_call") == len(self.control["calls"]):
            raise RuntimeError("paid replay native failure")
        self.state += 1
        return TrainingDiagnostic("fake_rehearsed_probe", float(self.state[0]))

    def predict(self, features):
        self.control["scores"].append(float(self.state[0]))
        if self.control.get("score_hook"):
            self.control["score_hook"]()
        probability = 0.6 + 0.001 * float(self.state[0])
        output = np.full((len(features), 3), (1 - probability) / 2)
        output[:, 1 if features[0, 0] < 0 else 0] = probability
        return output


def session(*, updates=None, replay=None):
    return ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=updates, max_replay_examples=replay), lambda: 0.0
    )


def run(*, fail_call=None, factory_fault=None, score_hook=None, use_return=True):
    source = Learner()
    source.control.update(fail_call=fail_call, score_hook=score_hook)
    training, evaluation, monitors = cohorts(episode=True)
    returns = return_templates(monitors)
    before = pickle.dumps((training, evaluation, monitors, returns))
    wake, replay, returning = session(updates=16), session(replay=288), session(updates=16)
    retained = []

    def factory(learner, budget, count):
        schema = EpisodeReplaySchema(2, 9, 3)
        bank = tuple(packet.episode for packet in monitors)
        helper = StreamRehearsal(
            learner,
            budget,
            RecentStreamReplayPolicy(
                1, 18, ReplayRetentionBudget(18, 2 * 8 * (schema.encoded_dim + 1))
            ),
            replay_budget=replay,
            total_updates=count,
            input_dim=2,
            episode_schema=schema,
            fixed_training_episodes=bank,
            wake_role="probe_train",
        )
        retained.append(helper)
        if factory_fault == "model":
            learner.state += 1
        elif factory_fault == "scope":
            helper._wake_role = "train"
        elif factory_fault == "raise":
            raise RuntimeError("construction failed")
        elif factory_fault == "limits":
            budget.budget = replace(budget.budget, max_wall_seconds=123.0)
        elif factory_fault == "clock":
            budget.clock = lambda: 0.0
        elif factory_fault == "started":
            budget.started_at = -1.0
        elif factory_fault == "progress":
            budget.progress = object()
        return helper

    report = run_fresh_task_probe(
        source,
        np.array([0.0]),
        training,
        evaluation,
        budget=wake,
        checkpoint_updates=(0, 4, 8, 16),
        timer=lambda: 1.0,
        training_readouts=monitors,
        decision_diagnostics=True,
        rehearsal_factory=factory,
        **(dict(return_training=returns, return_budget=returning) if use_return else {}),
    )
    assert pickle.dumps((training, evaluation, monitors, returns)) == before
    assert source.state[0] == 99
    return report, source, wake, replay, returning, retained


def test_should_measure_f_and_old_a_with_replay_then_return_on_the_same_candidate():
    report, source, wake, replay, returning, helpers = run()
    assert report["status"] == "complete", report["reason"]
    assert report["completed_updates"] == wake.updates_completed == 16
    assert report["completed_training_examples"] == 144
    assert report["score_forward_calls_completed"] == 8
    assert report["training_readouts"]["score_forward_calls_completed"] == 8
    assert report["return_phase"]["status"] == "complete"
    assert report["return_phase"]["completed_updates"] == returning.updates_completed == 16
    assert replay.replay_examples_completed == report["rehearsal"]["charged_exposures"] == 288
    assert (
        replay.updates_completed
        == wake.replay_examples_completed
        == returning.replay_examples_completed
        == 0
    )
    assert len(source.control["calls"]) == 64
    assert sum(len(features) for features, _ in source.control["calls"]) == 576
    helper = helpers[0]
    assert helper._runtime._candidate.state[0] == 64
    assert helper._runtime.actor.snapshot_state().state[0] == 0
    assert helper._learner.state[0] == 0
    assert report["probe_stream"]["training_role"] == "probe_train"
    assert report["probe_stream"]["score_forward_calls"] == 0
    assert "rehearsal" not in report["probe_stream"]
    assert report["rehearsal"]["wake_role"] == "probe_train"
    assert all(row.role == "probe_train" for row in helper._runtime._inbox._experiences.values())
    assert all(row.role == "probe_train" for row in helper._runtime._inbox._labels.values())
    assert len(report["rehearsal"]["events"]) == 16
    for event in report["rehearsal"]["events"]:
        assert event["status"] == "complete" and event["charged_exposures"] == 18
        assert [row["role"] for row in event["selected_episodes"]] == ["train", "train"]
    assert source.control["scores"][0:4] == [0.0] * 4
    assert source.control["scores"][12:20] == [48.0] * 8  # F16, then explicit Areturn0.


def test_should_keep_paid_wake_and_failed_second_replay_without_scoring_or_return():
    report, source, wake, replay, returning, _ = run(fail_call=3)
    assert report["status"] == "failed" and "paid replay native failure" in report["reason"]
    assert report["attempted_updates"] == report["completed_updates"] == wake.updates_completed == 1
    assert report["completed_training_examples"] == 9
    assert replay.replay_examples_completed == 18 and returning.updates_completed == 0
    assert report["rehearsal"]["attempted_updates"] == 2
    assert report["rehearsal"]["returned_updates"] == 1
    assert report["rehearsal"]["committed_updates"] == 0
    assert len(report["probe_stream"]["applied_updates"]) == 1
    assert report["probe_stream"]["applied_consolidations"] == []
    assert len(report["checkpoints"]) == 1 and len(source.control["scores"]) == 4
    assert report["return_phase"]["status"] != "complete"


@pytest.mark.parametrize(
    "fault", ["model", "scope", "raise", "limits", "clock", "started", "progress"]
)
def test_should_refuse_changed_or_failed_factory_before_f_work(fault):
    report, source, wake, replay, returning, _ = run(factory_fault=fault)
    assert report["status"] == "failed"
    assert (
        wake.updates_completed
        == replay.replay_examples_completed
        == returning.updates_completed
        == 0
    )
    assert source.control["calls"] == source.control["scores"] == []


def test_should_refuse_uncounted_replay_charge_and_keep_failed_readback_explicit():
    source = Learner()
    training, evaluation, monitors = cohorts(episode=True)
    wake, replay = session(updates=16), session(replay=288)
    schema = EpisodeReplaySchema(2, 9, 3)

    def factory(learner, budget, count):
        return StreamRehearsal(
            learner,
            budget,
            RecentStreamReplayPolicy(1, 18, ReplayRetentionBudget(18, 10000)),
            replay_budget=replay,
            total_updates=count,
            input_dim=2,
            episode_schema=schema,
            fixed_training_episodes=tuple(row.episode for row in monitors),
            wake_role="probe_train",
        )

    def tamper():
        replay.replay_examples_completed += 1

    source.control["score_hook"] = tamper
    report = run_fresh_task_probe(
        source,
        np.array([0.0]),
        training,
        evaluation,
        budget=wake,
        checkpoint_updates=(0, 4, 8, 16),
        timer=lambda: 1.0,
        rehearsal_factory=factory,
    )
    assert report["status"] == "failed" and report["rehearsal"] is None
    assert report["replay_budget_charged_exposures"] == 1
    assert report["completed_updates"] == wake.updates_completed == 0
    assert source.control["calls"] == []
