"""Teacher replay through real runtime isolation, budget and CLI boundaries."""

from copy import deepcopy
import json
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_stream_rehearsal_pilot as cli
import src.adapters.slow_teacher_pilot as pilot
from src.adapters.coupled_learners import CoupledLearner
from src.adapters.matched_backprop_learner import MatchedBackpropLearner
from src.adapters.slow_teacher_learner import SlowTeacherLearner
from src.app.coupled_attribution import AttributionBatch, run_attribution_stream
from src.app.stream_rehearsal import StreamRehearsal, StreamReplayPolicy
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import ReplayRetentionBudget, replay_sample_id
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.learner_ports import TrainingDiagnostic
from src.core.matched_backprop import MatchedBackpropNetwork
from src.core.shared_replay_schedule import ReplayAllocation
from src.core.slow_teacher import SlowTeacherPolicy, TeacherTargets


def policy(weight=0.25):
    return SlowTeacherPolicy("integration-teacher", "fixture_x2_v1", mixture_weight=weight)


class FakeTeacher:
    teacher_storage_bytes = 104

    def __init__(self, trace=None, state=None, *, weight=0.25, fault=None):
        self.teacher_policy = policy(weight)
        self.trace = [] if trace is None else trace
        self.state = [0, 0] if state is None else list(state)
        self.fault = fault

    def fork(self):
        return FakeTeacher(
            self.trace, self.state, weight=self.teacher_policy.mixture_weight, fault=self.fault
        )

    def snapshot_state(self):
        return list(self.state)

    def restore_state(self, state):
        self.state = list(state)

    def train_batch(self, x, y):
        self.trace.append(("wake", x.copy(), y.copy()))
        self.state[0] += 1
        return TrainingDiagnostic("fake_wake_v1", 0.5)

    def make_teacher_targets(self, x, y, **schemas):
        self.trace.append(("teacher", x.copy(), y.copy(), schemas))
        if self.fault == "forward":
            raise RuntimeError("teacher failed after attempted forward")
        ids = tuple(replay_sample_id(x[i : i + 1], y[i : i + 1]) for i in range(len(x)))
        return TeacherTargets(
            1, self.teacher_policy, self.state[0], sum(self.state), ids, np.full_like(y, 0.4)
        )

    def train_distillation_batch(self, x, y, packet):
        self.trace.append(("replay", x.copy(), y.copy(), packet))
        self.state[1] += 1
        if self.fault == "native":
            raise RuntimeError("student failed after attempted native replay")
        return TrainingDiagnostic("fake_mixed_v1", 0.6)

    def predict(self, x):
        self.trace.append(("score", x.copy()))
        return np.full((len(x), 1), 0.45)


def setup(source):
    clock = lambda: 0.0
    wake = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2, max_wall_seconds=10), clock)
    replay = ToyBudgetSession(ToyExecutionBudget(max_replay_examples=2, max_wall_seconds=10), clock)
    selection = StreamReplayPolicy(
        2,
        2,
        ReplayRetentionBudget(4, 96),
        ReplayAllocation(ReplayRetentionBudget(2, 48), ReplayRetentionBudget(2, 48)),
    )
    helper = StreamRehearsal(
        source,
        wake,
        selection,
        replay_budget=replay,
        total_updates=2,
        input_dim=2,
        distillation=source.teacher_policy,
    )
    return wake, replay, helper


def run(source, wake, helper, observe=None):
    training = tuple(
        AttributionBatch(
            "train",
            "A",
            (f"t{k}a", f"t{k}b"),
            np.array([[0.1 + k, 0.2], [0.3 + k, 0.4]]),
            np.array([[0.0], [1.0]]),
        )
        for k in range(2)
    )
    final = (
        AttributionBatch("final_test", "A", ("f0",), np.array([[99.0, 98.0]]), np.array([[1.0]])),
    )
    return run_attribution_stream(
        source,
        training,
        final,
        checkpoint_updates=(0, 2),
        label_delay=1,
        budget=wake,
        rehearsal=helper,
        observe_checkpoint=observe,
    )


def test_should_generate_only_arrived_row_targets_and_keep_actor_source_isolated():
    source = FakeTeacher()
    wake, replay, helper = setup(source)
    result = run(source, wake, helper)
    assert source.state == [0, 0]
    assert result["completed_updates"] == 2 and replay.replay_examples_completed == 2
    assert [r[0] for r in source.trace] == [
        "wake",
        "wake",
        "teacher",
        "replay",
        "teacher",
        "replay",
        "score",
        "score",
    ]
    for record in source.trace:
        if record[0] == "teacher":
            assert len(record[1]) == 1 and np.all(record[1] < 3)
            assert record[3] == dict(
                role="train", feature_schema="fixture_x2_v1", output_schema="binary_probability_v1"
            )
    costs = helper.report["distillation"]
    assert costs["teacher_forward_attempts"] == costs["teacher_forward_returns"] == 2
    assert (
        costs["teacher_probability_bytes_generated"] == costs["mixed_target_bytes_returned"] == 16
    )
    assert costs["peak_probability_array_bytes"] == 8 and costs["retained_target_array_bytes"] == 0
    packets = helper.report["events"][0]["teacher_targets"]
    assert [(p["teacher_revision"], p["student_updates"]) for p in packets] == [(2, 2), (2, 3)]
    assert all("probabilities" not in p for p in packets)
    detached = helper.report
    detached["distillation"]["teacher_forward_attempts"] = -1
    assert helper.report["distillation"]["teacher_forward_attempts"] == 2


@pytest.mark.parametrize(
    "fault,forwards,returned,charged", [("forward", 1, 0, 0), ("native", 1, 1, 1)]
)
def test_should_preserve_failed_teacher_work_and_native_charges(fault, forwards, returned, charged):
    source = FakeTeacher(fault=fault)
    wake, replay, helper = setup(source)
    with pytest.raises(RuntimeError, match="failed after attempted"):
        run(source, wake, helper)
    report = helper.report
    assert report["stopped"] and report["events"][0]["status"] == "failed"
    assert report["distillation"]["teacher_forward_attempts"] == forwards
    assert report["distillation"]["teacher_forward_returns"] == returned
    assert report["distillation"]["mixed_target_bytes_returned"] == 0
    assert replay.replay_examples_completed == report["attempted_updates"] == charged
    assert report["committed_updates"] == 0 and source.state == [0, 0]
    assert not any(r[0] == "score" for r in source.trace)


def test_should_skip_teacher_forward_at_zero_mixture():
    source = FakeTeacher(weight=0, fault="forward")
    source.teacher_storage_bytes = 0
    wake, _, helper = setup(source)
    run(source, wake, helper)
    assert not any(r[0] == "teacher" for r in source.trace)
    assert all(r[3] is None for r in source.trace if r[0] == "replay")
    assert helper.report["distillation"]["teacher_forward_attempts"] == 0


def test_should_refuse_replaced_teacher_policy_before_runtime_forks():
    source = FakeTeacher()
    wake, _, helper = setup(source)
    source.teacher_policy = SlowTeacherPolicy("replaced", "fixture_x2_v1")
    with pytest.raises(ValueError, match="original declared policy"):
        helper.bind(source, wake, 2, 2)
    assert source.trace == [] and not helper.report["bound"]


def test_should_refuse_teacher_continuation_explicitly():
    source = FakeTeacher()
    wake, _, helper = setup(source)
    helper.bind(source, wake, 2, 2)
    with pytest.raises(ValueError, match="not yet supported"):
        helper._capture_continuation()


@pytest.mark.parametrize("method", ("bp", "pc", "neutral_cpc", "gated_cpc"))
def test_should_execute_native_wake_and_teacher_replay_on_owned_candidate(method):
    model: Any = CoupledPredictiveCodingNetwork(2, (3,), 23)
    student: CoupledLearner | MatchedBackpropLearner
    if method == "bp":
        student = MatchedBackpropLearner(
            MatchedBackpropNetwork(model.snapshot_parameters()), learning_rate=0.03
        )
    else:
        if method != "pc":
            config = (
                CoupledCircadianConfig.matched_pc_control()
                if method == "neutral_cpc"
                else CoupledCircadianConfig()
            )
            model = CoupledCircadianPredictiveCodingNetwork(2, (3,), 23, config=config)
        student = CoupledLearner(
            model, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.04
        )
    source = SlowTeacherLearner(student, policy())
    before = pickle.dumps(source.snapshot_state())
    wake, replay, helper = setup(source)
    states = []
    result = run(source, wake, helper, lambda u, s, b: states.append(deepcopy(s)))
    assert pickle.dumps(source.snapshot_state()) == before
    assert [(s.wake_updates, s.replay_updates, s.teacher_revision) for s in states] == [
        (0, 0, 0),
        (2, 2, 2),
    ]
    assert helper.report["committed_updates"] == replay.replay_examples_completed == 2
    assert result["score_forward_calls"] == 2
    assert helper.report["distillation"]["teacher_storage_array_bytes"] == 104
    assert all(
        "slow_teacher_mixture" in d["definition"] for d in helper.report["events"][0]["diagnostics"]
    )


def forbidden(*args, **kwargs):
    pytest.fail("unexpected native work")


def test_should_describe_frozen_protocol_without_models_or_sampler(monkeypatch):
    for name in (
        "build_stream",
        "build_learner",
        "CoupledPredictiveCodingNetwork",
        "ProcessRssSampler",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    report = pilot.describe_protocol()
    assert report["seeds"] == [23] and report["replay"]["seed"] == 53
    assert report["planned_work"]["optimizer_updates"] == 120
    assert report["planned_work"]["teacher_forward_calls"] == 48
    assert report["teacher_storage_array_bytes_per_arm"] == 840
    assert all(
        p["mixture_weight"] == 0.25 and p["ema_decay"] == 0.9 for p in report["teachers"].values()
    )
    assert report["resources"]["storage_bytes"] == 2 * 1024**2


def test_should_preserve_all_unreached_arms_on_sampler_failure(monkeypatch):
    def fail():
        raise RuntimeError("sampler unavailable")

    monkeypatch.setattr(pilot, "ProcessRssSampler", fail)
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", forbidden)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and report["wake_updates"] == 0
    assert len(report["runs"]) == 4 and all(r["status"] == "not_run" for r in report["runs"])
    assert "sampler unavailable" in report["global_errors"][0]


@pytest.mark.parametrize("action", ("describe", "run"))
def test_should_route_teacher_cli_without_running_other_scenarios(monkeypatch, capsys, action):
    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    monkeypatch.setattr(
        cli,
        "describe_teacher_protocol" if action == "describe" else "run_teacher_pilot",
        lambda: dict(status="complete", selected=action),
    )
    monkeypatch.setattr(sys, "argv", ["rehearsal", f"--{action}", "--scenario", "slow-teacher"])
    assert cli.main() == 0 and json.loads(capsys.readouterr().out)["selected"] == action


@pytest.mark.parametrize("flag", ("--recent-only", "--diagnose-learnability"))
def test_should_refuse_single_case_flags_before_teacher_execution(monkeypatch, flag):
    monkeypatch.setattr(cli, "run_teacher_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["rehearsal", "--run", "--scenario", "slow-teacher", flag])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
