"""Owned fake learner/environment checks for the frozen actual-game learner pilot."""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
import pickle
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_learning_pilot as pilot
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.torch_feature_learner import TorchFeaturePolicy
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
from src.app.coupled_attribution import AttributionBatch, AttributionStream
import src.app.toy_execution_budget as budgets
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    episode_arrays,
)
from src.core.learner_ports import TrainingDiagnostic
from src.core.minigrid_key_door import predefined_layouts
from test_torch_feature_rehearsal_pilot import Sampler, TorchStub, forbidden


METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc", "recurrent_bc")
POINTS = (0, 32, 64, 96, 128)


def same_state(left, right):
    return pickle.dumps(left) == pickle.dumps(right)


def observation(step=0):
    return (-1.0,) * 147 + (float((-1, 1)[step % 2]),)


def source_batch(context, cue, *, steps=9):
    episode = DemonstrationEpisode(
        f"source-{context}-{cue}",
        context,
        "train",
        tuple(
            DemonstrationTransition(
                index,
                observation(index),
                1,
                0.5 if index == steps - 1 else 0.0,
                observation(index + 1),
                index == steps - 1,
                7,
            )
            for index in range(steps)
        ),
        7,
    )
    features, targets = episode_arrays(episode)
    return AttributionBatch(
        "train",
        context,
        tuple(f"{episode.episode_id}:step:{index}" for index in range(steps)),
        features,
        targets,
        episode,
        7,
    )


@dataclass
class State:
    policy: Any
    parameters: dict[str, Any]
    optimizer_state: dict[str, Any]
    gradients: dict[str, Any]
    completed_updates: int = 0
    training_examples: int = 0


class Learner:
    work: dict[str, Any] = {}

    def __init__(self, policy=None):
        if policy is None:
            policy = TorchFeaturePolicy("bp", 148, 8, 17, num_classes=7)
        self.policy = policy
        self.state = State(
            policy, {"weight": np.zeros((1, 1))}, {"momentum": 0.0}, {"weight": None}
        )
        self.work.setdefault("owners", []).append(self)

    def snapshot_state(self):
        return deepcopy(self.state)

    def restore_state(self, state):
        self.work.setdefault("restores", []).append((self, deepcopy(state)))
        self.state = deepcopy(state)
        self.policy = deepcopy(state.policy)
        callback = self.work.get("on_restore")
        if callback:
            callback(self)

    def fork(self):
        self.work.setdefault("forks", []).append(self)
        if self.work.get("identity_fork"):
            return self
        result = Learner(self.policy)
        result.state = self.snapshot_state()
        return result

    def predict(self, features):
        self.work.setdefault("predicts", []).append((self, features.copy()))
        callback = self.work.get("on_predict")
        if callback:
            callback(self)
        if self.work.get("prediction_error"):
            raise RuntimeError("fake prediction failed")
        if self.work.get("bad_probability"):
            return np.ones((len(features), 6), dtype=np.float64)
        return np.tile(np.array([[0.5, 0.5, 0, 0, 0, 0, 0]], dtype=np.float64), (len(features), 1))

    def train_batch(self, features, targets):
        self.work.setdefault("trains", []).append((self, features.copy(), targets.copy()))
        fail = self.work.get("fail_train")
        if fail and fail(self):
            raise RuntimeError("fake training failed")
        self.state.completed_updates += 1
        self.state.training_examples += len(features)
        self.state.parameters["weight"] += 0.01
        self.state.optimizer_state["momentum"] += 1.0
        callback = self.work.get("on_train")
        if callback:
            callback(self)
        return TrainingDiagnostic("fake_native_objective_v1", 0.5)

    def start_episode(self):
        session = Session(self)
        self.work.setdefault("sessions", []).append(session)
        return session


class Session:
    def __init__(self, owner):
        self.owner = owner
        self.state = owner.snapshot_state()
        self.completed_steps = 0

    def snapshot_state(self):
        return deepcopy(self.state)

    def predict(self, features):
        self.completed_steps += 1
        self.owner.work.setdefault("session_inputs", []).append((self, features.copy()))
        callback = self.owner.work.get("on_session_predict")
        if callback:
            callback(self)
        return self.owner.predict(features)


class Environment:
    work: dict[str, Any] = {}

    def __init__(self, layout, cue, seed=17, max_steps=16, *, role="train", episode_id="fake-live"):
        self.layout, self.cue, self.role = layout, cue, role
        self.seed, self.max_steps, self.episode_id = seed, max_steps, episode_id
        self.observation = observation()
        self.terminal = False
        self.actions: list[int] = []
        self.returned = 0
        self.transitions: list[dict[str, Any]] = []
        self.unobserved = False
        self.closed = False
        self.work.setdefault("environments", []).append(self)
        callback = self.work.get("on_environment")
        if callback:
            callback(self)

    @property
    def legal_actions(self):
        return () if self.terminal else tuple(range(7))

    @property
    def report(self):
        return dict(
            environment_steps_attempted=len(self.actions),
            environment_steps=self.returned,
            terminated=self.terminal and not self.work.get("timeout"),
            truncated=self.terminal and bool(self.work.get("timeout")),
            total_return=0.0 if self.work.get("timeout") else (0.5 if self.terminal else 0.0),
            status="complete" if self.terminal else "running",
            transitions=deepcopy(self.transitions),
            failure_work_may_be_unobserved=self.unobserved,
        )

    def step(self, action):
        self.actions.append(action)
        self.work.setdefault("steps", []).append((self, action))
        before = self.observation
        callback = self.work.get("on_step")
        if callback:
            callback(self)
        if self.work.get("step_error"):
            self.unobserved = True
            raise RuntimeError("fake native step failed")
        self.returned += 1
        self.terminal = not self.work.get("never_ends") and len(self.actions) >= (
            16 if self.work.get("timeout") else 2
        )
        self.observation = observation(len(self.actions))
        returned = dict(
            observation=self.observation,
            reward=0.0 if self.work.get("timeout") else (0.5 if self.terminal else 0.0),
            terminated=self.terminal and not self.work.get("timeout"),
            truncated=self.terminal and bool(self.work.get("timeout")),
        )
        self.transitions.append(
            dict(
                observation=before,
                next_observation=self.observation,
                action=action,
                reward=returned["reward"],
                terminated=returned["terminated"],
                truncated=returned["truncated"],
                raw_next_observation={"omitted_fixture": True},
            )
        )
        return returned

    def close(self):
        self.closed = True
        if self.work.get("close_error"):
            raise RuntimeError("fake close failed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@pytest.fixture
def evaluator(monkeypatch):
    work: dict[str, Any] = {}
    monkeypatch.setattr(Learner, "work", work)
    monkeypatch.setattr(Environment, "work", work)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", Environment)
    monkeypatch.setattr(evaluation, "_same_state", same_state)
    source = Learner()
    guard_calls = []

    def guard():
        guard_calls.append(len(work.get("steps", [])))
        callback = work.get("on_guard")
        if callback:
            callback()

    sources = tuple(
        source_batch(context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")
    )
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0)
    return SimpleNamespace(
        work=work,
        source=source,
        sources=sources,
        budget=budget,
        guard=guard,
        guard_calls=guard_calls,
    )


def readout(fixture, *, method="bp", final=False, state=None):
    layouts = predefined_layouts()
    if method == "recurrent_bc":
        fixture.source.policy = TorchRecurrentPolicy(148, 8, 7, 17)
        fixture.source.state.policy = fixture.source.policy
    if state is None:
        state = fixture.source.snapshot_state()
        state.completed_updates = 128
        state.training_examples = 1152
    return evaluation.evaluate_checkpoint(
        fixture.source,
        state,
        method=method,
        updates=128,
        layouts=(layouts[-1],) if final else layouts[:3],
        role="final_test" if final else "train",
        sources=() if final else fixture.sources,
        budget=fixture.budget,
        check_boundary=fixture.guard,
    )


def test_should_score_frozen_raw_probabilities_and_use_first_argmax_without_action_repair(
    evaluator,
):
    before = evaluator.source.snapshot_state()
    report = readout(evaluator)
    assert report["status"] == "complete" and report["state_unchanged"] is True
    assert len(report["scores"]) == len(report["episodes"]) == 6
    assert [(row["context"], row["cue"]) for row in report["scores"]] == [
        (context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")
    ]
    for score in report["scores"]:
        assert score["ce"] == pytest.approx(float(np.log(2)))
        assert score["accuracy"] == score["exact_sequence_accuracy"] == 0
    for episode in report["episodes"]:
        assert episode["status"] == "complete" and episode["success"] is True
        assert episode["return"] == 0.5 and episode["source_role"] == "train"
        assert len(episode["predictions"]) == 2
        assert [row["action"] for row in episode["predictions"]] == [0, 0]
        assert all(
            row["probabilities"] == [0.5, 0.5, 0, 0, 0, 0, 0] for row in episode["predictions"]
        )
        assert all(
            "raw_next_observation" not in row for row in episode["environment"]["transitions"]
        )
    work = report["work"]
    assert work["raw_score_forwards_completed"] == work["raw_score_forwards_attempted"] == 6
    assert work["raw_score_rows_completed"] == work["raw_score_rows_attempted"] == 54
    assert work["live_prediction_calls_returned"] == work["live_prediction_calls_attempted"] == 12
    assert work["environment_steps_returned"] == work["environment_steps_attempted"] == 12
    assert work["environment_resets_completed"] == 6 and work["live_sessions_started"] == 0
    assert len(evaluator.guard_calls) > 12
    assert same_state(before, evaluator.source.snapshot_state())
    assert not evaluator.work.get("trains")
    assert all(environment.closed for environment in evaluator.work["environments"])


def test_should_create_zero_step_owned_recurrent_sessions_for_every_live_episode(evaluator):
    report = readout(evaluator, method="recurrent_bc")
    sessions = evaluator.work["sessions"]
    assert report["work"]["live_sessions_started"] == len(sessions) == 6
    assert len({id(session) for session in sessions}) == 6
    assert all(session.completed_steps == 2 for session in sessions)
    assert all(session.owner is not evaluator.source for session in sessions)
    inputs = evaluator.work["session_inputs"]
    assert len(inputs) == 12 and all(values.shape == (1, 148) for _, values in inputs)
    for session in sessions:
        values = [value for owner, value in inputs if owner is session]
        np.testing.assert_array_equal(values[0], [observation()])
        np.testing.assert_array_equal(values[1], [observation(1)])


def test_should_run_endpoint_heldout_without_source_labels_or_teacher_calls(evaluator):
    report = readout(evaluator, final=True)
    assert report["role"] == "final_test" and report["updates"] == 128
    assert report["scores"] == []
    assert report["work"]["raw_score_forwards_attempted"] == 0
    assert [
        (game.layout.context, game.cue, game.role) for game in evaluator.work["environments"]
    ] == [("D", "blue", "final_test"), ("D", "red", "final_test")]
    assert len(evaluator.work["predicts"]) == 4 and not evaluator.work.get("trains")


def test_should_check_completed_training_guard_before_fork_or_heldout_construction(evaluator):
    def refuse():
        raise ValueError("completed training evidence changed")

    evaluator.work["on_guard"] = refuse
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as raised:
        readout(evaluator, final=True)
    assert "completed training" in raised.value.report["reason"]
    assert not evaluator.work.get("forks") and not evaluator.work.get("environments")
    assert raised.value.report["work"]["live_prediction_calls_attempted"] == 0


def test_should_apply_original_budget_before_fork_without_renewing_training_allowance(
    monkeypatch, evaluator
):
    def refuse():
        raise RuntimeError("original wall budget exhausted")

    monkeypatch.setattr(evaluator.budget, "before_final", refuse)
    with pytest.raises(evaluation.MiniGridEvaluationFailed, match="original wall budget"):
        readout(evaluator, final=True)
    assert evaluator.budget.updates_completed == 0
    assert not evaluator.work.get("forks") and not evaluator.work.get("environments")


def test_should_refuse_identity_fork_before_prediction_or_environment(evaluator):
    evaluator.work["identity_fork"] = True
    with pytest.raises(evaluation.MiniGridEvaluationFailed, match="independent"):
        readout(evaluator)
    assert not evaluator.work.get("predicts") and not evaluator.work.get("environments")


@pytest.mark.parametrize("fault", ["method", "seed", "dimension", "updates", "rows"])
def test_should_refuse_checkpoint_metadata_that_does_not_bind_its_declared_point(evaluator, fault):
    state = evaluator.source.snapshot_state()
    state.completed_updates, state.training_examples = 128, 1152
    if fault == "method":
        state.policy = replace(state.policy, method="pc")
    elif fault == "seed":
        state.policy = replace(state.policy, seed=18)
    elif fault == "dimension":
        state.policy = replace(state.policy, feature_dim=16)
    elif fault == "updates":
        state.completed_updates = 127
    else:
        state.training_examples = 1151
    with pytest.raises(evaluation.MiniGridEvaluationFailed):
        readout(evaluator, state=state)
    assert not evaluator.work.get("forks") and not evaluator.work.get("predicts")
    assert not evaluator.work.get("environments")


@pytest.mark.parametrize("fault", ["order", "features", "role", "updates", "final_labels"])
def test_should_refuse_changed_source_or_role_contract_before_fork(evaluator, fault):
    sources = evaluator.sources
    options: dict[str, Any] = dict(
        method="bp",
        updates=128,
        layouts=predefined_layouts()[:3],
        role="train",
        sources=sources,
        budget=evaluator.budget,
        check_boundary=evaluator.guard,
    )
    if fault == "order":
        options["sources"] = tuple(reversed(sources))
    elif fault == "features":
        sources[0].features[0, 0] = 0.0
    elif fault == "role":
        options["role"] = "probe_test"
    elif fault == "updates":
        options["updates"] = True
    else:
        options.update(role="final_test", layouts=(predefined_layouts()[-1],))
    with pytest.raises(ValueError):
        evaluation.evaluate_checkpoint(
            evaluator.source, evaluator.source.snapshot_state(), **options
        )
    assert not evaluator.work.get("forks") and not evaluator.work.get("environments")


@pytest.mark.parametrize("target", ["source", "scorer", "optimizer"])
def test_should_refuse_changed_learned_state_after_first_raw_score_without_live_work(
    evaluator, target
):
    def mutate(owner):
        selected = evaluator.source if target == "source" else owner
        if target == "optimizer":
            selected.state.optimizer_state["momentum"] += 1
        else:
            selected.state.parameters["weight"] += 1

    evaluator.work["on_predict"] = mutate
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as raised:
        readout(evaluator)
    report = raised.value.report
    assert report["status"] == "failed" and report["work"]["raw_score_forwards_completed"] == 1
    assert report["work"]["raw_score_rows_completed"] == 9
    assert len(evaluator.work["predicts"]) == 1 and not evaluator.work.get("environments")


@pytest.mark.parametrize("fault", ["learned_state", "activation_counter"])
def test_should_refuse_recurrent_prediction_corruption_before_native_action(evaluator, fault):
    def mutate(session):
        if fault == "learned_state":
            session.state.parameters["weight"] += 1
        else:
            session.completed_steps += 1

    evaluator.work["on_session_predict"] = mutate
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as raised:
        readout(evaluator, method="recurrent_bc")
    report = raised.value.report
    assert report["work"]["raw_score_forwards_completed"] == 6
    assert report["work"]["live_prediction_calls_returned"] == 1
    assert report["work"]["environment_steps_attempted"] == 0
    assert len(evaluator.work["sessions"]) == 1 and len(evaluator.work["environments"]) == 1
    assert evaluator.work["environments"][0].closed


@pytest.mark.parametrize("fault", ["nonzero", "reused"])
def test_should_refuse_nonfresh_recurrent_session_before_its_environment(
    monkeypatch, evaluator, fault
):
    shared: list[Session] = []

    def session(owner):
        if fault == "reused" and shared:
            return shared[0]
        current = Session(owner)
        if fault == "nonzero":
            current.completed_steps = 1
        shared.append(current)
        return current

    monkeypatch.setattr(Learner, "start_episode", session)
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as raised:
        readout(evaluator, method="recurrent_bc", final=True)
    report = raised.value.report
    assert report["work"]["environment_resets_completed"] == (0 if fault == "nonzero" else 1)
    assert report["work"]["environment_steps_returned"] == (0 if fault == "nonzero" else 2)


@pytest.mark.parametrize("fault", ["prediction", "probability", "step"])
def test_should_preserve_partial_final_work_and_stop_after_first_operational_failure(
    evaluator, fault
):
    evaluator.work[
        {"prediction": "prediction_error", "probability": "bad_probability", "step": "step_error"}[
            fault
        ]
    ] = True
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as raised:
        readout(evaluator, final=True)
    report = raised.value.report
    work = report["work"]
    assert report["status"] == "failed" and len(report["episodes"]) == 1
    assert work["live_prediction_calls_attempted"] == 1
    assert work["live_prediction_calls_returned"] == (0 if fault == "prediction" else 1)
    assert work["environment_steps_attempted"] == (1 if fault == "step" else 0)
    assert work["environment_steps_returned"] == 0
    assert len(evaluator.work["environments"]) == 1
    assert evaluator.work["environments"][0].closed
    assert work["failure_work_may_be_unobserved"] is True


def test_should_keep_primary_physics_fault_and_observed_prefix_when_close_also_fails(evaluator):
    evaluator.work.update(step_error=True, close_error=True)
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as raised:
        readout(evaluator, final=True)
    report = raised.value.report
    assert "fake native step failed" in report["reason"]
    assert "fake close failed" in report["episodes"][0]["close_error"]
    assert report["work"]["live_prediction_calls_returned"] == 1
    assert report["work"]["environment_steps_attempted"] == 1
    assert report["work"]["environment_steps_returned"] == 0
    assert report["work"]["failure_work_may_be_unobserved"] is True
    assert len(evaluator.work["environments"]) == 1
    assert evaluator.work["environments"][0].actions == [0]


def test_should_retain_step_and_prediction_when_completed_guard_fails_after_native_return(
    evaluator,
):
    def stop_after_step():
        if evaluator.work.get("steps"):
            raise ValueError("completed receipt changed after physics")

    evaluator.work["on_guard"] = stop_after_step
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as raised:
        readout(evaluator, final=True)
    report = raised.value.report
    assert (
        report["work"]["environment_steps_attempted"]
        == report["work"]["environment_steps_returned"]
        == 1
    )
    assert report["work"]["live_prediction_calls_returned"] == 1
    assert len(report["episodes"][0]["environment"]["transitions"]) == 1
    assert len(evaluator.work["predicts"]) == 1


def test_should_keep_native_timeout_distinct_from_success_at_fixed_sixteen_steps(evaluator):
    evaluator.work["timeout"] = True
    report = readout(evaluator, final=True)
    assert report["status"] == "complete"
    assert report["work"]["environment_steps_returned"] == 32
    for slot in report["episodes"]:
        assert slot["success"] is False and slot["return"] == 0.0
        assert slot["environment"]["truncated"] is True
        assert slot["environment"]["terminated"] is False
        assert len(slot["predictions"]) == 16


def test_should_refuse_missing_native_end_flag_without_more_than_sixteen_steps(evaluator):
    evaluator.work["never_ends"] = True
    with pytest.raises(evaluation.MiniGridEvaluationFailed, match="bounded ending") as raised:
        readout(evaluator, final=True)
    assert raised.value.report["work"]["environment_steps_returned"] == 16
    assert len(evaluator.work["environments"]) == 1


def pilot_environment(monkeypatch, evaluator):
    captured: dict[str, Any] = dict(
        streams=[], finished=[], collection_calls=0, evaluation_calls=[]
    )
    monkeypatch.setattr(pilot, "TorchFeatureLearner", Learner)
    monkeypatch.setattr(pilot, "TorchRecurrentLearner", Learner)
    monkeypatch.setattr(pilot, "require_torch", lambda: TorchStub)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "_same_state", same_state)

    def collect(check_boundary, reports):
        captured["collection_calls"] += 1
        check_boundary()
        for packet in evaluator.sources:
            reports.append(
                dict(
                    context=packet.context,
                    role="train",
                    status="complete",
                    reset_calls=1,
                    environment_steps=9,
                    environment_steps_attempted=9,
                    terminated=True,
                    truncated=False,
                    total_return=0.5,
                    transitions=[asdict(transition) for transition in packet.episode.transitions],
                )
            )
        check_boundary()
        return deepcopy(evaluator.sources)

    def stream(source, training, heldout, **kwargs):
        assert heldout == () and kwargs["training_only"] is True
        assert kwargs["checkpoint_updates"] == POINTS and kwargs["label_delay"] == 1
        assert not any(kwargs.get(name) for name in ("rehearsal", "renewal", "capacity"))
        actual = AttributionStream(source, training, heldout, **kwargs)
        captured["streams"].append((actual, source, deepcopy(training), kwargs["budget"]))
        finish = actual.finish

        def finished():
            result = finish()
            captured["finished"].append(actual)
            return result

        monkeypatch.setattr(actual, "finish", finished)
        return actual

    evaluate = evaluation.evaluate_checkpoint

    def completed_readout(source, state, **kwargs):
        assert len(captured["finished"]) == 5, (
            "readout exposed before all five training runs finished"
        )
        assert all(item[3].updates_completed == 128 for item in captured["streams"])
        captured["evaluation_calls"].append((kwargs["method"], kwargs["updates"], kwargs["role"]))
        return evaluate(source, state, **kwargs)

    def before_environment(game):
        assert len(captured["finished"]) == 5
        if game.layout.context == "D":
            assert game.role == "final_test"
        else:
            assert game.role == "train"

    evaluator.work["on_environment"] = before_environment
    monkeypatch.setattr(pilot, "_collect_sources", collect)
    monkeypatch.setattr(pilot, "AttributionStream", stream)
    monkeypatch.setattr(pilot, "evaluate_checkpoint", completed_readout)
    return captured


def test_should_finish_all_five_matched_streams_before_any_known_or_final_evaluation(
    monkeypatch, evaluator
):
    captured = pilot_environment(monkeypatch, evaluator)
    report = pilot.run_pilot()
    assert report["status"] == "complete", report["reason"]
    assert [row["method"] for row in report["methods"]] == list(METHODS)
    assert captured["collection_calls"] == 1 and len(captured["streams"]) == 5
    assert len(captured["finished"]) == 5
    assert len(evaluator.work["trains"]) == 640
    assert sum(len(features) for _, features, _ in evaluator.work["trains"]) == 5760
    for (_, isolated, training, budget), row in zip(
        captured["streams"], report["methods"], strict=True
    ):
        policy = isolated.policy
        assert getattr(policy, "feature_dim", getattr(policy, "input_dim", None)) == 148
        assert (policy.hidden_dim, policy.num_classes, policy.seed) == (8, 7, 17)
        assert policy.learning_rate == 0.03 and policy.momentum == 0.9
        assert len(training) == len({packet.episode.episode_id for packet in training}) == 128
        assert len({identity for packet in training for identity in packet.row_ids}) == 1152
        assert [packet.context for packet in training] == ["A"] * 32 + ["B"] * 32 + [
            "C_train"
        ] * 32 + ["A"] * 32
        assert all(
            packet.role == packet.episode.role == "train" and packet.features.shape == (9, 148)
            for packet in training
        )
        assert all(packet.num_classes == packet.episode.num_classes == 7 for packet in training)
        for index, packet in enumerate(training):
            base = next(
                source
                for source in evaluator.sources
                if source.context == packet.context
                and source.episode.episode_id.endswith("blue" if index % 2 == 0 else "red")
            )
            np.testing.assert_array_equal(packet.features, base.features)
            np.testing.assert_array_equal(packet.targets, base.targets)
            assert not np.shares_memory(packet.features, base.features)
        assert budget.updates_completed == row["charged_updates"] == 128
        assert budget.replay_examples_completed == 0
        assert row["native_training_examples"] == 1152
        assert row["source_unchanged"] is row["actor_unchanged"] is True
        assert isolated.state.completed_updates == 0
        assert row["stream_result"]["completed_updates"] == 128
        assert row["stream_result"]["score_forward_calls"] == 0
        assert [item["updates"] for item in row["readouts"]] == list(POINTS)
        assert row["final_evaluation"]["role"] == "final_test"
        assert row["final_evaluation"]["scores"] == []
        assert row["acquisition_conditions"] == dict(A=False, B=False, C_train=False)
        assert row["initial_A_acquired"] is False
        assert row["interpretation"] == "inconclusive_initial_A_not_acquired"
        assert all(
            value == 0 for metrics in row["signed_metrics"].values() for value in metrics.values()
        )
        assert row["work"]["raw_score_forwards_completed"] == 30
        assert row["work"]["raw_score_rows_completed"] == 270
    assert len(captured["evaluation_calls"]) == 30
    assert len(evaluator.work["environments"]) == 160
    assert sum(game.role == "final_test" for game in evaluator.work["environments"]) == 10
    assert len(evaluator.work["sessions"]) == 32
    assert all(game.closed for game in evaluator.work["environments"])
    assert report["scientific_advantage_verified"] is False
    assert all(value == 0 for pair in report["comparisons"].values() for value in pair.values())


def test_should_keep_partial_training_charges_and_refuse_all_evaluation_after_failure(
    monkeypatch, evaluator
):
    captured = pilot_environment(monkeypatch, evaluator)

    def fail(owner):
        return getattr(owner.policy, "method", None) == "pc" and owner.state.completed_updates == 2

    evaluator.work["fail_train"] = fail
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and "fake training failed" in report["reason"]
    assert len(captured["finished"]) == 1 and len(captured["streams"]) == 2
    assert captured["streams"][0][3].updates_completed == 128
    assert captured["streams"][1][3].updates_completed == 2
    assert report["methods"][0]["charged_updates"] == 128
    assert report["methods"][1]["charged_updates"] == 2
    assert not captured["evaluation_calls"] and not evaluator.work.get("environments")


def test_should_validate_preceding_success_before_second_peer_update(monkeypatch, evaluator):
    captured = pilot_environment(monkeypatch, evaluator)

    def corrupt_preceding(owner):
        if getattr(owner.policy, "method", None) == "pc":
            receipt = captured["streams"][0][0]._runtime.applied_updates[0]
            object.__setattr__(
                receipt, "diagnostic", TrainingDiagnostic("altered_evidence_v1", 9.0)
            )

    evaluator.work["on_train"] = corrupt_preceding
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and "preceding completed" in report["reason"]
    assert len(captured["streams"]) == 2 and len(captured["finished"]) == 1
    assert [item[3].updates_completed for item in captured["streams"]] == [128, 1]
    assert len(evaluator.work["trains"]) == 129
    assert not captured["evaluation_calls"] and not evaluator.work.get("environments")


def test_should_refuse_corrupted_completed_checkpoint_after_first_score_without_final_work(
    monkeypatch, evaluator
):
    captured = pilot_environment(monkeypatch, evaluator)

    def corrupt_checkpoint(owner):
        state = captured["streams"][0][0]._states[0][1]
        state.parameters["weight"] += 1

    evaluator.work["on_predict"] = corrupt_checkpoint
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and "preceding completed" in report["reason"]
    assert len(captured["finished"]) == 5 and len(evaluator.work["trains"]) == 640
    assert len(evaluator.work["predicts"]) == 1
    assert report["methods"][0]["readouts"][0]["work"]["raw_score_forwards_completed"] == 1
    assert all(item[3].updates_completed == 128 for item in captured["streams"])
    assert not evaluator.work.get("environments")


def test_should_describe_frozen_counts_and_native_policies_without_resources(monkeypatch):
    for name in (
        "require_torch",
        "ProcessRssSampler",
        "_collect_sources",
        "_learner",
        "AttributionStream",
        "evaluate_checkpoint",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    protocol = pilot.describe_protocol()
    assert json.loads(json.dumps(protocol, allow_nan=False)) == protocol
    assert protocol["methods"] == list(METHODS)
    assert protocol["checkpoint_updates"] == list(POINTS)
    assert protocol["planned_work"]["native_train_calls"] == 640
    assert protocol["planned_work"]["training_rows"] == 5760


@pytest.mark.parametrize("primary", ["teacher", "step"])
@pytest.mark.parametrize("cleanup", ["close", "readback"])
def test_should_preserve_source_collection_primary_error_and_cleanup_evidence(
    monkeypatch, evaluator, primary, cleanup
):
    class CollectionEnvironment(Environment):
        @property
        def report(self):
            if cleanup == "readback":
                raise RuntimeError("source report unavailable")
            return super().report

    def choose(public):
        if primary == "teacher":
            raise RuntimeError("source teacher primary")
        return 2

    evaluator.work.update(step_error=primary == "step", close_error=cleanup == "close")
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", CollectionEnvironment)
    monkeypatch.setattr(pilot, "CueKeyDoorTeacher", lambda: SimpleNamespace(choose_action=choose))
    reports: list[dict[str, Any]] = []
    expected_primary = (
        "source teacher primary" if primary == "teacher" else "fake native step failed"
    )
    with pytest.raises(RuntimeError, match=expected_primary):
        pilot._collect_sources(lambda: None, reports)
    assert len(reports) == len(evaluator.work["environments"]) == 1
    observed = reports[0]
    assert expected_primary in observed["primary_error"]
    assert evaluator.work["environments"][0].closed
    if cleanup == "close":
        assert "fake close failed" in observed["close_error"]
        assert observed["environment_steps_attempted"] == (1 if primary == "step" else 0)
        assert observed["environment_steps"] == 0
    else:
        assert "source report unavailable" in observed["readback_error"]
        assert observed["failure_work_may_be_unobserved"] is True


@pytest.mark.parametrize("body", ["success", "failure"])
def test_should_stop_after_arm_readback_fault_and_keep_original_failure_if_present(
    monkeypatch, evaluator, body
):
    captured = pilot_environment(monkeypatch, evaluator)
    original = pilot._readback
    calls: dict[int, int] = {}

    def readback(row, source, initial, stream, budget):
        original(row, source, initial, stream, budget)
        identity = id(row)
        calls[identity] = calls.get(identity, 0) + 1
        if calls[identity] == (2 if body == "success" else 1):
            raise RuntimeError("arm readback secondary")

    monkeypatch.setattr(pilot, "_readback", readback)
    if body == "failure":
        evaluator.work["fail_train"] = lambda owner: True
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    assert len(captured["streams"]) == 1
    assert not captured["evaluation_calls"] and not evaluator.work.get("environments")
    first = report["methods"][0]
    assert "arm readback secondary" in first["readback_error"]
    assert all(row["status"] == "not_run" for row in report["methods"][1:])
    if body == "success":
        assert "arm readback secondary" in report["reason"]
        assert captured["streams"][0][3].updates_completed == first["charged_updates"] == 128
        assert len(captured["finished"]) == 1
    else:
        assert "fake training failed" in report["reason"]
        assert "fake training failed" in first["reason"]
        assert "arm readback secondary" not in first["reason"]
        assert captured["streams"][0][3].updates_completed == first["charged_updates"] == 0
        assert len(evaluator.work["trains"]) == 1


def test_should_preserve_outer_primary_error_when_completed_arm_final_readback_fails(
    monkeypatch, evaluator
):
    captured = pilot_environment(monkeypatch, evaluator)
    learner = pilot._learner
    readback = pilot._readback
    peer_failed: list[bool] = []

    def create(method):
        if method == "pc":
            peer_failed.append(True)
            raise RuntimeError("peer factory primary")
        return learner(method)

    def final_readback(row, source, initial, stream, budget):
        readback(row, source, initial, stream, budget)
        if peer_failed and row["method"] == "bp":
            raise RuntimeError("outer readback secondary")

    monkeypatch.setattr(pilot, "_learner", create)
    monkeypatch.setattr(pilot, "_readback", final_readback)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    assert "peer factory primary" in report["reason"]
    assert "outer readback secondary" not in report["reason"]
    assert "outer readback secondary" in report["readback_error"]
    assert report["methods"][0]["charged_updates"] == 128
    assert report["methods"][1]["status"] == "incomplete"
    assert len(captured["streams"]) == len(captured["finished"]) == 1
    assert not captured["evaluation_calls"] and not evaluator.work.get("environments")


@pytest.mark.parametrize("fault", ["equal_observation", "role", "reward_type", "array_values"])
def test_should_refuse_replaced_immutable_episode_leaves_or_changed_array_bytes(fault):
    packet = source_batch("A", "blue")
    packets = (packet,)
    frozen = pilot._signature(packets)
    assert pilot._check_signature(packets, frozen) is True
    episode = packet.episode
    assert episode is not None
    if fault == "equal_observation":
        transition = episode.transitions[0]
        replacement = tuple(value for value in transition.observation)
        assert replacement == transition.observation and replacement is not transition.observation
        object.__setattr__(transition, "observation", replacement)
    elif fault == "role":
        object.__setattr__(episode, "role", "final_test")
    elif fault == "reward_type":
        assert episode.transitions[0].reward == 0.0
        object.__setattr__(episode.transitions[0], "reward", 0)
    else:
        packet.features[0, 0] = 0.0
    assert pilot._check_signature(packets, frozen) is False


@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_only_explicit_minigrid_learning_cli(monkeypatch, capsys, describe):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-minigrid-cue-key-door"]
        + (["--describe"] if describe else []),
    )
    calls = []

    def selected():
        calls.append(True)
        return dict(status="complete", selected="actual-minigrid-learning")

    monkeypatch.setattr(cli, "describe_minigrid_protocol", selected if describe else forbidden)
    monkeypatch.setattr(cli, "run_minigrid_pilot", forbidden if describe else selected)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    cli.main()
    assert calls == [True]
    assert json.loads(capsys.readouterr().out)["selected"] == "actual-minigrid-learning"
