"""Train-only live caller admission, completed-stream binding and partial evidence."""

from copy import deepcopy
import json
import math
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.key_door_pilot as shared
import src.adapters.train_acquisition_pilot as pilot
import src.adapters.train_policy_diagnostics as train_live
from test_recurrent_key_door_pilot import Learner as RecurrentLearner
from test_torch_feature_rehearsal_pilot import forbidden
from test_train_acquisition_pilot import Learner, environment as acquisition_environment


class Session:
    work: dict[str, Any] = {}

    def __init__(self, owner):
        self.owner = owner
        self.state = owner.snapshot_state()
        self.completed_steps = 0
        self.inputs: list[np.ndarray] = []

    def snapshot_state(self):
        return deepcopy(self.state)

    def predict(self, features):
        assert features.shape == (1, 8) and features.dtype == np.float64
        self.inputs.append(features.copy())
        self.completed_steps += 1
        callback = self.work.get("on_prediction")
        if callback is not None:
            callback(self)
        return np.array([[0.6, 0.1, 0.1, 0.1, 0.1]], dtype=np.float64)


def environment(monkeypatch):
    captured = acquisition_environment(monkeypatch)
    captured.update(
        actual_streams=[],
        finished=[],
        sessions=[],
        diagnostics=[],
        teacher_inputs=[],
        on_prediction=None,
    )
    original_stream = pilot.AttributionStream
    original_diagnostic = train_live.evaluate_train_policy
    original_collect = shared._collect

    def collect(*args, **kwargs):
        result = original_collect(*args, **kwargs)
        captured["teacher_inputs"].append(result)
        return result

    def stream(*args, **kwargs):
        result = original_stream(*args, **kwargs)
        captured["actual_streams"].append(result)
        original_finish = result.finish

        def finish():
            returned = original_finish()
            captured["finished"].append(deepcopy(returned))
            return returned

        monkeypatch.setattr(result, "finish", finish)
        return result

    def start_episode(owner):
        assert len(RecurrentLearner.trains) == 64, "live diagnostics began before all64updates"
        assert len(captured["finished"]) == 1, "live diagnostics preceded stream.finish"
        session = Session(owner)
        captured["sessions"].append(session)
        return session

    def diagnostic(report, source, states, reference, budget, **kwargs):
        assert len(captured["finished"]) == 1 and len(RecurrentLearner.trains) == 64
        assert [point for point, _ in states] == [0, 64]
        captured["diagnostics"].append((report, source, states, reference, budget))
        return original_diagnostic(report, source, states, reference, budget, **kwargs)

    monkeypatch.setattr(pilot, "AttributionStream", stream)
    monkeypatch.setattr(shared, "_collect", collect)
    monkeypatch.setattr(Learner, "start_episode", start_episode)
    monkeypatch.setattr(Session, "work", captured)
    monkeypatch.setattr(train_live, "evaluate_train_policy", diagnostic)
    monkeypatch.setattr(train_live, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))
    return captured


def test_should_finish_unchanged_train_stream_before_four_owned_live_episodes_and_prior(
    monkeypatch,
):
    captured = environment(monkeypatch)
    global_rng = pickle.dumps(np.random.get_state())
    report = pilot.run_live_pilot()
    assert report["status"] == "complete", report
    assert report["charged_updates"] == len(RecurrentLearner.trains) == 64
    assert report["stream_result"]["completed_updates"] == 64
    assert report["stream_result"]["training_examples"] == 512
    assert len(report["stream_result"]["applied_updates"]) == 64
    assert len(captured["collections"]) == 2 and len(captured["finished"]) == 1
    isolated, training, budget = captured["streams"][0]
    runtime = captured["actual_streams"][0]._runtime
    assert budget.updates_completed == 64 and budget.replay_examples_completed == 0
    assert budget.budget.max_training_updates == 64
    assert len(training) == 64 and len({batch.episode.episode_id for batch in training}) == 64
    assert all(batch.role == batch.episode.role == "train" for batch in training)
    assert all(batch.role == batch.episode.role == "train" for batch in captured["collections"])
    assert [point["updates"] for point in report["checkpoints"]] == [0, 8, 16, 32, 64]
    assert len(RecurrentLearner.classifications) == 10
    assert len(captured["sessions"]) == 4
    assert [session.state.completed_updates for session in captured["sessions"]] == [0, 0, 64, 64]
    for session in captured["sessions"]:
        assert session.owner is not Learner.sources[0] and session.owner is not isolated
        assert (
            session.owner is not runtime._candidate and session.owner is not runtime.actor._learner
        )
        assert len(session.inputs) == session.completed_steps == 12
        assert session.inputs[0][0, 6] in (-1, 1)
        assert all(features[0, 6] == 0 for features in session.inputs[1:])
    for field in ("source_unchanged", "isolated_source_unchanged", "actor_unchanged"):
        assert report[field] is True
    assert report["partial_stream"]["native_candidate_updates"] == 64
    assert report["partial_stream"]["native_candidate_training_examples"] == 512
    assert "policy_evaluation" not in report and "probes" not in report and "replay" not in report
    diagnostics = report["train_policy_diagnostics"]
    assert diagnostics["status"] == "complete"
    assert (
        diagnostics["source_role"] == "train"
        and diagnostics["classification"] == "train_resubstitution"
    )
    assert diagnostics["outcomes"] == [
        dict(updates=0, success_rate=0.0, mean_return=0.0),
        dict(updates=64, success_rate=0.0, mean_return=0.0),
    ]
    assert len(captured["diagnostics"]) == 1
    assert captured["diagnostics"][0][4] is budget
    assert captured["diagnostics"][0][3] is report["majority_control"]
    for slot in diagnostics["episodes"]:
        assert slot["status"] == "complete" and slot["state_unchanged"] is True
        assert slot["action_conditioning"] == "legal_actions_v1"
        assert slot["outcome"] == "step_limit" and slot["return"] == 0
        assert all(step["action"] in step["legal_actions"] for step in slot["transitions"])
    reference = diagnostics["reference"]
    expected_prior = [0.5, 0.125, 0.0, 0.125, 0.25]
    assert (
        reference["probabilities"] == report["majority_control"]["probabilities"] == expected_prior
    )
    assert len(reference["episodes"]) == 2 and reference["outcomes"] is None
    for slot in reference["episodes"]:
        assert slot["status"] == "refused" and slot["refusal"] == "zero_legal_mass"
        assert slot["success"] is None and slot["return"] == 0
        last = slot["transitions"][-1]
        assert last["raw_probabilities"] == tuple(expected_prior)
        assert math.fsum(expected_prior[action] for action in last["legal_actions"]) == 0
        assert last["probabilities"] is last["action"] is None and last["applied"] is False
        assert slot["prediction_calls_returned"] == slot["steps_executed"] + 1
    assert (
        diagnostics["comparison"] is None
        and "prior_zero_legal_mass" in diagnostics["comparison_reason"]
    )
    assert diagnostics["work"] == dict(
        learner_callback_attempts=48,
        learner_callback_returns=48,
        learner_applied_steps=48,
        learner_sessions_started=4,
        prior_callback_attempts=8,
        prior_callback_returns=8,
        prior_applied_steps=6,
        failure_work_may_be_unobserved=False,
    )
    assert report["scientific_advantage_verified"] is False
    assert pickle.dumps(np.random.get_state()) == global_rng


def test_should_preserve_training_failure_and_skip_all_live_work(monkeypatch):
    captured = environment(monkeypatch)
    monkeypatch.setattr(Learner, "fail_third", True)
    report = pilot.run_live_pilot()
    assert report["status"] == "incomplete" and "third train-only" in report["reason"]
    assert report["charged_updates"] == 2
    assert len(RecurrentLearner.trains) == 2 and len(Learner.training_owners) == 3
    assert captured["finished"] == captured["sessions"] == []
    assert captured["diagnostics"] == []
    assert report["partial_stream"]["native_candidate_updates"] == 2
    assert len(report["partial_stream"]["applied_updates"]) == 2


@pytest.mark.parametrize(
    "target",
    ["source", "actor", "candidate", "budget", "checkpoint", "receipt", "reference", "teacher"],
)
def test_should_stop_live_callbacks_when_completed_training_authority_changes(monkeypatch, target):
    captured = environment(monkeypatch)

    def corrupt(session):
        stream = captured["actual_streams"][0]
        if target == "source":
            Learner.sources[0].state.parameters["weight_input"][0, 0] += 1
        elif target == "actor":
            stream._runtime.actor._learner.state.parameters["weight_input"][0, 0] += 1
        elif target == "candidate":
            stream._runtime._candidate.state.optimizer_state["weight_input"][0, 0] += 1
        elif target == "budget":
            captured["streams"][0][2].updates_completed = 63
        elif target == "checkpoint":
            stream._states[0][1].parameters["weight_input"][0, 0] += 1
        elif target == "receipt":
            receipt = stream._runtime.applied_updates[0]
            object.__setattr__(receipt.diagnostic, "value", receipt.diagnostic.value + 1)
        elif target == "reference":
            captured["diagnostics"][0][3]["probabilities"][0] -= 0.125
            captured["diagnostics"][0][3]["probabilities"][2] += 0.125
        else:
            captured["teacher_inputs"][0].features[0, 0] += 1

    captured["on_prediction"] = corrupt
    report = pilot.run_live_pilot()
    assert report["status"] == "incomplete"
    assert len(RecurrentLearner.trains) == 64 and len(captured["finished"]) == 1
    assert len(captured["sessions"]) == 1 and len(captured["sessions"][0].inputs) == 1
    assert len(report["partial_stream"]["applied_updates"]) == 64
    assert report["partial_stream"]["native_candidate_updates"] == 64
    assert report["charged_updates"] == (63 if target == "budget" else 64)
    diagnostics = report["train_policy_diagnostics"]
    assert diagnostics["status"] == "failed" and diagnostics["live_sessions_started"] == 1
    assert all(slot["status"] == "not_run" for slot in diagnostics["episodes"][1:])
    assert all(slot["status"] == "not_run" for slot in diagnostics["reference"]["episodes"])


def test_should_preserve_live_failure_after_spent_prediction_without_more_learning(monkeypatch):
    captured = environment(monkeypatch)

    def fail(session):
        if len(session.inputs) == 2:
            raise RuntimeError("second train-only live prediction failed")

    captured["on_prediction"] = fail
    report = pilot.run_live_pilot()
    assert report["status"] == "incomplete"
    assert report["charged_updates"] == len(RecurrentLearner.trains) == 64
    assert len(captured["sessions"]) == 1
    assert len(captured["sessions"][0].inputs) == 2
    assert report["stream_result"]["completed_updates"] == 64
    assert len(report["stream_result"]["applied_updates"]) == 64
    diagnostics = report["train_policy_diagnostics"]
    first = diagnostics["episodes"][0]
    assert diagnostics["status"] == "failed"
    assert first["prediction_calls_attempted"] == 2 and first["prediction_calls_returned"] == 1
    assert first["steps_executed"] == 1
    assert all(slot["status"] == "not_run" for slot in diagnostics["episodes"][1:])


def test_should_refuse_changed_selected_checkpoint_copy_before_any_live_session(monkeypatch):
    captured = environment(monkeypatch)
    original = train_live.evaluate_train_policy

    def replace_checkpoint(report, source, states, reference, budget, **kwargs):
        states[0][1].parameters["weight_input"][0, 0] += 1
        return original(report, source, states, reference, budget, **kwargs)

    monkeypatch.setattr(train_live, "evaluate_train_policy", replace_checkpoint)
    report = pilot.run_live_pilot()
    assert report["status"] == "incomplete"
    assert report["charged_updates"] == len(RecurrentLearner.trains) == 64
    assert len(captured["finished"]) == 1 and captured["sessions"] == []
    assert report["stream_result"]["completed_updates"] == 64
    assert report["train_policy_diagnostics"]["status"] == "failed"


def test_should_describe_explicit_train_live_cost_without_resources(monkeypatch):
    for name in (
        "TorchRecurrentLearner",
        "ProcessRssSampler",
        "require_torch",
        "AttributionStream",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    protocol = pilot.describe_live_protocol()
    assert protocol["protocol_id"] == "key_door_train_only_closed_loop_acquisition_v1"
    assert protocol["architecture"] == dict(input_dim=8, hidden_dim=8, num_classes=5, seed=17)
    assert protocol["checkpoint_updates"] == [0, 8, 16, 32, 64]
    assert protocol["planned_work"]["native_train_calls"] == 64
    assert protocol["planned_work"]["training_rows"] == 512
    assert protocol["planned_work"]["recurrent_unroll_steps"] == 640
    assert protocol["planned_work"]["maximum_native_predict_calls"] == 58
    assert protocol["planned_work"]["maximum_prior_callbacks"] == 24
    assert protocol["planned_work"]["environment_constructions"] == 8
    assert protocol["planned_work"]["maximum_environment_steps"] == 88
    assert pilot.describe_protocol()["protocol_id"] == "key_door_train_only_acquisition_v1"


def test_should_keep_ordinary_default_route_free_of_live_work_and_new_fields(monkeypatch):
    captured = environment(monkeypatch)
    report = pilot.run_pilot()
    assert report["status"] == "complete"
    assert "train_policy_diagnostics" not in report
    assert captured["sessions"] == [] and len(RecurrentLearner.trains) == 64


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_native_live_mode_bool_before_work(monkeypatch, invalid):
    for name in (
        "TorchRecurrentLearner",
        "ProcessRssSampler",
        "require_torch",
        "AttributionStream",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    with pytest.raises(ValueError):
        pilot._run_pilot(train_policy_diagnostics=invalid)


@pytest.mark.parametrize(
    "mix",
    [
        dict(return_context=True),
        dict(evaluate_policy_after_training=True),
        dict(episode_replay=True),
        dict(fresh_task_probes=True),
        dict(four_phase=True),
        dict(four_phase_policy=True),
        dict(four_phase_replay="coverage"),
        dict(four_phase_probes=True),
    ],
)
def test_should_refuse_live_mode_mixed_with_other_recipes_before_work(monkeypatch, mix):
    for name in (
        "TorchRecurrentLearner",
        "ProcessRssSampler",
        "require_torch",
        "AttributionStream",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    with pytest.raises(ValueError):
        pilot._run_pilot(train_policy_diagnostics=True, **mix)


@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_only_explicit_train_live_cli_without_arguments(
    monkeypatch, capsys, describe
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-key-door-train-live"] + (["--describe"] if describe else []),
    )
    calls = []

    def selected():
        calls.append(True)
        return dict(status="complete", selected="train-live")

    monkeypatch.setattr(cli, "describe_train_live_protocol", selected if describe else forbidden)
    monkeypatch.setattr(cli, "run_train_live_pilot", forbidden if describe else selected)
    monkeypatch.setattr(cli, "run_train_acquisition_pilot", forbidden)
    monkeypatch.setattr(cli, "run_train_context_return_pilot", forbidden)
    cli.main()
    assert calls == [True]
    assert json.loads(capsys.readouterr().out)["selected"] == "train-live"


def test_should_refuse_normalized_prior_replacement_during_training_before_diagnostic(monkeypatch):
    captured = environment(monkeypatch)
    references: list[dict[str, Any]] = []
    original_reference, original_train = pilot._reference, Learner.train_batch

    def reference(base):
        result = original_reference(base)
        references.append(result)
        return result

    def train(self, *args):
        result = original_train(self, *args)
        if self.snapshot_state().completed_updates == 1:
            references[0]["probabilities"][0] -= 0.125
            references[0]["probabilities"][2] += 0.125
        return result

    monkeypatch.setattr(pilot, "_reference", reference)
    monkeypatch.setattr(Learner, "train_batch", train)
    report = pilot.run_live_pilot()
    assert report["status"] == "incomplete"
    assert "prior differs from original empirical train labels" in report["reason"]
    assert report["charged_updates"] == len(RecurrentLearner.trains) == 64
    assert len(captured["finished"]) == 1
    assert captured["sessions"] == captured["diagnostics"] == []
