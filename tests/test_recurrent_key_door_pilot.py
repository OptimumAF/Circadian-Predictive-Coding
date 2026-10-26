"""Existing key/door orchestration with owned fake recurrent episode sessions."""

from copy import deepcopy
from dataclasses import dataclass
import json
import pickle
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.key_door_pilot as pilot
from src.core.learner_ports import TrainingDiagnostic
from test_torch_feature_rehearsal_pilot import Sampler, TorchStub, environment, forbidden


@dataclass
class State:
    parameters: dict[str, Any]
    optimizer_state: dict[str, Any]
    gradients: dict[str, Any]
    completed_updates: int = 0
    training_examples: int = 0


class Session:
    def __init__(self, learner):
        self.state = learner.snapshot_state()
        self.completed_steps = 0
        self.inputs: list[np.ndarray] = []

    def snapshot_state(self):
        return deepcopy(self.state)

    def predict(self, features):
        assert features.shape == (1, 8) and features.dtype == np.float64
        self.inputs.append(features.copy())
        self.completed_steps += 1
        if Learner.change_session_state:
            self.state.completed_updates += 1
        return np.array([[0.6, 0.1, 0.1, 0.1, 0.1]], dtype=np.float64)


class Learner:
    sessions: list[Session] = []
    trains: list[tuple[np.ndarray, np.ndarray]] = []
    classifications: list[np.ndarray] = []
    change_session_state = False
    fail_session_copy = False

    def __init__(self, policy):
        self.policy = policy
        shapes = dict(
            weight_input=(8, 8),
            weight_recurrent=(8, 8),
            bias_hidden=(1, 8),
            weight_output=(8, 5),
            bias_output=(1, 5),
        )
        self.state = State(
            {key: np.zeros(shape, dtype=np.float32) for key, shape in shapes.items()},
            {},
            {key: None for key in shapes},
        )

    def snapshot_state(self):
        return deepcopy(self.state)

    def restore_state(self, state):
        self.state = deepcopy(state)

    def fork(self):
        return deepcopy(self)

    def train_batch(self, features, targets):
        assert features.shape == (8, 8) and targets.shape == (8, 1)
        assert features.dtype == targets.dtype == np.float64
        assert features[0, 6] in (-1, 1) and np.all(features[1:, 6] == 0)
        assert np.all(targets == np.floor(targets)) and np.all((targets >= 0) & (targets < 5))
        Learner.trains.append((features.copy(), targets.copy()))
        self.state.completed_updates += 1
        self.state.training_examples += 8
        for key, tensor in self.state.parameters.items():
            tensor += float(np.mean(features)) * 0.01
            self.state.gradients[key] = np.full_like(tensor, 0.1)
        self.state.optimizer_state = {
            key: tensor.copy() for key, tensor in self.state.gradients.items()
        }
        return TrainingDiagnostic("fake_recurrent_episode_ce_v1", 0.5)

    def predict(self, features):
        assert features.shape == (8, 8)
        Learner.classifications.append(features.copy())
        return np.tile(np.array([[0.6, 0.1, 0.1, 0.1, 0.1]]), (len(features), 1))

    def start_episode(self):
        if Learner.fail_session_copy:
            raise RuntimeError("session copy failed")
        result = Session(self)
        Learner.sessions.append(result)
        return result


def observation(updates, state):
    def size(mapping):
        return sum(value.nbytes for value in mapping.values() if isinstance(value, np.ndarray))

    return dict(
        wake_updates=updates,
        native_updates=state.completed_updates,
        native_training_examples=state.training_examples,
        parameters={key: value.tolist() for key, value in state.parameters.items()},
        head_tensor_bytes=size(state.parameters),
        optimizer_tensor_bytes=size(state.optimizer_state),
        gradient_tensor_bytes=size(state.gradients),
        architecture="elman_tanh",
        activation_tensor_bytes=32,
        training_unroll_steps=state.training_examples,
    )


def fake_environment(monkeypatch):
    environment(monkeypatch)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "require_torch", lambda: TorchStub)
    monkeypatch.setattr(pilot, "TorchFeatureLearner", forbidden)
    monkeypatch.setattr(pilot, "TorchRecurrentLearner", Learner)
    monkeypatch.setattr(pilot, "_recurrent_observation", observation)
    monkeypatch.setattr(pilot, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))
    monkeypatch.setattr(Learner, "sessions", [])
    monkeypatch.setattr(Learner, "trains", [])
    monkeypatch.setattr(Learner, "classifications", [])
    monkeypatch.setattr(Learner, "change_session_state", False)
    monkeypatch.setattr(Learner, "fail_session_copy", False)


def test_should_describe_frozen_recurrent_work_without_native_or_collection_calls(monkeypatch):
    for name in (
        "build_stream",
        "KeyDoorNavigation",
        "TorchFeatureLearner",
        "TorchRecurrentLearner",
        "require_torch",
        "ProcessRssSampler",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    protocol = pilot.describe_recurrent_protocol()
    assert protocol["methods"] == ["recurrent_bc"]
    assert protocol["replay_modes"] == [False, True]
    assert protocol["planned_work"]["native_train_calls"] == 20
    assert protocol["planned_work"]["training_rows"] == 160
    assert protocol["planned_work"]["score_forwards"] == 80
    assert protocol["planned_work"]["live_episodes"] == 80
    assert protocol["planned_work"]["maximum_live_forwards"] == 960
    assert protocol["seed"] == 17 and protocol["feature_dim"] == protocol["hidden_dim"] == 8
    assert protocol["num_classes"] == 5
    assert pilot.describe_protocol()["planned_work"]["native_train_calls"] == 80
    assert pilot.describe_protocol()["planned_work"]["live_episodes"] == 320


def test_should_report_recurrent_checkpoint_geometry_activation_and_separate_storage(monkeypatch):
    source = Learner(SimpleNamespace(hidden_dim=8))
    state = source.snapshot_state()
    state.completed_updates, state.training_examples = 3, 24
    state.gradients = {key: np.ones_like(value) for key, value in state.parameters.items()}
    state.optimizer_state = {key: value.copy() for key, value in state.gradients.items()}

    def bytes_in(value):
        if isinstance(value, np.ndarray):
            return value.nbytes
        if isinstance(value, dict):
            return sum(bytes_in(item) for item in value.values())
        return 0

    monkeypatch.setattr(pilot, "_tensor_bytes", bytes_in)
    record = pilot._recurrent_observation(2, SimpleNamespace(**vars(state), policy=source.policy))
    expected = (8 * 8 + 8 * 8 + 8 + 8 * 5 + 5) * 4
    assert record["head_tensor_bytes"] == expected
    assert record["optimizer_tensor_bytes"] == record["gradient_tensor_bytes"] == expected
    assert record["activation_tensor_bytes"] == 32
    assert record["training_unroll_steps"] == record["native_training_examples"] == 24
    assert record["wake_updates"] == 2 and record["native_updates"] == 3
    assert record["architecture"] == "elman_tanh"
    assert record["parameters"] == {key: value.tolist() for key, value in state.parameters.items()}


def test_should_run_two_arms_and_all_eighty_fresh_sessions_with_honest_negative_outcomes(
    monkeypatch,
):
    fake_environment(monkeypatch)
    report = pilot.run_recurrent_pilot()
    assert report["status"] == "complete", report["runs"]
    assert report["global_errors"] == []
    assert len(report["runs"]) == 2 and len(report["demonstrations"]) == 16
    assert all(row["method"] == "recurrent_bc" for row in report["runs"])
    assert report["pc_neutral_exact"] is report["shared_replay_plans_exact"] is None
    assert report["shared_initial_parameters_exact"] is True
    assert len(Learner.trains) == 20 and sum(len(x) for x, _ in Learner.trains) == 160
    assert len(Learner.classifications) == 80
    episodes = [episode for row in report["runs"] for episode in row["episodes"]]
    assert len(episodes) == len(Learner.sessions) == 80
    assert len({id(session) for session in Learner.sessions}) == 80
    assert all(episode["outcome"] == "illegal_action" for episode in episodes)
    assert all(episode["success"] is False and episode["return"] == 0 for episode in episodes)
    assert all(episode["state_unchanged"] is True for episode in episodes)
    assert sum(e["layout_partition"] == "heldout_layout" for e in episodes) == 20
    for session, episode in zip(Learner.sessions, episodes, strict=True):
        assert (
            len(session.inputs) == session.completed_steps == episode["prediction_calls_attempted"]
        )
        assert session.inputs[0][0, 6] == episode["cue"]
        assert all(features[0, 6] == 0 for features in session.inputs[1:])
        assert episode["identity"]["method"] == "recurrent_bc"
        assert episode["phase"] == "evaluation" and episode["learning_active"] is False
    assert sum(session.completed_steps for session in Learner.sessions) == 200
    for row in report["runs"]:
        expected_updates = [0, 3, 6, 9, 12] if row["replay_enabled"] else [0, 2, 4, 6, 8]
        assert [point["native_updates"] for point in row["observations"]] == expected_updates
        assert all(point["architecture"] == "elman_tanh" for point in row["observations"])
        assert row["wake_updates"] == 8
        assert row["replay_charged_examples"] == (32 if row["replay_enabled"] else 0)
        assert row["source_unchanged"] is row["actor_unchanged"] is True
        work = row["work"]
        assert work["observed_checkpoint_training_unroll_steps"] == (
            96 if row["replay_enabled"] else 64
        )
        assert work["classification_unroll_steps"] == 320
        assert work["live_callback_attempts"] == work["live_callback_returns"] == 100
        assert work["live_applied_steps"] == 60 and work["live_sessions_started"] == 40
        assert work["failure_work_may_be_unobserved"] is False
    assert report["scientific_advantage_verified"] is report["full_R42_complete"] is False


def test_should_stop_changed_session_state_and_preserve_partial_and_unreached_slots(monkeypatch):
    fake_environment(monkeypatch)
    monkeypatch.setattr(Learner, "change_session_state", True)
    report = pilot.run_recurrent_pilot()
    assert report["status"] == "incomplete"
    assert len(Learner.sessions) == 2
    for row in report["runs"]:
        assert row["status"] == "failed" and row["wake_updates"] == 8
        first = row["episodes"][0]
        assert first["status"] == "failed" and first["prediction_calls_attempted"] == 1
        assert "state changed" in first["reason"]
        assert all(slot["status"] == "not_run" for slot in row["episodes"][1:])
        assert row["source_unchanged"] is row["actor_unchanged"] is True


def test_should_keep_primary_live_failure_and_secondary_readback_with_permanent_costs(monkeypatch):
    fake_environment(monkeypatch)

    def readback():
        raise RuntimeError("secondary recurrent readback")

    def fail(row, source, states, budget):
        monkeypatch.setattr(source, "snapshot_state", readback)
        raise RuntimeError("primary recurrent live fault")

    monkeypatch.setattr(pilot, "_live", fail)
    report = pilot.run_recurrent_pilot()
    assert report["status"] == "incomplete"
    for row in report["runs"]:
        assert row["wake_updates"] == 8
        assert row["replay_charged_examples"] == (32 if row["replay_enabled"] else 0)
        assert "primary recurrent live fault" in row["reason"]
        assert "secondary recurrent readback" in row["readback_error"]
        assert all(slot["status"] == "not_run" for slot in row["episodes"])


def test_should_keep_all_planned_slots_when_fresh_session_copy_fails(monkeypatch):
    fake_environment(monkeypatch)
    monkeypatch.setattr(Learner, "fail_session_copy", True)
    report = pilot.run_recurrent_pilot()
    assert report["status"] == "incomplete"
    assert len(report["runs"]) == 2
    for row in report["runs"]:
        assert "session copy failed" in row["reason"]
        assert row["wake_updates"] == 8 and len(row["episodes"]) == 40
        assert row["episodes"][0]["status"] == "failed"
        assert all(slot["status"] == "not_run" for slot in row["episodes"][1:])
        assert row["source_unchanged"] is row["actor_unchanged"] is True


def test_should_preserve_all_eighty_slots_when_resource_setup_fails(monkeypatch):
    def fail():
        raise RuntimeError("resource setup")

    monkeypatch.setattr(pilot, "ProcessRssSampler", fail)
    report = pilot.run_recurrent_pilot()
    assert report["status"] == "incomplete" and report["global_errors"]
    assert len(report["runs"]) == 2
    assert sum(len(row["episodes"]) for row in report["runs"]) == 80
    assert all(row["status"] == "not_run" for row in report["runs"])


@pytest.mark.parametrize("describe", [False, True])
def test_should_route_only_selected_recurrent_key_door_cli(monkeypatch, capsys, describe):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-key-door-recurrent"] + (["--describe"] if describe else []),
    )
    selected = "describe_recurrent_protocol" if describe else "run_recurrent_pilot"
    for name in ("describe_recurrent_protocol", "run_recurrent_pilot"):
        monkeypatch.setattr(
            cli,
            name,
            (lambda: dict(status="complete", selected="recurrent"))
            if name == selected
            else forbidden,
        )
    monkeypatch.setattr(cli, "describe_key_door_protocol", forbidden)
    monkeypatch.setattr(cli, "run_key_door_pilot", forbidden)
    cli.main()
    assert json.loads(capsys.readouterr().out)["selected"] == "recurrent"
