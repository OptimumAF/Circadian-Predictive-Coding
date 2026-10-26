"""Real actor/inbox reward routing with owned fake native recurrent models."""

from copy import deepcopy
import json
import pickle
import sys

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.key_door_pilot as shared
import src.adapters.key_door_reward_pilot as pilot
from src.adapters.torch_reward_update_learner import RewardUpdate
from src.core.learner_ports import TrainingDiagnostic
from test_recurrent_key_door_pilot import State, fake_environment
from test_torch_feature_rehearsal_pilot import Sampler, TorchStub, forbidden


class Session:
    def __init__(self, source):
        self.state = source.snapshot_state()
        self.inputs: list[np.ndarray] = []
        self.completed_steps = 0
        self.number = len(Learner.sessions) + 1

    def snapshot_state(self):
        return deepcopy(self.state)

    def predict(self, features):
        assert features.shape == (1, 8) and features.dtype == np.float64
        self.inputs.append(features.copy())
        if Learner.fail_collection and self.number == 1 and len(self.inputs) == 2:
            raise RuntimeError("second sampled prediction failed")
        self.completed_steps += 1
        return np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])


class Learner:
    sessions: list[Session] = []
    updates: list[RewardUpdate] = []
    identifiers: list[str] = []
    classifications: list[np.ndarray] = []
    sources: list["Learner"] = []
    fail_collection = False
    fail_update = False

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
        Learner.sources.append(self)

    @property
    def completed_updates(self):
        return self.state.completed_updates

    @property
    def training_examples(self):
        return self.state.training_examples

    def snapshot_state(self):
        return deepcopy(self.state)

    def restore_state(self, state):
        self.state = deepcopy(state)

    def fork(self):
        return deepcopy(self)

    def train_batch(self, update, identifier):
        assert type(update) is RewardUpdate
        assert type(identifier) is str and identifier == update.episode.episode_id
        assert update.episode.role == "train"
        assert pickle.dumps(update.behavior_state) == pickle.dumps(self.state)
        Learner.updates.append(deepcopy(update))
        Learner.identifiers.append(identifier)
        if Learner.fail_update and len(Learner.updates) == 3:
            raise RuntimeError("third atomic reward update failed")
        self.state.completed_updates += 1
        self.state.training_examples += len(update.episode.transitions)
        for key, value in self.state.parameters.items():
            self.state.gradients[key] = np.zeros_like(value)
        self.state.optimizer_state = {
            key: value.copy() for key, value in self.state.gradients.items()
        }
        return TrainingDiagnostic("torch_recurrent_reward_logprob_sum_v1", 0.0)

    def predict(self, features):
        assert features.shape == (8, 8)
        Learner.classifications.append(features.copy())
        return np.tile(np.array([[1.0, 0.0, 0.0, 0.0, 0.0]]), (len(features), 1))

    def start_episode(self):
        result = Session(self)
        Learner.sessions.append(result)
        return result


def environment(monkeypatch):
    fake_environment(monkeypatch)
    monkeypatch.setattr(pilot, "TorchRewardUpdateLearner", Learner)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "require_torch", lambda: TorchStub)
    monkeypatch.setattr(
        pilot, "_same_state", lambda left, right: pickle.dumps(left) == pickle.dumps(right)
    )
    for name in ("sessions", "updates", "identifiers", "classifications", "sources"):
        monkeypatch.setattr(Learner, name, [])
    monkeypatch.setattr(Learner, "fail_collection", False)
    monkeypatch.setattr(Learner, "fail_update", False)


def test_should_describe_frozen_reward_route_without_native_or_collection_calls(monkeypatch):
    for name in (
        "TorchRewardUpdateLearner",
        "ProcessRssSampler",
        "require_torch",
        "collect_key_door_reward_episode",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "KeyDoorNavigation", forbidden)
    protocol = pilot.describe_protocol()
    assert protocol["methods"] == ["recurrent_rl"]
    assert protocol["replay_modes"] == [False]
    assert (
        protocol["architecture"]
        == "Elman tanh/full episode BPTT/reward logprob sum; zero hidden per packet"
    )
    assert protocol["seed"] == 17
    assert protocol["planned_work"]["native_train_calls"] == 8
    assert protocol["planned_work"]["live_episodes"] == 40
    assert protocol["planned_work"]["maximum_recurrent_unroll_steps"] == 992
    assert shared.describe_protocol()["planned_work"]["native_train_calls"] == 80
    assert shared.describe_recurrent_protocol()["planned_work"]["native_train_calls"] == 20


def test_should_train_only_sampled_reward_records_through_original_runtime_and_keep_all_negatives(
    monkeypatch,
):
    environment(monkeypatch)
    report = pilot.run_pilot()
    assert report["status"] == "complete", report["runs"]
    assert report["global_errors"] == [] and len(report["runs"]) == 1
    assert len(report["demonstrations"]) == 8
    assert all(episode["role"] == "final_test" for episode in report["demonstrations"])
    row = report["runs"][0]
    assert row["method"] == "recurrent_rl" and row["replay_enabled"] is False
    assert row["wake_updates"] == 8 and row["source_unchanged"] is row["actor_unchanged"] is True
    assert [slot["context"] for slot in row["training"]] == list("AABBCCAA")
    assert [slot["cue"] for slot in row["training"]] == [1, -1] * 4
    assert [point["wake_updates"] for point in row["observations"]] == [0, 2, 4, 6, 8]
    assert [point["native_updates"] for point in row["observations"]] == [0, 2, 4, 6, 8]
    assert [point["native_training_examples"] for point in row["observations"]] == [
        0,
        8,
        10,
        12,
        20,
    ]
    assert len(Learner.updates) == 8 and len(set(Learner.identifiers)) == 8
    lengths = [len(update.episode.transitions) for update in Learner.updates]
    assert lengths == [4, 4, 1, 1, 1, 1, 4, 4]
    expected_rng = np.random.Generator(np.random.PCG64(29))
    for index, slot in enumerate(row["training"]):
        assert slot["status"] == "complete"
        assert (
            slot["update_attempted"] is slot["update_committed"] is slot["commit_validated"] is True
        )
        assert slot["receipt"] is not None
        assert (
            slot["charged_updates_before"] == index and slot["charged_updates_after"] == index + 1
        )
        assert slot["native_updates_before"] == index
        assert slot["native_training_examples_before"] == sum(lengths[:index])
        assert slot["sampler_before"] == expected_rng.bit_generator.state
        expected_rng.random(lengths[index])
        assert slot["sampler_after"] == expected_rng.bit_generator.state
        collection = slot["collection"]
        assert collection["status"] == "complete" and collection["termination"] == "illegal_action"
        assert collection["total_return"] == 0 and collection["sampling_draws"] == lengths[index]
        assert collection["behavior_native_updates"] == index
        assert collection["behavior_training_examples"] == sum(lengths[:index])
        assert collection["episode"]["role"] == "train"
        assert "behavior_state" not in collection
        assert all(step["reward"] == 0 for step in collection["episode"]["transitions"])
    result = row["stream_result"]
    assert result["completed_updates"] == 8 and result["training_examples"] == 20
    assert result["score_forward_calls"] == 40 and len(result["applied_updates"]) == 8
    assert [p["training_context"] for p in result["checkpoints"]] == [
        None,
        "A-blue",
        "B-blue",
        "C-blue",
        "A-blue",
    ]
    assert len(Learner.classifications) == 40 and len(Learner.sessions) == 48
    assert len(row["episodes"]) == 40 and row["live_sessions_started"] == 40
    assert all(
        episode["outcome"] == "illegal_action" and episode["state_unchanged"] is True
        for episode in row["episodes"]
    )
    assert sum(episode["prediction_calls_attempted"] for episode in row["episodes"]) == 100
    assert row["collection_attempts"] == 8
    assert row["work"]["observed_checkpoint_training_unroll_steps"] == 20
    assert row["work"]["classification_unroll_steps"] == 320
    assert row["work"]["live_callback_attempts"] == row["work"]["live_callback_returns"] == 100
    assert row["work"]["live_applied_steps"] == 60
    assert row["work"]["live_sessions_started"] == 40
    assert row["work"]["failure_work_may_be_unobserved"] is False
    assert report["scientific_advantage_verified"] is report["full_R42_complete"] is False


@pytest.mark.parametrize("phase", ["collection", "update"])
def test_should_keep_partial_cost_spent_rng_and_unreached_slots_when_reward_work_fails(
    monkeypatch, phase
):
    environment(monkeypatch)
    monkeypatch.setattr(
        Learner, "fail_collection" if phase == "collection" else "fail_update", True
    )
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    row = report["runs"][0]
    assert row["status"] == "failed" and len(row["training"]) == 8
    failed_index, committed, draws = (0, 0, 1) if phase == "collection" else (2, 2, 9)
    assert row["wake_updates"] == committed
    failed = row["training"][failed_index]
    assert failed["status"] == "failed" and failed["receipt"] is None
    assert failed["update_committed"] is False and failed["commit_validated"] is False
    assert failed["charged_updates_after"] == committed
    expected_rng = np.random.Generator(np.random.PCG64(29))
    expected_rng.random(draws)
    assert failed["sampler_after"] == expected_rng.bit_generator.state
    assert all(slot["status"] == "not_run" for slot in row["training"][failed_index + 1 :])
    assert all(slot["status"] == "not_run" for slot in row["episodes"])
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    if phase == "collection":
        assert failed["collection"]["prediction_calls_attempted"] == 2
        assert failed["collection"]["prediction_calls_returned"] == 1
        assert failed["collection"]["sampling_draws"] == 1
        assert failed["update_attempted"] is False
    else:
        assert failed["collection"]["status"] == "complete"
        assert failed["update_attempted"] is True
        assert "third atomic reward update failed" in failed["reason"]


def test_should_preserve_primary_runtime_error_and_secondary_source_readback(monkeypatch):
    environment(monkeypatch)

    def readback():
        raise RuntimeError("secondary reward source readback")

    def fail(owner, update, identifier):
        monkeypatch.setattr(Learner.sources[0], "snapshot_state", readback)
        raise RuntimeError("primary reward runtime fault")

    monkeypatch.setattr(Learner, "train_batch", fail)
    result = pilot.run_pilot()
    row = result["runs"][0]
    assert result["status"] == "incomplete" and row["status"] == "failed"
    assert "primary reward runtime fault" in row["reason"]
    assert "secondary reward source readback" in row["readback_error"]
    assert row["wake_updates"] == 0
    assert row["training"][0]["collection"]["sampling_draws"] == 4
    assert all(slot["status"] == "not_run" for slot in row["training"][1:])
    assert all(slot["status"] == "not_run" for slot in row["episodes"])


def test_should_preserve_eight_training_and_forty_live_slots_when_resource_setup_fails(monkeypatch):
    def fail():
        raise RuntimeError("reward resource setup")

    monkeypatch.setattr(pilot, "ProcessRssSampler", fail)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and report["global_errors"]
    assert len(report["runs"]) == 1
    row = report["runs"][0]
    assert row["status"] == "not_run" and row["wake_updates"] == 0
    assert len(row["training"]) == 8 and len(row["episodes"]) == 40
    assert all(slot["status"] == "not_run" for slot in (*row["training"], *row["episodes"]))


@pytest.mark.parametrize("describe", [False, True])
def test_should_route_only_explicit_reward_cli(monkeypatch, capsys, describe):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-key-door-rl"] + (["--describe"] if describe else []),
    )
    selected = "describe_reward_protocol" if describe else "run_reward_pilot"
    for name in ("describe_reward_protocol", "run_reward_pilot"):
        monkeypatch.setattr(
            cli,
            name,
            (lambda: dict(status="complete", selected="reward-only"))
            if name == selected
            else forbidden,
        )
    for name in (
        "describe_key_door_protocol",
        "run_key_door_pilot",
        "describe_recurrent_protocol",
        "run_recurrent_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    cli.main()
    assert json.loads(capsys.readouterr().out)["selected"] == "reward-only"
