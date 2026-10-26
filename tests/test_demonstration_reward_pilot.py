"""Explicit BC then reward phases preserve one owned runtime and native state."""

from copy import deepcopy
from dataclasses import replace
import json
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.key_door_pilot as shared
import src.adapters.key_door_reward_pilot as pilot
from src.adapters.torch_demonstration_reward_learner import DemonstrationUpdate
from src.adapters.torch_reward_update_learner import RewardUpdate
from src.core.learner_ports import TrainingDiagnostic
from test_key_door_reward_pilot import Learner as RewardLearner, environment as reward_environment
from test_torch_feature_rehearsal_pilot import forbidden


class Learner(RewardLearner):
    packets: list[Any] = []
    commits = 0
    fail_phase: str | None = None

    def train_batch(self, update, identifier):
        assert type(update) in (DemonstrationUpdate, RewardUpdate)
        assert update.episode.role == "train" and identifier == update.episode.episode_id
        phase = "demonstration" if type(update) is DemonstrationUpdate else "reward"
        if phase == "reward":
            assert pickle.dumps(update.behavior_state) == pickle.dumps(self.state)
            assert self.state.completed_updates >= 8
            assert self.state.optimizer_state
        Learner.packets.append(deepcopy(update))
        phase_calls = sum(type(item) is type(update) for item in Learner.packets)
        if Learner.fail_phase == phase and phase_calls == 3:
            raise RuntimeError(f"third {phase} update failed atomically")
        for key, value in self.state.parameters.items():
            gradient = np.full_like(value, 0.25 if phase == "demonstration" else 0.0)
            previous = self.state.optimizer_state.get(key, np.zeros_like(value))
            momentum = previous * 0.9 + gradient
            self.state.gradients[key] = gradient
            self.state.optimizer_state[key] = momentum
            value -= 0.03 * momentum
        self.state.completed_updates += 1
        self.state.training_examples += len(update.episode.transitions)
        Learner.commits += 1
        return TrainingDiagnostic(
            "torch_recurrent_episode_cross_entropy_mean_v1"
            if phase == "demonstration"
            else "torch_recurrent_reward_logprob_sum_v1",
            0.5 if phase == "demonstration" else 0.0,
        )

    def predict(self, features):
        assert Learner.commits == 16, "final scoring happened before every original commit"
        return super().predict(features)


def environment(monkeypatch):
    reward_environment(monkeypatch)
    monkeypatch.setattr(pilot, "TorchDemonstrationRewardLearner", Learner)
    monkeypatch.setattr(Learner, "packets", [])
    monkeypatch.setattr(Learner, "commits", 0)
    monkeypatch.setattr(Learner, "fail_phase", None)


def test_should_describe_explicit_phases_without_collecting_or_native_work(monkeypatch):
    for name in (
        "TorchRewardUpdateLearner",
        "TorchDemonstrationRewardLearner",
        "ProcessRssSampler",
        "require_torch",
        "collect_key_door_reward_episode",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "KeyDoorNavigation", forbidden)
    protocol = pilot.describe_warm_start_protocol()
    assert protocol["methods"] == ["recurrent_rl"] and protocol["replay_modes"] == [False]
    assert protocol["planned_work"]["native_train_calls"] == 16
    assert protocol["planned_work"]["score_forwards"] == 48
    assert protocol["planned_work"]["live_episodes"] == 48
    assert pilot.describe_protocol()["planned_work"]["native_train_calls"] == 8


def test_should_keep_continuous_full_native_state_from_eight_demo_to_eight_reward_commits(
    monkeypatch,
):
    environment(monkeypatch)
    report = pilot.run_warm_start_pilot()
    assert report["status"] == "complete", report["runs"]
    assert report["global_errors"] == [] and len(report["runs"]) == 1
    row = report["runs"][0]
    assert row["phase_schedule"] == ["demonstration"] * 8 + ["reward"] * 8
    assert row["wake_updates"] == Learner.commits == 16
    assert len(row["training"]) == len(Learner.packets) == 16
    assert [point["wake_updates"] for point in row["observations"]] == [0, 8, 10, 12, 14, 16]
    assert [point["native_updates"] for point in row["observations"]] == [0, 8, 10, 12, 14, 16]
    assert [point["native_training_examples"] for point in row["observations"]] == [
        0,
        64,
        72,
        74,
        76,
        84,
    ]
    rng = np.random.Generator(np.random.PCG64(29))
    for index, slot in enumerate(row["training"]):
        assert slot["index"] == index and slot["status"] == "complete"
        assert (
            slot["charged_updates_before"] == index and slot["charged_updates_after"] == index + 1
        )
        assert slot["update_committed"] is slot["commit_validated"] is True
        assert slot["sampler_before"] == rng.bit_generator.state
        if index < 8:
            assert type(Learner.packets[index]) is DemonstrationUpdate
            assert slot["phase"] == "demonstration" and slot["collection"] is None
            assert slot["demonstration"]["episode_id"] == f"warmup-train-{index}"
            assert slot["demonstration"]["role"] == "train" and slot["training_rows"] == 8
            assert (
                slot["receipt"]["diagnostic"]["definition"]
                == "torch_recurrent_episode_cross_entropy_mean_v1"
            )
        else:
            packet = Learner.packets[index]
            assert type(packet) is RewardUpdate
            assert slot["phase"] == "reward" and slot["demonstration"] is None
            assert slot["collection"]["episode"]["episode_id"] == f"reward-train-{index - 8}"
            assert packet.behavior_state.completed_updates == index
            assert packet.behavior_state.training_examples == 64 + sum(
                len(item.episode.transitions) for item in Learner.packets[8:index]
            )
            assert (
                slot["receipt"]["diagnostic"]["definition"]
                == "torch_recurrent_reward_logprob_sum_v1"
            )
            rng.random(len(packet.episode.transitions))
        assert slot["sampler_after"] == rng.bit_generator.state
    first_reward = Learner.packets[8]
    assert first_reward.behavior_state.completed_updates == 8
    assert first_reward.behavior_state.training_examples == 64
    assert all(np.any(value != 0) for value in first_reward.behavior_state.parameters.values())
    assert all(np.any(value != 0) for value in first_reward.behavior_state.optimizer_state.values())
    assert row["phase_work"]["demonstration"] == dict(
        completed_updates=8, training_rows=64, validated_commits=8
    )
    assert row["phase_work"]["reward"] == dict(
        completed_updates=8, training_rows=20, validated_commits=8
    )
    assert row["stream_result"]["completed_updates"] == 16
    assert row["stream_result"]["training_examples"] == 84
    assert row["stream_result"]["score_forward_calls"] == len(RewardLearner.classifications) == 48
    assert len(row["episodes"]) == row["live_sessions_started"] == 48
    assert len(RewardLearner.sessions) == 56
    assert all(
        episode["outcome"] == "illegal_action" and episode["state_unchanged"] is True
        for episode in row["episodes"]
    )
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert report["scientific_advantage_verified"] is report["full_R42_complete"] is False


@pytest.mark.parametrize("phase", ["demonstration", "reward"])
def test_should_preserve_committed_phase_charges_and_unreached_work_on_atomic_failure(
    monkeypatch, phase
):
    environment(monkeypatch)
    monkeypatch.setattr(Learner, "fail_phase", phase)
    report = pilot.run_warm_start_pilot()
    assert report["status"] == "incomplete"
    row = report["runs"][0]
    failed_index, charges = (2, 2) if phase == "demonstration" else (10, 10)
    assert row["wake_updates"] == Learner.commits == charges
    assert row["training"][failed_index]["status"] == "failed"
    assert row["training"][failed_index]["update_committed"] is False
    assert row["training"][failed_index]["receipt"] is None
    assert row["training"][failed_index]["charged_updates_after"] == charges
    assert all(slot["status"] == "not_run" for slot in row["training"][failed_index + 1 :])
    assert all(slot["status"] == "not_run" for slot in row["episodes"])
    assert RewardLearner.classifications == []
    expected = np.random.Generator(np.random.PCG64(29))
    if phase == "demonstration":
        assert RewardLearner.sessions == []
        assert row["phase_work"]["demonstration"]["completed_updates"] == 2
        assert row["phase_work"]["demonstration"]["training_rows"] == 16
        assert row["phase_work"]["reward"]["completed_updates"] == 0
    else:
        expected.random(9)
        assert row["phase_work"]["demonstration"]["completed_updates"] == 8
        assert row["phase_work"]["demonstration"]["training_rows"] == 64
        assert row["phase_work"]["reward"]["completed_updates"] == 2
        assert row["phase_work"]["reward"]["training_rows"] == 8
    assert row["training"][failed_index]["sampler_after"] == expected.bit_generator.state
    assert row["source_unchanged"] is row["actor_unchanged"] is True


def test_should_refuse_nontrain_demonstration_before_any_sampled_reward_or_native_commit(
    monkeypatch,
):
    environment(monkeypatch)
    original = shared._collect

    def wrong_role(identity, layout, cue, role):
        result = original(identity, layout, cue, role)
        if role == "train":
            assert result.episode is not None
            return replace(
                result, role="final_test", episode=replace(result.episode, role="final_test")
            )
        return result

    monkeypatch.setattr(shared, "_collect", wrong_role)
    report = pilot.run_warm_start_pilot()
    row = report["runs"][0]
    assert report["status"] == "incomplete" and row["status"] == "failed"
    assert row["wake_updates"] == Learner.commits == 0 and Learner.packets == []
    assert RewardLearner.sessions == RewardLearner.classifications == []
    assert row["training"][0]["demonstration"]["role"] == "final_test"
    assert row["training"][0]["charged_updates_after"] == 0
    assert row["training"][0]["update_attempted"] is False
    assert all(slot["status"] == "not_run" for slot in row["training"][1:])
    assert all(slot["status"] == "not_run" for slot in row["episodes"])
    assert row["training"][0]["sampler_before"] == row["training"][0]["sampler_after"]


def test_should_keep_all_sixteen_training_and_forty_eight_live_slots_on_resource_failure(
    monkeypatch,
):
    def fail():
        raise RuntimeError("warm-start resource setup")

    monkeypatch.setattr(pilot, "ProcessRssSampler", fail)
    report = pilot.run_warm_start_pilot()
    row = report["runs"][0]
    assert report["status"] == "incomplete" and report["global_errors"]
    assert row["status"] == "not_run" and row["wake_updates"] == 0
    assert len(row["training"]) == 16 and len(row["episodes"]) == 48
    assert all(slot["status"] == "not_run" for slot in (*row["training"], *row["episodes"]))


def test_should_preserve_primary_refusal_when_corrupted_phase_cost_cannot_be_summarized(
    monkeypatch,
):
    environment(monkeypatch)
    original_run, original_observation = pilot._run_arm, shared._recurrent_observation
    captured: dict[str, Any] = {}

    def observe(updates, state):
        if updates == 8:
            captured["row"]["training"][0]["training_rows"] = None
        return original_observation(updates, state)

    def run(row, *args, **kwargs):
        captured["row"] = row
        return original_run(row, *args, **kwargs)

    monkeypatch.setattr(shared, "_recurrent_observation", observe)
    monkeypatch.setattr(pilot, "_run_arm", run)
    report = pilot.run_warm_start_pilot()
    row = report["runs"][0]
    assert report["status"] == "incomplete" and row["status"] == "failed"
    assert "preceding RL evidence" in row["reason"]
    assert "corrupted committed evidence" in row["phase_work_error"]
    assert row["phase_work"] is None
    assert row["wake_updates"] == Learner.commits == 8
    assert row["collection_attempts"] == 0
    assert all(s["status"] == "not_run" for s in row["training"][8:])
    assert RewardLearner.classifications == []


@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_only_explicit_warm_start_cli(monkeypatch, capsys, describe):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-key-door-warm-start-rl"]
        + (["--describe"] if describe else []),
    )
    selected = "describe_warm_start_protocol" if describe else "run_warm_start_pilot"
    for name in ("describe_warm_start_protocol", "run_warm_start_pilot"):
        monkeypatch.setattr(
            cli,
            name,
            (lambda: dict(status="complete", selected="warm-start"))
            if name == selected
            else forbidden,
        )
    for name in (
        "describe_reward_protocol",
        "run_reward_pilot",
        "describe_recurrent_protocol",
        "run_recurrent_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    cli.main()
    assert json.loads(capsys.readouterr().out)["selected"] == "warm-start"
