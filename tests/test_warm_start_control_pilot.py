"""Matched warm checkpoints and deferred pair evaluation using owned fake models."""

from copy import deepcopy
from dataclasses import dataclass, replace
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


@dataclass
class State:
    policy: Any
    parameters: dict[str, Any]
    optimizer_state: dict[str, Any]
    gradients: dict[str, Any]
    completed_updates: int
    training_examples: int


class Learner(RewardLearner):
    packets: list[tuple[int, Any]] = []
    commits = 0
    fail_phase: str | None = None
    corrupt_warm: str | None = None
    control_states: list[State] = []
    session_commits: list[int] = []

    def __init__(self, policy):
        super().__init__(policy)
        self.arm_number = len(RewardLearner.sources)

    def snapshot_state(self):
        result = State(
            deepcopy(self.policy),
            deepcopy(self.state.parameters),
            deepcopy(self.state.optimizer_state),
            deepcopy(self.state.gradients),
            self.state.completed_updates,
            self.state.training_examples,
        )
        if self.arm_number == 1 and result.completed_updates == 8:
            Learner.control_states.append(deepcopy(result))
        if self.arm_number == 2 and result.completed_updates == 8:
            if Learner.corrupt_warm == "gradient":
                result.gradients["weight_input"][0, 0] += 0.1
            elif Learner.corrupt_warm == "momentum":
                result.optimizer_state["weight_input"][0, 0] += 0.1
            elif Learner.corrupt_warm == "policy":
                result.policy = replace(result.policy, seed=18)
        return result

    def restore_state(self, state):
        self.policy = deepcopy(state.policy)
        self.state = deepcopy(state)

    def train_batch(self, update, identifier):
        assert type(update) in (DemonstrationUpdate, RewardUpdate)
        assert update.episode.role == "train" and identifier == update.episode.episode_id
        phase = "demonstration" if type(update) is DemonstrationUpdate else "reward"
        if phase == "reward":
            assert self.arm_number == 2 and self.state.completed_updates >= 8
            assert pickle.dumps(update.behavior_state) == pickle.dumps(self.snapshot_state())
        Learner.packets.append((self.arm_number, deepcopy(update)))
        calls = sum(
            arm == self.arm_number and type(packet) is type(update)
            for arm, packet in Learner.packets
        )
        if (
            Learner.fail_phase == phase
            and calls == 3
            and (self.arm_number == 1 if phase == "demonstration" else self.arm_number == 2)
        ):
            raise RuntimeError(f"third paired {phase} update failed")
        for key, value in self.state.parameters.items():
            gradient = np.full_like(value, 0.25 if phase == "demonstration" else 0.0)
            momentum = self.state.optimizer_state.get(key, np.zeros_like(value)) * 0.9 + gradient
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
        assert Learner.commits == 24, "final scoring began before both arms trained"
        return super().predict(features)

    def start_episode(self):
        Learner.session_commits.append(Learner.commits)
        return super().start_episode()


def environment(monkeypatch):
    reward_environment(monkeypatch)
    monkeypatch.setattr(pilot, "TorchDemonstrationRewardLearner", Learner)
    for name in ("packets", "control_states", "session_commits"):
        monkeypatch.setattr(Learner, name, [])
    monkeypatch.setattr(Learner, "commits", 0)
    monkeypatch.setattr(Learner, "fail_phase", None)
    monkeypatch.setattr(Learner, "corrupt_warm", None)


def test_should_describe_matched_arms_without_native_data_or_resource_work(monkeypatch):
    for name in (
        "TorchDemonstrationRewardLearner",
        "ProcessRssSampler",
        "require_torch",
        "collect_key_door_reward_episode",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "KeyDoorNavigation", forbidden)
    protocol = pilot.describe_warm_start_comparison_protocol()
    assert protocol["planned_work"]["native_train_calls"] == 24
    assert protocol["planned_work"]["score_forwards"] == 64
    assert protocol["planned_work"]["live_episodes"] == 64
    assert protocol["seed"] == 17
    assert pilot.describe_protocol()["planned_work"]["native_train_calls"] == 8
    assert pilot.describe_warm_start_protocol()["planned_work"]["native_train_calls"] == 16


def test_should_match_complete_bc_state_before_reward_and_defer_all_pair_evaluation(monkeypatch):
    environment(monkeypatch)
    report = pilot.run_warm_start_comparison_pilot()
    assert report["status"] == "complete", report["runs"]
    assert report["warm_start_matched"] is True and report["global_errors"] == []
    control, reward = report["runs"]
    assert (control["arm"], control["method"]) == ("bc_only", "recurrent_bc")
    assert (reward["arm"], reward["method"]) == ("warm_start_rl", "recurrent_rl")
    assert control["wake_updates"] == 8 and reward["wake_updates"] == 16
    assert all(row["warm_start_matches_control"] is True for row in report["runs"])
    assert [point["native_updates"] for point in control["observations"]] == [0, 8]
    assert [point["native_updates"] for point in reward["observations"]] == [0, 8, 10, 12, 14, 16]
    assert len(Learner.packets) == Learner.commits == 24
    control_packets = [packet for arm, packet in Learner.packets if arm == 1]
    reward_packets = [packet for arm, packet in Learner.packets if arm == 2]
    assert len(control_packets) == 8 and all(
        type(packet) is DemonstrationUpdate for packet in control_packets
    )
    assert [packet.episode for packet in control_packets] == [
        packet.episode for packet in reward_packets[:8]
    ]
    assert all(type(packet) is RewardUpdate for packet in reward_packets[8:])
    first_reward = reward_packets[8].behavior_state
    assert first_reward.completed_updates == 8 and first_reward.training_examples == 64
    assert pickle.dumps(first_reward) == pickle.dumps(Learner.control_states[0])
    assert all(
        pickle.dumps(state) == pickle.dumps(first_reward) for state in Learner.control_states
    )
    assert first_reward.policy.seed == 17 and first_reward.policy.hidden_dim == 8
    assert all(np.any(value != 0) for value in first_reward.gradients.values())
    assert all(np.any(value != 0) for value in first_reward.optimizer_state.values())
    assert Learner.session_commits == list(range(16, 24)) + [24] * 64
    assert len(RewardLearner.classifications) == 64 and len(RewardLearner.sessions) == 72
    assert len(control["episodes"]) == 16 and len(reward["episodes"]) == 48
    assert control["phase_work"]["demonstration"]["training_rows"] == 64
    assert control["phase_work"]["reward"]["completed_updates"] == 0
    assert reward["phase_work"]["reward"]["training_rows"] == 20
    for row in report["runs"]:
        assert row["source_unchanged"] is row["actor_unchanged"] is True
        assert all(slot["charged_updates_after"] == slot["index"] + 1 for slot in row["training"])
        assert all(
            e["outcome"] == "illegal_action" and e["state_unchanged"] is True
            for e in row["episodes"]
        )
    assert all(slot["sampler_before"] == slot["sampler_after"] for slot in control["training"])
    comparison = report["comparison"]
    assert comparison["endpoint_native_updates"] == dict(bc_only=8, warm_start_rl=16)
    assert len(comparison["endpoint_rows"]) == 8
    for key in (
        "ce_reduction_bc_minus_rl",
        "accuracy_gain_rl_minus_bc",
        "success_gain_rl_minus_bc",
        "return_gain_rl_minus_bc",
    ):
        assert comparison[key] == 0
    for point in comparison["endpoint_rows"]:
        for metric in ("ce", "accuracy", "success", "return"):
            assert point[f"{metric}_bc"] == point[f"{metric}_rl"]
    assert comparison["success_rate"] == dict(bc_only=0.0, warm_start_rl=0.0)
    assert report["scientific_advantage_verified"] is report["full_R42_complete"] is False


@pytest.mark.parametrize("corruption", ["gradient", "momentum", "policy"])
def test_should_stop_mismatched_full_warm_state_before_any_reward_collection_or_final(
    monkeypatch, corruption
):
    environment(monkeypatch)
    monkeypatch.setattr(Learner, "corrupt_warm", corruption)
    report = pilot.run_warm_start_comparison_pilot()
    assert report["status"] == "incomplete" and report["warm_start_matched"] is False
    control, reward = report["runs"]
    assert control["wake_updates"] == reward["wake_updates"] == 8
    assert reward["warm_start_matches_control"] is False
    assert report["comparison"] is None
    assert RewardLearner.classifications == RewardLearner.sessions == []
    assert all(type(packet) is DemonstrationUpdate for _, packet in Learner.packets)
    assert all(slot["status"] == "not_run" for slot in reward["training"][8:])
    assert all(slot["status"] == "not_run" for row in report["runs"] for slot in row["episodes"])


@pytest.mark.parametrize("phase", ["demonstration", "reward"])
def test_should_preserve_original_per_arm_charges_and_skip_all_finals_after_pair_training_failure(
    monkeypatch, phase
):
    environment(monkeypatch)
    monkeypatch.setattr(Learner, "fail_phase", phase)
    report = pilot.run_warm_start_comparison_pilot()
    assert report["status"] == "incomplete" and report["comparison"] is None
    control, reward = report["runs"]
    assert RewardLearner.classifications == []
    assert all(slot["status"] == "not_run" for row in report["runs"] for slot in row["episodes"])
    if phase == "demonstration":
        assert control["wake_updates"] == 2
        assert control["training"][2]["status"] == "failed"
        assert control["phase_work"]["demonstration"]["training_rows"] == 16
        assert RewardLearner.sessions == []
    else:
        assert control["wake_updates"] == 8 and control["status"] == "not_evaluated"
        assert reward["wake_updates"] == 10
        assert reward["training"][10]["status"] == "failed"
        assert reward["phase_work"]["reward"]["training_rows"] == 8
        expected = np.random.Generator(np.random.PCG64(29))
        expected.random(9)
        assert reward["training"][10]["sampler_after"] == expected.bit_generator.state


@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_only_explicit_warm_control_cli(monkeypatch, capsys, describe):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-key-door-warm-start-control"]
        + (["--describe"] if describe else []),
    )
    selected = (
        "describe_warm_start_comparison_protocol" if describe else "run_warm_start_comparison_pilot"
    )
    for name in ("describe_warm_start_comparison_protocol", "run_warm_start_comparison_pilot"):
        monkeypatch.setattr(
            cli,
            name,
            (lambda: dict(status="complete", selected="matched-control"))
            if name == selected
            else forbidden,
        )
    for name in (
        "describe_reward_protocol",
        "run_reward_pilot",
        "describe_warm_start_protocol",
        "run_warm_start_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    cli.main()
    assert json.loads(capsys.readouterr().out)["selected"] == "matched-control"
