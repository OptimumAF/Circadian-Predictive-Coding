"""Explicit mean/weighted BC routes with owned fake learners and actual toy episodes."""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.key_door_pilot as shared
import src.adapters.key_door_reward_pilot as pilot
from src.adapters.torch_demonstration_reward_learner import (
    ClassWeightedDemonstrationUpdate,
    DemonstrationUpdate,
)
from src.adapters.torch_recurrent_learner import TORCH_RECURRENT_CE_ID
from src.adapters.torch_recurrent_weighted import TORCH_RECURRENT_WEIGHTED_CE_ID
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
    packets: list[Any] = []
    commits = 0
    fail_third = False
    wrong_diagnostic = False
    session_commits: list[int] = []

    def snapshot_state(self):
        return State(
            deepcopy(self.policy),
            deepcopy(self.state.parameters),
            deepcopy(self.state.optimizer_state),
            deepcopy(self.state.gradients),
            self.state.completed_updates,
            self.state.training_examples,
        )

    def restore_state(self, state):
        self.policy = deepcopy(state.policy)
        self.state = deepcopy(state)

    def train_batch(self, update, identifier):
        assert type(update) in (DemonstrationUpdate, ClassWeightedDemonstrationUpdate)
        assert update.episode.role == "train" and identifier == update.episode.episode_id
        Learner.packets.append(deepcopy(update))
        if Learner.fail_third and len(Learner.packets) == 3:
            raise RuntimeError("third BC update failed")
        for key, value in self.state.parameters.items():
            gradient = np.full_like(value, 0.25)
            momentum = self.state.optimizer_state.get(key, np.zeros_like(value)) * 0.9 + gradient
            self.state.gradients[key] = gradient
            self.state.optimizer_state[key] = momentum
            value -= 0.03 * momentum
        self.state.completed_updates += 1
        self.state.training_examples += len(update.episode.transitions)
        Learner.commits += 1
        weighted = type(update) is ClassWeightedDemonstrationUpdate
        diagnostic = TORCH_RECURRENT_WEIGHTED_CE_ID if weighted else TORCH_RECURRENT_CE_ID
        if Learner.wrong_diagnostic:
            diagnostic = TORCH_RECURRENT_CE_ID
        return TrainingDiagnostic(diagnostic, 0.5)

    def predict(self, features):
        assert Learner.commits == 8, "final scoring began before all eight train commits"
        return super().predict(features)

    def start_episode(self):
        assert Learner.commits == 8, "live evaluation began before all eight train commits"
        Learner.session_commits.append(Learner.commits)
        return super().start_episode()


def environment(monkeypatch):
    reward_environment(monkeypatch)
    monkeypatch.setattr(pilot, "TorchDemonstrationRewardLearner", Learner)
    monkeypatch.setattr(pilot, "collect_key_door_reward_episode", forbidden)
    monkeypatch.setattr(Learner, "packets", [])
    monkeypatch.setattr(Learner, "session_commits", [])
    monkeypatch.setattr(Learner, "commits", 0)
    monkeypatch.setattr(Learner, "fail_third", False)
    monkeypatch.setattr(Learner, "wrong_diagnostic", False)


@pytest.mark.parametrize("value", [0, None, "true"])
@pytest.mark.parametrize("entry", ["describe_demonstration_protocol", "run_demonstration_pilot"])
def test_should_reject_non_bool_objective_choice_before_resource_work(monkeypatch, entry, value):
    for name in ("TorchDemonstrationRewardLearner", "ProcessRssSampler", "require_torch"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    with pytest.raises(ValueError, match="class_weighted"):
        getattr(pilot, entry)(class_weighted=value)


@pytest.mark.parametrize("weighted", [False, True])
def test_should_describe_fixed_bc_only_work_without_native_or_data_calls(monkeypatch, weighted):
    for name in ("TorchDemonstrationRewardLearner", "ProcessRssSampler", "require_torch"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    protocol = pilot.describe_demonstration_protocol(class_weighted=weighted)
    assert protocol["methods"] == ["recurrent_bc"]
    assert protocol["checkpoint_updates"] == [0, 8]
    assert protocol["phase_schedule"] == ["demonstration"] * 8
    assert protocol["seed"] == 17 and protocol["replay_modes"] == [False]
    work = protocol["planned_work"]
    assert work["native_train_calls"] == 8 and work["maximum_training_rows"] == 64
    assert work["demonstration_training_rows"] == 64
    assert work["score_forwards"] == work["live_episodes"] == 16
    assert protocol["objective"] == (
        TORCH_RECURRENT_WEIGHTED_CE_ID if weighted else TORCH_RECURRENT_CE_ID
    )


@pytest.mark.parametrize("weighted", [False, True])
def test_should_release_only_train_packets_then_score_all_contexts_without_changing_actor(
    monkeypatch, weighted
):
    environment(monkeypatch)
    # Why: explicit False must preserve the default call shape and mean objective.
    report = (
        pilot.run_demonstration_pilot(class_weighted=True)
        if weighted
        else pilot.run_demonstration_pilot()
    )
    assert report["status"] == "complete", report["runs"]
    assert report["global_errors"] == [] and len(report["runs"]) == 1
    row = report["runs"][0]
    objective = TORCH_RECURRENT_WEIGHTED_CE_ID if weighted else TORCH_RECURRENT_CE_ID
    packet_type = ClassWeightedDemonstrationUpdate if weighted else DemonstrationUpdate
    assert row["demonstration_objective"] == objective
    assert row["method"] == "recurrent_bc" and row["replay_enabled"] is False
    assert row["wake_updates"] == Learner.commits == len(Learner.packets) == 8
    assert row["demonstration_collection_attempts"] == 8 and row["collection_attempts"] == 0
    assert [slot["context"] for slot in row["training"]] == list("AABBCCAA")
    assert [point["native_updates"] for point in row["observations"]] == [0, 8]
    assert [point["native_training_examples"] for point in row["observations"]] == [0, 64]
    assert all(episode["role"] == "final_test" for episode in report["demonstrations"])
    final_ids = {episode["episode_id"] for episode in report["demonstrations"]}
    for index, (packet, slot) in enumerate(zip(Learner.packets, row["training"], strict=True)):
        assert type(packet) is packet_type
        assert packet.episode.role == "train" and packet.episode.episode_id not in final_ids
        assert len(packet.episode.transitions) == slot["training_rows"] == 8
        assert slot["status"] == "complete" and slot["commit_validated"] is True
        assert slot["charged_updates_before"] == index
        assert slot["charged_updates_after"] == index + 1
        assert slot["receipt"]["diagnostic"]["definition"] == objective
        assert slot["sampler_before"] == slot["sampler_after"]
        if weighted:
            assert type(packet) is ClassWeightedDemonstrationUpdate
            counts = Counter(step.action for step in packet.episode.transitions)
            expected = tuple(8 / (len(counts) * counts[c]) if counts[c] else 1.0 for c in range(5))
            assert packet.class_weights == expected
            assert slot["class_weights"] == list(expected)
        else:
            assert "class_weights" not in slot
    assert row["phase_work"]["demonstration"]["training_rows"] == 64
    assert row["phase_work"]["reward"]["completed_updates"] == 0
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert len(RewardLearner.classifications) == len(RewardLearner.sessions) == 16
    assert Learner.session_commits == [8] * 16
    assert len(row["episodes"]) == 16
    assert {(e["context"], e["cue"]) for e in row["episodes"]} == {
        (context, cue) for context in "ABCD" for cue in (1, -1)
    }
    assert all(
        e["outcome"] == "illegal_action"
        and not e["success"]
        and e["return"] == 0
        and e["state_unchanged"] is True
        for e in row["episodes"]
    )
    assert report["scientific_advantage_verified"] is report["full_R42_complete"] is False


@pytest.mark.parametrize("weighted", [False, True])
def test_should_keep_two_permanent_charges_and_skip_finals_after_third_training_failure(
    monkeypatch, weighted
):
    environment(monkeypatch)
    monkeypatch.setattr(Learner, "fail_third", True)
    report = pilot.run_demonstration_pilot(class_weighted=weighted)
    assert report["status"] == "incomplete"
    row = report["runs"][0]
    assert row["wake_updates"] == Learner.commits == 2
    assert len(Learner.packets) == 3
    assert row["training"][2]["status"] == "failed"
    assert row["training"][2]["charged_updates_after"] == 2
    assert row["phase_work"]["demonstration"]["training_rows"] == 16
    assert RewardLearner.classifications == RewardLearner.sessions == []
    assert all(slot["status"] == "not_run" for slot in row["training"][3:])
    assert all(slot["status"] == "not_run" for slot in row["episodes"])


def test_should_refuse_mean_diagnostic_on_weighted_commit_and_preserve_its_charge(monkeypatch):
    environment(monkeypatch)
    monkeypatch.setattr(Learner, "wrong_diagnostic", True)
    report = pilot.run_demonstration_pilot(class_weighted=True)
    assert report["status"] == "incomplete"
    row = report["runs"][0]
    slot = row["training"][0]
    assert row["wake_updates"] == Learner.commits == len(Learner.packets) == 1
    assert slot["status"] == "failed" and slot["update_committed"] is True
    assert slot["commit_validated"] is False and slot["charged_updates_after"] == 1
    assert "progression differs" in slot["reason"]
    assert RewardLearner.classifications == RewardLearner.sessions == []


def test_should_refuse_non_train_demonstration_before_any_update_or_final(monkeypatch):
    environment(monkeypatch)
    collect = shared._collect

    def wrong_role(identity, layout, cue, role):
        batch = collect(identity, layout, cue, role)
        if role == "train":
            assert batch.episode is not None
            return replace(batch, episode=replace(batch.episode, role="final_test"))
        return batch

    monkeypatch.setattr(shared, "_collect", wrong_role)
    report = pilot.run_demonstration_pilot(class_weighted=True)
    assert report["status"] == "incomplete"
    row = report["runs"][0]
    assert row["wake_updates"] == 0 and Learner.packets == []
    assert row["training"][0]["status"] == "failed"
    assert "train-role" in row["training"][0]["reason"]
    assert RewardLearner.classifications == RewardLearner.sessions == []


@pytest.mark.parametrize("phase", ["collection", "update"])
def test_should_refuse_objective_switch_before_more_collection_or_final(monkeypatch, phase):
    environment(monkeypatch)
    captured: dict[str, Any] = {}
    new_row, collect, train = pilot._new_row, shared._collect, Learner.train_batch
    train_collections = []

    def capture_row(**kwargs):
        row = new_row(**kwargs)
        captured["row"] = row
        return row

    def change_during_collection(identity, layout, cue, role):
        batch = collect(identity, layout, cue, role)
        if role == "train":
            train_collections.append(identity)
            assert len(train_collections) == 1, "objective mutation admitted another collection"
            if phase == "collection":
                captured["row"]["demonstration_objective"] = TORCH_RECURRENT_CE_ID
        return batch

    def change_during_update(self, update, identifier):
        diagnostic = train(self, update, identifier)
        if phase == "update":
            captured["row"]["demonstration_objective"] = TORCH_RECURRENT_CE_ID
        return diagnostic

    monkeypatch.setattr(pilot, "_new_row", capture_row)
    monkeypatch.setattr(shared, "_collect", change_during_collection)
    monkeypatch.setattr(Learner, "train_batch", change_during_update)
    report = pilot.run_demonstration_pilot(class_weighted=True)
    assert report["status"] == "incomplete"
    row = report["runs"][0]
    expected_commits = int(phase == "update")
    assert row["wake_updates"] == Learner.commits == len(Learner.packets) == expected_commits
    assert len(train_collections) == 1 and "objective changed" in row["reason"]
    assert row["training"][0]["charged_updates_after"] == expected_commits
    assert RewardLearner.classifications == RewardLearner.sessions == []
    assert all(slot["status"] == "not_run" for slot in row["training"][1:])


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_only_explicit_bc_cli_with_weighted_option_only_when_requested(
    monkeypatch, capsys, weighted, describe
):
    scenario = "torch-key-door-weighted-bc" if weighted else "torch-key-door-bc"
    monkeypatch.setattr(
        sys, "argv", ["pilot", "--scenario", scenario] + (["--describe"] if describe else [])
    )
    seen = []

    def selected(**kwargs):
        seen.append(kwargs)
        return dict(status="complete", selected=scenario)

    monkeypatch.setattr(cli, "describe_demonstration_protocol", selected if describe else forbidden)
    monkeypatch.setattr(cli, "run_demonstration_pilot", forbidden if describe else selected)
    for name in (
        "describe_reward_protocol",
        "run_reward_pilot",
        "describe_warm_start_protocol",
        "run_warm_start_pilot",
        "describe_warm_start_comparison_protocol",
        "run_warm_start_comparison_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    cli.main()
    assert seen == ([dict(class_weighted=True)] if weighted else [{}])
    assert json.loads(capsys.readouterr().out)["selected"] == scenario
