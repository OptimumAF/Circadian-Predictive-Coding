"""Opt-in legal reward routing through the actual actor runtime and local game."""

from copy import deepcopy
from dataclasses import replace
import json
import pickle
import sys

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.key_door_pilot as shared
import src.adapters.key_door_reward_pilot as pilot
from src.adapters.torch_reward_update_learner import LegalRewardUpdate
from src.core.learner_ports import TrainingDiagnostic
from src.core.legal_policy_gradient import LegalPolicyGradientEpisode
from test_key_door_reward_pilot import (
    Learner as OriginalLearner,
    environment as original_environment,
)
from test_torch_feature_rehearsal_pilot import forbidden


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
            raise RuntimeError("second legal prediction failed")
        self.completed_steps += 1
        return np.array([Learner.raw], dtype=np.float64)


class Learner:
    sessions: list[Session] = []
    updates: list[LegalRewardUpdate] = []
    sources: list["Learner"] = []
    classifications: list[np.ndarray] = []
    identifiers: list[str] = []
    committed = 0
    fail_collection = False
    fail_update = False
    raw = (0.2,) * 5
    snapshot_state = OriginalLearner.snapshot_state
    restore_state = OriginalLearner.restore_state
    fork = OriginalLearner.fork

    @property
    def completed_updates(self):
        return self.state.completed_updates

    @property
    def training_examples(self):
        return self.state.training_examples

    def __init__(self, policy):
        self.policy = policy
        self.state = OriginalLearner(policy).state
        Learner.sources.append(self)

    def train_batch(self, update, identifier):
        assert type(update) is LegalRewardUpdate
        assert type(update.episode) is LegalPolicyGradientEpisode
        packet = update.episode.episode
        assert packet.role == "train" and packet.episode_id == identifier
        assert pickle.dumps(update.behavior_state) == pickle.dumps(self.state)
        Learner.updates.append(deepcopy(update))
        Learner.identifiers.append(identifier)
        if Learner.fail_update and len(Learner.updates) == 3:
            raise RuntimeError("third legal atomic update failed")
        self.state.completed_updates += 1
        self.state.training_examples += len(packet.transitions)
        for name, value in self.state.parameters.items():
            self.state.gradients[name] = np.zeros_like(value)
        self.state.optimizer_state = {
            name: value.copy() for name, value in self.state.gradients.items()
        }
        Learner.committed += 1
        return TrainingDiagnostic("torch_recurrent_legal_reward_logprob_sum_v1", 0.0)

    def predict(self, features):
        assert Learner.committed == 8, "final scoring became available before training finished"
        assert features.shape == (8, 8)
        Learner.classifications.append(features.copy())
        return np.tile(np.array([Learner.raw]), (len(features), 1))

    def start_episode(self):
        result = Session(self)
        Learner.sessions.append(result)
        return result


def environment(monkeypatch):
    original_environment(monkeypatch)
    monkeypatch.setattr(pilot, "TorchRewardUpdateLearner", Learner)
    for name in ("sessions", "updates", "identifiers", "classifications", "sources"):
        monkeypatch.setattr(Learner, name, [])
    monkeypatch.setattr(Learner, "fail_collection", False)
    monkeypatch.setattr(Learner, "fail_update", False)
    monkeypatch.setattr(Learner, "committed", 0)
    monkeypatch.setattr(Learner, "raw", (0.2,) * 5)
    collected, budgets = [], []
    original_collect, original_session = shared._collect, shared._session

    def collect(*args, **kwargs):
        assert Learner.committed == 8, "final teacher data became available before all commits"
        assert args[3] == "final_test"
        collected.append((args, kwargs))
        return original_collect(*args, **kwargs)

    def session(*args, **kwargs):
        result = original_session(*args, **kwargs)
        budgets.append(result)
        return result

    monkeypatch.setattr(shared, "_collect", collect)
    monkeypatch.setattr(shared, "_session", session)
    return collected, budgets


def test_should_complete_eight_legal_updates_with_original_costs_and_keep_negative_curves(
    monkeypatch,
):
    final_collections, budgets = environment(monkeypatch)
    global_rng = pickle.dumps(np.random.get_state())
    report = pilot.run_pilot(condition_on_legal_actions=True)
    assert report["status"] == "complete", report
    assert report["global_errors"] == [] and len(report["runs"]) == 1
    row = report["runs"][0]
    assert row["action_conditioning"] == "legal_actions_v1"
    assert row["method"] == "recurrent_rl" and row["replay_enabled"] is False
    assert row["wake_updates"] == Learner.committed == len(Learner.updates) == 8
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert len(budgets) == 1 and budgets[0].budget.max_training_updates == 8
    assert budgets[0].updates_completed == 8 and budgets[0].replay_examples_completed == 0
    assert len(final_collections) == len(report["demonstrations"]) == 8
    assert all(packet["role"] == "final_test" for packet in report["demonstrations"])
    assert [slot["context"] for slot in row["training"]] == list("AABBCCAA")
    assert [slot["cue"] for slot in row["training"]] == [1, -1] * 4
    assert len(set(Learner.identifiers)) == 8
    expected_rng, examples = np.random.Generator(np.random.PCG64(29)), 0
    for index, (slot, update) in enumerate(zip(row["training"], Learner.updates, strict=True)):
        packet = update.episode.episode
        count = len(packet.transitions)
        assert slot["status"] == "complete" and slot["commit_validated"] is True
        assert slot["update_attempted"] is slot["update_committed"] is True
        assert slot["charged_updates_before"] == slot["native_updates_before"] == index
        assert slot["charged_updates_after"] == index + 1
        assert slot["native_training_examples_before"] == examples
        assert slot["sampler_before"] == expected_rng.bit_generator.state
        expected_rng.random(count)
        assert slot["sampler_after"] == expected_rng.bit_generator.state
        collection = slot["collection"]
        assert collection["status"] == "complete" and collection["termination"] != "illegal_action"
        assert collection["action_conditioning"] == "legal_actions_v1"
        assert collection["episode"]["episode"]["role"] == "train"
        assert collection["episode"]["episode"]["episode_id"] == Learner.identifiers[index]
        assert "behavior_state" not in collection
        assert collection["behavior_native_updates"] == index
        assert collection["behavior_training_examples"] == examples
        assert collection["sampling_draws"] == collection["environment_steps"] == count
        assert slot["receipt"]["diagnostic"]["definition"] == (
            "torch_recurrent_legal_reward_logprob_sum_v1"
        )
        for step, transition in enumerate(collection["transitions"]):
            legal = update.episode.legal_actions[step]
            assert transition["legal_actions"] == legal
            assert transition["action"] in legal and transition["applied"] is True
            assert transition["raw_probabilities"] == update.episode.raw_probabilities[step]
            expected = tuple(1 / len(legal) if action in legal else 0.0 for action in range(5))
            np.testing.assert_allclose(transition["probabilities"], expected, rtol=0, atol=1e-15)
            assert packet.behavior_probabilities[step] == transition["probabilities"]
        examples += count
    assert row["sampling_state"] == expected_rng.bit_generator.state
    assert row["stream_result"]["completed_updates"] == 8
    assert row["stream_result"]["training_examples"] == examples
    assert len(row["stream_result"]["applied_updates"]) == 8
    assert row["stream_result"]["score_forward_calls"] == len(Learner.classifications) == 40
    assert [point["wake_updates"] for point in row["observations"]] == [0, 2, 4, 6, 8]
    assert len(row["episodes"]) == row["live_sessions_started"] == 40
    assert len(Learner.sessions) == 48
    for slot in row["episodes"]:
        assert slot["status"] == "complete" and slot["state_unchanged"] is True
        assert slot["action_conditioning"] == "legal_actions_v1"
        assert slot["outcome"] != "illegal_action" and slot["success"] is False
        assert slot["return"] == 0
        for transition in slot["transitions"]:
            assert transition["action"] in transition["legal_actions"]
            assert transition["action"] == int(np.argmax(transition["probabilities"]))
            assert transition["raw_probabilities"] == Learner.raw
    assert all(value == 0 for value in row["outcomes"].values())
    assert report["scientific_advantage_verified"] is report["full_R42_complete"] is False
    assert pickle.dumps(np.random.get_state()) == global_rng


@pytest.mark.parametrize("phase", ["collection", "update", "zero_mass"])
def test_should_keep_failed_legal_work_rng_and_original_charges_without_final_data(
    monkeypatch, phase
):
    final_collections, budgets = environment(monkeypatch)
    if phase == "zero_mass":
        monkeypatch.setattr(Learner, "raw", (0.0, 0.0, 0.0, 0.0, 1.0))
    else:
        monkeypatch.setattr(
            Learner, "fail_collection" if phase == "collection" else "fail_update", True
        )
    report = pilot.run_pilot(condition_on_legal_actions=True)
    row = report["runs"][0]
    assert report["status"] == "incomplete" and row["status"] == "failed"
    completed = 2 if phase == "update" else 0
    assert row["wake_updates"] == budgets[0].updates_completed == Learner.committed == completed
    failed = row["training"][completed]
    assert failed["status"] == "failed" and failed["receipt"] is None
    assert failed["update_committed"] is failed["commit_validated"] is False
    assert failed["charged_updates_after"] == completed
    assert all(slot["status"] == "not_run" for slot in row["training"][completed + 1 :])
    assert all(slot["status"] == "not_run" for slot in row["episodes"])
    assert final_collections == report["demonstrations"] == Learner.classifications == []
    expected_rng = np.random.Generator(np.random.PCG64(29))
    for slot in row["training"][: completed + 1]:
        expected_rng.random(slot["collection"]["sampling_draws"])
    assert failed["sampler_after"] == row["sampling_state"] == expected_rng.bit_generator.state
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    if phase == "zero_mass":
        collection = failed["collection"]
        assert collection["prediction_calls_returned"] == 1
        assert collection["sampling_draws"] == collection["environment_steps_attempted"] == 0
        assert collection["transitions"][0]["raw_probabilities"] == Learner.raw
    elif phase == "collection":
        assert failed["collection"]["prediction_calls_attempted"] == 2
        assert failed["collection"]["sampling_draws"] == 1 and failed["update_attempted"] is False
    else:
        assert failed["update_attempted"] is True
        assert "third legal atomic update failed" in failed["reason"]


@pytest.mark.parametrize("when", ["collection", "update"])
def test_should_refuse_changed_conditioning_without_admitting_more_training(monkeypatch, when):
    final_collections, budgets = environment(monkeypatch)
    rows = []
    original_row = pilot._new_row

    def make_row(*args, **kwargs):
        result = original_row(*args, **kwargs)
        rows.append(result)
        return result

    monkeypatch.setattr(pilot, "_new_row", make_row)
    owner = pilot if when == "collection" else Learner
    name = "collect_key_door_reward_episode" if when == "collection" else "train_batch"
    original = getattr(owner, name)

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        rows[0]["action_conditioning"] = None
        return result

    monkeypatch.setattr(owner, name, mutate)
    report = pilot.run_pilot(condition_on_legal_actions=True)
    row = report["runs"][0]
    assert report["status"] == "incomplete" and row["status"] == "failed"
    expected = int(when == "update")
    assert row["wake_updates"] == budgets[0].updates_completed == Learner.committed == expected
    assert row["collection_attempts"] == 1
    assert len(Learner.updates) == expected
    assert final_collections == report["demonstrations"] == []
    assert all(slot["status"] == "not_run" for slot in row["training"][1:])
    first = row["training"][0]
    assert first["charged_updates_after"] == expected
    assert first["update_committed"] is (when == "update")
    assert first["commit_validated"] is False
    if when == "update":
        assert first["receipt"] is not None
        assert first["receipt"]["update_number"] == 1
    else:
        assert first["receipt"] is None


@pytest.mark.parametrize("fault", ["role", "reported_conditioning"])
def test_should_refuse_replaced_role_or_conditioning_in_collected_packet(monkeypatch, fault):
    final_collections, budgets = environment(monkeypatch)
    original = pilot.collect_key_door_reward_episode

    def forged(*args, **kwargs):
        result = original(*args, **kwargs)
        if fault == "role":
            wrapper = result["episode"]
            result["episode"] = replace(
                wrapper, episode=replace(wrapper.episode, role="final_test")
            )
        else:
            result["action_conditioning"] = None
        return result

    monkeypatch.setattr(pilot, "collect_key_door_reward_episode", forged)
    report = pilot.run_pilot(condition_on_legal_actions=True)
    assert report["status"] == "incomplete"
    assert report["runs"][0]["wake_updates"] == budgets[0].updates_completed == 0
    assert Learner.updates == [] and final_collections == report["demonstrations"] == []


@pytest.mark.parametrize(
    "phase,target",
    [("final", "conditioning"), ("score", "source"), ("score", "actor"), ("live", "conditioning")],
)
def test_should_stop_evaluation_at_first_changed_boundary_and_preserve_all_training_charges(
    monkeypatch, phase, target
):
    final_collections, budgets = environment(monkeypatch)
    rows, runtimes = [], []
    original_row, original_runtime = pilot._new_row, pilot.ActorShadowRuntime

    def make_row(*args, **kwargs):
        result = original_row(*args, **kwargs)
        rows.append(result)
        return result

    def runtime(*args, **kwargs):
        result = original_runtime(*args, **kwargs)
        runtimes.append(result)
        return result

    monkeypatch.setattr(pilot, "_new_row", make_row)
    monkeypatch.setattr(pilot, "ActorShadowRuntime", runtime)

    def corrupt():
        if target == "conditioning":
            rows[0]["action_conditioning"] = None
        else:
            owner = Learner.sources[0] if target == "source" else runtimes[0].actor._learner
            assert isinstance(owner, Learner)
            owner.state.parameters["weight_input"][0, 0] += 1

    if phase == "final":
        original = shared._collect

        def collect(*args, **kwargs):
            result = original(*args, **kwargs)
            corrupt()
            return result

        monkeypatch.setattr(shared, "_collect", collect)
    else:
        owner = Learner if phase == "score" else Session
        original_predict = owner.predict

        def predict(*args, **kwargs):
            result = original_predict(*args, **kwargs)
            if Learner.committed == 8:
                corrupt()
            return result

        monkeypatch.setattr(owner, "predict", predict)
    report = pilot.run_pilot(condition_on_legal_actions=True)
    row = report["runs"][0]
    assert report["status"] == "incomplete" and row["status"] == "failed"
    assert row["wake_updates"] == budgets[0].updates_completed == Learner.committed == 8
    assert len(Learner.updates) == 8
    assert all(slot["status"] == "complete" for slot in row["training"])
    assert len(final_collections) == (1 if phase == "final" else 8)
    assert len(Learner.classifications) == {"final": 0, "score": 1, "live": 40}[phase]
    assert row["live_sessions_started"] == int(phase == "live")
    assert all(slot["status"] == "not_run" for slot in row["episodes"][int(phase == "live") :])
    assert row["source_unchanged"] is (target != "source")
    assert row["actor_unchanged"] is (target != "actor")


def test_should_describe_separate_legal_information_without_native_resources(monkeypatch):
    for name in (
        "TorchRewardUpdateLearner",
        "ProcessRssSampler",
        "require_torch",
        "collect_key_door_reward_episode",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    monkeypatch.setattr(shared, "KeyDoorNavigation", forbidden)
    old = pilot.describe_protocol()
    assert old == pilot.describe_protocol(condition_on_legal_actions=False)
    assert "action_conditioning" not in old
    legal = pilot.describe_protocol(condition_on_legal_actions=True)
    assert legal["protocol_id"] != old["protocol_id"]
    assert legal["action_conditioning"] == "legal_actions_v1"
    assert legal["planned_work"] == old["planned_work"]
    assert legal["methods"] == ["recurrent_rl"] and legal["sampling_seed"] == 29


def test_should_keep_original_public_default_route_explicitly_unconditioned(monkeypatch):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return dict(status="default-sentinel")

    monkeypatch.setattr(pilot, "_run_pilot", run)
    implicit = pilot.run_pilot()
    explicit = pilot.run_pilot(condition_on_legal_actions=False)
    assert implicit == explicit == dict(status="default-sentinel")
    assert calls == [dict(warm_start=False, condition_on_legal_actions=False)] * 2


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", np.bool_(True)])
@pytest.mark.parametrize("function", ["describe_protocol", "run_pilot"])
def test_should_refuse_non_native_bool_before_resource_work(monkeypatch, invalid, function):
    for name in (
        "TorchRewardUpdateLearner",
        "ProcessRssSampler",
        "require_torch",
        "collect_key_door_reward_episode",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    with pytest.raises(ValueError):
        getattr(pilot, function)(condition_on_legal_actions=invalid)


@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_explicit_legal_cli_with_conditioning_enabled(
    monkeypatch, capsys, describe
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-key-door-legal-rl"] + (["--describe"] if describe else []),
    )
    selected = "describe_legal_reward_protocol" if describe else "run_legal_reward_pilot"
    calls = []

    def chosen(**kwargs):
        calls.append(kwargs)
        return dict(status="complete", selected="legal-reward")

    for name in ("describe_legal_reward_protocol", "run_legal_reward_pilot"):
        monkeypatch.setattr(cli, name, chosen if name == selected else forbidden)
    for name in (
        "describe_reward_protocol",
        "run_reward_pilot",
        "describe_warm_start_protocol",
        "run_warm_start_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    cli.main()
    assert calls == [{"condition_on_legal_actions": True}]
    assert json.loads(capsys.readouterr().out)["selected"] == "legal-reward"
