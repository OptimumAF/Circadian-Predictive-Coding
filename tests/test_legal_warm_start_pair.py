"""Legal live comparison with actual runtime and independently owned fake SGD state."""

from copy import deepcopy
import json
import pickle
import sys

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.key_door_pilot as shared
import src.adapters.key_door_reward_pilot as pilot
from src.adapters.torch_demonstration_reward_learner import DemonstrationUpdate
from src.adapters.torch_reward_update_learner import LegalRewardUpdate
from src.core.learner_ports import TrainingDiagnostic
from test_key_door_reward_pilot import Learner as RewardLearner, Session as RewardSession
from test_torch_feature_rehearsal_pilot import forbidden
from test_warm_start_control_pilot import Learner as WarmLearner, environment as warm_environment


class Session(RewardSession):
    def __init__(self, source):
        self.state = source.snapshot_state()
        self.completed_steps = 0
        self.inputs: list[np.ndarray] = []

    def snapshot_state(self):
        return deepcopy(self.state)

    def predict(self, features):
        assert features.shape == (1, 8) and features.dtype == np.float64
        self.inputs.append(features.copy())
        self.completed_steps += 1
        return np.array([[0.2] * 5])


class Learner(WarmLearner):
    def train_batch(self, update, identifier):
        assert type(update) in (DemonstrationUpdate, LegalRewardUpdate)
        reward = type(update) is LegalRewardUpdate
        packet = update.episode.episode if reward else update.episode
        assert packet.role == "train" and identifier == packet.episode_id
        phase = "reward" if reward else "demonstration"
        if reward:
            assert self.arm_number == 2 and self.state.completed_updates >= 8
            assert pickle.dumps(update.behavior_state) == pickle.dumps(self.snapshot_state())
        WarmLearner.packets.append((self.arm_number, deepcopy(update)))
        calls = sum(
            arm == self.arm_number and type(record) is type(update)
            for arm, record in WarmLearner.packets
        )
        if (
            WarmLearner.fail_phase == phase
            and calls == 3
            and self.arm_number == (2 if reward else 1)
        ):
            raise RuntimeError(f"third legal paired {phase} update failed")
        for name, value in self.state.parameters.items():
            gradient = np.full_like(value, 0.0 if reward else 0.25)
            momentum = self.state.optimizer_state.get(name, np.zeros_like(value)) * 0.9 + gradient
            self.state.gradients[name] = gradient
            self.state.optimizer_state[name] = momentum
            value -= 0.03 * momentum
        self.state.completed_updates += 1
        self.state.training_examples += len(packet.transitions)
        WarmLearner.commits += 1
        return TrainingDiagnostic(
            "torch_recurrent_legal_reward_logprob_sum_v1"
            if reward
            else "torch_recurrent_episode_cross_entropy_mean_v1",
            0.0 if reward else 0.5,
        )

    def predict(self, features):
        assert WarmLearner.commits == 24, "final scoring started before both arms finished"
        assert features.shape == (8, 8)
        RewardLearner.classifications.append(features.copy())
        return np.full((8, 5), 0.2)

    def start_episode(self):
        WarmLearner.session_commits.append(WarmLearner.commits)
        result = Session(self)
        RewardLearner.sessions.append(result)
        return result


def environment(monkeypatch):
    warm_environment(monkeypatch)
    monkeypatch.setattr(pilot, "TorchDemonstrationRewardLearner", Learner)
    collected, budgets, rows, runtimes = [], [], [], []
    original_collect, original_session = shared._collect, shared._session
    original_row, original_runtime = pilot._new_row, pilot.ActorShadowRuntime

    def collect(*args, **kwargs):
        role = args[3]
        assert role in ("train", "final_test")
        if role == "final_test":
            assert WarmLearner.commits == 24, "final teacher data created before all24commits"
        collected.append((role, args[0], WarmLearner.commits))
        return original_collect(*args, **kwargs)

    def session(*args, **kwargs):
        result = original_session(*args, **kwargs)
        budgets.append(result)
        return result

    def make_row(*args, **kwargs):
        result = original_row(*args, **kwargs)
        rows.append(result)
        return result

    def runtime(*args, **kwargs):
        result = original_runtime(*args, **kwargs)
        runtimes.append(result)
        return result

    monkeypatch.setattr(shared, "_collect", collect)
    monkeypatch.setattr(shared, "_session", session)
    monkeypatch.setattr(pilot, "_new_row", make_row)
    monkeypatch.setattr(pilot, "ActorShadowRuntime", runtime)
    return collected, budgets, rows, runtimes


def test_should_preserve_full_warm_match_original_budgets_and_one_shared_final_after24commits(
    monkeypatch,
):
    collected, budgets, _, _ = environment(monkeypatch)
    global_rng = pickle.dumps(np.random.get_state())
    report = pilot.run_warm_start_comparison_pilot(condition_on_legal_actions=True)
    assert report["status"] == "complete", report
    assert report["warm_start_matched"] is True and report["global_errors"] == []
    control, reward = report["runs"]
    assert [row["arm"] for row in report["runs"]] == ["bc_only", "warm_start_rl"]
    assert [row["wake_updates"] for row in report["runs"]] == [8, 16]
    assert [budget.budget.max_training_updates for budget in budgets] == [8, 16]
    assert [budget.updates_completed for budget in budgets] == [8, 16]
    assert all(budget.replay_examples_completed == 0 for budget in budgets)
    assert len(WarmLearner.packets) == WarmLearner.commits == 24
    assert [role for role, _, _ in collected] == ["train"] * 16 + ["final_test"] * 8
    assert len({identity for role, identity, _ in collected if role == "final_test"}) == 8
    assert all(commits == 24 for role, _, commits in collected if role == "final_test")
    assert len(report["demonstrations"]) == 8
    assert all(packet["role"] == "final_test" for packet in report["demonstrations"])
    control_packets = [packet for arm, packet in WarmLearner.packets if arm == 1]
    reward_packets = [packet for arm, packet in WarmLearner.packets if arm == 2]
    assert len(control_packets) == 8 and len(reward_packets) == 16
    assert all(
        type(packet) is DemonstrationUpdate for packet in control_packets + reward_packets[:8]
    )
    assert [packet.episode for packet in control_packets] == [
        packet.episode for packet in reward_packets[:8]
    ]
    assert all(type(packet) is LegalRewardUpdate for packet in reward_packets[8:])
    first_behavior = reward_packets[8].behavior_state
    assert first_behavior.completed_updates == 8 and first_behavior.training_examples == 64
    assert pickle.dumps(first_behavior) == pickle.dumps(WarmLearner.control_states[0])
    assert all(np.any(value != 0) for value in first_behavior.gradients.values())
    assert all(np.any(value != 0) for value in first_behavior.optimizer_state.values())
    assert [point["native_updates"] for point in control["observations"]] == [0, 8]
    assert [point["native_updates"] for point in reward["observations"]] == [0, 8, 10, 12, 14, 16]
    expected_rng, examples = np.random.Generator(np.random.PCG64(29)), 64
    for index, (slot, update) in enumerate(
        zip(reward["training"][8:], reward_packets[8:], strict=True)
    ):
        packet = update.episode.episode
        count = len(packet.transitions)
        assert slot["sampler_before"] == expected_rng.bit_generator.state
        expected_rng.random(count)
        assert slot["sampler_after"] == expected_rng.bit_generator.state
        assert slot["charged_updates_after"] == index + 9
        assert slot["collection"]["behavior_training_examples"] == examples
        assert slot["collection"]["episode"]["episode"]["role"] == "train"
        assert slot["collection"]["action_conditioning"] == "legal_actions_v1"
        assert (
            slot["receipt"]["diagnostic"]["definition"]
            == "torch_recurrent_legal_reward_logprob_sum_v1"
        )
        for transition, legal, raw, behavior in zip(
            packet.transitions,
            update.episode.legal_actions,
            update.episode.raw_probabilities,
            packet.behavior_probabilities,
            strict=True,
        ):
            assert transition.action in legal and raw == (0.2,) * 5
            expected = tuple(1 / len(legal) if action in legal else 0 for action in range(5))
            np.testing.assert_allclose(behavior, expected, rtol=0, atol=1e-15)
        examples += count
    assert reward["sampling_state"] == expected_rng.bit_generator.state
    assert reward["phase_work"]["reward"]["training_rows"] == examples - 64
    assert all(slot["sampler_before"] == slot["sampler_after"] for slot in control["training"])
    assert WarmLearner.session_commits == list(range(16, 24)) + [24] * 64
    assert len(RewardLearner.classifications) == 64 and len(RewardLearner.sessions) == 72
    for row in report["runs"]:
        assert row["action_conditioning"] == "legal_actions_v1"
        assert row["warm_start_matches_control"] is True
        assert row["source_unchanged"] is row["actor_unchanged"] is True
        assert row["phase_work"]["demonstration"]["training_rows"] == 64
        assert all(slot["commit_validated"] is True for slot in row["training"])
        for slot in row["episodes"]:
            assert slot["status"] == "complete" and slot["state_unchanged"] is True
            assert slot["action_conditioning"] == "legal_actions_v1"
            assert slot["outcome"] != "illegal_action" and slot["success"] is False
            assert slot["return"] == 0
            for transition in slot["transitions"]:
                assert transition["action"] in transition["legal_actions"]
                assert transition["raw_probabilities"] == (0.2,) * 5
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
    assert comparison["learning_benefit_established"] is False
    assert report["scientific_advantage_verified"] is report["full_R42_complete"] is False
    assert pickle.dumps(np.random.get_state()) == global_rng


@pytest.mark.parametrize("corruption", ["gradient", "momentum", "policy"])
def test_should_refuse_full_warm_mismatch_before_legal_collection_or_any_final(
    monkeypatch, corruption
):
    collected, budgets, _, _ = environment(monkeypatch)
    monkeypatch.setattr(WarmLearner, "corrupt_warm", corruption)
    report = pilot.run_warm_start_comparison_pilot(condition_on_legal_actions=True)
    assert report["status"] == "incomplete" and report["warm_start_matched"] is False
    assert [budget.updates_completed for budget in budgets] == [8, 8]
    assert WarmLearner.commits == 16
    assert all(type(packet) is DemonstrationUpdate for _, packet in WarmLearner.packets)
    assert len(collected) == 16 and report["demonstrations"] == []
    assert RewardLearner.classifications == RewardLearner.sessions == []
    assert report["comparison"] is None


@pytest.mark.parametrize("phase", ["demonstration", "reward"])
def test_should_preserve_failed_pair_costs_and_skip_final_creation(monkeypatch, phase):
    collected, budgets, _, _ = environment(monkeypatch)
    monkeypatch.setattr(WarmLearner, "fail_phase", phase)
    report = pilot.run_warm_start_comparison_pilot(condition_on_legal_actions=True)
    assert report["status"] == "incomplete" and report["comparison"] is None
    control, reward = report["runs"]
    assert [budget.updates_completed for budget in budgets] == (
        [2] if phase == "demonstration" else [8, 10]
    )
    failed = control["training"][2] if phase == "demonstration" else reward["training"][10]
    assert failed["status"] == "failed" and failed["receipt"] is None
    assert failed["update_committed"] is failed["commit_validated"] is False
    assert report["demonstrations"] == RewardLearner.classifications == []
    assert all(role == "train" for role, _, _ in collected)
    assert all(slot["status"] == "not_run" for row in report["runs"] for slot in row["episodes"])


@pytest.mark.parametrize("target", ["source", "actor", "candidate", "receipt", "conditioning"])
def test_should_validate_deferred_control_again_before_first_peer_commit(monkeypatch, target):
    collected, budgets, rows, runtimes = environment(monkeypatch)
    original = shared._collect

    def collect(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[3] == "train" and WarmLearner.commits == 8:
            control = next(row for row in rows if row.get("arm") == "bc_only")
            if target == "conditioning":
                control["action_conditioning"] = None
            elif target == "receipt":
                control["training"][0]["receipt"]["diagnostic"]["value"] += 1
            else:
                owner = (
                    RewardLearner.sources[0]
                    if target == "source"
                    else runtimes[0].actor._learner
                    if target == "actor"
                    else runtimes[0]._candidate
                )
                assert isinstance(owner, Learner)
                owner.state.parameters["weight_input"][0, 0] += 1
        return result

    monkeypatch.setattr(shared, "_collect", collect)
    report = pilot.run_warm_start_comparison_pilot(condition_on_legal_actions=True)
    assert report["status"] == "incomplete" and report["comparison"] is None
    assert [budget.updates_completed for budget in budgets] == [8, 0]
    assert WarmLearner.commits == len(WarmLearner.packets) == 8
    assert len(collected) == 9 and report["demonstrations"] == []
    assert RewardLearner.classifications == RewardLearner.sessions == []


def test_should_keep_committed_peer_receipt_when_control_changes_during_native_return(monkeypatch):
    _, budgets, rows, _ = environment(monkeypatch)
    original = Learner.train_batch

    def train(self, *args):
        result = original(self, *args)
        if self.arm_number == 2 and self.completed_updates == 1:
            control = next(row for row in rows if row.get("arm") == "bc_only")
            control["action_conditioning"] = None
        return result

    monkeypatch.setattr(Learner, "train_batch", train)
    report = pilot.run_warm_start_comparison_pilot(condition_on_legal_actions=True)
    assert report["status"] == "incomplete"
    assert [budget.updates_completed for budget in budgets] == [8, 1]
    assert WarmLearner.commits == 9 and len(WarmLearner.packets) == 9
    first = report["runs"][1]["training"][0]
    assert first["charged_updates_after"] == 1 and first["receipt"]["update_number"] == 1
    assert first["update_committed"] is True and first["commit_validated"] is False
    assert all(slot["status"] == "not_run" for slot in report["runs"][1]["training"][1:])
    assert report["demonstrations"] == RewardLearner.classifications == RewardLearner.sessions == []


@pytest.mark.parametrize("phase", ["final", "score"])
def test_should_guard_both_completed_arms_around_first_final_callback(monkeypatch, phase):
    collected, budgets, rows, _ = environment(monkeypatch)

    def corrupt():
        reward = next(row for row in rows if row.get("arm") == "warm_start_rl")
        reward["action_conditioning"] = None

    if phase == "final":
        original = shared._collect

        def collect(*args, **kwargs):
            result = original(*args, **kwargs)
            if args[3] == "final_test":
                corrupt()
            return result

        monkeypatch.setattr(shared, "_collect", collect)
    else:
        original_predict = Learner.predict

        def predict(*args, **kwargs):
            result = original_predict(*args, **kwargs)
            corrupt()
            return result

        monkeypatch.setattr(Learner, "predict", predict)
    report = pilot.run_warm_start_comparison_pilot(condition_on_legal_actions=True)
    assert report["status"] == "incomplete" and report["comparison"] is None
    assert [budget.updates_completed for budget in budgets] == [8, 16]
    assert WarmLearner.commits == 24
    assert sum(role == "final_test" for role, _, _ in collected) == (1 if phase == "final" else 8)
    assert len(RewardLearner.classifications) == int(phase == "score")
    assert len(RewardLearner.sessions) == 8
    assert all(slot["status"] == "not_run" for row in report["runs"] for slot in row["episodes"])


def test_should_describe_opt_in_legal_pair_without_resources_and_preserve_other_defaults(
    monkeypatch,
):
    for name in (
        "TorchDemonstrationRewardLearner",
        "ProcessRssSampler",
        "require_torch",
        "collect_key_door_reward_episode",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    old = pilot.describe_warm_start_comparison_protocol()
    assert old == pilot.describe_warm_start_comparison_protocol(condition_on_legal_actions=False)
    legal = pilot.describe_warm_start_comparison_protocol(condition_on_legal_actions=True)
    assert legal["protocol_id"] == "actual_key_door_legal_warm_start_comparison_v1"
    assert legal["action_conditioning"] == "legal_actions_v1"
    assert legal["planned_work"] == old["planned_work"]
    assert legal["planned_work"]["native_train_calls"] == 24
    assert legal["planned_work"]["score_forwards"] == legal["planned_work"]["live_episodes"] == 64
    assert "action_conditioning" not in old
    assert (
        pilot.describe_protocol(condition_on_legal_actions=True)["protocol_id"]
        == "actual_key_door_legal_on_policy_recurrent_rl_v1"
    )


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", np.bool_(True)])
@pytest.mark.parametrize(
    "function", ["describe_warm_start_comparison_protocol", "run_warm_start_comparison_pilot"]
)
def test_should_refuse_non_native_conditioning_bool_before_any_resources(
    monkeypatch, invalid, function
):
    for name in (
        "TorchDemonstrationRewardLearner",
        "ProcessRssSampler",
        "require_torch",
        "collect_key_door_reward_episode",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(shared, "_collect", forbidden)
    with pytest.raises(ValueError):
        getattr(pilot, function)(condition_on_legal_actions=invalid)


@pytest.mark.parametrize("describe", [False, True])
def test_should_route_only_explicit_legal_pair_cli_with_native_bool(monkeypatch, capsys, describe):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-key-door-legal-warm-start-control"]
        + (["--describe"] if describe else []),
    )
    chosen = (
        "describe_legal_warm_start_comparison_protocol"
        if describe
        else "run_legal_warm_start_comparison_pilot"
    )
    calls = []

    def selected(**kwargs):
        calls.append(kwargs)
        return dict(status="complete", selected="legal-pair")

    for name in (
        "describe_legal_warm_start_comparison_protocol",
        "run_legal_warm_start_comparison_pilot",
    ):
        monkeypatch.setattr(cli, name, selected if name == chosen else forbidden)
    for name in (
        "describe_warm_start_comparison_protocol",
        "run_warm_start_comparison_pilot",
        "describe_legal_reward_protocol",
        "run_legal_reward_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    cli.main()
    assert calls == [dict(condition_on_legal_actions=True)]
    assert json.loads(capsys.readouterr().out)["selected"] == "legal-pair"


@pytest.mark.parametrize("changed", ["score", "outcome"])
def test_should_refuse_changed_completed_control_final_before_more_peer_scoring(
    monkeypatch, changed
):
    _, budgets, rows, _ = environment(monkeypatch)
    original = Learner.predict
    corrupted: list[str] = []

    def predict(self, features):
        result = original(self, features)
        control = next(row for row in rows if row.get("arm") == "bc_only")
        if self.arm_number == 2 and control["status"] == "complete" and not corrupted:
            if changed == "score":
                control["stream_result"]["checkpoints"][-1]["scores"]["A-red"]["ce"] += 1.0
            else:
                control["episodes"][-1]["return"] += 1.0
            corrupted.append(changed)
        return result

    monkeypatch.setattr(Learner, "predict", predict)
    report = pilot.run_warm_start_comparison_pilot(condition_on_legal_actions=True)
    assert corrupted == [changed]
    assert report["status"] == "incomplete" and report["comparison"] is None
    assert [budget.updates_completed for budget in budgets] == [8, 16]
    assert WarmLearner.commits == 24
    control, reward = report["runs"]
    assert control["status"] == "complete" and reward["status"] == "failed"
    assert "completed paired final evidence changed" in reward["reason"]
    assert len(RewardLearner.classifications) == 17
    assert all(slot["status"] == "not_run" for slot in reward["episodes"])
