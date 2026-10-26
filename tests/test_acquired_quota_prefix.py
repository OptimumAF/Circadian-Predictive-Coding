"""Owned A-only prefixes through the ordinary actor, receipts and quota runner."""

from copy import deepcopy
import json
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.key_door_acquired_quota as pilot
import src.adapters.key_door_pilot as shared
import src.adapters.key_door_reward_pilot as recipe
from src.adapters.torch_demonstration_reward_learner import DemonstrationUpdate
from src.adapters.torch_reward_update_learner import LegalRewardUpdate
from src.app.key_door_training import KeyDoorTrainingFailed
from src.core.learner_ports import TrainingDiagnostic
from test_key_door_reward_pilot import environment as base_environment
from test_legal_reward_pilot import Learner
from test_torch_feature_rehearsal_pilot import forbidden


PREFIX_UPDATES = 192
PREFIX_ROWS = 1536
PHYSICAL_POINTS = [0, 24, 48, 72, 96]


def environment(monkeypatch):
    """Fake native arithmetic, real recipe/runtime/collector; no acquisition claim."""
    base_environment(monkeypatch)
    # Both forms support the adapter's resource-port import during composition.
    monkeypatch.setattr(pilot, "ProcessRssSampler", recipe.ProcessRssSampler, raising=False)
    monkeypatch.setattr(pilot, "require_torch", recipe.require_torch, raising=False)
    monkeypatch.setattr(recipe, "TorchRewardUpdateLearner", Learner)
    monkeypatch.setattr(recipe, "TorchDemonstrationRewardLearner", Learner)
    for name in ("sessions", "updates", "identifiers", "classifications", "sources"):
        monkeypatch.setattr(Learner, name, [])
    monkeypatch.setattr(Learner, "committed", 0)
    monkeypatch.setattr(Learner, "fail_collection", False)
    monkeypatch.setattr(Learner, "fail_update", False)
    monkeypatch.setattr(Learner, "raw", (0.2,) * 5)
    evidence: dict[str, Any] = dict(
        active={},
        rows=[],
        budgets=[],
        updates=[],
        attempts=[],
        clocks=[],
        prefix_states={},
        physical={},
        reward_collections=[],
        demonstrations=[],
        final_packets=[],
        predictions=[],
        final_started=False,
        after_train=None,
        fail_demo_at=None,
        deadlines=[],
    )
    original_prepare = recipe._prepare_update
    original_train_one = recipe._train_one
    original_collect = recipe.collect_key_door_reward_episode
    original_packet = shared._collect
    original_budget = shared._session
    original_prefix_budget = recipe.ToyBudgetSession
    original_arm = recipe._run_arm

    def arm(*args, **options):
        evidence["deadlines"].append(args[2])
        return original_arm(*args, **options)

    def prepare(row, source, state, rng, budget, index, identity, **options):
        if all(row is not old for old in evidence["rows"]):
            evidence["rows"].append(row)
        evidence["active"].update(
            row=row,
            source=source,
            index=index,
            budget=budget,
            quota=options.get("interaction_quota"),
        )
        return original_prepare(row, source, state, rng, budget, index, identity, **options)

    def train_one(*args, **options):
        row, clock = args[0], args[8]
        evidence["clocks"].append((row["arm"], id(clock), args[7], clock.now()))
        return original_train_one(*args, **options)

    def train(self, update, identity):
        row = evidence["active"]["row"]
        demonstration = type(update) is DemonstrationUpdate
        assert demonstration or type(update) is LegalRewardUpdate
        packet = update.episode if demonstration else update.episode.episode
        assert packet.role == "train" and packet.episode_id == identity
        if demonstration:
            assert (
                packet.context
                == f"A-{'red' if row['training'][evidence['active']['index']]['cue'] == 1 else 'blue'}"
            )
        else:
            assert packet.context in ("A", "B", "C")
        if not demonstration:
            assert row["arm"] == "reward_prefix", "frozen control entered native reward training"
            assert self.state.completed_updates >= PREFIX_UPDATES
            assert pickle.dumps(update.behavior_state) == pickle.dumps(self.state)
        evidence["attempts"].append((row["arm"], "demonstration" if demonstration else "reward"))
        if demonstration and evidence["fail_demo_at"] == self.state.completed_updates + 1:
            raise RuntimeError("declared prefix update failed atomically")
        before = self.snapshot_state()
        for name, value in self.state.parameters.items():
            gradient = np.full_like(value, 0.25 if demonstration else 0.125)
            velocity = self.state.optimizer_state.get(name, np.zeros_like(value)) * 0.9 + gradient
            self.state.gradients[name] = gradient
            self.state.optimizer_state[name] = velocity
            value -= 0.03 * velocity
        self.state.completed_updates += 1
        self.state.training_examples += len(packet.transitions)
        if evidence["after_train"] is not None:
            evidence["after_train"](self, update, row)
        if self.state.completed_updates == PREFIX_UPDATES:
            evidence["prefix_states"][row["arm"]] = self.snapshot_state()
        evidence["updates"].append((row["arm"], deepcopy(update), before, self.snapshot_state()))
        return TrainingDiagnostic(
            "torch_recurrent_episode_cross_entropy_mean_v1"
            if demonstration
            else "torch_recurrent_legal_reward_logprob_sum_v1",
            0.5 if demonstration else 0.0,
        )

    def collect(source, layout, cue, **options):
        arm = evidence["active"]["row"]["arm"]
        try:
            report = original_collect(source, layout, cue, **options)
        except KeyDoorTrainingFailed as error:
            evidence["physical"][arm] = (
                evidence["physical"].get(arm, 0) + error.report["environment_steps"]
            )
            evidence["reward_collections"].append((arm, deepcopy(error.report)))
            raise
        assert report["behavior_state"].completed_updates >= PREFIX_UPDATES
        evidence["physical"][arm] = evidence["physical"].get(arm, 0) + report["environment_steps"]
        evidence["reward_collections"].append((arm, deepcopy(report)))
        return report

    def packet(identity, layout, cue, role):
        if role == "train":
            assert layout.layout_id == "A"
            assert (layout.rotation, layout.branch_row, layout.swap_keys, layout.appearance) == (
                0,
                1,
                False,
                0.0,
            )
            assert cue == (1 if evidence["active"]["index"] % 2 == 0 else -1)
            result = original_packet(identity, layout, cue, role)
            evidence["demonstrations"].append((evidence["active"]["row"]["arm"], result))
            return result
        assert role == "final_test"
        assert set(evidence["prefix_states"]) == {"frozen_prefix", "reward_prefix"}
        assert evidence["physical"] == {"frozen_prefix": 96, "reward_prefix": 96}, (
            "final data preceded both completed arms"
        )
        evidence["final_started"] = True
        result = original_packet(identity, layout, cue, role)
        evidence["final_packets"].append(result)
        return result

    def predict(self, features):
        assert features.shape == (8, 8) and features.dtype == np.float64
        assert self.state.completed_updates >= PREFIX_UPDATES
        if evidence["final_started"]:
            assert evidence["physical"] == {"frozen_prefix": 96, "reward_prefix": 96}
        else:
            assert any(
                np.array_equal(features, batch.features) for _, batch in evidence["demonstrations"]
            ), "unknown pre-final prediction input"
        evidence["predictions"].append(
            (evidence["final_started"], self.state.completed_updates, features.copy())
        )
        Learner.classifications.append(features.copy())
        return np.tile(np.array([Learner.raw]), (len(features), 1))

    def budget(*args, **options):
        result = original_budget(*args, **options)
        if all(result is not prior for prior in evidence["budgets"]):
            evidence["budgets"].append(result)
        return result

    def prefix_budget(*args, **options):
        result = original_prefix_budget(*args, **options)
        if all(result is not prior for prior in evidence["budgets"]):
            evidence["budgets"].append(result)
        return result

    monkeypatch.setattr(recipe, "_prepare_update", prepare)
    monkeypatch.setattr(recipe, "_run_arm", arm)
    monkeypatch.setattr(recipe, "_train_one", train_one)
    monkeypatch.setattr(recipe, "collect_key_door_reward_episode", collect)
    monkeypatch.setattr(shared, "_collect", packet)
    monkeypatch.setattr(shared, "_session", budget)
    monkeypatch.setattr(recipe, "ToyBudgetSession", prefix_budget)
    monkeypatch.setattr(Learner, "train_batch", train)
    monkeypatch.setattr(Learner, "predict", predict)
    return evidence


def test_should_keep_original_receipts_and_budget_through_owned_prefix_collection_and_reward(
    monkeypatch,
):
    evidence = environment(monkeypatch)
    report = pilot.run_pilot()
    assert report["status"] == "complete", (
        report["global_errors"],
        [(r["status"], r["reason"]) for r in report["runs"]],
    )
    rows = report["runs"]
    assert [row["arm"] for row in rows] == ["frozen_prefix", "reward_prefix"]
    assert report["prefix_matched"] is True
    assert len(evidence["budgets"]) == 2
    assert len(evidence["deadlines"]) == 2 and evidence["deadlines"][0] == evidence["deadlines"][1]
    assert [budget.budget.max_training_updates for budget in evidence["budgets"]] == [192, 208]
    assert pickle.dumps(evidence["prefix_states"]["frozen_prefix"]) == pickle.dumps(
        evidence["prefix_states"]["reward_prefix"]
    )
    assert all(
        np.any(value != 0)
        for value in evidence["prefix_states"]["frozen_prefix"].optimizer_state.values()
    )
    assert evidence["physical"] == {"frozen_prefix": 96, "reward_prefix": 96}
    assert len(evidence["demonstrations"]) == 384
    assert len(evidence["final_packets"]) == 8
    for row, budget in zip(rows, evidence["budgets"], strict=True):
        arm = row["arm"]
        assert row["acquired_prefix"] == "a_only_192_v1" and row["prefix_matches_control"] is True
        assert row["prefix"]["updates"] == 192 and row["prefix"]["training_examples"] == 1536
        assert row["prefix"]["source_role"] == "train" and row["prefix"]["context"] == "A"
        assert row["prefix"]["acquired_skill_verified"] is False
        assert row["environment_seeds"] == dict(training=0, final_only=23)
        prefix, quota = row["training"][:192], row["training"][192:]
        assert len(prefix) == 192 and 8 <= len(quota) <= 16
        rng = np.random.Generator(np.random.PCG64(29))
        for index, slot in enumerate(prefix):
            assert slot["context"] == "A" and slot["cue"] == (1 if index % 2 == 0 else -1)
            assert slot["phase"] == "demonstration" and slot["training_rows"] == 8
            assert slot["demonstration"]["role"] == "train" and slot["collection"] is None
            assert slot["update_committed"] is slot["commit_validated"] is True
            assert slot["receipt"]["update_number"] == index + 1
            assert (
                slot["charged_updates_before"] == index
                and slot["charged_updates_after"] == index + 1
            )
            assert slot["sampler_before"] == slot["sampler_after"] == rng.bit_generator.state
        for slot in quota:
            assert slot["collection"]["sampler_before"] == rng.bit_generator.state
            rng.random(slot["collection"]["sampling_draws"])
            assert slot["collection"]["sampler_after"] == rng.bit_generator.state
            if arm == "frozen_prefix":
                assert slot["collection_only"] is slot["collection_validated"] is True
                assert (
                    slot["receipt"] is None
                    and slot["update_committed"] is slot["commit_validated"] is False
                )
                assert slot["charged_updates_before"] == slot["charged_updates_after"] == 192
                assert slot["collection"]["behavior_native_updates"] == 192
                assert slot["collection"]["behavior_training_examples"] == 1536
            else:
                assert slot["update_committed"] is slot["commit_validated"] is True
                assert slot["receipt"] is not None
        assert row["sampling_state"] == rng.bit_generator.state
        expected_updates = 192 if arm == "frozen_prefix" else 192 + len(quota)
        expected_rows = 1536 if arm == "frozen_prefix" else 1632
        assert row["wake_updates"] == budget.updates_completed == expected_updates
        assert row["stream_result"]["completed_updates"] == expected_updates
        assert row["stream_result"]["training_examples"] == expected_rows
        assert row["stream_result"]["score_forward_calls"] == 40
        assert [
            point["physical_steps"] for point in row["stream_result"]["checkpoints"]
        ] == PHYSICAL_POINTS
        assert [point["physical_steps"] for point in row["quota_checkpoints"]] == PHYSICAL_POINTS
        assert row["quota_checkpoints"][0]["native_updates"] == 192
        if arm == "frozen_prefix":
            assert all(point["native_updates"] == 192 for point in row["quota_checkpoints"])
            assert all(point["updates"] == 192 for point in row["stream_result"]["checkpoints"])
        assert len(row["episodes"]) == row["live_sessions_started"] == 40
        for point in PHYSICAL_POINTS:
            assert (
                len(
                    [
                        episode
                        for episode in row["episodes"]
                        if episode["checkpoint_physical_steps"] == point
                    ]
                )
                == 8
            )
        assert row["source_unchanged"] is row["actor_unchanged"] is True
        clocks = [record for record in evidence["clocks"] if record[0] == arm]
        assert len({record[1] for record in clocks}) == 1
        assert [record[2] for record in clocks] == list(range(expected_updates))
        assert all(index == observed for _, _, index, observed in clocks)
    frozen_collections = [
        collection for arm, collection in evidence["reward_collections"] if arm == "frozen_prefix"
    ]
    assert all(
        pickle.dumps(collection["behavior_state"])
        == pickle.dumps(evidence["prefix_states"]["frozen_prefix"])
        for collection in frozen_collections
    )
    assert report["comparison"] is not None
    assert report["comparison"]["initial_A_acquired"] == dict(
        frozen_prefix=False, reward_prefix=False
    )
    assert "inconclusive" in report["comparison"]["interpretation"]
    assert report["scientific_advantage_verified"] is report["full_R43_complete"] is False


def test_should_stop_failed_prefix_before_any_quota_or_second_arm_and_keep_native_charges(
    monkeypatch,
):
    evidence = environment(monkeypatch)
    evidence["fail_demo_at"] = 3
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    first, second = report["runs"]
    assert first["status"] == "failed" and second["status"] == "not_run"
    assert len(evidence["budgets"]) == 1 and evidence["budgets"][0].updates_completed == 2
    assert first["wake_updates"] == 2
    assert first["training"][2]["status"] == "failed" and first["training"][2]["receipt"] is None
    assert all(slot["status"] == "not_run" for slot in first["training"][3:])
    assert len(evidence["updates"]) == 2 and len(evidence["attempts"]) == 3
    assert not evidence["reward_collections"] and not evidence["final_packets"]
    assert report["comparison"] is None and report["prefix_matched"] is False


def test_should_refuse_optimizer_only_prefix_mismatch_before_reward_continuation_or_finals(
    monkeypatch,
):
    evidence = environment(monkeypatch)

    def mismatch(self, update, row):
        if row["arm"] == "reward_prefix" and self.state.completed_updates == 192:
            self.state.optimizer_state["bias_hidden"][0, 0] += 0.25

    evidence["after_train"] = mismatch
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and report["prefix_matched"] is False
    assert evidence["physical"] == {"frozen_prefix": 96}
    assert [budget.updates_completed for budget in evidence["budgets"]] == [192, 192]
    assert len(evidence["updates"]) == 384
    assert not evidence["final_packets"] and report["comparison"] is None
    frozen, reward = (
        evidence["prefix_states"][name] for name in ("frozen_prefix", "reward_prefix")
    )
    for name in frozen.parameters:
        np.testing.assert_array_equal(frozen.parameters[name], reward.parameters[name])
    assert not np.array_equal(
        frozen.optimizer_state["bias_hidden"], reward.optimizer_state["bias_hidden"]
    )


def test_should_refuse_changed_prior_prefix_receipt_without_erasing_later_paid_commit(monkeypatch):
    evidence = environment(monkeypatch)

    def mutate(self, update, row):
        if self.state.completed_updates == 3:
            row["training"][0]["receipt"]["update_number"] = 0

    evidence["after_train"] = mutate
    report = pilot.run_pilot()
    first = report["runs"][0]
    assert report["status"] == "incomplete" and first["status"] == "failed"
    assert first["training"][0]["receipt"]["update_number"] == 0
    assert first["training"][2]["update_committed"] is True
    assert evidence["budgets"][0].updates_completed == first["wake_updates"] == 3
    assert not evidence["reward_collections"] and not evidence["final_packets"]
    assert report["runs"][1]["status"] == "not_run"


@pytest.mark.parametrize("describe", (False, True))
def test_should_dispatch_fixed_acquired_prefix_cli_without_user_options(
    monkeypatch, capsys, describe
):
    calls = []

    def selected():
        calls.append(True)
        return dict(status="complete")

    monkeypatch.setattr(cli, "describe_acquired_prefix_protocol", selected)
    monkeypatch.setattr(cli, "run_acquired_prefix_pilot", selected)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run", "--scenario", "torch-key-door-acquired-interaction-quota"]
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [True] and json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "flags",
    (
        ["--environment-seed", "0", "--heldout-environment-seed", "23"],
        ["--source-artifact", "old-prefix.json"],
    ),
)
def test_should_refuse_caller_seed_or_state_artifact_in_fixed_prefix_cli(monkeypatch, flags):
    monkeypatch.setattr(cli, "describe_acquired_prefix_protocol", forbidden)
    monkeypatch.setattr(cli, "run_acquired_prefix_pilot", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["run", "--scenario", "torch-key-door-acquired-interaction-quota", *flags]
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


@pytest.mark.parametrize("selected", ("describe_protocol", "run_pilot"))
def test_should_refuse_caller_control_of_fixed_prefix_dose_or_environment(selected):
    with pytest.raises(TypeError):
        getattr(pilot, selected)(prefix_updates=8)
    with pytest.raises(TypeError):
        getattr(pilot, selected)(environment_seed=1)


def test_should_keep_default_reward_and_frozen_descriptors_unchanged_by_prefix_description(
    monkeypatch,
):
    route = dict(
        condition_on_legal_actions=True, action_remapping=True, fill_interaction_quota=True
    )
    old = deepcopy(recipe.describe_protocol())
    frozen = deepcopy(recipe.describe_protocol(**route, frozen_recurrent_control=True))
    for name in ("ProcessRssSampler", "require_torch", "TorchDemonstrationRewardLearner"):
        monkeypatch.setattr(recipe, name, forbidden)
    protocol = pilot.describe_protocol()
    assert protocol["protocol_id"] == "actual_key_door_acquired_quota_prefix_v1"
    assert protocol["arms"] == ["frozen_prefix", "reward_prefix"]
    assert protocol["prefix"]["updates"] == 192 and protocol["prefix"]["training_rows"] == 1536
    assert protocol["prefix"]["context"] == "A" and protocol["prefix"]["cues"] == [1, -1]
    assert protocol["native_caps"] == dict(frozen_prefix=192, reward_prefix=208)
    assert protocol["maximum_wall_seconds"] == 240
    assert protocol["physical_boundaries"] == PHYSICAL_POINTS
    assert protocol["base_protocol"]["environment_seeds"] == dict(training=0, final_only=23)
    assert (
        protocol["base_protocol"]["seed"] == 17 and protocol["base_protocol"]["sampling_seed"] == 29
    )
    assert protocol["planned_work"] == dict(
        maximum_native_train_calls=400,
        maximum_training_rows=3168,
        prefix_updates=384,
        prefix_rows=3072,
        collection_steps=192,
        maximum_collection_episodes=32,
        score_calls=80,
        score_rows=640,
        live_episodes=80,
        maximum_live_steps=960,
        maximum_prediction_calls=1232,
        maximum_unroll_steps=4960,
        maximum_environment_steps=4288,
        maximum_environment_episodes=504,
        replay_updates=0,
        probe_updates=0,
    )
    assert recipe.describe_protocol() == old
    assert recipe.describe_protocol(**route, frozen_recurrent_control=True) == frozen


def test_should_refuse_corrupted_preceding_receipt_during_collection_before_next_prefix_sgd(
    monkeypatch,
):
    evidence = environment(monkeypatch)
    original = shared._collect

    def corrupt(identity, layout, cue, role):
        packet = original(identity, layout, cue, role)
        if role == "train" and evidence["active"]["index"] == 1:
            evidence["active"]["row"]["training"][0]["receipt"]["update_number"] = 0
        return packet

    monkeypatch.setattr(shared, "_collect", corrupt)
    report = pilot.run_pilot()
    first, second = report["runs"]
    assert report["status"] == "incomplete" and first["status"] == "failed"
    assert first["wake_updates"] == evidence["budgets"][0].updates_completed == 1
    assert len(evidence["attempts"]) == len(evidence["updates"]) == 1
    assert first["training"][0]["receipt"]["update_number"] == 0
    assert first["training"][1]["receipt"] is None
    assert first["training"][1]["demonstration"] is not None
    assert second["status"] == "not_run" and not evidence["final_packets"]


@pytest.mark.parametrize("field,value", [("cue", True), ("index", False)])
def test_should_refuse_bool_plan_substitution_after_prefix_collection_before_sgd(
    monkeypatch, field, value
):
    evidence = environment(monkeypatch)
    original = shared._collect

    def corrupt(identity, layout, cue, role):
        packet = original(identity, layout, cue, role)
        if role == "train":
            evidence["active"]["row"]["training"][0][field] = value
        return packet

    monkeypatch.setattr(shared, "_collect", corrupt)
    report = pilot.run_pilot()
    first, second = report["runs"]
    assert report["status"] == "incomplete" and first["status"] == "failed"
    assert first["wake_updates"] == evidence["budgets"][0].updates_completed == 0
    assert not evidence["attempts"] and not evidence["updates"]
    assert first["training"][0]["demonstration"] is not None
    assert first["demonstration_collection_attempts"] == 1
    assert type(first["training"][0][field]) is bool
    assert second["status"] == "not_run" and not evidence["final_packets"]
