"""A frozen recurrent control spends physical steps without native training."""

from copy import deepcopy
import json
import pickle
import sys
from typing import Any, TypedDict

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.key_door_pilot as shared
import src.adapters.key_door_reward_pilot as pilot
from src.app.key_door_training import KeyDoorTrainingFailed
from src.core.key_door_navigation import KeyDoorNavigation, key_door_layout_record, scripted_actions
from test_key_door_reward_pilot import environment as base_environment
from test_legal_reward_pilot import Learner, Session
from test_torch_feature_rehearsal_pilot import forbidden


ROUTE = dict(condition_on_legal_actions=True, action_remapping=True, fill_interaction_quota=True)


class SeedOptions(TypedDict):
    environment_seed: int
    heldout_environment_seed: int


SEEDS: SeedOptions = dict(environment_seed=17, heldout_environment_seed=6)
PHYSICAL_POINTS = [0, 24, 48, 72, 96]


def environment(monkeypatch):
    """Use the existing actor and pure game with frozen fake probability sources."""
    base_environment(monkeypatch)
    monkeypatch.setattr(pilot, "TorchRewardUpdateLearner", Learner)
    for name in ("sessions", "updates", "identifiers", "classifications", "sources"):
        monkeypatch.setattr(Learner, name, [])
    monkeypatch.setattr(Learner, "committed", 0)
    monkeypatch.setattr(Learner, "fail_collection", False)
    monkeypatch.setattr(Learner, "fail_update", False)
    monkeypatch.setattr(Learner, "raw", (0.2,) * 5)
    monkeypatch.setattr(Learner, "train_batch", forbidden)
    evidence: dict[str, Any] = dict(
        physics=0, attempts=0, collections=[], finals=[], budgets=[], active={}
    )
    original_collect = pilot.collect_key_door_reward_episode
    original_final = shared._collect
    original_budget = shared._session
    original_prepare = pilot._prepare_update

    def prepare(row, source, state, rng, budget, index, identity, **options):
        evidence["active"].update(
            row=row, source=source, index=index, quota=options["interaction_quota"]
        )
        return original_prepare(row, source, state, rng, budget, index, identity, **options)

    def collect(source, layout, cue, **options):
        try:
            result = original_collect(source, layout, cue, **options)
        except KeyDoorTrainingFailed as error:
            result = error.report
            evidence["physics"] += result["environment_steps"]
            evidence["attempts"] += result["environment_steps_attempted"]
            evidence["collections"].append((layout, cue, deepcopy(result)))
            raise
        evidence["physics"] += result["environment_steps"]
        evidence["attempts"] += result["environment_steps_attempted"]
        evidence["collections"].append((layout, cue, deepcopy(result)))
        return result

    def final(*args, **options):
        assert evidence["physics"] == evidence["attempts"] == 96, (
            "final packet preceded complete physical allowance"
        )
        assert args[3] == "final_test"
        result = original_final(*args, **options)
        evidence["finals"].append((args, result))
        return result

    def predict(self, features):
        assert evidence["physics"] == 96, "final scoring preceded collection"
        assert features.shape == (8, 8) and features.dtype == np.float64
        assert self.completed_updates == self.training_examples == 0
        Learner.classifications.append(features.copy())
        return np.tile(np.array([Learner.raw]), (len(features), 1))

    def budget(*args, **options):
        result = original_budget(*args, **options)
        evidence["budgets"].append(result)
        return result

    monkeypatch.setattr(pilot, "_prepare_update", prepare)
    monkeypatch.setattr(pilot, "collect_key_door_reward_episode", collect)
    monkeypatch.setattr(shared, "_collect", final)
    monkeypatch.setattr(shared, "_session", budget)
    monkeypatch.setattr(Learner, "predict", predict)
    return evidence


def run(**options):
    return pilot.run_pilot(**ROUTE, frozen_recurrent_control=True, **options)


@pytest.mark.parametrize("describe", (False, True))
@pytest.mark.parametrize("invalid", (None, 0, 1, "true", np.bool_(True)))
def test_should_refuse_non_bool_frozen_mode_before_resources_and_clock(
    monkeypatch, describe, invalid
):
    for name in ("monotonic", "ProcessRssSampler", "require_torch", "TorchRewardUpdateLearner"):
        monkeypatch.setattr(pilot, name, forbidden)
    selected = pilot.describe_protocol if describe else pilot.run_pilot
    with pytest.raises(ValueError):
        selected(**ROUTE, frozen_recurrent_control=invalid)


@pytest.mark.parametrize("describe", (False, True))
@pytest.mark.parametrize(
    "route",
    (
        {},
        {"condition_on_legal_actions": True},
        {"condition_on_legal_actions": True, "action_remapping": True},
    ),
)
def test_should_refuse_frozen_mode_outside_the_standalone_mapped_quota_route(
    monkeypatch, describe, route
):
    for name in ("monotonic", "ProcessRssSampler", "require_torch", "TorchRewardUpdateLearner"):
        monkeypatch.setattr(pilot, name, forbidden)
    selected = pilot.describe_protocol if describe else pilot.run_pilot
    with pytest.raises(ValueError):
        selected(**route, frozen_recurrent_control=True)


@pytest.mark.parametrize(
    "route",
    (
        {},
        {"condition_on_legal_actions": True},
        {"condition_on_legal_actions": True, "action_remapping": True},
        ROUTE,
        dict(**ROUTE, **SEEDS),
    ),
)
def test_should_preserve_old_descriptors_when_frozen_control_is_disabled(monkeypatch, route):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    original = deepcopy(pilot.describe_protocol(**route))
    assert original == pilot.describe_protocol(**route, frozen_recurrent_control=False)
    assert "learning_mode" not in original


def test_should_describe_zero_training_and_full_collection_evaluation_costs_without_models(
    monkeypatch,
):
    for name in (
        "ProcessRssSampler",
        "require_torch",
        "TorchRewardUpdateLearner",
        "collect_key_door_reward_episode",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    old = deepcopy(pilot.describe_protocol(**ROUTE, **SEEDS))
    protocol = pilot.describe_protocol(**ROUTE, **SEEDS, frozen_recurrent_control=True)
    assert protocol["protocol_id"] == "actual_key_door_frozen_interaction_quota_v1"
    assert protocol["classification"] == "engineering_nonlearning_control"
    assert protocol["learning_mode"] == "frozen_recurrent_control_v1"
    assert protocol["methods"] == ["recurrent_rl"]
    assert protocol["seed"] == 17 and protocol["sampling_seed"] == 29
    assert protocol["environment_seeds"] == {"training": 17, "final_only": 6}
    assert (
        protocol["layouts"] == old["layouts"]
        and protocol["native_defaults"] == old["native_defaults"]
    )
    assert protocol["checkpoint_physical_steps"] == PHYSICAL_POINTS
    work = protocol["planned_work"]
    for field in (
        "wake_updates",
        "native_train_calls",
        "minimum_native_train_calls",
        "maximum_native_train_calls",
        "exact_training_rows",
        "maximum_training_rows",
    ):
        assert work[field] == 0
    assert work["exact_collection_rows"] == 96
    assert work["score_forwards"] == work["live_episodes"] == work["live_sessions"] == 40
    assert work["score_rows"] == 320
    assert old == pilot.describe_protocol(**ROUTE, **SEEDS)


def test_should_preserve_old_run_dispatch_without_forwarding_false_control_keyword(monkeypatch):
    calls = []

    def selected(**options):
        calls.append(options)
        return {"status": "sentinel"}

    monkeypatch.setattr(pilot, "_run_pilot", selected)
    pilot.run_pilot(**ROUTE)
    pilot.run_pilot(**ROUTE, frozen_recurrent_control=False)
    assert calls == [dict(warm_start=False, **ROUTE)] * 2


@pytest.mark.parametrize("seeds", ({}, SEEDS))
@pytest.mark.parametrize("scripted", (False, True))
def test_should_collect_own_physics_with_no_training_then_evaluate_five_distinct_physical_stages(
    monkeypatch, seeds, scripted
):
    evidence = environment(monkeypatch)
    if scripted:
        original_predict = Session.predict
        layouts = {layout.layout_id: layout for layout in pilot.action_remapping_layouts(**seeds)}

        def predict(self, features):
            if evidence["physics"] == 96:
                return original_predict(self, features)
            active = evidence["active"]
            slot = active["row"]["training"][active["index"]]
            action = scripted_actions(layouts[slot["context"]], slot["cue"])[self.completed_steps]
            self.inputs.append(features.copy())
            self.completed_steps += 1
            result = np.zeros((1, 5), dtype=np.float64)
            result[0, action] = 1
            return result

        monkeypatch.setattr(Session, "predict", predict)
    global_rng = pickle.dumps(np.random.get_state())
    report = run(**seeds)
    row = report["runs"][0]
    assert report["status"] == "complete", report
    assert row["method"] == "recurrent_rl" and row["learning_mode"] == "frozen_recurrent_control_v1"
    assert row["wake_updates"] == Learner.committed == 0
    assert Learner.updates == Learner.identifiers == []
    assert len(evidence["budgets"]) == 1
    budget = evidence["budgets"][0]
    assert budget.budget.max_training_updates == budget.updates_completed == 0
    assert budget.replay_examples_completed == 0
    assert evidence["physics"] == evidence["attempts"] == 96
    assert 8 <= len(evidence["collections"]) == len(row["training"]) <= 16
    if scripted:
        assert [slot["collection"]["environment_steps"] for slot in row["training"]] == [8, 4] * 8
        assert [slot["collection"]["termination"] for slot in row["training"]] == [
            "terminal",
            "truncated",
        ] * 8
    assert (
        row["physical_quotas"]
        == [dict(maximum_steps=12, attempted_steps=12, completed_steps=12, failed_steps=0)] * 8
    )
    assert row["quota_checkpoints"] == [
        dict(physical_steps=point, native_updates=0) for point in PHYSICAL_POINTS
    ]
    expected_rng = np.random.Generator(np.random.PCG64(29))
    declared = {layout.layout_id: layout for layout in pilot.action_remapping_layouts(**seeds)}
    for slot in row["training"]:
        assert (
            slot["status"] == "complete"
            and slot["collection_only"] is slot["collection_validated"] is True
        )
        assert (
            slot["update_attempted"]
            is slot["update_committed"]
            is slot["commit_validated"]
            is False
        )
        assert slot["receipt"] is None
        assert slot["charged_updates_before"] == slot["charged_updates_after"] == 0
        assert slot["native_updates_before"] == slot["native_training_examples_before"] == 0
        assert slot["context"] in ("A", "B", "C")
        assert slot["layout"] == key_door_layout_record(declared[slot["context"]])
        collection = slot["collection"]
        assert (
            collection["behavior_native_updates"] == collection["behavior_training_examples"] == 0
        )
        assert collection["sampler_before"] == expected_rng.bit_generator.state
        expected_rng.random(collection["sampling_draws"])
        assert collection["sampler_after"] == expected_rng.bit_generator.state
    assert row["sampling_state"] == expected_rng.bit_generator.state
    assert len(evidence["finals"]) == 8 and all(
        args[3] == "final_test" for args, _ in evidence["finals"]
    )
    assert {args[1].layout_id for args, _ in evidence["finals"]} == {"A", "B", "C", "D"}
    assert all(args[1] == declared[args[1].layout_id] for args, _ in evidence["finals"])
    assert len(Learner.classifications) == row["stream_result"]["score_forward_calls"] == 40
    assert (
        row["stream_result"]["completed_updates"] == row["stream_result"]["training_examples"] == 0
    )
    assert [point["updates"] for point in row["stream_result"]["checkpoints"]] == [0] * 5
    assert [
        point["physical_steps"] for point in row["stream_result"]["checkpoints"]
    ] == PHYSICAL_POINTS
    assert [
        point["physical_steps"] for point in row["physical_evaluation_progress"]
    ] == PHYSICAL_POINTS
    assert all(point["status"] == "complete" for point in row["physical_evaluation_progress"])
    assert all(
        point["score_progress"]["prediction_calls_completed"] == 8
        for point in row["physical_evaluation_progress"]
    )
    assert [point["physical_steps"] for point in row["observations"]] == PHYSICAL_POINTS
    assert all(
        point["native_updates"] == point["native_training_examples"] == 0
        for point in row["observations"]
    )
    assert len(row["episodes"]) == row["live_sessions_started"] == 40
    for point in PHYSICAL_POINTS:
        episodes = [
            episode for episode in row["episodes"] if episode["checkpoint_physical_steps"] == point
        ]
        assert len(episodes) == 8
        assert all(
            episode["checkpoint_updates"] == 0 and episode["status"] == "complete"
            for episode in episodes
        )
        assert all(episode["state_unchanged"] is True for episode in episodes)
    assert len(Learner.sessions) == len(row["training"]) + 40
    assert all(session.inputs[0][0, 6] in (-1, 1) for session in Learner.sessions)
    assert all(
        source.completed_updates == source.training_examples == 0 for source in Learner.sources
    )
    reference = pickle.dumps(getattr(Learner.sources[0], "snapshot_state")())
    assert all(pickle.dumps(session.snapshot_state()) == reference for session in Learner.sessions)
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert (
        row["work"]["collection_physical_steps"]
        == row["work"]["collection_physical_attempts"]
        == 96
    )
    assert row["work"]["collection_unroll_steps"] == 96
    assert row["work"]["collection_episodes"] == len(row["training"])
    assert row["work"]["classification_unroll_steps"] == 320
    assert row["work"]["observed_checkpoint_training_unroll_steps"] == 0
    assert pickle.dumps(np.random.get_state()) == global_rng


@pytest.mark.parametrize("mutation", ("source", "seed", "prior_collection"))
def test_should_refuse_changed_frozen_state_or_preceding_evidence_and_keep_physical_charges(
    monkeypatch, mutation
):
    evidence = environment(monkeypatch)
    original = Session.predict
    changed: list[bool] = []

    def predict(self, features):
        result = original(self, features)
        active = evidence["active"]
        if not changed and (mutation != "prior_collection" or active["index"] > 0):
            if mutation == "source":
                active["source"].state.parameters["bias_hidden"][0, 0] = 1
            elif mutation == "seed":
                active["row"]["environment_seeds"]["training"] = 18
            else:
                active["row"]["training"][0]["collection"]["total_return"] = 77.0
            changed.append(True)
        return result

    monkeypatch.setattr(Session, "predict", predict)
    report = run(**SEEDS)
    row = report["runs"][0]
    assert changed and report["status"] == "incomplete" and row["status"] == "failed"
    assert Learner.updates == [] and evidence["budgets"][0].updates_completed == 0
    assert not evidence["finals"] and not Learner.classifications
    assert sum(quota["attempted_steps"] for quota in row["physical_quotas"]) > 0
    assert row["work"]["collection_physical_steps"] > 0
    if mutation == "prior_collection":
        assert row["training"][0]["collection_only"] is True
        assert row["training"][0]["receipt"] is None


def test_should_keep_failed_physical_attempt_paid_without_native_update_or_final_access(
    monkeypatch,
):
    evidence = environment(monkeypatch)

    def fail(self, action):
        raise RuntimeError("frozen physical fault")

    monkeypatch.setattr(KeyDoorNavigation, "step", fail)
    report = run(**SEEDS)
    row = report["runs"][0]
    assert report["status"] == "incomplete" and row["status"] == "failed"
    assert row["physical_quotas"][0] == dict(
        maximum_steps=12, attempted_steps=1, completed_steps=0, failed_steps=1
    )
    assert row["training"][0]["collection"]["environment_steps_attempted"] == 1
    assert row["training"][0]["collection"]["environment_steps"] == 0
    assert row["training"][0]["receipt"] is None
    assert evidence["budgets"][0].updates_completed == 0 and Learner.updates == []
    assert not evidence["finals"] and not Learner.classifications
    assert row["work"]["collection_physical_attempts"] == 1
    assert row["work"]["collection_physical_steps"] == 0


def test_should_preserve_two_returned_final_scores_when_the_third_predictor_call_fails(monkeypatch):
    evidence = environment(monkeypatch)
    original = Learner.predict
    attempts = []

    def predict(self, features):
        attempts.append(features.copy())
        if len(attempts) == 3:
            raise RuntimeError("third frozen final score failed")
        return original(self, features)

    monkeypatch.setattr(Learner, "predict", predict)
    report = run(**SEEDS)
    row = report["runs"][0]
    assert report["status"] == "incomplete" and row["status"] == "failed"
    assert evidence["physics"] == 96 and len(evidence["finals"]) == 8
    assert len(attempts) == 3 and len(Learner.classifications) == 2
    assert row["stream_result"]["score_forward_calls"] == 2
    assert row["work"]["classification_unroll_steps"] == 16
    assert row["work"]["collection_physical_steps"] == 96
    progress = row["physical_evaluation_progress"][0]["score_progress"]
    assert (
        progress["prediction_calls_attempted"] == 3 and progress["prediction_calls_completed"] == 2
    )
    assert (
        progress["prediction_rows_attempted"] == 24 and progress["prediction_rows_completed"] == 16
    )
    assert all(point["status"] == "not_run" for point in row["physical_evaluation_progress"][1:])
    assert row["live_sessions_started"] == 0
    assert evidence["budgets"][0].updates_completed == 0 and Learner.updates == []


@pytest.mark.parametrize("describe", (False, True))
@pytest.mark.parametrize("seeded", (False, True))
def test_should_route_frozen_cli_to_same_pilot_with_explicit_mode_and_optional_pair(
    monkeypatch, capsys, describe, seeded
):
    calls = []

    def selected(**options):
        calls.append(options)
        return dict(status="complete")

    monkeypatch.setattr(cli, "run_legal_reward_pilot", selected)
    monkeypatch.setattr(cli, "describe_legal_reward_protocol", selected)
    flags = ["--environment-seed", "17", "--heldout-environment-seed", "6"] if seeded else []
    monkeypatch.setattr(
        sys,
        "argv",
        ["run", "--scenario", "torch-key-door-frozen-interaction-quota", *flags]
        + (["--describe"] if describe else []),
    )
    cli.main()
    expected: dict[str, Any] = dict(**ROUTE, frozen_recurrent_control=True)
    if seeded:
        expected.update(SEEDS)
    assert calls == [expected]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "flags",
    (
        ["--environment-seed", "17"],
        ["--heldout-environment-seed", "6"],
        ["--environment-seed", "6", "--heldout-environment-seed", "6"],
    ),
)
def test_should_refuse_invalid_frozen_seed_cli_before_recipe_invocation(monkeypatch, flags):
    monkeypatch.setattr(cli, "run_legal_reward_pilot", forbidden)
    monkeypatch.setattr(cli, "describe_legal_reward_protocol", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["run", "--scenario", "torch-key-door-frozen-interaction-quota", *flags]
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
