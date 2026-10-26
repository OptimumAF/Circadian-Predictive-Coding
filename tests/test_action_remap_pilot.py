"""Mapped dynamics through the existing actor, reward and final-evaluation recipe."""

from copy import deepcopy
import json
import pickle
import sys

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.key_door_reward_pilot as pilot
from src.core.key_door_navigation import (
    KeyDoorNavigation,
    key_door_layout_record,
    scripted_actions,
)
from test_legal_reward_pilot import Learner, environment
from test_torch_feature_rehearsal_pilot import forbidden


def test_should_run_mapped_own_trajectories_and_final_layouts_after_all_commits(monkeypatch):
    final_collections, budgets = environment(monkeypatch)
    report = pilot.run_pilot(condition_on_legal_actions=True, action_remapping=True)
    assert report["status"] == "complete", report
    assert report["full_R43_complete"] is report["scientific_advantage_verified"] is False
    row = report["runs"][0]
    assert row["action_mapping"] == "movement_permutation_v1"
    assert len(Learner.updates) == budgets[0].updates_completed == 8
    assert len(final_collections) == len(Learner.classifications) // 5 == 8
    assert len(row["episodes"]) == 40
    layouts = {layout.layout_id: layout for layout in pilot.action_remapping_layouts()}
    rng = np.random.Generator(np.random.PCG64(29))
    rows = 0
    for index, slot in enumerate(row["training"]):
        layout = layouts[slot["context"]]
        game = KeyDoorNavigation(layout, slot["cue"])
        collected = slot["collection"]
        assert slot["layout"] == key_door_layout_record(layout)
        assert collected.get("layout") == (
            key_door_layout_record(layout) if slot["context"] == "B" else None
        )
        assert collected["behavior_native_updates"] == index
        assert collected["behavior_training_examples"] == rows
        assert slot["sampler_before"] == rng.bit_generator.state
        for step in collected["transitions"]:
            assert step["observation"] == game.observation
            assert step["legal_actions"] == game.legal_actions
            outcome = game.step(step["action"])
            assert step["next_observation"] == outcome.observation
            assert step["reward"] == outcome.reward
        count = len(collected["transitions"])
        assert collected["sampling_draws"] == collected["environment_steps"] == count
        rng.random(count)
        assert slot["sampler_after"] == rng.bit_generator.state
        assert slot["charged_updates_before"] == index
        assert slot["charged_updates_after"] == index + 1
        rows += count
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert all(session.inputs[0][0, 6] in (-1, 1) for session in Learner.sessions)
    assert len({id(session) for session in Learner.sessions}) == len(Learner.sessions)
    for args, _ in final_collections:
        assert args[1] == layouts[args[1].layout_id] and args[3] == "final_test"
    for slot in row["episodes"]:
        assert slot["layout"] == key_door_layout_record(layouts[slot["context"]])
        assert slot["learning_active"] is False


@pytest.mark.parametrize("fault", ["law", "layout", "collected_layout"])
def test_should_refuse_changed_mapping_before_more_training_and_keep_prior_charges(
    monkeypatch, fault
):
    collected, budgets = environment(monkeypatch)
    original = pilot._prepare_update

    def prepare(row, source, state, rng, budget, index, identity):
        if index == 2 and fault == "law":
            row.pop("action_mapping")
        if index == 2 and fault == "layout":
            row["training"][index]["layout"]["movement_action_map"] = (0, 1, 2, 3)
        if fault == "collected_layout" and index == 2:
            original_collect = pilot.collect_key_door_reward_episode

            def collect(*args, **kwargs):
                result = original_collect(*args, **kwargs)
                result["layout"]["movement_action_map"] = (0, 1, 2, 3)
                return result

            monkeypatch.setattr(pilot, "collect_key_door_reward_episode", collect)
        return original(row, source, state, rng, budget, index, identity)

    monkeypatch.setattr(pilot, "_prepare_update", prepare)
    report = pilot.run_pilot(condition_on_legal_actions=True, action_remapping=True)
    assert report["status"] == "incomplete"
    row = report["runs"][0]
    assert row["status"] == "failed" and row["training"][2]["status"] == "failed"
    assert budgets[0].updates_completed == len(Learner.updates) == 2
    assert not collected and not Learner.classifications
    assert all(slot["status"] == "not_run" for slot in row["training"][3:])
    assert row["training"][2]["charged_updates_after"] == 2
    if fault == "collected_layout":
        paid = row["training"][2]["collection"]
        assert paid["sampling_draws"] > 0 and paid["prediction_calls_returned"] > 0


def test_should_describe_different_controls_without_different_inputs_or_teacher_training(
    monkeypatch,
):
    for name in ("require_torch", "ProcessRssSampler", "TorchRewardUpdateLearner"):
        monkeypatch.setattr(pilot, name, forbidden)
    old = deepcopy(pilot.describe_protocol(condition_on_legal_actions=True))
    new = pilot.describe_protocol(condition_on_legal_actions=True, action_remapping=True)
    assert new["planned_work"] == old["planned_work"]
    assert new["protocol_id"] == "actual_key_door_legal_action_remapping_v1"
    assert new["sampling_seed"] == 29 and new["seed"] == 17
    a, b, c, _ = pilot.action_remapping_layouts()
    assert KeyDoorNavigation(a, 1).observation == KeyDoorNavigation(b, 1).observation
    assert KeyDoorNavigation(c, -1).observation == KeyDoorNavigation(b, -1).observation
    assert scripted_actions(a, 1) != scripted_actions(b, 1)
    assert "no learned remapping" in new["limitations"]
    assert old == pilot.describe_protocol(condition_on_legal_actions=True, action_remapping=False)


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", np.bool_(True)])
@pytest.mark.parametrize("entry", ["describe_protocol", "run_pilot"])
def test_should_refuse_invalid_mapping_switch_before_resource_work(monkeypatch, invalid, entry):
    monkeypatch.setattr(pilot, "_run_pilot", forbidden)
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    with pytest.raises(ValueError):
        getattr(pilot, entry)(condition_on_legal_actions=True, action_remapping=invalid)


def test_should_refuse_unconditioned_remap_before_training(monkeypatch):
    monkeypatch.setattr(pilot, "_run_pilot", forbidden)
    with pytest.raises(ValueError):
        pilot.run_pilot(action_remapping=True)


@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_mapped_scenario_through_existing_cli(monkeypatch, capsys, describe):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(cli, "describe_legal_reward_protocol", run)
    monkeypatch.setattr(cli, "run_legal_reward_pilot", run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run", "--scenario", "torch-key-door-legal-action-remap"]
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [dict(condition_on_legal_actions=True, action_remapping=True)]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


def test_should_keep_same_observation_but_different_own_policy_action_exposure(monkeypatch):
    # An ordinary fixed distribution observes its own mapped controls, never
    # a common teacher path substituted to manufacture identical trajectories.
    environment(monkeypatch)
    monkeypatch.setattr(Learner, "raw", (0.1, 0.2, 0.3, 0.15, 0.25))
    report = pilot.run_pilot(condition_on_legal_actions=True, action_remapping=True)
    assert report["status"] == "complete"
    row = report["runs"][0]
    first_a = row["training"][0]["collection"]["transitions"][0]
    first_b = row["training"][2]["collection"]["transitions"][0]
    assert first_a["observation"] == first_b["observation"]
    assert first_a["legal_actions"] != first_b["legal_actions"]
    assert pickle.dumps(first_a["probabilities"]) != pickle.dumps(first_b["probabilities"])
