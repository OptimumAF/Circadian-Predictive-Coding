"""Actual policy sampling and owned reward updates stay outside the serving actor."""

from copy import deepcopy
from dataclasses import asdict
import json
import pickle

import numpy as np
import pytest

from src.adapters.torch_recurrent_learner import TorchRecurrentLearner, TorchRecurrentPolicy
from src.app.actor_shadow import StableActor
from src.app.key_door_training import collect_key_door_reward_episode
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.key_door_navigation import KeyDoorLayout
from src.shared.torch_runtime import require_torch
from test_torch_feature_learner import assert_tree


def test_should_collect_real_policy_actions_update_rewards_and_continue_owned_state(tmp_path):
    torch = require_torch()
    torch_before = torch.get_rng_state().clone()
    numpy_before = pickle.dumps(np.random.get_state())
    source = TorchRecurrentLearner(TorchRecurrentPolicy(8, 4, 5, 17))
    initial = source.snapshot_state()
    actor = StableActor(source, version="before-reward-updates")
    candidate = source.fork()
    sampler = np.random.Generator(np.random.PCG64(29))
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    layout = KeyDoorLayout("reward-train-A")

    def collect(owner, rng, identity):
        return collect_key_door_reward_episode(
            owner,
            layout,
            1,
            episode_id=identity,
            sampler=rng,
            before_prediction=budget.before_final,
        )

    first = collect(candidate, sampler, "policy-first")
    assert first["status"] == "complete" and first["episode"].role == "train"
    assert_tree(torch, initial, candidate.snapshot_state())
    budget.before_update()
    diagnostic = candidate.train_reward_episode(first["episode"], first["behavior_state"])
    budget.record_update()
    saved = candidate.snapshot_state()
    rng_saved = deepcopy(sampler.bit_generator.state)
    resumed, rng_resumed = candidate.fork(), np.random.Generator(np.random.PCG64(0))
    rng_resumed.bit_generator.state = deepcopy(rng_saved)
    second = collect(candidate, sampler, "policy-next")
    next_resumed = collect(resumed, rng_resumed, "policy-next")
    assert second["episode"] == next_resumed["episode"]
    assert second["sampler_before"] == next_resumed["sampler_before"] == rng_saved
    assert second["sampler_after"] == next_resumed["sampler_after"]
    assert_tree(torch, second["behavior_state"], next_resumed["behavior_state"])
    budget.before_update()
    candidate.train_reward_episode(second["episode"], second["behavior_state"])
    budget.record_update()
    branch_budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    branch_budget.before_update()
    resumed.train_reward_episode(next_resumed["episode"], next_resumed["behavior_state"])
    branch_budget.record_update()
    assert_tree(torch, candidate.snapshot_state(), resumed.snapshot_state())
    assert_tree(torch, initial, source.snapshot_state())
    assert_tree(torch, initial, actor.snapshot_state().state)
    assert_tree(torch, saved, second["behavior_state"])
    assert budget.updates_completed == 2 and branch_budget.updates_completed == 1
    assert candidate.completed_updates == 2
    assert candidate.training_examples == sum(
        len(x["episode"].transitions) for x in (first, second)
    )
    assert torch.equal(torch_before, torch.get_rng_state())
    assert pickle.dumps(np.random.get_state()) == numpy_before
    for report in (first, second, next_resumed):
        episode = report["episode"]
        assert report["prediction_calls_attempted"] == report["prediction_calls_returned"]
        assert report["sampling_draws"] == len(episode.transitions)
        assert report["environment_steps_attempted"] == report["environment_steps"]
        assert report["total_return"] == sum(t.reward for t in episode.transitions)
    evidence = dict(
        classification="native engineering integration, no learning-quality gate or comparison",
        episodes=[asdict(x["episode"]) for x in (first, second, next_resumed)],
        counters=[
            {
                k: x[k]
                for k in (
                    "prediction_calls_attempted",
                    "prediction_calls_returned",
                    "sampling_draws",
                    "environment_steps_attempted",
                    "environment_steps",
                    "total_return",
                    "sampler_before",
                    "sampler_after",
                )
            }
            for x in (first, second, next_resumed)
        ],
        diagnostic=asdict(diagnostic),
        original_budget_charges=2,
        branch_budget_charges=1,
        source_actor_global_rng_unchanged=True,
        full_optimizer_continuation_exact=True,
    )
    (tmp_path / "reward-engineering.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf8"
    )


def test_should_refuse_source_changed_during_real_collection_before_any_reward_update():
    source = TorchRecurrentLearner(TorchRecurrentPolicy(8, 4, 5, 17))
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)

    def change_source():
        budget.before_update()
        source.train_batch(np.zeros((1, 8)), np.zeros((1, 1)))
        budget.record_update()

    report = collect_key_door_reward_episode(
        source,
        KeyDoorLayout("source-fault"),
        1,
        episode_id="stale-policy",
        sampler=np.random.Generator(np.random.PCG64(29)),
        max_steps=1,
        before_prediction=change_source,
    )
    assert report["behavior_state"].completed_updates == 0
    assert source.completed_updates == budget.updates_completed == 1
    with pytest.raises(ValueError, match="stale|differs"):
        source.train_reward_episode(report["episode"], report["behavior_state"])
    assert source.completed_updates == budget.updates_completed == 1
