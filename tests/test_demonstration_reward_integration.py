"""Train-only native phase routing and owned continuation through the real runtime."""

from copy import deepcopy
from dataclasses import asdict
import json
import pickle
from typing import Any

import numpy as np

from src.adapters import key_door_reward_pilot as pilot
from src.adapters import torch_recurrent_learner as native, torch_recurrent_reward as reward_native
from src.adapters.torch_demonstration_reward_learner import TorchDemonstrationRewardLearner
from src.adapters.torch_recurrent_learner import TORCH_RECURRENT_CE_ID, TorchRecurrentPolicy
from src.adapters.torch_reward_update_learner import RewardUpdate
from src.app.actor_shadow import ActorShadowRuntime
from src.app.key_door_training import collect_key_door_reward_episode
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import LogicalClock
from src.core.key_door_navigation import KeyDoorNavigation
from src.shared.process_memory import ProcessRssSampler
from src.shared.torch_runtime import require_torch
from test_torch_feature_learner import assert_tree, tensors


def _engineering_counters(monkeypatch, torch):
    caps = dict(sgd=12, train_rows=128, unroll=256, predict=128, environment_step=200)
    counts = dict.fromkeys(caps, 0)

    def charge(key, amount=1):
        if counts[key] + amount > caps[key]:
            raise AssertionError(f"{key} engineering allowance exhausted before native entry")
        counts[key] += amount

    def measured(owner, name, key):
        original = getattr(owner, name)

        def call(*args, **kwargs):
            charge(key)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, call)

    original_forward = native._forward

    def forward(parameters, features, hidden):
        charge("unroll", len(features))
        if torch.is_grad_enabled():
            charge("train_rows", len(features))
        return original_forward(parameters, features, hidden)

    monkeypatch.setattr(native, "_forward", forward)
    monkeypatch.setattr(reward_native, "_forward", forward)
    measured(torch.optim.SGD, "step", "sgd")
    measured(native.TorchRecurrentLearner, "predict", "predict")
    measured(native.RecurrentEpisodePredictor, "predict", "predict")
    measured(KeyDoorNavigation, "step", "environment_step")
    return caps, counts


def test_should_continue_native_optimizer_across_train_only_phases_and_owned_resume(
    tmp_path, monkeypatch
):
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    caps, counts = _engineering_counters(monkeypatch, torch)
    torch_before = torch.get_rng_state().clone()
    numpy_before = pickle.dumps(np.random.get_state())
    source = TorchDemonstrationRewardLearner(TorchRecurrentPolicy(8, 8, 5, 17))
    initial = source.snapshot_state()
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=16), lambda: 0.0)
    birth = pilot._binding(budget)
    runtime: Any = ActorShadowRuntime(
        source,
        actor_version="a0",
        candidate_version="c0",
        clock=clock,
        budget=budget,
        max_experiences=16,
        max_consolidations=0,
    )
    rng = np.random.Generator(np.random.PCG64(29))
    row: dict[str, Any] = dict(
        training=pilot._training_slots(True),
        collection_attempts=0,
        demonstration_collection_attempts=0,
    )
    with ProcessRssSampler() as memory:
        previous = runtime.candidate_snapshot().state
        for i in range(8):
            previous = pilot._train_one(
                row, source, runtime, previous, rng, budget, birth, i, clock
            )
        warm = deepcopy(previous)
        assert warm.completed_updates == budget.updates_completed == 8
        assert warm.training_examples == 64 and row["collection_attempts"] == 0
        assert rng.bit_generator.state == np.random.PCG64(29).state
        assert any(bool(t.abs().sum().item()) for t in tensors(torch, warm.optimizer_state))
        assert any(bool(t.abs().sum().item()) for t in tensors(torch, warm.gradients))
        assert not pilot._same_state(initial.parameters, warm.parameters)
        assert all(
            s["receipt"]["diagnostic"]["definition"] == TORCH_RECURRENT_CE_ID
            for s in row["training"][:8]
        )
        previous = pilot._train_one(row, source, runtime, previous, rng, budget, birth, 8, clock)
        saved, rng_saved = deepcopy(previous), deepcopy(rng.bit_generator.state)
        branch, branch_rng = source.fork(), np.random.Generator(np.random.PCG64(0))
        branch.restore_state(deepcopy(saved))
        branch_rng.bit_generator.state = deepcopy(rng_saved)
        branch_budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
        report = collect_key_door_reward_episode(
            branch,
            pilot.shared.layouts()[0],
            -1,
            episode_id="reward-train-1",
            sampler=branch_rng,
            before_prediction=branch_budget.before_final,
        )
        previous = pilot._train_one(row, source, runtime, previous, rng, budget, birth, 9, clock)
        recorded = row["training"][9]["collection"]
        assert recorded["episode"] == asdict(report["episode"])
        assert recorded["sampler_before"] == rng_saved == report["sampler_before"]
        assert (
            recorded["sampler_after"] == report["sampler_after"] == branch_rng.bit_generator.state
        )
        assert_tree(torch, saved, report["behavior_state"])
        branch_budget.before_update()
        branch.train_batch(
            RewardUpdate(report["episode"], report["behavior_state"]), "reward-train-1"
        )
        branch_budget.record_update()
        assert_tree(torch, previous, branch.snapshot_state())
        assert_tree(torch, initial, source.snapshot_state())
        assert_tree(torch, initial, runtime.actor.snapshot_state().state)
        assert budget.updates_completed == previous.completed_updates == 10
        assert branch_budget.updates_completed == 1
        assert previous.training_examples == 64 + sum(
            len(s["collection"]["episode"]["transitions"]) for s in row["training"][8:10]
        )
        assert all(
            s["status"] == "complete" and s["commit_validated"] for s in row["training"][:10]
        )
        assert all(s["status"] == "not_run" for s in row["training"][10:])
        assert torch.equal(torch_before, torch.get_rng_state())
        assert pickle.dumps(np.random.get_state()) == numpy_before
    rss = asdict(memory.snapshot())
    assert rss["peak_bytes"] <= 768 * 1024**2
    assert counts["sgd"] == 11
    assert all(counts[key] <= value for key, value in caps.items())
    evidence = dict(
        classification="train-only native engineering; no final evaluation or skill comparison",
        original_budget_charges=10,
        original_budget_limit=16,
        branch_budget_charges=1,
        demonstration_updates=8,
        demonstration_rows=64,
        reward_updates=3,
        full_optimizer_continuation_exact=True,
        source_actor_global_rng_unchanged=True,
        observations=[
            pilot.shared._recurrent_observation(p.completed_updates, p)
            for p in (initial, warm, saved, previous)
        ],
        training=row["training"],
        branch_collection=pilot._serialize_collection(report),
        sampled_process_rss=rss,
        native_caps=caps,
        native_counts=counts,
    )
    (tmp_path / "phase-engineering.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf8"
    )
