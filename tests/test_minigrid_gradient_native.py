"""One gated actual TRAIN checkpoint composition, not a learning comparison."""

from copy import deepcopy
import json
import pickle
import random
from time import monotonic
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_learnability as pilot
from src.adapters import minigrid_learning_pilot as shared
from src.adapters import torch_recurrent_learner as native
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.shared.torch_runtime import require_torch
from test_torch_feature_learner import assert_tree


def test_should_read_three_original_train_pairs_after_four_fixed_native_updates(
    monkeypatch, record_property
):
    torch = require_torch()
    costs: dict[str, Any] = dict(
        optimizer_steps=0,
        training_calls_attempted=0,
        training_rows_attempted=0,
        training_rows=0,
        source_graphs=0,
        diagnostic_pairs=0,
        raw_score_forwards=0,
        raw_score_rows=0,
        gradient_forwards=0,
        gradient_forward_rows=0,
        gradient_backwards=0,
        gradient_backward_rows=0,
        reading=False,
        last_rows=0,
    )
    original_step, original_forward, original_grad = (
        torch.optim.SGD.step,
        native._forward,
        torch.autograd.grad,
    )

    def step(owner, *args, **kwargs):
        costs["optimizer_steps"] += 1
        assert not costs["reading"], "readout must never apply an SGD step"
        return original_step(owner, *args, **kwargs)

    def forward(parameters, values, hidden, **kwargs):
        if costs["reading"]:
            costs["last_rows"] = len(values)
            if torch.is_grad_enabled():
                costs["gradient_forwards"] += 1
                costs["gradient_forward_rows"] += len(values)
            else:
                costs["raw_score_forwards"] += 1
                costs["raw_score_rows"] += len(values)
        return original_forward(parameters, values, hidden, **kwargs)

    def grad(*args, **kwargs):
        if costs["reading"]:
            costs["gradient_backwards"] += 1
            costs["gradient_backward_rows"] += costs["last_rows"]
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.optim.SGD, "step", step)
    monkeypatch.setattr(native, "_forward", forward)
    monkeypatch.setattr(torch.autograd, "grad", grad)
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=4, max_wall_seconds=30), monotonic
    )
    reports: list[dict[str, Any]] = []
    result = None
    try:
        sources = shared._collect_sources(budget.before_final, reports)
        source_bytes = pickle.dumps(sources)
        binding = shared._signature(sources)
        source = native.TorchRecurrentLearner(native.TorchRecurrentPolicy(148, 8, 7, 17))
        costs["source_graphs"] += 1
        for index in range(4):
            budget.before_update()
            costs["training_calls_attempted"] += 1
            costs["training_rows_attempted"] += 9
            source.train_episode(shared._episode(sources[index % 2]))
            costs["training_rows"] += 9
            budget.record_update()
        state = source.snapshot_state()
        before, source_before = deepcopy(state), source.snapshot_state()
        rng: Any = torch.get_rng_state().clone(), deepcopy(np.random.get_state()), random.getstate()
        started_at, original_budget = budget.started_at, budget.budget

        def guard():
            budget.before_final()
            assert shared._check_signature(sources, binding)
            assert budget.started_at == started_at and budget.budget is original_budget
            assert budget.updates_completed == 4

        costs["reading"] = True
        result = pilot._score_checkpoint(
            source,
            state,
            method="recurrent_bc",
            updates=4,
            sources=sources,
            budget=budget,
            check_boundary=guard,
            gradient_geometry=True,
        )
        costs["reading"] = False
        assert result["status"] == "complete", result["reason"]
        assert result["source_unchanged"] is result["state_unchanged"] is True
        assert len(result["scores"]) == 6
        pairs = result["cue_gradient_geometry"]
        assert [entry["context"] for entry in pairs] == ["A", "B", "C_train"]
        costs["diagnostic_pairs"] = len(pairs)
        for index, entry in enumerate(pairs):
            diagnostic = entry["diagnostic"]
            assert diagnostic["status"] == "complete"
            assert diagnostic["objective"] == native.TORCH_RECURRENT_CE_ID
            assert [loss["episode_id"] for loss in diagnostic["losses"]] == [
                shared._episode(packet).episode_id for packet in sources[index * 2 : index * 2 + 2]
            ]
            for loss, score in zip(
                diagnostic["losses"], result["scores"][index * 2 : index * 2 + 2], strict=True
            ):
                assert loss["mean_ce"] == pytest.approx(score["ce"], rel=2e-6, abs=2e-6)
        work = result["work"]["cue_gradient_geometry"]
        assert work["pair_calls_attempted"] == work["pair_calls_completed"] == 3
        assert work["forward_rows_completed"] == work["backward_rows_completed"] == 54
        assert work["train_rows_admitted"] == 54
        assert work["input_array_equivalent_bytes"] == 64368
        assert work["geometry_reductions_completed"] == 18
        assert work["optimizer_steps"] == work["published_updates"] == 0
        assert work["failure_work_may_be_unobserved"] is False
        assert_tree(torch, state, before)
        assert_tree(torch, source.snapshot_state(), source_before)
        assert pickle.dumps(sources) == source_bytes
        assert torch.equal(torch.get_rng_state(), rng[0])
        now: Any = np.random.get_state()
        assert now[0] == rng[1][0] and now[2:] == rng[1][2:]
        np.testing.assert_array_equal(now[1], rng[1][1])
        assert random.getstate() == rng[2]
        assert budget.updates_completed == source.snapshot_state().completed_updates == 4
        assert costs["optimizer_steps"] == 4 and costs["training_rows"] == 36
        assert (
            costs["raw_score_forwards"]
            == costs["gradient_forwards"]
            == costs["gradient_backwards"]
            == 6
        )
        assert (
            costs["raw_score_rows"]
            == costs["gradient_forward_rows"]
            == costs["gradient_backward_rows"]
            == 54
        )
        json.dumps(result, allow_nan=False)
    finally:
        for key, value in costs.items():
            if type(value) is int and key != "last_rows":
                record_property(key, value)
        record_property("teacher_episodes", len(reports))
        record_property(
            "teacher_environment_steps", sum(row.get("environment_steps", 0) for row in reports)
        )
        record_property("teacher_reports", json.dumps(reports, allow_nan=False))
        if result is not None:
            record_property("checkpoint_report", json.dumps(result, allow_nan=False))
