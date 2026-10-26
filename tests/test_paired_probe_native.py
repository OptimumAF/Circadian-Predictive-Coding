"""Four prospective native cases: eight physical SGD steps / 34 training rows.

Warm steps, direct continuation controls and failed unpublished steps are all
charged explicitly. No environment, old native group or acquisition run occurs.
"""

from copy import deepcopy
import json
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import torch_paired_probe as adapter
from src.adapters import torch_recurrent_episode_batch as native
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
from src.app.coupled_attribution import AttributionBatch
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    episode_arrays,
)
from src.shared.torch_runtime import require_torch
from test_torch_feature_learner import assert_tree


@pytest.fixture
def cost(monkeypatch, record_property):
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    observed = SimpleNamespace(
        torch=torch,
        steps=0,
        rows=0,
        active_rows=None,
        expected=(0, 0),
        probe_calls=[],
        failed_native_work=None,
    )
    step = torch.optim.SGD.step
    train = native.TorchRecurrentEpisodeBatchLearner.train_probe_episode_pair

    def measured_step(owner, *args, **kwargs):
        assert type(observed.active_rows) is int, "unbudgeted physical optimizer step"
        observed.steps += 1
        observed.rows += observed.active_rows
        return step(owner, *args, **kwargs)

    def measured_probe(owner, episodes):
        observed.probe_calls.append(deepcopy(episodes))
        return train(owner, episodes)

    monkeypatch.setattr(torch.optim.SGD, "step", measured_step)
    monkeypatch.setattr(
        native.TorchRecurrentEpisodeBatchLearner, "train_probe_episode_pair", measured_probe
    )
    yield observed
    record_property("physical_sgd_steps", observed.steps)
    record_property("physical_training_rows", observed.rows)
    if observed.failed_native_work is not None:
        record_property(
            "failed_native_work", json.dumps(observed.failed_native_work, allow_nan=False)
        )
    assert (observed.steps, observed.rows) == observed.expected


def paid(cost, rows, operation):
    assert cost.active_rows is None
    cost.active_rows = rows
    try:
        return operation()
    finally:
        cost.active_rows = None


def demonstration(name, labels, *, role="probe_train", context="E-blue"):
    observations = tuple(
        (float(index) / 10, 0.3 if index % 2 else -0.3) for index in range(len(labels) + 1)
    )
    return DemonstrationEpisode(
        name,
        context,
        role,
        tuple(
            DemonstrationTransition(
                index,
                observations[index],
                label,
                0.0,
                observations[index + 1],
                index == len(labels) - 1,
                3,
            )
            for index, label in enumerate(labels)
        ),
        3,
    )


def probe_episodes():
    return (
        demonstration("probe-blue-0", (0,)),
        demonstration("probe-blue-1", (1, 2)),
        demonstration("probe-red-0", (2, 1), context="E-red"),
        demonstration("probe-red-1", (0,), context="E-red"),
    )


def pair_arrays(episodes):
    arrays = [episode_arrays(episode) for episode in episodes]
    return np.concatenate([item[0] for item in arrays]), np.concatenate(
        [item[1] for item in arrays]
    )


def probe_data(episodes):
    training = []
    for index in range(len(episodes) // 2):
        pair = episodes[2 * index : 2 * index + 2]
        features, targets = pair_arrays(pair)
        training.append(
            AttributionBatch(
                "probe_train",
                pair[0].context,
                tuple(f"probe-update-{index}-row-{row}" for row in range(len(features))),
                features,
                targets,
                None,
                3,
            )
        )
    evaluation = []
    for context in ("E-blue", "E-red"):
        features, targets = episode_arrays(
            demonstration(f"test-{context}", (0, 2), role="probe_test", context=context)
        )
        evaluation.append(
            AttributionBatch(
                "probe_test",
                context,
                tuple(f"test-{context}-row-{row}" for row in range(2)),
                features,
                targets,
                None,
                3,
            )
        )
    return tuple(training), tuple(evaluation)


def warm(cost):
    source = native.TorchRecurrentEpisodeBatchLearner(TorchRecurrentPolicy(2, 3, 3, 17))
    episodes = (
        demonstration("old-blue", (0, 1), role="train", context="A"),
        demonstration("old-red", (2, 2, 1), role="train", context="A"),
    )
    paid(cost, 5, lambda: source.train_episode_pair(episodes))
    return source, source.snapshot_state()


def test_should_preserve_original_probe_roles_and_full_warm_native_continuation(cost):
    cost.expected = (3, 15)
    source, state = warm(cost)
    episodes = (
        demonstration("probe-blue", (0, 1)),
        demonstration("probe-red", (2, 2, 1), context="E-red"),
    )
    immutable = pickle.dumps(episodes)
    before = deepcopy(state)
    direct = source.fork()
    probe = adapter.TorchPairedProbeLearner(state, episodes)
    rng, numpy_rng = cost.torch.get_rng_state().clone(), pickle.dumps(np.random.get_state())
    expected = paid(cost, 5, lambda: direct.train_probe_episode_pair(episodes))
    actual = paid(cost, 5, lambda: probe.train_batch(*pair_arrays(episodes)))
    assert actual.definition == expected.definition == native.TORCH_RECURRENT_EPISODE_PAIR_CE_ID
    assert actual.value == expected.value
    assert_tree(cost.torch, probe.snapshot_state(), direct.snapshot_state())
    assert (
        probe.snapshot_state().completed_updates,
        probe.snapshot_state().training_episodes,
        probe.snapshot_state().training_examples,
    ) == (2, 4, 10)
    assert_tree(cost.torch, state, before)
    assert_tree(cost.torch, source.snapshot_state(), before)
    assert all(value is not None for value in state.gradients.values())
    assert any(
        bool(cost.torch.count_nonzero(value["momentum_buffer"]).item())
        for value in state.optimizer_state["state"].values()
    )
    assert pickle.dumps(episodes) == immutable
    assert all(
        tuple(episode.role for episode in call) == ("probe_train", "probe_train")
        for call in cost.probe_calls
    )
    assert (
        cost.torch.equal(cost.torch.get_rng_state(), rng)
        and pickle.dumps(np.random.get_state()) == numpy_rng
    )


def test_should_run_existing_probe_with_local_budget_and_absolute_warm_pair_clocks(
    monkeypatch, cost
):
    cost.expected = (3, 11)
    _, state = warm(cost)
    episodes = probe_episodes()
    source = adapter.TorchPairedProbeLearner(state, episodes)
    before = source.snapshot_state()
    training, evaluation = probe_data(episodes)
    original_inputs = pickle.dumps((state, episodes, training, evaluation))
    children: list[Any] = []
    fork = adapter.TorchPairedProbeLearner.fork

    def capture(owner):
        child = fork(owner)
        children.append(child)
        return child

    monkeypatch.setattr(adapter.TorchPairedProbeLearner, "fork", capture)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 10.0)
    report = paid(
        cost,
        3,
        lambda: run_fresh_task_probe(
            source,
            state,
            training,
            evaluation,
            budget=budget,
            checkpoint_updates=(0, 1, 2),
            timer=lambda: 10.0,
            decision_diagnostics=True,
        ),
    )
    assert report["status"] == "complete", report["reason"]
    assert report["completed_updates"] == budget.updates_completed == 2
    assert report["completed_training_examples"] == 6
    assert [point["updates"] for point in report["checkpoints"]] == [0, 1, 2]
    assert report["score_forward_calls_completed"] == 6
    assert "episode_scope" not in report and "failed_native_work" not in report
    assert len(children) == 1
    end = children[0].snapshot_state()
    assert (end.completed_updates, end.training_episodes, end.training_examples) == (3, 6, 11)
    assert_tree(cost.torch, source.snapshot_state(), before)
    assert pickle.dumps((state, episodes, training, evaluation)) == original_inputs
    assert report["reserved_data_bytes"] == sum(
        packet.features.nbytes + packet.targets.nbytes for packet in (*training, *evaluation)
    )
    assert all(
        diagnostic["definition"] == native.TORCH_RECURRENT_EPISODE_PAIR_CE_ID
        for diagnostic in report["diagnostics"]
    )
    assert all(item.episode is None for item in (*training, *evaluation))
    assert all(episode.role == "probe_train" for call in cost.probe_calls for episode in call)
    for point in report["checkpoints"]:
        assert set(point["scores"]) == {"E-blue", "E-red"}
        assert all(
            detail["sequence_kind"] == "batch_row_order"
            for detail in point["decision_diagnostics"].values()
        )


def test_should_keep_train_entrypoint_train_only_and_probe_entrypoint_probe_train_only(cost):
    source = native.TorchRecurrentEpisodeBatchLearner(TorchRecurrentPolicy(2, 3, 3, 17))
    before = source.snapshot_state()
    for role in ("train", "final_test", "probe_test"):
        episodes = (
            demonstration("first", (0,), role=role),
            demonstration("second", (1,), role=role),
        )
        with pytest.raises(ValueError):
            source.train_probe_episode_pair(episodes)
    with pytest.raises(ValueError):
        source.train_episode_pair(probe_episodes()[:2])
    assert_tree(cost.torch, source.snapshot_state(), before)


def test_should_preserve_paid_unpublished_native_step_in_generic_probe_failure(monkeypatch, cost):
    cost.expected = (2, 8)
    _, state = warm(cost)
    episodes = probe_episodes()[:2]
    source = adapter.TorchPairedProbeLearner(state, episodes)
    before = source.snapshot_state()
    training, evaluation = probe_data(episodes)
    capture = native._capture

    def refuse_after_step(parameters, optimizer, policy, updates, exposed_episodes, examples):
        if updates == 2 and cost.steps == 2:
            raise FloatingPointError("paid probe finite capture refused")
        return capture(parameters, optimizer, policy, updates, exposed_episodes, examples)

    monkeypatch.setattr(native, "_capture", refuse_after_step)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 10.0)
    report = paid(
        cost,
        3,
        lambda: run_fresh_task_probe(
            source,
            state,
            training,
            evaluation,
            budget=budget,
            checkpoint_updates=(0, 1),
            timer=lambda: 10.0,
        ),
    )
    assert report["status"] == "failed" and "paid probe finite capture refused" in report["reason"]
    assert (
        report["attempted_updates"] == 1
        and report["completed_updates"] == budget.updates_completed == 0
    )
    assert report["attempted_training_examples"] == 3 and report["completed_training_examples"] == 0
    assert [point["updates"] for point in report["checkpoints"]] == [0]
    failed = report["failed_native_work"]
    cost.failed_native_work = failed
    assert failed["definition"] == native.TORCH_RECURRENT_EPISODE_PAIR_CE_ID
    assert failed["work"]["optimizer_steps"] == 1 and failed["work"]["published_updates"] == 0
    assert failed["work"]["failure_work_may_be_unobserved"] is True
    assert_tree(cost.torch, source.snapshot_state(), before)
