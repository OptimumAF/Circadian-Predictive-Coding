"""Actual complete recurrent probe episodes stay outside serving and source state."""

import pickle

import numpy as np
import pytest

from src.adapters.key_door_pilot import _collect
from src.adapters.torch_recurrent_learner import TorchRecurrentLearner, TorchRecurrentPolicy
from src.app.actor_shadow import StableActor
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.key_door_navigation import KeyDoorLayout
from src.shared.torch_runtime import require_torch
from test_torch_feature_learner import assert_tree


def test_should_probe_complete_categorical_episodes_without_changing_source_or_serving_actor(
    monkeypatch,
):
    torch = require_torch()
    source = TorchRecurrentLearner(TorchRecurrentPolicy(8, 4, 5, 17))
    prior = _collect("probe-component-prior", KeyDoorLayout("A"), 1, "train")
    assert prior.episode is not None
    source.train_episode(prior.episode)
    state = source.snapshot_state()
    actor = StableActor(source, version="component-actor-before-probe")
    actor_before = actor.snapshot_state()
    layout = KeyDoorLayout("E", 3, 3, True, 0.75)
    training = tuple(
        _collect(f"reserved-train-{cue}", layout, cue, "probe_train") for cue in (1, -1)
    )
    evaluation = tuple(
        _collect(f"reserved-test-{cue}", layout, cue, "probe_test") for cue in (1, -1)
    )
    for packet in (training[0], evaluation[0]):
        assert packet.episode is not None
        with pytest.raises(ValueError):
            source.train_episode(packet.episode)
    data_before = pickle.dumps((training, evaluation))
    torch_before, numpy_before = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state())
    children = []
    original_fork = source.fork

    def fork():
        child = original_fork()
        children.append(child)
        return child

    monkeypatch.setattr(source, "fork", fork)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    report = run_fresh_task_probe(
        source,
        state,
        training,
        evaluation,
        budget=budget,
        checkpoint_updates=(0, 1, 2),
        timer=lambda: 0.0,
    )
    assert report["status"] == "complete" and report["num_classes"] == 5
    assert (
        report["completed_updates"] == report["attempted_updates"] == budget.updates_completed == 2
    )
    assert report["completed_training_examples"] == report["attempted_training_examples"] == 16
    assert report["score_forward_calls_completed"] == report["score_forward_calls_attempted"] == 6
    assert report["reserved_data_bytes"] == 2304
    assert report["episode_scope"]["training_episode_ids"] == [
        "reserved-train-1",
        "reserved-train--1",
    ]
    assert report["episode_scope"]["evaluation_episode_ids"] == [
        "reserved-test-1",
        "reserved-test--1",
    ]
    assert report["episode_scope"]["training_transitions"] == 16
    assert report["episode_scope"]["evaluation_transitions"] == 16
    assert [p["updates"] for p in report["checkpoints"]] == [0, 1, 2]
    for point in report["checkpoints"]:
        assert set(point["scores"]) == {"E-red", "E-blue"}
        for score in point["scores"].values():
            assert np.isfinite(score["ce"]) and 0 <= score["accuracy"] <= 1
            assert set(score) == {"ce", "accuracy"}
    assert len(children) == 1
    assert children[0] is not source and not actor._owns(children[0])
    child_state = children[0].snapshot_state()
    assert child_state.completed_updates == 3 and child_state.training_examples == 24
    assert_tree(torch, source.snapshot_state(), state)
    assert_tree(torch, actor.snapshot_state(), actor_before)
    assert pickle.dumps((training, evaluation)) == data_before
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert pickle.dumps(np.random.get_state()) == numpy_before
