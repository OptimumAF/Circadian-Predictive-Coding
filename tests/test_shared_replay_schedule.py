"""The shared schedule matches bounded unprioritized circadian selection."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
    ReplayRetentionBudget,
    replay_sample_id,
)
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.shared_replay_schedule import SharedReplayBuffer


@pytest.mark.parametrize(
    "policy",
    [ReplayRetentionPolicy("recent_fifo"), ReplayRetentionPolicy("seeded_reservoir", 53)],
)
def test_should_match_retained_and_selected_rows_under_equal_caps(
    policy: ReplayRetentionPolicy,
) -> None:
    budget = ReplayRetentionBudget(4, 96)
    shared = SharedReplayBuffer(2, budget, policy)
    model = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=4,
        seed=41,
        circadian_config=CircadianConfig(
            replay_memory_size=1,
            replay_steps=2,
            replay_prioritized=False,
            replay_class_balanced=False,
        ),
    )
    model.configure_replay_retention(budget, policy=policy)
    phases = (
        (np.arange(8, dtype=np.float64).reshape(4, 2) / 10.0, np.zeros((4, 1))),
        (np.arange(8, 16, dtype=np.float64).reshape(4, 2) / 10.0, np.ones((4, 1))),
    )
    for inputs, targets in phases:
        shared.observe_train_batch(inputs, targets)
        model.train_epoch(inputs, targets, 0.02, 2, 0.1)
        assert shared.retention == model.get_replay_retention()
        selected = shared.select_recent(2)
        actual_ids = tuple(
            replay_sample_id(item.input_batch, item.target_batch)
            for item in model._select_replay_snapshots(2)
        )
        assert selected.sample_ids == actual_ids
        assert selected.sample_ids == shared.retained_order_ids[-2:]
    repeated = next(
        item
        for item in model._replay_memory
        if replay_sample_id(item.input_batch, item.target_batch) in shared.retention.sample_ids
    )
    shared.observe_train_batch(repeated.input_batch, repeated.target_batch)
    model.train_epoch(repeated.input_batch, repeated.target_batch, 0.02, 2, 0.1)
    assert shared.retention == model.get_replay_retention()
    assert shared.select_recent(2).sample_ids == tuple(
        replay_sample_id(item.input_batch, item.target_batch)
        for item in model._select_replay_snapshots(2)
    )
    assert shared.retention.example_count == 4
    assert shared.retention.retained_bytes == 96
    if policy.name == "recent_fifo":
        assert set(shared.retention.sample_ids) == {
            replay_sample_id(phases[1][0][index : index + 1], phases[1][1][index : index + 1])
            for index in range(4)
        }
    else:
        a_ids = {
            replay_sample_id(phases[0][0][index : index + 1], phases[0][1][index : index + 1])
            for index in range(4)
        }
        b_ids = {
            replay_sample_id(phases[1][0][index : index + 1], phases[1][1][index : index + 1])
            for index in range(4)
        }
        assert len(set(shared.retention.sample_ids) & a_ids) == 2
        assert len(set(shared.retention.sample_ids) & b_ids) == 2


def test_should_reject_bad_budget_or_batch_before_mutating_retention() -> None:
    with pytest.raises(ValueError, match="cannot hold one"):
        SharedReplayBuffer(2, ReplayRetentionBudget(4, 16), ReplayRetentionPolicy("recent_fifo"))
    shared = SharedReplayBuffer(
        2, ReplayRetentionBudget(4, 96), ReplayRetentionPolicy("recent_fifo")
    )
    before = shared.retention
    with pytest.raises(ValueError, match="targets must be between"):
        shared.observe_train_batch(np.zeros((2, 2)), np.asarray([[0.0], [2.0]]))
    assert shared.retention == before


@pytest.mark.parametrize("budget", [ReplayRetentionBudget(2, 96), ReplayRetentionBudget(10, 48)])
def test_should_enforce_count_and_copied_array_byte_caps(budget: ReplayRetentionBudget) -> None:
    shared = SharedReplayBuffer(2, budget, ReplayRetentionPolicy("recent_fifo"))
    inputs = np.arange(8, dtype=np.float64).reshape(4, 2)
    targets = np.zeros((4, 1), dtype=np.float64)
    shared.observe_train_batch(inputs, targets)
    assert shared.retention.example_count == 2
    assert shared.retention.retained_bytes == 48
    assert shared.select_recent(2).sample_ids == tuple(
        replay_sample_id(inputs[index : index + 1], targets[index : index + 1]) for index in (2, 3)
    )
