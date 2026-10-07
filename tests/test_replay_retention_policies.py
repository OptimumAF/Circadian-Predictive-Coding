"""Equal-capacity FIFO and seeded reservoir retention on arrived wake rows."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
    ReplayRetentionBudget,
    ReplayRetentionSnapshot,
    replay_sample_id,
)
from src.core.replay_retention import ReplayRetentionPolicy


_BUDGET = ReplayRetentionBudget(max_examples=4, max_bytes=96)


def _model(
    policy: ReplayRetentionPolicy | None, budget: ReplayRetentionBudget = _BUDGET
) -> CircadianPredictiveCodingNetwork:
    model = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=4,
        seed=41,
        circadian_config=CircadianConfig(replay_memory_size=1, replay_steps=0),
    )
    model.configure_replay_retention(budget, policy=policy)
    return model


def _phase_rows() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    phase_a = np.arange(8, dtype=np.float64).reshape(4, 2) / 10.0
    phase_b = np.arange(8, 16, dtype=np.float64).reshape(4, 2) / 10.0
    return (phase_a, np.zeros((4, 1))), (phase_b, np.ones((4, 1)))


def _row_ids(inputs: np.ndarray, targets: np.ndarray) -> set[str]:
    return {
        replay_sample_id(inputs[index : index + 1], targets[index : index + 1])
        for index in range(len(inputs))
    }


def _train(model: CircadianPredictiveCodingNetwork, phases: tuple[str, ...]) -> None:
    roles = dict(zip(("a", "b"), _phase_rows(), strict=True))
    for phase in phases:
        inputs, targets = roles[phase]
        model.train_epoch(inputs, targets, 0.02, 2, 0.1)


def test_equal_caps_expose_fifo_recency_and_reservoir_order_invariance() -> None:
    fifo_ab = _model(ReplayRetentionPolicy("recent_fifo"))
    fifo_ba = _model(ReplayRetentionPolicy("recent_fifo"))
    reservoir_ab = _model(ReplayRetentionPolicy("seeded_reservoir", seed=53))
    reservoir_ba = _model(ReplayRetentionPolicy("seeded_reservoir", seed=53))
    for model, order in (
        (fifo_ab, ("a", "b")),
        (fifo_ba, ("b", "a")),
        (reservoir_ab, ("a", "b")),
        (reservoir_ba, ("b", "a")),
    ):
        _train(model, order)

    (a_inputs, a_targets), (b_inputs, b_targets) = _phase_rows()
    a_ids = _row_ids(a_inputs, a_targets)
    b_ids = _row_ids(b_inputs, b_targets)
    assert set(fifo_ab.get_replay_retention().sample_ids) == b_ids
    assert set(fifo_ba.get_replay_retention().sample_ids) == a_ids
    assert reservoir_ab.get_replay_retention() == reservoir_ba.get_replay_retention()
    reservoir_ids = set(reservoir_ab.get_replay_retention().sample_ids)
    assert reservoir_ids.issubset(a_ids | b_ids)
    assert len(reservoir_ids & a_ids) == len(reservoir_ids & b_ids) == 2
    for model in (fifo_ab, fifo_ba, reservoir_ab, reservoir_ba):
        retained = model.get_replay_retention()
        assert retained.example_count == 4
        assert retained.retained_bytes == 96
        assert len(set(retained.sample_ids)) == 4


@pytest.mark.parametrize("policy", ["recent_fifo", "seeded_reservoir"])
@pytest.mark.parametrize("budget", [ReplayRetentionBudget(2, 240), ReplayRetentionBudget(10, 48)])
def test_count_and_byte_caps_each_limit_the_same_unique_rows(
    policy: str, budget: ReplayRetentionBudget
) -> None:
    selected = ReplayRetentionPolicy(policy, seed=53 if policy == "seeded_reservoir" else None)
    model = _model(selected, budget)
    _train(model, ("a", "b"))
    retained = model.get_replay_retention()
    assert retained.example_count == 2
    assert retained.retained_bytes == 48
    assert len(set(retained.sample_ids)) == 2


@pytest.mark.parametrize("policy", ["recent_fifo", "seeded_reservoir"])
def test_repeated_retained_row_refreshes_without_using_another_slot(policy: str) -> None:
    selected = ReplayRetentionPolicy(policy, seed=53 if policy == "seeded_reservoir" else None)
    model = _model(selected)
    _train(model, ("a", "b"))
    old_ids = model.get_replay_retention().sample_ids
    row = next(
        item
        for item in model._replay_memory
        if replay_sample_id(item.input_batch, item.target_batch) in old_ids
    )
    row_id = replay_sample_id(row.input_batch, row.target_batch)
    model.train_epoch(row.input_batch, row.target_batch, 0.02, 2, 0.1)
    retained = model.get_replay_retention()
    refreshed = next(
        item
        for item in model._replay_memory
        if replay_sample_id(item.input_batch, item.target_batch) == row_id
    )
    assert retained.sample_ids == old_ids
    assert refreshed is not row
    assert retained.example_count == 4
    if policy == "recent_fifo":
        assert (
            replay_sample_id(
                model._replay_memory[-1].input_batch,
                model._replay_memory[-1].target_batch,
            )
            == row_id
        )


@pytest.mark.parametrize(
    ("name", "seed"),
    [
        ("unknown", None),
        ("recent_fifo", 1),
        ("seeded_reservoir", None),
        ("seeded_reservoir", -1),
        ("seeded_reservoir", True),
    ],
)
def test_invalid_policy_rejects_before_training(name: str, seed: Any) -> None:
    with pytest.raises((TypeError, ValueError), match="replay.*policy|seed"):
        ReplayRetentionPolicy(name, seed=seed)


def test_policy_identity_and_budget_reject_before_snapshot_restore() -> None:
    policy = ReplayRetentionPolicy("seeded_reservoir", seed=53)
    source = _model(policy)
    _train(source, ("a", "b"))
    saved = source.snapshot_state()
    matching = _model(policy)
    matching.restore_state(saved)
    assert matching.get_replay_retention() == source.get_replay_retention()
    _train(matching, ("b",))
    _train(source, ("b",))
    assert matching.get_replay_retention() == source.get_replay_retention()

    for incompatible in (
        _model(ReplayRetentionPolicy("recent_fifo")),
        _model(ReplayRetentionPolicy("seeded_reservoir", seed=54)),
        _model(None),
        _model(policy, ReplayRetentionBudget(3, 72)),
    ):
        before = incompatible.snapshot_state()
        with pytest.raises(ValueError, match="replay|snapshot"):
            incompatible.restore_state(saved)
        assert incompatible.get_replay_retention() == ReplayRetentionSnapshot((), 0, 0)
        assert incompatible.snapshot_state().state.keys() == before.state.keys()

    over_budget = replace(saved, state=deepcopy(saved.state))
    over_budget.state["_replay_memory"].append(saved.state["_replay_memory"][0])
    with pytest.raises(ValueError, match="replay"):
        _model(policy).restore_state(over_budget)


def test_default_bounded_hash_keeps_old_snapshot_identity() -> None:
    model = _model(None)
    _train(model, ("a", "b"))
    snapshot = model.snapshot_state()
    assert "_replay_retention_policy" not in snapshot.state
    assert model.get_replay_retention().sample_ids == tuple(
        sorted(set().union(*(_row_ids(*role) for role in _phase_rows())))[:4]
    )
    with pytest.raises(ValueError, match="explicit retention policy"):
        model.get_replay_exposure()


def test_opt_in_exposure_tracks_duplicates_and_restores_without_forged_ids() -> None:
    policy = ReplayRetentionPolicy("recent_fifo")

    def new_model() -> CircadianPredictiveCodingNetwork:
        model = CircadianPredictiveCodingNetwork(
            input_dim=2,
            hidden_dim=4,
            seed=41,
            circadian_config=CircadianConfig(
                sleep_mode="components",
                max_split_per_sleep=0,
                max_prune_per_sleep=0,
                replay_steps=1,
                replay_memory_size=1,
            ),
        )
        model.configure_replay_retention(_BUDGET, policy=policy)
        return model

    model = new_model()
    inputs, targets = _phase_rows()[0]
    for _ in range(2):
        model.train_epoch(inputs, targets, 0.02, 2, 0.1)
    before = model.get_replay_exposure()
    assert set(before.observed_ids) == _row_ids(inputs, targets)
    assert before.duplicate_ids == before.observed_ids
    assert before.duplicate_occurrences == len(inputs)
    assert before.exposed_ids == ()
    assert before.replay_updates == 0

    model.sleep_event(force_sleep=True)
    after = model.get_replay_exposure()
    assert len(after.exposed_ids) == after.replay_updates == 1
    assert set(after.exposed_ids) <= set(after.observed_ids)
    saved = model.snapshot_state()
    restored = new_model()
    restored.restore_state(saved)
    assert restored.get_replay_exposure() == after

    forged = replace(saved, state=deepcopy(saved.state))
    forged.state["_replay_exposed_ids"].add("f" * 64)
    with pytest.raises(ValueError, match="replay exposure"):
        new_model().restore_state(forged)
