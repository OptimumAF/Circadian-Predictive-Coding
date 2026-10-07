"""Deterministic retention observations before changing replay policy."""

from __future__ import annotations

import numpy as np

from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
    ReplayRetentionBudget,
    replay_sample_id,
)


def _model(*, batch_slots: int) -> CircadianPredictiveCodingNetwork:
    return CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=4,
        seed=37,
        circadian_config=CircadianConfig(replay_memory_size=batch_slots, replay_steps=0),
    )


def _train(
    model: CircadianPredictiveCodingNetwork, inputs: np.ndarray, targets: np.ndarray
) -> None:
    model.train_epoch(inputs, targets, 0.02, 2, 0.1)


def _row_ids(inputs: np.ndarray, targets: np.ndarray) -> set[str]:
    return {
        replay_sample_id(inputs[index : index + 1], targets[index : index + 1])
        for index in range(len(inputs))
    }


def test_legacy_batch_slots_have_variable_example_bytes_and_evict_old_task() -> None:
    model = _model(batch_slots=2)
    inputs = np.arange(14, dtype=np.float64).reshape(7, 2) / 10.0
    targets = np.array([[0], [1], [0], [1], [0], [1], [0]], dtype=np.float64)
    phase_a_ids = _row_ids(inputs[:1], targets[:1])

    for start, stop in ((0, 1), (1, 5), (5, 7)):
        _train(model, inputs[start:stop], targets[start:stop])

    memory = model.snapshot_state().state["_replay_memory"]
    assert memory.maxlen == 2
    assert [item.input_batch.shape[0] for item in memory] == [4, 2]
    assert sum(item.input_batch.nbytes + item.target_batch.nbytes for item in memory) == 144
    retained_ids = set().union(*(_row_ids(item.input_batch, item.target_batch) for item in memory))
    assert len(retained_ids) == 6
    assert retained_ids.isdisjoint(phase_a_ids)


def test_legacy_priorities_do_not_age_and_duplicate_batches_use_more_slots() -> None:
    model = _model(batch_slots=3)
    phase_a = np.array([[0.0, 0.1]], dtype=np.float64)
    phase_b = np.array([[1.0, 1.1]], dtype=np.float64)
    a_target = np.zeros((1, 1), dtype=np.float64)
    b_target = np.ones((1, 1), dtype=np.float64)

    _train(model, phase_a, a_target)
    saved_priority = model._replay_memory[0].priority
    prior_weights = model.weight_hidden_output.copy()
    _train(model, phase_b, b_target)
    assert not np.array_equal(model.weight_hidden_output, prior_weights)
    assert model._replay_memory[0].priority == saved_priority
    _train(model, phase_a, a_target)

    memory = model.snapshot_state().state["_replay_memory"]
    assert len(memory) == 3
    retained_ids = [replay_sample_id(item.input_batch, item.target_batch) for item in memory]
    assert len(set(retained_ids)) == 2
    assert retained_ids[0] == retained_ids[2]
    assert memory[0].priority == saved_priority


def test_bounded_content_retention_measures_phase_a_survival_after_shift() -> None:
    model = _model(batch_slots=1)
    model.configure_replay_retention(ReplayRetentionBudget(max_examples=4, max_bytes=96))
    phase_a = np.arange(8, dtype=np.float64).reshape(4, 2) / 10.0
    phase_b = np.arange(8, 16, dtype=np.float64).reshape(4, 2) / 10.0
    a_target = np.zeros((4, 1), dtype=np.float64)
    b_target = np.ones((4, 1), dtype=np.float64)
    phase_a_ids = _row_ids(phase_a, a_target)
    phase_b_ids = _row_ids(phase_b, b_target)

    _train(model, phase_a, a_target)
    after_a = model.get_replay_retention()
    _train(model, phase_b, b_target)
    after_b = model.get_replay_retention()

    assert after_a.example_count == after_b.example_count == 4
    assert after_a.retained_bytes == after_b.retained_bytes == 96
    assert set(after_a.sample_ids) == phase_a_ids
    assert set(after_b.sample_ids).issubset(phase_a_ids | phase_b_ids)
    assert after_b.sample_ids == tuple(sorted(phase_a_ids | phase_b_ids)[:4])
    assert len(phase_a_ids.intersection(after_b.sample_ids)) == 2

    surviving_a_index = next(
        index
        for index in range(len(phase_a))
        if replay_sample_id(phase_a[index : index + 1], a_target[index : index + 1])
        in after_b.sample_ids
    )
    repeated_input = phase_a[surviving_a_index : surviving_a_index + 1]
    repeated_target = a_target[surviving_a_index : surviving_a_index + 1]
    repeated_id = replay_sample_id(repeated_input, repeated_target)
    old_snapshot = next(
        item
        for item in model._replay_memory
        if replay_sample_id(item.input_batch, item.target_batch) == repeated_id
    )
    _train(model, repeated_input, repeated_target)
    new_snapshot = next(
        item
        for item in model._replay_memory
        if replay_sample_id(item.input_batch, item.target_batch) == repeated_id
    )
    assert model.get_replay_retention().sample_ids == after_b.sample_ids
    assert new_snapshot is not old_snapshot
    assert new_snapshot.priority == abs(
        float(model.predict_proba(repeated_input)[0, 0] - repeated_target[0, 0])
    )
