"""Native tracked histories cross owned ports and the existing rehearsal stream."""

from dataclasses import asdict, replace
import pickle
from typing import Any, cast

import numpy as np
import pytest

from src.adapters.coupled_learners import CoupledLearner
from src.adapters.native_unit_diagnostics import read_coupled_unit_diagnostics
from src.app.coupled_attribution import AttributionBatch, run_attribution_stream
from src.app.stream_rehearsal import RecentStreamReplayPolicy, StreamRehearsal
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_predictive_coding import CoupledSnapshot
from src.core.tracked_coupled_predictive_coding import (
    TrackedCoupledCircadianPredictiveCodingNetwork,
    TrackedCoupledPredictiveCodingNetwork,
    TrackedCoupledSnapshot,
)

KINDS = ("pc", "neutral", "active")


def model(kind):
    if kind == "pc":
        return TrackedCoupledPredictiveCodingNetwork(2, (3,), 17, history_decay=0.75)
    return TrackedCoupledCircadianPredictiveCodingNetwork(
        2,
        (3,),
        17,
        history_decay=0.75,
        config=CoupledCircadianConfig.matched_pc_control() if kind == "neutral" else None,
    )


def learner(native):
    return CoupledLearner(
        native, learning_rate=0.02, inference_steps=2, inference_learning_rate=0.04
    )


def batch():
    return np.array([[0.2, -0.4], [-0.5, 0.3]]), np.array([[1.0], [0.0]])


def arrays(snapshot):
    p = snapshot.parameters
    return (*p.hidden_weights, *p.hidden_biases, p.output_weight, p.output_bias)


@pytest.mark.parametrize("kind", KINDS)
def test_should_own_fork_restore_and_continue_complete_native_history_through_existing_port(kind):
    native = model(kind)
    source_before = pickle.dumps(vars(native))
    owned = learner(native)
    x, y = batch()
    owned.train_batch(x, y)
    owned.train_batch(x, y)
    saved = owned.snapshot_state()
    assert type(saved) is TrackedCoupledSnapshot
    assert saved.unit_history is not None
    assert saved.unit_history.updates == 2
    child = owned.fork()
    child.train_batch(x, y)
    assert owned.snapshot_state().unit_history.updates == 2
    assert child.snapshot_state().unit_history.updates == 3
    child.restore_state(saved)
    left = owned.train_batch(x, y)
    right = child.train_batch(x, y)
    assert left == right
    a, b = owned.snapshot_state(), child.snapshot_state()
    for first, second in zip(arrays(a), arrays(b), strict=True):
        np.testing.assert_array_equal(first, second)
        assert not np.shares_memory(first, second)
    for name in ("unit_ids", "importance_ema", "update_age"):
        first, second = getattr(a.unit_history, name), getattr(b.unit_history, name)
        np.testing.assert_array_equal(first, second)
        assert not np.shares_memory(first, second)
    assert pickle.dumps(vars(native)) == source_before
    assert a.unit_history.updates == b.unit_history.updates == 3
    if kind != "pc":
        assert a.wake_state.wake_updates == b.wake_state.wake_updates == 3
    saved.unit_history.importance_ema.fill(999)
    saved.unit_history.update_age.fill(999)
    assert owned.snapshot_state().unit_history.updates == 3
    assert np.all(owned.snapshot_state().unit_history.update_age == 3)


def stream_batches():
    x, y = batch()
    training = (
        AttributionBatch("train", "A", ("train-a0", "train-a1"), x, y),
        AttributionBatch("train", "B", ("train-b0", "train-b1"), x + 0.1, 1 - y),
    )
    heldout = (
        AttributionBatch("final_test", "A", ("final-a0", "final-a1"), x + 0.2, y.copy()),
        AttributionBatch("final_test", "B", ("final-b0", "final-b1"), x + 0.3, 1 - y),
    )
    return training, heldout


@pytest.mark.parametrize("kind", KINDS)
def test_should_count_replay_in_history_through_actual_delayed_stream_and_consolidation(
    kind, monkeypatch
):
    owned = learner(model(kind))
    source_before = pickle.dumps(owned.snapshot_state())
    training, heldout = stream_batches()
    wake = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=10), lambda: 0.0
    )
    replay = ToyBudgetSession(
        ToyExecutionBudget(max_replay_examples=2, max_wall_seconds=10), lambda: 0.0
    )
    policy = RecentStreamReplayPolicy(1, 1, ReplayRetentionBudget(4, 96))
    rehearsal = StreamRehearsal(
        owned, wake, policy, replay_budget=replay, total_updates=2, input_dim=2
    )
    observations: list[tuple[int, dict[str, Any]]] = []
    predict = CoupledLearner.predict

    def final_only(self, features):
        assert [entry[0] for entry in observations] == [0, 1, 2]
        return predict(self, features)

    monkeypatch.setattr(CoupledLearner, "predict", final_only)

    def observe(updates, state, arrived):
        assert type(state) is TrackedCoupledSnapshot
        assert state.unit_history is not None
        diagnostics = read_coupled_unit_diagnostics(state)
        observations.append((updates, asdict(diagnostics)))
        assert diagnostics.successful_updates == 2 * updates
        assert diagnostics.history.age.minimum == diagnostics.history.age.maximum == 2 * updates
        assert diagnostics.array_bytes == 72
        assert (arrived is None) == (updates == 0)
        if kind != "pc":
            assert state.wake_state is not None
            assert state.wake_state.wake_updates == 2 * updates
        # Observer state is detached from both the candidate and saved score states.
        state.unit_history.importance_ema.fill(999)
        for array in arrays(state):
            array.fill(999)

    result = run_attribution_stream(
        owned,
        training,
        heldout,
        checkpoint_updates=(0, 1, 2),
        label_delay=1,
        budget=wake,
        observe_checkpoint=observe,
        rehearsal=rehearsal,
    )
    assert result["completed_updates"] == 2 and result["training_examples"] == 4
    assert (
        result["rehearsal"]["attempted_updates"]
        == result["rehearsal"]["returned_updates"]
        == result["rehearsal"]["committed_updates"]
        == 2
    )
    assert len(result["applied_consolidations"]) == 2 and result["score_forward_calls"] == 6
    assert wake.updates_completed == replay.replay_examples_completed == 2
    assert pickle.dumps(owned.snapshot_state()) == source_before
    assert [point["updates"] for point in result["checkpoints"]] == [0, 1, 2]
    assert observations[-1][1]["history"]["importance"]["maximum"] < 999


def test_should_detach_read_only_diagnostics_and_refuse_untracked_or_corrupt_history():
    native = model("pc")
    snapshot = native.snapshot_state()
    before = pickle.dumps(snapshot)
    summary = read_coupled_unit_diagnostics(snapshot)
    assert pickle.dumps(snapshot) == before
    assert [u.neuron_id for u in summary.history.units] == [0, 1, 2]
    assert all(u.parent_id is None for u in summary.history.units)
    assert summary.successful_updates == 0 and summary.array_bytes == 72
    assert "batch-aggregate" in summary.importance_definition
    assert "including replay" in summary.age_definition
    snapshot.unit_history.importance_ema.fill(1)
    assert summary.history.importance.maximum == 0
    old = CoupledSnapshot(1, snapshot.parameters, snapshot.precisions)
    with pytest.raises(ValueError, match="tracked snapshot"):
        read_coupled_unit_diagnostics(cast(TrackedCoupledSnapshot, old))
    with pytest.raises(ValueError, match="UnitHistory"):
        read_coupled_unit_diagnostics(replace(snapshot, unit_history=None))
    with pytest.raises(ValueError):
        read_coupled_unit_diagnostics(
            replace(snapshot, unit_history=replace(snapshot.unit_history, update_age=np.ones(3)))
        )
