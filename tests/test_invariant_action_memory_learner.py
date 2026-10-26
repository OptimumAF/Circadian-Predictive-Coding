"""Explicit representation binding and existing helper/stream array fixtures.

These are deterministic declared-TRAIN fixture invariants, not game physics or
independent scientific evidence. Raw controls and their saved states are intact.
"""

from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters.action_memory_learner import ActionMemoryLearner
from src.adapters.fifo_action_memory_learner import FifoActionMemoryLearner
from src.adapters.invariant_action_memory_learner import (
    InvariantActionMemoryLearner,
    InvariantMemoryPolicy,
    InvariantMemorySnapshot,
    copy_invariant_memory_snapshot,
)
from src.app.coupled_attribution import AttributionBatch, run_attribution_stream
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.action_memory import ActionMemoryEntry, ActionMemoryPolicy
from src.core.fifo_action_memory import FifoActionMemoryPolicy, FifoActionMemorySnapshot
from src.core.minigrid_invariant_encoding import INVARIANT_HISTORY_ENCODING_ID
from test_minigrid_invariant_encoding import frame


def policy(fifo=False, capacity=2):
    kind = FifoActionMemoryPolicy if fifo else ActionMemoryPolicy
    return InvariantMemoryPolicy(
        kind(
            feature_dim=296, num_classes=7, max_entries=capacity, max_payload_bytes=2424 * capacity
        )
    )


def features(cue=0, *, wall=5, direction=3):
    current = frame(wall_color=wall, direction=direction, cue_color=cue)
    return np.array([current + current], dtype=np.float64)


def labels(action):
    return np.array([[float(action)]], dtype=np.float64)


@pytest.mark.parametrize("fifo", [False, True])
def test_should_transfer_only_declared_nuisance_changes_and_preserve_count_native_objective(fifo):
    declared = policy(fifo)
    learner = InvariantActionMemoryLearner(declared)
    raw_kind: Any = FifoActionMemoryLearner if fifo else ActionMemoryLearner
    raw = raw_kind(declared.native_policy)
    original = features()
    variant = features(wall=2, direction=0)
    before = pickle.dumps((original, variant, declared))
    diagnostic = learner.train_batch(original, labels(6))
    assert diagnostic == raw.train_batch(original, labels(6))
    assert diagnostic.value == 1
    np.testing.assert_array_equal(learner.predict(variant), [[0, 0, 0, 0, 0, 0, 1]])
    np.testing.assert_array_equal(raw.predict(variant), np.full((1, 7), 1 / 7))
    np.testing.assert_array_equal(
        learner.predict(features(2, wall=2, direction=0)), np.full((1, 7), 1 / 7)
    )
    state = learner.snapshot_state()
    assert state.policy.encoding_id == INVARIANT_HISTORY_ENCODING_ID
    assert state.policy.method == f"public_invariant_{declared.native_policy.method}"
    assert (state.completed_updates, state.training_examples) == (1, 1)
    assert len(state.native_state.entries) == 1
    assert before == pickle.dumps((original, variant, declared))


@pytest.mark.parametrize("fifo", [False, True])
def test_should_own_policy_forks_snapshots_and_input_arrays_without_sharing_learning(fifo):
    declared = policy(fifo)
    learner = InvariantActionMemoryLearner(declared)
    inputs = features()
    learner.train_batch(inputs, labels(6))
    original = learner.snapshot_state()
    before = pickle.dumps(original)
    child = learner.fork()
    assert child is not learner and child.snapshot_state() == original
    inputs.fill(0)
    object.__setattr__(declared.native_policy, "max_entries", 1)
    assert learner.policy.max_entries == 2
    child.train_batch(features(2), labels(3))
    assert child.completed_updates == 2 and learner.completed_updates == 1
    assert learner.snapshot_state() == original
    copy = child.snapshot_state()
    object.__setattr__(copy.policy, "encoding_id", "raw_history")
    assert child.policy.encoding_id == INVARIANT_HISTORY_ENCODING_ID
    assert pickle.dumps(original) == before


@pytest.mark.parametrize("fifo", [False, True])
@pytest.mark.parametrize(
    "fault",
    ["raw", "encoding", "native_kind", "raw_keys", "extra", "extra_policy", "cap", "bool_count"],
)
def test_should_refuse_raw_mismatched_or_corrupted_snapshots_atomically(fifo, fault):
    learner = InvariantActionMemoryLearner(policy(fifo))
    learner.train_batch(features(), labels(6))
    before = learner.snapshot_state()
    state: Any = learner.snapshot_state()
    if fault == "raw":
        state = state.native_state
    elif fault == "encoding":
        object.__setattr__(state.policy, "encoding_id", "raw_history")
    elif fault == "native_kind":
        other = InvariantActionMemoryLearner(policy(not fifo))
        other.train_batch(features(), labels(6))
        state = InvariantMemorySnapshot(state.policy, other.snapshot_state().native_state)
    elif fault == "raw_keys":
        entry = ActionMemoryEntry(tuple(float(v) for v in features()[0]), (0, 0, 0, 0, 0, 0, 1))
        state = replace(state, native_state=replace(state.native_state, entries=(entry,)))
    elif fault == "extra":
        object.__setattr__(state, "undeclared_encoding", "other")
    elif fault == "extra_policy":
        object.__setattr__(state.policy, "undeclared_encoding", "other")
    elif fault == "cap":
        other = InvariantActionMemoryLearner(policy(fifo, 3))
        other.train_batch(features(), labels(6))
        state = other.snapshot_state()
    else:
        object.__setattr__(state.native_state.entries[0], "counts", (0, 0, 0, 0, 0, 0, True))
    with pytest.raises(ValueError):
        learner.restore_state(state)
    assert learner.snapshot_state() == before


@pytest.mark.parametrize("fifo", [False, True])
def test_should_refuse_bound_snapshot_at_the_original_raw_native_port(fifo):
    learner = InvariantActionMemoryLearner(policy(fifo))
    raw_kind: Any = FifoActionMemoryLearner if fifo else ActionMemoryLearner
    raw = raw_kind(learner.policy.native_policy)
    with pytest.raises(ValueError):
        raw.restore_state(learner.snapshot_state())


@pytest.mark.parametrize("fifo", [False, True])
def test_should_transform_every_row_before_committing_or_predicting_a_batch(fifo):
    learner = InvariantActionMemoryLearner(policy(fifo))
    learner.train_batch(features(), labels(6))
    before = learner.snapshot_state()
    malformed = np.concatenate((features(), features(2)), axis=0)
    malformed[1, 0] = 0.123
    with pytest.raises(ValueError):
        learner.train_batch(malformed, np.array([[6.0], [3.0]]))
    with pytest.raises(ValueError):
        learner.predict(malformed)
    assert learner.snapshot_state() == before


def test_should_preserve_atomic_refusal_and_fifo_retired_exposure_with_the_same_one_key_cap():
    exact, fifo = (InvariantActionMemoryLearner(policy(kind, 1)) for kind in (False, True))
    for learner in (exact, fifo):
        learner.train_batch(features(), labels(6))
    before = exact.snapshot_state()
    with pytest.raises(ValueError, match="capacity"):
        exact.train_batch(features(2), labels(3))
    assert exact.snapshot_state() == before
    fifo.train_batch(features(2), labels(3))
    state = fifo.snapshot_state()
    assert isinstance(state.native_state, FifoActionMemorySnapshot)
    assert (state.completed_updates, state.training_examples) == (2, 2)
    assert (state.native_state.evictions, state.native_state.retired_examples) == (1, 1)
    assert len(state.native_state.entries) == 1
    assert copy_invariant_memory_snapshot(state) == state


def batch(role, context, identity, cue, action, *, wall=5, direction=3):
    return AttributionBatch(
        role,
        context,
        (identity,),
        features(cue, wall=wall, direction=direction),
        labels(action),
        num_classes=7,
    )


@pytest.mark.parametrize("fifo", [False, True])
def test_should_use_existing_isolated_probe_with_raw_inputs_and_bound_native_state(fifo):
    source = InvariantActionMemoryLearner(policy(fifo))
    state = source.snapshot_state()
    training = (
        batch("probe_train", "A", "train0", 0, 6),
        batch("probe_train", "B", "train1", 2, 3),
        batch("probe_train", "A", "train2", 0, 6, wall=2, direction=0),
    )
    evaluation = (
        batch("probe_test", "A", "evalA", 0, 6, wall=1, direction=1),
        batch("probe_test", "B", "evalB", 2, 3, wall=4, direction=2),
    )
    before = pickle.dumps((state, training, evaluation))
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=3), lambda: 0.0)

    def guard():
        assert source.snapshot_state() == state
        assert pickle.dumps((state, training, evaluation)) == before

    report = run_fresh_task_probe(
        source,
        state,
        training,
        evaluation,
        budget=budget,
        checkpoint_updates=(0, 1, 2, 3),
        timer=lambda: 0.0,
        decision_diagnostics=True,
        check_boundary=guard,
    )
    assert report["status"] == "complete"
    assert report["completed_updates"] == budget.updates_completed == 3
    assert report["completed_training_examples"] == 3
    assert report["score_forward_calls_completed"] == 8
    assert [
        [point["scores"][context]["accuracy"] for context in ("A", "B")]
        for point in report["checkpoints"]
    ] == [[0, 0], [1, 0], [1, 1], [1, 1]]
    assert source.snapshot_state() == state and before == pickle.dumps(
        (state, training, evaluation)
    )


@pytest.mark.parametrize("fifo", [False, True])
def test_should_use_existing_delayed_train_stream_without_final_scores_or_source_updates(fifo):
    source = InvariantActionMemoryLearner(policy(fifo))
    state = source.snapshot_state()
    training = (
        batch("train", "A", "train0", 0, 6),
        batch("train", "B", "train1", 2, 3),
        batch("train", "A", "train2", 0, 6, wall=2, direction=0),
    )
    before = pickle.dumps(training)
    seen = []
    checkpoints = []

    def prearrival(observation, candidate):
        seen.append((observation.tick, candidate.state.completed_updates))
        assert source.snapshot_state() == state

    def checkpoint(update, owned, packet):
        checkpoints.append((update, owned))
        assert owned.policy.encoding_id == INVARIANT_HISTORY_ENCODING_ID

    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=3), lambda: 0.0)
    report = run_attribution_stream(
        source,
        training,
        (),
        checkpoint_updates=(0, 1, 2, 3),
        label_delay=1,
        budget=budget,
        observe_features=prearrival,
        observe_checkpoint=checkpoint,
        training_only=True,
    )
    assert (
        report["completed_updates"] == report["training_examples"] == budget.updates_completed == 3
    )
    assert report["score_forward_calls"] == 0 and report["evaluation_scope"] == "training_only"
    assert seen == [(0, 0), (1, 0), (2, 1)]
    assert [update for update, owned in checkpoints] == [0, 1, 2, 3]
    assert len(checkpoints[-1][1].native_state.entries) == 2
    assert source.snapshot_state() == state and pickle.dumps(training) == before
