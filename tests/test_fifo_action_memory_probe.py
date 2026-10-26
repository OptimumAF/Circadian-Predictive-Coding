"""Array-fixture old/new/return through the existing isolated probe helper.

These tests demonstrate bounded learner/helper composition, not MiniGrid physics,
independent fresh-task confirmation, generalization or a comparative benefit.
"""

import pickle

import numpy as np
import pytest

from src.adapters.action_memory_learner import ActionMemoryLearner
from src.adapters.fifo_action_memory_learner import FifoActionMemoryLearner
from src.app.coupled_attribution import AttributionBatch, run_attribution_stream
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.action_memory import ActionMemoryPolicy
from src.core.fifo_action_memory import (
    FIFO_ACTION_MEMORY_COUNT_DIAGNOSTIC,
    FifoActionMemoryPolicy,
    convert_action_memory_to_fifo,
)


def frame(value, width):
    result = np.zeros((1, width), dtype=np.float64)
    result[0, 0] = value
    return result


@pytest.mark.parametrize("width,classes", [(16, 5), (296, 7)])
def test_should_learn_forget_and_relearn_on_a_disposable_bounded_copy(monkeypatch, width, classes):
    table_bytes = 2 * (width + classes) * 8
    original = ActionMemoryLearner(
        ActionMemoryPolicy(
            feature_dim=width,
            num_classes=classes,
            max_entries=2,
            max_payload_bytes=table_bytes,
        )
    )
    original.train_batch(frame(0.25, width), np.array([[float(classes - 1)]]))
    old_state = original.snapshot_state()
    source = FifoActionMemoryLearner(convert_action_memory_to_fifo(old_state).policy)
    source.restore_state(convert_action_memory_to_fifo(old_state))
    state = source.snapshot_state()
    contexts = {"A": (0.25, classes - 1), "B": (0.5, 1), "C": (0.75, 2)}

    def packet(role, context, index):
        value, target = contexts[context]
        return AttributionBatch(
            role,
            context,
            (f"{role}-{context}-{index}",),
            frame(value, width),
            np.array([[float(target)]], dtype=np.float64),
            num_classes=classes,
        )

    training = tuple(packet("probe_train", context, i) for i, context in enumerate("BCA"))
    evaluation = tuple(packet("probe_test", context, i) for i, context in enumerate("ABC"))
    before = pickle.dumps((old_state, state, training, evaluation))
    children = []
    fork = FifoActionMemoryLearner.fork

    def observed_fork(owner):
        child = fork(owner)
        children.append(child)
        return child

    monkeypatch.setattr(FifoActionMemoryLearner, "fork", observed_fork)
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=3, max_wall_seconds=1.0), lambda: 0.0
    )

    def check_source():
        assert original.snapshot_state() == old_state
        assert source.snapshot_state() == state
        assert pickle.dumps((old_state, state, training, evaluation)) == before

    report = run_fresh_task_probe(
        source,
        state,
        training,
        evaluation,
        budget=budget,
        checkpoint_updates=(0, 1, 2, 3),
        timer=lambda: 0.0,
        decision_diagnostics=True,
        check_boundary=check_source,
    )
    assert report["status"] == "complete" and report["reason"] is None
    assert (
        report["completed_updates"] == report["attempted_updates"] == budget.updates_completed == 3
    )
    assert report["completed_training_examples"] == report["attempted_training_examples"] == 3
    assert report["score_forward_calls_completed"] == report["score_forward_calls_attempted"] == 12
    assert report["reserved_data_bytes"] == 6 * (width + 1) * 8
    assert (
        report["diagnostics"]
        == [dict(definition=FIFO_ACTION_MEMORY_COUNT_DIAGNOSTIC, value=1.0)] * 3
    )
    expected = [(1, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1)]
    for point, accuracy in zip(report["checkpoints"], expected, strict=True):
        assert tuple(point["scores"][context]["accuracy"] for context in "ABC") == accuracy
        for context, correct in zip("ABC", accuracy, strict=True):
            score = point["scores"][context]
            assert score["ce"] == pytest.approx(0.0 if correct else np.log(classes))
            (decision,) = point["decision_diagnostics"][context]["decisions"]
            assert decision["correct"] == bool(correct)
    assert len(children) == 1 and children[0] is not source
    final = children[0].snapshot_state()
    assert (final.completed_updates, final.training_examples) == (4, 4)
    assert (final.evictions, final.retired_examples) == (2, 2)
    assert [entry.features[0] for entry in final.entries] == [0.75, 0.25]
    assert sum(sum(entry.counts) for entry in final.entries) + final.retired_examples == 4
    assert len(final.entries) * final.policy.entry_payload_bytes == table_bytes
    check_source()


@pytest.mark.parametrize("width,classes", [(16, 5), (296, 7)])
def test_should_replace_memory_only_after_labels_arrive_in_the_existing_train_stream(
    width, classes
):
    source = FifoActionMemoryLearner(
        FifoActionMemoryPolicy(
            feature_dim=width,
            num_classes=classes,
            max_entries=2,
            max_payload_bytes=2 * (width + classes) * 8,
        )
    )
    original = source.snapshot_state()
    contexts = {"A": (0.25, classes - 1), "B": (0.5, 1), "C": (0.75, 2)}
    training = tuple(
        AttributionBatch(
            "train",
            context,
            (f"train-{i}",),
            frame(contexts[context][0], width),
            np.array([[float(contexts[context][1])]]),
            num_classes=classes,
        )
        for i, context in enumerate("ABCA")
    )
    old_inputs = pickle.dumps(training)
    prearrival = []
    observed = []

    def before_labels(features, candidate):
        prearrival.append((features.tick, candidate.state.completed_updates))
        assert source.snapshot_state() == original

    def checkpoint(update, state, batch):
        observed.append((update, state))
        assert state.completed_updates == update
        assert (batch is None) == (update == 0)
        assert source.snapshot_state() == original

    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=4, max_wall_seconds=1.0), lambda: 0.0
    )
    result = run_attribution_stream(
        source,
        training,
        (),
        checkpoint_updates=(0, 1, 2, 3, 4),
        label_delay=1,
        budget=budget,
        observe_features=before_labels,
        observe_checkpoint=checkpoint,
        training_only=True,
    )
    assert (
        result["completed_updates"] == result["training_examples"] == budget.updates_completed == 4
    )
    assert result["score_forward_calls"] == 0 and result["evaluation_scope"] == "training_only"
    assert len(result["applied_updates"]) == 4 and result["checkpoints"] == []
    assert prearrival == [(0, 0), (1, 0), (2, 1), (3, 2)]
    assert [update for update, state in observed] == [0, 1, 2, 3, 4]
    assert [tuple(entry.features[0] for entry in state.entries) for update, state in observed] == [
        (),
        (0.25,),
        (0.25, 0.5),
        (0.5, 0.75),
        (0.75, 0.25),
    ]
    final = observed[-1][1]
    assert (final.evictions, final.retired_examples, final.training_examples) == (2, 2, 4)
    assert source.snapshot_state() == original and pickle.dumps(training) == old_inputs
