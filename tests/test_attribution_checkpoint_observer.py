"""Checkpoint callbacks borrow detached states and only already-arrived labels."""

from typing import Any

import numpy as np
import pytest

from src.app.coupled_attribution import AttributionBatch, run_attribution_stream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.learner_ports import TrainingDiagnostic


class SpyLearner:
    def __init__(self, trace=None, state=None):
        self.trace = [] if trace is None else trace
        self.state = np.array([0.0]) if state is None else state.copy()

    def fork(self):
        self.trace.append(("fork", float(self.state[0])))
        return SpyLearner(self.trace, self.state)

    def snapshot_state(self):
        return self.state.copy()

    def restore_state(self, state):
        self.state = state.copy()

    def train_batch(self, x, y):
        self.trace.append(("train", float(self.state[0]), x.copy(), y.copy()))
        self.state += 1
        return TrainingDiagnostic("observer_spy_v1", float(self.state[0]))

    def predict(self, x):
        self.trace.append(("predict", float(self.state[0]), x.copy()))
        return np.full((len(x), 1), (self.state[0] + 1) / 10)


def batches():
    training = tuple(
        AttributionBatch(
            "train", "A", (f"train-{i}",), np.array([[0.1 + i, 0.2]]), np.array([[float(i)]])
        )
        for i in range(2)
    )
    heldout = (
        AttributionBatch("final_test", "A", ("test-0",), np.array([[0.7, 0.8]]), np.array([[0.3]])),
    )
    return training, heldout


def budget():
    return ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)


def run(source, **options):
    training, heldout = batches()
    return run_attribution_stream(
        source,
        training,
        heldout,
        checkpoint_updates=(0, 1, 2),
        label_delay=3,
        budget=options.pop("budget", budget()),
        **options,
    )


def test_should_preserve_default_stream_results_and_native_call_order_exactly():
    a, b = SpyLearner(), SpyLearner()
    assert run(a) == run(b, observe_checkpoint=None)
    np.testing.assert_equal(a.trace, b.trace)


def test_should_observe_initial_state_before_labels_then_only_actual_arrived_training_batches():
    source = SpyLearner()
    observed = []

    def observe(updates, state, batch):
        source.trace.append(("observe", updates))
        observed.append((updates, state.copy(), batch))

    result = run(source, observe_checkpoint=observe)
    assert [e[0] for e in observed] == [0, 1, 2]
    assert [float(e[1][0]) for e in observed] == [0, 1, 2]
    assert observed[0][2] is None
    training, _ = batches()
    for (_, _, actual), expected in zip(observed[1:], training, strict=True):
        assert actual.role == "train"
        assert actual.row_ids == expected.row_ids
        np.testing.assert_array_equal(actual.features, expected.features)
        np.testing.assert_array_equal(actual.targets, expected.targets)
    events = [e[0] for e in source.trace if e[0] in ("observe", "train", "predict")]
    assert events == [
        "observe",
        "train",
        "observe",
        "train",
        "observe",
        "predict",
        "predict",
        "predict",
    ]
    assert [r["arrived_at"] for r in result["applied_updates"]] == [3, 4]
    assert source.state[0] == 0


def test_should_detach_observer_state_data_and_snapshots_used_for_scoring():
    source = SpyLearner()
    observed = []

    def mutate(updates, state, batch):
        observed.append(state)
        state[:] = 900
        if batch is not None:
            batch.features[:] = 901
            batch.targets[:] = 902

    result = run(source, observe_checkpoint=mutate)
    assert [row["updates"] for row in result["checkpoints"]] == [0, 1, 2]
    assert [e[1] for e in source.trace if e[0] == "predict"] == [0, 1, 2]
    actual = [e for e in source.trace if e[0] == "train"]
    for event, expected in zip(actual, batches()[0], strict=True):
        np.testing.assert_array_equal(event[2], expected.features)
        np.testing.assert_array_equal(event[3], expected.targets)
    observed[0][:] = -1
    assert source.state[0] == 0


@pytest.mark.parametrize("fail_at", (0, 1, 2))
def test_should_propagate_observer_fault_and_never_start_heldout_scoring(fail_at):
    source = SpyLearner()

    def broken(updates, state, batch):
        if updates == fail_at:
            raise RuntimeError("observer failed")

    with pytest.raises(RuntimeError, match="observer failed"):
        run(source, observe_checkpoint=broken)
    assert len([e for e in source.trace if e[0] == "train"]) == fail_at
    assert not any(e[0] == "predict" for e in source.trace)


def test_should_check_budget_after_observer_before_any_next_update_or_score():
    source = SpyLearner()
    now = [0.0]
    session = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=1.0), lambda: now[0]
    )

    def consume(updates, state, batch):
        if updates == 1:
            now[0] = 2.0

    with pytest.raises(ToyExecutionStopped):
        run(source, budget=session, observe_checkpoint=consume)
    assert session.updates_completed == 1
    assert len([e for e in source.trace if e[0] == "train"]) == 1
    assert not any(e[0] == "predict" for e in source.trace)


@pytest.mark.parametrize("observer", (1, {}, "callback"))
def test_should_refuse_noncallable_observer_before_any_learner_or_budget_callback(
    observer: Any, monkeypatch
):
    source = SpyLearner()
    session = budget()
    calls = []
    monkeypatch.setattr(session, "before_final", lambda: calls.append("budget"))
    with pytest.raises(ValueError):
        run(source, budget=session, observe_checkpoint=observer)
    assert source.trace == [] and calls == []
