"""Explicit routed residual updates; synthetic software fixtures, no native study."""

from copy import deepcopy
from dataclasses import replace
import random

import numpy as np
import pytest

from src.adapters import context_linear_fit_learner as context
from src.adapters import linear_fit_learner as linear
from src.core.categorical_linear_fit import CategoricalLinearFit
from src.core.categorical_linear_state import LinearFitSnapshot, MAX_COUNT
from src.core.context_linear_state import ContextLinearFitPolicy, ContextLinearFitSnapshot
from src.core.learner_ports import TrainingUpdateFailed
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget


def packet(index=0, sign=1.0, label=1.0, rows=1) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((rows, 4))
    x[:, 0] = sign
    x[:, 2 + index] = 1
    return x, np.full((rows, 1), label)


def model(rows=1):
    # Literal prior metadata, NOT uncounted source training.
    return CategoricalLinearFit(2, 2, rows, ((0.2, -0.1),) * 3, 1, (1.0,) * min(rows, 3), 0.0)


def owner(*, warm=False, cls=context.ContextLinearFitLearner):
    source = cls(ContextLinearFitPolicy(4, 2, (2, 3)))
    if warm:
        state = source.snapshot_state()
        heads = (
            LinearFitSnapshot(state.heads[0].policy, model(), 1, 1),
            LinearFitSnapshot(state.heads[1].policy, model(), 1, 1),
        )
        source.restore_state(ContextLinearFitSnapshot(state.policy, heads, 2, 2))
    return source


def test_should_route_both_heads_preserve_inactive_and_learn_orthogonal_return():
    source = owner()
    before = source.snapshot_state()
    child = source.fork()
    for i, (index, sign, label) in enumerate(
        ((0, 1.0, 1.0), (1, 1.0, 0.0), (0, -1.0, 0.0), (1, -1.0, 1.0), (0, 1.0, 1.0), (1, 1.0, 0.0))
    ):
        previous = child.snapshot_state()
        result = child.train_batch_minimum_change(*packet(index, sign, label))
        after = child.snapshot_state()
        assert (
            result.definition == context.CONTEXT_MINIMUM_CHANGE_RESIDUAL_ID and result.value < 1e-24
        )
        assert after.heads[1 - index] == previous.heads[1 - index]
        assert after.completed_updates == after.training_examples == i + 1
        assert after.heads[index].completed_updates == previous.heads[index].completed_updates + 1
    mixed = np.vstack(
        [packet(0, 1.0)[0], packet(0, -1.0)[0], packet(1, 1.0)[0], packet(1, -1.0)[0]]
    )
    assert (child.predict(mixed)[:, 0] > 0.5).astype(int).tolist() == [1, 0, 0, 1]
    assert source.snapshot_state() == before
    assert [h.completed_updates for h in after.heads] == [3, 3]
    assert set(after.__dataclass_fields__) == {
        "policy",
        "heads",
        "completed_updates",
        "training_examples",
    }
    object.__setattr__(after.heads[1].model, "squared_error", 42.0)
    assert child.snapshot_state() != after


def test_should_keep_first_fit_equivalence_and_default_nonsticky():
    a, b = owner(), owner()
    a.train_batch(*packet())
    b.train_batch_minimum_change(*packet())
    assert a.snapshot_state() == b.snapshot_state()
    b.train_batch_minimum_change(*packet(sign=-1.0, label=0.0))
    inactive = b.snapshot_state().heads[1]
    b.train_batch(*packet(sign=-1.0, label=0.0))
    a.train_batch(*packet(sign=-1.0, label=0.0))
    assert b.snapshot_state().heads[0].model == a.snapshot_state().heads[0].model
    assert b.snapshot_state().heads[1] == inactive


def test_should_use_generic128_row_prior_and_keep_other_head():
    source = owner()
    initial = source.snapshot_state()
    first = LinearFitSnapshot(initial.heads[0].policy, model(128), 1, 128)
    source.restore_state(
        ContextLinearFitSnapshot(initial.policy, (first, initial.heads[1]), 1, 128)
    )
    x, y = packet(rows=128)
    x[:, 0] = 0
    source.train_batch_minimum_change(x, y)
    after = source.snapshot_state()
    assert after.completed_updates == 2 and after.training_examples == 256
    assert after.heads[0].model.fit_rows == 128 and after.heads[1] == initial.heads[1]


@pytest.mark.parametrize(
    "fault",
    ["missing", "both", "fraction", "mixed", "width", "dtype", "target_dtype", "class", "counter"],
)
def test_should_refuse_invalid_route_or_target_before_private_operation(fault, monkeypatch):
    source = owner()
    x, y = packet()
    if fault == "missing":
        x[:, 2:] = 0
    elif fault == "both":
        x[:, 2:] = 1
    elif fault == "fraction":
        x[:, 2:] = 0.5
    elif fault == "mixed":
        x, y = np.vstack((x, packet(1)[0])), np.vstack((y, y))
    elif fault == "width":
        x = x[:, :3]
    elif fault == "dtype":
        x = x.astype(np.float32)
    elif fault == "target_dtype":
        y = y.astype(np.int64)
    elif fault == "class":
        y[:] = 2
    else:
        s = source.snapshot_state()
        head = LinearFitSnapshot(s.heads[0].policy, model(), MAX_COUNT, MAX_COUNT)
        source.restore_state(
            ContextLinearFitSnapshot(s.policy, (head, s.heads[1]), MAX_COUNT, MAX_COUNT)
        )
    before = source.snapshot_state()
    monkeypatch.setattr(
        linear.LinearFitLearner,
        "train_batch_minimum_change",
        lambda *a: pytest.fail("private operation"),
    )
    with pytest.raises(ValueError):
        source.train_batch_minimum_change(x, y)
    assert source.snapshot_state() == before


@pytest.mark.parametrize(
    "fault",
    [
        "lost",
        "raw",
        "targets",
        "inactive",
        "active",
        "policy",
        "state",
        "staged_prior",
        "rng",
        "return_rows",
    ],
)
def test_should_refuse_changed_state_or_lost_result_and_preserve_work(fault, monkeypatch):
    source = owner(warm=True)
    before = source.snapshot_state()
    x, y = packet()
    original = linear.update_categorical_linear
    rng = random.getstate()

    def update(prior, features, targets):
        result = original(prior, features, targets)
        if fault == "lost":
            raise RuntimeError("lost routed residual")
        if fault == "raw":
            x[:, 2:] = [0, 1]
        elif fault == "targets":
            y[:] = 0
        elif fault in ("inactive", "active"):
            index = int(fault == "inactive")
            object.__setattr__(source._state.heads[index].model, "coefficients", ((0.0, 0.0),) * 3)
        elif fault == "policy":
            source._policy = replace(source.policy)
        elif fault == "state":
            source._state = deepcopy(source._state)
        elif fault == "staged_prior":
            object.__setattr__(prior, "squared_error", False)
        elif fault == "rng":
            random.seed(71)  # deliberate fault, restored below
        elif fault == "return_rows":
            object.__setattr__(result, "fit_rows", 2)
        return result

    monkeypatch.setattr(linear, "update_categorical_linear", update)
    try:
        with pytest.raises(TrainingUpdateFailed) as failed:
            source.train_batch_minimum_change(x, y)
        work = failed.value.work
        assert failed.value.definition == context.CONTEXT_MINIMUM_CHANGE_RESIDUAL_ID
        assert work["fit_attempts"] == work["fit_rows_attempted"] == 1
        assert work["fit_returns"] == work["fit_rows_returned"] == int(fault != "lost")
        assert work["unreturned_work_unknown"] == (fault == "lost")
        assert work["candidate_commits"] == 0 and work["context_index"] == 0
        assert work["private_candidate_commits"] == int(
            fault in ("raw", "inactive", "active", "policy", "state")
        )
        if fault in ("inactive", "active"):
            assert source.snapshot_state() != before
        else:
            assert source.snapshot_state() == before
        if fault == "raw":
            assert x[0, 3] == 1
        if fault == "rng":
            assert random.getstate() != rng
    finally:
        random.setstate(rng)


@pytest.mark.parametrize("minimum_change", [False, True])
@pytest.mark.parametrize("kind", ["head", "model", "head_policy", "state_policy"])
def test_should_refuse_equal_valued_nested_owner_replacement(kind, minimum_change, monkeypatch):
    source = owner(warm=True)
    before = source.snapshot_state()
    replacement = []
    name = "update_categorical_linear" if minimum_change else "fit_categorical_linear"
    original = getattr(linear, name)

    def operation(*args):
        result = original(*args)
        if kind == "head":
            new = deepcopy(source._state.heads[1])
            object.__setattr__(source._state, "heads", (source._state.heads[0], new))
        elif kind == "model":
            new = deepcopy(source._state.heads[1].model)
            object.__setattr__(source._state.heads[1], "model", new)
        elif kind == "head_policy":
            new = replace(source._state.heads[1].policy)
            object.__setattr__(source._state.heads[1], "policy", new)
        else:
            new = replace(source._state.policy)
            object.__setattr__(source._state, "policy", new)
        replacement.append(new)
        return result

    monkeypatch.setattr(linear, name, operation)
    method = source.train_batch_minimum_change if minimum_change else source.train_batch
    with pytest.raises(TrainingUpdateFailed, match="changed") as failed:
        method(*packet())
    assert source.snapshot_state() == before and replacement
    assert (
        failed.value.work["private_candidate_commits"] == 1
        and failed.value.work["candidate_commits"] == 0
    )
    current = {
        "head": source._state.heads[1],
        "model": source._state.heads[1].model,
        "head_policy": source._state.heads[1].policy,
        "state_policy": source._state.policy,
    }[kind]
    assert current is replacement[0]  # failure never restores away foreign ownership


class ExplicitContextPort(context.ContextLinearFitLearner):
    """Only test composition chooses the new operation; no production mode."""

    def train_batch(self, features, targets):
        return self.train_batch_minimum_change(features, targets)

    def fork(self):
        child = ExplicitContextPort(self.policy)
        child.restore_state(self.snapshot_state())
        return child


def stream_for(source, role="train"):
    sequence = (
        (0, 1.0, 1.0),
        (1, 1.0, 0.0),
        (0, -1.0, 0.0),
        (1, -1.0, 1.0),
        (0, 1.0, 1.0),
        (1, 1.0, 0.0),
    )
    packets = tuple(
        AttributionBatch(role, f"p-{i}", (f"row-{i}",), *packet(index, sign, label), num_classes=2)
        for i, (index, sign, label) in enumerate(sequence)
    )
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=6), lambda: 0.0)
    return AttributionStream(
        source,
        packets,
        (),
        checkpoint_updates=(0, 6),
        label_delay=1,
        budget=budget,
        training_only=True,
    ), budget


def test_should_use_existing_delayed_train_stream_and_keep_source_actor():
    source = owner(cls=ExplicitContextPort)
    before = source.snapshot_state()
    stream, budget = stream_for(source)
    stream.run_until(6)
    state = stream.candidate_snapshot()
    report = stream.finish()
    assert state.completed_updates == state.training_examples == budget.updates_completed == 6
    assert report["score_forward_calls"] == 0 and [h.completed_updates for h in state.heads] == [
        3,
        3,
    ]
    assert source.snapshot_state() == stream._runtime.actor.snapshot_state().state == before
    reader = source.fork()
    reader.restore_state(state)
    x = np.vstack((packet(0, 1.0)[0], packet(0, -1.0)[0], packet(1, 1.0)[0], packet(1, -1.0)[0]))
    assert (reader.predict(x)[:, 0] > 0.5).astype(int).tolist() == [1, 0, 0, 1]
    assert all(r.observed_at + 1 == r.arrived_at <= r.applied_at for r in stream.applied_updates)


@pytest.mark.parametrize("role", ["probe_train", "final_test"])
def test_should_keep_nontrain_roles_out_of_ordinary_stream(role, monkeypatch):
    monkeypatch.setattr(linear, "update_categorical_linear", lambda *a: pytest.fail("solver"))
    with pytest.raises(ValueError):
        stream_for(owner(warm=True, cls=ExplicitContextPort), role)


def test_should_stop_existing_stream_after_lost_routed_update(monkeypatch):
    source = owner(warm=True, cls=ExplicitContextPort)
    before = source.snapshot_state()
    stream, budget = stream_for(source)
    original = linear.update_categorical_linear
    attempts = []

    def lost(*args):
        attempts.append(1)
        original(*args)
        raise RuntimeError("lost routed residual")

    monkeypatch.setattr(linear, "update_categorical_linear", lost)
    with pytest.raises(TrainingUpdateFailed) as failed:
        stream.run_until(6)
    assert attempts == [1] and budget.updates_completed == 0 and stream._runtime._stopped
    assert failed.value.work["context_index"] == 0 and failed.value.work["unreturned_work_unknown"]
    assert source.snapshot_state() == stream._runtime.actor.snapshot_state().state == before
    assert stream._runtime._candidate.snapshot_state() == before
    with pytest.raises(ValueError):
        stream.run_until(6)
    assert attempts == [1]
