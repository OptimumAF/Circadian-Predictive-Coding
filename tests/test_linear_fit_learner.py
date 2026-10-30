"""Owned current-packet fits and actual delayed-stream failure/isolation behavior."""

from copy import deepcopy
from dataclasses import replace
import random

import numpy as np
import pytest

from src.adapters import linear_fit_learner as adapter
from src.core.categorical_linear_fit import CategoricalLinearFit
from src.core.categorical_linear_state import LinearFitPolicy, LinearFitSnapshot, MAX_COUNT
from src.core.learner_ports import TrainingUpdateFailed
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget


def data():
    return np.eye(3, dtype=np.float64), np.arange(3, dtype=np.float64).reshape(3, 1)


def learner():
    return adapter.LinearFitLearner(LinearFitPolicy(3, 3))


def literal_model(rows=1):
    return CategoricalLinearFit(3, 3, rows, ((0.0, 0.0, 0.0),) * 4, 0, (0.0,) * min(rows, 4), 0.0)


def test_should_predict_uniform_unfitted_without_fitting_or_rng_draws(monkeypatch):
    owner = learner()
    x, _ = data()
    monkeypatch.setattr(adapter, "fit_categorical_linear", lambda *a: pytest.fail("fit"))
    before = owner.snapshot_state()
    assert before.model is None and before.completed_updates == before.training_examples == 0
    assert np.array_equal(owner.predict(x), np.full((3, 3), 1 / 3))
    assert np.array_equal(owner.predict_logits(x), np.zeros((3, 3)))
    assert owner.snapshot_state() == before


def test_should_replace_only_current_packet_and_own_forks_snapshots_outputs():
    owner = learner()
    x, y = data()
    initial = owner.snapshot_state()
    diagnostic = owner.train_batch(x, y)
    assert diagnostic.definition == adapter.LINEAR_RESIDUAL_ID and diagnostic.value < 1e-20
    first = owner.snapshot_state()
    assert np.argmax(owner.predict(x), axis=1).tolist() == [0, 1, 2]
    child = owner.fork()
    shifted = np.array([[2.0], [0.0], [1.0]])
    child.train_batch(x, shifted)
    state = child.snapshot_state()
    assert state.completed_updates == 2 and state.training_examples == 6
    assert state.model is not None and state.model.fit_rows == 3
    assert np.argmax(child.predict(x), axis=1).tolist() == [2, 0, 1]
    assert owner.snapshot_state() == first
    # No observations or success chain are retained in the public snapshot.
    assert set(state.__dataclass_fields__) == {
        "policy",
        "model",
        "completed_updates",
        "training_examples",
    }
    assert len(state.model.coefficients) == 4 and all(
        len(row) == 3 for row in state.model.coefficients
    )
    object.__setattr__(state.model, "squared_error", 123.0)
    assert child.snapshot_state().model != state.model
    output = child.predict(x)
    output[:] = 0
    assert np.allclose(child.predict(x).sum(axis=1), 1)
    owner.restore_state(initial)
    assert owner.snapshot_state() == initial and owner.snapshot_state().model is None


def test_should_follow_existing_binary_probability_contract():
    owner = adapter.LinearFitLearner(LinearFitPolicy(1, 2))
    x = np.array([[-1.0], [1.0]])
    y = np.array([[0.0], [1.0]])
    assert owner.predict(x).shape == (2, 1)
    owner.train_batch(x, y)
    assert (owner.predict(x)[:, 0] > 0.5).tolist() == [False, True]
    assert owner.predict_logits(x).shape == (2, 2)


@pytest.mark.parametrize(
    "fault", ["width", "dtype", "nonfinite", "target_shape", "target_class", "target_dtype"]
)
def test_should_refuse_bad_packet_before_solver_and_preserve_state(fault, monkeypatch):
    owner = learner()
    before = owner.snapshot_state()
    x, y = data()
    if fault == "width":
        x = x[:, :2]
    elif fault == "dtype":
        x = x.astype(np.float32)
    elif fault == "nonfinite":
        x[0, 0] = np.nan
    elif fault == "target_shape":
        y = y[:, 0]
    elif fault == "target_class":
        y[0, 0] = 3
    else:
        y = y.astype(np.int64)
    monkeypatch.setattr(adapter, "fit_categorical_linear", lambda *a: pytest.fail("solver"))
    with pytest.raises(ValueError):
        owner.train_batch(x, y)
    assert owner.snapshot_state() == before


@pytest.mark.parametrize(
    "fault", ["policy", "unfitted_count", "row_count", "model_geometry", "overflow"]
)
def test_should_reject_invalid_snapshot_or_policy_before_restore(fault):
    owner = learner()
    before = owner.snapshot_state()
    with pytest.raises(ValueError):
        if fault == "policy":
            owner.restore_state(LinearFitSnapshot(LinearFitPolicy(2, 3), None, 0, 0))
        elif fault == "unfitted_count":
            owner.restore_state(replace(before, completed_updates=1))
        elif fault == "row_count":
            owner.restore_state(LinearFitSnapshot(before.policy, literal_model(), 2, 1))
        elif fault == "model_geometry":
            owner.restore_state(LinearFitSnapshot(LinearFitPolicy(2, 3), literal_model(), 1, 1))
        else:
            owner.restore_state(
                LinearFitSnapshot(before.policy, literal_model(), MAX_COUNT + 1, MAX_COUNT + 1)
            )
    assert owner.snapshot_state() == before


def test_should_refuse_counter_overflow_before_solver(monkeypatch):
    owner = learner()
    owner.restore_state(LinearFitSnapshot(owner.policy, literal_model(), MAX_COUNT, MAX_COUNT))
    before = owner.snapshot_state()
    x, y = data()
    monkeypatch.setattr(adapter, "fit_categorical_linear", lambda *a: pytest.fail("solver"))
    with pytest.raises(ValueError, match="overflow"):
        owner.train_batch(x, y)
    assert owner.snapshot_state() == before


@pytest.mark.parametrize(
    "fault", ["lost", "source", "policy", "state_replace", "input", "staged", "rng"]
)
def test_should_refuse_lost_or_foreign_mutated_fit_without_erasing_changes(fault, monkeypatch):
    owner = learner()
    before = owner.snapshot_state()
    x, y = data()
    original = adapter.fit_categorical_linear
    numpy_rng = deepcopy(np.random.get_state())
    python_rng = random.getstate()
    changed = None

    def fit(features, targets, classes):
        nonlocal changed
        result = original(features, targets, classes)
        if fault == "lost":
            raise RuntimeError("lost solver return")
        if fault == "source":
            owner.restore_state(LinearFitSnapshot(owner.policy, literal_model(), 1, 1))
            changed = owner.snapshot_state()
        elif fault == "policy":
            owner._policy = replace(owner.policy)
        elif fault == "state_replace":
            owner._state = deepcopy(owner._state)
        elif fault == "input":
            x[0, 0] = 0.5
        elif fault == "staged":
            targets[0, 0] = 2
        elif fault == "rng":
            random.seed(71)
        return result

    monkeypatch.setattr(adapter, "fit_categorical_linear", fit)
    try:
        with pytest.raises(TrainingUpdateFailed, match="lost|changed") as failure:
            owner.train_batch(x, y)
        assert failure.value.definition == adapter.LINEAR_RESIDUAL_ID
        assert (
            failure.value.work["fit_attempts"] == 1
            and failure.value.work["fit_rows_attempted"] == 3
        )
        assert failure.value.work["candidate_commits"] == 0
        assert failure.value.work["fit_returns"] == int(fault != "lost")
        assert failure.value.work["unreturned_work_unknown"] == (fault == "lost")
        if fault == "source":
            assert owner.snapshot_state() == changed
        else:
            assert owner.snapshot_state() == before
        if fault == "input":
            assert x[0, 0] == 0.5
        if fault == "rng":
            assert random.getstate() != python_rng
    finally:
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)


def make_stream(owner, *, role="train", budget=None):
    x, y = data()
    packets = tuple(
        AttributionBatch(
            role,
            f"p-{i}",
            tuple(f"{i}-{j}" for j in range(3)),
            x.copy(),
            (y if i == 0 else (y + 1) % 3).copy(),
            num_classes=3,
        )
        for i in range(2)
    )
    budget = budget or ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=15), lambda: 0.0
    )
    return AttributionStream(
        owner,
        packets,
        (),
        checkpoint_updates=(0, 2),
        label_delay=1,
        budget=budget,
        training_only=True,
    ), budget


def test_should_run_two_delayed_train_updates_in_existing_stream_and_preserve_owners(monkeypatch):
    owner = learner()
    before = owner.snapshot_state()
    stream, budget = make_stream(owner)
    original = adapter.fit_categorical_linear
    observed = []

    def fit(x, y, k):
        inbox = stream._runtime._inbox
        ready = inbox._ready(inbox._last_time)
        assert len(ready) == 1
        source, label = inbox._experiences[ready[0]], inbox._labels[ready[0]]
        assert source.observed_at + 1 == label.arrived_at <= inbox._last_time
        assert source.role == label.role == "train" and source.permissions.training
        assert np.array_equal(x, source.features) and np.array_equal(y, label.targets)
        observed.append(True)
        return original(x, y, k)

    monkeypatch.setattr(adapter, "fit_categorical_linear", fit)
    stream.run_until(2)
    result = stream.finish()
    assert observed == [True, True]
    assert budget.updates_completed == result["completed_updates"] == 2
    assert result["training_examples"] == 6 and result["score_forward_calls"] == 0
    assert (
        owner.snapshot_state() == before and stream._runtime.actor.snapshot_state().state == before
    )
    assert [r.update_number for r in stream.applied_updates] == [1, 2]
    assert all(r.observed_at + 1 == r.arrived_at <= r.applied_at for r in stream.applied_updates)


@pytest.mark.parametrize("role", ["probe_train", "final_test"])
def test_should_refuse_other_roles_before_any_fit(role, monkeypatch):
    monkeypatch.setattr(adapter, "fit_categorical_linear", lambda *a: pytest.fail("fit"))
    with pytest.raises(ValueError):
        make_stream(learner(), role=role)


def test_should_stop_existing_stream_after_lost_fit_and_preserve_attempt_cost(monkeypatch):
    owner = learner()
    before = owner.snapshot_state()
    stream, budget = make_stream(owner)
    original = adapter.fit_categorical_linear
    attempts = []

    def lost(x, y, k):
        attempts.append(len(x))
        original(x, y, k)
        raise RuntimeError("lost solver return")

    monkeypatch.setattr(adapter, "fit_categorical_linear", lost)
    with pytest.raises(TrainingUpdateFailed, match="lost solver") as failure:
        stream.run_until(2)
    assert failure.value.work == dict(
        fit_attempts=1,
        fit_rows_attempted=3,
        fit_returns=0,
        fit_rows_returned=0,
        candidate_commits=0,
        unreturned_work_unknown=True,
        unpublished_work=True,
    )
    detached = failure.value.work
    detached["fit_attempts"] = 999
    assert failure.value.work["fit_attempts"] == 1
    assert attempts == [3] and budget.updates_completed == 0
    assert stream._runtime._stopped and owner.snapshot_state() == before
    assert stream._runtime.actor.snapshot_state().state == before
    assert stream._runtime._candidate.snapshot_state() == before
    with pytest.raises(ValueError):
        stream.run_until(2)
    assert attempts == [3]


def test_should_admit_existing_d9_ticket_train_packets_in_the_same_stream():
    from scripts.recurring_ticket_workflow_example import prepare_training

    spec, packets = prepare_training(context_return=True)
    owner = adapter.LinearFitLearner(LinearFitPolicy(len(spec.vocabulary), 6))
    before = owner.snapshot_state()
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=15), lambda: 0.0
    )
    stream = AttributionStream(
        owner,
        (packets[0], packets[4]),
        (),
        checkpoint_updates=(0, 2),
        label_delay=1,
        budget=budget,
        training_only=True,
    )
    stream.run_until(2)
    state = stream.candidate_snapshot()
    assert state.completed_updates == 2 and state.training_examples == 12
    assert state.model is not None and state.model.fit_rows == 6
    assert state.model.feature_dim == 9 and state.model.num_classes == 6
    assert stream.finish()["score_forward_calls"] == 0
    assert owner.snapshot_state() == stream._runtime.actor.snapshot_state().state == before
