"""Explicit owned residual updates; analytic fixtures, no native task study."""

from copy import deepcopy
from dataclasses import replace
import random
from typing import Any

import numpy as np
import pytest

from src.adapters import linear_fit_learner as adapter
from src.core import categorical_linear_fit as core
from src.core.categorical_linear_state import LinearFitPolicy, LinearFitSnapshot, MAX_COUNT
from src.core.learner_ports import TrainingUpdateFailed
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget


def prior(*, rows=1, coefficients=((0.2, -0.1),) * 3):
    # Handwritten valid coefficients/count metadata, not executed training.
    return core.CategoricalLinearFit(2, 2, rows, coefficients, 1, (1.0,) * min(rows, 3), 0.0)


def owner(model=None, cls=adapter.LinearFitLearner):
    result = cls(LinearFitPolicy(2, 2))
    if model is not None:
        result.restore_state(LinearFitSnapshot(result.policy, model, 1, model.fit_rows))
    return result


def packet():
    return np.array([[1.0, 0.0]]), np.array([[1.0]])


def test_should_preserve_invisible_prior_and_match_independent_rank_one_oracle():
    old = prior()
    x, y = packet()
    before = deepcopy(old), x.copy(), y.copy()
    updated = core.update_categorical_linear(old, x, y)
    # z=[1,0,1], delta=z^T*([0,1]-zW)/2; middle coordinate is invisible.
    np.testing.assert_allclose(updated.coefficients, [[0, 0.5], [0.2, -0.1], [0, 0.5]], atol=1e-14)
    assert updated.coefficients[1] == old.coefficients[1]
    assert updated.rank == updated.fit_rows == 1 and updated.squared_error < 1e-28
    assert old == before[0] and not hasattr(updated, "prior")
    np.testing.assert_array_equal(x, before[1])
    np.testing.assert_array_equal(y, before[2])


def test_should_keep_an_already_solved_batch_and_disclose_overlap_limit():
    old = prior(coefficients=((0.0, 0.0), (2.0, -3.0), (0.0, 1.0)))
    x = np.zeros((1, 2))
    unchanged = core.update_categorical_linear(old, x, np.array([[1.0]]))
    assert unchanged.coefficients == old.coefficients and unchanged.squared_error == 0
    conflicting = core.update_categorical_linear(old, x, np.array([[0.0]]))
    assert np.argmax(core.predict_categorical_linear(conflicting, x), axis=1).tolist() == [0]
    assert np.argmax(core.predict_categorical_linear(old, x), axis=1).tolist() == [1]
    assert conflicting.coefficients[1] == old.coefficients[1]


def test_should_admit_generic128_row_prior_and_packet_without_widening_wide_contract():
    source = owner(prior(rows=128))
    result = source.train_batch_minimum_change(np.zeros((128, 2)), np.ones((128, 1)))
    state = source.snapshot_state()
    assert result.definition == adapter.MINIMUM_CHANGE_RESIDUAL_ID and result.value < 1e-24
    assert state.completed_updates == 2 and state.training_examples == 256
    assert state.model is not None and state.model.fit_rows == 128
    assert state.model.coefficients[:2] == ((0.2, -0.1),) * 2
    from src.core.wide_categorical_linear_fit import (
        WideCategoricalLinearFit,
        update_wide_categorical_linear,
    )

    wide = WideCategoricalLinearFit(2, 2, 1, ((0.2, -0.1),) * 3, 1, (1.0,), 0.0)
    with pytest.raises(ValueError, match="features"):
        update_wide_categorical_linear(wide, np.zeros((17, 2)), np.zeros((17, 1)))


def test_should_update_all296_generic_coordinates_with_one_analytic_projection():
    old = core.CategoricalLinearFit(296, 2, 1, ((0.2, -0.1),) * 297, 1, (1.0,), 0.0)
    updated = core.update_categorical_linear(old, np.ones((1, 296)), np.array([[1.0]]))
    weights = np.array(old.coefficients)
    expected = weights + (np.array([0.0, 1.0]) - weights.sum(axis=0)) / 297
    np.testing.assert_allclose(updated.coefficients, expected, rtol=0, atol=1e-14)
    assert updated.feature_dim == 296 and updated.rank == 1


@pytest.mark.parametrize("fault", ["object", "wide", "metadata", "overflow"])
def test_should_refuse_invalid_core_prior_before_solver(monkeypatch, fault):
    from src.core.wide_categorical_linear_fit import WideCategoricalLinearFit

    model = prior()
    if fault == "metadata":
        object.__setattr__(model, "squared_error", False)
    elif fault == "overflow":
        model = prior(coefficients=((1e308, 1e308),) * 3)
    candidate: Any = (
        object()
        if fault == "object"
        else WideCategoricalLinearFit(2, 2, 1, model.coefficients, 1, (1.0,), 0.0)
        if fault == "wide"
        else model
    )
    monkeypatch.setattr(core, "_solve_design", lambda *a: pytest.fail("solver"))
    with pytest.raises((ValueError, FloatingPointError)):
        core.update_categorical_linear(candidate, np.ones((1, 2)), np.array([[0.0]]))


def test_should_refuse_nonfinite_backend_result_without_changing_prior(monkeypatch):
    model = prior()
    before = deepcopy(model)
    monkeypatch.setattr(
        np.linalg, "lstsq", lambda *a, **kw: (np.full((3, 2), np.nan), [], 1, np.array([1.0]))
    )
    with pytest.raises(ValueError, match="invalid coefficients"):
        core.update_categorical_linear(model, *packet())
    assert model == before


def test_should_keep_unfitted_equivalence_forks_and_default_operation_nonsticky():
    x, y = packet()
    cold, warm = owner(), owner()
    cold.train_batch(x, y)
    diagnostic = warm.train_batch_minimum_change(x, y)
    assert warm.snapshot_state() == cold.snapshot_state()
    assert diagnostic.definition == adapter.MINIMUM_CHANGE_RESIDUAL_ID
    before = warm.snapshot_state()
    child = warm.fork()
    other_x, other_y = -x, np.array([[0.0]])
    child.train_batch_minimum_change(other_x, other_y)
    assert np.argmax(child.predict_logits(np.concatenate((x, other_x))), axis=1).tolist() == [1, 0]
    assert warm.snapshot_state() == before
    child.train_batch(other_x, other_y)
    cold.train_batch(other_x, other_y)
    assert child.snapshot_state().model == cold.snapshot_state().model
    # The cold refit's old-point scores are a numerical tie; do not assert a
    # platform-dependent sign of rounding error as a task success/failure.
    np.testing.assert_allclose(child.predict_logits(x), np.zeros((1, 2)), atol=1e-14)
    assert set(child.snapshot_state().__dataclass_fields__) == {
        "policy",
        "model",
        "completed_updates",
        "training_examples",
    }


@pytest.mark.parametrize("fault", ["width", "dtype", "rows", "class", "target_dtype", "overflow"])
def test_should_refuse_invalid_admission_without_solver(fault, monkeypatch):
    source = owner(prior())
    x, y = packet()
    if fault == "width":
        x = np.zeros((1, 297))
    elif fault == "dtype":
        x = x.astype(np.float32)
    elif fault == "rows":
        x, y = np.zeros((129, 2)), np.zeros((129, 1))
    elif fault == "class":
        y[0, 0] = 2
    elif fault == "target_dtype":
        y = y.astype(np.int64)
    else:
        source.restore_state(LinearFitSnapshot(source.policy, prior(), MAX_COUNT, MAX_COUNT))
    before = source.snapshot_state()
    monkeypatch.setattr(adapter, "update_categorical_linear", lambda *a: pytest.fail("solver"))
    with pytest.raises(ValueError):
        source.train_batch_minimum_change(x, y)
    assert source.snapshot_state() == before


@pytest.mark.parametrize(
    "fault",
    [
        "lost",
        "source",
        "policy",
        "state_replace",
        "input",
        "staged",
        "prior",
        "prior_type",
        "both_priors",
        "rng",
        "return_rows",
        "return_object",
    ],
)
def test_should_refuse_foreign_changes_and_keep_paid_failure_work(fault, monkeypatch):
    source = owner(prior())
    before = source.snapshot_state()
    x, y = packet()
    original = adapter.update_categorical_linear
    numpy_rng, python_rng = deepcopy(np.random.get_state()), random.getstate()

    def update(staged_prior, features, targets):
        assert staged_prior is not source._state.model and staged_prior == before.model
        result = original(staged_prior, features, targets)
        if fault == "lost":
            raise RuntimeError("lost residual return")
        if fault == "source":
            source.restore_state(
                LinearFitSnapshot(source.policy, prior(coefficients=((0.0, 0.0),) * 3), 1, 1)
            )
        elif fault == "policy":
            source._policy = replace(source.policy)
        elif fault == "state_replace":
            source._state = deepcopy(source._state)
        elif fault == "input":
            x[0, 0] = 0.25
        elif fault == "staged":
            targets[0, 0] = 0
        elif fault in ("prior", "both_priors"):
            object.__setattr__(staged_prior, "coefficients", ((0.0, 0.0),) * 3)
            if fault == "both_priors":
                object.__setattr__(source._state.model, "coefficients", staged_prior.coefficients)
        elif fault == "prior_type":
            object.__setattr__(staged_prior, "squared_error", False)
        elif fault == "rng":
            random.seed(71)  # fault injection only, never a model/seed selection
        elif fault == "return_rows":
            object.__setattr__(result, "fit_rows", 2)
        elif fault == "return_object":
            return object()
        return result

    monkeypatch.setattr(adapter, "update_categorical_linear", update)
    try:
        with pytest.raises(TrainingUpdateFailed) as failure:
            source.train_batch_minimum_change(x, y)
        assert failure.value.definition == adapter.MINIMUM_CHANGE_RESIDUAL_ID
        assert failure.value.work == dict(
            fit_attempts=1,
            fit_rows_attempted=1,
            fit_returns=int(fault != "lost"),
            fit_rows_returned=int(fault != "lost"),
            candidate_commits=0,
            unreturned_work_unknown=fault == "lost",
            unpublished_work=True,
        )
        if fault in ("source", "both_priors"):
            assert source.snapshot_state() != before  # foreign mutation is not erased
        else:
            assert source.snapshot_state() == before
        if fault == "input":
            assert x[0, 0] == 0.25
        if fault == "rng":
            assert random.getstate() != python_rng
    finally:
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)


class ExplicitMinimumChangePort(adapter.LinearFitLearner):
    """Test-only caller composition; production defaults have no sticky mode."""

    def train_batch(self, features, targets):
        return self.train_batch_minimum_change(features, targets)

    def fork(self):
        child = ExplicitMinimumChangePort(self.policy)
        child.restore_state(self.snapshot_state())
        return child


def stream_for(source, role="train"):
    batches = tuple(
        AttributionBatch(
            role,
            f"packet-{i}",
            (f"row-{i}",),
            np.array([[sign, 0.0]]),
            np.array([[label]]),
            num_classes=2,
        )
        for i, (sign, label) in enumerate(((1.0, 1.0), (-1.0, 0.0), (1.0, 1.0)))
    )
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=3), lambda: 0.0)
    return AttributionStream(
        source,
        batches,
        (),
        checkpoint_updates=(0, 3),
        label_delay=1,
        budget=budget,
        training_only=True,
    ), budget


def test_should_use_existing_delayed_train_stream_and_preserve_source_actor():
    source = owner(cls=ExplicitMinimumChangePort)
    before = source.snapshot_state()
    stream, budget = stream_for(source)
    stream.run_until(3)
    result = stream.finish()
    candidate = source.fork()
    candidate.restore_state(stream._runtime.candidate_snapshot().state)
    assert np.argmax(
        candidate.predict_logits(np.array([[1.0, 0.0], [-1.0, 0.0]])), axis=1
    ).tolist() == [1, 0]
    assert (
        budget.updates_completed == result["completed_updates"] == 3
        and result["training_examples"] == 3
    )
    assert source.snapshot_state() == stream._runtime.actor.snapshot_state().state == before
    assert all(r.observed_at + 1 == r.arrived_at <= r.applied_at for r in stream.applied_updates)


@pytest.mark.parametrize("role", ["probe_train", "final_test"])
def test_should_refuse_nontrain_stream_before_new_port(role, monkeypatch):
    monkeypatch.setattr(adapter, "update_categorical_linear", lambda *a: pytest.fail("solver"))
    with pytest.raises(ValueError):
        stream_for(owner(prior(), ExplicitMinimumChangePort), role)


def test_should_stop_delayed_stream_on_lost_update_without_more_attempts(monkeypatch):
    source = owner(prior(), ExplicitMinimumChangePort)
    before = source.snapshot_state()
    stream, budget = stream_for(source)
    original = adapter.update_categorical_linear
    attempted = []

    def lost(model, x, y):
        attempted.append(len(x))
        original(model, x, y)
        raise RuntimeError("lost residual return")

    monkeypatch.setattr(adapter, "update_categorical_linear", lost)
    with pytest.raises(TrainingUpdateFailed) as failure:
        stream.run_until(3)
    assert attempted == [1] and budget.updates_completed == 0
    assert failure.value.work["fit_attempts"] == 1 and failure.value.work["unreturned_work_unknown"]
    assert stream._runtime._stopped and source.snapshot_state() == before
    assert stream._runtime.actor.snapshot_state().state == before
    assert stream._runtime._candidate.snapshot_state() == before
    with pytest.raises(ValueError):
        stream.run_until(3)
    assert attempted == [1]
