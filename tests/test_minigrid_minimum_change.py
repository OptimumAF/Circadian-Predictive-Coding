"""Tiny296/K7 adapter fixtures, not learned navigation or a native study."""

from dataclasses import replace

import numpy as np
import pytest

from src.adapters import minigrid_context_linear_learner as port
from src.core.categorical_linear_fit import CategoricalLinearFit
from src.core.learner_ports import TrainingUpdateFailed
from src.core.minigrid_context_encoding import context_public_history


def inputs(cue=2):
    frame = [-0.8, -1.0, -1.0] * 49 + [-1.0]
    frame[:3] = [0.2, 2 * cue / 5 - 1, -1.0]
    return np.array([frame + frame], dtype=np.float64)


def labels(action=2):
    return np.array([[float(action)]], dtype=np.float64)


def synthetic_prior():
    """Declared fixture coefficients; not a trained source or renewed authority."""
    weights = np.zeros((297, 7), dtype=np.float64)
    weights[0, 3], weights[1, 3], weights[40, 5] = 0.75, -0.25, 0.5
    return CategoricalLinearFit(
        296, 7, 1, tuple(tuple(r) for r in weights.tolist()), 1, (1.0,), 0.0
    )


def seeded():
    learner = port.ContextLinearLearner()
    state = replace(
        learner.snapshot_state(),
        heads=(synthetic_prior(), synthetic_prior()),
        completed_updates=2,
        training_examples=2,
    )
    learner.restore_state(state)
    return learner


def test_first_minimum_change_is_the_original_cold_fit():
    incremental, cold = port.ContextLinearLearner(), port.ContextLinearLearner()
    x, y = inputs(), labels()
    result = incremental.train_batch_minimum_change(x, y)
    default = cold.train_batch(x, y)
    assert incremental.snapshot_state() == cold.snapshot_state()
    assert result.definition == port.MINIMUM_CHANGE_DIAGNOSTIC
    assert default.definition == port.DIAGNOSTIC and result.value == default.value


def test_warm_fit_matches_analytic_minimum_change_and_preserves_null_space():
    learner = seeded()
    before = learner.snapshot_state()
    x, y = inputs(), labels()
    design = np.append(np.array(context_public_history(tuple(x[0].tolist())))[:296], 1.0)
    prior = np.array(before.heads[0].coefficients)
    residual = np.eye(7)[2] - design @ prior
    expected_delta = np.outer(design / (design @ design), residual)
    # A nontrivial coefficient direction invisible to this one current row.
    null = np.zeros(297)
    null[0], null[1] = design[1], -design[0]
    result = learner.train_batch_minimum_change(x, y)
    after = learner.snapshot_state()
    actual_delta = np.array(after.heads[0].coefficients) - prior
    np.testing.assert_allclose(actual_delta, expected_delta, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(null @ actual_delta, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        design @ np.array(after.heads[0].coefficients), np.eye(7)[2], atol=1e-12
    )
    assert result.value < 1e-20 and after.heads[1] == before.heads[1]
    assert (after.completed_updates, after.training_examples, after.heads[0].fit_rows) == (3, 3, 1)


def test_incremental_operation_does_not_change_later_default_behavior():
    learner, cold = seeded(), port.ContextLinearLearner()
    learner.train_batch_minimum_change(inputs(), labels())
    diagnostic = learner.train_batch(inputs(), labels(5))
    cold.train_batch(inputs(), labels(5))
    assert learner.snapshot_state().heads[0] == cold.snapshot_state().heads[0]
    assert diagnostic.definition == port.DIAGNOSTIC


def test_forked_continuation_updates_only_selected_cue():
    source = seeded()
    before = source.snapshot_state()
    child = source.fork()
    child.train_batch_minimum_change(inputs(0), labels(5))
    after = child.snapshot_state()
    assert source.snapshot_state() == before
    assert after.heads[0] == before.heads[0] and after.heads[1] != before.heads[1]
    assert after.heads[0] is not before.heads[0]
    child.restore_state(after)
    assert child.snapshot_state() == after


@pytest.mark.parametrize(
    "fault",
    [
        "dtype",
        "width",
        "empty",
        "nan",
        "offgrid",
        "mixed",
        "target_dtype",
        "fractional",
        "range",
        "target_shape",
    ],
)
def test_invalid_incremental_batch_never_enters_a_solver(monkeypatch, fault):
    learner = seeded()
    x, y = inputs(), labels()
    if fault == "dtype":
        x = x.astype(np.float32)
    elif fault == "width":
        x = x[:, :-1]
    elif fault == "empty":
        x, y = x[:0], y[:0]
    elif fault == "nan":
        x[0, 0] = np.nan
    elif fault == "offgrid":
        x[0, 0] = 0.123
    elif fault == "mixed":
        x, y = np.concatenate([x, inputs(0)]), np.concatenate([y, y])
    elif fault == "target_dtype":
        y = y.astype(np.int64)
    elif fault == "fractional":
        y[0, 0] = 1.5
    elif fault == "range":
        y[0, 0] = 7
    else:
        y = y[:, 0]
    before = learner.snapshot_state()
    monkeypatch.setattr(
        port, "fit_categorical_linear", lambda *a: pytest.fail("cold solver reached")
    )
    monkeypatch.setattr(
        port, "update_categorical_linear", lambda *a: pytest.fail("warm solver reached")
    )
    with pytest.raises(ValueError):
        learner.train_batch_minimum_change(x, y)
    assert learner.snapshot_state() == before


@pytest.mark.parametrize(
    "fault",
    [
        "lost",
        "features",
        "staged_targets",
        "caller_targets",
        "caller_features",
        "prior",
        "prior_type",
        "state",
        "equal_state",
        "geometry",
    ],
)
def test_failed_warm_fit_preserves_paid_known_unknown_work_and_latches_owner(monkeypatch, fault):
    learner = seeded()
    before = learner.snapshot_state()
    x, y = inputs(), labels()
    original = port.update_categorical_linear

    def update(prior, features, targets):
        assert prior is not before.heads[0]
        model = original(prior, features, targets)
        if fault == "lost":
            raise RuntimeError("lost numerical return")
        if fault == "features":
            features.fill(0)
        elif fault == "staged_targets":
            targets.fill(1)
        elif fault == "caller_targets":
            y.fill(1)
        elif fault == "caller_features":
            x.fill(0)
        elif fault == "prior":
            object.__setattr__(prior, "squared_error", prior.squared_error + 1.0)
        elif fault == "prior_type":
            object.__setattr__(prior, "fit_rows", True)
        elif fault == "state":
            learner._state = replace(before, training_examples=3)
        elif fault == "equal_state":
            learner._state = replace(learner._state)
        elif fault == "geometry":
            model = replace(model, fit_rows=2, singular_values=(1.0, 0.0))
        return model

    monkeypatch.setattr(port, "update_categorical_linear", update)
    with pytest.raises(TrainingUpdateFailed) as caught:
        learner.train_batch_minimum_change(x, y)
    work = caught.value.work
    assert caught.value.definition == port.MINIMUM_CHANGE_DIAGNOSTIC
    assert work["fit_calls_attempted"] == work["fit_rows_attempted"] == 1
    assert work["fit_calls_completed"] == work["fit_rows_completed"] == int(fault != "lost")
    assert work["failure_work_may_be_unobserved"] is True
    failed = learner.snapshot_state()
    assert failed.heads == before.heads and failed.completed_updates == before.completed_updates
    assert failed.training_examples == before.training_examples
    work["fit_calls_attempted"] = 0
    assert caught.value.work["fit_calls_attempted"] == 1
    with pytest.raises(ValueError, match="paid work"):
        learner.restore_state(before)
    for owner in (learner, learner.fork()):
        for method in (owner.train_batch, owner.train_batch_minimum_change):
            with pytest.raises(TrainingUpdateFailed):
                method(inputs(), labels())


@pytest.mark.parametrize("fault", ["counter", "model"])
def test_corrupt_preceding_state_is_refused_without_new_attempt(monkeypatch, fault):
    learner = seeded()
    if fault == "counter":
        object.__setattr__(learner._state, "completed_updates", True)
    else:
        object.__setattr__(learner._state.heads[0], "coefficients", ())
    monkeypatch.setattr(port, "update_categorical_linear", lambda *a: pytest.fail("solver reached"))
    with pytest.raises(ValueError):
        learner.train_batch_minimum_change(inputs(), labels())
