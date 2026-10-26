"""Explicit training-only admission, continuation and terminal behavior with fakes."""

from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.app import coupled_attribution as app
from src.app.toy_execution_budget import ToyExecutionStopped
from tests.test_coupled_attribution import SpyLearner, batches, budget


def stream(source=None, *, training=None, heldout=(), **options):
    return app.AttributionStream(
        SpyLearner() if source is None else source,
        batches()[0] if training is None else training,
        heldout,
        checkpoint_updates=(0, 1, 2),
        label_delay=options.pop("label_delay", 1),
        budget=options.pop("budget", budget()),
        training_only=options.pop("training_only", True),
        **options,
    )


@pytest.mark.parametrize("mode", [0, 1, None, "true", np.bool_(True)])
def test_should_require_exact_bool_before_any_learner_call(mode):
    source = SpyLearner()
    with pytest.raises(ValueError, match="exact bool"):
        stream(source, training_only=mode)
    assert source.trace == []


class EmptyTuple(tuple):
    pass


@pytest.mark.parametrize("heldout", [[], EmptyTuple(), batches()[1]])
def test_should_refuse_any_heldout_data_or_nonexact_empty_tuple(heldout):
    source = SpyLearner()
    with pytest.raises(ValueError, match="exact empty heldout tuple"):
        stream(source, heldout=heldout)
    assert source.trace == []


def test_should_keep_default_empty_heldout_refusal_and_ordinary_result_wire():
    source = SpyLearner()
    with pytest.raises(ValueError, match="nonempty tuple"):
        stream(source, training_only=False)
    training, heldout = batches()
    ordinary = app.run_attribution_stream(
        source,
        training,
        heldout,
        checkpoint_updates=(0, 1, 2),
        label_delay=1,
        budget=budget(),
    )
    assert ordinary["score_forward_calls"] == 6
    assert len(ordinary["checkpoints"]) == 3
    assert "evaluation_scope" not in ordinary
    assert source.state[0] == 0


def invalid_training():
    training = batches()[0]
    first: Any = training[0]
    return [
        (),
        list(training),
        training * 257,
        (replace(first, role="final_test"), training[1]),
        (replace(first, features=np.full((1, 2), np.nan)), training[1]),
        (replace(first, targets=np.array([[2.0]])), training[1]),
        (replace(first, row_ids=training[1].row_ids), training[1]),
        (replace(first, features=np.ones((1, 3))), training[1]),
        (replace(first, num_classes=3), training[1]),
    ]


@pytest.mark.parametrize("training", invalid_training())
def test_should_preserve_train_role_geometry_identity_and_bounds_checks(training):
    source = SpyLearner()
    with pytest.raises(ValueError):
        stream(source, training=training)
    assert source.trace == []


@pytest.mark.parametrize("delay", [0, 2])
def test_should_finish_full_receipts_without_scorer_fork_prediction_or_final_data(
    monkeypatch, delay
):
    source, session = SpyLearner(), budget()
    observed = []

    def observe(count, state, arrived):
        observed.append((count, state.copy(), arrived))
        state.fill(999)
        if arrived is not None:
            assert arrived.role == "train"
            arrived.features.fill(999)

    def forbidden(*args, **kwargs):
        raise AssertionError("heldout scoring is unreachable")

    monkeypatch.setattr(app, "_score_checkpoints", forbidden)
    training = batches()[0]
    original = pickle.dumps(training)
    owner = stream(
        source, training=training, budget=session, label_delay=delay, observe_checkpoint=observe
    )
    actor_before = owner._runtime.actor.snapshot_state().state
    owner.run_until(2)
    before_finish = list(source.trace)
    result = owner.finish()
    assert source.trace == before_finish
    assert result["evaluation_scope"] == "training_only"
    assert result["checkpoints"] == [] and result["score_forward_calls"] == 0
    assert result["completed_updates"] == session.updates_completed == 2
    assert result["training_examples"] == 2
    assert [receipt["update_number"] for receipt in result["applied_updates"]] == [1, 2]
    assert [receipt["arrived_at"] for receipt in result["applied_updates"]] == [delay, delay + 1]
    assert [point for point, _, _ in observed] == [0, 1, 2]
    assert [state[0] for _, state, _ in observed] == [0, 1, 2]
    assert observed[0][2] is None
    assert pickle.dumps(training) == original
    assert source.state[0] == 0
    np.testing.assert_array_equal(owner._runtime.actor.snapshot_state().state, actor_before)
    assert not any(event[0] == "predict" for event in source.trace)
    for action in (owner.finish, lambda: owner.run_until(2), owner.pause):
        with pytest.raises(ValueError, match="failed or finished"):
            action()


def test_should_expose_same_explicit_mode_through_existing_runner(monkeypatch):
    source = SpyLearner()

    def forbidden(*args, **kwargs):
        raise AssertionError("scorer must not run")

    monkeypatch.setattr(app, "_score_checkpoints", forbidden)
    result = app.run_attribution_stream(
        source,
        batches()[0],
        (),
        checkpoint_updates=(0, 2),
        label_delay=1,
        budget=budget(),
        training_only=True,
    )
    assert result["evaluation_scope"] == "training_only"
    assert len(result["applied_updates"]) == 2


def test_should_resume_training_only_with_original_budget_receipts_and_owned_data():
    source, session = SpyLearner(), budget()
    training = batches()[0]
    owner = stream(source, training=training, budget=session)
    training[1].features.fill(222)
    owner.run_until(1)
    token = owner.pause()
    saved = token.snapshot_state()
    assert saved["authority"]["training_only"] is True
    saved["candidate_state"].fill(999)
    with pytest.raises(ValueError, match="paused"):
        owner.finish()
    owner.resume(token)
    assert owner._budget is session and session.updates_completed == 1
    owner.run_until(2)
    result = owner.finish()
    assert result["completed_updates"] == session.updates_completed == 2
    assert result["checkpoints"] == []
    assert not any(event[0] == "predict" for event in source.trace)
    trained = [event for event in source.trace if event[0] == "train"]
    np.testing.assert_array_equal(trained[1][2], np.array([[0.3, 0.4]]))


@pytest.mark.parametrize("boundary", ["wake", "pause", "resume", "finish"])
@pytest.mark.parametrize("mutation", ["mode", "heldout", "nonbool"])
def test_should_refuse_changed_scope_before_more_work_at_each_boundary(boundary, mutation):
    source, session = SpyLearner(), budget()
    owner = stream(source, budget=session)
    token = None
    if boundary in ("pause", "resume"):
        owner.run_until(1)
    if boundary == "resume":
        token = owner.pause()
    if boundary == "finish":
        owner.run_until(2)
    before = session.updates_completed
    if mutation == "heldout":
        owner._heldout = batches()[1]
    else:
        owner._training_only = False if mutation == "mode" else 1
    action = {
        "wake": lambda: owner.run_until(1),
        "pause": owner.pause,
        "resume": lambda: owner.resume(token),
        "finish": owner.finish,
    }[boundary]
    with pytest.raises(ValueError, match="original evaluation scope"):
        action()
    assert session.updates_completed == before
    assert not any(event[0] == "predict" for event in source.trace)


def test_should_refuse_default_stream_changed_to_training_only():
    source = SpyLearner()
    owner = stream(source, heldout=batches()[1], training_only=False)
    owner.run_until(2)
    owner._training_only = True
    with pytest.raises(ValueError, match="original evaluation scope"):
        owner.finish()
    assert not any(event[0] == "predict" for event in source.trace)


def test_should_refuse_replaced_scope_in_pause_observation_without_new_work():
    source, session = SpyLearner(), budget()
    owner = stream(source, budget=session)
    owner.run_until(1)
    token = owner.pause()
    token._snapshot["authority"]["training_only"] = False
    with pytest.raises(ValueError, match="corrupted or replaced"):
        owner.resume(token)
    assert session.updates_completed == 1
    assert not any(event[0] == "predict" for event in source.trace)


def test_should_keep_committed_charge_and_seal_after_observer_failure():
    source, session = SpyLearner(), budget()

    def failing_observer(count, state, arrived):
        if count == 1:
            raise RuntimeError("observed failure")

    owner = stream(source, budget=session, observe_checkpoint=failing_observer)
    with pytest.raises(RuntimeError, match="observed failure"):
        owner.run_until(2)
    assert owner.completed_updates == session.updates_completed == 1
    assert len(owner._runtime.applied_updates) == 1
    with pytest.raises(ValueError, match="failed or finished"):
        owner.finish()
    assert source.state[0] == 0
    assert not any(event[0] == "predict" for event in source.trace)


def test_should_keep_partial_budget_failure_without_scoring_or_budget_renewal():
    source, session = SpyLearner(), budget(1)
    owner = stream(source, budget=session)
    with pytest.raises(ToyExecutionStopped):
        owner.run_until(2)
    assert owner._budget is session and session.updates_completed == 1
    assert len(owner._runtime.applied_updates) == 1
    with pytest.raises(ValueError, match="failed or finished"):
        owner.finish()
    assert not any(event[0] == "predict" for event in source.trace)


@pytest.mark.parametrize("corruption", ["checkpoint", "receipt"])
def test_should_require_complete_receipts_and_checkpoint_capture_even_without_scores(corruption):
    source = SpyLearner()
    owner = stream(source)
    owner.run_until(2)
    if corruption == "checkpoint":
        owner._states.pop()
    else:
        owner._runtime._inbox._applied.popitem()
    with pytest.raises(
        (ValueError, RuntimeError), match="receipts or checkpoints|completed receipts"
    ):
        owner.finish()
    assert not any(event[0] == "predict" for event in source.trace)
