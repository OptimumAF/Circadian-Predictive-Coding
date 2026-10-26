"""Real weighted native updates, frozen source isolation and paid late failures.

One D2/H2/A3 initializer draws10 values. The module performs23 batch attempts
(11returns) on47 attempted rows,19 gradient attempts/38rows (18returns/36rows),
19 successful forwards/38rows. Four gradient returns are manual momentum oracles.
No prediction, fit, GN, teacher or SDK. All data is synthetic TRAIN supervision.
"""

from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.adapters import numpy_continuous_bc as native
from src.core import continuous_bc as core
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.learner_ports import TrainingUpdateFailed
from src.core.observation_standardization import ObservationStandardization


def forbidden(*args, **kwargs):
    pytest.fail("weighted native fixture admitted prediction, fitting, GN or refused-input work")


def spec():
    return ContinuousActionSpec(
        2, ("m", "normalized_gripper", "fixed"), (-2.0, -1.0, 4.0), (6.0, 1.0, 4.0)
    )


def arrays():
    return np.array([[0.2, -0.3], [0.7, 0.4]]), np.array([[1.0, -1.0, 4.0], [4.0, 1.0, 4.0]])


def episode(role="train"):
    x, y = arrays()
    # Why: demonstration records require native scalars, not NumPy scalar views.
    a, b, c, d = (tuple(map(float, row)) for row in (x[0], y[0], x[1], y[1]))
    return ContinuousEpisode(
        "weights-TRAIN",
        "weights",
        role,
        spec(),
        (
            ContinuousTransition(0, a, b, 0.0, c, False, False),
            ContinuousTransition(1, c, d, 0.0, (1.0, 0.5), True, False),
        ),
    )


def assert_same(a, b):
    assert a.policy == b.policy
    assert (a.completed_updates, a.training_examples) == (b.completed_updates, b.training_examples)
    for x, y in zip(a.parameters + a.velocity, b.parameters + b.velocity, strict=True):
        np.testing.assert_array_equal(x, y)


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(
        initializations=0,
        normal_values=0,
        batch_attempts=0,
        batch_returns=0,
        batch_rows_attempted=0,
        batch_rows_returned=0,
        gradient_attempts=0,
        gradient_returns=0,
        gradient_rows_attempted=0,
        gradient_rows_returned=0,
        forward_attempts=0,
        forward_returns=0,
        forward_rows_attempted=0,
        forward_rows_returned=0,
    )
    initialize, train = (
        native.NumpyContinuousBCLearner.__init__,
        native.NumpyContinuousBCLearner.train_batch,
    )
    gradient, forward = core.continuous_bc_loss_and_gradients, core._forward

    def initialized(owner, policy):
        counts["initializations"] += 1
        counts["normal_values"] += policy.hidden_dim * (
            policy.spec.observation_dim + len(policy.spec.action_units)
        )
        initialize(owner, policy)

    def counted(original, kind):
        def call(*args, **kwargs):
            x = args[1] if kind == "batch" else args[0]
            counts[kind + "_attempts"] += 1
            counts[kind + "_rows_attempted"] += len(x)
            value = original(*args, **kwargs)
            counts[kind + "_returns"] += 1
            counts[kind + "_rows_returned"] += len(x)
            return value

        return call

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(native.NumpyContinuousBCLearner, "__init__", initialized)
        patch.setattr(native.NumpyContinuousBCLearner, "train_batch", counted(train, "batch"))
        patch.setattr(core, "continuous_bc_loss_and_gradients", counted(gradient, "gradient"))
        patch.setattr(core, "_forward", counted(forward, "forward"))
        patch.setattr(native.NumpyContinuousBCLearner, "predict", forbidden)
        patch.setattr(native.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
        patch.setattr(native, "fit_training_standardization", forbidden)
        yield counts
    if os.environ.get("WEIGHTED_BC_NATIVE_FIXTURE_LEDGER"):
        Path(os.environ["WEIGHTED_BC_NATIVE_FIXTURE_LEDGER"]).write_text(
            json.dumps(counts, sort_keys=True) + "\n"
        )
    assert counts == dict(
        initializations=1,
        normal_values=10,
        batch_attempts=23,
        batch_returns=11,
        batch_rows_attempted=47,
        batch_rows_returned=22,
        gradient_attempts=19,
        gradient_returns=18,
        gradient_rows_attempted=38,
        gradient_rows_returned=36,
        forward_attempts=19,
        forward_returns=19,
        forward_rows_attempted=38,
        forward_rows_returned=38,
    )


@pytest.fixture(scope="module")
def owner(ledger):
    model = native.NumpyContinuousBCLearner(
        native.ContinuousBCPolicy(spec(), hidden_dim=2, seed=17, learning_rate=0.03, momentum=0.8)
    )
    model.train_batch(*arrays())
    return model


@pytest.fixture
def model(owner):
    return owner.fork()


def configured(model, axis):
    model._policy = replace(model._policy, binary_gripper_axis=axis)
    model._state = replace(model._state, policy=model._policy)
    return model


@pytest.mark.parametrize("axis", [None, 1])
def test_should_keep_all_ones_update_exact_with_inherited_momentum_but_explicit_provenance(
    model, owner, axis
):
    configured(model, axis)
    original_owner = owner.snapshot_state()
    weighted = model.fork()
    before = model.snapshot_state()
    assert any(np.any(a != 0.0) for a in before.velocity)
    plain = model.train_batch(*arrays())
    result = weighted.train_batch(*arrays(), row_weights=np.ones(2))
    assert result.definition == (
        core.WEIGHTED_OBJECTIVE if axis is None else core.WEIGHTED_MIXED_OBJECTIVE
    )
    assert plain.definition == (core.OBJECTIVE if axis is None else core.MIXED_OBJECTIVE)
    assert result.value == plain.value
    assert_same(model.snapshot_state(), weighted.snapshot_state())
    assert_same(owner.snapshot_state(), original_owner)
    assert not any(
        np.shares_memory(a, b)
        for a in model._state.parameters + model._state.velocity
        for b in weighted._state.parameters + weighted._state.velocity
    )


@pytest.mark.parametrize("axis", [None, 1])
def test_should_apply_two_weighted_steps_using_frozen_normalizer_and_independent_momentum_arithmetic(
    model, owner, axis
):
    configured(model, axis)
    transform = ObservationStandardization((10.0, 100.0), (2.0, 4.0), 2)
    model._policy = replace(model._policy, observation_standardization=transform)
    model._state = replace(model._state, policy=model._policy)
    raw = np.array([[10.4, 98.8], [11.4, 101.6]])
    values = (raw - np.array(transform.means)) / np.array(transform.scales)
    targets = arrays()[1]
    weights = np.array([3.0, 1.0])
    original = (raw.copy(), targets.copy(), weights.copy())
    source = owner.snapshot_state()
    for _ in range(2):
        before = model.snapshot_state()
        result = core.continuous_bc_loss_and_gradients(
            values,
            targets,
            before.parameters,
            spec(),
            binary_gripper_axis=axis,
            row_weights=weights,
        )
        velocity = tuple(
            0.8 * a + b for a, b in zip(before.velocity, result.gradients, strict=True)
        )
        parameters = tuple(a - 0.03 * b for a, b in zip(before.parameters, velocity, strict=True))
        diagnostic = model.train_batch(raw, targets, row_weights=weights)
        after = model.snapshot_state()
        assert diagnostic.value == result.loss
        assert (after.completed_updates, after.training_examples) == (
            before.completed_updates + 1,
            before.training_examples + 2,
        )
        assert after.policy == before.policy
        for a, b in zip(after.parameters + after.velocity, parameters + velocity, strict=True):
            np.testing.assert_array_equal(a, b)
    for a, b in zip((raw, targets, weights), original, strict=True):
        np.testing.assert_array_equal(a, b)
    assert_same(owner.snapshot_state(), source)


@pytest.mark.parametrize(
    "fault",
    ["list", "dtype", "shape", "nan", "zero", "negative", "underflow", "normalized_underflow"],
)
def test_should_refuse_bad_weights_before_gradient_without_changing_state(
    model, monkeypatch, fault
):
    weights: Any = np.array([3.0, 1.0])
    features, targets = arrays()
    if fault == "list":
        weights = [3.0, 1.0]
    if fault == "dtype":
        weights = np.array([3, 1])
    if fault == "shape":
        weights = np.ones((2, 1))
    if fault == "nan":
        weights[0] = np.nan
    if fault == "zero":
        weights[0] = 0.0
    if fault == "negative":
        weights[0] = -1.0
    if fault == "underflow":
        weights = np.array([np.nextafter(0.0, 1.0), np.finfo(np.float64).max])
    if fault == "normalized_underflow":
        weights = np.array([np.nextafter(0.0, 1.0), 1.0, 1.0])
        features, targets = np.vstack((features, features[:1])), np.vstack((targets, targets[:1]))
    initial = model.snapshot_state()
    monkeypatch.setattr(core, "continuous_bc_loss_and_gradients", forbidden)
    with pytest.raises(ValueError):
        model.train_batch(features, targets, row_weights=weights)
    assert_same(model.snapshot_state(), initial)


@pytest.mark.parametrize("fault", ["unreturned", "replaced", "mutated"])
def test_should_preserve_weighted_paid_failure_and_refuse_publication_after_late_failure(
    model, monkeypatch, fault
):
    configured(model, 1)
    initial = model.snapshot_state()
    state_owner = model._state
    gradient = core.continuous_bc_loss_and_gradients

    def failed(*args, **kwargs):
        result = gradient(*args, **kwargs)
        if fault == "unreturned":
            raise RuntimeError("weighted gradient return unavailable")
        if fault == "replaced":
            model._state = replace(model._state)
        if fault == "mutated":
            model._state.parameters[3][0] += 0.125
        return result

    monkeypatch.setattr(core, "continuous_bc_loss_and_gradients", failed)
    with pytest.raises(TrainingUpdateFailed) as caught:
        model.train_batch(*arrays(), row_weights=np.array([3.0, 1.0]))
    assert caught.value.definition == core.WEIGHTED_MIXED_OBJECTIVE
    assert caught.value.work == dict(
        gradient_evaluations_attempted=1,
        gradient_evaluations_completed=int(fault != "unreturned"),
        parameter_proposals_attempted=int(fault != "unreturned"),
        parameter_proposals_completed=int(fault != "unreturned"),
        published_updates=0,
    )
    current = model.snapshot_state()
    assert (current.completed_updates, current.training_examples) == (
        initial.completed_updates,
        initial.training_examples,
    )
    if fault != "mutated":
        assert_same(current, initial)
    else:
        np.testing.assert_array_equal(
            current.parameters[3], initial.parameters[3] + np.array([0.125, 0.0, 0.0])
        )
        for a, b in zip(current.velocity, initial.velocity, strict=True):
            np.testing.assert_array_equal(a, b)
    assert (model._state is state_owner) is (fault != "replaced")


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_should_refuse_nonmatching_episode_role_before_weight_copy(model, monkeypatch, role):
    before = model.snapshot_state()
    record = episode(role)
    monkeypatch.setattr(core, "_row_weights", forbidden)
    with pytest.raises(ValueError):
        model.train_episode(record, row_weights=np.array([3.0, 1.0]))
    assert_same(model.snapshot_state(), before)


def test_should_train_weighted_matching_episode_without_retaining_weight_array(model):
    configured(model, 1)
    weights = np.array([2.0, 1.0])
    before = model.snapshot_state()
    diagnostic = model.train_episode(episode(), row_weights=weights)
    after = model.snapshot_state()
    weights.fill(100.0)
    assert diagnostic.definition == core.WEIGHTED_MIXED_OBJECTIVE
    assert (after.completed_updates, after.training_examples) == (
        before.completed_updates + 1,
        before.training_examples + 2,
    )
    assert_same(model.snapshot_state(), after)


def test_should_keep_default_episode_and_core_callback_signatures_unchanged(model, monkeypatch):
    original = core.continuous_bc_loss_and_gradients

    def loss(x, y, p, s, *, binary_gripper_axis=None):
        return original(x, y, p, s, binary_gripper_axis=binary_gripper_axis)

    batch = model.train_batch

    def trained(x, y):
        return batch(x, y)

    monkeypatch.setattr(core, "continuous_bc_loss_and_gradients", loss)
    monkeypatch.setattr(model, "train_batch", trained)
    assert model.train_episode(episode(), row_weights=None).definition == core.OBJECTIVE


def test_should_keep_real_numeric_overflow_failure_weighted_and_unpublished(model):
    configured(model, 1)
    initial = model.snapshot_state()
    parameters = tuple(np.zeros_like(a) for a in initial.parameters)
    parameters[3][1] = 1e308
    model.restore_state(replace(initial, parameters=parameters))
    before = model.snapshot_state()
    with pytest.raises(TrainingUpdateFailed) as caught:
        model.train_batch(np.zeros((2, 2)), arrays()[1], row_weights=np.array([3.0, 1.0]))
    assert caught.value.definition == core.WEIGHTED_MIXED_OBJECTIVE
    assert caught.value.work == dict(
        gradient_evaluations_attempted=1,
        gradient_evaluations_completed=0,
        parameter_proposals_attempted=0,
        parameter_proposals_completed=0,
        published_updates=0,
    )
    assert_same(model.snapshot_state(), before)
