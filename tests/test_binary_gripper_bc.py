"""Independent mixed-loss oracles on synthetic data; no robot or adaptation claims."""

from dataclasses import replace
import math

import numpy as np
import pytest

from src.adapters import numpy_continuous_bc as adapter
from src.core import continuous_bc as core
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.learner_ports import TrainingUpdateFailed
from src.core.observation_standardization import ObservationStandardization


def spec():
    return ContinuousActionSpec(
        2, ("m", "normalized_gripper", "fixed"), (-2.0, -1.0, 4.0), (6.0, 1.0, 4.0)
    )


def arrays():
    return (
        np.array([[0.2, -0.3], [0.7, 0.4]], dtype=np.float64),
        np.array([[1.0, -1.0, 4.0], [4.0, 1.0, 4.0]], dtype=np.float64),
    )


def parameters():
    return (
        np.array([[0.1, -0.2], [0.3, 0.15]]),
        np.array([0.02, -0.03]),
        np.array([[0.2, -0.1, 0.12], [-0.3, 0.25, -0.15]]),
        np.array([0.05, -0.04, 0.07]),
    )


def oracle_loss(features, targets, weights):
    first, bias_first, second, bias_second = weights
    logits = np.tanh(features @ first + bias_first) @ second + bias_second
    position_error = np.tanh(logits[:, 0]) - (targets[:, 0] - 2.0) / 4.0
    grip_loss = np.logaddexp(0.0, -2.0 * targets[:, 1] * logits[:, 1])
    return float(np.sum(0.5 * position_error**2 + grip_loss) / (2 * len(features)))


def learner(axis=1, transform=None, role="train"):
    return adapter.NumpyContinuousBCLearner(
        adapter.ContinuousBCPolicy(
            spec(),
            hidden_dim=2,
            learning_rate=0.03,
            momentum=0.8,
            training_role=role,
            observation_standardization=transform,
            binary_gripper_axis=axis,
        )
    )


def episode(role="train"):
    return ContinuousEpisode(
        "synthetic",
        "grip",
        role,
        spec(),
        (
            ContinuousTransition(0, (0.2, -0.3), (1.0, -1.0, 4.0), 0.0, (0.7, 0.4), False, False),
            ContinuousTransition(1, (0.7, 0.4), (4.0, 1.0, 4.0), 0.0, (1.0, 0.5), True, False),
        ),
    )


def assert_same_state(first, second):
    assert first.policy == second.policy
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    for group in ("parameters", "velocity"):
        for left, right in zip(getattr(first, group), getattr(second, group), strict=True):
            np.testing.assert_array_equal(left, right)


def test_should_match_independent_mixed_objective_and_every_parameter_finite_difference():
    features, targets = arrays()
    weights = parameters()
    result = core.continuous_bc_loss_and_gradients(
        features, targets, weights, spec(), binary_gripper_axis=1
    )
    assert result.loss == pytest.approx(oracle_loss(features, targets, weights), abs=1e-15)
    for block, gradient in enumerate(result.gradients):
        finite_difference = np.zeros_like(gradient)
        for index in np.ndindex(gradient.shape):
            plus, minus = (
                tuple(value.copy() for value in weights),
                tuple(value.copy() for value in weights),
            )
            plus[block][index] += 1e-6
            minus[block][index] -= 1e-6
            finite_difference[index] = (
                oracle_loss(features, targets, plus) - oracle_loss(features, targets, minus)
            ) / 2e-6
        np.testing.assert_allclose(gradient, finite_difference, rtol=3e-6, atol=2e-9)
    np.testing.assert_array_equal(result.gradients[2][:, 2], [0.0, 0.0])
    assert result.gradients[3][2] == 0.0


def test_should_match_hand_zero_logit_loss_with_constant_axis_excluded():
    weights = tuple(np.zeros_like(value) for value in parameters())
    targets = np.array([[-2.0, -1.0, 4.0], [6.0, 1.0, 4.0]])
    result = core.continuous_bc_loss_and_gradients(
        np.zeros((2, 2)), targets, weights, spec(), binary_gripper_axis=1
    )
    assert result.loss == pytest.approx((0.5 + math.log(2.0)) / 2)
    np.testing.assert_array_equal(result.predictions, [[2.0, 0.0, 4.0]] * 2)


@pytest.mark.parametrize("label", [-1.0, 1.0])
def test_should_include_factor_two_in_binary_logit_gradient(label):
    weights = tuple(np.zeros_like(value) for value in parameters())
    result = core.continuous_bc_loss_and_gradients(
        np.zeros((1, 2)),
        np.array([[2.0, label, 4.0]]),
        weights,
        spec(),
        binary_gripper_axis=1,
    )
    assert result.loss == pytest.approx(math.log(2.0) / 2)
    assert result.gradients[3][1] == -label / 2


@pytest.mark.parametrize("logit", [-1000.0, 1000.0])
@pytest.mark.parametrize("correct", [False, True])
def test_should_keep_saturated_binary_loss_and_gradient_stable_without_tanh_gradient(
    logit, correct
):
    weights = tuple(np.zeros_like(value) for value in parameters())
    weights[3][1] = logit
    sign = 1.0 if logit > 0 else -1.0
    label = sign if correct else -sign
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = core.continuous_bc_loss_and_gradients(
            np.zeros((1, 2)),
            np.array([[2.0, label, 4.0]]),
            weights,
            spec(),
            binary_gripper_axis=1,
        )
    assert result.predictions[0, 1] == sign
    assert result.loss == (0.0 if correct else 1000.0)
    assert result.gradients[3][1] == (0.0 if correct else sign)


def test_should_leave_default_mse_and_prediction_arrays_exactly_unchanged():
    features, targets = arrays()
    weights = parameters()
    implicit = core.continuous_bc_loss_and_gradients(features, targets, weights, spec())
    explicit = core.continuous_bc_loss_and_gradients(
        features, targets, weights, spec(), binary_gripper_axis=None
    )
    assert implicit.loss == explicit.loss
    for left, right in zip(implicit.gradients, explicit.gradients, strict=True):
        np.testing.assert_array_equal(left, right)
    mixed = core.continuous_bc_loss_and_gradients(
        features, targets, weights, spec(), binary_gripper_axis=1
    )
    np.testing.assert_array_equal(implicit.predictions, mixed.predictions)
    np.testing.assert_array_equal(
        mixed.predictions, core.continuous_bc_predict(features, weights, spec())
    )
    default, binary = learner(None), learner()
    first, second = default.snapshot_state(), binary.snapshot_state()
    for name in ("parameters", "velocity"):
        for left, right in zip(getattr(first, name), getattr(second, name), strict=True):
            np.testing.assert_array_equal(left, right)
    assert first.completed_updates == second.completed_updates == 0
    assert core.MIXED_OBJECTIVE != core.OBJECTIVE


@pytest.mark.parametrize("axis", [-1, 3, True, 1.0, np.int64(1), "1"])
def test_should_refuse_non_native_or_invalid_binary_axis_in_core_and_policy(axis):
    features, targets = arrays()
    with pytest.raises(ValueError):
        core.continuous_bc_loss_and_gradients(
            features, targets, parameters(), spec(), binary_gripper_axis=axis
        )
    with pytest.raises(ValueError):
        learner(axis)


@pytest.mark.parametrize("bad", ["unit", "lower", "upper", "constant"])
def test_should_refuse_binary_axis_with_wrong_declared_controller_contract(bad):
    declared = spec()
    if bad == "unit":
        declared = replace(declared, action_units=("m", "N", "fixed"))
    elif bad == "lower":
        declared = replace(declared, lower_bounds=(-2.0, -2.0, 4.0))
    elif bad == "upper":
        declared = replace(declared, upper_bounds=(6.0, 2.0, 4.0))
    else:
        declared = replace(declared, lower_bounds=(-2.0, 1.0, 4.0))
    with pytest.raises(ValueError):
        adapter.ContinuousBCPolicy(declared, binary_gripper_axis=1)


@pytest.mark.parametrize("label", [0.0, 0.5, -1.0 + 1e-15, np.nan, np.inf])
def test_should_refuse_non_endpoint_targets_before_adapter_gradient_attempt(monkeypatch, label):
    model = learner()
    before = model.snapshot_state()
    features, targets = arrays()
    targets[0, 1] = label
    monkeypatch.setattr(
        core,
        "continuous_bc_loss_and_gradients",
        lambda *_args, **_kwargs: pytest.fail("invalid target reached gradient attempt"),
    )
    with pytest.raises(ValueError):
        model.train_batch(features, targets)
    assert_same_state(before, model.snapshot_state())


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_should_refuse_wrong_role_before_binary_gradient_attempt(monkeypatch, role):
    model = learner()
    before = model.snapshot_state()
    monkeypatch.setattr(
        core,
        "continuous_bc_loss_and_gradients",
        lambda *_args, **_kwargs: pytest.fail("wrong role reached gradient attempt"),
    )
    with pytest.raises(ValueError):
        model.train_episode(episode(role))
    assert_same_state(before, model.snapshot_state())


def test_should_preserve_failed_overflow_work_and_entire_state_atomically():
    model = learner()
    state = model.snapshot_state()
    weights = tuple(np.zeros_like(value) for value in state.parameters)
    weights[3][1] = 1e308
    model.restore_state(replace(state, parameters=weights))
    before = model.snapshot_state()
    with pytest.raises(TrainingUpdateFailed) as caught:
        model.train_batch(np.zeros((1, 2)), np.array([[2.0, -1.0, 4.0]]))
    assert caught.value.definition == core.MIXED_OBJECTIVE
    assert caught.value.work == dict(
        gradient_evaluations_attempted=1,
        gradient_evaluations_completed=0,
        parameter_proposals_attempted=0,
        parameter_proposals_completed=0,
        published_updates=0,
    )
    assert_same_state(before, model.snapshot_state())


def test_should_use_frozen_normalizer_and_distinct_training_name_but_common_read_only_diagnostics():
    transform = ObservationStandardization((10.0, 100.0), (2.0, 4.0), 2)
    model, manual = learner(transform=transform), learner()
    raw = np.array([[10.4, 98.8], [11.4, 101.6]])
    features, targets = arrays()
    observed, expected = model.train_batch(raw, targets), manual.train_batch(features, targets)
    assert observed.definition == expected.definition == core.MIXED_OBJECTIVE
    assert observed.value == pytest.approx(expected.value, rel=1e-13)
    for group in ("parameters", "velocity"):
        for left, right in zip(
            getattr(model.snapshot_state(), group),
            getattr(manual.snapshot_state(), group),
            strict=True,
        ):
            np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-14)
    plain = learner()
    plain.train_episode(episode())
    before = plain.snapshot_state()
    predictions = plain.predict(features)
    report = plain.diagnose_training_episode(episode(), ("open", "closed"))
    errors = np.column_stack(
        ((predictions[:, 0] - targets[:, 0]) / 4, predictions[:, 1] - targets[:, 1])
    )
    assert report.residuals.overall.normalized_half_mse == pytest.approx(
        float(np.mean(errors**2) / 2)
    )
    assert_same_state(before, plain.snapshot_state())


def test_should_preserve_full_axis_config_and_momentum_across_fork_restore_continuation():
    model = learner()
    model.train_episode(episode())
    snapshot = model.snapshot_state()
    restored, forked = learner(), model.fork()
    restored.restore_state(snapshot)
    object.__setattr__(snapshot.policy, "binary_gripper_axis", None)
    snapshot.velocity[0].fill(999.0)
    assert (
        model.policy.binary_gripper_axis
        == restored.policy.binary_gripper_axis
        == forked.policy.binary_gripper_axis
        == 1
    )
    for item in (model, restored, forked):
        item.train_episode(episode())
    assert_same_state(model.snapshot_state(), restored.snapshot_state())
    assert_same_state(model.snapshot_state(), forked.snapshot_state())
    before = model.snapshot_state()
    with pytest.raises(ValueError):
        model.restore_state(learner(None).snapshot_state())
    assert_same_state(before, model.snapshot_state())
