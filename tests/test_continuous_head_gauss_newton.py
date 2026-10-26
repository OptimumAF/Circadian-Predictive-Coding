"""Independent head-only Gauss-Newton arithmetic and atomic BP integration.

Prospective:7 real least-squares solves/21 augmented rows plus4 patched solver
attempts;20 numerical linearizations/29 source rows. One initializer draws2
normals;4 one-row GN attempts and1 ordinary one-row SGD. No stats fit/native.
"""

from dataclasses import replace
import math
from typing import Any, cast

import numpy as np
import pytest

from src.adapters import numpy_continuous_bc as adapter
from src.core import continuous_bc as task
from src.core import continuous_head_gauss_newton as core
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.learner_ports import TrainingUpdateFailed


def spec(actions=1):
    return ContinuousActionSpec(
        1,
        ("normalized_gripper",) if actions == 1 else ("a", "b"),
        (-1.0,) * actions,
        (1.0,) * actions,
    )


def parameters(actions=1, *, bias=0.0):
    return cast(
        adapter.Parameters,
        (np.zeros((1, 1)), np.zeros(1), np.zeros((1, actions)), np.full(actions, bias)),
    )


def assert_state(first, second):
    assert first.policy == second.policy
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    for a, b in zip(
        first.parameters + first.velocity, second.parameters + second.velocity, strict=True
    ):
        np.testing.assert_array_equal(a, b)


@pytest.fixture(scope="module")
def cached():
    declared = ContinuousActionSpec(1, ("physical_a", "physical_b"), (-2.0, 3.0), (2.0, 7.0))
    features = np.array([[-0.5], [0.75]])
    targets = np.array([[-0.5, 4.25], [0.75, 6.5]])
    params = cast(
        adapter.Parameters,
        (np.array([[0.7]]), np.array([-0.1]), np.array([[0.3, -0.4]]), np.array([0.2, 0.1])),
    )
    linear = core.linearize_continuous_head(features, targets, params, declared)
    finite_differences = []
    for axis in range(2):
        for group in (2, 3):
            plus, minus = (
                tuple(value.copy() for value in params),
                tuple(value.copy() for value in params),
            )
            plus[group].flat[axis] += 1e-6
            minus[group].flat[axis] -= 1e-6
            positive = core.linearize_continuous_head(
                features, targets, cast(adapter.Parameters, plus), declared
            )
            negative = core.linearize_continuous_head(
                features, targets, cast(adapter.Parameters, minus), declared
            )
            finite_differences.append(
                (axis, group - 2, (positive.residuals - negative.residuals) / 2e-6)
            )
    inactive_spec = ContinuousActionSpec(1, ("constant", "active"), (2.0, -1.0), (2.0, 1.0))
    inactive_params = parameters(2)
    inactive_params[2][0, 0], inactive_params[3][0] = 0.75, -0.4
    constant_spec = ContinuousActionSpec(1, ("constant",), (2.0,), (2.0,))
    cases = {
        "scalar": (np.zeros((1, 1)), np.array([[0.25]]), parameters(), spec()),
        "cap": (np.zeros((1, 1)), -np.ones((1, 2)), parameters(2, bias=2.0), spec(2)),
        "inactive": (np.zeros((1, 1)), np.array([[2.0, 0.25]]), inactive_params, inactive_spec),
        "saturated": (np.zeros((1, 1)), np.array([[-1.0]]), parameters(bias=1000.0), spec()),
        "constant": (np.zeros((1, 1)), np.array([[2.0]]), parameters(), constant_spec),
    }
    proposals = {name: core.propose_head_gauss_newton(*inputs) for name, inputs in cases.items()}
    return linear, finite_differences, proposals, cases, features, targets, params


def test_should_match_every_head_weight_and_bias_jacobian_by_central_differences(cached):
    linear, differences = cached[:2]
    for axis, column, derivative in differences:
        np.testing.assert_allclose(
            linear.jacobians[axis][:, column], derivative[:, axis], atol=2e-8, rtol=2e-6
        )
        np.testing.assert_array_equal(derivative[:, 1 - axis], [0.0, 0.0])


def test_should_scale_residuals_and_jacobians_by_active_normalized_axes_not_physical_width(cached):
    linear, _, _, _, features, targets, _ = cached
    hidden = np.tanh(0.7 * features[:, 0] - 0.1)
    normalized = np.tanh(hidden[:, None] * np.array([0.3, -0.4]) + np.array([0.2, 0.1]))
    expected = (normalized - (targets - np.array([0.0, 5.0])) / 2.0) / 2.0
    np.testing.assert_allclose(linear.residuals, expected, atol=1e-14)
    np.testing.assert_allclose(linear.design, np.column_stack((hidden, np.ones(2))))
    for axis in range(2):
        np.testing.assert_allclose(
            linear.jacobians[axis], linear.design * ((1 - normalized[:, axis] ** 2) / 2)[:, None]
        )
    assert linear.loss == pytest.approx(0.5 * float(np.sum(expected**2)))


def test_should_match_the_hand_damped_scalar_bias_step_and_original_normalized_loss(cached):
    proposal = cached[2]["scalar"]
    delta = 0.25 / (1 + 0.001)
    assert core.DAMPING == 0.001 and core.MAX_STEP_NORM == 1.0 and core.RCOND == 1e-12
    assert proposal.accepted is True
    np.testing.assert_allclose(proposal.head_weights, [[0.0]], atol=1e-14)
    np.testing.assert_allclose(proposal.head_bias, [delta], atol=1e-14)
    assert proposal.loss_before == pytest.approx(0.5 * 0.25**2)
    assert proposal.loss_after == pytest.approx(0.5 * (math.tanh(delta) - 0.25) ** 2)
    assert proposal.step_norm == pytest.approx(delta)
    assert proposal.work["solve_attempts"] == proposal.work["solve_completed"] == 1
    assert proposal.work["solve_rows"] == 3 and proposal.work["solve_columns"] == 2
    assert proposal.work["solve_input_values"] == 9
    assert proposal.work["solver_work_known"] is False


def test_should_cap_the_joint_head_step_globally_instead_of_capping_each_axis(cached):
    proposal = cached[2]["cap"]
    assert proposal.accepted is True and proposal.loss_after < proposal.loss_before
    realized = np.concatenate((proposal.head_weights.ravel(), proposal.head_bias - 2.0))
    assert float(np.linalg.norm(realized)) == pytest.approx(1.0)
    assert proposal.step_norm == pytest.approx(1.0)
    assert proposal.work["solve_attempts"] == proposal.work["solve_completed"] == 2
    np.testing.assert_allclose(proposal.head_bias, [2 - 1 / math.sqrt(2)] * 2, atol=1e-12)


def test_should_leave_constant_axis_head_columns_unchanged_while_fitting_only_active_axes(cached):
    proposal = cached[2]["inactive"]
    assert proposal.accepted is True
    assert proposal.head_weights[0, 0] == 0.75 and proposal.head_bias[0] == -0.4
    np.testing.assert_array_equal(proposal.predictions_after[:, 0], [2.0])


def test_should_reject_a_saturated_zero_jacobian_without_publishing_an_alternative(cached):
    proposal = cached[2]["saturated"]
    assert proposal.accepted is False and proposal.loss_before == proposal.loss_after == 2.0
    np.testing.assert_array_equal(proposal.head_bias, [1000.0])
    np.testing.assert_array_equal(proposal.predictions_before, proposal.predictions_after)


def test_should_skip_every_solve_when_all_axes_are_constant(cached):
    proposal = cached[2]["constant"]
    assert proposal.accepted is False and proposal.loss_before == proposal.loss_after == 0.0
    np.testing.assert_array_equal(proposal.predictions_after, [[2.0]])
    np.testing.assert_array_equal(proposal.head_weights, [[0.0]])
    np.testing.assert_array_equal(proposal.head_bias, [0.0])
    assert proposal.work["solve_attempts"] == proposal.work["solve_completed"] == 0


def test_should_own_proposals_and_linearizations_without_aliasing_source_arrays(cached):
    linear, _, proposals, cases = cached[:4]
    proposal = proposals["scalar"]
    inputs = cases["scalar"]
    for output in (
        proposal.head_weights,
        proposal.head_bias,
        proposal.predictions_before,
        proposal.predictions_after,
    ):
        assert not any(np.shares_memory(output, value) for value in (*inputs[:2], *inputs[2]))
    original = proposal.head_bias.copy()
    try:
        proposal.head_bias.fill(99.0)
        np.testing.assert_array_equal(inputs[2][3], [0.0])
    finally:
        proposal.head_bias[:] = original
    assert not np.shares_memory(linear.residuals, cached[5])


def test_should_reject_a_finite_loss_increasing_solver_proposal_and_keep_the_original_head(
    monkeypatch,
):
    def wrong(design, rhs, *, rcond):
        delta = np.zeros((design.shape[1],) + rhs.shape[1:])
        delta[-1] = -0.5
        return delta, np.array([0.0]), np.int64(design.shape[1]), np.ones(design.shape[1])

    monkeypatch.setattr(np.linalg, "lstsq", wrong)
    proposal = core.propose_head_gauss_newton(
        np.zeros((1, 1)), np.array([[0.25]]), parameters(), spec()
    )
    assert proposal.accepted is False and proposal.trial_loss is not None
    assert proposal.trial_loss > proposal.loss_before
    assert proposal.loss_after == proposal.loss_before
    np.testing.assert_array_equal(proposal.head_weights, [[0.0]])
    np.testing.assert_array_equal(proposal.head_bias, [0.0])
    np.testing.assert_array_equal(proposal.predictions_after, proposal.predictions_before)


def test_should_raise_a_typed_failure_with_owned_paid_solver_work(monkeypatch):
    calls = []

    def fail(design, rhs, *, rcond):
        calls.append((design.shape, rhs.shape))
        raise np.linalg.LinAlgError("one paid axis solve failed")

    monkeypatch.setattr(np.linalg, "lstsq", fail)
    with pytest.raises(core.HeadGaussNewtonFailed) as captured:
        core.propose_head_gauss_newton(np.zeros((1, 1)), np.array([[0.25]]), parameters(), spec())
    assert len(calls) == 1 and calls[0][0] == (3, 2)
    paid = captured.value.work
    assert paid["solve_attempts"] == 1 and paid["solve_completed"] == 0
    assert paid["solve_rows"] == 3 and paid["solve_columns"] == 2
    assert paid["failure_work_may_be_unobserved"] is True
    paid["caller_only"] = 99
    assert "caller_only" not in captured.value.work


@pytest.mark.parametrize("fault", ["dtype", "shape", "nonfinite", "bounds", "parameters"])
def test_should_refuse_invalid_inputs_before_a_linear_solve(monkeypatch, fault):
    features, targets, params = np.zeros((1, 1)), np.zeros((1, 1)), parameters()
    if fault == "dtype":
        features = features.astype(np.float32)
    elif fault == "shape":
        features = np.zeros((1, 2))
    elif fault == "nonfinite":
        features[0, 0] = np.inf
    elif fault == "bounds":
        targets[0, 0] = 2.0
    else:
        params[0][0, 0] = np.nan

    def forbidden(*args, **kwargs):
        pytest.fail("invalid input reached a solver")

    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    with pytest.raises(ValueError):
        core.propose_head_gauss_newton(features, targets, params, spec())


@pytest.fixture(scope="module")
def source():
    owner = adapter.NumpyContinuousBCLearner(
        adapter.ContinuousBCPolicy(spec(), hidden_dim=1, seed=17)
    )
    initial = owner.snapshot_state()
    velocity = cast(
        adapter.Parameters,
        (np.array([[0.3]]), np.array([-0.2]), np.array([[0.4]]), np.array([0.5])),
    )
    owner.restore_state(
        replace(
            initial,
            parameters=parameters(),
            velocity=velocity,
            completed_updates=3,
            training_examples=7,
        )
    )
    return owner


@pytest.fixture(scope="module")
def updates(source):
    results = {}
    for name in ("accepted", "saturated", "failure"):
        owner = source.fork()
        if name == "saturated":
            owner.restore_state(replace(owner.snapshot_state(), parameters=parameters(bias=1000.0)))
        before = owner.snapshot_state()
        with pytest.MonkeyPatch.context() as patch:
            if name == "failure":

                def fail(*args, **kwargs):
                    raise np.linalg.LinAlgError("learner paid solve failure")

                patch.setattr(np.linalg, "lstsq", fail)
            try:
                returned: Any
                if name == "accepted":
                    episode = ContinuousEpisode(
                        "one",
                        "context",
                        "train",
                        spec(),
                        (ContinuousTransition(0, (0.0,), (0.25,), 1.0, (0.0,), True, False),),
                    )
                    returned = owner.train_head_gauss_newton_episode(episode)
                else:
                    returned = owner.train_head_gauss_newton_batch(
                        np.zeros((1, 1)), np.array([[-1.0 if name == "saturated" else 0.25]])
                    )
            except TrainingUpdateFailed as error:
                returned = error
        results[name] = owner, before, returned
    return results


def test_should_commit_only_the_head_reset_only_head_momentum_and_charge_rows_once(updates, source):
    owner, before, returned = updates["accepted"]
    assert not isinstance(returned, TrainingUpdateFailed)
    after = owner.snapshot_state()
    assert returned.diagnostic.definition == task.OBJECTIVE
    assert returned.diagnostic.value == pytest.approx(0.5 * 0.25**2)
    assert returned.loss_after < returned.diagnostic.value
    assert after.completed_updates == before.completed_updates + 1
    assert after.training_examples == before.training_examples + 1
    for index in (0, 1):
        np.testing.assert_array_equal(after.parameters[index], before.parameters[index])
        np.testing.assert_array_equal(after.velocity[index], before.velocity[index])
    np.testing.assert_array_equal(after.velocity[2], [[0.0]])
    np.testing.assert_array_equal(after.velocity[3], [0.0])
    np.testing.assert_allclose(after.parameters[3], [0.25 / 1.001], atol=1e-14)
    assert returned.work["published_updates"] == 1
    assert_state(before, source.snapshot_state())


def test_should_refuse_a_nonimproving_learner_step_without_changing_any_state(updates):
    owner, before, failure = updates["saturated"]
    assert isinstance(failure, TrainingUpdateFailed)
    assert failure.definition == task.OBJECTIVE
    assert failure.work["published_updates"] == 0
    assert_state(before, owner.snapshot_state())


def test_should_preserve_full_state_and_paid_work_when_the_learner_solver_fails(updates):
    owner, before, failure = updates["failure"]
    assert isinstance(failure, TrainingUpdateFailed)
    assert "paid solve" in str(failure)
    assert failure.work["published_updates"] == 0
    assert failure.work["solve_attempts"] == 1 and failure.work["solve_completed"] == 0
    assert_state(before, owner.snapshot_state())


def test_should_keep_deliberate_source_corruption_visible_when_a_patched_proposal_raises(
    source, monkeypatch
):
    owner = source.fork()
    before = owner.snapshot_state()

    def corrupt(*args):
        owner._state.parameters[0][0, 0] += 0.125
        raise RuntimeError("synthetic proposal callback failure")

    monkeypatch.setattr(core, "propose_head_gauss_newton", corrupt)
    with pytest.raises(TrainingUpdateFailed, match="changed its source") as captured:
        owner.train_head_gauss_newton_batch(np.zeros((1, 1)), np.array([[0.25]]))
    assert captured.value.work["published_updates"] == 0
    assert captured.value.work["normalization_completed"] == 1
    assert owner.snapshot_state().parameters[0][0, 0] == before.parameters[0][0, 0] + 0.125
    assert_state(before, source.snapshot_state())


def test_should_preserve_the_original_sgd_and_momentum_update_equations(source):
    owner = source.fork()
    before = owner.snapshot_state()
    returned = owner.train_batch(np.zeros((1, 1)), np.array([[0.25]]))
    assert returned.definition == task.OBJECTIVE
    after = owner.snapshot_state()
    expected_velocity = (0.27, -0.18, 0.36, 0.2)
    for parameter, velocity, expected in zip(
        after.parameters, after.velocity, expected_velocity, strict=True
    ):
        np.testing.assert_allclose(velocity, expected, atol=1e-14)
        np.testing.assert_allclose(parameter, -0.01 * expected, atol=1e-14)
    assert after.completed_updates == 4 and after.training_examples == 8
    assert_state(before, source.snapshot_state())


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "inner_guard", "final_test"])
def test_should_refuse_nontrain_episode_roles_before_head_work(source, monkeypatch, role):
    episode = ContinuousEpisode(
        "one",
        "context",
        role,
        spec(),
        (ContinuousTransition(0, (0.0,), (0.25,), 1.0, (0.0,), True, False),),
    )
    owner = source.fork()
    before = owner.snapshot_state()
    monkeypatch.setattr(
        core, "propose_head_gauss_newton", lambda *args: pytest.fail("heldout head work")
    )
    with pytest.raises(ValueError):
        owner.train_head_gauss_newton_episode(episode)
    assert_state(before, owner.snapshot_state())


@pytest.mark.parametrize("fault", ["probe_policy", "binary_policy", "subclass"])
def test_should_refuse_nontrain_mixed_or_foreign_learner_modes_before_head_math(
    source, monkeypatch, fault
):
    owner = source.fork()
    before = owner.snapshot_state()
    if fault == "subclass":

        class Foreign(adapter.NumpyContinuousBCLearner):
            pass

        owner.__class__ = Foreign
    else:
        policy = replace(
            owner.policy,
            **(
                {"training_role": "probe_train"}
                if fault == "probe_policy"
                else {"binary_gripper_axis": 0}
            ),
        )
        owner._policy = policy
        owner.restore_state(replace(before, policy=policy))
    monkeypatch.setattr(
        core, "propose_head_gauss_newton", lambda *args: pytest.fail("wrong mode head math")
    )
    with pytest.raises(ValueError):
        owner.train_head_gauss_newton_batch(np.zeros((1, 1)), np.array([[0.25]]))


def test_should_charge_a_nonfinite_returned_solver_without_admitting_a_trial(monkeypatch):
    def nonfinite(matrix, rhs, *, rcond):
        return (
            np.full(matrix.shape[1], np.nan),
            np.zeros(1),
            np.int64(matrix.shape[1]),
            np.ones(matrix.shape[1]),
        )

    monkeypatch.setattr(np.linalg, "lstsq", nonfinite)
    with pytest.raises(core.HeadGaussNewtonFailed) as captured:
        core.propose_head_gauss_newton(np.zeros((1, 1)), np.array([[0.25]]), parameters(), spec())
    paid = captured.value.work
    assert paid["solve_attempts"] == paid["solve_completed"] == 1
    assert paid["head_forward_attempts"] == paid["head_forward_completed"] == 1
    assert paid["parameter_proposals_completed"] == paid["accepted_proposals"] == 0
    assert paid["failure_work_may_be_unobserved"] is True
