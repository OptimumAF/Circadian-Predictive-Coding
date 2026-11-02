"""Tiny deterministic GGN proofs and owned commits; no datasets/RNG/SDK."""

from collections import defaultdict
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from src.core import continuous_bc as bc
from src.core import continuous_full_gauss_newton as gn
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.learner_ports import TrainingUpdateFailed
from src.adapters import numpy_continuous_bc as bp

WORK: defaultdict[str, int] = defaultdict(int)
CAPS = dict(
    sources=48,
    forwards=2048,
    forward_rows=8192,
    gradients=1024,
    gradient_rows=4096,
    solves=96,
    linearizations=256,
    proposals=256,
    updates=48,
    update_rows=256,
    published=48,
    snapshots=65536,
    snapshot_bytes=256 * 1024**2,
    finite_difference_coordinates=65536,
    forbidden=0,
)


def charge(key, amount=1):
    WORK[key] += amount
    assert WORK[key] <= CAPS[key], "permanent409fixturecap:" + key


def forbidden(*args, **options):
    charge("forbidden")


@pytest.fixture(autouse=True)
def meter(monkeypatch):
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "__init__", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    forward, gradient, solve, linearize, propose, update, snapshot = (
        bc._forward,
        bc.continuous_bc_loss_and_gradients,
        np.linalg.lstsq,
        gn._linearize,
        gn.propose_full_gauss_newton,
        bp.NumpyContinuousBCLearner.train_full_gauss_newton_batch,
        bp.NumpyContinuousBCLearner.snapshot_state,
    )

    def forwarded(x, *args, **kwargs):
        charge("forwards")
        charge("forward_rows", len(x))
        return forward(x, *args, **kwargs)

    def gradients(x, *args, **kwargs):
        charge("gradients")
        charge("gradient_rows", len(x))
        return gradient(x, *args, **kwargs)

    def solved(matrix, *args, **kwargs):
        charge("solves")
        assert matrix.shape[1] <= 64, "no H64/large solve in software fixtures"
        return solve(matrix, *args, **kwargs)

    def linearized(*args, **kwargs):
        charge("linearizations")
        return linearize(*args, **kwargs)

    def proposed(*args, **kwargs):
        charge("proposals")
        return propose(*args, **kwargs)

    def updated(owner, x, *args, **kwargs):
        charge("updates")
        charge("update_rows", len(x))
        result = update(owner, x, *args, **kwargs)
        charge("published", result.work["published_updates"])
        return result

    def snap(owner):
        charge("snapshots")
        value = snapshot(owner)
        charge("snapshot_bytes", sum(a.nbytes for a in value.parameters + value.velocity))
        return value

    for owner, name, value in (
        (bc, "_forward", forwarded),
        (bc, "continuous_bc_loss_and_gradients", gradients),
        (np.linalg, "lstsq", solved),
        (gn, "_linearize", linearized),
        (gn, "propose_full_gauss_newton", proposed),
        (bp.NumpyContinuousBCLearner, "train_full_gauss_newton_batch", updated),
        (bp.NumpyContinuousBCLearner, "snapshot_state", snap),
    ):
        monkeypatch.setattr(owner, name, value)


def fixture(*, axis=None):
    spec = ContinuousActionSpec(2, ("x", "y", "z", "normalized_gripper"), (-1.0,) * 4, (1.0,) * 4)
    parameters = cast(
        bc.Parameters,
        (
            np.array([[0.2, -0.1], [0.1, 0.3]]),
            np.array([0.05, -0.04]),
            np.array([[0.1, 0.2, -0.15, 0.12], [-0.2, 0.15, 0.2, -0.1]]),
            np.array([0.01, -0.02, 0.03, 0.04]),
        ),
    )
    x = np.array([[0.2, -0.3], [0.6, 0.1], [-0.4, 0.5]])
    y = np.array([[0.3, -0.2, 0.1, 1.0], [-0.3, 0.2, 0.1, -1.0], [0.0, 0.1, -0.2, 1.0]])
    return spec, parameters, x, y


def shifted(parameters, vector, scale):
    blocks, offset = [], 0
    for value in parameters:
        blocks.append(value + scale * vector[offset : offset + value.size].reshape(value.shape))
        offset += value.size
    return cast(bc.Parameters, tuple(blocks))


@pytest.mark.parametrize("axis", [None, 3])
@pytest.mark.parametrize("weighted", [False, True])
def test_full_logit_jacobian_and_original_gradient_match_finite_differences(axis, weighted):
    spec, parameters, x, y = fixture()
    weights = np.array([1.0, 2.0, 3.0]) if weighted else None
    result = gn.linearize_full_network(
        x, y, parameters, spec, binary_gripper_axis=axis, row_weights=weights
    )
    p = sum(a.size for a in parameters)
    eps = 1e-6
    for column in range(p):
        direction = np.zeros(p)
        direction[column] = 1.0
        high = bc._forward(x, shifted(parameters, direction, eps), spec)[3]
        low = bc._forward(x, shifted(parameters, direction, -eps), spec)[3]
        charge("finite_difference_coordinates", high.size)
        np.testing.assert_allclose(
            result.logit_jacobian[:, :, column], (high - low) / (2 * eps), rtol=1e-6, atol=1e-9
        )
    for direction in (np.linspace(-0.2, 0.3, p), np.cos(np.arange(p))):
        high = bc.continuous_bc_loss_and_gradients(
            x,
            y,
            shifted(parameters, direction, eps),
            spec,
            binary_gripper_axis=axis,
            row_weights=weights,
        ).loss
        low = bc.continuous_bc_loss_and_gradients(
            x,
            y,
            shifted(parameters, direction, -eps),
            spec,
            binary_gripper_axis=axis,
            row_weights=weights,
        ).loss
        assert float(result.gradient @ direction) == pytest.approx(
            (high - low) / (2 * eps), rel=1e-6, abs=1e-9
        )


def test_mixed_curvature_has_factor_two_logit_ce_and_allones_preserve_arithmetic():
    spec, parameters, x, y = fixture()
    result = gn.linearize_full_network(x, y, parameters, spec, binary_gripper_axis=3)
    ones = gn.linearize_full_network(
        x, y, parameters, spec, binary_gripper_axis=3, row_weights=np.ones(3)
    )
    np.testing.assert_array_equal(ones.gradient, result.gradient)
    np.testing.assert_array_equal(ones.curvature_weights, result.curvature_weights)
    logits = bc._forward(x, parameters, spec)[3]
    eps = 1e-4
    high = np.logaddexp(0.0, -2 * y[:, 3] * (logits[:, 3] + eps))
    base = np.logaddexp(0.0, -2 * y[:, 3] * logits[:, 3])
    low = np.logaddexp(0.0, -2 * y[:, 3] * (logits[:, 3] - eps))
    np.testing.assert_allclose(
        result.curvature_weights[:, 3], (high - 2 * base + low) / eps**2 / 12, rtol=1e-5, atol=1e-8
    )


def test_weight_scale_invariance_and_constant_axes_have_zero_curvature():
    spec, parameters, x, y = fixture()
    spec = replace(spec, lower_bounds=(-1.0, 0.0, -1.0, -1.0), upper_bounds=(1.0, 0.0, 1.0, 1.0))
    y[:, 1] = 0.0
    first = gn.linearize_full_network(x, y, parameters, spec, row_weights=np.array([1.0, 2.0, 3.0]))
    second = gn.linearize_full_network(
        x, y, parameters, spec, row_weights=np.array([1.0, 2.0, 3.0]) * 1e200
    )
    assert first.active_axes == 3
    assert np.all(first.curvature_weights[:, 1] == 0)
    np.testing.assert_allclose(first.gradient, second.gradient, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("axis", [None, 3])
def test_one_full_proposal_strictly_reduces_original_objective_and_keeps_inputs(axis):
    spec, parameters, x, y = fixture()
    before = tuple(a.copy() for a in parameters)
    result = gn.propose_full_gauss_newton(x, y, parameters, spec, binary_gripper_axis=axis)
    assert result.accepted and result.reason == "strict_improvement", result
    assert 0 < result.step_norm <= 1.0
    observed = bc.continuous_bc_loss_and_gradients(
        x, y, result.parameters, spec, binary_gripper_axis=axis
    ).loss
    assert result.loss_after == observed < result.loss_before
    assert any(
        not np.array_equal(a, b) for a, b in zip(result.parameters[:2], before[:2], strict=True)
    )
    for a, b in zip(before, parameters, strict=True):
        np.testing.assert_array_equal(a, b)
    assert result.work["gradient_requests"] == result.work["gradient_returns"] == 2
    assert result.work["solve_attempts"] == result.work["solve_returns"] == 1


def test_wrong_saturated_binary_logit_retains_gradient_with_zero_curvature():
    spec, parameters, x, y = fixture()
    parameters[3][3] = 1000.0
    y[:, 3] = -1.0
    result = gn.linearize_full_network(x, y, parameters, spec, binary_gripper_axis=3)
    assert np.all(result.curvature_weights[:, 3] == 0)
    assert result.gradient[-1] == pytest.approx(0.5)  # 2/(4active axes)
    assert np.isfinite(result.gradient).all()


def test_literal_h64_one_row_linearization_requires_no_large_solve():
    spec = ContinuousActionSpec(12, ("x", "y", "z", "normalized_gripper"), (-1.0,) * 4, (1.0,) * 4)
    parameters = (np.zeros((12, 64)), np.zeros(64), np.zeros((64, 4)), np.zeros(4))
    result = gn.linearize_full_network(
        np.zeros((1, 12)), np.array([[0.0, 0.0, 0.0, 1.0]]), parameters, spec, binary_gripper_axis=3
    )
    assert result.logit_jacobian.shape == (1, 4, 1092)
    assert result.work["solve_attempts"] == 0


@pytest.mark.parametrize("fault", ["rows", "parameters", "nonfinite", "weights", "axis", "targets"])
def test_bad_inputs_are_refused_before_original_kernel(fault):
    spec, parameters, x, y = fixture()
    options: dict[str, Any] = {}
    before = WORK["forwards"]
    if fault == "rows":
        x, y = np.zeros((221, 2)), np.zeros((221, 4))
    elif fault == "parameters":
        spec = replace(spec, observation_dim=64)
        x = np.zeros((3, 64))
        parameters = cast(
            bc.Parameters, (np.zeros((64, 64)), np.zeros(64), np.zeros((64, 4)), np.zeros(4))
        )
    elif fault == "nonfinite":
        x[0, 0] = float("nan")
    elif fault == "weights":
        options["row_weights"] = np.array([1.0, 0.0, 1.0])
    elif fault == "axis":
        options["binary_gripper_axis"] = True
    else:
        y[0, 0] = 2.0
    with pytest.raises(ValueError):
        gn.propose_full_gauss_newton(x, y, parameters, spec, **options)
    assert WORK["forwards"] == before


@pytest.mark.parametrize("fault", ["throw", "rank", "shape", "nonfinite"])
def test_failed_solve_retains_attempt_and_return_costs(fault, monkeypatch):
    spec, parameters, x, y = fixture()
    solve = np.linalg.lstsq

    def broken(*args, **options):
        if fault == "throw":
            raise np.linalg.LinAlgError("solver failed")
        step, errors, rank, singular = solve(*args, **options)
        if fault == "nonfinite":
            step[0] = np.nan
        return (
            (step[:-1] if fault == "shape" else step),
            errors,
            (0 if fault == "rank" else rank),
            singular,
        )

    monkeypatch.setattr(np.linalg, "lstsq", broken)
    with pytest.raises(gn.FullGaussNewtonFailed) as error:
        gn.propose_full_gauss_newton(x, y, parameters, spec)
    assert error.value.work["solve_attempts"] == 1
    assert error.value.work["solve_returns"] == (0 if fault == "throw" else 1)
    assert error.value.work["unknown_partial_work"]


def test_returned_nonfinite_objective_retains_gradient_ack(monkeypatch):
    spec, parameters, x, y = fixture()
    objective = bc.continuous_bc_loss_and_gradients

    def invalid(*args, **options):
        return replace(objective(*args, **options), loss=float("nan"))

    monkeypatch.setattr(bc, "continuous_bc_loss_and_gradients", invalid)
    with pytest.raises(gn.FullGaussNewtonFailed) as error:
        gn.propose_full_gauss_newton(x, y, parameters, spec)
    assert error.value.work["gradient_requests"] == 1
    assert error.value.work["gradient_returns"] == 1
    assert error.value.work["solve_attempts"] == 0


def test_zero_step_rejects_without_publishing_changed_parameters(monkeypatch):
    spec, parameters, x, y = fixture()
    solve = np.linalg.lstsq

    def zero(*args, **options):
        step, errors, rank, singular = solve(*args, **options)
        return np.zeros_like(step), errors, rank, singular

    monkeypatch.setattr(np.linalg, "lstsq", zero)
    result = gn.propose_full_gauss_newton(x, y, parameters, spec)
    assert not result.accepted and result.reason == "no_strict_improvement"
    for a, b in zip(result.parameters, parameters, strict=True):
        np.testing.assert_array_equal(a, b)


def source(*, role="train", axis=3):
    charge("sources")
    spec, parameters, x, y = fixture()
    owner = object.__new__(bp.NumpyContinuousBCLearner)
    owner._policy = bp.ContinuousBCPolicy(
        spec, hidden_dim=2, training_role=role, binary_gripper_axis=axis
    )
    velocity = cast(bc.Parameters, tuple(np.full_like(a, 0.02) for a in parameters))
    owner._state = bp.ContinuousBCSnapshot(owner._policy, parameters, velocity, 0, 0)
    return owner, x, y


@pytest.mark.parametrize("weighted", [False, True])
def test_owned_update_reduces_original_loss_resets_momentum_and_leaves_fork(weighted):
    owner, x, y = source()
    actor = owner.fork()
    old = actor.snapshot_state()
    result = owner.train_full_gauss_newton_batch(
        x, y, row_weights=np.array([1.0, 2.0, 3.0]) if weighted else None
    )
    assert result.loss_after < result.diagnostic.value
    assert result.work["published_updates"] == 1
    assert owner._state.completed_updates == 1 and owner._state.training_examples == 3
    assert all(np.all(a == 0) for a in owner._state.velocity)
    assert actor._state.completed_updates == 0
    for a, b in zip(
        actor._state.parameters + actor._state.velocity, old.parameters + old.velocity, strict=True
    ):
        np.testing.assert_array_equal(a, b)
    assert result.diagnostic.definition == (
        bc.WEIGHTED_MIXED_OBJECTIVE if weighted else bc.MIXED_OBJECTIVE
    )


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_new_adapter_has_no_nontrain_authority(role):
    owner, x, y = source(role="probe_train")
    # Invalid policies cannot be constructed normally; exercise the boundary's
    # refusal of a later-corrupted role as well as the valid PROBE_TRAIN role.
    if role != "probe_train":
        object.__setattr__(owner._policy, "training_role", role)
    before = WORK["forwards"]
    with pytest.raises(ValueError, match="TRAIN|training role"):
        owner.train_full_gauss_newton_batch(x, y)
    assert WORK["forwards"] == before and owner._state.completed_updates == 0


@pytest.mark.parametrize("fault", ["parameters", "policy", "velocity_tuple", "counter"])
def test_returned_solver_cannot_replace_or_corrupt_source_before_commit(monkeypatch, fault):
    owner, x, y = source()
    solve = np.linalg.lstsq

    def changed(*args, **options):
        result = solve(*args, **options)
        if fault == "parameters":
            owner._state.parameters[0][0, 0] += 0.1
        elif fault == "policy":
            object.__setattr__(owner._state, "policy", replace(owner._state.policy))
        elif fault == "velocity_tuple":
            object.__setattr__(owner._state, "velocity", tuple(list(owner._state.velocity)))
        else:
            object.__setattr__(owner._state, "completed_updates", False)
        return result

    monkeypatch.setattr(np.linalg, "lstsq", changed)
    with pytest.raises(TrainingUpdateFailed) as error:
        owner.train_full_gauss_newton_batch(x, y)
    assert error.value.work["published_updates"] == 0
    assert error.value.work["solve_returns"] == 1 and owner._state.completed_updates == 0


def test_malformed_numerical_failure_remains_a_paid_unpublished_failure(monkeypatch):
    owner, x, y = source()
    before = owner._state

    def failed(*args, **options):
        raise gn.FullGaussNewtonFailed("invalid receipt", {"solve_returns": False})

    monkeypatch.setattr(gn, "propose_full_gauss_newton", failed)
    with pytest.raises(TrainingUpdateFailed) as error:
        owner.train_full_gauss_newton_batch(x, y)
    assert owner._state is before
    assert error.value.work["unknown_partial_work"] is True
    assert error.value.work["published_updates"] == 0


def test_constant_objective_rejects_before_any_solve():
    spec, parameters, x, y = fixture()
    spec = replace(spec, lower_bounds=(0.0,) * 4, upper_bounds=(0.0,) * 4)
    result = gn.propose_full_gauss_newton(x, np.zeros_like(y), parameters, spec)
    assert not result.accepted and result.reason == "no_active_axes"
    assert result.loss_before == 0 and result.work["solve_attempts"] == 0


def test_full_update_applies_owned_observation_transform_exactly_once(monkeypatch):
    owner, x, y = source()
    transform = bp.observations.ObservationStandardization((0.1, -0.2), (2.0, 3.0), 3)
    policy = replace(owner._policy, observation_standardization=transform)
    owner._policy = policy
    owner._state = replace(owner._state, policy=policy)
    propose = gn.propose_full_gauss_newton
    seen = []

    def observed(values, *args, **options):
        seen.append(values.copy())
        return propose(values, *args, **options)

    monkeypatch.setattr(gn, "propose_full_gauss_newton", observed)
    result = owner.train_full_gauss_newton_batch(x, y)
    assert result.work["published_updates"] == 1 and len(seen) == 1
    np.testing.assert_array_equal(seen[0], (x - [0.1, -0.2]) / [2.0, 3.0])


def test_equal_replacement_of_effective_scale_owner_is_refused(monkeypatch):
    owner, x, y = source()
    transform = bp.observations.ScaleFlooredObservationStandardization(
        (0.1, -0.2), (2.0, 3.0), 3, (1.0, 1.0)
    )
    policy = replace(owner._policy, observation_standardization=transform)
    owner._policy = policy
    owner._state = replace(owner._state, policy=policy)
    solve = np.linalg.lstsq

    def changed(*args, **options):
        result = solve(*args, **options)
        native_transform = owner._policy.observation_standardization
        object.__setattr__(
            native_transform, "effective_scales", tuple(list(native_transform.effective_scales))
        )
        return result

    monkeypatch.setattr(np.linalg, "lstsq", changed)
    with pytest.raises(TrainingUpdateFailed) as error:
        owner.train_full_gauss_newton_batch(x, y)
    assert error.value.work["published_updates"] == 0
    assert error.value.work["solve_returns"] == 1


def test_rejected_adapter_publishes_nothing_and_retains_paid_solve(monkeypatch):
    owner, x, y = source()
    before = owner._state
    solve = np.linalg.lstsq

    def zero(*args, **options):
        step, errors, rank, singular = solve(*args, **options)
        return np.zeros_like(step), errors, rank, singular

    monkeypatch.setattr(np.linalg, "lstsq", zero)
    with pytest.raises(TrainingUpdateFailed) as error:
        owner.train_full_gauss_newton_batch(x, y)
    assert owner._state is before
    assert error.value.work["solve_returns"] == 1 and error.value.work["published_updates"] == 0


def test_episode_entrypoint_admits_train_and_refuses_reserved_episode_roles():
    owner, x, y = source()
    episode = ContinuousEpisode(
        "literal-full-fit",
        "literal-training",
        "train",
        owner._policy.spec,
        tuple(
            ContinuousTransition(
                i,
                tuple(float(v) for v in x[i]),
                tuple(float(v) for v in y[i]),
                0.0,
                tuple(float(v) for v in x[min(i + 1, 2)]),
                i == 2,
                False,
            )
            for i in range(3)
        ),
    )
    for role in ("probe_train", "probe_test", "final_test"):
        with pytest.raises(ValueError):
            owner.train_full_gauss_newton_episode(replace(episode, role=role))
    result = owner.train_full_gauss_newton_episode(episode)
    assert result.work["published_updates"] == 1
