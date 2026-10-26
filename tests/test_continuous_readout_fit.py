"""Frozen-hidden least-squares diagnostics with hand-specified hidden features.

Prospective per platform: six returned solves/16 rows and one raised solve/2
rows; one finite preactivation-overflow refusal before solve. Two tiny learner
initializers draw16 normals; zero SGD, empirical fits, serving or native work.
All successful fits are cached; refusal tests cannot enter the solver.
"""

from dataclasses import replace
import math
from typing import Any, cast

import numpy as np
import pytest

from src.adapters import continuous_readout_fit as adapter
from src.adapters import numpy_continuous_bc as bp
from src.adapters import numpy_continuous_pc as pc
from src.core import continuous_readout_fit as core
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.observation_standardization import ObservationStandardization


def spec():
    return ContinuousActionSpec(2, ("physical_a", "physical_b"), (-2.0, 3.0), (2.0, 7.0))


def parameters():
    return cast(bp.Parameters, (np.eye(2), np.zeros(2), np.zeros((2, 2)), np.zeros(2)))


def physical(hidden, coefficients, declared):
    logits = np.asarray([(*row, 1.0) for row in hidden]) @ np.asarray(coefficients)
    normalized = np.tanh(logits)
    lower, upper = np.asarray(declared.lower_bounds), np.asarray(declared.upper_bounds)
    return (lower + upper) / 2 + (upper - lower) / 2 * normalized


@pytest.fixture(scope="module")
def fits():
    rank_hidden = np.array([[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]])
    rank_coefficients = np.array([[0.5, -1.0], [0.0, 0.0], [0.25, -0.125]])
    full_hidden = np.array([[-0.5, -0.25], [-0.5, 0.25], [0.5, -0.25], [0.5, 0.25]])
    full_coefficients = np.array([[0.5, -1.0], [0.25, 0.5], [0.125, -0.25]])
    constant_spec = ContinuousActionSpec(2, ("constant", "varying"), (2.0, -1.0), (2.0, 1.0))
    singleton_spec = ContinuousActionSpec(2, ("constant_a", "constant_b"), (2.0, -3.0), (2.0, -3.0))
    cases = {
        "rank": (
            np.arctanh(rank_hidden),
            physical(rank_hidden, rank_coefficients, spec()),
            parameters(),
            spec(),
        ),
        "clip": (
            np.arctanh(np.array([[-0.5, 0.0], [0.5, 0.0]])),
            np.array([[2.0, -1.0], [2.0, 1.0]]),
            parameters(),
            constant_spec,
        ),
        "singleton": (np.zeros((1, 2)), np.array([[2.0, -3.0]]), parameters(), singleton_spec),
        "full": (
            np.arctanh(full_hidden),
            physical(full_hidden, full_coefficients, spec()),
            parameters(),
            spec(),
        ),
    }
    calls = []
    solve = np.linalg.lstsq

    def solving(design, targets, *, rcond):
        calls.append((design.copy(), targets.copy(), rcond))
        return solve(design, targets, rcond=rcond)

    results = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(np.linalg, "lstsq", solving)
        for name, inputs in cases.items():
            results[name] = core.fit_continuous_readout(*inputs)
    return results, cases, calls, rank_coefficients, full_coefficients


def test_should_recover_an_exact_representable_physical_head_without_changing_hidden_parameters(
    fits,
):
    results, cases, _, expected, _ = fits
    result = results["rank"]
    assert (
        result.objective_definition == "frozen_hidden_clipped_inverse_tanh_minimum_norm_readout_v1"
    )
    assert result.target_clip == 1e-6 and result.rcond == 1e-12
    np.testing.assert_allclose(result.coefficients, expected, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(result.predictions_after, cases["rank"][1], atol=2e-12, rtol=2e-12)
    np.testing.assert_array_equal(result.predictions_before, [[0.0, 5.0]] * 3)
    expected_mse = float(np.mean((cases["rank"][1] - np.array([0.0, 5.0])) ** 2))
    assert result.physical_mse_before == pytest.approx(expected_mse)
    assert result.physical_mse_after < 1e-24
    assert result.coefficient_norm == pytest.approx(math.sqrt(1.328125))
    assert result.clipped_targets_per_axis == (0, 0)
    np.testing.assert_array_equal(cases["rank"][2][0], np.eye(2))


def test_should_report_rank_deficiency_effective_condition_and_one_fixed_solve_cost(fits):
    results, _, calls, _, _ = fits
    result = results["rank"]
    assert len(calls) == 4 and all(call[2] == 1e-12 == core.RCOND for call in calls)
    assert result.rank == 2 and result.condition_number is None
    np.testing.assert_allclose(result.singular_values[:2], [math.sqrt(3), math.sqrt(0.5)])
    assert result.effective_condition_number == pytest.approx(math.sqrt(6))
    np.testing.assert_allclose(calls[0][0], [[-0.5, 0.0, 1.0], [0.0, 0.0, 1.0], [0.5, 0.0, 1.0]])
    work = result.work
    assert work["solve_attempts"] == work["solve_completed"] == work["hidden_forward_passes"] == 1
    assert work["head_forward_passes"] == 2 and work["rows"] == 3
    assert work["design_values"] == 9 and work["design_numeric_bytes"] == 72
    assert work["design_columns"] == 3 and work["column_nullity"] == 1
    assert work["coefficient_values"] == 6 and work["coefficient_numeric_bytes"] == 48
    assert work["prediction_values"] == 12 and work["prediction_numeric_bytes"] == 96
    assert work["dense_multiply_accumulates"] == 36
    assert work["solver_work_known"] is False


def test_should_clip_only_inverse_targets_and_score_the_original_endpoints_including_constant_axes(
    fits,
):
    result = fits[0]["clip"]
    assert core.TARGET_CLIP == 1e-6 and result.clipped_targets_per_axis == (0, 2)
    np.testing.assert_allclose(result.coefficients[:, 0], 0.0, atol=1e-14)
    np.testing.assert_array_equal(result.predictions_after[:, 0], [2.0, 2.0])
    np.testing.assert_allclose(
        result.predictions_after[:, 1], [-1 + core.TARGET_CLIP, 1 - core.TARGET_CLIP], atol=1e-14
    )
    assert result.physical_mse_before == pytest.approx(0.5)
    assert result.physical_mse_after == pytest.approx(core.TARGET_CLIP**2 / 2, abs=1e-20)


def test_should_still_solve_an_all_constant_singleton_and_return_the_minimum_norm_zero_head(fits):
    result = fits[0]["singleton"]
    assert result.rows == result.rank == 1
    assert result.condition_number == result.effective_condition_number == 1.0
    assert result.coefficient_norm == 0.0 and result.clipped_targets_per_axis == (0, 0)
    np.testing.assert_array_equal(result.coefficients, np.zeros((3, 2)))
    np.testing.assert_array_equal(result.predictions_before, [[2.0, -3.0]])
    np.testing.assert_array_equal(result.predictions_after, [[2.0, -3.0]])
    assert result.physical_mse_before == result.physical_mse_after == 0.0
    assert result.work["solve_completed"] == 1
    assert result.work["design_columns"] == 3 and result.work["column_nullity"] == 2


def test_should_recover_both_hidden_coordinates_in_a_full_rank_design(fits):
    result = fits[0]["full"]
    assert result.rank == 3 and result.condition_number is not None
    np.testing.assert_allclose(result.coefficients, fits[4], atol=2e-12, rtol=2e-12)
    assert result.condition_number == pytest.approx(result.effective_condition_number)


def test_should_own_all_returned_arrays_without_aliasing_inputs_or_other_output_arrays(fits):
    result, inputs = fits[0]["full"], fits[1]["full"]
    outputs = (
        result.coefficients,
        result.singular_values,
        result.predictions_before,
        result.predictions_after,
    )
    for output in outputs:
        assert not any(np.shares_memory(output, original) for original in (*inputs[:2], *inputs[2]))
    for i, first in enumerate(outputs):
        assert not any(np.shares_memory(first, second) for second in outputs[i + 1 :])
    originals = tuple(value.copy() for value in (*inputs[:2], *inputs[2]))
    saved = tuple(output.copy() for output in outputs)
    try:
        for output in outputs:
            output.fill(99.0)
        for current, original in zip((*inputs[:2], *inputs[2]), originals, strict=True):
            np.testing.assert_array_equal(current, original)
    finally:
        for output, original in zip(outputs, saved, strict=True):
            output[:] = original


@pytest.mark.parametrize(
    "fault",
    [
        "feature_dtype",
        "feature_nan",
        "target_nan",
        "target_bounds",
        "feature_shape",
        "target_shape",
        "parameter_shape",
        "parameter_nan",
        "rows",
        "dimensions",
        "overflow",
    ],
)
def test_should_refuse_invalid_or_unrepresentable_inputs_before_the_linear_solver(
    monkeypatch, fault
):
    features, targets, params, declared = (
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        parameters(),
        ContinuousActionSpec(2, ("a", "b"), (-1.0,) * 2, (1.0,) * 2),
    )
    if fault == "feature_dtype":
        features = features.astype(np.float32)
    elif fault == "feature_nan":
        features[0, 0] = np.nan
    elif fault == "target_nan":
        targets[0, 0] = np.nan
    elif fault == "target_bounds":
        targets[0, 0] = 2.0
    elif fault == "feature_shape":
        features = np.zeros((2, 1))
    elif fault == "target_shape":
        targets = np.zeros((1, 2))
    elif fault == "parameter_shape":
        params = (np.zeros((3, 2)), *params[1:])
    elif fault == "parameter_nan":
        params[0][0, 0] = np.inf
    elif fault == "rows":
        features, targets = np.zeros((129, 2)), np.zeros((129, 2))
    elif fault == "dimensions":
        declared = replace(declared, observation_dim=65)
        features, params = np.zeros((2, 65)), (np.zeros((65, 2)), *params[1:])
    else:
        features.fill(1e308)
        params[0].fill(1e308)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid input reached the linear solver")

    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    with pytest.raises((ValueError, FloatingPointError)):
        core.fit_continuous_readout(features, targets, params, declared)


def test_should_propagate_one_solver_failure_and_preserve_all_source_arrays(monkeypatch):
    features, targets, params = np.zeros((2, 2)), np.array([[0.0, 5.0], [0.0, 5.0]]), parameters()
    originals = tuple(array.copy() for array in (features, targets, *params))
    attempts = []

    def fail(design, values, *, rcond):
        attempts.append((design.shape, values.shape, rcond))
        raise np.linalg.LinAlgError("synthetic one-shot solver failure")

    monkeypatch.setattr(np.linalg, "lstsq", fail)
    with pytest.raises(np.linalg.LinAlgError, match="one-shot"):
        core.fit_continuous_readout(features, targets, params, spec())
    assert attempts == [((2, 3), (2, 2), core.RCOND)]
    for array, original in zip((features, targets, *params), originals, strict=True):
        np.testing.assert_array_equal(array, original)


def episodes(features, targets):
    records = []
    for name, start, end in (("first", 0, 2), ("second", 2, 3)):
        rows = tuple(
            ContinuousTransition(
                i - start,
                tuple(float(value) for value in features[i]),
                tuple(float(value) for value in targets[i]),
                float(i == end - 1),
                tuple(float(value) for value in features[min(i + 1, end - 1)]),
                i == end - 1,
                False,
            )
            for i in range(start, end)
        )
        records.append(ContinuousEpisode(name, "train-context", "train", spec(), rows))
    return tuple(records)


@pytest.fixture(scope="module")
def models(fits):
    features, targets, params, _ = fits[1]["rank"]
    stats = ObservationStandardization((1.0, -2.0), (2.0, 0.5), 3)
    raw = features * np.array(stats.scales) + np.array(stats.means)
    data = episodes(raw, targets)
    base = bp.ContinuousBCPolicy(spec(), hidden_dim=2, seed=17, observation_standardization=stats)
    records: dict[str, Any] = {}
    for family in ("bp", "pc"):
        owner: Any = (
            bp.NumpyContinuousBCLearner(base)
            if family == "bp"
            else pc.NumpyContinuousPCLearner(
                pc.ContinuousPCPolicy(base, circadian=CoupledCircadianConfig())
            )
        )
        initial = owner.snapshot_state()
        velocity = cast(bp.Parameters, tuple(np.full_like(array, 0.125) for array in params))
        options: dict[str, Any] = (
            {"wake_state": replace(initial.wake_state, wake_updates=2)} if family == "pc" else {}
        )
        owner.restore_state(
            replace(
                initial,
                parameters=params,
                velocity=velocity,
                completed_updates=2,
                training_examples=3,
                **options,
            )
        )
        records[family] = owner
    return records, data


@pytest.fixture(scope="module")
def adapter_fits(models):
    results = {}
    for family, owner in models[0].items():
        before = owner.snapshot_state()
        result = adapter.diagnose_readout_fit(owner, models[1])
        results[family] = result, before
    return results


def assert_state(first, second):
    assert first.policy == second.policy
    assert (
        first.completed_updates == second.completed_updates
        and first.training_examples == second.training_examples
    )
    for a, b in zip(
        first.parameters + first.velocity, second.parameters + second.velocity, strict=True
    ):
        np.testing.assert_array_equal(a, b)
    if type(first) is pc.ContinuousPCSnapshot:
        assert first.wake_state is not None and second.wake_state is not None
        assert first.wake_state.wake_updates == second.wake_state.wake_updates == 2
        np.testing.assert_array_equal(
            first.wake_state.chemical_state, second.wake_state.chemical_state
        )
        np.testing.assert_array_equal(
            first.wake_state.plasticity_state, second.wake_state.plasticity_state
        )


@pytest.mark.parametrize("family", ["bp", "pc"])
def test_should_apply_only_existing_frozen_statistics_and_leave_full_parameters_momentum_and_wake_unchanged(
    models, adapter_fits, fits, family
):
    result, before = adapter_fits[family]
    assert result.standardization_calls == 1 and result.standardization_rows == 3
    assert result.standardization_values == 6 and result.standardization_applied is True
    np.testing.assert_allclose(result.fit.coefficients, fits[3], atol=2e-12, rtol=2e-12)
    assert_state(before, models[0][family].snapshot_state())
    assert before.completed_updates == 2 and before.training_examples == 3
    assert all(np.any(value != 0.0) for value in before.velocity)


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test", "inner_guard"])
def test_should_refuse_a_later_heldout_episode_before_normalization_or_solve(
    models, monkeypatch, role
):
    data = (models[1][0], replace(models[1][1], role=role))
    monkeypatch.setattr(
        adapter, "standardize_observations", lambda *args: pytest.fail("heldout normalization")
    )
    monkeypatch.setattr(adapter, "fit_continuous_readout", lambda *args: pytest.fail("heldout fit"))
    with pytest.raises(ValueError):
        adapter.diagnose_readout_fit(models[0]["bp"], data)


@pytest.mark.parametrize("fault", ["list", "duplicate", "spec"])
def test_should_refuse_changed_episode_container_identity_or_spec_before_normalization(
    models, monkeypatch, fault
):
    data: Any = models[1]
    if fault == "list":
        data = list(data)
    elif fault == "duplicate":
        data = (data[0], data[0])
    else:
        data = (
            data[0],
            replace(data[1], spec=replace(spec(), action_units=("changed", "physical_b"))),
        )
    monkeypatch.setattr(
        adapter, "standardize_observations", lambda *args: pytest.fail("invalid normalization")
    )
    with pytest.raises(ValueError):
        adapter.diagnose_readout_fit(models[0]["bp"], data)


def test_should_refuse_a_probe_training_policy_before_extracting_train_rows(models, monkeypatch):
    owner = models[0]["bp"].fork()
    before = owner.snapshot_state()
    owner._policy = replace(owner.policy, training_role="probe_train")
    owner.restore_state(replace(before, policy=owner._policy))
    monkeypatch.setattr(
        adapter, "standardize_observations", lambda *args: pytest.fail("probe policy normalization")
    )
    with pytest.raises(ValueError):
        adapter.diagnose_readout_fit(owner, models[1])


def test_should_refuse_source_mutation_in_a_raising_fit_callback_without_restoring_away_the_evidence(
    models, monkeypatch
):
    owner = models[0]["bp"].fork()
    before = owner.snapshot_state()

    def corrupt(*args):
        owner._state.parameters[3][0] += 0.25
        raise ValueError("synthetic diagnostic failure")

    monkeypatch.setattr(adapter, "fit_continuous_readout", corrupt)
    with pytest.raises(RuntimeError, match="changed its source"):
        adapter.diagnose_readout_fit(owner, models[1])
    assert owner.snapshot_state().parameters[3][0] == before.parameters[3][0] + 0.25
    assert_state(before, models[0]["bp"].snapshot_state())
