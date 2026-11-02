"""Analytical physical affine references; all genuine tiny LS work is capped."""

from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.core import continuous_affine_fit as core
from src.core.cube_taskspace_state import cube_taskspace_spec

WORK = dict(attempts=0, rows=0, returns=0)


@pytest.fixture(autouse=True)
def budgeted_solve(monkeypatch):
    original = np.linalg.lstsq

    def solve(matrix, targets, **options):
        WORK["attempts"] += 1
        WORK["rows"] += len(matrix)
        assert WORK["attempts"] <= 64 and WORK["rows"] <= 1024
        result = original(matrix, targets, **options)
        WORK["returns"] += 1
        return result

    monkeypatch.setattr(np.linalg, "lstsq", solve)


def inputs(rows=13):
    x = np.zeros((rows, 12), dtype=np.float64)
    for index in range(1, min(rows, 13)):
        x[index, index - 1] = 1.0
    weights = np.arange(1, rows + 1, dtype=np.float64)
    coefficients = np.arange(52, dtype=np.float64).reshape(13, 4) / 100
    y = np.column_stack((x, np.ones(rows))) @ coefficients
    return x, y, weights, coefficients


def test_should_recover_known_full_rank_physical_affine_map_with_unequal_weights():
    x, y, weights, expected = inputs()
    before = pickle.dumps((x, y, weights, np.random.get_state()))
    fit = core.fit_continuous_affine(x, y, weights, cube_taskspace_spec())
    assert fit.rank == 13 and fit.fit_rows == 13 and fit.condition_number is not None
    np.testing.assert_allclose(fit.coefficients, expected, atol=3e-14, rtol=0)
    result = core.predict_continuous_affine(fit, x)
    np.testing.assert_allclose(result.raw, y, atol=3e-14, rtol=0)
    assert result.clipped_per_axis == (0, 0, 0, 0)
    assert pickle.dumps((x, y, weights, np.random.get_state())) == before
    result.raw[0, 0] = -10
    assert result.bounded[0, 0] >= -1 and not np.shares_memory(result.raw, result.bounded)
    assert not np.shares_memory(result.bounded, x)


def test_should_fit_weighted_mean_on_identical_inputs_without_dropping_rows():
    x = np.zeros((2, 12), dtype=np.float64)
    y = np.array([[0.2] * 4, [0.8] * 4], dtype=np.float64)
    fit = core.fit_continuous_affine(x, y, np.array([1.0, 3.0]), cube_taskspace_spec())
    assert fit.fit_rows == 2 and fit.rank == 1 and fit.condition_number is None
    np.testing.assert_allclose(core.predict_continuous_affine(fit, x).raw, 0.65, atol=1e-15)


def test_should_clip_extrapolation_only_after_validating_raw_predictions():
    x = np.zeros((2, 12), dtype=np.float64)
    x[1, 0] = 1
    y = np.array([[0.0] * 4, [1.0] * 4], dtype=np.float64)
    fit = core.fit_continuous_affine(x, y, np.ones(2), cube_taskspace_spec())
    after = x[1:].copy() * 2
    prediction = core.predict_continuous_affine(fit, after)
    np.testing.assert_allclose(prediction.raw, 2, atol=1e-14)
    np.testing.assert_array_equal(prediction.bounded, np.ones((1, 4)))
    assert prediction.clipped_per_axis == (1, 1, 1, 1)


def test_should_admit_exact198_physical_rows_without_widening_old_hidden_readout():
    x, y, weights, _ = inputs(198)
    fit = core.fit_continuous_affine(x, y, weights, cube_taskspace_spec())
    assert fit.fit_rows == 198 and len(fit.singular_values) == 13
    np.testing.assert_allclose(core.predict_continuous_affine(fit, x).raw, y, atol=1e-13)


@pytest.mark.parametrize(
    "fault",
    [
        "dtype",
        "width",
        "empty",
        "199",
        "nan",
        "target_dtype",
        "target_width",
        "bounds",
        "weight_dtype",
        "zero_weight",
        "negative_weight",
        "nan_weight",
        "weight_shape",
        "spec",
    ],
)
def test_should_refuse_invalid_inputs_before_any_solve(fault):
    x, y, weights, _ = inputs()
    spec = cube_taskspace_spec()
    if fault == "dtype":
        x = x.astype(np.float32)
    elif fault == "width":
        x = x[:, :-1]
    elif fault == "empty":
        x, y, weights = x[:0], y[:0], weights[:0]
    elif fault == "199":
        x, y, weights, _ = inputs(199)
    elif fault == "nan":
        x[0, 0] = np.nan
    elif fault == "target_dtype":
        y = y.astype(np.float32)
    elif fault == "target_width":
        y = y[:, :-1]
    elif fault == "bounds":
        y[0, 0] = 2
    elif fault == "weight_dtype":
        weights = weights.astype(np.float32)
    elif fault == "zero_weight":
        weights[0] = 0
    elif fault == "negative_weight":
        weights[0] = -1
    elif fault == "nan_weight":
        weights[0] = np.nan
    elif fault == "weight_shape":
        weights = weights[:, None]
    else:
        spec = replace(spec, observation_dim=13)
    before = WORK.copy()
    with pytest.raises(ValueError):
        core.fit_continuous_affine(x, y, weights, spec)
    assert WORK == before


def test_should_preserve_relative_weights_even_at_large_common_scale():
    x, y, weights, _ = inputs()
    ordinary = core.fit_continuous_affine(x, y, weights, cube_taskspace_spec())
    scaled = core.fit_continuous_affine(x, y, weights * 1e300, cube_taskspace_spec())
    np.testing.assert_allclose(scaled.coefficients, ordinary.coefficients, atol=3e-14)


@pytest.mark.parametrize("fault", ["features", "targets", "weights", "design", "rhs", "lost"])
def test_should_refuse_mutated_or_unreturned_solver_work_without_a_model(monkeypatch, fault):
    x, y, weights, _ = inputs()
    original = core._solve

    def solve(matrix, rhs):
        result = original(matrix, rhs)
        if fault == "features":
            x[0, 0] = 0.5
        elif fault == "targets":
            y[0, 0] = 0.5
        elif fault == "weights":
            weights[0] = 0.5
        elif fault == "design":
            matrix[0, 0] = 0.5
        elif fault == "rhs":
            rhs[0, 0] = 0.5
        else:
            raise RuntimeError("lost solver return")
        return result

    monkeypatch.setattr(core, "_solve", solve)
    before = WORK["attempts"]
    with pytest.raises((ValueError, RuntimeError)):
        core.fit_continuous_affine(x, y, weights, cube_taskspace_spec())
    assert WORK["attempts"] == before + 1


@pytest.mark.parametrize("fault", ["rank", "coefficients", "singular", "condition", "rows"])
def test_should_refuse_inconsistent_fitted_state(fault):
    x, y, w, _ = inputs(2)
    fit = core.fit_continuous_affine(x, y, w, cube_taskspace_spec())
    changes: dict[str, dict[str, Any]] = {
        "rank": dict(rank=0),
        "coefficients": dict(coefficients=((0.0,) * 4,) * 12),
        "singular": dict(singular_values=(1.0, -1.0)),
        "condition": dict(condition_number=True),
        "rows": dict(fit_rows=True),
    }
    with pytest.raises(ValueError):
        replace(fit, **changes[fault])


@pytest.mark.parametrize("fault", ["input", "model", "equal_coefficients"])
def test_should_refuse_prediction_changes_and_equal_model_field_replacement(monkeypatch, fault):
    x, y, w, _ = inputs(2)
    fit = core.fit_continuous_affine(x, y, w, cube_taskspace_spec())
    original = np.matmul

    def multiply(*args):
        result = original(*args)
        if fault == "input":
            x[0, 0] = 0.5
        elif fault == "model":
            object.__setattr__(fit, "rank", 0)
        else:
            object.__setattr__(fit, "coefficients", tuple(list(fit.coefficients)))
        return result

    monkeypatch.setattr(np, "matmul", multiply)
    with pytest.raises(ValueError, match="source/model changed"):
        core.predict_continuous_affine(fit, x)
