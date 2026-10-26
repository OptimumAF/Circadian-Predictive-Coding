"""Warm residual solves against independent projection oracles, no model assets."""

from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from src.core import wide_categorical_linear_fit as wide


def prior(width=2):
    # A legal nonzero prior; latest fit metadata is not a lifetime work counter.
    return wide.WideCategoricalLinearFit(
        width, 2, 1, tuple((0.2, -0.1) for _ in range(width + 1)), 1, (1.0,), 0.0
    )


def test_should_preserve_nullspace_and_apply_minimum_change_not_cold_refit():
    model = prior()
    x, y = np.array([[1.0, 0.0]]), np.array([[1.0]])
    before = asdict(model), x.copy(), y.copy()
    updated = wide.update_wide_categorical_linear(model, x, y)
    # For z=[1,0,1], delta=z^T * ([0,1]-zW)/2.
    expected = np.array([[0.0, 0.5], [0.2, -0.1], [0.0, 0.5]])
    np.testing.assert_allclose(updated.coefficients, expected, atol=1e-14)
    assert updated.rank == 1 and updated.squared_error < 1e-28
    assert updated.fit_rows == 1 and len(updated.singular_values) == 1
    assert asdict(model) == before[0]
    np.testing.assert_array_equal(x, before[1])
    np.testing.assert_array_equal(y, before[2])
    assert not hasattr(updated, "prior")


def test_should_report_inconsistent_rows_and_actual_last_batch_residual():
    updated = wide.update_wide_categorical_linear(
        prior(), np.zeros((2, 2)), np.array([[0.0], [1.0]])
    )
    assert updated.rank == 1 and updated.squared_error == pytest.approx(1.0)
    np.testing.assert_allclose(updated.coefficients, [[0.2, -0.1]] * 2 + [[0.5, 0.5]])


def test_should_update_all4096_coordinates_with_one_analytic_rank_one_projection():
    x = np.full((1, 4096), 1 / 64, dtype=np.float64)
    model = prior(4096)
    updated = wide.update_wide_categorical_linear(model, x, np.array([[1.0]]))
    z = np.append(x[0], 1)
    weights = np.array(model.coefficients)
    expected = weights + z[:, None] * (np.array([0.0, 1.0]) - z @ weights) / 2
    np.testing.assert_allclose(updated.coefficients, expected, rtol=1e-12, atol=1e-13)
    assert updated.rank == 1 and updated.squared_error < 1e-23


@pytest.mark.parametrize("fault", ["width", "dtype", "nan", "target", "model", "overflow"])
def test_should_refuse_bad_updates_without_solver(monkeypatch, fault):
    model: Any = prior()
    x, y = np.zeros((2, 2)), np.array([[0.0], [1.0]])
    if fault == "width":
        x = np.zeros((2, 3))
    elif fault == "dtype":
        x = x.astype(np.float32)
    elif fault == "nan":
        x[0, 0] = np.nan
    elif fault == "target":
        y[0, 0] = 2
    elif fault == "model":
        model = object()
    else:
        model = replace(model, coefficients=((1e308, 1e308),) * 3)
        x[:] = 1
    monkeypatch.setattr(np.linalg, "lstsq", lambda *a, **k: pytest.fail("invalid solve"))
    with pytest.raises((ValueError, FloatingPointError)):
        wide.update_wide_categorical_linear(model, x, y)


def test_should_refuse_nonfinite_solver_return(monkeypatch):
    monkeypatch.setattr(
        np.linalg, "lstsq", lambda *a, **k: (np.full((3, 2), np.nan), [], 1, np.array([1.0]))
    )
    with pytest.raises(ValueError, match="invalid coefficients"):
        wide.update_wide_categorical_linear(prior(), np.zeros((1, 2)), np.array([[0.0]]))
