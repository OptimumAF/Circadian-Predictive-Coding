"""Analytic fit/readout references and distinct old/new geometry contracts."""

from dataclasses import FrozenInstanceError, asdict, replace
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from src.core import categorical_linear_fit as old
from src.core import wide_categorical_linear_fit as wide


def data():
    return np.array([[-1.0], [0.0], [1.0]]), np.array([[0.0], [1.0], [1.0]])


def test_should_match_independent_scalar_fit_residual_spectrum_and_softmax():
    x, y = data()
    before = x.copy(), y.copy()
    model = wide.fit_wide_categorical_linear(x, y, 2)
    np.testing.assert_allclose(model.coefficients, [[-0.5, 0.5], [1 / 3, 2 / 3]], atol=1e-14)
    assert model.rank == 2 and model.squared_error == pytest.approx(1 / 3)
    np.testing.assert_allclose(model.singular_values, [np.sqrt(3), np.sqrt(2)])
    expected_scores = np.array([[5 / 6, 1 / 6], [1 / 3, 2 / 3], [-1 / 6, 7 / 6]])
    expected = np.exp(expected_scores - expected_scores.max(axis=1, keepdims=True))
    expected /= expected.sum(axis=1, keepdims=True)
    p = wide.predict_wide_categorical_linear(model, x)
    np.testing.assert_allclose(p, expected, atol=1e-14)
    np.testing.assert_array_equal(x, before[0])
    np.testing.assert_array_equal(y, before[1])
    p[:] = 0
    assert np.any(wide.predict_wide_categorical_linear(model, x) > 0)
    with pytest.raises(FrozenInstanceError):
        setattr(model, "rank", 0)


def test_should_keep_incompatible_identical_rows_as_negative_fit_and_first_tie():
    model = wide.fit_wide_categorical_linear(
        np.zeros((4, 3)), np.array([[0.0], [1.0], [0.0], [1.0]]), 3
    )
    np.testing.assert_allclose(model.coefficients, [[0, 0, 0]] * 3 + [[0.5, 0.5, 0]], atol=1e-14)
    assert model.rank == 1 and model.squared_error == pytest.approx(2.0)
    p = wide.predict_wide_categorical_linear(model, np.zeros((4, 3)))
    assert np.argmax(p, axis=1).tolist() == [0] * 4  # Two contradictory labels stay wrong.


def test_should_preserve_saved_before_legacy_math_and_distinct_admission():
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/clm-linear-fit-control-20261025/categorical_before.py"
    )
    spec = importlib.util.spec_from_file_location("before_wide_linear", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    x, y = data()
    assert asdict(module.fit_categorical_linear(x, y, 2)) == asdict(
        old.fit_categorical_linear(x, y, 2)
    )
    with pytest.raises(ValueError):
        old.fit_categorical_linear(np.zeros((2, 297)), np.zeros((2, 1)), 2)
    m = wide.fit_wide_categorical_linear(np.zeros((2, 297)), np.zeros((2, 1)), 2)
    wrong_model: Any = m
    with pytest.raises(ValueError):
        old.predict_categorical_linear(wrong_model, np.zeros((2, 297)))


@pytest.mark.parametrize(
    "fault", ["width", "rows", "dtype", "subclass", "nan", "range", "target", "classes"]
)
def test_should_refuse_invalid_wide_inputs_before_solver(monkeypatch, fault):
    x: Any
    y: Any
    c: Any = 2
    x, y = data()
    if fault == "width":
        x = np.zeros((3, 4097))
    elif fault == "rows":
        x, y = np.zeros((17, 1)), np.zeros((17, 1))
    elif fault == "dtype":
        x = x.astype(np.float32)
    elif fault == "subclass":

        class Sub(np.ndarray):
            pass

        x = x.view(Sub)
    elif fault == "nan":
        x[0, 0] = np.nan
    elif fault == "range":
        x[0, 0] = 1.1
    elif fault == "target":
        y[0, 0] = 0.5
    else:
        c = True
    monkeypatch.setattr(np.linalg, "lstsq", lambda *a, **k: pytest.fail("invalid solve"))
    with pytest.raises(ValueError):
        wide.fit_wide_categorical_linear(x, y, c)


@pytest.mark.parametrize(
    "field,value",
    [
        ("feature_dim", 4097),
        ("fit_rows", 17),
        ("rank", True),
        ("squared_error", float("nan")),
        ("coefficients", ((1, 0), (0.0, 1.0))),
    ],
)
def test_should_refuse_corrupted_wide_model(field, value):
    model = wide.WideCategoricalLinearFit(1, 2, 2, ((1.0, 0.0), (0.0, 1.0)), 2, (1.0, 1.0), 0.0)
    with pytest.raises(ValueError):
        replace(model, **{field: value})


def test_should_refuse_wrong_solver_geometry_before_model_publication(monkeypatch):
    monkeypatch.setattr(
        np.linalg, "lstsq", lambda *a, **k: (np.zeros((1, 2)), np.array([]), 1, np.array([1.0]))
    )
    with pytest.raises(ValueError, match="coefficients"):
        x, y = data()
        wide.fit_wide_categorical_linear(x, y, 2)
