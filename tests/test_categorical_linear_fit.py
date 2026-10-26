"""Closed-form linear TRAIN-fit math, bounded admission and detached results.

These tests use analytic examples rather than a second call to the solver as an
oracle. No learner, game, evaluation dataset or native training is involved.
"""

from dataclasses import FrozenInstanceError
import pickle
import random
from typing import Any

import numpy as np
import pytest

from src.core import categorical_linear_fit as linear


def data():
    return np.array([[-1.0], [0.0], [1.0]]), np.array([[0.0], [1.0], [1.0]])


def manual_model():
    return linear.CategoricalLinearFit(
        feature_dim=1,
        num_classes=3,
        fit_rows=3,
        coefficients=((1000.0, 0.0, -1000.0), (0.0, 0.0, 0.0)),
        rank=2,
        singular_values=(2.0, 1.0),
        squared_error=0.0,
    )


def test_should_match_closed_form_intercept_slopes_and_unconditional_residual():
    x, y = data()
    model = linear.fit_categorical_linear(x, y, 2)
    assert model.feature_dim == 1 and model.num_classes == 2 and model.fit_rows == 3
    # Least squares on symmetric x: intercept is each class mean; slope is
    # sum(x*y)/sum(x*x). Residual is 2*(1/36+1/9+1/36)=1/3.
    np.testing.assert_allclose(model.coefficients, [[-0.5, 0.5], [1 / 3, 2 / 3]], atol=1e-14)
    assert model.squared_error == pytest.approx(1 / 3)
    assert model.rank == 2
    np.testing.assert_allclose(model.singular_values, [np.sqrt(3), np.sqrt(2)])
    predicted = linear.predict_categorical_linear(model, x)
    assert predicted.dtype == np.float64 and predicted.shape == (3, 2)
    assert np.argmax(predicted, axis=1).tolist() == [0, 1, 1]


def test_should_use_one_fixed_rcond_solve_and_recompute_even_when_solver_residual_is_empty(
    monkeypatch,
):
    x, y = data()
    original = np.linalg.lstsq
    calls = []

    def solve(design, targets, *, rcond):
        calls.append((design.copy(), targets.copy(), rcond))
        coefficients, _, rank, singular = original(design, targets, rcond=rcond)
        return coefficients, np.empty(0, dtype=np.float64), rank, singular

    monkeypatch.setattr(np.linalg, "lstsq", solve)
    model = linear.fit_categorical_linear(x, y, 2)
    assert len(calls) == 1 and calls[0][2] == linear.RCOND == 1e-12
    np.testing.assert_array_equal(calls[0][0], [[-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]])
    np.testing.assert_array_equal(calls[0][1], [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    assert model.squared_error == pytest.approx(1 / 3)


def test_should_return_minimum_norm_rank_deficient_fit_with_absent_class_support():
    x = np.zeros((4, 2), dtype=np.float64)
    y = np.array([[0.0], [0.0], [1.0], [1.0]])
    model = linear.fit_categorical_linear(x, y, 3)
    np.testing.assert_allclose(
        np.array(model.coefficients),
        np.array([[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0]], dtype=np.float64),
    )
    assert model.rank == 1 and len(model.singular_values) == 3
    assert model.squared_error == pytest.approx(2.0)
    probabilities = linear.predict_categorical_linear(model, x)
    expected = np.exp([0.5, 0.5, 0.0])
    expected /= expected.sum()
    np.testing.assert_allclose(probabilities, np.tile(expected, (4, 1)))
    assert np.argmax(probabilities, axis=1).tolist() == [0] * 4


def test_should_softmax_large_finite_scores_stably_and_resolve_ties_to_first_class():
    model = manual_model()
    predictions = linear.predict_categorical_linear(model, np.array([[-1.0], [0.0], [1.0]]))
    assert np.isfinite(predictions).all()
    np.testing.assert_allclose(predictions.sum(axis=1), 1.0)
    np.testing.assert_array_equal(predictions[0], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(predictions[2], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(predictions[1], [1 / 3] * 3)
    assert np.argmax(predictions, axis=1).tolist() == [2, 0, 0]


def test_should_admit_upper_geometry_and_single_row_without_coefficient_aliases():
    x = np.zeros((128, 296), dtype=np.float64)
    y = np.full((128, 1), 15.0)
    model = linear.fit_categorical_linear(x, y, 16)
    assert model.rank == 1 and len(model.singular_values) == 128
    assert len(model.coefficients) == 297
    assert all(type(row) is tuple and len(row) == 16 for row in model.coefficients)
    assert all(type(value) is float for row in model.coefficients for value in row)
    assert np.argmax(linear.predict_categorical_linear(model, x), axis=1).tolist() == [15] * 128
    singleton = linear.fit_categorical_linear(np.array([[1.0]]), np.array([[1.0]]), 2)
    np.testing.assert_allclose(singleton.coefficients, [[0.0, 0.5], [0.0, 0.5]])
    assert singleton.rank == 1 and singleton.squared_error == pytest.approx(0.0, abs=1e-25)


def test_should_preserve_inputs_global_rng_and_model_when_outputs_or_inputs_are_changed():
    x, y = data()
    before = (x.copy(), y.copy(), pickle.dumps(np.random.get_state()), random.getstate())
    model = linear.fit_categorical_linear(x, y, 2)
    saved = pickle.dumps(model)
    probabilities = linear.predict_categorical_linear(model, x)
    probabilities.fill(0)
    np.testing.assert_array_equal(x, before[0])
    np.testing.assert_array_equal(y, before[1])
    assert pickle.dumps(np.random.get_state()) == before[2] and random.getstate() == before[3]
    x.fill(1)
    y.fill(0)
    assert pickle.dumps(model) == saved
    with pytest.raises((FrozenInstanceError, AttributeError)):
        setattr(model, "rank", 0)


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "float32",
        "array_subclass",
        "flat",
        "empty",
        "too_many_rows",
        "no_columns",
        "too_wide",
        "nan",
        "inf",
        "above",
        "below",
        "target_list",
        "target_dtype",
        "target_flat",
        "target_rows",
        "target_nan",
        "target_fraction",
        "target_negative",
        "target_outside",
        "classes_bool",
        "classes_numpy",
        "classes_low",
        "classes_high",
    ],
)
def test_should_refuse_invalid_fit_admission_before_any_solver(monkeypatch, fault):
    x: Any
    y: Any
    x, y = data()
    classes: Any = 2
    if fault == "list":
        x = x.tolist()
    elif fault == "float32":
        x = x.astype(np.float32)
    elif fault == "array_subclass":

        class ArraySubclass(np.ndarray):
            pass

        x = x.view(ArraySubclass)
    elif fault == "flat":
        x = x[:, 0]
    elif fault in ("empty", "too_many_rows", "no_columns", "too_wide"):
        rows, columns = {
            "empty": (0, 1),
            "too_many_rows": (129, 1),
            "no_columns": (3, 0),
            "too_wide": (3, 297),
        }[fault]
        x, y = np.zeros((rows, columns)), np.zeros((rows, 1))
    elif fault in ("nan", "inf", "above", "below"):
        x[0, 0] = {"nan": np.nan, "inf": np.inf, "above": 1.0001, "below": -1.0001}[fault]
    elif fault == "target_list":
        y = y.tolist()
    elif fault == "target_dtype":
        y = y.astype(np.float32)
    elif fault == "target_flat":
        y = y[:, 0]
    elif fault == "target_rows":
        y = y[:-1]
    elif fault.startswith("target_"):
        y[0, 0] = {
            "target_nan": np.nan,
            "target_fraction": 0.5,
            "target_negative": -1,
            "target_outside": 2,
        }[fault]
    else:
        classes = {
            "classes_bool": True,
            "classes_numpy": np.int64(2),
            "classes_low": 1,
            "classes_high": 17,
        }[fault]
    monkeypatch.setattr(np.linalg, "lstsq", lambda *a, **k: pytest.fail("invalid data solved"))
    with pytest.raises(ValueError):
        linear.fit_categorical_linear(x, y, classes)


@pytest.mark.parametrize(
    "field,value",
    [
        ("feature_dim", True),
        ("feature_dim", 297),
        ("num_classes", 1),
        ("fit_rows", 129),
        ("rank", True),
        ("rank", 3),
        ("singular_values", (2.0,)),
        ("singular_values", (2.0, float("nan"))),
        ("squared_error", -0.1),
        ("squared_error", float("inf")),
        ("coefficients", ((1.0, 2.0),)),
        ("coefficients", ((1.0, 2.0, 3.0), (1.0, 2.0, float("nan")))),
    ],
)
def test_should_refuse_tampered_model_before_prediction_without_repair(field, value):
    model = manual_model()
    object.__setattr__(model, field, value)
    with pytest.raises(ValueError):
        linear.predict_categorical_linear(model, np.array([[0.0]]))
    assert getattr(model, field) is value


@pytest.mark.parametrize("fault", ["object", "extra", "coefficient_int", "width", "dtype", "range"])
def test_should_refuse_nonexact_models_and_prediction_geometry(fault):
    model: Any = manual_model()
    x = np.array([[0.0]])
    if fault == "object":
        model = object()
    elif fault == "extra":
        with pytest.raises(AttributeError):
            object.__setattr__(model, "foreign", 1)
        return
    elif fault == "coefficient_int":
        object.__setattr__(model, "coefficients", ((1000, 0.0, -1000.0), (0.0, 0.0, 0.0)))
    elif fault == "width":
        x = np.zeros((1, 2))
    elif fault == "dtype":
        x = x.astype(np.float32)
    else:
        x[0, 0] = 1.1
    with pytest.raises(ValueError):
        linear.predict_categorical_linear(model, x)
