"""Pure signed gradient geometry uses bounded float32 inputs and float64 math."""

import json
import pickle
from typing import Any

import numpy as np
import pytest

from src.core.gradient_geometry import compare_gradient_geometry


def array(values):
    return np.array(values, dtype=np.float32)


@pytest.mark.parametrize(
    "first,second,norm1,norm2,dot,cosine",
    [
        (
            (array([[1, 2], [3, 4]]),),
            (array([[2, 4], [6, 8]]),),
            np.sqrt(30),
            2 * np.sqrt(30),
            60,
            1,
        ),
        (
            (array([[1, 2], [3, 4]]),),
            (array([[-1, -2], [-3, -4]]),),
            np.sqrt(30),
            np.sqrt(30),
            -30,
            -1,
        ),
        ((array([[1, 0]]),), (array([[0, 2]]),), 1, 2, 0, 0),
        ((array([[1, 2]]),), (array([[1, -2]]),), np.sqrt(5), np.sqrt(5), -3, -0.6),
        ((array([[3]]), array([[4]])), (array([[6]]), array([[8]])), 5, 10, 50, 1),
    ],
)
def test_should_match_analytical_joint_signed_geometry(first, second, norm1, norm2, dot, cosine):
    before = pickle.dumps((first, second))
    result = compare_gradient_geometry(first, second)
    assert set(result) == {"first_norm", "second_norm", "dot", "cosine", "cosine_reason"}
    assert result["first_norm"] == pytest.approx(norm1)
    assert result["second_norm"] == pytest.approx(norm2)
    assert result["dot"] == pytest.approx(dot)
    assert result["cosine"] == pytest.approx(cosine) and result["cosine_reason"] is None
    assert all(type(result[key]) is float for key in ("first_norm", "second_norm", "dot", "cosine"))
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert pickle.dumps((first, second)) == before


@pytest.mark.parametrize("zero_first", [False, True])
def test_should_expose_undefined_cosine_for_either_zero_norm(zero_first):
    zero, nonzero = (array([[0, 0]]),), (array([[3, 4]]),)
    first, second = (zero, nonzero) if zero_first else (nonzero, zero)
    result = compare_gradient_geometry(first, second)
    assert result["first_norm"] == (0.0 if zero_first else 5.0)
    assert result["second_norm"] == (5.0 if zero_first else 0.0)
    assert result["dot"] == 0 and result["cosine"] is None
    assert result["cosine_reason"] == "zero_norm"
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("scale", [1e30, 1e-30])
def test_should_accumulate_large_and_tiny_float32_gradients_without_float32_overflow_or_underflow(
    scale,
):
    first = (array([[scale, -scale]]),)
    second = (array([[scale, -scale]]),)
    exact = float(first[0][0, 0])
    result = compare_gradient_geometry(first, second)
    assert result["first_norm"] == pytest.approx(np.sqrt(2) * abs(exact), rel=1e-12, abs=0)
    assert result["dot"] == pytest.approx(2 * exact * exact, rel=1e-12, abs=0)
    assert result["dot"] > 0 and result["cosine"] == pytest.approx(1.0)
    json.dumps(result, allow_nan=False)


def test_should_accept_readonly_noncontiguous_views_without_alias_mutation():
    backing = np.arange(12, dtype=np.float32).reshape(3, 4)
    first, second = backing[:, ::2], -backing[:, ::2]
    first.setflags(write=False)
    second.setflags(write=False)
    before = backing.tobytes(), second.tobytes()
    result = compare_gradient_geometry((first,), (second,))
    assert result["dot"] < 0 and result["cosine"] == pytest.approx(-1.0)
    assert (backing.tobytes(), second.tobytes()) == before
    assert not first.flags.writeable and not second.flags.writeable


def test_should_accept_exact_aggregate_cell_boundary_and_all_five_parameter_arrays():
    shapes = ((1, 16384), (1, 16384), (1, 16384), (1, 8192), (1, 8192))
    first = tuple(np.ones(shape, dtype=np.float32) for shape in shapes)
    result = compare_gradient_geometry(first, first)
    assert result["first_norm"] == result["second_norm"] == 256.0
    assert result["dot"] == 65536.0 and result["cosine"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "array",
        "empty",
        "six",
        "count",
        "shape",
        "one_dim",
        "three_dim",
        "empty_rows",
        "empty_columns",
        "float64",
        "float16",
        "int32",
        "bool",
        "nan",
        "inf",
        "negative_inf",
        "array_cell_limit",
        "aggregate_cell_limit",
    ],
)
def test_should_refuse_invalid_pair_before_changing_any_inputs(fault):
    first: Any = (array([[1, 2]]),)
    second: Any = (array([[3, 4]]),)
    if fault == "list":
        first = list(first)
    elif fault == "array":
        first = first[0]
    elif fault == "empty":
        first, second = (), ()
    elif fault == "six":
        first, second = first * 6, second * 6
    elif fault == "count":
        second = second * 2
    elif fault == "shape":
        second = (array([[1], [2]]),)
    elif fault == "one_dim":
        first = (array([1, 2]),)
    elif fault == "three_dim":
        first = (array([[[1, 2]]]),)
    elif fault in ("empty_rows", "empty_columns"):
        first = (np.zeros((0, 2) if fault == "empty_rows" else (2, 0), dtype=np.float32),)
    elif fault in ("float64", "float16", "int32", "bool"):
        first = (first[0].astype(getattr(np, fault) if fault != "bool" else np.bool_),)
    elif fault in ("nan", "inf", "negative_inf"):
        first[0][0, 0] = {"nan": np.nan, "inf": np.inf, "negative_inf": -np.inf}[fault]
    elif fault == "array_cell_limit":
        first = second = (np.ones((1, 65537), dtype=np.float32),)
    else:
        first = second = (
            np.ones((1, 32768), dtype=np.float32),
            np.ones((1, 32769), dtype=np.float32),
        )
    before = pickle.dumps((first, second))
    with pytest.raises(ValueError):
        compare_gradient_geometry(first, second)
    assert pickle.dumps((first, second)) == before


def test_should_validate_second_tuple_and_every_array_not_only_first_parameter():
    first = (array([[1]]), array([[2]]))
    second = (array([[1]]), array([[np.nan]]))
    with pytest.raises(ValueError):
        compare_gradient_geometry(first, second)
    with pytest.raises(ValueError):
        bad_second: Any = list(first)
        compare_gradient_geometry(first, bad_second)


@pytest.mark.parametrize("aggregate", [False, True])
def test_should_refuse_cell_overflow_before_any_finite_scan_or_copy(monkeypatch, aggregate):
    gradients = (
        (np.ones((1, 32768), dtype=np.float32), np.ones((1, 32769), dtype=np.float32))
        if aggregate
        else (np.ones((1, 65537), dtype=np.float32),)
    )
    before = pickle.dumps(gradients)

    def forbidden(*args, **kwargs):
        pytest.fail("bounded size admission must precede even the first finite scan")

    with monkeypatch.context() as context:
        context.setattr(np, "isfinite", forbidden)
        with pytest.raises(ValueError, match="65536"):
            compare_gradient_geometry(gradients, gradients)
    assert pickle.dumps(gradients) == before
