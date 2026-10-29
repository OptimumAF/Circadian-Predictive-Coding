"""Literal coefficient admission; no solver, training or model prediction."""

from dataclasses import replace

import numpy as np
import pytest

from src.core.categorical_linear_fit import CategoricalLinearFit

ERROR = "complete immutable finite float coefficients"


class TupleSubclass(tuple):
    pass


def model():
    return CategoricalLinearFit(296, 7, 9, ((0.0,) * 7,) * 297, 1, (1.0,) * 9, 0.0)


@pytest.mark.parametrize("width,classes,rows", [(1, 2, 1), (296, 7, 9)])
def test_should_accept_complete_finite_literal_coefficients(width, classes, rows):
    coefficients = tuple((0.25,) * classes for _ in range(width + 1))
    owned = CategoricalLinearFit(width, classes, rows, coefficients, 1, (1.0,) * rows, 0.0)
    owned.__post_init__()
    assert owned.coefficients == coefficients and owned.coefficients is coefficients


@pytest.mark.parametrize("kind", ["list", "short", "long", "subclass"])
def test_should_refuse_inexact_or_wrong_length_outer_coefficients(kind):
    original = model()
    coefficients = {
        "list": list(original.coefficients),
        "short": original.coefficients[:-1],
        "long": original.coefficients + (original.coefficients[-1],),
        "subclass": TupleSubclass(original.coefficients),
    }[kind]
    with pytest.raises(ValueError, match=ERROR):
        replace(original, coefficients=coefficients)


@pytest.mark.parametrize("position", [0, 148, 296])
@pytest.mark.parametrize("row", [[0.0] * 7, None, (0.0,) * 6, (0.0,) * 8])
def test_should_refuse_malformed_rows_anywhere_in_coefficients(position, row):
    original = model()
    coefficients = list(original.coefficients)
    coefficients[position] = row
    with pytest.raises(ValueError, match=ERROR):
        replace(original, coefficients=tuple(coefficients))


@pytest.mark.parametrize("position,column", [(0, 0), (148, 3), (296, 6)])
@pytest.mark.parametrize(
    "value", [True, 0, np.float64(0.0), float("nan"), float("inf"), -float("inf"), "0"]
)
def test_should_refuse_nonfinite_or_inexact_scalars_anywhere(position, column, value):
    original = model()
    row = list(original.coefficients[position])
    row[column] = value
    coefficients = list(original.coefficients)
    coefficients[position] = tuple(row)
    with pytest.raises(ValueError, match=ERROR):
        replace(original, coefficients=tuple(coefficients))


@pytest.mark.parametrize("position", [0, 148, 296])
def test_should_revalidate_corruption_after_a_preceding_success(position):
    owned = model()
    owned.__post_init__()
    coefficients = list(owned.coefficients)
    coefficients[position] = (float("nan"),) + (0.0,) * 6
    object.__setattr__(owned, "coefficients", tuple(coefficients))
    with pytest.raises(ValueError, match=ERROR):
        owned.__post_init__()
