"""Independent active-softmax oracles; literal coefficients, no fitting."""

from dataclasses import asdict

import numpy as np
import pytest

from src.core.wide_categorical_linear_fit import (
    WideCategoricalLinearFit,
    predict_wide_categorical_linear,
    predict_wide_categorical_active,
)


def model(bias=(10000.0, 1.0, 2.0, 3.0)):
    # Readout-only oracle, not an acquired model or native task result.
    return WideCategoricalLinearFit(2, 4, 1, ((0.0,) * 4, (0.0,) * 4, tuple(bias)), 1, (1.0,), 7.0)


@pytest.mark.parametrize(
    "columns,bias", [((1, 2, 3), (10000.0, 1.0, 2.0, 3.0)), ((0, 1, 3), (1.0, 2.0, 10000.0, 3.0))]
)
def test_should_mask_inactive_extreme_before_softmax_without_touching_fit(columns, bias):
    source = model(bias)
    before = asdict(source)
    x = np.array([[0.6, 0.8]])
    probabilities = predict_wide_categorical_active(source, x, columns)
    values = np.exp(np.array([1.0, 2.0, 3.0]) - 3)
    np.testing.assert_allclose(probabilities, [values / values.sum()], atol=1e-14)
    assert asdict(source) == before and source.squared_error == 7.0 and source.fit_rows == 1
    np.testing.assert_array_equal(x, [[0.6, 0.8]])


def test_should_equal_original_readout_for_all_columns():
    source = model((0.0, 1.0, 2.0, 3.0))
    x = np.zeros((1, 2))
    np.testing.assert_array_equal(
        predict_wide_categorical_active(source, x, (0, 1, 2, 3)),
        predict_wide_categorical_linear(source, x),
    )


@pytest.mark.parametrize("columns", [(), (1,), (1, 1), (2, 1), (1, 4), (False, 1), [0, 1]])
def test_should_refuse_nonoriginal_column_declarations(columns):
    with pytest.raises(ValueError, match="ordered"):
        predict_wide_categorical_active(model(), np.zeros((1, 2)), columns)
