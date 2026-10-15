"""One exact step verifies the residual entry point preserves stable products."""

from math import ldexp

import numpy as np

from src.core.residual_linear_pc import (
    LinearResidualConfig,
    LinearResidualState,
    effective_residual_weights,
    step_residual_linear_pc,
)


def test_should_preserve_representable_delta_in_the_no_middle_block_case():
    gamma = ldexp(1.0, -530)
    config = LinearResidualConfig("mean_field", 1, 1, 1, gamma)
    state = LinearResidualState(config, (np.array([[1.0]]), np.array([[gamma]])))
    rate = ldexp(1.0, -20)
    assert rate * config.raw_learning_rate_factor == 0.0
    result = step_residual_linear_pc(state, np.array([[1.0]]), np.array([[0.0]]), rate)
    np.testing.assert_array_equal(result.evaluation.weight_gradients[1], [[0.25]])
    np.testing.assert_array_equal(
        effective_residual_weights(result.state)[1], [[1.0 - ldexp(1.0, -22)]]
    )
    np.testing.assert_array_equal(result.state.raw_weights[1], [[gamma - ldexp(1.0, -552)]])
    np.testing.assert_array_equal(state.raw_weights[0], [[1.0]])
    np.testing.assert_array_equal(state.raw_weights[1], [[gamma]])
