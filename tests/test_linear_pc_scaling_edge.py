"""Independent review regression for a representable update product (one step)."""

from math import ldexp

import numpy as np

from src.core.linear_pc_scaling import (
    LinearScalingConfig,
    LinearScalingState,
    effective_linear_weights,
    step_linear_pc,
)


def test_should_preserve_representable_raw_delta_when_partial_rate_product_underflows():
    gamma = ldexp(1.0, -530)
    config = LinearScalingConfig("mean_field", 1, 1, 1, gamma)
    state = LinearScalingState(config, (np.array([[1.0]]), np.array([[gamma]])))
    rate = ldexp(1.0, -20)
    assert rate * config.raw_learning_rate_factor == 0.0
    result = step_linear_pc(state, np.array([[1.0]]), np.array([[0.0]]), rate)
    # Effective gradient .25, effective output-rate factor 1: delta=2**-22.
    np.testing.assert_array_equal(result.evaluation.weight_gradients[1], [[0.25]])
    np.testing.assert_array_equal(
        effective_linear_weights(result.state)[1], [[1.0 - ldexp(1.0, -22)]]
    )
    np.testing.assert_array_equal(result.state.raw_weights[1], [[gamma - ldexp(1.0, -552)]])
    np.testing.assert_array_equal(state.raw_weights[0], [[1.0]])
    np.testing.assert_array_equal(state.raw_weights[1], [[gamma]])
