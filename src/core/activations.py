"""Activation functions used by both learning approaches."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def sigmoid(values: Array) -> Array:
    """Compute sigmoid activation with stable clipping."""
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def sigmoid_derivative_from_linear(values: Array) -> Array:
    """Derivative of sigmoid with respect to its linear input."""
    activated = sigmoid(values)
    return activated * (1.0 - activated)


def tanh(values: Array) -> Array:
    """Compute tanh activation."""
    return np.tanh(values)


def tanh_derivative_from_linear(values: Array) -> Array:
    """Derivative of tanh with respect to its linear input."""
    activated = np.tanh(values)
    return 1.0 - np.square(activated)

