"""Validate NumPy binary training inputs and finite update candidates.

Why this: the three binary trainers share the same batch contract; checking it
once at each entry point prevents broadcasting and empty-batch NaNs.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import isfinite

import numpy as np
from numpy.typing import NDArray


def require_finite_training_arrays(arrays: Iterable[NDArray[np.float64]], context: str) -> None:
    """Reject a nonfinite model value or candidate before committing a step."""
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise FloatingPointError(f"nonfinite {context}")


def validate_positive_finite_learning_rate(learning_rate: float) -> None:
    """Require a usable weight-step size before training begins."""
    if not isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")


def validate_binary_training_batch(
    features: NDArray[np.float64], targets: NDArray[np.float64], input_dim: int
) -> None:
    """Require a nonempty real binary/soft-label batch with exact dimensions."""
    if not isinstance(features, np.ndarray) or features.ndim != 2:
        raise ValueError("features must be 2D NumPy arrays")
    if features.shape[0] == 0:
        raise ValueError("training batch must be nonempty")
    if features.shape[1] != input_dim:
        raise ValueError(f"feature width must be {input_dim}; got {features.shape[1]}")
    if not isinstance(targets, np.ndarray) or targets.ndim != 2:
        raise ValueError("targets must be 2D NumPy arrays")
    if targets.shape[1] != 1:
        raise ValueError("targets must have one column")
    if targets.shape[0] != features.shape[0]:
        raise ValueError("features and targets must have the same batch size")
    if any(
        not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating)
        for array in (features, targets)
    ):
        raise ValueError("features and targets must be real numeric arrays")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
        raise ValueError("features and targets must be finite")
    if np.any((targets < 0.0) | (targets > 1.0)):
        raise ValueError("targets must be between 0 and 1")
