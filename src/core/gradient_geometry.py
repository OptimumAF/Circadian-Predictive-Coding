"""Compare bounded supplied gradients without changing their signs or inputs.

Inputs are matching tuples of finite float32 matrices; output is scalar geometry.
This pure reducer knows neither objectives nor learners, data roles or causality.
"""

from math import sqrt
from typing import Any

import numpy as np


def _validate_pair(first: tuple, second: tuple) -> None:
    if (
        type(first) is not tuple
        or type(second) is not tuple
        or not 1 <= len(first) <= 5
        or len(first) != len(second)
    ):
        raise ValueError("gradients require matching exact tuples with 1..5 matrices")
    cells = 0
    for left, right in zip(first, second, strict=True):
        for value in (left, right):
            if (
                type(value) is not np.ndarray
                or value.dtype != np.float32
                or value.ndim != 2
                or not value.size
            ):
                raise ValueError("gradient matrices must be nonempty finite exact float32 arrays")
        if left.shape != right.shape:
            raise ValueError("corresponding gradient shapes must match")
        cells += left.size
        if cells > 65536:
            raise ValueError("each gradient tuple exceeds 65536 cells")
    # Admit the entire bounded shape before allocating any finite-check masks.
    for left, right in zip(first, second, strict=True):
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("gradient matrices must be finite")


def compare_gradient_geometry(
    first: tuple[np.ndarray, ...], second: tuple[np.ndarray, ...]
) -> dict[str, Any]:
    """Return signed dot/cosine and norms; zero norms have undefined cosine.

    Why: conversion precedes multiplication, so finite large or subnormal
    float32 gradients retain their magnitude in the float64 accumulation.
    """
    _validate_pair(first, second)
    first_squared = second_squared = dot = 0.0
    for left, right in zip(first, second, strict=True):
        a, b = left.astype(np.float64), right.astype(np.float64)
        first_squared += float(np.sum(a * a, dtype=np.float64))
        second_squared += float(np.sum(b * b, dtype=np.float64))
        dot += float(np.sum(a * b, dtype=np.float64))
    first_norm, second_norm = sqrt(first_squared), sqrt(second_squared)
    zero = first_norm == 0 or second_norm == 0
    # Only roundoff at the mathematical endpoints is bounded; negative values
    # remain signed evidence, rather than being converted into magnitudes.
    cosine = None if zero else max(-1.0, min(1.0, dot / (first_norm * second_norm)))
    return dict(
        first_norm=first_norm,
        second_norm=second_norm,
        dot=dot,
        cosine=cosine,
        cosine_reason="zero_norm" if zero else None,
    )
