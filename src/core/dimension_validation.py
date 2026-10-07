"""Validate model widths before NumPy or Torch allocates parameter arrays."""

from __future__ import annotations

from numbers import Integral


def require_positive_integer_dimension(value: object, name: str) -> int:
    """Accept Python/NumPy integral widths, excluding booleans and fractions."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
