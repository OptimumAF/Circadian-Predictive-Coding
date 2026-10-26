"""Owned fixed-width histories of committed coupled-energy gradient magnitudes.

Importance is an EMA of the absolute batch-aggregate outgoing gradient before
plasticity gating, not activity, chemistry or causal importance. Ages count all
committed native calls, including replay. No model, runtime, IO or RNG is owned.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from src.core.coupled_energy import Array

MAX_HISTORY_UPDATES = 2**53 - 1


def _width(value: int) -> int:
    if type(value) is not int or not 1 <= value <= 4096:
        raise ValueError("unit history width must be a native integer in 1..4096")
    return value


def _decay(value: float) -> float:
    if type(value) not in {int, float} or not 0 <= value < 1 or not isfinite(value):
        raise ValueError("unit history ema_decay must be a finite native number in [0, 1)")
    return float(value)


@dataclass(frozen=True)
class UnitHistory:
    format_version: int
    ema_decay: float
    unit_ids: NDArray[np.int64]
    importance_ema: Array
    update_age: Array
    updates: int

    def __post_init__(self) -> None:
        _validate(self)
        object.__setattr__(self, "ema_decay", _decay(self.ema_decay))
        for name in ("unit_ids", "importance_ema", "update_age"):
            object.__setattr__(self, name, getattr(self, name).copy())


def _validate(history: UnitHistory) -> int:
    if type(history) is not UnitHistory:
        raise ValueError("unit history must be an exact UnitHistory record")
    if type(history.format_version) is not int or history.format_version != 1:
        raise ValueError("unsupported unit history format")
    _decay(history.ema_decay)
    if type(history.updates) is not int or not 0 <= history.updates <= MAX_HISTORY_UPDATES:
        raise ValueError("unit history updates must be a native integer in 0..2**53-1")
    ids = history.unit_ids
    if type(ids) is not np.ndarray or ids.dtype != np.int64 or ids.ndim != 1:
        raise ValueError("unit history IDs must be an int64 vector")
    width = _width(len(ids))
    for values in (history.importance_ema, history.update_age):
        if type(values) is not np.ndarray or values.dtype != np.float64 or values.shape != (width,):
            raise ValueError("unit history values must be aligned float64 vectors")
    if not np.array_equal(ids, np.arange(width, dtype=np.int64)):
        raise ValueError("fixed-width unit history IDs must equal arange(width)")
    for values in (history.importance_ema, history.update_age):
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("unit history importance and ages must be finite and nonnegative")
    if np.any(history.update_age != float(history.updates)):
        raise ValueError("fixed-width unit ages must equal the committed update count")
    if history.updates == 0 and np.any(history.importance_ema != 0):
        raise ValueError("zero-update unit history requires zero importance")
    return width


def initial_unit_history(width: int, ema_decay: float = 0.95) -> UnitHistory:
    width, decay = _width(width), _decay(ema_decay)
    return UnitHistory(
        1,
        decay,
        np.arange(width, dtype=np.int64),
        np.zeros(width, dtype=np.float64),
        np.zeros(width, dtype=np.float64),
        0,
    )


def copy_unit_history(history: UnitHistory, width: int, ema_decay: float) -> UnitHistory:
    width, decay = _width(width), _decay(ema_decay)
    if _validate(history) != width or history.ema_decay != decay:
        raise ValueError("unit history width or EMA policy differs from this model")
    return UnitHistory(
        1, decay, history.unit_ids, history.importance_ema, history.update_age, history.updates
    )


def advance_unit_history(history: UnitHistory, outgoing_gradient: Array) -> UnitHistory:
    width = _validate(history)
    if history.updates == MAX_HISTORY_UPDATES:
        raise ValueError("unit history update counter is exhausted")
    if (
        type(outgoing_gradient) is not np.ndarray
        or outgoing_gradient.dtype != np.float64
        or outgoing_gradient.shape != (width, 1)
        or not np.isfinite(outgoing_gradient).all()
    ):
        raise ValueError("outgoing gradient must be a finite float64 matrix of shape (width, 1)")
    magnitude = np.abs(outgoing_gradient[:, 0])
    scale = np.maximum(history.importance_ema, magnitude)
    # A scaled convex combination avoids overflowing finite representable EMAs.
    with np.errstate(under="ignore", over="raise", invalid="raise"):
        old = np.divide(history.importance_ema, scale, out=np.zeros(width), where=scale != 0)
        new = np.divide(magnitude, scale, out=np.zeros(width), where=scale != 0)
        fraction = history.ema_decay * old + (1 - history.ema_decay) * new
        importance = scale * np.clip(fraction, 0.0, 1.0)
    updates = history.updates + 1
    return UnitHistory(
        1, history.ema_decay, history.unit_ids, importance, np.full(width, float(updates)), updates
    )
