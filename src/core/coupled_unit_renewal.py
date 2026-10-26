"""Bounded renewable unit histories and deterministic mature/low-gradient choice.

History records native updates and replacement identity, not permission to spend
a replacement allowance. Selection uses no data, predictions, context or RNG.
Importance remains the absolute batch-aggregate outgoing gradient EMA before gates.
"""

from dataclasses import dataclass, replace
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from src.core.coupled_energy import Array
from src.core.coupled_unit_history import MAX_HISTORY_UPDATES, _decay, _width

MAX_NEXT_UNIT_ID = 2**63 - 1


@dataclass(frozen=True)
class RenewalPolicy:
    min_age_updates: int
    max_importance: float
    max_per_event: int
    seed: int = 29

    def __post_init__(self) -> None:
        if (
            type(self.min_age_updates) is not int
            or not 1 <= self.min_age_updates <= MAX_HISTORY_UPDATES
        ):
            raise ValueError("renewal minimum age must be a native integer in 1..2**53-1")
        if type(self.max_importance) not in {int, float} or self.max_importance < 0:
            raise ValueError("renewal maximum importance must be finite, native and nonnegative")
        try:
            finite = isfinite(self.max_importance)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("renewal maximum importance must be finite")
        if type(self.max_per_event) is not int or not 0 <= self.max_per_event <= 4096:
            raise ValueError("renewal max_per_event must be a native integer in 0..4096")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("renewal seed must be a native integer in 0..2**32-1")


def copy_renewal_policy(policy: RenewalPolicy) -> RenewalPolicy:
    if type(policy) is not RenewalPolicy:
        raise ValueError("renewal requires an exact RenewalPolicy")
    return replace(policy)


@dataclass(frozen=True)
class RenewableUnitHistory:
    format_version: int
    ema_decay: float
    unit_ids: NDArray[np.int64]
    importance_ema: Array
    update_age: Array
    updates: int
    next_unit_id: int

    def __post_init__(self) -> None:
        _validate(self)
        object.__setattr__(self, "ema_decay", _decay(self.ema_decay))
        for name in ("unit_ids", "importance_ema", "update_age"):
            object.__setattr__(self, name, getattr(self, name).copy())


def _validate(history: RenewableUnitHistory) -> int:
    if type(history) is not RenewableUnitHistory:
        raise ValueError("renewal history must be an exact RenewableUnitHistory")
    if type(history.format_version) is not int or history.format_version != 2:
        raise ValueError("unsupported renewable unit history format")
    _decay(history.ema_decay)
    if type(history.updates) is not int or not 0 <= history.updates <= MAX_HISTORY_UPDATES:
        raise ValueError("renewal history update count must be a native integer in 0..2**53-1")
    if type(history.next_unit_id) is not int or not 1 <= history.next_unit_id <= MAX_NEXT_UNIT_ID:
        raise ValueError("renewal next_unit_id must be a native integer in 1..2**63-1")
    ids = history.unit_ids
    if type(ids) is not np.ndarray or ids.dtype != np.int64 or ids.ndim != 1:
        raise ValueError("renewal unit IDs must be an int64 vector")
    width = _width(len(ids))
    for values in (history.importance_ema, history.update_age):
        if type(values) is not np.ndarray or values.dtype != np.float64 or values.shape != (width,):
            raise ValueError("renewal history values must be aligned float64 vectors")
    if np.any(ids < 0) or np.any(ids >= history.next_unit_id) or len(np.unique(ids)) != width:
        raise ValueError("renewal unit IDs must be unique, nonnegative and below next_unit_id")
    for values in (history.importance_ema, history.update_age):
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("renewal importance and ages must be finite and nonnegative")
    if np.any(history.update_age > history.updates) or np.any(
        history.update_age != np.floor(history.update_age)
    ):
        raise ValueError(
            "renewal ages must be integral native-update counts no greater than updates"
        )
    if np.any(history.importance_ema[history.update_age == 0] != 0):
        raise ValueError("newborn units require zero importance")
    return width


def initial_renewal_history(width: int, ema_decay: float = 0.95) -> RenewableUnitHistory:
    width, decay = _width(width), _decay(ema_decay)
    return RenewableUnitHistory(
        2, decay, np.arange(width, dtype=np.int64), np.zeros(width), np.zeros(width), 0, width
    )


def copy_renewal_history(
    history: RenewableUnitHistory, width: int, ema_decay: float
) -> RenewableUnitHistory:
    width, decay = _width(width), _decay(ema_decay)
    if _validate(history) != width or history.ema_decay != decay:
        raise ValueError("renewal history width or EMA policy differs from this model")
    return RenewableUnitHistory(
        2,
        decay,
        history.unit_ids,
        history.importance_ema,
        history.update_age,
        history.updates,
        history.next_unit_id,
    )


def advance_renewal_history(
    history: RenewableUnitHistory, outgoing_gradient: Array
) -> RenewableUnitHistory:
    width = _validate(history)
    if history.updates == MAX_HISTORY_UPDATES:
        raise ValueError("renewal history update counter is exhausted")
    if (
        type(outgoing_gradient) is not np.ndarray
        or outgoing_gradient.dtype != np.float64
        or outgoing_gradient.shape != (width, 1)
        or not np.isfinite(outgoing_gradient).all()
    ):
        raise ValueError("outgoing gradient must be a finite float64 matrix of shape (width, 1)")
    magnitude = np.abs(outgoing_gradient[:, 0])
    scale = np.maximum(history.importance_ema, magnitude)
    with np.errstate(under="ignore", over="raise", invalid="raise"):
        old = np.divide(history.importance_ema, scale, out=np.zeros(width), where=scale != 0)
        new = np.divide(magnitude, scale, out=np.zeros(width), where=scale != 0)
        # Same scaled convex EMA as tracked fixed-width history, without false overflow.
        importance = scale * np.clip(history.ema_decay * old + (1 - history.ema_decay) * new, 0, 1)
    return RenewableUnitHistory(
        2,
        history.ema_decay,
        history.unit_ids,
        importance,
        history.update_age + 1.0,
        history.updates + 1,
        history.next_unit_id,
    )


def select_renewal_indices(
    history: RenewableUnitHistory, policy: RenewalPolicy, max_units: int
) -> tuple[int, ...]:
    _validate(history)
    policy = copy_renewal_policy(policy)
    if type(max_units) is not int or max_units < 0:
        raise ValueError("renewal allowance must be a native nonnegative integer")
    eligible = [
        index
        for index in range(len(history.unit_ids))
        if float(history.update_age[index]) >= policy.min_age_updates
        and float(history.importance_ema[index]) <= policy.max_importance
    ]
    eligible.sort(
        key=lambda index: (
            float(history.importance_ema[index]),
            int(history.unit_ids[index]),
            index,
        )
    )
    return tuple(eligible[: min(max_units, policy.max_per_event)])


@dataclass(frozen=True)
class RenewalResult:
    indices: tuple[int, ...]
    old_unit_ids: tuple[int, ...]
    new_unit_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        for values in (self.indices, self.old_unit_ids, self.new_unit_ids):
            if (
                type(values) is not tuple
                or len(values) > 4096
                or any(type(value) is not int or value < 0 for value in values)
            ):
                raise ValueError(
                    "renewal results require bounded tuples of native nonnegative integers"
                )
            if len(set(values)) != len(values):
                raise ValueError("renewal result entries must be distinct")
        if not len(self.indices) == len(self.old_unit_ids) == len(self.new_unit_ids):
            raise ValueError("renewal result tuples must align")
        if any(index >= 4096 for index in self.indices) or any(
            value >= MAX_NEXT_UNIT_ID for value in (*self.old_unit_ids, *self.new_unit_ids)
        ):
            raise ValueError("renewal result indices or stable IDs exceed their bounds")
