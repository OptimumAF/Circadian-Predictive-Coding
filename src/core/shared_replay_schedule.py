"""Prediction-independent retained rows and recent replay selection.

Inputs are arrived training arrays, one bounded retention policy, and a
predeclared per-sleep update limit. Outputs are detached selected rows and
retention facts. This module does not train models, access decision roles,
or decide when sleep occurs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.core.circadian_predictive_coding import (
    ReplayRetentionBudget,
    ReplayRetentionSnapshot,
    replay_sample_id,
)
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.training_validation import validate_binary_training_batch


Array = NDArray[np.float64]


@dataclass(frozen=True)
class SharedReplaySelection:
    """One ordered replay batch list; each consumer gets detached arrays."""

    sample_ids: tuple[str, ...]
    _rows: tuple[tuple[Array, Array], ...] = field(repr=False, compare=False)

    def training_batches(self) -> tuple[tuple[Array, Array], ...]:
        """Give one model private copies of the same selected labeled rows."""
        return tuple((inputs.copy(), targets.copy()) for inputs, targets in self._rows)


class SharedReplayBuffer:
    """Retain distinct labeled rows with the existing bounded FIFO/bottom-k rule."""

    def __init__(
        self, input_dim: int, budget: ReplayRetentionBudget, policy: ReplayRetentionPolicy
    ) -> None:
        if type(input_dim) is not int or input_dim <= 0:
            raise ValueError("shared replay input_dim must be positive")
        if type(budget) is not ReplayRetentionBudget or type(policy) is not ReplayRetentionPolicy:
            raise TypeError("shared replay requires a retention budget and policy")
        if budget.max_bytes < 8 * (input_dim + 1):
            raise ValueError("shared replay byte budget cannot hold one float64 labeled row")
        self.input_dim = input_dim
        self.budget = budget
        self.policy = policy
        self._rows: dict[str, tuple[Array, Array]] = {}

    @property
    def retention(self) -> ReplayRetentionSnapshot:
        return ReplayRetentionSnapshot(
            sample_ids=tuple(sorted(self._rows)),
            example_count=len(self._rows),
            retained_bytes=sum(
                inputs.nbytes + targets.nbytes for inputs, targets in self._rows.values()
            ),
        )

    @property
    def retained_order_ids(self) -> tuple[str, ...]:
        """Expose the order used by the fixed newest-retained sampler."""
        return tuple(self._rows)

    def observe_train_batch(self, inputs: Array, targets: Array) -> None:
        """Copy one wake batch after validating every row before mutation."""
        validate_binary_training_batch(inputs, targets, self.input_dim)
        # Why this: use the same float64 copied-array byte scope as the
        # opt-in circadian buffer, regardless of the caller's input dtype.
        copied_inputs = np.ascontiguousarray(inputs, dtype=np.float64)
        copied_targets = np.ascontiguousarray(targets, dtype=np.float64)
        retained = self._rows.copy()
        for index in range(len(copied_inputs)):
            input_row = copied_inputs[index : index + 1].copy()
            target_row = copied_targets[index : index + 1].copy()
            sample_id = replay_sample_id(input_row, target_row)
            if self.policy.name == "recent_fifo":
                retained.pop(sample_id, None)
            retained[sample_id] = (input_row, target_row)
            while (
                len(retained) > self.budget.max_examples
                or _retained_bytes(retained) > self.budget.max_bytes
            ):
                del retained[self.policy.eviction_id(retained)]
        self._rows = retained

    def select_recent(self, max_updates: int) -> SharedReplaySelection:
        """Select newest retained rows, independent of model predictions."""
        if type(max_updates) is not int or max_updates <= 0:
            raise ValueError("shared replay update budget must be positive")
        selected = tuple(self._rows.items())[-max_updates:]
        return SharedReplaySelection(
            sample_ids=tuple(sample_id for sample_id, _ in selected),
            _rows=tuple((inputs.copy(), targets.copy()) for _, (inputs, targets) in selected),
        )


def _retained_bytes(rows: dict[str, tuple[Array, Array]]) -> int:
    return sum(inputs.nbytes + targets.nbytes for inputs, targets in rows.values())
