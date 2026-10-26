"""Prediction-independent retained rows and recent replay selection.

Inputs are arrived training arrays, bounded retention settings, optional recent/
reservoir quotas, and a predeclared per-sleep update limit. Outputs are detached
selected rows and retention facts. This module does not train models, access
decision roles, decide when sleep occurs, or extend native model retention.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace

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


@dataclass(frozen=True)
class ReplayAllocation:
    """Separate row/byte quotas, with detached and revalidated nested budgets."""

    recent: ReplayRetentionBudget
    reservoir: ReplayRetentionBudget

    def __post_init__(self) -> None:
        if (
            type(self.recent) is not ReplayRetentionBudget
            or type(self.reservoir) is not ReplayRetentionBudget
        ):
            raise ValueError("replay allocation requires exact recent and reservoir budgets")
        object.__setattr__(self, "recent", replace(self.recent))
        object.__setattr__(self, "reservoir", replace(self.reservoir))


@dataclass(frozen=True)
class PartitionRetention:
    recent: ReplayRetentionSnapshot
    reservoir: ReplayRetentionSnapshot
    unique: ReplayRetentionSnapshot
    overlap_examples: int


@dataclass(frozen=True)
class ReplayRotationCursor:
    """Next requested partition and one last-scanned reservoir content ID."""

    next_partition: int
    reservoir_after: str | None

    def __post_init__(self) -> None:
        if type(self.next_partition) is not int or self.next_partition not in (0, 1):
            raise ValueError("rotation next_partition must be native 0 (recent) or 1 (reservoir)")
        content_id = self.reservoir_after
        if content_id is not None and (
            type(content_id) is not str
            or len(content_id) != 64
            or any(character not in "0123456789abcdef" for character in content_id)
        ):
            raise ValueError("rotation reservoir_after must be a canonical content ID or None")


class SharedReplayBuffer:
    """Retain distinct labeled rows with the existing bounded FIFO/bottom-k rule."""

    def __init__(
        self,
        input_dim: int,
        budget: ReplayRetentionBudget,
        policy: ReplayRetentionPolicy,
        *,
        allocation: ReplayAllocation | None = None,
    ) -> None:
        if type(input_dim) is not int or input_dim <= 0:
            raise ValueError("shared replay input_dim must be positive")
        if type(budget) is not ReplayRetentionBudget or type(policy) is not ReplayRetentionPolicy:
            raise TypeError("shared replay requires a retention budget and policy")
        owned_allocation = None
        if allocation is not None:
            owned_allocation = _validate_allocation(input_dim, budget, policy, allocation)
            budget, policy = replace(budget), replace(policy)
        if budget.max_bytes < 8 * (input_dim + 1):
            raise ValueError("shared replay byte budget cannot hold one float64 labeled row")
        self.input_dim = input_dim
        self.budget = budget
        self.policy = policy
        self._rows: dict[str, tuple[Array, Array]] = {}
        self._allocation = owned_allocation
        self._recent: dict[str, None] = {}
        self._reservoir: dict[str, None] = {}
        self._rotation_cursor = ReplayRotationCursor(0, None)

    def _partition_config(self) -> ReplayAllocation | None:
        if self._allocation is None:
            return None
        return _validate_allocation(self.input_dim, self.budget, self.policy, self._allocation)

    @property
    def allocation(self) -> ReplayAllocation | None:
        return self._partition_config()

    @property
    def rotation_cursor(self) -> ReplayRotationCursor:
        return replace(self._rotation_cursor)

    @property
    def partition_retention(self) -> PartitionRetention | None:
        if self._partition_config() is None:
            return None
        row_bytes = 8 * (self.input_dim + 1)
        return PartitionRetention(
            _partition_snapshot(self._recent, row_bytes),
            _partition_snapshot(self._reservoir, row_bytes),
            self.retention,
            len(self._recent.keys() & self._reservoir.keys()),
        )

    @property
    def retention(self) -> ReplayRetentionSnapshot:
        self._partition_config()
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
        self._partition_config()
        return tuple(self._rows)

    def observe_train_batch(self, inputs: Array, targets: Array) -> None:
        """Copy one wake batch after validating every row before mutation."""
        allocation = self._partition_config()
        validate_binary_training_batch(inputs, targets, self.input_dim)
        # Why this: use the same float64 copied-array byte scope as the
        # opt-in circadian buffer, regardless of the caller's input dtype.
        copied_inputs = np.ascontiguousarray(inputs, dtype=np.float64)
        copied_targets = np.ascontiguousarray(targets, dtype=np.float64)
        if allocation is not None:
            # Conversion to the retained float64 representation must remain valid.
            validate_binary_training_batch(copied_inputs, copied_targets, self.input_dim)
            self._observe_partitioned(copied_inputs, copied_targets, allocation)
            return
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
        self._partition_config()
        selected = tuple(self._rows.items())[-max_updates:]
        return SharedReplaySelection(
            sample_ids=tuple(sample_id for sample_id, _ in selected),
            _rows=tuple((inputs.copy(), targets.copy()) for _, (inputs, targets) in selected),
        )

    def _observe_partitioned(
        self, inputs: Array, targets: Array, allocation: ReplayAllocation
    ) -> None:
        retained, recent, reservoir = self._rows.copy(), self._recent.copy(), self._reservoir.copy()
        row_bytes = 8 * (self.input_dim + 1)
        for index in range(len(inputs)):
            input_row, target_row = inputs[index : index + 1], targets[index : index + 1]
            sample_id = replay_sample_id(input_row, target_row)
            if sample_id not in retained:
                retained[sample_id] = (input_row.copy(), target_row.copy())
            recent.pop(sample_id, None)
            recent[sample_id] = None
            reservoir[sample_id] = None
            _trim_partition(recent, allocation.recent, row_bytes, None)
            _trim_partition(reservoir, allocation.reservoir, row_bytes, self.policy)
            # Membership overlap pays both quotas, but arrays live only in this union.
            ordered = (*[key for key in reservoir if key not in recent], *recent)
            retained = {key: retained[key] for key in ordered}
        self._rows, self._recent, self._reservoir = retained, recent, reservoir

    def select_partitioned(self, max_updates: int) -> SharedReplaySelection:
        """Alternate distinct recent-newest and ascending-rank reservoir proposals.

        Duplicate IDs are skipped within a partition's turn; an exhausted
        partition falls back to the other. No prediction or context is consulted.
        """
        if type(max_updates) is not int or max_updates <= 0:
            raise ValueError("shared replay update budget must be positive")
        if self._partition_config() is None:
            raise ValueError("partitioned selection requires a configured replay allocation")
        sources = (
            iter(reversed(self._recent)),
            iter(sorted(self._reservoir, key=lambda key: (self.policy._reservoir_rank(key), key))),
        )
        selected: list[str] = []
        seen: set[str] = set()
        turn = 0
        while len(selected) < min(max_updates, len(self._rows)):
            sample_id = _next_distinct(sources[turn], seen)
            if sample_id is None:
                sample_id = _next_distinct(sources[1 - turn], seen)
            if sample_id is None:
                break
            selected.append(sample_id)
            seen.add(sample_id)
            turn = 1 - turn
        return SharedReplaySelection(
            tuple(selected),
            tuple((self._rows[key][0].copy(), self._rows[key][1].copy()) for key in selected),
        )

    def select_partitioned_rotating(self, max_updates: int) -> SharedReplaySelection:
        """Continue alternating turns and reservoir rank traversal across calls.

        On a stable nonempty reservoir of R members, each member is proposed
        within at most 2R successful nonempty calls with positive budgets.
        This does not promise coverage for changing pools or recent-only rows.
        Trusted in-memory deepcopy/pickle retains continuation, not authority.
        """
        if type(max_updates) is not int or max_updates <= 0:
            raise ValueError("shared replay update budget must be positive")
        if self._partition_config() is None:
            raise ValueError("rotating selection requires a configured replay allocation")
        cursor = self.rotation_cursor
        sources = (
            iter(reversed(self._recent)),
            iter(_rotating_reservoir(self._reservoir, self.policy, cursor.reservoir_after)),
        )
        selected: list[str] = []
        seen: set[str] = set()
        turn, after = cursor.next_partition, cursor.reservoir_after
        while len(selected) < min(max_updates, len(self._rows)):
            sample_id, after = _scan_rotating(sources[turn], seen, after, reservoir=turn == 1)
            if sample_id is None:
                sample_id, after = _scan_rotating(
                    sources[1 - turn], seen, after, reservoir=turn == 0
                )
            if sample_id is None:
                break
            selected.append(sample_id)
            seen.add(sample_id)
            turn = 1 - turn
        selection = SharedReplaySelection(
            tuple(selected),
            tuple((self._rows[key][0].copy(), self._rows[key][1].copy()) for key in selected),
        )
        # Copy/construction failures must not spend a continuation position.
        if selected:
            self._rotation_cursor = ReplayRotationCursor(turn, after)
        return selection


def _rotating_reservoir(
    membership: dict[str, None], policy: ReplayRetentionPolicy, after: str | None
) -> tuple[str, ...]:
    ranked = sorted((policy._reservoir_rank(key), key) for key in membership)
    if after is None:
        return tuple(key for _, key in ranked)
    previous = (policy._reservoir_rank(after), after)
    start = next((index for index, item in enumerate(ranked) if item > previous), len(ranked))
    return tuple(key for _, key in (*ranked[start:], *ranked[:start]))


def _scan_rotating(
    source: Iterator[str], seen: set[str], after: str | None, *, reservoir: bool
) -> tuple[str | None, str | None]:
    for sample_id in source:
        if reservoir:
            after = sample_id
        if sample_id not in seen:
            return sample_id, after
    return None, after


def _validate_allocation(
    input_dim: int,
    budget: ReplayRetentionBudget,
    policy: ReplayRetentionPolicy,
    allocation: ReplayAllocation,
) -> ReplayAllocation:
    if type(input_dim) is not int or input_dim <= 0:
        raise ValueError("partitioned replay input_dim must be positive")
    if type(allocation) is not ReplayAllocation:
        raise ValueError("allocation must be an exact ReplayAllocation")
    if type(budget) is not ReplayRetentionBudget or type(policy) is not ReplayRetentionPolicy:
        raise ValueError("partitioned replay requires exact global budget and policy records")
    budget = replace(budget)
    if type(policy.name) is not str or policy.name != "seeded_reservoir":
        raise ValueError("partitioned replay requires the seeded_reservoir policy")
    replace(policy)
    allocation = ReplayAllocation(allocation.recent, allocation.reservoir)
    if any(
        part.max_bytes < 8 * (input_dim + 1) for part in (allocation.recent, allocation.reservoir)
    ):
        raise ValueError("each replay partition must hold one float64 labeled row")
    if (
        allocation.recent.max_examples + allocation.reservoir.max_examples > budget.max_examples
        or allocation.recent.max_bytes + allocation.reservoir.max_bytes > budget.max_bytes
    ):
        raise ValueError("replay partition budgets must sum to at most the global caps")
    return allocation


def _partition_snapshot(membership: dict[str, None], row_bytes: int) -> ReplayRetentionSnapshot:
    return ReplayRetentionSnapshot(
        tuple(sorted(membership)), len(membership), len(membership) * row_bytes
    )


def _trim_partition(
    membership: dict[str, None],
    budget: ReplayRetentionBudget,
    row_bytes: int,
    policy: ReplayRetentionPolicy | None,
) -> None:
    while len(membership) > budget.max_examples or len(membership) * row_bytes > budget.max_bytes:
        evicted = next(iter(membership)) if policy is None else policy.eviction_id(membership)
        del membership[evicted]


def _next_distinct(source: Iterator[str], seen: set[str]) -> str | None:
    return next((sample_id for sample_id in source if sample_id not in seen), None)


def _retained_bytes(rows: dict[str, tuple[Array, Array]]) -> int:
    return sum(inputs.nbytes + targets.nbytes for inputs, targets in rows.values())
