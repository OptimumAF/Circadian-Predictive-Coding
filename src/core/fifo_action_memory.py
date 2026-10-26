"""Bounded exact action counts with deterministic insertion-order retirement.

Inputs are the existing public feature/label arrays or complete owned snapshots.
Outputs are count-frequency probabilities and atomic FIFO states. Hits never
refresh order; evictions retire all exposure counts into two bounded aggregates.
The retained-table allowance excludes the explicit 16-byte retirement metadata
and Python/RSS overhead. No labels, context, IO, history chain, timing or serving
authority is inferred or released by this component.
"""

from dataclasses import dataclass, replace

import numpy as np

from src.core.action_memory import (
    ActionMemoryEntry,
    ActionMemoryPolicy,
    ActionMemorySnapshot,
    _feature_keys,
    _record,
    copy_action_memory_snapshot,
)
from src.core.categorical_probability_scores import validate_class_targets
from src.core.coupled_energy import Array

FIFO_RETIREMENT_METADATA_BYTES = 16
FIFO_ACTION_MEMORY_COUNT_DIAGNOSTIC = "fifo_observation_action_count_increment_rows_v1"
_MAX_COUNT = 2**53 - 1
_MAX_INT64 = 2**63 - 1


@dataclass(frozen=True)
class FifoActionMemoryPolicy:
    method: str = "observation_memory_fifo"
    feature_dim: int = 16
    num_classes: int = 5
    max_entries: int = 64
    max_payload_bytes: int = 10752

    def __post_init__(self) -> None:
        if type(self.method) is not str or self.method != "observation_memory_fifo":
            raise ValueError(
                "FIFO action memory requires its distinct observation_memory_fifo identity"
            )
        # Reuse the refusal control's exact geometry/cap law without changing
        # its policy identity or silently expanding the MiniGrid byte allowance.
        ActionMemoryPolicy(
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
            max_entries=self.max_entries,
            max_payload_bytes=self.max_payload_bytes,
        )

    @property
    def entry_payload_bytes(self) -> int:
        return (self.feature_dim + self.num_classes) * 8


@dataclass(frozen=True)
class FifoActionMemorySnapshot:
    policy: FifoActionMemoryPolicy
    entries: tuple[ActionMemoryEntry, ...]
    completed_updates: int
    training_examples: int
    evictions: int
    retired_examples: int


def copy_fifo_action_memory_policy(policy: FifoActionMemoryPolicy) -> FifoActionMemoryPolicy:
    _record(policy, FifoActionMemoryPolicy)
    return replace(policy)


def _capacity(policy: FifoActionMemoryPolicy) -> int:
    return min(policy.max_entries, policy.max_payload_bytes // policy.entry_payload_bytes)


def _owned_entries(
    entries: tuple[ActionMemoryEntry, ...], policy: FifoActionMemoryPolicy
) -> tuple[ActionMemoryEntry, ...]:
    if type(entries) is not tuple or len(entries) > _capacity(policy):
        raise ValueError("FIFO entries must be an exact tuple within retained capacity")
    owned = []
    seen = set()
    for entry in entries:
        _record(entry, ActionMemoryEntry)
        # Why: reject coercible/tampered containers before copying; the new
        # entry then validates every scalar once, without a redundant replace.
        if type(entry.features) is not tuple or type(entry.counts) is not tuple:
            raise ValueError("FIFO entries require exact feature and count tuples")
        copied = ActionMemoryEntry(
            tuple(value for value in entry.features), tuple(value for value in entry.counts)
        )
        if len(copied.features) != policy.feature_dim or len(copied.counts) != policy.num_classes:
            raise ValueError("FIFO entry geometry differs from its bound policy")
        if copied.features in seen:
            raise ValueError("FIFO entries require unique exact observation keys")
        seen.add(copied.features)
        owned.append(copied)
    return tuple(owned)


def copy_fifo_action_memory_snapshot(
    snapshot: FifoActionMemorySnapshot,
) -> FifoActionMemorySnapshot:
    """Validate queue, lifetime exposure and retirement invariants before copying."""
    _record(snapshot, FifoActionMemorySnapshot)
    policy = copy_fifo_action_memory_policy(snapshot.policy)
    entries = _owned_entries(snapshot.entries, policy)
    updates, examples = snapshot.completed_updates, snapshot.training_examples
    evictions, retired = snapshot.evictions, snapshot.retired_examples
    if any(
        type(value) is not int or not 0 <= value <= _MAX_COUNT
        for value in (updates, examples, evictions, retired)
    ):
        raise ValueError("FIFO counters must be bounded nonnegative native integers")
    retained = sum(sum(entry.counts) for entry in entries)
    if (
        retained + retired != examples
        or not updates <= examples <= 128 * updates
        or not evictions <= retired <= examples
        or (evictions == 0) != (retired == 0)
        or (examples == 0) != (len(entries) == 0)
    ):
        raise ValueError("FIFO retained/retired exposure and lifetime counters disagree")
    if evictions and len(entries) != _capacity(policy):
        raise ValueError("an evicted FIFO table must remain full at its effective capacity")
    return FifoActionMemorySnapshot(policy, entries, updates, examples, evictions, retired)


def predict_fifo_action_memory(snapshot: FifoActionMemorySnapshot, features: Array) -> Array:
    """Use unchanged empirical counts or uniform probabilities for unseen keys."""
    owned = copy_fifo_action_memory_snapshot(snapshot)
    keys = _feature_keys(features, owned.policy.feature_dim)
    counts = {entry.features: entry.counts for entry in owned.entries}
    return np.array(
        [
            tuple(value / sum(counts[key]) for value in counts[key])
            if key in counts
            else (1.0 / owned.policy.num_classes,) * owned.policy.num_classes
            for key in keys
        ],
        dtype=np.float64,
    )


def count_fifo_action_observations(
    snapshot: FifoActionMemorySnapshot, features: Array, targets: Array
) -> FifoActionMemorySnapshot:
    """Construct the complete ordered update locally, publishing no partial batch."""
    owned = copy_fifo_action_memory_snapshot(snapshot)
    keys = _feature_keys(features, owned.policy.feature_dim)
    validate_class_targets(targets, len(keys), owned.policy.num_classes)
    labels = tuple(int(value) for value in targets[:, 0])
    capacity = _capacity(owned.policy)
    if capacity == 0:
        raise ValueError("FIFO retained capacity is zero; batch cannot be admitted")
    updates, examples = owned.completed_updates + 1, owned.training_examples + len(keys)
    if updates > _MAX_COUNT or examples > _MAX_COUNT:
        raise ValueError("FIFO native counter overflow")
    counts = {entry.features: list(entry.counts) for entry in owned.entries}
    evictions, retired = owned.evictions, owned.retired_examples
    for key, label in zip(keys, labels, strict=True):
        if key not in counts:
            if len(counts) == capacity:
                oldest = next(iter(counts))
                retired += sum(counts.pop(oldest))
                evictions += 1
                if retired > _MAX_COUNT or evictions > _MAX_COUNT:
                    raise ValueError("FIFO retirement counter overflow")
            counts[key] = [0] * owned.policy.num_classes
        if counts[key][label] == _MAX_INT64:
            raise ValueError("FIFO action int64 count overflow")
        # Updating a value in an insertion-ordered dict does not refresh its key.
        counts[key][label] += 1
    entries = tuple(ActionMemoryEntry(key, tuple(values)) for key, values in counts.items())
    result = FifoActionMemorySnapshot(
        copy_fifo_action_memory_policy(owned.policy), entries, updates, examples, evictions, retired
    )
    return copy_fifo_action_memory_snapshot(result)


def convert_action_memory_to_fifo(snapshot: ActionMemorySnapshot) -> FifoActionMemorySnapshot:
    """Explicitly convert owned old counts; no app budget or serving state resumes."""
    owned = copy_action_memory_snapshot(snapshot)
    policy = FifoActionMemoryPolicy(
        feature_dim=owned.policy.feature_dim,
        num_classes=owned.policy.num_classes,
        max_entries=owned.policy.max_entries,
        max_payload_bytes=owned.policy.max_payload_bytes,
    )
    return copy_fifo_action_memory_snapshot(
        FifoActionMemorySnapshot(
            policy, owned.entries, owned.completed_updates, owned.training_examples, 0, 0
        )
    )
