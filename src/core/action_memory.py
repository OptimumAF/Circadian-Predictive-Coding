"""Bounded exact public-observation action counts, without a neural model.

Inputs are float64 public 16/296-coordinate rows with five/seven-class labels.
Outputs are owned immutable snapshots or count-frequency probabilities; unseen
keys are uniform. No context IDs, temporal inference, eviction, masking, label
release, IO, or gradient objective belongs here. Updates are whole-batch atomic.
"""

from dataclasses import dataclass, fields, replace
from math import isfinite
from typing import Any

import numpy as np

from src.core.categorical_probability_scores import validate_class_targets
from src.core.coupled_energy import Array

ACTION_MEMORY_ENTRY_BYTES = 16 * 8 + 5 * 8
ACTION_MEMORY_COUNT_DIAGNOSTIC = "observation_action_count_increment_rows_v1"
_MAX_COUNT = 2**53 - 1
_MAX_INT64 = 2**63 - 1
_GEOMETRIES = ((16, 5), (296, 7))


def _record(value: Any, expected: type) -> None:
    if type(value) is not expected or set(vars(value)) != {
        field.name for field in fields(expected)
    }:
        raise ValueError(f"an exact complete {expected.__name__} is required")


@dataclass(frozen=True)
class ActionMemoryPolicy:
    method: str = "observation_memory"
    feature_dim: int = 16
    num_classes: int = 5
    max_entries: int = 64
    max_payload_bytes: int = 10752

    def __post_init__(self) -> None:
        if (
            type(self.method) is not str
            or self.method != "observation_memory"
            or type(self.feature_dim) is not int
            or type(self.num_classes) is not int
            or (self.feature_dim, self.num_classes) not in _GEOMETRIES
        ):
            raise ValueError(
                "action memory requires observation_memory with 16/5 or 296/7 features/classes"
            )
        for name, maximum in (
            ("max_entries", 64),
            ("max_payload_bytes", 64 * self.entry_payload_bytes),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be a native integer in 1..{maximum}")

    @property
    def entry_payload_bytes(self) -> int:
        """Logical float64 keys plus int64 counts, excluding object overhead."""
        # Why: larger MiniGrid keys cost more, but opting into their geometry
        # must not silently enlarge the declared default payload allowance.
        return (self.feature_dim + self.num_classes) * 8


@dataclass(frozen=True)
class ActionMemoryEntry:
    features: tuple[float, ...]
    counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.features) is not tuple
            or len(self.features) not in (16, 296)
            or any(
                type(value) is not float or not isfinite(value) or not -1 <= value <= 1
                for value in self.features
            )
        ):
            raise ValueError("memory keys require sixteen or 296 native finite float coordinates")
        if (
            type(self.counts) is not tuple
            or (len(self.features), len(self.counts)) not in _GEOMETRIES
            or any(type(value) is not int or not 0 <= value <= _MAX_INT64 for value in self.counts)
            or sum(self.counts) == 0
        ):
            raise ValueError(
                "memory counts require matching five/seven nonnegative int64 values with positive total"
            )


@dataclass(frozen=True)
class ActionMemorySnapshot:
    policy: ActionMemoryPolicy
    entries: tuple[ActionMemoryEntry, ...]
    completed_updates: int
    training_examples: int


def copy_action_memory_policy(policy: ActionMemoryPolicy) -> ActionMemoryPolicy:
    _record(policy, ActionMemoryPolicy)
    return replace(policy)


def copy_action_memory_snapshot(snapshot: ActionMemorySnapshot) -> ActionMemorySnapshot:
    """Validate even frozen-record tampering and detach every stored record."""
    _record(snapshot, ActionMemorySnapshot)
    policy = copy_action_memory_policy(snapshot.policy)
    if (
        type(snapshot.entries) is not tuple
        or len(snapshot.entries) > policy.max_entries
        or len(snapshot.entries) * policy.entry_payload_bytes > policy.max_payload_bytes
    ):
        raise ValueError("memory entries exceed their exact tuple or declared payload capacity")
    owned = []
    seen = set()
    for entry in snapshot.entries:
        _record(entry, ActionMemoryEntry)
        validated = replace(entry)
        if (
            len(validated.features) != policy.feature_dim
            or len(validated.counts) != policy.num_classes
        ):
            raise ValueError("memory entry geometry differs from its bound policy")
        copied = ActionMemoryEntry(
            tuple(value for value in validated.features), tuple(value for value in validated.counts)
        )
        if copied.features in seen:
            raise ValueError("memory entries must have unique exact observation keys")
        seen.add(copied.features)
        owned.append(copied)
    updates, examples = snapshot.completed_updates, snapshot.training_examples
    if any(type(value) is not int or not 0 <= value <= _MAX_COUNT for value in (updates, examples)):
        raise ValueError("memory counters must be bounded nonnegative native integers")
    # Counts have an int64 logical representation; the aggregate exposure clock
    # is more tightly bounded by the repository's exact accounting integer law.
    if (
        sum(sum(entry.counts) for entry in owned) != examples
        or not updates <= examples <= 128 * updates
        or (examples == 0) != (len(owned) == 0)
    ):
        raise ValueError("memory counts, examples, and complete batch update counters disagree")
    return ActionMemorySnapshot(policy, tuple(owned), updates, examples)


def _feature_keys(features: Array, feature_dim: int) -> tuple[tuple[float, ...], ...]:
    if (
        type(features) is not np.ndarray
        or features.dtype != np.float64
        or features.ndim != 2
        or features.shape[1] != feature_dim
        or not 1 <= len(features) <= 128
        or not np.isfinite(features).all()
        or not ((features >= -1) & (features <= 1)).all()
    ):
        raise ValueError(
            f"memory features must be exact finite float64 Bx{feature_dim} in [-1, 1], B in 1..128"
        )
    return tuple(tuple(float(value) for value in row) for row in features)


def predict_action_memory(snapshot: ActionMemorySnapshot, features: Array) -> Array:
    """Use exact numeric keys without quantization or nearest-neighbor repair."""
    owned = copy_action_memory_snapshot(snapshot)
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


def count_action_observations(
    snapshot: ActionMemorySnapshot, features: Array, targets: Array
) -> ActionMemorySnapshot:
    """Preflight all exposures and capacities, then return one complete new state."""
    owned = copy_action_memory_snapshot(snapshot)
    keys = _feature_keys(features, owned.policy.feature_dim)
    validate_class_targets(targets, len(keys), owned.policy.num_classes)
    labels = tuple(int(value) for value in targets[:, 0])
    updates, examples = owned.completed_updates + 1, owned.training_examples + len(keys)
    if updates > _MAX_COUNT or examples > _MAX_COUNT:
        raise ValueError("memory native counter overflow")
    counts = {entry.features: list(entry.counts) for entry in owned.entries}
    for key, label in zip(keys, labels, strict=True):
        current = counts.setdefault(key, [0] * owned.policy.num_classes)
        if current[label] == _MAX_INT64:
            raise ValueError("memory action int64 count overflow")
        current[label] += 1
    if (
        len(counts) > owned.policy.max_entries
        or len(counts) * owned.policy.entry_payload_bytes > owned.policy.max_payload_bytes
    ):
        raise ValueError("memory batch exceeds entry or logical payload capacity")
    entries = tuple(ActionMemoryEntry(key, tuple(values)) for key, values in counts.items())
    return ActionMemorySnapshot(copy_action_memory_policy(owned.policy), entries, updates, examples)
