"""Deterministic four-role source split for a future continual decision protocol.

Inputs are one phase's generated training fields and a declared final-test
count. Outputs are disjoint train, inner-guard, and outer-selection arrays
with stable row IDs and hashes. Final-test values are read only by the
explicit release function. This module does not train or choose settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from numbers import Real
from types import MappingProxyType
from typing import Mapping, Protocol

import numpy as np

from src.infra.datasets import LabeledData


class PhaseSource(Protocol):
    """Source fields used at development arrival and final release."""

    @property
    def train_input(self) -> np.ndarray: ...

    @property
    def train_target(self) -> np.ndarray: ...

    @property
    def test_input(self) -> np.ndarray: ...

    @property
    def test_target(self) -> np.ndarray: ...


@dataclass(frozen=True)
class RoleAvailability:
    """Declared source and label release events; runtime proof belongs to the app."""

    source_at: str
    labels_at: str


@dataclass(frozen=True)
class PhaseDecisionRoles:
    """One phase's disjoint decision roles and unopened final-test identity."""

    phase: str
    seed: int
    train: LabeledData
    inner_guard: LabeledData
    outer_selection: LabeledData
    sample_ids: Mapping[str, tuple[str, ...]]
    split_hashes: Mapping[str, str]
    release_policy: Mapping[str, RoleAvailability]
    expected_final_count: int
    final_test: LabeledData | None = None
    final_released: bool = False
    _source: PhaseSource | None = field(default=None, repr=False, compare=False)


def split_phase_decision_roles(
    source: PhaseSource,
    *,
    phase: str,
    seed: int,
    split_seed: int,
    inner_guard_fraction: float,
    outer_selection_fraction: float,
    expected_final_count: int,
    source_row_indices: tuple[int, ...] | None = None,
) -> PhaseDecisionRoles:
    """Split arrived training fields; predeclare final IDs without reading tests."""
    inputs, targets = _validated_training_fields(
        source,
        phase=phase,
        seed=seed,
        split_seed=split_seed,
        inner_guard_fraction=inner_guard_fraction,
        outer_selection_fraction=outer_selection_fraction,
        expected_final_count=expected_final_count,
    )
    sorted_rows = _partition_training_rows(
        targets, split_seed, inner_guard_fraction, outer_selection_fraction
    )
    source_rows = _source_row_indices(source_row_indices, len(inputs))
    samples = {
        role: LabeledData(inputs[indices], targets[indices])
        for role, indices in sorted_rows.items()
    }
    ids: dict[str, tuple[str, ...]] = {
        role: tuple(
            f"phase_{phase}/seed_{seed}/development/{source_rows[index]}" for index in indices
        )
        for role, indices in sorted_rows.items()
    }
    ids["final_test"] = tuple(
        f"phase_{phase}/seed_{seed}/final/{index}" for index in range(expected_final_count)
    )
    hashes = {role: _hash_role(phase, seed, role, ids[role], samples[role]) for role in samples}
    arrival = RoleAvailability(f"phase_{phase}_arrival", f"phase_{phase}_arrival")
    release_policy = {role: arrival for role in samples}
    release_policy["final_test"] = RoleAvailability("global_freeze", "global_freeze")
    return PhaseDecisionRoles(
        phase=phase,
        seed=seed,
        train=samples["train"],
        inner_guard=samples["inner_guard"],
        outer_selection=samples["outer_selection"],
        sample_ids=MappingProxyType(ids),
        split_hashes=MappingProxyType(hashes),
        release_policy=MappingProxyType(release_policy),
        expected_final_count=expected_final_count,
        _source=source,
    )


def _validated_training_fields(
    source: PhaseSource,
    *,
    phase: str,
    seed: int,
    split_seed: int,
    inner_guard_fraction: float,
    outer_selection_fraction: float,
    expected_final_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reject invalid identities, budgets, or arrived training arrays."""
    if phase not in {"a", "b"} or any(type(value) is not int for value in (seed, split_seed)):
        raise ValueError("phase must be a/b and seeds must be Python integers")
    if (
        any(
            not isinstance(value, Real) or isinstance(value, bool)
            for value in (inner_guard_fraction, outer_selection_fraction)
        )
        or not np.isfinite(inner_guard_fraction)
        or not np.isfinite(outer_selection_fraction)
        or not 0.0 < inner_guard_fraction < 1.0
        or not 0.0 < outer_selection_fraction < 1.0
        or inner_guard_fraction + outer_selection_fraction >= 1.0
    ):
        raise ValueError("inner/outer fractions must be positive and sum below one")
    if type(expected_final_count) is not int or expected_final_count <= 0:
        raise ValueError("expected_final_count must be positive")
    inputs, targets = source.train_input, source.train_target
    if (
        not isinstance(inputs, np.ndarray)
        or not isinstance(targets, np.ndarray)
        or inputs.dtype != np.float64
        or targets.dtype != np.float64
        or inputs.ndim != 2
        or inputs.shape[1] != 2
        or targets.shape != (inputs.shape[0], 1)
        or not np.all(np.isfinite(inputs))
        or not np.all(np.isin(targets, (0.0, 1.0)))
    ):
        raise ValueError("phase training fields must be finite binary two-feature rows")
    return inputs, targets


def _source_row_indices(requested: tuple[int, ...] | None, count: int) -> tuple[int, ...]:
    """Preserve original source positions after a phase-local training cap."""
    if requested is None:
        return tuple(range(count))
    if (
        not isinstance(requested, tuple)
        or len(requested) != count
        or any(type(index) is not int or index < 0 for index in requested)
        or len(set(requested)) != count
    ):
        raise ValueError("source_row_indices must be unique source positions")
    return requested


def _partition_training_rows(
    targets: np.ndarray,
    split_seed: int,
    inner_guard_fraction: float,
    outer_selection_fraction: float,
) -> dict[str, np.ndarray]:
    """Reserve both decision roles within each class and keep source order."""
    rng = np.random.default_rng(split_seed)
    rows: dict[str, list[int]] = {"train": [], "inner_guard": [], "outer_selection": []}
    for class_label in (0.0, 1.0):
        class_rows = rng.permutation(np.flatnonzero(targets[:, 0] == class_label))
        inner_count = max(1, round(len(class_rows) * inner_guard_fraction))
        outer_count = max(1, round(len(class_rows) * outer_selection_fraction))
        if len(class_rows) - inner_count - outer_count < 1:
            raise ValueError("each class needs train, inner-guard, and outer-selection rows")
        rows["inner_guard"].extend(class_rows[:inner_count].tolist())
        rows["outer_selection"].extend(class_rows[inner_count : inner_count + outer_count].tolist())
        rows["train"].extend(class_rows[inner_count + outer_count :].tolist())

    return {role: np.asarray(sorted(indices), dtype=np.int64) for role, indices in rows.items()}


def release_final_test(roles: PhaseDecisionRoles) -> PhaseDecisionRoles:
    """Read and bind final-test content after the app's global freeze gate."""
    if roles.final_released or roles._source is None:
        raise ValueError("final test already released")
    source = roles._source
    final = LabeledData(source.test_input, source.test_target)
    if (
        not isinstance(final.input, np.ndarray)
        or not isinstance(final.target, np.ndarray)
        or final.input.dtype != np.float64
        or final.target.dtype != np.float64
        or final.input.shape != (roles.expected_final_count, 2)
        or final.target.shape != (roles.expected_final_count, 1)
        or not np.all(np.isfinite(final.input))
        or not np.all(np.isin(final.target, (0.0, 1.0)))
    ):
        raise ValueError("incompatible final-test count or binary fields")
    hashes = dict(roles.split_hashes)
    hashes["final_test"] = _hash_role(
        roles.phase, roles.seed, "final_test", roles.sample_ids["final_test"], final
    )
    return replace(
        roles,
        final_test=final,
        final_released=True,
        split_hashes=MappingProxyType(hashes),
        _source=None,
    )


def _hash_role(
    phase: str, seed: int, role: str, sample_ids: tuple[str, ...], samples: LabeledData
) -> str:
    """Bind both source row identity and values, preserving role order."""
    digest = sha256(f"continual_decision_role_v1/{phase}/{seed}/{role}".encode("ascii"))
    for sample_id in sample_ids:
        digest.update(sample_id.encode("ascii"))
        digest.update(b"\0")
    for values in (samples.input, samples.target):
        canonical = np.ascontiguousarray(values, dtype="<f8")
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()
