"""Phase-local decision roles have stable identities and a sealed final source."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from src.infra.continual_roles import (
    PhaseDecisionRoles,
    release_final_test,
    split_phase_decision_roles,
)
from src.infra.datasets import generate_two_cluster_dataset_with_transform


class SealedFinalSource:
    """Expose training data while counting attempts to open held-out fields."""

    def __init__(self, source: Any) -> None:
        self.source = source
        self.train_input = source.train_input
        self.train_target = source.train_target
        self.released = False
        self.final_reads: list[str] = []

    @property
    def test_input(self) -> Any:
        if not self.released:
            raise AssertionError("final input opened before global freeze")
        self.final_reads.append("input")
        return self.source.test_input

    @property
    def test_target(self) -> Any:
        if not self.released:
            raise AssertionError("final label opened before global freeze")
        self.final_reads.append("label")
        return self.source.test_target


def _split(
    source: Any,
    *,
    phase: str = "a",
    split_seed: int = 23,
    inner_fraction: float = 0.2,
    outer_fraction: float = 0.2,
    final_count: int = 10,
) -> PhaseDecisionRoles:
    return split_phase_decision_roles(
        source,
        phase=phase,
        seed=17,
        split_seed=split_seed,
        inner_guard_fraction=inner_fraction,
        outer_selection_fraction=outer_fraction,
        expected_final_count=final_count,
    )


@pytest.mark.parametrize("phase", ["a", "b"])
def test_should_split_disjoint_arrived_roles_without_opening_final_source(phase: str) -> None:
    source = generate_two_cluster_dataset_with_transform(
        sample_count=40,
        noise_scale=0.8,
        seed=17 if phase == "a" else 118,
        test_ratio=0.25,
    )
    sealed = SealedFinalSource(source)
    first = _split(sealed, phase=phase)
    second = _split(sealed, phase=phase)
    changed_seed = _split(sealed, phase=phase, split_seed=29)

    assert not sealed.final_reads
    assert set(first.split_hashes) == {"train", "inner_guard", "outer_selection"}
    assert set(first.sample_ids) == {
        "train",
        "inner_guard",
        "outer_selection",
        "final_test",
    }
    assert dict(first.split_hashes) == dict(second.split_hashes)
    assert dict(first.sample_ids) == dict(second.sample_ids)
    assert first.split_hashes["inner_guard"] != changed_seed.split_hashes["inner_guard"]
    assert first.final_test is None
    assert all(
        first.release_policy[role].source_at == f"phase_{phase}_arrival"
        and first.release_policy[role].labels_at == f"phase_{phase}_arrival"
        for role in ("train", "inner_guard", "outer_selection")
    )
    assert first.release_policy["final_test"].source_at == "global_freeze"
    assert first.release_policy["final_test"].labels_at == "global_freeze"
    assert sum(
        len(first.sample_ids[role]) for role in ("train", "inner_guard", "outer_selection")
    ) == len(source.train_input)
    all_ids = tuple(sample_id for role_ids in first.sample_ids.values() for sample_id in role_ids)
    assert len(all_ids) == len(set(all_ids))
    for role in (first.train, first.inner_guard, first.outer_selection):
        assert set(role.target.reshape(-1)) == {0.0, 1.0}

    sealed.released = True
    bound = release_final_test(first)
    assert sealed.final_reads == ["input", "label"]
    assert set(bound.split_hashes) == {
        "train",
        "inner_guard",
        "outer_selection",
        "final_test",
    }
    assert bound.final_released
    assert bound.final_test is not None
    np.testing.assert_array_equal(bound.final_test.target, source.test_target)
    with pytest.raises(ValueError, match="already released"):
        release_final_test(bound)


def test_should_keep_development_identity_when_only_final_labels_change() -> None:
    source = generate_two_cluster_dataset_with_transform(
        sample_count=60, noise_scale=0.8, seed=17, test_ratio=0.25
    )
    changed = replace(source, test_target=1.0 - source.test_target)
    original_roles = _split(source, final_count=source.test_input.shape[0])
    changed_roles = _split(changed, final_count=source.test_input.shape[0])

    assert dict(original_roles.sample_ids) == dict(changed_roles.sample_ids)
    assert dict(original_roles.split_hashes) == dict(changed_roles.split_hashes)
    assert (
        release_final_test(original_roles).split_hashes["final_test"]
        != release_final_test(changed_roles).split_hashes["final_test"]
    )


def test_should_keep_role_ids_disjoint_across_phase_arrivals() -> None:
    source_a = generate_two_cluster_dataset_with_transform(
        sample_count=40, noise_scale=0.8, seed=17, test_ratio=0.25
    )
    source_b = generate_two_cluster_dataset_with_transform(
        sample_count=40, noise_scale=0.8, seed=118, test_ratio=0.25
    )
    phase_a = _split(SealedFinalSource(source_a), phase="a")
    phase_b = _split(SealedFinalSource(source_b), phase="b")

    a_ids = {sample_id for ids in phase_a.sample_ids.values() for sample_id in ids}
    b_ids = {sample_id for ids in phase_b.sample_ids.values() for sample_id in ids}
    assert a_ids.isdisjoint(b_ids)
    assert phase_a.release_policy["train"].labels_at == "phase_a_arrival"
    assert phase_b.release_policy["train"].labels_at == "phase_b_arrival"


def test_should_reject_invalid_role_budget_and_mismatched_final_count() -> None:
    source = generate_two_cluster_dataset_with_transform(
        sample_count=40, noise_scale=0.8, seed=17, test_ratio=0.25
    )
    with pytest.raises(ValueError, match="fractions"):
        _split(source, outer_fraction=0.9)
    with pytest.raises(ValueError, match="fractions"):
        _split(source, inner_fraction=cast(Any, "0.2"))
    with pytest.raises(ValueError, match="each class"):
        _split(replace(source, train_target=np.zeros_like(source.train_target)))
    roles = _split(source, final_count=11)
    with pytest.raises(ValueError, match="final-test count"):
        release_final_test(roles)


def test_should_preserve_original_source_ids_after_training_source_reduction() -> None:
    source = generate_two_cluster_dataset_with_transform(
        sample_count=40, noise_scale=0.8, seed=17, test_ratio=0.25
    )
    selected = tuple(range(0, len(source.train_input), 2))
    reduced = replace(
        source,
        train_input=source.train_input[list(selected)],
        train_target=source.train_target[list(selected)],
    )
    roles = split_phase_decision_roles(
        reduced,
        phase="b",
        seed=17,
        split_seed=23,
        inner_guard_fraction=0.2,
        outer_selection_fraction=0.2,
        expected_final_count=10,
        source_row_indices=selected,
    )
    reported = {
        int(sample_id.rsplit("/", 1)[1])
        for role in ("train", "inner_guard", "outer_selection")
        for sample_id in roles.sample_ids[role]
    }
    assert reported == set(selected)
    with pytest.raises(ValueError, match="unique source positions"):
        split_phase_decision_roles(
            reduced,
            phase="b",
            seed=17,
            split_seed=23,
            inner_guard_fraction=0.2,
            outer_selection_fraction=0.2,
            expected_final_count=10,
            source_row_indices=(0,) * len(selected),
        )
