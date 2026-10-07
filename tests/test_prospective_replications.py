"""Complete fabricated metadata declarations; no real seed/source authority."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src.core.prospective_replications import (
    ReplicaSeedBinding,
    ReplicaSlot,
    bind_replica_streams,
)


@pytest.fixture
def slots() -> tuple[ReplicaSlot, ...]:
    families = ("gating", "replay", "sleep", "schedule", "combined", "parent")
    return tuple(
        ReplicaSlot(
            family, index, f"{'gating_replay' if family in families[:2] else family}/{index}"
        )
        for family in families
        for index in range(10)
    )


@pytest.fixture
def bindings(slots: tuple[ReplicaSlot, ...]) -> tuple[ReplicaSeedBinding, ...]:
    groups = tuple(dict.fromkeys(row.source_group for row in slots))
    # Synthetic values test declarations only, without choosing any real seed.
    return tuple(
        ReplicaSeedBinding(row, 1_000_000 + 20_000 * groups.index(row.source_group))
        for row in slots
    )


def test_should_retain_all_sixty_views_and_fifty_declared_groups_without_independence_proof(
    slots: tuple[ReplicaSlot, ...],
    bindings: tuple[ReplicaSeedBinding, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("metadata binding performed IO")

    monkeypatch.setattr(Path, "open", forbidden)
    result = bind_replica_streams(slots, bindings)
    assert result.bindings == bindings and len(result.bindings) == 60
    assert len(result.source_groups) == 50 and len(result.streams) == 400
    assert result.source_groups[0] == ("gating_replay/0", 1_000_000)
    assert result.streams[7].value == 1_011_002
    assert result.independent_source_replications is None
    assert result.fresh_roles_authorized is False


@pytest.mark.parametrize("kind", ["omitted", "extra", "reordered", "late_family", "late_ordinal"])
def test_should_reject_any_partial_or_detached_ordered_binding(
    slots: tuple[ReplicaSlot, ...], bindings: tuple[ReplicaSeedBinding, ...], kind: str
) -> None:
    if kind == "omitted":
        bindings = bindings[:-1]
    elif kind == "extra":
        bindings += (bindings[-1],)
    elif kind == "reordered":
        bindings = tuple(reversed(bindings))
    else:
        changes: dict[str, Any] = {
            "family" if kind == "late_family" else "ordinal": "foreign"
            if kind == "late_family"
            else 10
        }
        slot = replace(bindings[-1].slot, **changes)
        bindings = bindings[:-1] + (replace(bindings[-1], slot=slot),)
    with pytest.raises(ValueError):
        bind_replica_streams(slots, bindings)


@pytest.mark.parametrize("value", [True, False, -1, 1.0, "1", None])
def test_should_reject_inexact_or_invalid_seed_types(
    slots: tuple[ReplicaSlot, ...], bindings: tuple[ReplicaSeedBinding, ...], value: Any
) -> None:
    with pytest.raises(ValueError):
        bind_replica_streams(slots, bindings[:-1] + (replace(bindings[-1], base_seed=value),))


def test_should_reject_a_shared_group_declaring_different_source_seeds(
    slots: tuple[ReplicaSlot, ...], bindings: tuple[ReplicaSeedBinding, ...]
) -> None:
    changed = bindings[:10] + (replace(bindings[10], base_seed=7_000_000),) + bindings[11:]
    with pytest.raises(ValueError, match="shared"):
        bind_replica_streams(slots, changed)


@pytest.mark.parametrize("ordinal", [True, 1.0])
def test_should_reject_a_binding_whose_inexact_ordinal_compares_equal_to_a_slot(
    slots: tuple[ReplicaSlot, ...], bindings: tuple[ReplicaSeedBinding, ...], ordinal: Any
) -> None:
    changed = (
        bindings[:1]
        + (replace(bindings[1], slot=replace(slots[1], ordinal=ordinal)),)
        + bindings[2:]
    )
    with pytest.raises(ValueError):
        bind_replica_streams(slots, changed)


def test_should_reject_a_family_reusing_one_group_for_two_replications(
    slots: tuple[ReplicaSlot, ...], bindings: tuple[ReplicaSeedBinding, ...]
) -> None:
    late = replace(slots[-1], source_group=slots[-2].source_group)
    layout = slots[:-1] + (late,)
    declared = bindings[:-1] + (ReplicaSeedBinding(late, bindings[-2].base_seed),)
    with pytest.raises(ValueError):
        bind_replica_streams(layout, declared)


@pytest.mark.parametrize("offset", [0, 17, 101, 138, 118, 1001, 5001, 11002])
def test_should_reject_all_base_or_cross_stream_reuse_between_distinct_groups(
    slots: tuple[ReplicaSlot, ...], bindings: tuple[ReplicaSeedBinding, ...], offset: int
) -> None:
    changed = bindings[:-1] + (replace(bindings[-1], base_seed=bindings[0].base_seed + offset),)
    with pytest.raises(ValueError, match="stream"):
        bind_replica_streams(slots, changed)


@pytest.mark.parametrize(
    "kind",
    [
        "mutable_slots",
        "mutable_bindings",
        "empty",
        "bool_ordinal",
        "empty_group",
        "missing_ordinal",
        "duplicate_slot",
    ],
)
def test_should_reject_inexact_or_ambiguous_slot_layouts(
    slots: tuple[ReplicaSlot, ...], bindings: tuple[ReplicaSeedBinding, ...], kind: str
) -> None:
    values: Any = slots
    declared: Any = bindings
    if kind == "mutable_slots":
        values = list(slots)
    elif kind == "mutable_bindings":
        declared = list(bindings)
    elif kind == "empty":
        values, declared = (), ()
    else:
        changes = (
            {"ordinal": True}
            if kind == "bool_ordinal"
            else {"source_group": ""}
            if kind == "empty_group"
            else {"ordinal": 11}
            if kind == "missing_ordinal"
            else {"ordinal": 8}
        )
        altered = replace(slots[-1], **changes)
        values = slots[:-1] + (altered,)
        declared = bindings[:-1] + (replace(bindings[-1], slot=altered),)
    with pytest.raises(ValueError):
        bind_replica_streams(values, declared)
