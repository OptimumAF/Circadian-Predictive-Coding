"""Bind ordered replica seed declarations without source construction or admission.

Inputs are complete immutable slot and seed-binding tuples. Outputs preserve
all views, planned shared-source groups and every current derived stream. Same
group declarations must agree; distinct groups cannot reuse numeric streams.
Declarations establish no actual freshness, statistical independence, execution,
request identity or prior-usage/resource acceptance. No IO/RNG/model belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.seed_stream_screening import DerivedSeedStream, confirmation_seed_streams


@dataclass(frozen=True)
class ReplicaSlot:
    family: str
    ordinal: int
    source_group: str


@dataclass(frozen=True)
class ReplicaSeedBinding:
    slot: ReplicaSlot
    base_seed: int


@dataclass(frozen=True)
class DeclaredReplicaStreams:
    bindings: tuple[ReplicaSeedBinding, ...]
    source_groups: tuple[tuple[str, int], ...]
    streams: tuple[DerivedSeedStream, ...]
    independent_source_replications: None = None
    fresh_roles_authorized: bool = False


def _validate_slots(slots: tuple[ReplicaSlot, ...]) -> None:
    if type(slots) is not tuple or not slots:
        raise ValueError("replica slots require the complete ordered immutable layout")
    next_ordinal: dict[str, int] = {}
    seen_groups: set[tuple[str, str]] = set()
    shared_ordinals: dict[str, int] = {}
    for slot in slots:
        if (
            type(slot) is not ReplicaSlot
            or type(slot.family) is not str
            or not slot.family
            or type(slot.source_group) is not str
            or not slot.source_group
            or type(slot.ordinal) is not int
            or slot.ordinal != next_ordinal.get(slot.family, 0)
        ):
            raise ValueError("replica slots require exact contiguous family ordinals and groups")
        if (slot.family, slot.source_group) in seen_groups or shared_ordinals.setdefault(
            slot.source_group, slot.ordinal
        ) != slot.ordinal:
            raise ValueError("replica slots cannot reuse a family group or shift shared ordinals")
        seen_groups.add((slot.family, slot.source_group))
        next_ordinal[slot.family] = slot.ordinal + 1


def _source_groups(
    slots: tuple[ReplicaSlot, ...], bindings: tuple[ReplicaSeedBinding, ...]
) -> tuple[tuple[str, int], ...]:
    if type(bindings) is not tuple or len(bindings) != len(slots):
        raise ValueError("replica bindings must retain every ordered slot")
    groups: dict[str, int] = {}
    for slot, binding in zip(slots, bindings, strict=True):
        if (
            type(binding) is not ReplicaSeedBinding
            or type(binding.slot) is not ReplicaSlot
            or type(binding.slot.family) is not str
            or type(binding.slot.source_group) is not str
            or type(binding.slot.ordinal) is not int
            or binding.slot != slot
            or type(binding.base_seed) is not int
            or binding.base_seed < 0
        ):
            raise ValueError("replica binding identity/order or seed type differs")
        prior = groups.setdefault(slot.source_group, binding.base_seed)
        if prior != binding.base_seed:
            raise ValueError("planned shared source group has different seed declarations")
    return tuple(groups.items())


def bind_replica_streams(
    slots: tuple[ReplicaSlot, ...], bindings: tuple[ReplicaSeedBinding, ...]
) -> DeclaredReplicaStreams:
    """Check declarations; neither numeric separation nor sharing proves independence."""
    _validate_slots(slots)
    groups = _source_groups(slots, bindings)
    streams = tuple(stream for _, seed in groups for stream in confirmation_seed_streams(seed))
    # Why this: a base can collide with another replica's source/split/model or
    # local-noise stream even when all declared base integers are different.
    if len({stream.value for stream in streams}) != len(streams):
        raise ValueError("distinct planned source groups reuse a derived seed stream")
    return DeclaredReplicaStreams(bindings, groups, streams)
