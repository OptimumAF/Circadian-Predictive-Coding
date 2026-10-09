"""Payload-free weak integrity witnesses for observed untrained tombstones.

The caller first proves the original arrival and pays its original admission.
These primitives retain no receipt, source, label, declaration or numeric payload
and perform no enrollment, cleanup, work accounting or authority checks.
"""

from dataclasses import dataclass
from hashlib import sha256
from typing import Any
from weakref import ReferenceType, ref

from src.app.erased_inbox_origins import _limits, _schema
from src.core.checkpoint_content import CheckpointContentLimits, checkpoint_content_stamp
from src.core.data_erasure import ErasedExperience
from src.core.untrained_inbox_origin import (
    UntrainedInboxOriginData,
    untrained_inbox_origin_metadata,
)


@dataclass(frozen=True, slots=True)
class UntrainedInboxOrigin:
    data: UntrainedInboxOriginData
    tombstone: ReferenceType[Any]
    tombstone_stamp: str


def _metadata(
    data: UntrainedInboxOriginData,
    tombstone: ErasedExperience,
    limits: CheckpointContentLimits,
) -> str:
    _limits(limits)
    encoded = untrained_inbox_origin_metadata(data, limits.max_metadata_bytes)
    _schema(tombstone, ErasedExperience)
    if type(tombstone.key) is not tuple or len(tombstone.key) != 2:
        raise ValueError("untrained inbox tombstone requires an exact paired key")
    strings: tuple[str, ...] = (*tombstone.key, tombstone.actor_version, tombstone.reason)
    if tombstone.event_id is not None:
        strings += (tombstone.event_id,)
    if any(type(value) is not str for value in strings):
        raise ValueError("untrained inbox tombstone identities require exact strings")
    if sum(len(value) * 6 for value in strings) > limits.max_metadata_bytes:
        raise ValueError("untrained inbox tombstone exceeds metadata capacity")
    if type(tombstone.erased_at) is not int or not 0 <= tombstone.erased_at < 2**63:
        raise ValueError("untrained inbox tombstone requires a bounded erasure time")
    for optional_counter in (tombstone.observed_at, tombstone.arrived_at):
        if optional_counter is not None:
            if type(optional_counter) is not int or not 0 <= optional_counter < 2**63:
                raise ValueError("untrained inbox tombstone times require bounded exact integers")
    # All compared values have exact primitive types before equality is used.
    if (
        tombstone.key != data.key
        or tombstone.actor_version != data.actor_version
        or tombstone.observed_at != data.observed_at
        or tombstone.event_id != data.event_id
        or tombstone.arrived_at != data.arrived_at
        or tombstone.erased_at != data.erased_at
        or tombstone.reason != data.reason
    ):
        raise ValueError("untrained inbox tombstone differs from original scalar metadata")
    scalar_data = (
        data.key,
        data.actor_version,
        data.learner_version,
        data.subject_id,
        data.source_id,
        data.observed_at,
        data.event_id,
        data.arrived_at,
        data.erased_at,
        data.reason,
        data.completed_updates,
    )
    # Why: bound the complete metadata pair before hashing its individual seal.
    checkpoint_content_stamp((scalar_data, tombstone), limits)
    return sha256(encoded).hexdigest()


def _seal(
    data: UntrainedInboxOriginData,
    metadata: str,
    tombstone: ErasedExperience,
    tombstone_ref: ReferenceType[Any],
    limits: CheckpointContentLimits,
) -> str:
    return checkpoint_content_stamp(
        (id(data), metadata, id(tombstone_ref), id(tombstone), tombstone), limits
    )


def prepare_untrained_inbox_origin(
    data: UntrainedInboxOriginData,
    tombstone: ErasedExperience,
    limits: CheckpointContentLimits,
) -> UntrainedInboxOrigin:
    """Prepare integrity after the caller has proved and paid original authority."""
    metadata = _metadata(data, tombstone, limits)
    tombstone_ref = ref(tombstone)
    seal = _seal(data, metadata, tombstone, tombstone_ref, limits)
    return UntrainedInboxOrigin(data, tombstone_ref, seal)


def untrained_inbox_origin_stamp(
    record: UntrainedInboxOrigin, limits: CheckpointContentLimits
) -> tuple:
    """Reprove the same original scalar data, weak identity and tombstone content."""
    _limits(limits)
    if type(record) is not UntrainedInboxOrigin:
        raise ValueError("untrained inbox origin requires its exact slotted witness type")
    if type(record.tombstone) is not ReferenceType:
        raise ValueError("untrained inbox origin requires an exact weak reference")
    seal = record.tombstone_stamp
    if (
        type(seal) is not str
        or len(seal) != 64
        or any(character not in "0123456789abcdef" for character in seal)
    ):
        raise ValueError("untrained inbox origin requires an exact bounded SHA256 seal")
    tombstone = record.tombstone()
    if tombstone is None:
        raise ValueError("untrained inbox origin original tombstone reference expired")
    metadata = _metadata(record.data, tombstone, limits)
    current = _seal(record.data, metadata, tombstone, record.tombstone, limits)
    if current != seal:
        raise ValueError("untrained inbox origin original identity/content seal changed")
    return (id(record), id(record.data), id(record.tombstone), id(tombstone), metadata, current)
