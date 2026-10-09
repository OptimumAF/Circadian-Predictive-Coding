"""Pure bounded untrained tombstone collections and caller memo binding.

Inputs are already admitted scalar witnesses and the caller's qualified memo
lookup. Outputs preserve the same scalar data with weak copied tombstone links.
This module owns no payload, receipt, consent, admission, cleanup or IO; the
checkpoint coordinator must separately prove the actual paid copy operation.
"""

from typing import Callable

from src.app.checkpoint_erased_inbox import _bound, _tuple
from src.app.untrained_inbox_origins import (
    UntrainedInboxOrigin,
    prepare_untrained_inbox_origin,
    untrained_inbox_origin_stamp,
)
from src.core.checkpoint_content import CheckpointContentLimits, checkpoint_content_stamp
from src.core.data_erasure import ErasedExperience
from src.core.untrained_inbox_origin import untrained_inbox_origin_metadata


def require_untrained_inbox_records(
    records: tuple[UntrainedInboxOrigin, ...],
    limits: CheckpointContentLimits,
    maximum: int,
) -> None:
    """Require complete bounded scalar witnesses before using keys or weak refs."""
    _bound(limits, maximum)
    _tuple(records, maximum)
    metadata_bytes = 0
    for record in records:
        untrained_inbox_origin_stamp(record, limits)
        metadata_bytes += len(
            untrained_inbox_origin_metadata(record.data, limits.max_metadata_bytes)
        )
        if metadata_bytes > limits.max_metadata_bytes:
            raise ValueError("untrained inbox collection exceeds aggregate metadata capacity")
    # Why: individual seals do not prove that the complete collection fits.
    checkpoint_content_stamp(
        tuple((tuple(vars(record.data).values()), record.tombstone()) for record in records), limits
    )
    keys: set[tuple[str, str]] = set()
    events: set[str] = set()
    tombstones: set[int] = set()
    for record in records:
        key, event, tombstone = record.data.key, record.data.event_id, record.tombstone()
        if key in keys or (event is not None and event in events) or id(tombstone) in tombstones:
            raise ValueError("untrained inbox collection has duplicate keys/events/tombstones")
        keys.add(key)
        if event is not None:
            events.add(event)
        tombstones.add(id(tombstone))


def require_same_untrained_inbox_history(
    original: UntrainedInboxOrigin,
    copied: UntrainedInboxOrigin,
    limits: CheckpointContentLimits,
) -> None:
    """Prove a distinct tombstone with the same admitted scalar data, not authority."""
    untrained_inbox_origin_stamp(original, limits)
    untrained_inbox_origin_stamp(copied, limits)
    if copied.data is not original.data or copied.tombstone() is original.tombstone():
        raise ValueError(
            "untrained inbox copy aliases its original or replaces admitted scalar data"
        )
    # Both seals already require every tombstone field to match the same data.
    checkpoint_content_stamp((original.tombstone(), copied.tombstone()), limits)


def bind_untrained_inbox_records(
    records: tuple[UntrainedInboxOrigin, ...],
    lookup: Callable[[object], object],
    limits: CheckpointContentLimits,
    maximum: int,
) -> tuple[UntrainedInboxOrigin, ...]:
    """Bind caller memo objects and reprove all witnesses after opaque lookup."""
    limit_values = _bound(limits, maximum)
    require_untrained_inbox_records(records, limits, maximum)
    before = tuple(untrained_inbox_origin_stamp(record, limits) for record in records)
    result: list[UntrainedInboxOrigin] = []
    stamps: list[tuple] = []
    for record in records:
        untrained_inbox_origin_stamp(record, limits)
        tombstone = record.tombstone()
        if tombstone is None:
            raise ValueError("untrained inbox memo original tombstone reference expired")
        try:
            copied = lookup(tombstone)
        except KeyError as error:
            raise ValueError("untrained inbox memo lacks the actual tombstone object") from error
        if _bound(limits, maximum) != limit_values:
            raise ValueError("untrained inbox memo original proof limits changed")
        if type(copied) is not ErasedExperience:
            raise ValueError("untrained inbox memo requires an exact tombstone object")
        rebound = prepare_untrained_inbox_origin(record.data, copied, limits)
        require_same_untrained_inbox_history(record, rebound, limits)
        result.append(rebound)
        stamps.append(untrained_inbox_origin_stamp(rebound, limits))
    copied_records = tuple(result)
    require_untrained_inbox_records(records, limits, maximum)
    require_untrained_inbox_records(copied_records, limits, maximum)
    if (
        _bound(limits, maximum) != limit_values
        or tuple(untrained_inbox_origin_stamp(record, limits) for record in records) != before
        or tuple(untrained_inbox_origin_stamp(record, limits) for record in copied_records)
        != tuple(stamps)
    ):
        raise ValueError("untrained inbox memo witnesses changed during opaque lookup")
    for original, rebound in zip(records, copied_records):
        require_same_untrained_inbox_history(original, rebound, limits)
    return copied_records
