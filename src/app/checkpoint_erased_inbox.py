"""Pure complete inbox partitions and actual erased receipt/tombstone memo binding.

These checks retain no payloads and grant no original consent, erasure, admission
or copy authority. The coordinator qualifies the actual cursor, live witnesses,
erased enrollment and memo lookup under its existing original operation leases.
"""

from math import isfinite
from typing import Any, Callable
from weakref import ReferenceType

from src.app.erased_inbox_origins import (
    ErasedInboxOrigin,
    _counter,
    _identifier,
    _limits,
    _receipt,
    _schema,
    _tombstone,
    erased_inbox_origin_stamp,
    prepare_erased_inbox_origin,
)
from src.app.untrained_inbox_origins import UntrainedInboxOrigin, untrained_inbox_origin_stamp
from src.core.checkpoint_content import CheckpointContentLimits, checkpoint_content_stamp
from src.core.data_erasure import ErasedExperience
from src.core.experience import AppliedExperience, Experience, ExperiencePermissions, LabelArrival
from src.core.inbox_cursor import InboxCursor
from src.core.inbox_origin import InboxOriginData


def _bound(limits: CheckpointContentLimits, maximum: int) -> tuple[int, ...]:
    _limits(limits)
    if type(maximum) is not int or not 0 < maximum <= min(4096, limits.max_nodes):
        raise ValueError("inbox partition requires a bounded exact original maximum")
    return (limits.max_nodes, limits.max_metadata_bytes, limits.max_array_bytes, limits.max_depth)


def _tuple(values: Any, maximum: int) -> None:
    if type(values) is not tuple or len(values) > maximum:
        raise ValueError("inbox partition requires bounded exact tuples")


def _erased_metadata(record: ErasedInboxOrigin, limits: CheckpointContentLimits) -> tuple:
    if type(record) is not ErasedInboxOrigin:
        raise ValueError("inbox erased history requires exact slotted witnesses")
    _schema(record.data, InboxOriginData)
    data = record.data
    if type(data.key) is not tuple or len(data.key) != 2:
        raise ValueError("inbox erased metadata requires an exact paired key")
    for identity in (
        *data.key,
        data.event_id,
        data.actor_version,
        data.learner_version,
        data.subject_id,
        data.source_id,
    ):
        _identifier(identity, limits.max_metadata_bytes)
    for counter in (data.observed_at, data.arrived_at, data.update_number, data.payload_bytes):
        _counter(counter)
    for digest in (data.features_digest, data.targets_digest):
        if type(digest) is not str or len(digest) != 64:
            raise ValueError("inbox erased metadata requires bounded digest strings")
    if type(record.receipt) is not ReferenceType or type(record.tombstone) is not ReferenceType:
        raise ValueError("inbox erased metadata requires exact weak references")
    receipt, tombstone = record.receipt(), record.tombstone()
    if receipt is None or tombstone is None:
        raise ValueError("inbox erased metadata references expired")
    _receipt(receipt, limits)
    _tombstone(tombstone, limits)
    values = (
        data.key,
        data.event_id,
        data.actor_version,
        data.learner_version,
        data.subject_id,
        data.source_id,
        data.observed_at,
        data.arrived_at,
        data.update_number,
        data.payload_bytes,
        data.features_digest,
        data.targets_digest,
    )
    return values, receipt, tombstone


def require_erased_inbox_records(
    records: tuple[ErasedInboxOrigin, ...], limits: CheckpointContentLimits, maximum: int
) -> None:
    """Prove original sealed weak records before hashing keys or dereferencing refs."""
    _bound(limits, maximum)
    _tuple(records, maximum)
    # Validate the aggregate metadata-only graph before per-record hash proofs.
    checkpoint_content_stamp(tuple(_erased_metadata(record, limits) for record in records), limits)
    seen_keys: set[tuple[str, str]] = set()
    seen_events: set[str] = set()
    seen_receipts: set[int] = set()
    seen_tombstones: set[int] = set()
    for record in records:
        erased_inbox_origin_stamp(record, limits)
        receipt, tombstone = record.receipt(), record.tombstone()
        if (
            record.data.key in seen_keys
            or record.data.event_id in seen_events
            or id(receipt) in seen_receipts
            or id(tombstone) in seen_tombstones
        ):
            raise ValueError("inbox erased history has duplicate keys/events/identities")
        seen_keys.add(record.data.key)
        seen_events.add(record.data.event_id)
        seen_receipts.add(id(receipt))
        seen_tombstones.add(id(tombstone))


def _source(source: Experience, limits: CheckpointContentLimits) -> tuple:
    _schema(source, Experience)
    for identity in (source.sample_id, source.episode_id, source.model_version):
        _identifier(identity, limits.max_metadata_bytes)
    _counter(source.observed_at)
    _schema(source.permissions, ExperiencePermissions)
    if any(
        type(flag) is not bool
        for flag in (
            source.permissions.training,
            source.permissions.replay,
            source.permissions.evaluation,
        )
    ):
        raise ValueError("inbox live permissions require exact boolean metadata")
    if (
        type(source.role) is not str
        or len(source.role) > 64
        or source.role != "train"
        or not source.permissions.training
    ):
        raise ValueError("inbox live partition requires original train-role sources")
    _tuple(source.candidate_ids, limits.max_nodes)
    for candidate in source.candidate_ids:
        _identifier(candidate, limits.max_metadata_bytes)
    if source.action_id is not None:
        _identifier(source.action_id, limits.max_metadata_bytes)
    reward = source.reward
    if reward is not None:
        if type(reward) is not int and type(reward) is not float:
            raise ValueError("inbox source reward requires an exact finite scalar")
        if type(reward) is int and reward.bit_length() > 1024:
            raise ValueError("inbox source reward exceeds scalar bounds")
        try:
            finite = isfinite(reward)
        except OverflowError as error:
            raise ValueError("inbox source reward exceeds finite scalar bounds") from error
        if not finite:
            raise ValueError("inbox source reward requires finite metadata")
    # Payloads are independently bound by the outer coordinator. Visit only
    # fixed scalar metadata here, so this partition check never reads arrays.
    values = (
        source.sample_id,
        source.episode_id,
        source.model_version,
        source.observed_at,
        source.role,
        source.permissions,
        source.candidate_ids,
        source.action_id,
        source.reward,
    )
    checkpoint_content_stamp(values, limits)
    Experience.__post_init__(source)
    return values


def _label(label: LabelArrival, limits: CheckpointContentLimits) -> tuple:
    _schema(label, LabelArrival)
    for identity in (label.sample_id, label.episode_id, label.event_id, label.model_version):
        _identifier(identity, limits.max_metadata_bytes)
    _counter(label.arrived_at)
    if type(label.role) is not str or len(label.role) > 64 or label.role != "train":
        raise ValueError("inbox live partition requires train-role labels")
    values = (
        label.sample_id,
        label.episode_id,
        label.event_id,
        label.model_version,
        label.arrived_at,
        label.role,
    )
    checkpoint_content_stamp(values, limits)
    return values


def _partition_tombstone(tombstone: ErasedExperience, limits: CheckpointContentLimits) -> None:
    """Validate optional arrival metadata before complete witness coverage checks."""
    _schema(tombstone, ErasedExperience)
    if type(tombstone.key) is not tuple or len(tombstone.key) != 2:
        raise ValueError("inbox partition tombstone requires an exact paired key")
    for identity in (*tombstone.key, tombstone.actor_version):
        _identifier(identity, limits.max_metadata_bytes)
    if tombstone.event_id is not None:
        _identifier(tombstone.event_id, limits.max_metadata_bytes)
    _counter(tombstone.erased_at)
    for counter in (tombstone.observed_at, tombstone.arrived_at):
        if counter is not None:
            _counter(counter)
    if (
        type(tombstone.reason) is not str
        or len(tombstone.reason) > 64
        or tombstone.reason not in ("deleted", "expired", "opt_out")
    ):
        raise ValueError("inbox partition tombstone reason changed")
    ErasedExperience.__post_init__(tombstone)


def require_inbox_partition(
    cursor: InboxCursor,
    live_receipts: tuple[AppliedExperience, ...],
    erased: tuple[ErasedInboxOrigin, ...],
    limits: CheckpointContentLimits,
    maximum: int,
    *,
    untrained: tuple[UntrainedInboxOrigin, ...] = (),
) -> None:
    """Prove applied live/trained coverage and complete trained/untrained erasure."""
    # The memo helper reuses this module's bounds; defer its import to avoid
    # a module initialization cycle while preserving the existing public API.
    from src.app.checkpoint_untrained_inbox import require_untrained_inbox_records

    _bound(limits, maximum)
    _schema(cursor, InboxCursor)
    _tuple(live_receipts, maximum)
    _tuple(erased, maximum)
    _tuple(untrained, maximum)
    for values in (cursor.experiences, cursor.labels, cursor.applied, cursor.erased):
        _tuple(values, maximum)
    for counter in (
        cursor.format_version,
        cursor.capacity,
        cursor.last_tick,
        cursor.completed_updates,
    ):
        _counter(counter)
    _identifier(cursor.learner_version, limits.max_metadata_bytes)
    if (
        type(cursor.format_version) is not int
        or cursor.format_version not in (1, 2)
        or type(cursor.stopped) is not bool
        or not 0 < cursor.capacity <= maximum
        or (cursor.format_version == 1) != (len(cursor.erased) == 0)
        or cursor.completed_updates != len(cursor.applied)
    ):
        raise ValueError("inbox partition cursor format/capacity/work metadata changed")
    if len(live_receipts) + len(erased) + len(untrained) > min(maximum, cursor.capacity):
        raise ValueError("inbox partition combined identity capacity exceeded")
    source_values = tuple(_source(source, limits) for source in cursor.experiences)
    label_values = tuple(_label(label, limits) for label in cursor.labels)
    for receipt in (*cursor.applied, *live_receipts):
        _receipt(receipt, limits)
    for tombstone in cursor.erased:
        _partition_tombstone(tombstone, limits)
    erased_values = tuple(_erased_metadata(record, limits) for record in erased)
    for untrained_record in untrained:
        untrained_inbox_origin_stamp(untrained_record, limits)
    untrained_values = tuple(
        (tuple(vars(record.data).values()), record.tombstone()) for record in untrained
    )
    history_values = (erased_values, untrained_values) if untrained else ()
    checkpoint_content_stamp(
        (
            cursor.format_version,
            cursor.learner_version,
            cursor.capacity,
            cursor.last_tick,
            cursor.stopped,
            cursor.completed_updates,
            source_values,
            label_values,
            cursor.applied,
            cursor.erased,
            live_receipts,
            *history_values,
        ),
        limits,
    )
    # All three scalar graphs fit together before any inventory key is hashed.
    require_erased_inbox_records(erased, limits, maximum)
    require_untrained_inbox_records(untrained, limits, maximum)
    sources: dict[tuple[str, str], Experience] = {}
    labels: dict[tuple[str, str], LabelArrival] = {}
    events: set[str] = set()
    for source in cursor.experiences:
        if source.key in sources:
            raise ValueError("inbox partition has duplicate live source keys")
        sources[source.key] = source
    for label in cursor.labels:
        if label.key in labels or label.event_id in events:
            raise ValueError("inbox partition has duplicate live label/event keys")
        labels[label.key] = label
        events.add(label.event_id)
    live: dict[tuple[str, str], AppliedExperience] = {}
    for receipt in live_receipts:
        _receipt(receipt, limits)
        checkpoint_content_stamp(receipt, limits)
        key = (receipt.episode_id, receipt.sample_id)
        if key in live:
            raise ValueError("inbox partition has duplicate live receipt keys")
        live[key] = receipt
    erased_receipts: dict[tuple[str, str], AppliedExperience] = {}
    tombstones: dict[tuple[str, str], ErasedExperience] = {}
    for erased_record in erased:
        erased_receipt, erased_tombstone = erased_record.receipt(), erased_record.tombstone()
        if erased_receipt is None or erased_tombstone is None:
            raise ValueError("inbox partition erased references expired")
        erased_receipts[erased_record.data.key] = erased_receipt
        tombstones[erased_record.data.key] = erased_tombstone
        if erased_record.data.event_id in events:
            raise ValueError("inbox partition erased/live event identities overlap")
        events.add(erased_record.data.event_id)
    untrained_keys: set[tuple[str, str]] = set()
    for untrained_record in untrained:
        untrained_tombstone = untrained_record.tombstone()
        if untrained_tombstone is None:
            raise ValueError("inbox partition untrained reference expired")
        key, event = untrained_record.data.key, untrained_record.data.event_id
        if (
            key in sources
            or key in labels
            or key in live
            or key in erased_receipts
            or key in tombstones
            or (event is not None and event in events)
        ):
            raise ValueError("inbox partition untrained keys/events overlap other histories")
        if (
            untrained_record.data.learner_version != cursor.learner_version
            or untrained_record.data.completed_updates > cursor.completed_updates
        ):
            raise ValueError("inbox partition untrained learner/work boundary changed")
        untrained_keys.add(key)
        tombstones[key] = untrained_tombstone
        if event is not None:
            events.add(event)
    if (
        sources.keys() != live.keys()
        or labels.keys() != live.keys()
        or live.keys() & erased_receipts.keys()
        or len(live) + len(erased_receipts) != len(cursor.applied)
        or len(live) + len(erased_receipts) + len(untrained_keys) > cursor.capacity
        or len(tombstones) != len(cursor.erased)
    ):
        raise ValueError("inbox partition is incomplete or live/erased keys overlap")
    seen_tombstones: set[tuple[str, str]] = set()
    for tombstone in cursor.erased:
        _partition_tombstone(tombstone, limits)
        checkpoint_content_stamp(tombstone, limits)
        if (
            tombstone.key in seen_tombstones
            or tombstones.get(tombstone.key) is not tombstone
            or tombstone.erased_at > cursor.last_tick
        ):
            raise ValueError("inbox partition tombstone lacks its exact original erased witness")
        seen_tombstones.add(tombstone.key)
    seen: set[tuple[str, str]] = set()
    last_applied = 0
    for number, receipt in enumerate(cursor.applied, 1):
        _receipt(receipt, limits)
        checkpoint_content_stamp(receipt, limits)
        key = (receipt.episode_id, receipt.sample_id)
        expected = live.get(key) if key in live else erased_receipts.get(key)
        if (
            key in seen
            or receipt is not expected
            or receipt.update_number != number
            or receipt.learner_version != cursor.learner_version
            or not last_applied <= receipt.applied_at <= cursor.last_tick
        ):
            raise ValueError("inbox partition applied identity/order/work coverage changed")
        if key in live:
            source, label = sources[key], labels[key]
            if (
                receipt.actor_version != source.model_version
                or source.model_version != label.model_version
                or receipt.event_id != label.event_id
                or receipt.observed_at != source.observed_at
                or receipt.arrived_at != label.arrived_at
            ):
                raise ValueError("inbox live receipt differs from original paired metadata")
        seen.add(key)
        last_applied = receipt.applied_at


def _receipt_values(receipt: AppliedExperience, limits: CheckpointContentLimits) -> tuple:
    _receipt(receipt, limits)
    checkpoint_content_stamp(receipt, limits)
    return (
        receipt.sample_id,
        receipt.episode_id,
        receipt.event_id,
        receipt.actor_version,
        receipt.learner_version,
        receipt.observed_at,
        receipt.arrived_at,
        receipt.applied_at,
        receipt.update_number,
        receipt.diagnostic.definition,
        receipt.diagnostic.value,
    )


def _tombstone_values(tombstone: ErasedExperience, limits: CheckpointContentLimits) -> tuple:
    _tombstone(tombstone, limits)
    checkpoint_content_stamp(tombstone, limits)
    return (
        tombstone.key,
        tombstone.actor_version,
        tombstone.observed_at,
        tombstone.event_id,
        tombstone.arrived_at,
        tombstone.erased_at,
        tombstone.reason,
    )


def require_same_erased_inbox_history(
    original: ErasedInboxOrigin, copied: ErasedInboxOrigin, limits: CheckpointContentLimits
) -> None:
    """Prove unchanged scalar history and distinct objects, without granting authority."""
    erased_inbox_origin_stamp(original, limits)
    erased_inbox_origin_stamp(copied, limits)
    original_receipt, original_tombstone = original.receipt(), original.tombstone()
    copied_receipt, copied_tombstone = copied.receipt(), copied.tombstone()
    if (
        original_receipt is None
        or original_tombstone is None
        or copied_receipt is None
        or copied_tombstone is None
    ):
        raise ValueError("inbox copied erased history references expired")
    checkpoint_content_stamp(
        (original_receipt, original_tombstone, copied_receipt, copied_tombstone), limits
    )
    if (
        copied.data is not original.data
        or copied_receipt is original_receipt
        or copied_tombstone is original_tombstone
        or copied_receipt.diagnostic is original_receipt.diagnostic
        or type(copied_receipt.diagnostic.value) is not type(original_receipt.diagnostic.value)
        or _receipt_values(copied_receipt, limits) != _receipt_values(original_receipt, limits)
        or _tombstone_values(copied_tombstone, limits)
        != _tombstone_values(original_tombstone, limits)
    ):
        raise ValueError("inbox erased copy contains aliases or changed scalar history")


def bind_erased_inbox_records(
    records: tuple[ErasedInboxOrigin, ...],
    lookup: Callable[[object], object],
    limits: CheckpointContentLimits,
    maximum: int,
) -> tuple[ErasedInboxOrigin, ...]:
    """Bind qualified actual memo objects; never infer memo authority from equality."""
    limit_values = _bound(limits, maximum)
    require_erased_inbox_records(records, limits, maximum)
    before = tuple(erased_inbox_origin_stamp(record, limits) for record in records)
    result: list[ErasedInboxOrigin] = []
    prepared_stamps: list[tuple] = []
    for record in records:
        # A previous opaque lookup may have replaced a later record's weak refs.
        erased_inbox_origin_stamp(record, limits)
        original_receipt, original_tombstone = record.receipt(), record.tombstone()
        if original_receipt is None or original_tombstone is None:
            raise ValueError("inbox memo original erased references expired")
        try:
            receipt = lookup(original_receipt)
            tombstone = lookup(original_tombstone)
        except KeyError as error:
            raise ValueError("inbox memo lacks actual erased receipt/tombstone objects") from error
        if _bound(limits, maximum) != limit_values:
            raise ValueError("inbox memo original proof limits changed")
        if type(receipt) is not AppliedExperience or type(tombstone) is not ErasedExperience:
            raise ValueError("inbox erased memo requires exact receipt/tombstone objects")
        rebound = prepare_erased_inbox_origin(record.data, receipt, tombstone, limits)
        require_same_erased_inbox_history(record, rebound, limits)
        result.append(rebound)
        prepared_stamps.append(erased_inbox_origin_stamp(rebound, limits))
    require_erased_inbox_records(records, limits, maximum)
    require_erased_inbox_records(tuple(result), limits, maximum)
    if tuple(erased_inbox_origin_stamp(record, limits) for record in records) != before or tuple(
        erased_inbox_origin_stamp(record, limits) for record in result
    ) != tuple(prepared_stamps):
        raise ValueError("inbox erased memo witnesses changed during opaque lookup")
    for original, copied in zip(records, result):
        require_same_erased_inbox_history(original, copied, limits)
    return tuple(result)
