"""Bounded payload-free integrity witnesses for already observed inbox erasure.

Inputs are original admitted metadata, a committed receipt and its actual
tombstone. Outputs retain only immutable metadata and weak receipt/tombstone
references. The caller proves original authority and pays admission first;
these helpers perform no enrollment, cleanup, payload access or ownership work.
"""

from dataclasses import dataclass, fields
from hashlib import sha256
from math import isfinite
from typing import Any
from weakref import ReferenceType, ref

from src.core.checkpoint_content import CheckpointContentLimits, checkpoint_content_stamp
from src.core.data_erasure import ErasedExperience
from src.core.experience import AppliedExperience, require_identifier
from src.core.inbox_origin import InboxOriginData, inbox_origin_metadata
from src.core.learner_ports import TrainingDiagnostic


@dataclass(frozen=True, slots=True)
class ErasedInboxOrigin:
    data: InboxOriginData
    receipt: ReferenceType[Any]
    tombstone: ReferenceType[Any]
    receipt_stamp: str
    tombstone_stamp: str


def _schema(value: Any, kind: type) -> None:
    if type(value) is not kind:
        raise ValueError("erased inbox origin requires exact supported metadata types")
    values = vars(value)
    names = tuple(field.name for field in fields(kind))
    if type(values) is not dict or len(values) != len(names):
        raise ValueError("erased inbox origin metadata field count changed")
    if any(type(key) is not str or len(key) > 64 for key in values):
        raise ValueError("erased inbox origin field keys require bounded exact strings")
    if values.keys() != set(names):
        raise ValueError("erased inbox origin metadata schema changed")


def _limits(limits: CheckpointContentLimits) -> None:
    _schema(limits, CheckpointContentLimits)
    CheckpointContentLimits.__post_init__(limits)


def _identifier(value: Any, maximum: int) -> None:
    if type(value) is not str or len(value) * 10 + 16 > maximum:
        raise ValueError("erased inbox origin identity requires a bounded exact string")
    require_identifier(value, "erased inbox origin identity")


def _counter(value: Any) -> None:
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("erased inbox origin counters require bounded exact integers")


def _receipt(receipt: AppliedExperience, limits: CheckpointContentLimits) -> None:
    _schema(receipt, AppliedExperience)
    for identity in (
        receipt.sample_id,
        receipt.episode_id,
        receipt.event_id,
        receipt.actor_version,
        receipt.learner_version,
    ):
        _identifier(identity, limits.max_metadata_bytes)
    for counter in (
        receipt.observed_at,
        receipt.arrived_at,
        receipt.applied_at,
        receipt.update_number,
    ):
        _counter(counter)
    _schema(receipt.diagnostic, TrainingDiagnostic)
    _identifier(receipt.diagnostic.definition, limits.max_metadata_bytes)
    diagnostic = receipt.diagnostic.value
    if type(diagnostic) is not int and type(diagnostic) is not float:
        raise ValueError("erased inbox diagnostic requires an exact finite scalar")
    if type(diagnostic) is int and diagnostic.bit_length() > 1024:
        raise ValueError("erased inbox diagnostic exceeds bounded scalar size")
    try:
        finite = isfinite(diagnostic)
    except OverflowError as error:
        raise ValueError("erased inbox diagnostic exceeds finite scalar bounds") from error
    if not finite:
        raise ValueError("erased inbox diagnostic requires a finite scalar")
    if not receipt.observed_at <= receipt.arrived_at <= receipt.applied_at:
        raise ValueError("erased inbox receipt chronology differs from original applied work")


def _tombstone(tombstone: ErasedExperience, limits: CheckpointContentLimits) -> None:
    _schema(tombstone, ErasedExperience)
    if type(tombstone.key) is not tuple or len(tombstone.key) != 2:
        raise ValueError("erased inbox tombstone requires an exact paired key")
    for identity in (*tombstone.key, tombstone.actor_version, tombstone.event_id):
        _identifier(identity, limits.max_metadata_bytes)
    for counter in (tombstone.observed_at, tombstone.arrived_at, tombstone.erased_at):
        _counter(counter)
    if (
        type(tombstone.reason) is not str
        or len(tombstone.reason) > 64
        or tombstone.reason not in ("deleted", "expired", "opt_out")
    ):
        raise ValueError("erased inbox tombstone reason changed")
    ErasedExperience.__post_init__(tombstone)


def _metadata(
    data: InboxOriginData,
    receipt: AppliedExperience,
    tombstone: ErasedExperience,
    limits: CheckpointContentLimits,
) -> str:
    _limits(limits)
    _schema(data, InboxOriginData)
    encoded = inbox_origin_metadata(data, limits.max_metadata_bytes)
    if data.payload_bytes > limits.max_array_bytes:
        raise ValueError("erased inbox original payload metadata exceeds its bound")
    _receipt(receipt, limits)
    _tombstone(tombstone, limits)
    if (
        (receipt.episode_id, receipt.sample_id) != data.key
        or receipt.event_id != data.event_id
        or receipt.actor_version != data.actor_version
        or receipt.learner_version != data.learner_version
        or receipt.observed_at != data.observed_at
        or receipt.arrived_at != data.arrived_at
        or receipt.update_number != data.update_number
        or tombstone.key != data.key
        or tombstone.actor_version != data.actor_version
        or tombstone.event_id != data.event_id
        or tombstone.observed_at != data.observed_at
        or tombstone.arrived_at != data.arrived_at
        or tombstone.erased_at < receipt.applied_at
    ):
        raise ValueError("erased inbox receipt/tombstone differs from original admitted metadata")
    # One proof bounds the aggregate scalar records before either seal is made.
    checkpoint_content_stamp((receipt, tombstone), limits)
    return sha256(encoded).hexdigest()


def _seals(
    data: InboxOriginData,
    metadata: str,
    receipt: AppliedExperience,
    tombstone: ErasedExperience,
    receipt_ref: ReferenceType[Any],
    tombstone_ref: ReferenceType[Any],
    limits: CheckpointContentLimits,
) -> tuple[str, str]:
    # Why: seal both original object and weak-reference identities, including
    # admitted metadata. Equal clones cannot replace an original weak witness.
    receipt_stamp = checkpoint_content_stamp(
        (id(data), metadata, id(receipt_ref), id(receipt), receipt), limits
    )
    tombstone_stamp = checkpoint_content_stamp(
        (id(data), metadata, id(tombstone_ref), id(tombstone), tombstone), limits
    )
    return receipt_stamp, tombstone_stamp


def prepare_erased_inbox_origin(
    data: InboxOriginData,
    receipt: AppliedExperience,
    tombstone: ErasedExperience,
    limits: CheckpointContentLimits,
) -> ErasedInboxOrigin:
    """Prepare only integrity; the caller has already proved and paid authority."""
    metadata = _metadata(data, receipt, tombstone, limits)
    receipt_ref, tombstone_ref = ref(receipt), ref(tombstone)
    receipt_stamp, tombstone_stamp = _seals(
        data, metadata, receipt, tombstone, receipt_ref, tombstone_ref, limits
    )
    return ErasedInboxOrigin(data, receipt_ref, tombstone_ref, receipt_stamp, tombstone_stamp)


def erased_inbox_origin_stamp(record: ErasedInboxOrigin, limits: CheckpointContentLimits) -> tuple:
    """Return a deterministic pure proof of exact current refs and contents."""
    _limits(limits)
    if type(record) is not ErasedInboxOrigin:
        raise ValueError("erased inbox origin requires its exact slotted witness type")
    if type(record.receipt) is not ReferenceType or type(record.tombstone) is not ReferenceType:
        raise ValueError("erased inbox origin requires exact weak references")
    for value in (record.receipt_stamp, record.tombstone_stamp):
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("erased inbox origin requires exact bounded SHA256 seals")
    receipt, tombstone = record.receipt(), record.tombstone()
    if receipt is None or tombstone is None:
        raise ValueError("erased inbox origin original receipt/tombstone reference expired")
    metadata = _metadata(record.data, receipt, tombstone, limits)
    receipt_stamp, tombstone_stamp = _seals(
        record.data, metadata, receipt, tombstone, record.receipt, record.tombstone, limits
    )
    if receipt_stamp != record.receipt_stamp or tombstone_stamp != record.tombstone_stamp:
        raise ValueError("erased inbox origin original metadata/identity/content seal changed")
    return (
        id(record),
        id(record.data),
        id(record.receipt),
        id(record.tombstone),
        id(receipt),
        id(tombstone),
        metadata,
        receipt_stamp,
        tombstone_stamp,
    )
