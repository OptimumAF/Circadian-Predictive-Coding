"""Prepay mixed original expiry history and reprove it after raw removal.

Caller owns original cleanup leases and proves birth/expiry/native authority.
The staged maps contain only original weak witnesses and scalar metadata. This
module invokes no ports, acquires no locks, removes no payloads and publishes
nothing. Failed reservations remain spent under the same original admission.
"""

from dataclasses import dataclass, fields
from math import isfinite
from typing import Any
from weakref import ReferenceType

from src.app.erased_inbox_origins import (
    ErasedInboxOrigin,
    _counter,
    _receipt,
    erased_inbox_origin_stamp,
    prepare_erased_inbox_origin,
)
from src.app.expiry_inbox_qualification import (
    _require_key,
    _require_original_bindings,
    _require_owner_metadata,
    _require_ready_time,
)
from src.app.expiry_untrained_qualification import (
    _persistent_data,
    _require_unchanged,
    prepare_expiry_untrained_erasure,
)
from src.app.managed_inbox_origins import (
    _InboxOrigin,
    _declaration_values,
    _digest,
    _limits,
    _record_schema,
    _schema,
)
from src.app.managed_replay_origins import _Row
from src.app.untrained_inbox_origins import UntrainedInboxOrigin, untrained_inbox_origin_stamp
from src.core.checkpoint_content import CheckpointContentLimits, checkpoint_content_stamp
from src.core.data_erasure import ErasedExperience
from src.core.data_lifecycle import (
    DataConsent,
    DataProvenance,
    LifecycleDeclaration,
    LifecycleLimits,
)
from src.core.data_retention import DataCleanupReport
from src.core.experience import AppliedExperience
from src.core.inbox_origin import InboxOriginData
from src.core.learner_ports import TrainingDiagnostic
from src.core.replay_origin import (
    ReplayOriginAccounting,
    ReplayOriginAdmission,
    ReplayOriginData,
    ReplayOriginLimits,
)
from src.core.untrained_inbox_origin import UntrainedInboxOriginData


@dataclass(frozen=True, slots=True)
class _ExpiryTransition:
    live: tuple[dict[Any, Any], dict[Any, Any]]
    erased: tuple[dict[Any, Any], dict[Any, Any]]
    untrained: tuple[dict[Any, Any], dict[Any, Any]]
    original: tuple[Any, ...]
    limits: CheckpointContentLimits
    keys: tuple[Any, ...]
    now: int
    prepared_id: int
    proof: tuple[Any, ...]
    proof_stamp: str
    output: tuple[Any, ...]
    output_stamp: str


_TYPES = (
    _InboxOrigin,
    _Row,
    InboxOriginData,
    ReplayOriginData,
    UntrainedInboxOriginData,
    LifecycleDeclaration,
    DataConsent,
    DataProvenance,
    LifecycleLimits,
    AppliedExperience,
    TrainingDiagnostic,
    ErasedExperience,
    ReplayOriginLimits,
    ReplayOriginAccounting,
    CheckpointContentLimits,
    ErasedInboxOrigin,
    UntrainedInboxOrigin,
)
_SLOTS = (ErasedInboxOrigin, UntrainedInboxOrigin)
_MAP_NAMES = (
    "_records",
    "_sealed",
    "_storage",
    "_erased_records",
    "_erased_sealed",
    "_erased_storage",
    "_untrained_records",
    "_untrained_sealed",
    "_untrained_storage",
)


class _Scalars:
    """Bound fixed metadata while projecting it, before allocating a large proof."""

    __slots__ = ("limits", "nodes", "size")

    def __init__(self, limits: CheckpointContentLimits) -> None:
        _limits(limits)
        self.limits, self.nodes, self.size = limits, 0, 0

    def take(self, count: int = 1, size: int = 16) -> None:
        self.nodes += count
        self.size += size
        if self.nodes > self.limits.max_nodes or self.size > self.limits.max_metadata_bytes:
            raise ValueError("expiry transition aggregate scalar proof exceeds original limits")

    def value(self, value: Any, depth: int = 0) -> Any:
        if depth > self.limits.max_depth:
            raise ValueError("expiry transition scalar proof exceeds original depth")
        kind = type(value)
        self.take()
        if value is None or kind is bool:
            return value
        if kind is int:
            if value.bit_length() > 1024:
                raise ValueError("expiry transition scalar integer exceeds original bounds")
            self.take(0, len(str(value)))
            return value
        if kind is float:
            if not isfinite(value):
                raise ValueError("expiry transition scalar must be finite")
            return value
        if kind is str:
            self.take(0, len(value) * 10)
            return value
        if kind is ReferenceType:
            return ("weak", id(value))
        if kind is tuple:
            if len(value) > self.limits.max_nodes - self.nodes:
                raise ValueError("expiry transition tuple exceeds original node bound")
            return (id(value), *(self.value(item, depth + 1) for item in value))
        if any(kind is supported for supported in _TYPES):
            return self.record(value, depth + 1)
        raise ValueError("expiry transition requires exact payload-free metadata")

    def record(self, value: Any, depth: int = 0) -> tuple:
        kind = type(value)
        if not any(kind is supported for supported in _TYPES):
            raise ValueError("expiry transition unsupported metadata record")
        if any(kind is supported for supported in _SLOTS):
            values = tuple(
                (field.name, object.__getattribute__(value, field.name)) for field in fields(kind)
            )
        else:
            values = tuple(_schema(value, kind).items())
        self.take(3 + len(values) * 2)
        return (id(value), *((name, self.value(item, depth + 1)) for name, item in values))


def _bound(proof: Any, limits: CheckpointContentLimits) -> None:
    # Saved proofs must also reject foreign equality operands before comparison.
    pending = [(proof, 0)]
    count = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > limits.max_nodes or depth > limits.max_depth:
            raise ValueError("expiry transition saved proof exceeds original bounds")
        kind = type(item)
        if kind is tuple:
            if count + len(pending) + len(item) > limits.max_nodes:
                raise ValueError("expiry transition saved tuple exceeds original bounds")
            pending.extend((child, depth + 1) for child in item)
        elif item is not None and not any(kind is match for match in (bool, int, float, str)):
            raise ValueError("expiry transition saved proof has foreign comparison operands")
    checkpoint_content_stamp(proof, limits)


def _mapping(mapping: Any, maximum: int, limits: Any, *, identities: bool = False) -> None:
    if type(mapping) is not dict or len(mapping) > maximum:
        raise ValueError("expiry transition requires bounded exact maps")
    for key in mapping:
        if identities:
            if type(key) is not int or not 0 < key < 2**64:
                raise ValueError("expiry transition original identity key changed")
        else:
            _require_key(key, limits.max_metadata_bytes)


def _history(history: Any, scalar: _Scalars) -> tuple:
    result = []
    for offset, kind in ((0, _InboxOrigin), (3, ErasedInboxOrigin), (6, UntrainedInboxOrigin)):
        records, sealed, storage = (
            getattr(history, name) for name in _MAP_NAMES[offset : offset + 3]
        )
        for mapping in (records, sealed):
            _mapping(mapping, history._capacity, scalar.limits, identities=offset == 0)
        if (
            type(storage) is not tuple
            or len(storage) != 2
            or storage[0] is not records
            or storage[1] is not sealed
            or records.keys() != sealed.keys()
        ):
            raise ValueError("expiry transition original weak storage changed")
        projected = []
        for key, record in records.items():
            if type(record) is not kind or sealed[key] is not record:
                raise ValueError("expiry transition original sealed witness replaced")
            if kind is _InboxOrigin:
                _record_schema(record, scalar.limits)
                if record.verified is not True or type(record.receipt) is not ReferenceType:
                    raise ValueError("expiry transition requires committed original live witnesses")
            elif kind is ErasedInboxOrigin:
                erased_inbox_origin_stamp(record, scalar.limits)
            else:
                untrained_inbox_origin_stamp(record, scalar.limits)
            projected.append((scalar.value(key), scalar.record(record)))
        result.append((id(records), id(sealed), id(storage), tuple(projected)))
    return tuple(result)


def _admission(history: Any, ledger: Any, scalar: _Scalars) -> tuple:
    admission = ledger._admission
    if (
        type(admission) is not ReplayOriginAdmission
        or len(vars(admission)) != 4
        or any(type(key) is not str or len(key) > 64 for key in vars(admission))
        or vars(admission).keys() != {"limits", "_original_limits", "_progress", "_minimum"}
    ):
        raise ValueError("expiry transition requires original unshadowed admission")
    if type(admission._minimum) is not tuple or len(admission._minimum) != 3:
        raise ValueError("expiry transition original charge minima changed")
    for item in admission._minimum:
        _counter(item)
    _schema(admission._progress, ReplayOriginAccounting)
    for item in vars(admission._progress).values():
        _counter(item)
    live = (
        len(ledger._rows)
        + ledger._copy_slots
        + len(history._records)
        + len(history._erased_records)
        + len(history._untrained_records)
    )
    ReplayOriginAdmission.accounting(admission, live)
    return (
        id(admission),
        scalar.record(admission.limits),
        scalar.record(admission._progress),
        scalar.value(admission._minimum),
    )


def _proof(history: Any, ledger: Any, owner: Any, runtime: Any, prepared: Any) -> tuple:
    scalar = _Scalars(history._content_limits)
    _require_owner_metadata(owner, history)
    inbox = runtime._inbox
    _mapping(owner._catalog, min(owner._limits.max_approved_records, 4096), scalar.limits)
    for mapping in (inbox._applied, prepared):
        _mapping(mapping, history._capacity, scalar.limits)
    if not owner._revoked_keys <= owner._catalog.keys():
        raise ValueError("expiry transition revocation lacks original declarations")
    _mapping(ledger._rows, history._capacity, scalar.limits, identities=True)
    rows = []
    for identity, row in ledger._rows.items():
        _schema(row, _Row)
        _schema(row.data, ReplayOriginData)
        if type(row.verified) is not bool:
            raise ValueError("expiry transition original row verification flag changed")
        _digest(row.metadata_digest, empty=True)
        for name in (
            "source",
            "label",
            "declaration",
            "snapshot",
            "features",
            "targets",
            "receipt",
        ):
            reference = getattr(row, name)
            if type(reference) is not ReferenceType and not (
                name in ("snapshot", "features", "targets", "receipt") and reference is None
            ):
                raise ValueError("expiry transition original row weak reference changed")
        metadata = scalar.record(row)
        ReplayOriginData.__post_init__(row.data)
        rows.append((scalar.value(identity), metadata))
    for flag in (inbox._stopped, inbox._draining):
        if type(flag) is not bool or flag:
            raise ValueError("expiry transition inbox stopped or draining")
    for counter in (
        inbox._capacity,
        inbox._last_time,
        ledger._copy_slots,
        ledger._minimum_copy_slots,
        ledger._last_work,
        runtime._budget.updates_completed,
    ):
        _counter(counter)
    if inbox._historical_completed_updates is not None:
        raise ValueError("expiry transition inbox is retired")
    if type(inbox._event_ids) is not set or len(inbox._event_ids) > history._capacity:
        raise ValueError("expiry transition requires bounded original event IDs")
    if any(
        type(event) is not str or len(event) > scalar.limits.max_metadata_bytes
        for event in inbox._event_ids
    ):
        raise ValueError("expiry transition requires bounded exact event IDs")
    catalog = []
    for key, declaration in owner._catalog.items():
        metadata = scalar.record(declaration)
        _declaration_values(declaration, scalar.limits)
        LifecycleLimits.require_supported(owner._limits, declaration)
        if declaration.key != key:
            raise ValueError("expiry transition original catalog key changed")
        catalog.append((scalar.value(key), metadata))
    receipts, tombstones = [], []
    for key, receipt in inbox._applied.items():
        metadata = scalar.record(receipt)
        _receipt(receipt, scalar.limits)
        if (receipt.episode_id, receipt.sample_id) != key:
            raise ValueError("expiry transition actual applied receipt key changed")
        receipts.append((scalar.value(key), metadata))
    for key, tombstone in prepared.items():
        _schema(tombstone, ErasedExperience)
        metadata = scalar.record(tombstone)
        ErasedExperience.__post_init__(tombstone)
        if tombstone.key != key:
            raise ValueError("expiry transition actual tombstone key changed")
        tombstones.append((scalar.value(key), metadata))
    events = {item.event_id for item in prepared.values() if item.event_id is not None}
    if inbox._event_ids != events:
        raise ValueError("expiry transition actual event index differs from complete tombstones")
    result = (
        tuple(
            id(value)
            for value in (
                history,
                ledger,
                owner,
                runtime,
                inbox,
                runtime._budget,
                owner._catalog,
                owner._revoked_keys,
                owner._opted_out,
                inbox._applied,
                inbox._event_ids,
                inbox._experiences,
                inbox._labels,
                ledger._anchors,
                prepared,
            )
        ),
        _history(history, scalar),
        _admission(history, ledger, scalar),
        scalar.record(owner._limits),
        tuple(catalog),
        tuple(receipts),
        tuple(tombstones),
        tuple(scalar.value(key) for key in sorted(owner._revoked_keys)),
        tuple(scalar.value(subject) for subject in sorted(owner._opted_out)),
        tuple(rows),
        (
            ledger._copy_slots,
            ledger._minimum_copy_slots,
            ledger._last_work,
            runtime._budget.updates_completed,
            inbox._capacity,
            inbox._last_time,
        ),
        scalar.value(runtime._candidate_version),
        scalar.value(runtime._base_actor_version),
        scalar.value(inbox._learner_version),
        tuple(sorted(inbox._event_ids)),
    )
    _bound(result, scalar.limits)
    return result


def _outputs(pairs: Any, limits: Any, maximum: int) -> tuple:
    scalar = _Scalars(limits)
    result = []
    total = 0
    if type(pairs) is not tuple or len(pairs) != 3:
        raise ValueError("expiry transition staged partition tuple changed")
    for pair, kind in zip(pairs, (_InboxOrigin, ErasedInboxOrigin, UntrainedInboxOrigin)):
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("expiry transition staged storage pair changed")
        records, sealed = pair
        for mapping in pair:
            _mapping(mapping, maximum, limits, identities=kind is _InboxOrigin)
        total += len(records)
        if total > maximum or records.keys() != sealed.keys():
            raise ValueError("expiry transition staged capacity/sealed membership changed")
        projected = []
        for key, record in records.items():
            if type(record) is not kind or sealed[key] is not record:
                raise ValueError("expiry transition staged sealed witness changed")
            if kind is ErasedInboxOrigin:
                erased_inbox_origin_stamp(record, limits)
            elif kind is UntrainedInboxOrigin:
                untrained_inbox_origin_stamp(record, limits)
            else:
                raise ValueError("expiry transition must remove all original current arrivals")
            if key != record.data.key:
                raise ValueError("expiry transition staged record key changed")
            projected.append((scalar.value(key), scalar.record(record)))
        result.append((id(pair), id(records), id(sealed), tuple(projected)))
    proof = tuple(result)
    _bound(proof, limits)
    return proof


def prepay_expiry_transition(
    history: Any,
    ledger: Any,
    owner: Any,
    runtime: Any,
    now: int,
    before4: Any,
    keys: Any,
    prepared: Any,
) -> _ExpiryTransition:
    """Pay all mixed records before persistent allocation; never publish."""
    arrivals = _require_unchanged(history, ledger, owner, runtime, now, before4, revoked=True)
    limits = history._content_limits
    if type(keys) is not tuple or not 0 < len(keys) <= history._capacity:
        raise ValueError("expiry transition requires bounded actual removal keys")
    for key in keys:
        _require_key(key, limits.max_metadata_bytes)
    trained = tuple(history._records.values())
    if len(set(keys)) != len(keys) or set(keys) != {row.data.key for row in trained} | {
        row.key for row in arrivals
    }:
        raise ValueError("expiry transition requires complete disjoint actual removal keys")
    _persistent_data(history, runtime, arrivals, prepared, now)
    initial = _proof(history, ledger, owner, runtime, prepared)
    live = (
        len(ledger._rows)
        + ledger._copy_slots
        + len(history._records)
        + len(history._erased_records)
        + len(history._untrained_records)
    )
    # Why: trained reservations precede the h2b reserve-all/allocate phase, so
    # every mixed record is paid before the first persistent witness/map exists.
    for index, record in enumerate(trained):
        ReplayOriginAdmission.reserve_inbox(ledger._admission, record.data, live + index)
    untrained = prepare_expiry_untrained_erasure(
        history, ledger, owner, runtime, now, before4, prepared, offset=len(trained)
    )
    _require_unchanged(history, ledger, owner, runtime, now, before4, revoked=True)
    erased: tuple[dict[Any, Any], dict[Any, Any]] = (
        history._erased_records.copy(),
        history._erased_sealed.copy(),
    )
    for record in trained:
        receipt = record.receipt()
        if receipt is None:
            raise ValueError("expiry transition original trained receipt expired")
        witness = prepare_erased_inbox_origin(
            record.data, receipt, prepared[record.data.key], limits
        )
        erased[0][record.data.key] = erased[1][record.data.key] = witness
    pairs: Any = ({}, {}), erased, untrained
    _require_unchanged(history, ledger, owner, runtime, now, before4, revoked=True)
    proof = _proof(history, ledger, owner, runtime, prepared)
    # Original accounting changes are checked by the fixed admission. Everything
    # else must remain the initial exact scalar/identity proof.
    if proof[:2] != initial[:2] or proof[3:] != initial[3:]:
        raise ValueError("expiry transition original metadata changed during preparation")
    output = (id(keys), keys, _outputs(pairs, limits, history._capacity))
    _bound(output, limits)
    original = tuple(getattr(history, name) for name in _MAP_NAMES)
    return _ExpiryTransition(
        pairs[0],
        pairs[1],
        pairs[2],
        original,
        limits,
        keys,
        now,
        id(prepared),
        proof,
        checkpoint_content_stamp(proof, limits),
        output,
        checkpoint_content_stamp(output, limits),
    )


def require_final_expiry_transition(
    staged: Any, history: Any, ledger: Any, owner: Any, runtime: Any, now: int, report: Any
) -> None:
    """Prove saved weak/scalar lineage after raw removal, with no publication."""
    if type(staged) is not _ExpiryTransition:
        raise ValueError("expiry transition requires exact original staged type")
    life = _require_original_bindings(history, ledger, owner, runtime)
    _require_ready_time(ledger, runtime, life, now)
    if staged.limits is not history._content_limits:
        raise ValueError("expiry transition original content limits replaced")
    _limits(staged.limits)
    if type(staged.keys) is not tuple or not 0 < len(staged.keys) <= history._capacity:
        raise ValueError("expiry transition original staged key tuple changed")
    for key in staged.keys:
        _require_key(key, staged.limits.max_metadata_bytes)
    _bound(staged.keys, staged.limits)
    for proof, seal in ((staged.proof, staged.proof_stamp), (staged.output, staged.output_stamp)):
        _bound(proof, staged.limits)
        if type(seal) is not str or checkpoint_content_stamp(proof, staged.limits) != seal:
            raise ValueError("expiry transition immutable saved proof changed")
    _counter(staged.now)
    if now != staged.now or type(staged.prepared_id) is not int:
        raise ValueError("expiry transition original logical time/prepared identity changed")
    if type(staged.original) is not tuple or len(staged.original) != len(_MAP_NAMES):
        raise ValueError("expiry transition original storage inventory changed")
    if any(getattr(history, name) is not value for name, value in zip(_MAP_NAMES, staged.original)):
        raise ValueError("expiry transition original history storage replaced")
    inbox = runtime._inbox
    for mapping in (inbox._experiences, inbox._labels):
        if type(mapping) is not dict or mapping:
            raise ValueError("expiry transition raw inbox removal is incomplete")
    if id(inbox._erased) != staged.prepared_id:
        raise ValueError("expiry transition actual prepared tombstone map replaced")
    current = _proof(history, ledger, owner, runtime, inbox._erased)
    if current != staged.proof:
        raise ValueError(
            "expiry transition original metadata/receipt/accounting changed after cleanup"
        )
    if (
        id(staged.keys),
        staged.keys,
        _outputs((staged.live, staged.erased, staged.untrained), staged.limits, history._capacity),
    ) != staged.output:
        raise ValueError("expiry transition prepared weak histories changed after cleanup")
    if staged.live[0] or staged.live[1]:
        raise ValueError("expiry transition prepared live history is not empty")
    if (
        staged.erased[0].keys() != inbox._applied.keys()
        or staged.erased[0].keys() | staged.untrained[0].keys() != inbox._erased.keys()
        or staged.erased[0].keys() & staged.untrained[0].keys()
    ):
        raise ValueError("expiry transition final receipt/tombstone partition is incomplete")
    for record in staged.erased[0].values():
        if (
            record.receipt() is not inbox._applied[record.data.key]
            or record.tombstone() is not inbox._erased[record.data.key]
        ):
            raise ValueError("expiry transition final trained identities changed")
    for record in staged.untrained[0].values():
        if record.tombstone() is not inbox._erased[record.data.key]:
            raise ValueError("expiry transition final untrained identity changed")
    _schema(report, DataCleanupReport)
    if type(staged.keys) is not tuple or len(staged.keys) > history._capacity:
        raise ValueError("expiry transition staged removal keys changed")
    for keys in (staged.keys, report.requested_keys, report.revoked_keys):
        if type(keys) is not tuple or len(keys) > 4096:
            raise ValueError("expiry transition requires bounded report keys")
        for key in keys:
            _require_key(key, staged.limits.max_metadata_bytes)
    if type(report.reason) is not str:
        raise ValueError("expiry transition report reason requires exact string")
    for counter in (
        report.model_snapshots_erased,
        report.inboxes_cleared,
        report.checkpoints_invalidated,
        report.promotions_invalidated,
    ):
        _counter(counter)
    checkpoint_content_stamp(
        (
            report.reason,
            report.requested_keys,
            report.revoked_keys,
            report.model_snapshots_erased,
            report.inboxes_cleared,
            report.checkpoints_invalidated,
            report.promotions_invalidated,
        ),
        staged.limits,
    )
    DataCleanupReport.__post_init__(report)
    if (
        report.reason != "expired"
        or not set(staged.keys) <= set(report.revoked_keys)
        or not set(report.revoked_keys) <= owner._revoked_keys
    ):
        raise ValueError("expiry transition report lacks actual original expiry revocation")
    for key in staged.keys:
        tombstone = inbox._erased[key]
        if tombstone.reason != "expired" or tombstone.erased_at != now:
            raise ValueError("expiry transition actual current expiry tombstone changed")
