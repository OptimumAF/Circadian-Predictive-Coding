"""Prospective weak inbox lineage under an original replay admission.

Only actual started/completed updates and a successful final poll admit history.
No payload ownership, retrospective enrollment, renewed allowance or native work.
The trusted ledger supplies consent/time/work checks; core stamps supply bounded
integrity checks. Copies may rebind already admitted witnesses without admission.
"""

from dataclasses import dataclass, fields, replace
from hashlib import sha256
from typing import Any, cast
from weakref import ReferenceType, ref

from src.core.checkpoint_content import CheckpointContentLimits, checkpoint_content_stamp
from src.core.data_erasure import ErasedExperience
from src.core.data_lifecycle import DataConsent, DataProvenance, LifecycleDeclaration
from src.core.experience import AppliedExperience, Experience, LabelArrival
from src.core.inbox_origin import (
    InboxOriginData,
    inbox_numeric_stamp,
    inbox_origin_metadata,
    inbox_payload_binding,
)
from src.core.native_update_origin import NativeUpdateOrigin
from src.core.learner_ports import TrainingDiagnostic
from src.core.replay_origin import ReplayOriginAdmission, ReplayOriginLimits
from src.app.erased_inbox_origins import (
    ErasedInboxOrigin,
    erased_inbox_origin_stamp,
    prepare_erased_inbox_origin,
)
from src.app.checkpoint_erased_inbox import (
    require_erased_inbox_records,
    require_same_erased_inbox_history,
)
from src.app.untrained_inbox_origins import UntrainedInboxOrigin, untrained_inbox_origin_stamp
from src.app.checkpoint_untrained_inbox import (
    require_untrained_inbox_records,
    require_same_untrained_inbox_history,
)
from src.core.untrained_inbox_origin import untrained_inbox_origin_metadata


@dataclass(frozen=True)
class _InboxOrigin:
    data: InboxOriginData
    source: ReferenceType[Any]
    label: ReferenceType[Any]
    declaration: ReferenceType[Any]
    features: ReferenceType[Any]
    targets: ReferenceType[Any]
    receipt: ReferenceType[Any] | None
    source_stamp: str
    receipt_stamp: str
    verified: bool


def _schema(value: Any, kind: type) -> dict[str, Any]:
    if type(value) is not kind:
        raise ValueError("inbox origin requires exact supported record types")
    values = vars(value)
    names = tuple(field.name for field in fields(kind))
    if type(values) is not dict or len(values) != len(names):
        raise ValueError("inbox origin fixed record field count changed")
    if any(type(key) is not str or len(key) > 64 for key in values):
        raise ValueError("inbox origin fixed record keys require bounded exact strings")
    if values.keys() != set(names):
        raise ValueError("inbox origin fixed record schema changed")
    return values


def _limits(limits: CheckpointContentLimits) -> tuple[int, ...]:
    _schema(limits, CheckpointContentLimits)
    CheckpointContentLimits.__post_init__(limits)
    return (limits.max_nodes, limits.max_metadata_bytes, limits.max_array_bytes, limits.max_depth)


def _weak(value: Any) -> ReferenceType[Any]:
    try:
        return ref(value)
    except TypeError as error:
        raise ValueError("inbox origin requires weak-referenceable original objects") from error


def _digest(value: Any, *, empty: bool = False) -> None:
    if type(value) is not str or not (
        (empty and value == "")
        or (len(value) == 64 and all(character in "0123456789abcdef" for character in value))
    ):
        raise ValueError("inbox origin content stamp changed")


def _declaration_values(declaration: Any, limits: CheckpointContentLimits) -> tuple:
    _schema(declaration, LifecycleDeclaration)
    _schema(declaration.provenance, DataProvenance)
    _schema(declaration.consent, DataConsent)
    result = (
        declaration.key,
        declaration.provenance.source_id,
        declaration.provenance.subject_id,
        declaration.provenance.verified,
        declaration.provenance.synthetic,
        declaration.consent.training,
        declaration.consent.replay,
        declaration.retention,
    )
    # Bound and reject callback metadata before trusted validators compare it.
    checkpoint_content_stamp(result, limits)
    if type(declaration.key) is not tuple or len(declaration.key) != 2:
        raise ValueError("inbox declaration requires an exact paired key")
    if any(type(item) is not str for item in declaration.key):
        raise ValueError("inbox declaration key requires exact strings")
    DataProvenance.__post_init__(declaration.provenance)
    DataConsent.__post_init__(declaration.consent)
    LifecycleDeclaration.__post_init__(declaration)
    return result


def _source_stamp(
    source: Any, label: Any, declaration: Any, limits: CheckpointContentLimits
) -> str:
    _schema(source, Experience)
    _schema(label, LabelArrival)
    declaration_values = _declaration_values(declaration, limits)
    declaration_stamp = checkpoint_content_stamp(declaration_values, limits)
    stamp = checkpoint_content_stamp(
        (id(source), source, id(label), label, id(declaration), declaration_stamp), limits
    )
    try:
        Experience.__post_init__(source)
    except OverflowError as error:
        raise ValueError("inbox source reward exceeds finite scalar bounds") from error
    LabelArrival.__post_init__(label)
    return stamp


def _receipt_values(receipt: Any, limits: CheckpointContentLimits) -> tuple:
    _schema(receipt, AppliedExperience)
    checkpoint_content_stamp(receipt, limits)
    for value in (
        receipt.sample_id,
        receipt.episode_id,
        receipt.event_id,
        receipt.actor_version,
        receipt.learner_version,
    ):
        if type(value) is not str or not value:
            raise ValueError("inbox receipt identities require exact nonempty strings")
    for value in (
        receipt.observed_at,
        receipt.arrived_at,
        receipt.applied_at,
        receipt.update_number,
    ):
        if type(value) is not int or not 0 <= value < 2**63:
            raise ValueError("inbox receipt counters require bounded exact integers")
    if receipt.update_number == 0:
        raise ValueError("inbox receipt requires positive committed work")
    _schema(receipt.diagnostic, TrainingDiagnostic)
    try:
        TrainingDiagnostic.__post_init__(receipt.diagnostic)
    except OverflowError as error:
        raise ValueError("inbox receipt diagnostic exceeds finite scalar bounds") from error
    # Diagnostic values are finite and exact before equality at copy binding.
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


def _receipt_stamp(receipt: Any, limits: CheckpointContentLimits) -> str:
    _receipt_values(receipt, limits)
    return checkpoint_content_stamp((id(receipt), receipt), limits)


def _record_schema(record: _InboxOrigin, content_limits: CheckpointContentLimits) -> bytes:
    _limits(content_limits)
    _schema(record, _InboxOrigin)
    _schema(record.data, InboxOriginData)
    metadata = inbox_origin_metadata(record.data, content_limits.max_metadata_bytes)
    if type(record.verified) is not bool:
        raise ValueError("inbox origin verification flag changed")
    _digest(record.source_stamp)
    _digest(record.receipt_stamp, empty=True)
    references = (record.source, record.label, record.declaration, record.features, record.targets)
    if any(type(reference) is not ReferenceType for reference in references):
        raise ValueError("inbox origin requires exact weak references")
    if record.receipt is not None and type(record.receipt) is not ReferenceType:
        raise ValueError("inbox receipt requires an exact weak reference")
    return metadata


def inbox_origin_stamp(record: _InboxOrigin, content_limits: CheckpointContentLimits) -> tuple:
    """Pure bounded original identity/content proof; never invokes ledger ports."""
    metadata = _record_schema(record, content_limits)
    references = (record.source, record.label, record.declaration, record.features, record.targets)
    source, label, declaration, features, targets = (reference() for reference in references)
    if (
        source is None
        or label is None
        or declaration is None
        or features is None
        or targets is None
    ):
        raise ValueError("inbox origin original source/declaration/payload reference expired")
    source_stamp = _source_stamp(source, label, declaration, content_limits)
    if source.features is not features or label.targets is not targets:
        raise ValueError("inbox origin actual borrowed payload references changed")
    if source_stamp != record.source_stamp:
        raise ValueError("inbox origin original source/declaration contents changed")
    if (
        inbox_numeric_stamp(features, content_limits) != record.data.features_digest
        or inbox_numeric_stamp(targets, content_limits) != record.data.targets_digest
    ):
        raise ValueError("inbox origin original numeric contents changed")
    if features.nbytes + targets.nbytes != record.data.payload_bytes:
        raise ValueError("inbox origin original payload size changed")
    if (
        record.data.key != source.key
        or source.key != label.key
        or record.data.event_id != label.event_id
        or record.data.actor_version != source.model_version
        or source.model_version != label.model_version
        or record.data.observed_at != source.observed_at
        or record.data.arrived_at != label.arrived_at
        or record.data.subject_id != declaration.provenance.subject_id
        or record.data.source_id != declaration.provenance.source_id
        or declaration.key != source.key
    ):
        raise ValueError("inbox origin bound metadata differs from original records")
    receipt = record.receipt() if record.receipt is not None else None
    if receipt is None:
        if record.verified or record.receipt_stamp != "":
            raise ValueError("inbox origin committed receipt expired")
    else:
        if _receipt_stamp(receipt, content_limits) != record.receipt_stamp:
            raise ValueError("inbox origin original receipt contents changed")
        if (
            (receipt.episode_id, receipt.sample_id) != record.data.key
            or receipt.event_id != record.data.event_id
            or receipt.actor_version != record.data.actor_version
            or receipt.learner_version != record.data.learner_version
            or receipt.observed_at != record.data.observed_at
            or receipt.arrived_at != record.data.arrived_at
            or receipt.update_number != record.data.update_number
        ):
            raise ValueError("inbox origin original receipt metadata changed")
    return (
        id(record),
        id(record.data),
        sha256(metadata).hexdigest(),
        tuple(id(reference) for reference in references),
        tuple(id(value) for value in (source, label, declaration, features, targets)),
        id(record.receipt),
        id(receipt),
        source_stamp,
        record.receipt_stamp,
        record.verified,
    )


class ManagedInboxOrigins:
    """Weak history installed only by the trusted ledger's original constructor."""

    __slots__ = (
        "_ledger",
        "_admission",
        "_limits",
        "_limit_values",
        "_content_limits",
        "_original_content_limits",
        "_content_limit_values",
        "_capacity",
        "_original_capacity",
        "_inbox_capacity",
        "_records",
        "_sealed",
        "_storage",
        "_erased_records",
        "_erased_sealed",
        "_erased_storage",
        "_untrained_records",
        "_untrained_sealed",
        "_untrained_storage",
        "__weakref__",
    )

    def __init__(self, ledger: Any) -> None:
        if ledger._enrolling_inbox_origins is not True:
            raise ValueError("inbox history requires the trusted original constructor birth scope")
        if (
            type(ledger._original_admission) is not ReferenceType
            or type(ledger._owner) is not ReferenceType
        ):
            raise ValueError("inbox history requires original weak admission and owner bindings")
        admission = ledger._admission
        if (
            type(admission) is not ReplayOriginAdmission
            or ledger._original_admission() is not admission
        ):
            raise ValueError("inbox origins require original replay admission")
        owner = ledger._owner()
        if owner is None:
            raise ValueError("inbox history original owner expired before birth")
        runtime = owner._shared._runtime
        if (
            type(ledger._last_work) is not int
            or ledger._last_work != 0
            or type(runtime._budget.updates_completed) is not int
            or runtime._budget.updates_completed != 0
            or type(ledger._rows) is not dict
            or len(ledger._rows) != 0
            or ledger._pending is not None
            or ledger._inbox_origins is not None
            or ledger._original_inbox_origins is not None
            or ledger._inbox_history_birth is not None
        ):
            raise ValueError(
                "inbox history requires fresh original work and absent prior enrollment"
            )
        if admission is None:
            raise ValueError("inbox origins original replay admission expired before birth")
        limit = admission.limits
        _schema(limit, ReplayOriginLimits)
        ReplayOriginLimits.__post_init__(limit)
        self._ledger, self._admission, self._limits = _weak(ledger), _weak(admission), _weak(limit)
        self._limit_values = tuple(vars(limit).values())
        self._content_limits = CheckpointContentLimits(
            4096, limit.max_metadata_bytes, limit.max_payload_bytes, 16
        )
        self._original_content_limits = self._content_limits
        self._content_limit_values = _limits(self._content_limits)
        # The trusted constructor has already bound these anchors. Calling its
        # binding check here would recurse into history before its weak install.
        capacities = (
            limit.max_live_records,
            limit.max_records_created,
            owner._limits.max_approved_records,
            owner._limits.max_replay_records,
            runtime._inbox._capacity,
            4096,
        )
        if any(type(value) is not int or not 0 < value < 2**63 for value in capacities):
            raise ValueError("inbox history requires positive original bounded capacity")
        self._capacity = min(capacities)
        self._original_capacity = self._capacity
        self._inbox_capacity = runtime._inbox._capacity
        self._records: dict[int, _InboxOrigin] = {}
        self._sealed: dict[int, _InboxOrigin] = {}
        self._storage = (self._records, self._sealed)
        self._erased_records: dict[tuple[str, str], ErasedInboxOrigin] = {}
        self._erased_sealed: dict[tuple[str, str], ErasedInboxOrigin] = {}
        self._erased_storage = (self._erased_records, self._erased_sealed)
        self._untrained_records: dict[tuple[str, str], UntrainedInboxOrigin] = {}
        self._untrained_sealed: dict[tuple[str, str], UntrainedInboxOrigin] = {}
        self._untrained_storage = (self._untrained_records, self._untrained_sealed)

    def _authority(self, ledger: Any) -> None:
        if any(
            type(value) is not ReferenceType
            for value in (self._ledger, self._admission, self._limits)
        ):
            raise ValueError("inbox origin authority requires exact weak references")
        admission, limit = self._admission(), self._limits()
        if (
            ledger is None
            or self._ledger() is not ledger
            or type(admission) is not ReplayOriginAdmission
            or cast(Any, ledger)._admission is not admission
            or type(limit) is not ReplayOriginLimits
            or admission.limits is not limit
            or admission._original_limits is not limit
        ):
            raise ValueError("inbox origin original admission/limits changed or expired")
        _schema(limit, ReplayOriginLimits)
        ReplayOriginLimits.__post_init__(limit)
        if (
            type(self._limit_values) is not tuple
            or len(self._limit_values) != 6
            or any(type(value) is not int or not 0 < value < 2**63 for value in self._limit_values)
            or tuple(vars(limit).values()) != self._limit_values
        ):
            raise ValueError("inbox origin original limit values changed")
        if self._content_limits is not self._original_content_limits:
            raise ValueError("inbox origin original content limits replaced")
        if (
            type(self._content_limit_values) is not tuple
            or len(self._content_limit_values) != 4
            or any(
                type(value) is not int or not 0 < value < 2**63
                for value in self._content_limit_values
            )
            or _limits(self._content_limits) != self._content_limit_values
        ):
            raise ValueError("inbox origin original content limit values changed")
        if (
            type(self._capacity) is not int
            or type(self._original_capacity) is not int
            or not 0 < self._original_capacity < 2**63
            or self._capacity != self._original_capacity
            or not 0 < self._capacity <= limit.max_live_records
        ):
            raise ValueError("inbox origin original history capacity changed")

    def _state(self, *, contents: bool = True) -> tuple[_InboxOrigin, ...]:
        if type(self._storage) is not tuple or len(self._storage) != 2:
            raise ValueError("inbox origin original storage tuple changed")
        if self._storage[0] is not self._records or self._storage[1] is not self._sealed:
            raise ValueError("inbox origin original history maps replaced")
        for mapping in (self._records, self._sealed):
            if type(mapping) is not dict or len(mapping) > self._capacity:
                raise ValueError("inbox origin history map type/capacity changed")
            if any(type(key) is not int or not 0 < key < 2**64 for key in mapping):
                raise ValueError("inbox origin history requires bounded exact identity keys")
        if self._records.keys() != self._sealed.keys():
            raise ValueError("inbox origin sealed history membership changed")
        self._erased_state(contents=contents)
        self._untrained_state(contents=contents)
        for key, record in self._records.items():
            if self._sealed[key] is not record:
                raise ValueError("inbox origin sealed record replaced")
            _record_schema(record, self._content_limits)
            if contents:
                inbox_origin_stamp(record, self._content_limits)
            source = record.source()
            if source is not None and id(source) != key:
                raise ValueError("inbox origin original source identity key changed")
        return tuple(self._records.values())

    def state_stamp(self) -> tuple:
        if type(self._ledger) is not ReferenceType:
            raise ValueError("inbox origin ledger reference changed")
        self._authority(self._ledger())
        records = self._state()
        return (
            id(self._records),
            id(self._sealed),
            id(self._storage),
            tuple(inbox_origin_stamp(record, self._content_limits) for record in records),
            id(self._erased_records),
            id(self._erased_sealed),
            id(self._erased_storage),
            tuple(
                erased_inbox_origin_stamp(record, self._content_limits)
                for record in self._erased_state()
            ),
            id(self._untrained_records),
            id(self._untrained_sealed),
            id(self._untrained_storage),
            tuple(
                untrained_inbox_origin_stamp(record, self._content_limits)
                for record in self._untrained_state()
            ),
        )

    def _erased_state(self, *, contents: bool = True) -> tuple[ErasedInboxOrigin, ...]:
        if type(self._records) is not dict:
            raise ValueError("original live inbox record storage changed")
        if (
            type(self._erased_storage) is not tuple
            or len(self._erased_storage) != 2
            or self._erased_storage[0] is not self._erased_records
            or self._erased_storage[1] is not self._erased_sealed
        ):
            raise ValueError("erased inbox origin original storage changed")
        for mapping in (self._erased_records, self._erased_sealed):
            if type(self._untrained_records) is not dict:
                raise ValueError("original untrained inbox record storage changed")
            if (
                type(mapping) is not dict
                or len(mapping) + len(self._records) + len(self._untrained_records) > self._capacity
            ):
                raise ValueError("erased inbox origin storage type/capacity changed")
            for key in mapping:
                if (
                    type(key) is not tuple
                    or len(key) != 2
                    or any(type(part) is not str for part in key)
                ):
                    raise ValueError("erased inbox origin requires exact paired key strings")
                checkpoint_content_stamp(key, self._content_limits)
        if self._erased_records.keys() != self._erased_sealed.keys():
            raise ValueError("erased inbox origin sealed membership changed")
        for key, record in self._erased_records.items():
            if type(record) is not ErasedInboxOrigin or self._erased_sealed[key] is not record:
                raise ValueError("erased inbox origin sealed record replaced")
            inbox_origin_metadata(record.data, self._content_limits.max_metadata_bytes)
            if (
                type(key) is not tuple
                or key != record.data.key
                or type(record.receipt) is not ReferenceType
                or type(record.tombstone) is not ReferenceType
            ):
                raise ValueError("erased inbox origin key or weak references changed")
            if contents:
                erased_inbox_origin_stamp(record, self._content_limits)
        return tuple(self._erased_records.values())

    def _untrained_state(self, *, contents: bool = True) -> tuple[UntrainedInboxOrigin, ...]:
        if (
            type(self._untrained_storage) is not tuple
            or len(self._untrained_storage) != 2
            or self._untrained_storage[0] is not self._untrained_records
            or self._untrained_storage[1] is not self._untrained_sealed
            or type(self._records) is not dict
            or type(self._erased_records) is not dict
        ):
            raise ValueError("untrained inbox original storage changed")
        for mapping in (self._untrained_records, self._untrained_sealed):
            if (
                type(mapping) is not dict
                or len(mapping) + len(self._records) + len(self._erased_records) > self._capacity
            ):
                raise ValueError("untrained inbox original map/capacity changed")
            for key in mapping:
                if (
                    type(key) is not tuple
                    or len(key) != 2
                    or any(type(part) is not str for part in key)
                ):
                    raise ValueError("untrained inbox requires exact paired keys")
                checkpoint_content_stamp(key, self._content_limits)
        if self._untrained_records.keys() != self._untrained_sealed.keys():
            raise ValueError("untrained inbox sealed membership changed")
        for key, record in self._untrained_records.items():
            if (
                type(record) is not UntrainedInboxOrigin
                or self._untrained_sealed[key] is not record
            ):
                raise ValueError("untrained inbox sealed record replaced")
            untrained_inbox_origin_metadata(record.data, self._content_limits.max_metadata_bytes)
            if key != record.data.key or type(record.tombstone) is not ReferenceType:
                raise ValueError("untrained inbox key/weak tombstone changed")
            if contents:
                untrained_inbox_origin_stamp(record, self._content_limits)
        return tuple(self._untrained_records.values())

    def _verify_untrained(self, owner: Any, runtime: Any) -> None:
        records = self._untrained_state()
        inbox = runtime._inbox
        if self._untrained_records.keys() != inbox._erased.keys() - inbox._applied.keys():
            raise ValueError("untrained tombstone lacks original observed erasure lineage")
        for record in records:
            key = record.data.key
            if (
                inbox._erased.get(key) is not record.tombstone()
                or key in inbox._applied
                or key in inbox._experiences
                or key in inbox._labels
                or key not in owner._revoked_keys
                or record.data.learner_version != runtime._candidate_version
                or record.data.completed_updates > runtime._budget.updates_completed
            ):
                raise ValueError("untrained original tombstone/work identities changed")

    def _verify_erased(self, owner: Any, runtime: Any) -> None:
        records = self._erased_state()
        inbox = runtime._inbox
        expected = inbox._erased.keys() & inbox._applied.keys()
        if self._erased_records.keys() != expected:
            raise ValueError("erased inbox receipt lacks original observed erasure lineage")
        for record in records:
            key = record.data.key
            if (
                inbox._applied.get(key) is not record.receipt()
                or inbox._erased.get(key) is not record.tombstone()
                or key in inbox._experiences
                or key in inbox._labels
                or key not in owner._revoked_keys
                or record.data.learner_version != runtime._candidate_version
                or record.data.update_number > runtime._budget.updates_completed
            ):
                raise ValueError("erased inbox original receipt/tombstone/work identities changed")

    def require_erasure_ready(self, ledger: Any, owner: Any, runtime: Any, now: int) -> tuple:
        """Separate actual untrained arrivals from verified applied pairs."""
        from src.app.untrained_inbox_erasure import observe_untrained_arrivals

        records = self.verify(ledger, owner, runtime, now)
        keys = {record.data.key for record in records}
        before = self.state_stamp()
        arrivals = observe_untrained_arrivals(self, ledger, owner, runtime, keys, now)
        if self.state_stamp() != before:
            raise ValueError("original history changed during untrained arrival qualification")
        return before, arrivals

    def prepare_erasure(
        self, ledger: Any, owner: Any, runtime: Any, keys: tuple, prepared: dict, before: tuple
    ) -> Any:
        """Prove and pay before removal; publish only prebuilt weak metadata afterwards."""
        self._authority(ledger)
        records = self._state()
        self._maps(runtime)
        if type(before) is not tuple or len(before) != 2 or self.state_stamp() != before[0]:
            raise ValueError("original inbox history changed during native erasure")
        arrivals = before[1]
        from src.app.untrained_inbox_erasure import (
            _Arrival,
            _require_arrival,
            prepare_untrained_erasure,
        )

        if (
            type(arrivals) is not tuple
            or len(arrivals) > self._capacity
            or any(type(arrival) is not _Arrival for arrival in arrivals)
        ):
            raise ValueError("original untrained arrival inventory changed")
        for arrival in arrivals:
            _require_arrival(self, owner, runtime, arrival)
        untrained_keys = {arrival.key for arrival in arrivals}
        live_by_key = {record.data.key: record for record in records}
        if (
            type(keys) is not tuple
            or not 0 < len(keys) <= self._capacity
            or type(prepared) is not dict
            or len(prepared) > self._inbox_capacity
        ):
            raise ValueError("original inbox erasure requires exact commit containers")
        for key in keys:
            if (
                type(key) is not tuple
                or len(key) != 2
                or any(type(part) is not str for part in key)
            ):
                raise ValueError("original erasure keys require exact paired strings")
            checkpoint_content_stamp(key, self._content_limits)
        for key, tombstone in prepared.items():
            if (
                type(key) is not tuple
                or len(key) != 2
                or any(type(part) is not str for part in key)
            ):
                raise ValueError("prepared erasure keys require exact paired strings")
            checkpoint_content_stamp(key, self._content_limits)
            _schema(tombstone, ErasedExperience)
        if (
            type(keys) is not tuple
            or not keys
            or len(keys) > self._capacity
            or type(prepared) is not dict
            or len(prepared) > self._inbox_capacity
            or set(keys) != live_by_key.keys() | untrained_keys
            or live_by_key.keys() & untrained_keys
            or len(set(keys)) != len(keys)
            or prepared.keys() != runtime._inbox._erased.keys() | set(keys)
            or any(prepared[key] is not value for key, value in runtime._inbox._erased.items())
        ):
            raise ValueError("original inbox erasure commit keys/history changed")
        updates: list[tuple[int, tuple[str, str], ErasedInboxOrigin]] = []
        for key in keys:
            if key in untrained_keys:
                continue
            record = live_by_key[key]
            if (
                not record.verified
                or record.receipt is None
                or runtime._inbox._experiences.get(key) is not record.source()
                or runtime._inbox._labels.get(key) is not record.label()
                or runtime._inbox._applied.get(key) is not record.receipt()
                or owner._catalog.get(key) is not record.declaration()
                or record.data.learner_version != runtime._candidate_version
                or record.data.update_number > runtime._budget.updates_completed
                or key not in owner._revoked_keys
            ):
                raise ValueError("inbox erasure lacks original committed source/receipt/revocation")
            # The original record reservation remains spent. Charge the new
            # erased witness before allocation, including conservative peak live.
            ledger._admission.reserve_inbox(record.data, ledger._live_records() + len(updates))
            erased = prepare_erased_inbox_origin(
                record.data,
                cast(AppliedExperience, record.receipt()),
                prepared[key],
                self._content_limits,
            )
            updates.append((id(record.source()), key, erased))

        live, live_sealed = self._records.copy(), self._sealed.copy()
        erased_records, erased_sealed = self._erased_records.copy(), self._erased_sealed.copy()
        for identity, key, erased in updates:
            del live[identity]
            del live_sealed[identity]
            erased_records[key] = erased_sealed[key] = erased
        untrained = prepare_untrained_erasure(
            self, ledger, owner, runtime, arrivals, prepared, len(updates)
        )
        # Allocate complete storage before payload removal; publication replaces
        # only prepared slot values, matching the existing checkpoint handoff.
        return (live, live_sealed), (erased_records, erased_sealed), untrained

    def _maps(self, runtime: Any) -> None:
        if (
            type(self._inbox_capacity) is not int
            or not 0 < self._inbox_capacity < 2**63
            or type(runtime._inbox._capacity) is not int
            or runtime._inbox._capacity != self._inbox_capacity
        ):
            raise ValueError("inbox origin original inbox capacity changed")
        for mapping, kind in (
            (runtime._inbox._experiences, Experience),
            (runtime._inbox._labels, LabelArrival),
            (runtime._inbox._applied, AppliedExperience),
            (runtime._inbox._erased, ErasedExperience),
        ):
            if type(mapping) is not dict or len(mapping) > self._capacity:
                raise ValueError("inbox origin current payload/history map is unsupported")
            for key, value in mapping.items():
                if (
                    type(key) is not tuple
                    or len(key) != 2
                    or any(type(part) is not str for part in key)
                ):
                    raise ValueError("inbox origin current key requires exact paired strings")
                checkpoint_content_stamp(key, self._content_limits)
                _schema(value, kind)

    def _origin(self, ledger: Any, runtime: Any, origin: Any) -> None:
        _schema(origin, NativeUpdateOrigin)
        if type(origin.completed_updates) is not int or not 0 <= origin.completed_updates < 2**63:
            raise ValueError("inbox origin work requires a bounded exact counter")
        if type(origin.learner_version) is not str:
            raise ValueError("inbox origin learner version requires an exact string")
        if (
            type(ledger._version) is not str
            or type(runtime._budget.updates_completed) is not int
            or not 0 <= runtime._budget.updates_completed < 2**63
        ):
            raise ValueError("inbox origin original producer/work metadata changed")
        if (
            origin.learner is not runtime._candidate
            or origin.learner_version != ledger._version
            or origin.completed_updates != runtime._budget.updates_completed
        ):
            raise ValueError("inbox origin original native producer/work differs")

    def started(self, ledger: Any, owner: Any, runtime: Any, origin: Any, declaration: Any) -> None:
        self._authority(ledger)
        self._state()
        self._maps(runtime)
        self._origin(ledger, runtime, origin)
        source_stamp = _source_stamp(origin.source, origin.label, declaration, self._content_limits)
        if (
            origin.receipt is not None
            or id(origin.source) in self._records
            or origin.source.key in runtime._inbox._applied
        ):
            raise ValueError("inbox history cannot retrospectively enroll original updates")
        if (
            ledger._source(owner, runtime, origin.source, origin.label, ledger._read_tick(runtime))
            is not declaration
        ):
            raise ValueError("inbox origin declaration differs from original eligible source")
        size, feature_digest, target_digest = inbox_payload_binding(
            origin.source.features,
            origin.label.targets,
            origin.features,
            origin.targets,
            self._content_limits,
        )
        if (
            _source_stamp(origin.source, origin.label, declaration, self._content_limits)
            != source_stamp
        ):
            raise ValueError("inbox source contents changed during original qualification")
        data = InboxOriginData(
            origin.source.key,
            origin.label.event_id,
            origin.source.model_version,
            origin.learner_version,
            declaration.provenance.subject_id,
            declaration.provenance.source_id,
            origin.source.observed_at,
            origin.label.arrived_at,
            origin.completed_updates + 1,
            size,
            feature_digest,
            target_digest,
        )
        ledger._admission.reserve_inbox(data, ledger._live_records())
        # Why: charges precede weak publication and remain spent on later failure.
        record = _InboxOrigin(
            data,
            _weak(origin.source),
            _weak(origin.label),
            _weak(declaration),
            _weak(origin.source.features),
            _weak(origin.label.targets),
            None,
            source_stamp,
            "",
            False,
        )
        if len(self._records) >= self._capacity:
            raise ValueError("inbox origin original history capacity exhausted")
        self._records[id(origin.source)] = self._sealed[id(origin.source)] = record

    def completed(self, ledger: Any, owner: Any, runtime: Any, origin: Any) -> None:
        self._authority(ledger)
        self._state()
        self._maps(runtime)
        self._origin(ledger, runtime, origin)
        _schema(origin.source, Experience)
        record = self._records.get(id(origin.source))
        if (
            record is None
            or record.verified
            or record.receipt is not None
            or record.source() is not origin.source
            or record.label() is not origin.label
            or record.data.update_number != origin.completed_updates
        ):
            raise ValueError("inbox origin completion lacks its original provisional witness")
        receipt = origin.receipt
        stamp = _receipt_stamp(receipt, self._content_limits)
        if runtime._inbox._applied.get(record.data.key) is not receipt:
            raise ValueError("inbox origin completion lacks actual inserted receipt")
        updated = replace(record, receipt=_weak(receipt), receipt_stamp=stamp)
        inbox_origin_stamp(updated, self._content_limits)
        self._records[id(origin.source)] = self._sealed[id(origin.source)] = updated

    def finish(self, ledger: Any, owner: Any, runtime: Any, receipts: Any) -> None:
        self._authority(ledger)
        records = self._state()
        before = self.state_stamp()
        self._maps(runtime)
        if type(receipts) is not tuple or len(receipts) > self._capacity:
            raise ValueError("inbox origin successful poll requires bounded exact receipts")
        for receipt in receipts:
            _receipt_values(receipt, self._content_limits)
        prepared = []
        for record in records:
            if record.verified or record.receipt is None:
                continue
            receipt = ledger._receipt(runtime, record)
            if not any(receipt is item for item in receipts):
                raise ValueError(
                    "inbox origin completed receipt is outside successful original poll"
                )
            declaration = ledger._source(
                owner, runtime, record.source(), record.label(), ledger._read_tick(runtime)
            )
            if declaration is not record.declaration():
                raise ValueError("inbox origin successful poll declaration changed")
            inbox_origin_stamp(record, self._content_limits)
            prepared.append((id(record.source()), replace(record, verified=True)))
        if self.state_stamp() != before:
            raise ValueError("inbox origin history changed during successful final qualification")
        if len(prepared) != len(receipts):
            raise ValueError(
                "successful poll receipts differ from newly completed original history"
            )
        for key, record in prepared:
            self._records[key] = self._sealed[key] = record

    def verify(
        self, ledger: Any, owner: Any, runtime: Any, now: int, *, lifecycle_leased: bool = False
    ) -> tuple[_InboxOrigin, ...]:
        self._authority(ledger)
        if type(now) is not int or not 0 <= now < 2**63 or type(lifecycle_leased) is not bool:
            raise ValueError("inbox origin verification requires exact bounded time and lease flag")
        self._state(contents=False)
        self._maps(runtime)
        self._verify_erased(owner, runtime)
        self._verify_untrained(owner, runtime)
        # Refuse foreign current tombstones before traversing borrowed live arrays.
        records = self._state()
        before = self.state_stamp()
        for record in records:
            if not record.verified:
                raise ValueError("inbox origin original pair remains provisional")
            declaration = ledger._source(
                owner,
                runtime,
                record.source(),
                record.label(),
                now,
                lifecycle_leased=lifecycle_leased,
            )
            if declaration is not record.declaration():
                raise ValueError("inbox origin original declaration reference changed")
            ledger._receipt(runtime, record)
            inbox_origin_stamp(record, self._content_limits)
        self._authority(ledger)
        if self.state_stamp() != before:
            raise ValueError("inbox origin history changed during original source qualification")
        # Opaque original liveness ports may alter current inbox membership while
        # leaving the witness-held scalar objects intact. Reprove both partitions.
        self._maps(runtime)
        self._verify_erased(owner, runtime)
        self._verify_untrained(owner, runtime)
        return records

    def prune(self, runtime: Any) -> None:
        if type(self._ledger) is not ReferenceType:
            raise ValueError("inbox origin ledger reference changed")
        self._authority(self._ledger())
        records = self._state(contents=False)
        self._maps(runtime)
        for record in records:
            key = record.data.key
            if (
                key in runtime._inbox._erased
                and key not in runtime._inbox._experiences
                and key not in runtime._inbox._labels
            ):
                tombstone = runtime._inbox._erased[key]
                checkpoint_content_stamp(tombstone, self._content_limits)
                ErasedExperience.__post_init__(tombstone)
                if (
                    tombstone.key != key
                    or tombstone.actor_version != record.data.actor_version
                    or tombstone.observed_at != record.data.observed_at
                    or tombstone.event_id != record.data.event_id
                    or tombstone.arrived_at != record.data.arrived_at
                ):
                    raise ValueError("inbox history tombstone differs from original pair")
                # Only the actual observed commit removes a live lineage record.
                # An unobserved tombstone cannot retrospectively create proof or
                # silently discard the missing erased receipt's original charge.
                raise ValueError("erased inbox receipt lacks original observed erasure lineage")

    def prepare_erased_handoff(self, records: tuple[ErasedInboxOrigin, ...]) -> tuple:
        """Prepare copied weak metadata maps before trusted plain publication."""
        if type(self._ledger) is not ReferenceType:
            raise ValueError("erased inbox original ledger reference changed")
        self._authority(self._ledger())
        self._state()
        originals = self._erased_state()
        require_erased_inbox_records(records, self._content_limits, self._capacity)
        if (
            len(records) != len(originals)
            or {record.data.key for record in records} != self._erased_records.keys()
        ):
            raise ValueError("erased inbox handoff does not cover original admitted history")
        result = {}
        for record in records:
            original = self._erased_records[record.data.key]
            require_same_erased_inbox_history(original, record, self._content_limits)
            result[record.data.key] = record
        # Actual memo binding and its copy reservations already qualified these
        # witnesses. Map publication allocates no new record or allowance.
        return result, dict(result)

    def prepare_untrained_handoff(self, records: tuple[UntrainedInboxOrigin, ...]) -> tuple:
        """Prebuild complete copied weak maps without a new grant or callback."""
        if type(self._ledger) is not ReferenceType:
            raise ValueError("untrained inbox original ledger reference changed")
        self._authority(self._ledger())
        self._state()
        originals = self._untrained_state()
        require_untrained_inbox_records(records, self._content_limits, self._capacity)
        if (
            len(records) != len(originals)
            or {record.data.key for record in records} != self._untrained_records.keys()
        ):
            raise ValueError("untrained inbox handoff does not cover original admitted history")
        result = {}
        for record in records:
            original = self._untrained_records[record.data.key]
            require_same_untrained_inbox_history(original, record, self._content_limits)
            result[record.data.key] = record
        # Actual memo binding and permanent reservations precede publication.
        return result, dict(result)

    def rebind(self, record: _InboxOrigin, source: Any, label: Any, receipt: Any) -> _InboxOrigin:
        if type(self._ledger) is not ReferenceType:
            raise ValueError("inbox origin ledger reference changed")
        self._authority(self._ledger())
        self._state()
        inbox_origin_stamp(record, self._content_limits)
        if not record.verified or self._records.get(id(record.source())) is not record:
            raise ValueError("inbox copy binding requires original verified enrolled history")
        _source_stamp(source, label, record.declaration(), self._content_limits)
        original_receipt = record.receipt() if record.receipt is not None else None
        if (
            source is record.source()
            or label is record.label()
            or receipt is original_receipt
            or source.features is record.features()
            or label.targets is record.targets()
        ):
            raise ValueError("inbox copy binding requires actual copied records and arrays")
        if _receipt_values(receipt, self._content_limits) != _receipt_values(
            original_receipt, self._content_limits
        ):
            raise ValueError("inbox copied receipt scalar history changed")
        rebound = replace(
            record,
            source=_weak(source),
            label=_weak(label),
            features=_weak(source.features),
            targets=_weak(label.targets),
            receipt=_weak(receipt),
            source_stamp=_source_stamp(source, label, record.declaration(), self._content_limits),
            receipt_stamp=_receipt_stamp(receipt, self._content_limits),
        )
        inbox_origin_stamp(rebound, self._content_limits)
        return rebound
