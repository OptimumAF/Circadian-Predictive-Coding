"""Audit committed trained inbox history before erasure, including after TTL.

Inputs are original birth-enrolled app objects and their current logical tick.
Output is the existing payload-free state stamp and an empty arrival tuple.
The caller must own the original leases and separately qualify lifecycle expiry,
native ports, policy/birth scalars and publication. This predicate grants no raw
access, cleanup, admission, copy or history transition. Untrained arrivals need
their own cleanup proof; they are explicitly unsupported here.
"""

from typing import Any, cast
from weakref import ReferenceType

import numpy as np

from src.app.actor_shadow import ActorShadowRuntime
from src.app.erased_inbox_origins import ErasedInboxOrigin, _counter
from src.app.experience_inbox import ExperienceInbox
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_experience import ManagedExperienceOwner
from src.app.managed_inbox_origins import (
    ManagedInboxOrigins,
    _InboxOrigin,
    _declaration_values,
    _digest,
    _schema,
    inbox_origin_stamp,
)
from src.app.managed_replay_origins import ManagedReplayOrigins
from src.app.resource_sharing import ResourceSharedRuntime
from src.app.toy_execution_budget import ToyBudgetSession
from src.app.untrained_inbox_origins import UntrainedInboxOrigin
from src.core.checkpoint_content import checkpoint_content_stamp
from src.core.data_lifecycle import (
    DataConsent,
    DataProvenance,
    LifecycleDeclaration,
    LifecycleLimits,
)
from src.core.experience import (
    AppliedExperience,
    Experience,
    ExperiencePermissions,
    LabelArrival,
    LogicalClock,
)
from src.core.learner_ports import TrainingDiagnostic
from src.core.inbox_origin import InboxOriginData
from src.core.replay_origin import ReplayOriginAccounting, ReplayOriginAdmission
from src.core.untrained_inbox_origin import UntrainedInboxOriginData


def _require_original_bindings(
    history: ManagedInboxOrigins,
    ledger: ManagedReplayOrigins,
    owner: ManagedExperienceOwner,
    runtime: ActorShadowRuntime,
) -> ManagedDataLifecycle:
    for value, kind in (
        (history, ManagedInboxOrigins),
        (ledger, ManagedReplayOrigins),
        (owner, ManagedExperienceOwner),
        (runtime, ActorShadowRuntime),
    ):
        if type(value) is not kind:
            raise ValueError("erasure audit requires exact supported original app objects")
    if (
        type(owner._shared) is not ResourceSharedRuntime
        or type(runtime._inbox) is not ExperienceInbox
        or type(runtime._budget) is not ToyBudgetSession
    ):
        raise ValueError("erasure audit requires exact original shared runtime/inbox/budget")
    if (
        type(ledger._owner) is not ReferenceType
        or ledger._owner() is not owner
        or owner._shared._runtime is not runtime
        or ManagedReplayOrigins._history(ledger) is not history
    ):
        raise ValueError("erasure audit requires original owner/runtime/history birth")
    ManagedInboxOrigins._authority(history, ledger)
    expected = dict(
        runtime=runtime,
        actor=runtime._actor,
        learner=runtime._candidate,
        inbox=runtime._inbox,
        budget=runtime._budget,
        clock=runtime._inbox._clock,
        budget_policy=runtime._budget.budget,
        owner_policy=owner._limits,
    )
    anchors = ledger._anchors
    if (
        type(anchors) is not dict
        or len(anchors) != len(expected) + 1
        or any(type(key) is not str or len(key) > 64 for key in anchors)
        or anchors.keys() != expected.keys() | {"model"}
        or any(type(value) is not ReferenceType for value in anchors.values())
        or any(anchors[name]() is not value for name, value in expected.items())
        or runtime._payload_lineage is not ledger._lineage
        or type(runtime._candidate_version) is not str
        or type(runtime._base_actor_version) is not str
        or type(ledger._version) is not str
        or runtime._candidate_version != ledger._version
    ):
        raise ValueError("erasure audit original structural bindings changed")
    life: ManagedDataLifecycle = owner._lifecycle
    if (
        type(life) is not ManagedDataLifecycle
        or type(ledger._lifecycle) is not ReferenceType
        or ledger._lifecycle() is not life
        or life._owner is not owner
        or life._clock is not runtime._inbox._clock
        or life._budget is not runtime._budget
        or life._lineage is not runtime._payload_lineage
    ):
        raise ValueError("erasure audit original lifecycle binding changed")
    return life


def _require_ready_time(ledger: Any, runtime: Any, life: Any, now: int) -> None:
    _counter(now)
    clock = runtime._inbox._clock
    if type(clock) is not LogicalClock:
        raise ValueError("erasure audit requires original exact logical clock")
    for tick in (clock._time, ledger._last_tick, runtime._inbox._last_time, life._last_tick):
        _counter(tick)
    # Why: read the supported scalar directly, without dispatching clock ports or
    # latching a newer time. Auditing expired data does not renew live access.
    if now != clock._time or now < max(
        ledger._last_tick, runtime._inbox._last_time, life._last_tick
    ):
        raise ValueError("erasure audit requires unchanged current monotone logical time")
    for flag in (
        ledger._busy,
        ledger._poisoned,
        runtime._stopped,
        runtime._retired,
        runtime._inbox._stopped,
        life._failed,
        life._retention_fault,
    ):
        if type(flag) is not bool or flag:
            raise ValueError("erasure audit refuses ambiguous stopped/pending/failed work")
    # _started stays true after successful training; _busy and active weak
    # bindings, rather than that historical flag, identify unfinished work.
    if type(ledger._started) is not bool:
        raise ValueError("erasure audit original started-work flag changed")
    if (
        ledger._pending is not None
        or ledger._active_source is not None
        or ledger._active_label is not None
    ):
        raise ValueError("erasure audit refuses pending work")
    _counter(ledger._last_work)
    _counter(runtime._budget.updates_completed)
    if runtime._budget.updates_completed < ledger._last_work:
        raise ValueError("erasure audit original completed work was rewound")


def _require_owner_metadata(owner: Any, history: Any) -> None:
    _schema(owner._limits, LifecycleLimits)
    LifecycleLimits.__post_init__(owner._limits)
    if type(owner._catalog) is not dict or len(owner._catalog) > 4096:
        raise ValueError("erasure audit requires bounded original declaration map")
    for key in owner._catalog:
        _require_key(key, history._content_limits.max_metadata_bytes)
    if type(owner._revoked_keys) is not set or len(owner._revoked_keys) > 4096:
        raise ValueError("erasure audit requires bounded exact revocation keys")
    for key in owner._revoked_keys:
        _require_key(key, history._content_limits.max_metadata_bytes)
    if type(owner._opted_out) is not set or len(owner._opted_out) > 4096:
        raise ValueError("erasure audit requires bounded exact opted-out subjects")
    if any(
        type(subject) is not str
        or not 0 < len(subject) <= history._content_limits.max_metadata_bytes
        for subject in owner._opted_out
    ):
        raise ValueError("erasure audit opted-out subjects require bounded exact strings")


def _require_key(key: Any, maximum: int) -> None:
    if (
        type(key) is not tuple
        or len(key) != 2
        or any(type(part) is not str or not 0 < len(part) <= maximum for part in key)
    ):
        raise ValueError("erasure audit requires bounded exact declaration/revocation keys")


def _require_admission(history: Any, ledger: Any) -> None:
    admission: ReplayOriginAdmission = ledger._admission
    if (
        type(ledger._original_admission) is not ReferenceType
        or ledger._original_admission() is not admission
        or type(admission._minimum) is not tuple
        or len(admission._minimum) != 3
    ):
        raise ValueError("erasure audit original admission or charge minima changed")
    for value in admission._minimum:
        _counter(value)
    _schema(admission._progress, ReplayOriginAccounting)
    if type(ledger._rows) is not dict or len(ledger._rows) > history._capacity:
        raise ValueError("erasure audit requires bounded original row map")
    _counter(ledger._copy_slots)
    _counter(ledger._minimum_copy_slots)
    if not ledger._minimum_copy_slots <= ledger._copy_slots <= admission.limits.max_live_records:
        raise ValueError("erasure audit original copy reservations were rewound")
    # The existing ledger counter dynamically calls self._history(). This pure
    # predicate uses fixed history methods, so a replaced instance port cannot run.
    live = (
        len(ledger._rows)
        + ledger._copy_slots
        + len(ManagedInboxOrigins._state(history, contents=False))
        + len(ManagedInboxOrigins._erased_state(history, contents=False))
        + len(ManagedInboxOrigins._untrained_state(history, contents=False))
    )
    ReplayOriginAdmission.accounting(admission, live)


def _require_complete_partitions(
    history: Any, ledger: Any, owner: Any, runtime: Any, records: tuple
) -> None:
    ManagedInboxOrigins._maps(history, runtime)
    ManagedInboxOrigins._verify_erased(history, owner, runtime)
    ManagedInboxOrigins._verify_untrained(history, owner, runtime)
    keys = {record.data.key for record in records}
    inbox = runtime._inbox
    if len(keys) != len(records) or inbox._applied.keys() != keys | history._erased_records.keys():
        raise ValueError("erasure audit requires complete original committed trained lineage")
    if inbox._experiences.keys() != keys or inbox._labels.keys() != keys:
        raise ValueError("erasure audit current untrained arrivals are not yet supported")
    if runtime._budget.updates_completed != ledger._last_work:
        raise ValueError("erasure audit completed work lacks original observed completion")


def _require_payload_bound(
    history: Any, owner: Any, runtime: Any, records: tuple, borrowed: tuple = ()
) -> None:
    total = 0
    metadata = [
        (tuple(owner._catalog), tuple(owner._revoked_keys), tuple(owner._opted_out)),
        tuple(
            tuple(mapping)
            for mapping in (
                runtime._inbox._experiences,
                runtime._inbox._labels,
                runtime._inbox._applied,
                runtime._inbox._erased,
            )
        ),
    ]
    for record in records:
        source, label, declaration = record.source(), record.label(), record.declaration()
        receipt = None if record.receipt is None else record.receipt()
        for value, kind in (
            (source, Experience),
            (label, LabelArrival),
            (declaration, LifecycleDeclaration),
            (receipt, AppliedExperience),
        ):
            _schema(value, kind)
        for array in (source.features, label.targets):
            if type(array) is not np.ndarray:
                raise ValueError("erasure audit requires exact numeric borrowed arrays")
            total += array.nbytes
            if total > history._content_limits.max_array_bytes:
                raise ValueError("erasure audit aggregate borrowed payload exceeds original bound")
        metadata.extend(
            (
                _scalar_fields(source, history, ("features",)),
                _scalar_fields(label, history, ("targets",)),
                _scalar_fields(declaration, history),
                _scalar_fields(receipt, history),
                _scalar_fields(record.data, history),
                (record.source_stamp, record.receipt_stamp),
            )
        )
    for record in (*history._erased_records.values(), *history._untrained_records.values()):
        metadata.append(_scalar_fields(record.data, history))
    # Why: a mixed cleanup must bound every current borrowed arrival together
    # with trained records before either path traverses any numeric contents.
    if type(borrowed) is not tuple or len(borrowed) > history._capacity:
        raise ValueError("erasure audit requires bounded borrowed arrival metadata")
    for row in borrowed:
        if type(row) is not tuple or len(row) != 3:
            raise ValueError("erasure audit requires exact borrowed arrival triples")
        source, label, declaration = row
        _schema(declaration, LifecycleDeclaration)
        metadata.append(_scalar_fields(declaration, history))
        for value, kind, field in (
            (source, Experience, "features"),
            (label, LabelArrival, "targets"),
        ):
            if value is None:
                continue
            _schema(value, kind)
            array = getattr(value, field)
            if type(array) is not np.ndarray:
                raise ValueError("erasure audit requires exact numeric borrowed arrays")
            total += array.nbytes
            if total > history._content_limits.max_array_bytes:
                raise ValueError("erasure audit aggregate borrowed payload exceeds original bound")
            metadata.append(_scalar_fields(value, history, (field,)))
    # No arrays occur in this graph. Complete metadata/node/depth bounds pass
    # before the later original content stamps traverse borrowed numeric buffers.
    checkpoint_content_stamp(tuple(metadata), history._content_limits)
    for record in records:
        if (
            runtime._inbox._experiences.get(record.data.key) is not record.source()
            or runtime._inbox._labels.get(record.data.key) is not record.label()
        ):
            raise ValueError("erasure audit original current source/label identity changed")


def _preflight_records(history: Any, runtime: Any) -> tuple:
    """Validate fixed record/map headers without traversing content or hashes."""
    for mapping, kind, data_kind in (
        (history._records, _InboxOrigin, InboxOriginData),
        (history._erased_records, ErasedInboxOrigin, InboxOriginData),
        (history._untrained_records, UntrainedInboxOrigin, UntrainedInboxOriginData),
    ):
        if type(mapping) is not dict or len(mapping) > history._capacity:
            raise ValueError("erasure audit requires bounded exact original history maps")
        for key, record in mapping.items():
            if kind is _InboxOrigin:
                if type(key) is not int or not 0 < key < 2**64:
                    raise ValueError("erasure audit requires exact original live identity keys")
                _schema(record, _InboxOrigin)
                for reference in (
                    record.source,
                    record.label,
                    record.declaration,
                    record.features,
                    record.targets,
                    record.receipt,
                ):
                    if type(reference) is not ReferenceType:
                        raise ValueError(
                            "erasure audit requires original committed weak references"
                        )
                _digest(record.source_stamp)
                _digest(record.receipt_stamp)
            else:
                _require_key(key, history._content_limits.max_metadata_bytes)
                if type(record) is not kind:
                    raise ValueError("erasure audit requires exact original erased witnesses")
            _schema(record.data, data_kind)
            _require_key(record.data.key, history._content_limits.max_metadata_bytes)
    for mapping in (
        runtime._inbox._experiences,
        runtime._inbox._labels,
        runtime._inbox._applied,
        runtime._inbox._erased,
    ):
        if type(mapping) is not dict or len(mapping) > history._capacity:
            raise ValueError("erasure audit requires bounded exact current inbox maps")
        for key in mapping:
            _require_key(key, history._content_limits.max_metadata_bytes)
    return tuple(history._records.values())


def _preflight_history(history: Any, owner: Any, runtime: Any, borrowed: tuple = ()) -> tuple:
    """Validate cheap fixed schemas and aggregate bounds before delegated hashes."""
    records = _preflight_records(history, runtime)
    _require_payload_bound(history, owner, runtime, records, borrowed)
    return records


def _scalar_fields(value: Any, history: Any, excluded: tuple = ()) -> tuple:
    result = []
    nested = (ExperiencePermissions, TrainingDiagnostic, DataConsent, DataProvenance)
    for name, field in vars(value).items():
        if name in excluded:
            continue
        if any(type(field) is kind for kind in nested):
            _schema(field, type(field))
            field = tuple(vars(field).values())
        if type(field) is tuple:
            if len(field) > history._content_limits.max_nodes:
                raise ValueError("erasure audit metadata tuple exceeds original node bound")
            values = field
        else:
            values = (field,)
        if any(
            item is not None and not any(type(item) is kind for kind in (bool, int, float, str))
            for item in values
        ):
            raise ValueError("erasure audit requires bounded exact scalar metadata")
        result.append((name, field))
    return tuple(result)


def _require_committed_record(
    history: Any,
    owner: Any,
    runtime: Any,
    record: _InboxOrigin,
    now: int,
    revoked: frozenset = frozenset(),
) -> None:
    if type(revoked) is not frozenset or len(revoked) > history._capacity:
        raise ValueError("erasure audit requires bounded exact expected cleanup revocation")
    for key in revoked:
        _require_key(key, history._content_limits.max_metadata_bytes)
    inbox_origin_stamp(record, history._content_limits)
    if record.verified is not True:
        raise ValueError("erasure audit original pair remains provisional")
    source, label, declaration = (
        cast(Experience, record.source()),
        cast(LabelArrival, record.label()),
        cast(LifecycleDeclaration, record.declaration()),
    )
    _declaration_values(declaration, history._content_limits)
    _schema(source.permissions, ExperiencePermissions)
    ExperiencePermissions.__post_init__(source.permissions)
    LifecycleLimits.require_supported(owner._limits, declaration)
    receipt = ManagedReplayOrigins._receipt(runtime, record)
    if (
        owner._catalog.get(record.data.key) is not declaration
        or (record.data.key in owner._revoked_keys and record.data.key not in revoked)
        or declaration.provenance.subject_id in owner._opted_out
        or type(source.role) is not str
        or type(label.role) is not str
        or source.role != "train"
        or label.role != "train"
        or not source.permissions.training
        or not source.permissions.replay
        or source.model_version != runtime._base_actor_version
        or record.data.learner_version != runtime._candidate_version
        or not record.data.observed_at <= record.data.arrived_at <= receipt.applied_at <= now
    ):
        raise ValueError("erasure audit original consent/receipt/chronology changed")


def require_committed_erasure_ready(
    history: Any, ledger: Any, owner: Any, runtime: Any, now: int
) -> tuple:
    """Return an audit stamp, never permission to use an expired raw payload.

    Source age limits guard live use. An unchanged originally committed receipt
    can instead be audited for erasure after that age, without a new consent grant.
    """
    # Validate outer exact types before dereferencing a possible foreign runtime.
    if type(runtime) is not ActorShadowRuntime:
        raise ValueError("erasure audit requires an exact original runtime")
    life = _require_original_bindings(history, ledger, owner, runtime)
    _require_ready_time(ledger, runtime, life, now)
    _require_owner_metadata(owner, history)
    records = _preflight_history(history, owner, runtime)
    ManagedInboxOrigins._state(history, contents=False)
    _require_admission(history, ledger)
    _require_complete_partitions(history, ledger, owner, runtime, records)
    before = ManagedInboxOrigins.state_stamp(history)
    for record in records:
        _require_committed_record(history, owner, runtime, record, now)
    _require_original_bindings(history, ledger, owner, runtime)
    _require_ready_time(ledger, runtime, life, now)
    _require_owner_metadata(owner, history)
    _require_admission(history, ledger)
    _require_complete_partitions(history, ledger, owner, runtime, records)
    if ManagedInboxOrigins.state_stamp(history) != before:
        raise ValueError("erasure audit history changed during original qualification")
    return before, ()
