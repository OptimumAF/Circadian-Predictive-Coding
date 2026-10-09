"""Original checkpoint graph admission and weak replay publication witnesses.

Inputs are the original row ledger, bounded graph ports and original builder
source port. Capture returns the controller's actual token; restore returns its
published runtime. The controller alone owns native preparation and publication.
No payload ownership, renewed allowance, persistence or provenance inference.
Inbox pairs require original retained rows or history enrolled before training.
"""

from collections import deque
from _thread import LockType
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, fields
from math import isfinite
from threading import Lock
from typing import Any, Callable, cast
from weakref import ReferenceType, WeakMethod

from src.app.candidate_checkpoint import (
    CandidateCheckpoint,
    CandidateCheckpointView,
    CandidateCheckpointController,
    _Pending,
)
from src.app.checkpoint_replay_handoff import _ReplayTransition, lease_replay_handoff
from src.app.managed_replay_copies import ManagedReplayCopies
from src.app.managed_replay_origins import ManagedReplayOrigins, _Row, _bound, _weak
from src.app.managed_inbox_origins import ManagedInboxOrigins, _InboxOrigin, inbox_origin_stamp
from src.app.erased_inbox_origins import ErasedInboxOrigin, erased_inbox_origin_stamp
from src.app.untrained_inbox_origins import UntrainedInboxOrigin, untrained_inbox_origin_stamp
from src.app.checkpoint_untrained_inbox import (
    bind_untrained_inbox_records,
    require_untrained_inbox_records,
    require_same_untrained_inbox_history,
)
from src.app.checkpoint_erased_inbox import (
    bind_erased_inbox_records,
    require_erased_inbox_records,
    require_inbox_partition,
    require_same_erased_inbox_history,
)
from src.app.managed_experience import ManagedExperienceOwner
from src.app.actor_shadow import ActorShadowRuntime, StableActor
from src.app.experience_inbox import ExperienceInbox
from src.app.serving_promotion import PromotableActor, _Slot, _Bundle
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.payload_ownership import (
    PayloadOwnershipBusy,
    PayloadOwnershipRegistry,
    lease_payload_lock,
)
from src.app.toy_execution_budget import (
    CircadianResumePosition,
    ToyExecutionBudget,
    ToyExecutionProgress,
    ToyBudgetSession,
)
from src.core.experience import (
    AppliedExperience,
    Experience,
    LabelArrival,
    LogicalClock,
    require_tick,
)
from src.core.data_erasure import ErasedExperience
from src.core.data_lifecycle import (
    LifecycleLimits,
    LifecycleDeclaration,
    DataConsent,
    DataProvenance,
)
from src.core.data_retention import DataRetentionPolicy
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.payload_bytes import PayloadCopyLimits
from src.core.replay_origin import ReplayOriginLimits, ReplayOriginData
from src.core.replay_write_origin import ReplayWriteLimits
from src.core.resource_sharing import SharingLimits, SharingSnapshot
from src.core.actor_ports import AppliedConsolidation
from src.core.learner_ports import TrainingDiagnostic
from src.core.checkpoint_content import CheckpointContentLimits, checkpoint_content_stamp
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork, ReplaySnapshot
from src.core.inbox_cursor import InboxCursor
from src.core.native_graph_copy import observe_graph_copies
from src.core.native_model_copy import ModelCopyLimits
from src.core.replay_graph_origin import ReplayGraphPorts
from src.shared.process_memory import ProcessRssSampler, ProcessRssSegment


@dataclass(frozen=True)
class _InboxPair:
    source: ReferenceType[Any]
    label: ReferenceType[Any]
    receipt: ReferenceType[Any]
    features: ReferenceType[Any]
    targets: ReferenceType[Any]
    origin: _Row | _InboxOrigin


@dataclass(frozen=True)
class _Checkpoint:
    token: ReferenceType[Any]
    controller: ReferenceType[Any]
    pending: ReferenceType[Any]
    view: ReferenceType[Any]
    state: ReferenceType[Any]
    cursor: ReferenceType[Any]
    rows: tuple[_Row, ...]
    inbox: tuple[_InboxPair, ...]
    erased_inbox: tuple[ErasedInboxOrigin, ...]
    erased_inbox_stamp: tuple
    untrained_inbox: tuple[UntrainedInboxOrigin, ...]
    untrained_inbox_stamp: tuple
    enrollment: int
    builder: ReferenceType[Any]
    policy: ReferenceType[Any]
    probe: ReferenceType[Any]
    models_identity: int
    attempts: int
    position: int
    attempt_limit: int
    pending_limit: int
    models: tuple[ReferenceType[Any], ...]
    budget_history: tuple
    state_content: str
    inbox_content: str
    pending_stamp: tuple
    view_history: tuple


class _Operation:
    """Strong borrowed roots exist only inside one explicitly admitted operation."""

    # These borrowed witnesses are installed only at their admitted lifecycle stage.
    original_content: str
    copy_gate: LockType
    manager_gate: LockType
    original_cursor: InboxCursor[Any, Any]
    original_inbox_content: str
    capture_inbox_content: str
    materialized_content: str
    capture_state_content: str
    restore_state: dict[str, Any]
    restored_stamps: tuple[tuple[Any, ...], ...]
    restored_content: str

    def __init__(self, ledger, owner, controller, life, enrollment, runtime, model, rows):
        self.ledger, self.owner, self.controller, self.life = ledger, owner, controller, life
        self.shared, self.registry, self.sharing = (
            owner._shared,
            life._registry,
            owner._shared._sharing,
        )
        self.enrollment, self.runtime, self.model = enrollment, runtime, model
        self.learner_type = type(runtime._candidate)
        self.originals = rows
        self.canonical = tuple(ledger._rows[id(row)] for row in rows)
        self.metadata_limit = min(ledger._admission.limits.max_metadata_bytes, 128 * 1024)
        self.history = ledger._history()
        self.history_records = (
            ()
            if self.history is None
            else ManagedInboxOrigins.verify(
                self.history,
                ledger,
                owner,
                runtime,
                ledger._read_tick(runtime),
                lifecycle_leased=True,
            )
        )
        self.history_stamp = (
            None if self.history is None else ManagedInboxOrigins.state_stamp(self.history)
        )
        self.erased_history = () if self.history is None else self.history._erased_state()
        self.untrained_history = () if self.history is None else self.history._untrained_state()
        self.history_content_limits = CheckpointContentLimits(
            4096, self.metadata_limit, ledger._admission.limits.max_payload_bytes, 16
        )
        self.original_row_stamps = tuple(_row_stamp(row) for row in self.canonical)
        self.declaration_history = tuple(
            _history_value(row.declaration(), max_metadata_bytes=self.metadata_limit)
            for row in (self.history_records if self.history is not None else self.canonical)
        )
        self.learner_stamp = tuple(
            (key, id(value)) for key, value in _metadata_fields(runtime._candidate).items()
        )
        self.learner_fields = tuple(name for name, _ in self.learner_stamp)
        self.row_capacity = min(ledger._window_limits.max_retained_snapshots, 4096)
        self.record_capacity = min(
            owner._limits.max_approved_records, owner._limits.max_replay_records, 4096
        )
        self.holder_capacity = min(life._policy.holders.max_lifetime_enrollments, 4096)
        self.inbox_capacity = min(runtime._inbox._capacity, self.record_capacity)
        self.pair_capacity = self.row_capacity if self.history is None else self.inbox_capacity
        self.consolidation_capacity = min(runtime._consolidation_limit, 4096)
        self.declaration_ticks = owner._declaration_ticks
        self.declaration_seconds = owner._declaration_seconds
        self.declaration_times = _declaration_times(owner, self.record_capacity)
        self.authorities = (
            ledger._admission,
            ledger._admission.limits,
            ledger._ports,
            tuple(ledger._port_functions),
            tuple(ledger._anchors.items()),
            life._copy_budget,
            life._copy_budget._limits,
            life._policy,
            life._measure,
            life._footprint,
            life._sampler,
            life._registry._limits,
        )
        self.limit_values = _limit_values(
            ledger, owner, life, max_metadata_bytes=self.metadata_limit
        )
        self.gates = _gate_bindings(ledger, owner, controller, runtime, life)
        sampler = life._sampler
        self.sampler_history = (
            None
            if sampler is None
            else (
                sampler.read_rss_bytes,
                sampler.interval_seconds,
                sampler._stop,
                sampler._thread,
                sampler.start_bytes,
                sampler.peak_bytes,
                sampler.sample_count,
            )
        )
        self.builder, self.policy, self.probe = (
            controller._build,
            controller._policy,
            controller._digest,
        )
        self.models_identity = id(controller._models)
        self.initial_models = tuple(controller._models)
        self.attempts, self.position = controller._attempts, len(controller._models)
        self.attempt_limit, self.pending_limit = controller._attempt_limit, controller._limit
        self.roots: dict[int, tuple[object, tuple[_Row, ...]]] = {}
        self.inboxes: dict[int, tuple[object, tuple[_InboxPair, ...]]] = {}
        self.inbox_erasures: dict[int, tuple[ErasedInboxOrigin, ...]] = {}
        self.inbox_untrained: dict[int, tuple[UntrainedInboxOrigin, ...]] = {}
        self.copying = ExitStack()
        self.event = None
        self.token = self.checkpoint = None
        self.restored_model = self.restored_rows = self.materialized = None
        self.materialized_erased_stamp: tuple = ()
        self.materialized_untrained_stamp: tuple = ()
        self.capture_state = self.capture_inbox = None
        self.published = False

    def close(self):
        try:
            self.copying.close()
        finally:
            # Why: even a failed copier must release transient graph ownership.
            self.__dict__.clear()


def _row_stamp(row):
    if (
        type(row) is not _Row
        or type(row.data) is not ReplayOriginData
        or type(row.metadata_digest) is not str
        or type(row.verified) is not bool
    ):
        raise ValueError("replay row stamp requires exact trusted metadata types")
    for reference in (
        row.source,
        row.label,
        row.declaration,
        row.snapshot,
        row.features,
        row.targets,
        row.receipt,
    ):
        if type(reference) is not ReferenceType:
            raise ValueError("replay row stamp requires exact original weak references")
    snapshot = row.snapshot() if row.snapshot is not None else None
    if snapshot is not None and type(snapshot) is not ReplaySnapshot:
        raise ValueError("replay row stamp requires exact supported native snapshot type")
    for value, expected in (
        (row.source(), Experience),
        (row.label(), LabelArrival),
        (row.declaration(), LifecycleDeclaration),
        (row.receipt() if row.receipt is not None else None, AppliedExperience),
    ):
        if value is not None and type(value) is not expected:
            raise ValueError("replay row origins require exact supported metadata record types")
    # Borrow actual object fields without invoking a user-supplied inventory port.
    fields = (
        ()
        if snapshot is None
        else tuple((key, id(value)) for key, value in _metadata_fields(snapshot).items())
    )
    origins = tuple(
        ()
        if value is None
        else tuple((key, id(field)) for key, field in _metadata_fields(value).items())
        for value in (
            row.source(),
            row.label(),
            row.receipt() if row.receipt is not None else None,
            row.declaration(),
        )
    )
    return (
        id(row.data),
        id(row.source()),
        id(row.label()),
        id(row.declaration()),
        id(snapshot),
        id(row.features() if row.features is not None else None),
        id(row.targets() if row.targets is not None else None),
        id(row.receipt() if row.receipt is not None else None),
        row.verified,
        row.metadata_digest,
        fields,
        origins,
        tuple((key, id(value)) for key, value in _metadata_fields(row.data).items()),
    )


def _gate_bindings(ledger, owner, controller, runtime, life):
    gates = (
        ledger._gate,
        owner._gate,
        controller._gate,
        runtime._write_gate,
        runtime._actor._read_gate,
        life._registry._gate,
        life._time_gate,
        life._copy_budget._gate,
        None if life._sampler is None else life._sampler._lock,
        owner._shared._sharing._gate,
    )
    if any(gate is not None and type(gate) is not LockType for gate in gates):
        raise ValueError("checkpoint ownership requires exact native lock objects")
    return gates


def _limit_values(ledger, owner, life, *, max_metadata_bytes):
    if type(life._policy) is not DataRetentionPolicy:
        raise ValueError("original lifecycle policy type changed")
    values = (
        (ledger._admission.limits, ReplayOriginLimits),
        (ledger._window_limits, ReplayWriteLimits),
        (owner._limits, LifecycleLimits),
        (life._copy_budget._limits, PayloadCopyLimits),
        (life._policy, DataRetentionPolicy),
        (life._policy.holders, PayloadOwnershipLimits),
        (owner._shared._sharing._limits, SharingLimits),
        (owner._shared._runtime._budget.budget, ToyExecutionBudget),
    )
    result = []
    for value, expected in values:
        if type(value) is not expected:
            raise ValueError("original policy limits require exact trusted metadata types")
        result.append((id(value), value))
    return _limit_value(tuple(result), max_metadata_bytes=max_metadata_bytes)


def _metadata_fields(value, expected=None):
    kind = type(value)
    known = (
        _Pending,
        _Row,
        ReplaySnapshot,
        ReplayOriginData,
        Experience,
        LabelArrival,
        AppliedExperience,
        LifecycleDeclaration,
        DataConsent,
        DataProvenance,
    )
    if expected is None and any(kind is match for match in known):
        expected = tuple(field.name for field in fields(kind))
    data = vars(value)
    if type(data) is not dict or expected is not None and len(data) != len(expected):
        raise ValueError("checkpoint metadata dictionary differs from its fixed field count")
    if any(type(key) is not str for key in data):
        raise ValueError("checkpoint metadata requires an exact dictionary with exact string keys")
    if expected is not None and data.keys() != set(expected):
        raise ValueError("checkpoint metadata dictionary differs from its fixed field schema")
    return data


def _history_value(value, *, max_metadata_bytes=128 * 1024):
    """Normalize bounded trusted metadata without invoking custom comparisons."""
    if type(max_metadata_bytes) is not int or not 0 < max_metadata_bytes <= 128 * 1024:
        raise ValueError("history normalization requires its original bounded metadata allowance")
    nodes = metadata_bytes = 0
    supported = (
        AppliedConsolidation,
        TrainingDiagnostic,
        SharingSnapshot,
        SharingLimits,
        LifecycleDeclaration,
        DataConsent,
        DataProvenance,
        ToyExecutionBudget,
        CircadianResumePosition,
        ReplayOriginLimits,
        ReplayWriteLimits,
        LifecycleLimits,
        PayloadCopyLimits,
        DataRetentionPolicy,
        PayloadOwnershipLimits,
    )

    def normalize(item, depth):
        nonlocal nodes, metadata_bytes
        nodes += 1
        metadata_bytes += 32
        if nodes > 4096 or depth > 16 or metadata_bytes > max_metadata_bytes:
            raise ValueError("checkpoint history node/depth/metadata bound exceeded")
        kind = type(item)
        if any(kind is match for match in (type(None), str, bool, int, float)):
            if kind is str:
                if len(item) > (max_metadata_bytes - metadata_bytes) // 6:
                    raise ValueError("checkpoint history string bound exceeded")
                metadata_bytes += len(item) * 6
            elif kind is int:
                if item.bit_length() > 1024:
                    raise ValueError("checkpoint history integer bound exceeded")
                metadata_bytes += (item.bit_length() + 7) // 8
            elif kind is float and not isfinite(item):
                raise ValueError("checkpoint history requires finite numeric metadata")
            if metadata_bytes > max_metadata_bytes:
                raise ValueError("checkpoint history metadata bound exceeded")
            return kind.__name__, item
        if kind is tuple:
            if len(item) > 4096 - nodes:
                raise ValueError("checkpoint history tuple bound exceeded")
            return "tuple", tuple(normalize(child, depth + 1) for child in item)
        if not any(kind is match for match in supported):
            raise ValueError("checkpoint history contains unsupported immutable metadata types")
        schema = tuple(field.name for field in fields(kind))
        data = _metadata_fields(item, schema)
        if len(data) > 4096 - nodes:
            raise ValueError("checkpoint history record bound exceeded")
        return kind.__name__, tuple(
            (name, normalize(field, depth + 1)) for name, field in data.items()
        )

    return normalize(value, 0)


def _view_history(view, *, max_metadata_bytes=128 * 1024):
    if type(view) is not CandidateCheckpointView:
        raise ValueError("checkpoint history requires exact original view type")
    _metadata_fields(view, tuple(field.name for field in fields(CandidateCheckpointView)))
    names = (
        "actor_version",
        "actor_generation",
        "learner_version",
        "attempted_ids",
        "consolidations",
        "consolidation_limit",
        "stopped",
        "sharing",
        "event_tick",
    )
    return _history_value(
        tuple((name, getattr(view, name)) for name in names), max_metadata_bytes=max_metadata_bytes
    )


def _segment_values(segment):
    if type(segment) is not ProcessRssSegment:
        raise ValueError("RSS observation requires exact original segment metadata type")
    expected = ("pid", "start_bytes", "peak_bytes", "sample_count", "interval_seconds")
    data = _metadata_fields(segment, expected)
    if tuple(data) != expected or any(type(data[name]) is not int for name in expected[:4]):
        raise ValueError("RSS observation requires exact integer counters")
    interval = data["interval_seconds"]
    if (
        not (type(interval) is int or type(interval) is float)
        or not isfinite(interval)
        or interval <= 0
    ):
        raise ValueError("RSS observation requires exact numeric interval metadata")
    return tuple(data[name] for name in expected)


def _limit_value(value, *, max_metadata_bytes):
    return _history_value(value, max_metadata_bytes=max_metadata_bytes)


def _budget_value(value, *, max_metadata_bytes):
    return _history_value(value, max_metadata_bytes=max_metadata_bytes)


def _declaration_times(owner, maximum):
    result = []
    for mapping, integer in ((owner._declaration_ticks, True), (owner._declaration_seconds, False)):
        if type(mapping) is not dict or len(mapping) > maximum:
            raise ValueError(
                "declaration time anchors require their original bounded exact dictionaries"
            )
        entries = []
        for key, value in mapping.items():
            if (
                type(key) is not tuple
                or len(key) != 2
                or any(type(item) is not str for item in key)
            ):
                raise ValueError("declaration time anchors require exact supported sample keys")
            if integer:
                if type(value) is not int or value < 0:
                    raise ValueError(
                        "declaration tick anchors require exact original nonnegative integers"
                    )
            elif (
                not (type(value) is int or type(value) is float) or not isfinite(value) or value < 0
            ):
                raise ValueError(
                    "declaration elapsed anchors require finite original numeric metadata"
                )
            entries.append((key, value))
        result.append(tuple(entries))
    return tuple(result)


def _require_partition_maps(mapping, maximum, kind):
    """Keep operation inventory lookup pure after the last opaque probe."""
    if type(maximum) is not int or not 0 < maximum <= 4096:
        raise ValueError("checkpoint partition map requires an exact bounded maximum")
    # Why13: each map root must originate in the existing graph event window.
    if type(mapping) is not dict or len(mapping) > 13:
        raise ValueError("checkpoint partition map requires an exact bounded dictionary")
    for key, records in mapping.items():
        if type(key) is not int or not 0 < key < 2**63:
            raise ValueError("checkpoint partition map keys require exact positive identities")
        if type(records) is not tuple or len(records) > maximum:
            raise ValueError("checkpoint partition map records require an exact bounded tuple")
        if any(type(record) is not kind for record in records):
            raise ValueError("checkpoint partition map records require exact witness types")


def _require_inbox_keys(mapping, maximum, metadata_maximum):
    """Validate exact scalar keys before dictionary equality or lookup."""
    if (
        type(maximum) is not int
        or not 0 < maximum <= 4096
        or type(metadata_maximum) is not int
        or not 0 < metadata_maximum <= 128 * 1024
    ):
        raise ValueError("checkpoint inbox keys require exact bounded original limits")
    if type(mapping) is not dict or len(mapping) > maximum:
        raise ValueError("checkpoint inbox keys require an exact bounded dictionary")
    for key in mapping:
        if type(key) is not tuple or len(key) != 2:
            raise ValueError("checkpoint inbox keys require exact paired strings")
        if any(type(part) is not str for part in key):
            raise ValueError("checkpoint inbox keys require exact paired strings")
        if sum(len(part) * 6 for part in key) > metadata_maximum or any(
            not part.strip() for part in key
        ):
            raise ValueError("checkpoint inbox keys exceed bounded identifier metadata")


def _require_untrained_stamp(value, maximum):
    """Keep stored proof equality pure even after an opaque callback."""
    if type(maximum) is not int or not 0 < maximum <= 4096:
        raise ValueError("untrained checkpoint stamp requires an exact bounded maximum")
    if type(value) is not tuple or len(value) > maximum:
        raise ValueError("untrained checkpoint stamp requires an exact bounded tuple")
    for stamp in value:
        if type(stamp) is not tuple or len(stamp) != 6:
            raise ValueError("untrained checkpoint stamp requires six exact scalar fields")
        if any(type(identity) is not int or not 0 <= identity < 2**63 for identity in stamp[:4]):
            raise ValueError("untrained checkpoint stamp identities require exact bounded integers")
        if any(
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in stamp[4:]
        ):
            raise ValueError("untrained checkpoint stamp requires exact SHA256 scalars")


def _checkpoint_references(witness, model_limit):
    if type(witness) is not _Checkpoint:
        raise ValueError("checkpoint requires its exact original witness type")
    references = (
        witness.token,
        witness.controller,
        witness.pending,
        witness.view,
        witness.state,
        witness.cursor,
        witness.builder,
        witness.policy,
        witness.probe,
    )
    if any(type(reference) is not ReferenceType for reference in references):
        raise ValueError(
            "checkpoint witness requires exact original weak references before dereference"
        )
    if (
        type(witness.models) is not tuple
        or len(witness.models) > model_limit
        or any(type(reference) is not ReferenceType for reference in witness.models)
    ):
        raise ValueError("checkpoint retained history requires bounded original weak references")


def _inbox_references(pairs, maximum, content_limits):
    if type(pairs) is not tuple or len(pairs) > maximum:
        raise ValueError("inbox lineage requires its original bounded witness tuple")
    for pair in pairs:
        if type(pair) is not _InboxPair:
            raise ValueError("inbox lineage requires exact trusted witness records")
        if any(
            type(reference) is not ReferenceType
            for reference in (pair.source, pair.label, pair.receipt, pair.features, pair.targets)
        ):
            raise ValueError("inbox lineage requires exact weak references before dereference")
        if type(pair.origin) is _Row:
            _row_stamp(pair.origin)
        elif type(pair.origin) is _InboxOrigin:
            inbox_origin_stamp(pair.origin, content_limits)
        else:
            raise ValueError("inbox pair lacks exact original lineage")


def _trusted_actor_stamp(actor):
    if type(actor) is StableActor:
        version, generation = actor._version, 0
    elif type(actor) is PromotableActor:
        slot = actor._slot
        if type(slot) is not _Slot:
            raise ValueError("actor generation requires its exact original serving slot type")
        bundle = slot.bundle
        if type(bundle) is not _Bundle:
            raise ValueError("actor version requires its exact original serving bundle type")
        version, generation = bundle.version, slot.generation
    else:
        raise ValueError("actor continuity requires an exact supported actor type")
    if type(version) is not str or type(generation) is not int or generation < 0:
        raise ValueError("actor continuity requires exact string version and integer generation")
    return version, generation


def _budget_history(state, *, max_metadata_bytes):
    if type(state) is not dict:
        raise ValueError("checkpoint cumulative budget state requires an exact dictionary")
    schema = (
        "budget",
        "started_at",
        "last_clock",
        "updates_completed",
        "replay_examples_completed",
        "hidden_width_observed",
        "peak_hidden_width_observed",
        "rejected_proposed_hidden_width",
        "checkpoint_position",
        "process_rss_segment",
        "progress_state",
    )
    if len(state) != len(schema):
        raise ValueError("checkpoint budget dictionary differs from fixed field count")
    if any(type(key) is not str for key in state):
        raise ValueError("checkpoint budget keys require exact strings")
    if state.keys() != set(schema):
        raise ValueError("checkpoint budget dictionary differs from fixed field schema")
    result = []
    for key, value in state.items():
        if type(key) is not str:
            raise ValueError("checkpoint budget keys require exact strings")
        if key in ("last_clock", "process_rss_segment"):
            continue
        if key == "progress_state":
            if value is not None and type(value) is not dict:
                raise ValueError("checkpoint cumulative progress requires exact field dictionary")
            progress_schema = tuple(field.name for field in fields(ToyExecutionProgress))
            if value is not None and len(value) != len(progress_schema):
                raise ValueError("checkpoint progress dictionary differs from fixed field count")
            if value is not None and any(type(name) is not str for name in value):
                raise ValueError("checkpoint progress keys require exact strings")
            if value is not None and value.keys() != set(progress_schema):
                raise ValueError("checkpoint progress dictionary differs from fixed field schema")
            normalized = (
                None
                if value is None
                else tuple(
                    (name, field) for name, field in value.items() if name != "process_rss_segment"
                )
            )
            result.append((key, normalized))
        else:
            result.append((key, value))
    return _budget_value(tuple(result), max_metadata_bytes=max_metadata_bytes)


def _require_collections(op, prepared=None):
    _require_partition_maps(op.inbox_erasures, op.pair_capacity, ErasedInboxOrigin)
    _require_partition_maps(op.inbox_untrained, op.pair_capacity, UntrainedInboxOrigin)
    """Reject collection replacements before their iteration can execute code."""
    if (
        type(op.owner) is not ManagedExperienceOwner
        or type(op.ledger) is not ManagedReplayOrigins
        or type(op.controller) is not CandidateCheckpointController
        or type(op.life) is not ManagedDataLifecycle
    ):
        raise ValueError("checkpoint publication authority object type changed")
    if (
        op.owner._shared is not op.shared
        or type(op.shared) is not ResourceSharedRuntime
        or op.shared._sharing is not op.sharing
        or type(op.sharing) is not ServingPriorityGate
        or op.shared._runtime is not op.runtime
    ):
        raise ValueError("checkpoint original shared runtime/sharing authority changed")
    if op.life._registry is not op.registry or type(op.registry) is not PayloadOwnershipRegistry:
        raise ValueError("checkpoint original registry authority changed")
    if (
        op.owner._declaration_ticks is not op.declaration_ticks
        or op.owner._declaration_seconds is not op.declaration_seconds
        or _declaration_times(op.owner, op.record_capacity) != op.declaration_times
    ):
        raise ValueError("original declaration retention time anchors changed or were renewed")
    for value, expected, maximum in (
        (op.ledger._rows, dict, op.row_capacity),
        (op.ledger._anchors, dict, 9),
        (op.controller._models, list, min(op.attempt_limit, op.position + 1)),
        (op.controller._pending, dict, op.pending_limit),
        (op.life._registry._holders, dict, op.holder_capacity),
        (op.owner._catalog, dict, op.record_capacity),
        (op.owner._revoked_keys, set, op.record_capacity),
        (op.owner._opted_out, set, op.record_capacity),
        (op.owner._shared._sharing._deferrals, dict, 5),
    ):
        if type(value) is not expected:
            raise ValueError(
                "checkpoint publication collection differs from its exact supported type"
            )
        if len(value) > maximum:
            raise ValueError(
                "checkpoint publication collection exceeds its original bounded capacity"
            )
    anchor_names = {
        "runtime",
        "actor",
        "learner",
        "model",
        "inbox",
        "budget",
        "clock",
        "budget_policy",
        "owner_policy",
    }
    if (
        any(
            type(name) is not str or type(reference) is not ReferenceType
            for name, reference in op.ledger._anchors.items()
        )
        or op.ledger._anchors.keys() != anchor_names
    ):
        raise ValueError("checkpoint original anchors require the fixed weak-reference schema")
    if any(type(key) is not int or type(row) is not _Row for key, row in op.ledger._rows.items()):
        raise ValueError("checkpoint original rows require exact identity keys and trusted records")
    if type(op.canonical) is not tuple or len(op.canonical) > op.row_capacity:
        raise ValueError("canonical row witnesses exceed their original bounded tuple schema")
    for row in op.ledger._rows.values():
        _row_stamp(row)
    for row in op.canonical:
        _row_stamp(row)
    if any(
        type(token) is not CandidateCheckpoint or type(pending) is not _Pending
        for token, pending in op.controller._pending.items()
    ):
        raise ValueError("checkpoint pending history requires exact original token/pending types")
    for number, entry in op.registry._holders.items():
        if (
            type(number) is not int
            or type(entry) is not tuple
            or len(entry) != 2
            or type(entry[0]) is not str
            or type(entry[1]) is not ReferenceType
        ):
            raise ValueError(
                "registry membership requires exact number/kind/weak reference records"
            )
    for reference in (
        op.ledger._owner,
        op.ledger._lifecycle,
        op.ledger._original_admission,
        op.ledger._original_copy_budget,
        op.ledger._original_copy_limits,
    ):
        if type(reference) is not ReferenceType:
            raise ValueError("original replay authority requires exact weak-reference bindings")
    if op.ledger._copy_guard is not None and not (
        type(op.ledger._copy_guard) is ReferenceType or type(op.ledger._copy_guard) is WeakMethod
    ):
        raise ValueError("original copy guard requires its original weak reference type")
    if any(
        type(value) is not int
        for value in (
            op.controller._attempts,
            op.controller._attempt_limit,
            op.controller._limit,
            op.registry._total,
            op.ledger._last_tick,
            op.ledger._last_work,
            op.ledger._copy_slots,
            op.ledger._minimum_copy_slots,
        )
    ):
        raise ValueError("checkpoint authority/history counters require exact integer metadata")
    if (
        not (
            type(op.ledger._last_budget_clock) is int or type(op.ledger._last_budget_clock) is float
        )
        or not isfinite(op.ledger._last_budget_clock)
        or type(op.life._clock) is not LogicalClock
        or type(op.life._clock._time) is not int
    ):
        raise ValueError(
            "checkpoint original time history requires exact supported primitive types"
        )
    if (
        type(op.ledger._version) is not str
        or not (type(op.ledger._budget_origin) is int or type(op.ledger._budget_origin) is float)
        or not isfinite(op.ledger._budget_origin)
    ):
        raise ValueError(
            "checkpoint original version/budget origin requires exact finite primitive metadata"
        )
    if op.life._last_seconds is not None and (
        not (type(op.life._last_seconds) is int or type(op.life._last_seconds) is float)
        or not isfinite(op.life._last_seconds)
    ):
        raise ValueError("checkpoint elapsed retention requires exact numeric metadata")
    runtimes = (op.runtime,) if prepared is None else (op.runtime, prepared)
    for runtime in runtimes:
        if (
            type(runtime) is not ActorShadowRuntime
            or type(runtime._inbox) is not ExperienceInbox
            or type(runtime._budget) is not ToyBudgetSession
        ):
            raise ValueError(
                "publication requires exact original runtime/inbox/budget object types"
            )
        if type(runtime._candidate) is not op.learner_type:
            raise ValueError("publication requires the original exact supported learner type")
        if not (type(runtime._actor) is StableActor or type(runtime._actor) is PromotableActor):
            raise ValueError("publication requires exact original supported actor type")
        for value in (
            runtime._retired,
            runtime._stopped,
            runtime._inbox._draining,
            runtime._inbox._stopped,
        ):
            if type(value) is not bool:
                raise ValueError("runtime/inbox status requires exact boolean metadata")
        for value in (
            runtime._revision,
            runtime._consolidation_limit,
            runtime._inbox._last_time,
            runtime._inbox._capacity,
            runtime._budget.updates_completed,
        ):
            if type(value) is not int:
                raise ValueError("runtime/inbox work history requires exact integer metadata")
        for value in (
            runtime._candidate_version,
            runtime._base_actor_version,
            runtime._inbox._learner_version,
        ):
            if type(value) is not str:
                raise ValueError("runtime/inbox versions require exact string metadata")
        inbox = runtime._inbox
        for value, expected, maximum in (
            (inbox._experiences, dict, op.inbox_capacity),
            (inbox._labels, dict, op.inbox_capacity),
            (inbox._applied, dict, op.inbox_capacity),
            (inbox._erased, dict, op.inbox_capacity),
            (inbox._event_ids, set, op.inbox_capacity),
            (runtime._attempted_ids, set, op.consolidation_capacity),
            (runtime._consolidations, list, op.consolidation_capacity),
            (vars(runtime._candidate), dict, len(op.learner_fields)),
            (vars(runtime._budget), dict, 13),
        ):
            if type(value) is not expected:
                raise ValueError(
                    "checkpoint runtime history differs from exact supported collections"
                )
            if len(value) > maximum:
                raise ValueError("checkpoint runtime history exceeds its original bounded capacity")
        _metadata_fields(runtime._candidate, op.learner_fields)
        for mapping in (inbox._experiences, inbox._labels, inbox._applied, inbox._erased):
            _require_inbox_keys(mapping, op.inbox_capacity, op.metadata_limit)
        for values, record_type in (
            (inbox._experiences.values(), Experience),
            (inbox._labels.values(), LabelArrival),
            (inbox._applied.values(), AppliedExperience),
            (inbox._erased.values(), ErasedExperience),
        ):
            if any(type(value) is not record_type for value in values):
                raise ValueError(
                    "checkpoint current history requires exact supported inbox records"
                )
        if any(type(value) is not str for value in inbox._event_ids):
            raise ValueError("checkpoint event index requires exact string identifiers")
        if any(type(value) is not str for value in runtime._attempted_ids):
            raise ValueError("checkpoint attempted history requires exact string identifiers")
        if any(type(value) is not AppliedConsolidation for value in runtime._consolidations):
            raise ValueError("checkpoint consolidation history requires exact receipt types")
    if any(
        type(key) is not str or type(value) is not int
        for key, value in op.owner._shared._sharing._deferrals.items()
    ):
        raise ValueError("checkpoint sharing deferral history requires exact string/int metadata")
    gate = op.owner._shared._sharing
    if any(
        type(value) is not bool
        for value in (
            gate._checkpointing,
            gate._training,
            gate._paused,
            op.life._failed,
            op.life._retention_fault,
        )
    ) or any(type(value) is not int for value in (gate._serving, gate._admitted)):
        raise ValueError("sharing/lifecycle status requires exact primitive metadata")
    if (
        type(op.model) is not CircadianPredictiveCodingNetwork
        or type(op.model.__dict__) is not dict
        or type(op.model.__dict__.get("_replay_memory")) is not deque
    ):
        raise ValueError(
            "checkpoint original native inventory requires exact state dict/replay deque"
        )
    if prepared is not None:
        if type(prepared._candidate.__dict__) is not dict:
            raise ValueError("prepared learner state requires exact dictionary")
        if (
            type(op.restored_model) is not CircadianPredictiveCodingNetwork
            or type(op.restored_model.__dict__) is not dict
            or type(op.restored_model.__dict__.get("_replay_memory")) is not deque
        ):
            raise ValueError("prepared native state requires exact state dictionary/replay deque")
    if op.checkpoint is not None:
        _checkpoint_references(op.checkpoint, op.attempt_limit)
        _inbox_references(op.checkpoint.inbox, op.pair_capacity, op.history_content_limits)
        require_erased_inbox_records(
            op.checkpoint.erased_inbox, op.history_content_limits, op.pair_capacity
        )
        require_untrained_inbox_records(
            op.checkpoint.untrained_inbox, op.history_content_limits, op.pair_capacity
        )
        _require_untrained_stamp(op.checkpoint.untrained_inbox_stamp, op.pair_capacity)
        if (
            tuple(
                untrained_inbox_origin_stamp(record, op.history_content_limits)
                for record in op.checkpoint.untrained_inbox
            )
            != op.checkpoint.untrained_inbox_stamp
        ):
            raise ValueError("captured untrained inbox original weak witnesses changed")
        if (
            tuple(
                erased_inbox_origin_stamp(record, op.history_content_limits)
                for record in op.checkpoint.erased_inbox
            )
            != op.checkpoint.erased_inbox_stamp
        ):
            raise ValueError("captured erased inbox original weak witnesses changed")
        view = op.checkpoint.view()
        if type(view) is not CandidateCheckpointView or type(view.budget_state) is not dict:
            raise ValueError(
                "checkpoint pending view/budget history differs from exact supported format"
            )
        cursor = view.inbox
        if type(cursor) is not InboxCursor or any(
            type(value) is not tuple
            for value in (
                cursor.experiences,
                cursor.labels,
                cursor.applied,
                cursor.erased,
                view.attempted_ids,
                view.consolidations,
            )
        ):
            raise ValueError("checkpoint pending history requires exact immutable tuples")
    if op.materialized is not None:
        if type(op.materialized) is not tuple or len(op.materialized) != 2:
            raise ValueError("materialized inbox requires its original admitted tuple")
        _inbox_references(op.materialized[1], op.pair_capacity, op.history_content_limits)
        erased = op.inbox_erasures.get(id(op.materialized[0]))
        untrained = op.inbox_untrained.get(id(op.materialized[0]))
        require_untrained_inbox_records(untrained, op.history_content_limits, op.pair_capacity)
        _require_untrained_stamp(op.materialized_untrained_stamp, op.pair_capacity)
        if (
            tuple(
                untrained_inbox_origin_stamp(record, op.history_content_limits)
                for record in untrained
            )
            != op.materialized_untrained_stamp
        ):
            raise ValueError("materialized untrained inbox original weak witnesses changed")
        require_erased_inbox_records(erased, op.history_content_limits, op.pair_capacity)
        if (
            tuple(erased_inbox_origin_stamp(record, op.history_content_limits) for record in erased)
            != op.materialized_erased_stamp
        ):
            raise ValueError("materialized erased inbox original weak witnesses changed")


class ManagedReplayCheckpoints:
    def __init__(
        self,
        ledger: ManagedReplayOrigins,
        graph_ports: ReplayGraphPorts,
        *,
        builder_source: Callable,
    ):
        if type(ledger) is not ManagedReplayOrigins or type(graph_ports) is not ReplayGraphPorts:
            raise ValueError(
                "managed replay checkpoints require original ledger and exact graph ports"
            )
        if not callable(builder_source):
            raise ValueError("managed replay checkpoints require original builder-source port")
        self._ledger = _weak(ledger)
        self._graph_ports = self._original_graph_ports = graph_ports
        self._graph_functions = (graph_ports.rows, graph_ports.payload_bytes)
        self._copies = self._original_copies = ManagedReplayCopies(
            ledger, builder_source=builder_source
        )
        self._content_limits = self._original_content_limits = CheckpointContentLimits(
            4096,
            ledger._admission.limits.max_metadata_bytes,
            ledger._admission.limits.max_payload_bytes,
            16,
        )
        self._content_limit_values = tuple(vars(self._content_limits).values())
        self._checkpoints: dict[int, _Checkpoint] = {}
        self._active: _Operation | None = None
        self._gate = Lock()
        self._original_gate, self._original_copy_gate = self._gate, self._copies._gate

    @contextmanager
    def _exclusive(self):
        gate = self._gate
        if type(gate) is not LockType or not gate.acquire(blocking=False):
            raise PayloadOwnershipBusy("managed replay checkpoints are busy")
        try:
            yield
        finally:
            gate.release()

    def _original_ledger(self):
        self._require_content_limits()
        if type(self._ledger) is not ReferenceType:
            raise ValueError("original manager ledger requires its exact weak reference")
        ledger = self._ledger()
        if (
            ledger is None
            or self._copies is not self._original_copies
            or self._copies._original_ledger() is not ledger
            or self._graph_ports is not self._original_graph_ports
            or self._graph_ports.rows is not self._graph_functions[0]
            or self._graph_ports.payload_bytes is not self._graph_functions[1]
            or self._content_limits is not self._original_content_limits
            or tuple(vars(self._content_limits).values()) != self._content_limit_values
            or self._gate is not self._original_gate
            or self._copies._gate is not self._original_copy_gate
        ):
            raise ValueError("original checkpoint ledger/copy/graph authority changed or expired")
        return ledger

    def _begin(self, controller):
        ledger = self._original_ledger()
        owner = ledger._original_owner()
        life = owner._lifecycle
        if life is None:
            raise ValueError("checkpoint graph admission requires original bounded lifecycle")
        initial_gates = _gate_bindings(ledger, owner, controller, owner._shared._runtime, life)
        with life._registry._lease():
            life, enrollment = self._copies._controller(owner, controller)
            runtime, model, rows = self._copies._source(ledger, owner)
            if not rows:
                raise ValueError(
                    "checkpoint graph admission requires nonempty original row origins"
                )
            runtime._budget.before_final()
            op = _Operation(ledger, owner, controller, life, enrollment, runtime, model, rows)
            if any(a is not b for a, b in zip(op.gates, initial_gates)):
                raise ValueError(
                    "original checkpoint holder gates changed during initial opaque validation"
                )
            op.original_content = self._stamp(model.__dict__)
            op.copy_gate, op.manager_gate = self._copies._gate, self._gate
            return op

    def _stamp(self, value):
        self._require_content_limits()
        return checkpoint_content_stamp(value, self._content_limits)

    def _require_content_limits(self):
        limits = self._content_limits
        if (
            type(limits) is not CheckpointContentLimits
            or limits is not self._original_content_limits
        ):
            raise ValueError("checkpoint original content proof limit authority changed")
        _metadata_fields(
            limits, ("max_nodes", "max_metadata_bytes", "max_array_bytes", "max_depth")
        )
        CheckpointContentLimits.__post_init__(limits)
        if tuple(vars(limits).values()) != self._content_limit_values:
            raise ValueError("checkpoint original content proof limit values changed")

    def _prove_current(self, op, *, publication=False):
        if op.ledger._history() is not op.history or (
            op.history is not None
            and ManagedInboxOrigins.state_stamp(op.history) != op.history_stamp
        ):
            raise ValueError("original committed inbox history changed during opaque validation")
        if self._stamp(op.model.__dict__) != op.original_content:
            raise ValueError("original native model contents changed during checkpoint operation")
        if op.checkpoint is not None:
            if (
                self._stamp(op.checkpoint.state()) != op.checkpoint.state_content
                or self._stamp(op.checkpoint.cursor()) != op.checkpoint.inbox_content
            ):
                raise ValueError("original pending checkpoint state/inbox contents changed")
        if (
            getattr(op, "original_cursor", None) is not None
            and self._stamp(op.original_cursor) != op.original_inbox_content
        ):
            raise ValueError(
                "original current inbox payload/history contents changed during opaque validation"
            )
        if publication:
            if self._stamp(op.restored_model.__dict__) != op.restored_content:
                raise ValueError(
                    "actual restored native contents changed after original copied event"
                )
            if self._stamp(op.materialized[0]) != op.materialized_content:
                raise ValueError(
                    "actual materialized inbox contents changed after original copied event"
                )

    def _limits(self, op):
        # One bounded window spans the entire graph chain; roots cannot renew it.
        count = 13
        rows = max(op.ledger._window_limits.max_retained_snapshots, op.pair_capacity)
        return ModelCopyLimits(count, 2 * count, count * (6 * rows + 16))

    def capture(self, controller):
        with self._exclusive():
            ledger = self._original_ledger()
            owner = ledger._original_owner()
            with ledger._exclusive(), owner._operation():
                op = self._begin(controller)
                self._active = op
                try:
                    with observe_graph_copies(self._observe, self._limits(op)):
                        token = controller.capture()
                    with ExitStack() as stack:
                        _, observe = self._copies._lease_copy(op.life, stack)
                        stack.enter_context(controller._exclusive())
                        stack.enter_context(op.runtime._exclusive())
                        self._current(op)
                        pending = controller._require(token)
                        if (
                            type(token) is not CandidateCheckpoint
                            or type(pending.view) is not CandidateCheckpointView
                            or pending.owner is not op.runtime
                            or pending.view.state is not op.capture_state
                            or pending.view.inbox is not op.capture_inbox
                        ):
                            raise ValueError(
                                "captured checkpoint differs from admitted actual graph chain"
                            )
                        op.runtime._budget._before_final_leased(observe)
                        self._current(op)
                        self._prove_current(op)
                        if (
                            self._stamp(pending.view.state) != op.capture_state_content
                            or self._stamp(pending.view.inbox) != op.capture_inbox_content
                        ):
                            raise ValueError(
                                "captured native/inbox contents changed after original copied events"
                            )
                        state, rows = op.roots[id(pending.view.state)]
                        cursor, pairs = op.inboxes[id(pending.view.inbox)]
                        self._verify_inbox(
                            cursor,
                            pairs,
                            op.inbox_erasures[id(cursor)],
                            untrained=op.inbox_untrained[id(cursor)],
                            content_limits=op.history_content_limits,
                            maximum=op.pair_capacity,
                        )
                        witness = _Checkpoint(
                            _weak(token),
                            _weak(controller),
                            _weak(pending),
                            _weak(pending.view),
                            _weak(state),
                            _weak(cursor),
                            rows,
                            pairs,
                            op.inbox_erasures[id(cursor)],
                            tuple(
                                erased_inbox_origin_stamp(record, op.history_content_limits)
                                for record in op.inbox_erasures[id(cursor)]
                            ),
                            op.inbox_untrained[id(cursor)],
                            tuple(
                                untrained_inbox_origin_stamp(record, op.history_content_limits)
                                for record in op.inbox_untrained[id(cursor)]
                            ),
                            op.enrollment,
                            _weak(op.builder),
                            _weak(op.policy),
                            _weak(op.probe),
                            op.models_identity,
                            op.attempts,
                            op.position,
                            op.attempt_limit,
                            op.pending_limit,
                            tuple(_weak(model) for model in op.initial_models),
                            _budget_history(
                                pending.view.budget_state, max_metadata_bytes=op.metadata_limit
                            ),
                            op.capture_state_content,
                            op.capture_inbox_content,
                            tuple(
                                (name, id(value))
                                for name, value in _metadata_fields(pending).items()
                            ),
                            _view_history(pending.view, max_metadata_bytes=op.metadata_limit),
                        )
                        self._checkpoints = {
                            key: old
                            for key, old in self._checkpoints.items()
                            if old.token() is not None and old.controller() is not None
                        }
                        self._checkpoints[id(token)] = witness
                    return token
                finally:
                    self._active = None
                    op.close()

    def restore(self, controller, token):
        with self._exclusive():
            ledger = self._original_ledger()
            witness = self._checkpoints.get(id(token))
            if type(witness) is not _Checkpoint or witness.token() is not token:
                raise ValueError("restore requires the actual originally captured checkpoint token")
            # The existing copy wrapper owns ledger/owner leases and real fork admission.
            with ledger._exclusive(), ledger._original_owner()._operation():
                op = self._begin(controller)
                op.token, op.checkpoint = token, witness
                self._require_checkpoint(op, controller._require(token))
                self._prove_current(op)
                op.roots[id(witness.state())] = witness.state(), witness.rows
                op.inboxes[id(witness.cursor())] = witness.cursor(), witness.inbox
                op.inbox_erasures[id(witness.cursor())] = witness.erased_inbox
                op.inbox_untrained[id(witness.cursor())] = witness.untrained_inbox
            self._active = op
            try:
                with (
                    observe_graph_copies(self._observe, self._limits(op)),
                    lease_replay_handoff(controller, self),
                ):
                    result = self._copies.restore(controller, token)
                return result
            finally:
                self._active = None
                op.close()

    def _require_checkpoint(self, op, pending: _Pending[Any, Any, Any, Any]):
        witness = op.checkpoint
        controller: CandidateCheckpointController[Any, Any, Any, Any] = op.controller
        _checkpoint_references(witness, op.attempt_limit)
        if (
            type(witness) is not _Checkpoint
            or type(pending) is not _Pending
            or witness.token() is not op.token
            or witness.controller() is not controller
            or witness.pending() is not pending
            or pending is not controller._pending.get(op.token)
            or witness.view() is not pending.view
            or witness.state()
            is not cast(CandidateCheckpointView[Any, Any, Any], pending.view).state
            or witness.cursor()
            is not cast(CandidateCheckpointView[Any, Any, Any], pending.view).inbox
            or witness.enrollment != op.enrollment
            or witness.builder() is not controller._build
            or witness.policy() is not controller._policy
            or witness.probe() is not controller._digest
            or witness.models_identity != id(controller._models)
            or witness.attempt_limit != controller._attempt_limit
            or witness.pending_limit != controller._limit
            or not witness.attempts <= controller._attempts <= controller._attempt_limit
            or witness.position > len(controller._models)
            or any(
                reference() is not model
                for reference, model in zip(witness.models, controller._models)
            )
            or tuple((name, id(value)) for name, value in _metadata_fields(pending).items())
            != witness.pending_stamp
        ):
            raise ValueError("checkpoint original membership/view/history or producer changed")

    def _current(self, op):
        _require_collections(op)
        self._require_gates(op)
        if self._original_ledger() is not op.ledger:
            raise ValueError("checkpoint original ledger changed")
        life, enrollment = self._copies._controller(op.owner, op.controller)
        runtime, model, rows = self._copies._source(op.ledger, op.owner, leased=True)
        controller = op.controller
        if (
            life is not op.life
            or enrollment != op.enrollment
            or runtime is not op.runtime
            or model is not op.model
            or len(rows) != len(op.originals)
            or any(a is not b for a, b in zip(rows, op.originals))
            or any(op.ledger._rows.get(id(row)) is not old for row, old in zip(rows, op.canonical))
            or controller._build is not op.builder
            or controller._policy is not op.policy
            or controller._digest is not op.probe
            or id(controller._models) != op.models_identity
            or controller._attempt_limit != op.attempt_limit
            or controller._limit != op.pending_limit
            or any(a is not b for a, b in zip(controller._models, op.initial_models))
        ):
            raise ValueError("checkpoint original source, history or controller authority changed")
        if op.checkpoint is None:
            if controller._attempts != op.attempts or len(controller._models) != op.position:
                raise ValueError("capture unexpectedly changed checkpoint preparation history")
        else:
            self._require_checkpoint(op, controller._require(op.token))
            if controller._attempts not in (op.attempts, op.attempts + 1) or len(
                controller._models
            ) not in (op.position, op.position + 1):
                raise ValueError("restore exceeds its original single preparation attempt")
        if runtime._inbox._draining or runtime._inbox._historical_completed_updates is not None:
            raise ValueError("checkpoint requires quiescent nonhistorical original inbox")
        _require_partition_maps(op.inbox_erasures, op.pair_capacity, ErasedInboxOrigin)
        _require_partition_maps(op.inbox_untrained, op.pair_capacity, UntrainedInboxOrigin)
        return runtime, model

    @staticmethod
    def _require_gates(op, *, held=False):
        current = _gate_bindings(op.ledger, op.owner, op.controller, op.runtime, op.life)
        if len(current) != len(op.gates) or any(a is not b for a, b in zip(current, op.gates)):
            raise ValueError(
                "checkpoint original operation/registry/time/copy/sampler/sharing/actor gates changed"
            )
        if held and any(not gate.locked() for gate in op.gates if gate is not None):
            raise ValueError("checkpoint original publication gate lease is no longer held")

    def _rows(self, op, root):
        rows = self._graph_ports.rows(root, op.ledger._window_limits.max_retained_snapshots)
        if type(rows) is not tuple or len(rows) != len(op.canonical):
            raise ValueError("graph replay inventory differs from original admitted row count")
        return rows

    def _verify_rows(self, op, root, records):
        rows = self._rows(op, root)
        if len(records) != len(rows):
            raise ValueError("graph replay witness count changed")
        now = op.ledger._read_tick(op.runtime)
        for row, record in zip(rows, records):
            op.ledger._verify_row(op.owner, op.runtime, row, record, now, lifecycle_leased=True)
        return rows

    def _qualify_native(self, op, producer, kind, source):
        controller = op.controller
        if kind == "native_snapshot":
            if producer is op.model and source is op.model.__dict__:
                records = op.canonical
            else:
                learner = (
                    controller._models[op.position]
                    if len(controller._models) == op.position + 1
                    else None
                )
                if (
                    learner is None
                    or op.ledger._ports.model_reference(learner) is not producer
                    or source is not producer.__dict__
                ):
                    raise ValueError(
                        "native snapshot source is not an actual original retained model dictionary"
                    )
                if op.restored_model is producer:
                    records = op.restored_rows
                else:
                    copy = self._copies._copies.get(id(producer))
                    self._copies._require_copy(copy, controller, learner, producer, op.enrollment)
                    assert copy is not None
                    records = copy.rows
            self._verify_rows(op, source, records)
            return records
        if producer is not controller and kind != "native_restore":
            raise ValueError("checkpoint graph producer differs from original controller")
        if kind == "checkpoint_capture":
            state = getattr(source, "state", None)
            bound = op.roots.get(id(state))
            if op.checkpoint is not None or bound is None or bound[0] is not state:
                raise ValueError(
                    "checkpoint capture lacks its actual preceding native snapshot root"
                )
            records = bound[1]
        elif kind in ("checkpoint_build_state", "checkpoint_restore_state"):
            if op.checkpoint is None or source is not op.checkpoint.state():
                raise ValueError("checkpoint preparation root differs from original pending state")
            records = op.checkpoint.rows
        elif kind == "native_restore":
            bound = op.roots.get(id(source))
            learner = (
                controller._models[op.position]
                if len(controller._models) == op.position + 1
                else None
            )
            if (
                learner is None
                or op.ledger._ports.model_reference(learner) is not producer
                or bound is None
                or bound[0] is not source
                or source is not getattr(op, "restore_state", None)
            ):
                raise ValueError(
                    "native restore root lacks original copied checkpoint restore state"
                )
            records = bound[1]
        else:
            raise ValueError("unsupported original native checkpoint graph kind")
        self._verify_rows(op, source, records)
        return records

    def _original_inbox(self, op, cursor):
        if type(cursor) is not InboxCursor:
            raise ValueError("inbox graph root requires an exact actual cursor")
        inbox = op.runtime._inbox
        sources = tuple(inbox._experiences[k] for k in sorted(inbox._experiences))
        labels = tuple(inbox._labels[k] for k in sorted(inbox._labels))
        receipts = tuple(inbox._applied.values())
        if (
            len(cursor.experiences) != len(sources)
            or len(cursor.labels) != len(labels)
            or len(cursor.applied) != len(receipts)
            or any(a is not b for a, b in zip(cursor.experiences, sources))
            or any(a is not b for a, b in zip(cursor.labels, labels))
            or any(a is not b for a, b in zip(cursor.applied, receipts))
            or cursor.last_tick != inbox._last_time
            or cursor.completed_updates != op.runtime._budget.updates_completed
            or cursor.learner_version != inbox._learner_version
            or cursor.capacity != inbox._capacity
            or cursor.stopped != inbox._stopped
            or cursor.erased != tuple(inbox._erased[k] for k in sorted(inbox._erased))
        ):
            raise ValueError(
                "inbox copy root differs from actual current original cursor identities/history"
            )
        result = []
        seen = set()
        originals = (
            op.canonical
            if op.history is None
            else ManagedInboxOrigins.verify(
                op.history,
                op.ledger,
                op.owner,
                op.runtime,
                op.ledger._read_tick(op.runtime),
                lifecycle_leased=True,
            )
        )
        for old in originals:
            source, label = old.source(), old.label()
            receipt = old.receipt() if old.receipt is not None else None
            if id(source) in seen:
                continue
            if (
                type(source) is not Experience
                or type(label) is not LabelArrival
                or type(receipt) is not AppliedExperience
            ):
                raise ValueError("inbox graph pair lacks original source/label/receipt identities")
            seen.add(id(source))
            result.append(
                _InboxPair(
                    _weak(source),
                    _weak(label),
                    _weak(receipt),
                    _weak(source.features),
                    _weak(label.targets),
                    old,
                )
            )
        if (
            len(result) != len(sources)
            or len(result) != len(labels)
            or {id(p.source()) for p in result} != {id(s) for s in sources}
            or {id(p.label()) for p in result} != {id(s) for s in labels}
        ):
            raise ValueError(
                "untracked or pending inbox payloads cannot acquire checkpoint provenance"
            )
        erased = op.erased_history
        untrained = op.untrained_history
        if op.history is None:
            applied_keys = {(receipt.episode_id, receipt.sample_id) for receipt in receipts}
            if any(tombstone.key not in applied_keys for tombstone in cursor.erased):
                raise ValueError(
                    "unenrolled untrained tombstones cannot acquire checkpoint provenance"
                )
            if len(result) != len(receipts) or {id(p.receipt()) for p in result} != {
                id(s) for s in receipts
            }:
                raise ValueError("unenrolled inbox receipts cannot acquire checkpoint provenance")
        else:
            require_inbox_partition(
                cursor,
                tuple(cast(AppliedExperience, pair.receipt()) for pair in result),
                erased,
                op.history_content_limits,
                op.pair_capacity,
                untrained=untrained,
            )
        op.inbox_erasures[id(cursor)] = erased
        op.inbox_untrained[id(cursor)] = untrained
        return tuple(result)

    @staticmethod
    def _verify_inbox(cursor, pairs, erased=(), *, untrained=(), content_limits=None, maximum=None):
        if (
            type(cursor) is not InboxCursor
            or type(pairs) is not tuple
            or type(erased) is not tuple
            or type(untrained) is not tuple
            or any(
                type(value) is not tuple
                for value in (cursor.experiences, cursor.labels, cursor.applied, cursor.erased)
            )
        ):
            raise ValueError("bound inbox verification requires exact cursor/witness tuples")
        if (
            type(cursor) is not InboxCursor
            or len(cursor.experiences) != len(pairs)
            or len(cursor.labels) != len(pairs)
            or len(cursor.applied) != len(pairs) + len(erased)
        ):
            raise ValueError("bound inbox copy inventory changed")
        if erased or untrained:
            if content_limits is None or maximum is None:
                raise ValueError(
                    "erased inbox verification requires original bounded content limits"
                )
            require_inbox_partition(
                cursor,
                tuple(pair.receipt() for pair in pairs),
                erased,
                content_limits,
                maximum,
                untrained=untrained,
            )
        for pair in pairs:
            source, label, receipt = pair.source(), pair.label(), pair.receipt()
            if (
                source is None
                or label is None
                or receipt is None
                or not any(source is item for item in cursor.experiences)
                or not any(label is item for item in cursor.labels)
                or not any(receipt is item for item in cursor.applied)
                or source.features is not pair.features()
                or label.targets is not pair.targets()
            ):
                raise ValueError(
                    "bound inbox source/label/receipt/payload identity changed or expired"
                )

    def _qualify_inbox(self, op, producer, kind, source):
        if producer is not op.runtime._inbox:
            raise ValueError("inbox graph producer differs from original current inbox")
        if kind == "inbox_capture":
            return self._original_inbox(op, source)
        if (
            kind != "inbox_materialize"
            or op.checkpoint is None
            or source is not op.checkpoint.cursor()
        ):
            raise ValueError("inbox materialization root differs from original pending cursor")
        self._verify_inbox(
            source,
            op.checkpoint.inbox,
            op.checkpoint.erased_inbox,
            untrained=op.checkpoint.untrained_inbox,
            content_limits=op.history_content_limits,
            maximum=op.pair_capacity,
        )
        op.inbox_erasures[id(source)] = op.checkpoint.erased_inbox
        op.inbox_untrained[id(source)] = op.checkpoint.untrained_inbox
        return op.checkpoint.inbox

    def _observe(self, producer, kind, stage, read, lookup):
        op = self._active
        if op is None:
            raise ValueError("graph copy requires its original active checkpoint operation")
        if stage == "before_copy":
            self._require_gates(op)
            reserve, observe = self._copies._lease_copy(op.life, op.copying)
            original = read()
            source_content = self._stamp(original.source)
            self._current(op)
            inbox = kind in ("inbox_capture", "inbox_materialize")
            records = (
                self._qualify_inbox(op, producer, kind, original.source)
                if inbox
                else self._qualify_native(op, producer, kind, original.source)
            )
            erased = op.inbox_erasures[id(original.source)] if inbox else ()
            untrained = op.inbox_untrained[id(original.source)] if inbox else ()
            if kind == "inbox_capture" and getattr(op, "original_cursor", None) is None:
                op.original_cursor, op.original_inbox_content = original.source, source_content
            op.runtime._budget._before_final_leased(observe)
            self._current(op)
            op.ledger._admission.start(op.ledger._live_records())
            for row in op.canonical:
                op.ledger._reserve_copied_row(row)
            if inbox:
                if op.history is not None:
                    for pair in records:
                        op.ledger._reserve_copied_inbox(pair.origin)
                    for record in erased:
                        op.ledger._reserve_copied_erased_inbox(record)
                    for untrained_record in untrained:
                        op.ledger._reserve_copied_untrained_inbox(untrained_record)
                size = sum(
                    op.life._port_bytes(op.life._measure, pair.features())
                    + op.life._port_bytes(op.life._measure, pair.targets())
                    for pair in records
                )
            else:
                size = self._graph_ports.payload_bytes(
                    original.source,
                    op.ledger._window_limits.max_retained_snapshots,
                    op.ledger._admission.limits.max_payload_bytes,
                )
                if size != sum(row.data.payload_bytes for row in records):
                    raise ValueError(
                        "graph replay footprint differs from verified original row bytes"
                    )
            require_tick(size, "checkpoint graph owned payload bytes")
            reserve(size)
            op.ledger._require_copy_budget(op.owner)
            self._current(op)
            self._prove_current(op)
            if self._stamp(original.source) != source_content:
                raise ValueError("admitted graph source contents changed during opaque validation")
            op.event = producer, kind, original.source, records, inbox, erased, untrained
            return
        try:
            original = read()
            # The trusted proof precedes all injected validation/inventory ports.
            copied_content = self._stamp(original.target)
            if op.event is None:
                raise ValueError("copied graph lacks its original event admission")
            expected_producer, expected_kind, source, records, inbox, erased, untrained = op.event
            if (
                producer is not expected_producer
                or kind != expected_kind
                or original.source is not source
                or original.target is None
            ):
                raise ValueError("copied graph differs from original admitted producer/root")
            self._current(op)
            if inbox:
                copied_erased = bind_erased_inbox_records(
                    erased, lookup, op.history_content_limits, op.pair_capacity
                )
                copied_untrained = bind_untrained_inbox_records(
                    untrained, lookup, op.history_content_limits, op.pair_capacity
                )
                result = self._bind_inbox(
                    original.target,
                    records,
                    lookup,
                    copied_erased,
                    op.history_content_limits,
                    op.pair_capacity,
                    untrained=copied_untrained,
                )
                self._bind_inbox_order(original.source, original.target, lookup, op.pair_capacity)
                # Order lookup is opaque too; reprove every partition afterwards.
                self._verify_inbox(
                    original.source,
                    records,
                    erased,
                    untrained=untrained,
                    content_limits=op.history_content_limits,
                    maximum=op.pair_capacity,
                )
                self._verify_inbox(
                    original.target,
                    result,
                    copied_erased,
                    untrained=copied_untrained,
                    content_limits=op.history_content_limits,
                    maximum=op.pair_capacity,
                )
                op.inboxes[id(original.target)] = original.target, result
                op.inbox_erasures[id(original.target)] = copied_erased
                op.inbox_untrained[id(original.target)] = copied_untrained
                if kind == "inbox_capture" and op.checkpoint is None:
                    op.capture_inbox = original.target
                    op.capture_inbox_content = copied_content
                elif kind == "inbox_materialize":
                    op.materialized = original.target, result
                    op.materialized_content = copied_content
                    op.materialized_erased_stamp = tuple(
                        erased_inbox_origin_stamp(record, op.history_content_limits)
                        for record in copied_erased
                    )
                    op.materialized_untrained_stamp = tuple(
                        untrained_inbox_origin_stamp(record, op.history_content_limits)
                        for record in copied_untrained
                    )
            else:
                result = self._bind_rows(op, source, original.target, records, lookup)
                op.roots[id(original.target)] = original.target, result
                if kind == "checkpoint_capture":
                    op.capture_state = original.target
                    op.capture_state_content = copied_content
                elif kind == "checkpoint_restore_state":
                    op.restore_state = original.target.state
                    op.roots[id(original.target.state)] = original.target.state, result
                elif kind == "native_restore":
                    op.restored_model, op.restored_rows = producer, result
                    op.restored_stamps = tuple(_row_stamp(row) for row in result)
                    op.restored_content = copied_content
            self._current(op)
            self._prove_current(op)
            if self._stamp(original.target) != copied_content:
                raise ValueError("actual copied graph contents changed during opaque binding ports")
        finally:
            op.event = None
            op.copying.close()
            op.copying = ExitStack()

    def _bind_rows(self, op, source, target, records, lookup):
        originals, copied = self._rows(op, source), self._rows(op, target)
        result = []
        for old_snapshot, old, snapshot in zip(originals, records, copied):
            features, targets = op.ledger._ports.payloads(snapshot)
            if (
                old.features is None
                or old.targets is None
                or snapshot is old_snapshot
                or lookup(old_snapshot) is not snapshot
                or features is old.features()
                or targets is old.targets()
                or lookup(old.features()) is not features
                or lookup(old.targets()) is not targets
            ):
                raise ValueError("graph row/payload lacks exact actual copier memo chain")
            row = _Row(
                old.data,
                old.source,
                old.label,
                old.declaration,
                _weak(snapshot),
                _weak(features),
                _weak(targets),
                old.receipt,
                True,
                old.metadata_digest,
            )
            op.ledger._verify_row(
                op.owner,
                op.runtime,
                snapshot,
                row,
                op.ledger._read_tick(op.runtime),
                lifecycle_leased=True,
            )
            result.append(row)
        return tuple(result)

    @staticmethod
    def _bind_inbox_order(source, target, lookup, maximum):
        """The actual memo preserves complete receipt/tombstone sequence order."""
        if type(source) is not InboxCursor or type(target) is not InboxCursor:
            raise ValueError("ordered inbox binding requires exact original/copied cursors")
        for name in ("applied", "erased"):
            originals, copied = getattr(source, name), getattr(target, name)
            if (
                type(originals) is not tuple
                or type(copied) is not tuple
                or len(originals) > maximum
                or len(copied) != len(originals)
            ):
                raise ValueError("copied inbox receipt/tombstone sequence bounds changed")
            if any(lookup(old) is not new for old, new in zip(originals, copied)):
                raise ValueError("copied inbox receipt/tombstone order lacks actual memo lineage")

    @staticmethod
    def _bind_inbox(
        target, records, lookup, erased=(), content_limits=None, maximum=None, *, untrained=()
    ):
        result = []
        for old in records:
            source, label, receipt = (
                lookup(old.source()),
                lookup(old.label()),
                lookup(old.receipt()),
            )
            features, targets = lookup(old.features()), lookup(old.targets())
            if (
                type(source) is not Experience
                or type(label) is not LabelArrival
                or type(receipt) is not AppliedExperience
                or source is old.source()
                or label is old.label()
                or receipt is old.receipt()
                or features is old.features()
                or targets is old.targets()
                or source.features is not features
                or label.targets is not targets
            ):
                raise ValueError("inbox copy lacks actual source/label/receipt/payload memo chain")
            result.append(
                _InboxPair(
                    _weak(source),
                    _weak(label),
                    _weak(receipt),
                    _weak(features),
                    _weak(targets),
                    old.origin,
                )
            )
        ManagedReplayCheckpoints._verify_inbox(
            target,
            tuple(result),
            erased,
            content_limits=content_limits,
            maximum=maximum,
            untrained=untrained,
        )
        return tuple(result)

    @contextmanager
    def _publication_lease(self, controller, original, prepared, pending):
        op = self._active
        if (
            op is None
            or controller is not op.controller
            or original is not op.runtime
            or op.checkpoint is None
        ):
            raise ValueError("publication requires original active restore operation")
        _require_collections(op, prepared)
        self._require_gates(op)
        if self._gate is not op.manager_gate or self._copies._gate is not op.copy_gate:
            raise ValueError("original manager/copy witness gate changed")
        with ExitStack() as stack:
            _, observe = self._copies._lease_copy(op.life, stack)
            live = op.life._registry._live()
            registry_identity, registry_total = op.life._registry._holders, op.life._registry._total
            holder_gates = []
            for _, kind, holder in live:
                if holder._payload_ready is not True:
                    raise ValueError("publication holder initialization is incomplete")
                op.life._require_holder(kind, holder)
                if (
                    holder is not controller
                    and holder is not original
                    and holder is not original._actor
                ):
                    holder_gate = holder._write_gate if kind == "candidate" else holder._gate
                    if type(holder_gate) is not LockType:
                        raise ValueError("enrolled holder requires its exact supported native lock")
                    stack.enter_context(lease_payload_lock(holder_gate, "final enrolled holder"))
                    holder_gates.append((holder, kind, holder_gate))
            if not any(kind == "candidate" and holder is prepared for _, kind, holder in live):
                raise ValueError("prepared runtime lacks actual original candidate enrollment")
            self._require_checkpoint(op, pending)
            self._current(op)
            if (
                controller._attempts != op.attempts + 1
                or len(controller._models) != op.position + 1
                or controller._models[op.position] is not prepared._candidate
            ):
                raise ValueError(
                    "publication preparation position/attempt differs from actual history"
                )
            model = op.ledger._ports.model_reference(prepared._candidate)
            learner_stamp = tuple(
                (key, id(value)) for key, value in vars(prepared._candidate).items()
            )
            if (
                model is not op.restored_model
                or op.restored_rows is None
                or op.materialized is None
            ):
                raise ValueError(
                    "publication lacks actual native restore/materialized inbox witnesses"
                )
            copy = self._copies._copies.get(id(model))
            self._copies._require_copy(copy, controller, prepared._candidate, model, op.enrollment)
            cursor, pairs = op.materialized
            erased = op.inbox_erasures[id(cursor)]
            untrained = op.inbox_untrained[id(cursor)]
            materialized_stamps = tuple(
                tuple((key, id(value)) for key, value in vars(item).items())
                for pair in pairs
                for item in (pair.source(), pair.label(), pair.receipt())
            )
            self._verify_inbox(
                cursor,
                pairs,
                erased,
                untrained=untrained,
                content_limits=op.history_content_limits,
                maximum=op.pair_capacity,
            )
            self._verify_rows(op, model.__dict__, op.restored_rows)
            op.runtime._budget._before_final_leased(observe)
            self._current(op)
            self._original_ledger()
            if (
                tuple((key, id(value)) for key, value in vars(prepared._candidate).items())
                != learner_stamp
            ):
                raise ValueError("prepared learner native bindings changed during final probes")
            self._verify_rows(op, pending.view.state, op.checkpoint.rows)
            self._verify_inbox(
                pending.view.inbox,
                op.checkpoint.inbox,
                op.checkpoint.erased_inbox,
                untrained=op.checkpoint.untrained_inbox,
                content_limits=op.history_content_limits,
                maximum=op.pair_capacity,
            )
            rows = {}
            for row in op.restored_rows:
                matches = [
                    pair
                    for pair in pairs
                    if pair.origin.source() is row.source()
                    and pair.origin.label() is row.label()
                    and pair.origin.receipt() is row.receipt()
                ]
                if len(matches) != 1:
                    raise ValueError("restored row lacks original materialized inbox lineage")
                pair = matches[0]
                rebound = _Row(
                    row.data,
                    pair.source,
                    pair.label,
                    row.declaration,
                    row.snapshot,
                    row.features,
                    row.targets,
                    pair.receipt,
                    True,
                    row.metadata_digest,
                )
                snapshot = row.snapshot()
                op.ledger._verify_row(
                    op.owner,
                    prepared,
                    snapshot,
                    rebound,
                    op.ledger._read_tick(original),
                    lifecycle_leased=True,
                )
                rows[id(snapshot)] = rebound
            anchors = dict(op.ledger._anchors)
            anchors.update(
                runtime=_weak(prepared),
                learner=_weak(prepared._candidate),
                model=_weak(model),
                inbox=_weak(prepared._inbox),
            )
            # Finish every opaque observation before pure identities/history checks.
            self._current(op)
            self._original_ledger()
            _require_collections(op, prepared)
            op.ledger._admission.start(op.ledger._live_records())
            self._prove_current(op, publication=True)
            stack.enter_context(
                lease_payload_lock(op.owner._shared._sharing._gate, "final checkpoint sharing")
            )
            self._require_gates(op, held=True)
            if (
                self._gate is not op.manager_gate
                or self._copies._gate is not op.copy_gate
                or not op.manager_gate.locked()
                or not op.copy_gate.locked()
            ):
                raise ValueError("original manager/copy gate lease changed or ended")
            if (
                op.life._registry._holders is not registry_identity
                or op.life._registry._total != registry_total
                or len(registry_identity) != len(live)
            ):
                raise ValueError("final registry membership/history changed during opaque ports")
            for holder, kind, held_gate in holder_gates:
                current_gate = holder._write_gate if kind == "candidate" else holder._gate
                if current_gate is not held_gate or not held_gate.locked():
                    raise ValueError("actual prepared/other holder gate identity or lease changed")
            self._pure_budget(op, pending)
            if (
                tuple((key, id(value)) for key, value in vars(prepared._candidate).items())
                != learner_stamp
            ):
                raise ValueError("prepared learner native bindings changed during final probes")
            if tuple(_row_stamp(row) for row in op.restored_rows) != op.restored_stamps:
                raise ValueError(
                    "restored row metadata/payload identities changed after native restore"
                )
            if (
                tuple(
                    tuple((key, id(value)) for key, value in vars(item).items())
                    for pair in pairs
                    for item in (pair.source(), pair.label(), pair.receipt())
                )
                != materialized_stamps
            ):
                raise ValueError("materialized inbox record identities changed during final probes")
            self._pure_publication(
                op, original, prepared, pending, model, cursor, pairs, live, rows
            )
            inbox_state = None
            erased_inbox_state = None
            untrained_inbox_state = None
            if op.history is not None:
                records = {}
                for pair in pairs:
                    record = ManagedInboxOrigins.rebind(
                        op.history, pair.origin, pair.source(), pair.label(), pair.receipt()
                    )
                    records[id(pair.source())] = record
                inbox_state = records, dict(records)
                erased_inbox_state = ManagedInboxOrigins.prepare_erased_handoff(op.history, erased)
                untrained_inbox_state = ManagedInboxOrigins.prepare_untrained_handoff(
                    op.history, untrained
                )
            transition = _ReplayTransition(
                op.ledger, anchors, rows, inbox_state, erased_inbox_state, untrained_inbox_state
            )
            op.published = True
            yield transition
            # Trusted controller commits plain assignments; no validation follows.

    @staticmethod
    def _pure_budget(op, pending):
        budget, view = op.runtime._budget, pending.view
        if (
            _budget_history(view.budget_state, max_metadata_bytes=op.metadata_limit)
            != op.checkpoint.budget_history
        ):
            raise ValueError("original captured budget/progress metadata changed")
        session_schema = (
            "budget",
            "clock",
            "progress",
            "started_at",
            "last_clock",
            "updates_completed",
            "replay_examples_completed",
            "hidden_width_observed",
            "peak_hidden_width_observed",
            "rejected_proposed_hidden_width",
            "checkpoint_position",
            "process_rss_sampler",
            "process_rss_segment",
        )
        budget_data = _metadata_fields(budget, session_schema)
        state = {
            key: value
            for key, value in budget_data.items()
            if key not in ("clock", "process_rss_sampler", "progress")
        }
        if budget.progress is not None and type(budget.progress) is not ToyExecutionProgress:
            raise ValueError("original cumulative progress format changed")
        state["progress_state"] = None if budget.progress is None else vars(budget.progress)
        if (
            _budget_history(state, max_metadata_bytes=op.metadata_limit)
            != op.checkpoint.budget_history
        ):
            raise ValueError("original cumulative budget/work/progress history changed")
        if (
            not (type(budget.last_clock) is int or type(budget.last_clock) is float)
            or not isfinite(budget.last_clock)
            or not (
                type(view.budget_state["last_clock"]) is int
                or type(view.budget_state["last_clock"]) is float
            )
            or not isfinite(view.budget_state["last_clock"])
            or budget.last_clock < view.budget_state["last_clock"]
        ):
            raise ValueError("original budget clock rewound after checkpoint capture")
        sampler = pending.sampler
        if sampler is None:
            if (
                pending.sampler_state is not None
                or budget.process_rss_segment is not view.budget_state["process_rss_segment"]
            ):
                raise ValueError(
                    "absent original RSS sampler acquired foreign cumulative resource state"
                )
            return
        if (
            type(sampler) is not ProcessRssSampler
            or type(pending.sampler_state) is not tuple
            or len(pending.sampler_state) != 2
        ):
            raise ValueError("original RSS checkpoint format changed")
        reader, captured = pending.sampler_state
        _segment_values(captured)
        if (
            not (type(sampler.interval_seconds) is int or type(sampler.interval_seconds) is float)
            or not isfinite(sampler.interval_seconds)
            or sampler.interval_seconds <= 0
        ):
            raise ValueError("original sampler interval requires exact numeric metadata")
        (
            original_reader,
            original_interval,
            original_stop,
            original_thread,
            baseline,
            peak,
            count,
        ) = op.sampler_history
        if (
            type(captured) is not ProcessRssSegment
            or sampler.read_rss_bytes is not reader
            or sampler.interval_seconds != captured.interval_seconds
            or type(sampler.start_bytes) is not int
            or sampler.start_bytes != captured.start_bytes
            or type(sampler.peak_bytes) is not int
            or sampler.peak_bytes < captured.peak_bytes
            or type(sampler.sample_count) is not int
            or sampler.sample_count < captured.sample_count
            or sampler.read_rss_bytes is not original_reader
            or sampler.interval_seconds != original_interval
            or sampler._stop is not original_stop
            or sampler._thread is not original_thread
            or sampler.start_bytes != baseline
            or sampler.peak_bytes < peak
            or sampler.sample_count < count
            or sampler._error is not None
            or type(sampler._stop._flag) is not bool
            or sampler._stop._flag
        ):
            raise ValueError("original RSS baseline/reader/counters changed or rewound")
        segment = budget.process_rss_segment
        _segment_values(segment)
        if (
            type(segment) is not ProcessRssSegment
            or segment.pid != captured.pid
            or segment.start_bytes != captured.start_bytes
            or segment.interval_seconds != captured.interval_seconds
            or segment.peak_bytes < captured.peak_bytes
            or segment.sample_count < captured.sample_count
            or segment.peak_bytes > sampler.peak_bytes
            or segment.sample_count > sampler.sample_count
            or budget.progress is not None
            and budget.progress.process_rss_segment is not segment
        ):
            raise ValueError(
                "original RSS budget/progress cumulative observations changed or rewound"
            )

    @staticmethod
    def _pure_publication(op, original, prepared, pending, model, cursor, pairs, live, rows):
        ledger, owner, controller = op.ledger, op.owner, op.controller
        if (
            _view_history(pending.view, max_metadata_bytes=op.metadata_limit)
            != op.checkpoint.view_history
        ):
            raise ValueError("original captured complete view history changed")
        for runtime in (original, prepared):
            if (
                _history_value(
                    tuple(sorted(runtime._attempted_ids)), max_metadata_bytes=op.metadata_limit
                )
                != _history_value(pending.view.attempted_ids, max_metadata_bytes=op.metadata_limit)
                or _history_value(
                    tuple(runtime._consolidations), max_metadata_bytes=op.metadata_limit
                )
                != _history_value(pending.view.consolidations, max_metadata_bytes=op.metadata_limit)
                or runtime._consolidation_limit != pending.view.consolidation_limit
                or runtime._stopped != pending.view.stopped
            ):
                raise ValueError(
                    "actual original/prepared candidate history differs from captured original view"
                )
        if (
            tuple(
                _history_value(row.declaration(), max_metadata_bytes=op.metadata_limit)
                for row in (op.history_records if op.history is not None else op.canonical)
            )
            != op.declaration_history
        ):
            raise ValueError("original declaration/consent/provenance metadata changed")
        if (
            _limit_values(ledger, owner, op.life, max_metadata_bytes=op.metadata_limit)
            != op.limit_values
        ):
            raise ValueError("original replay/copy/holder/owner/budget limit values changed")
        authorities = (
            ledger._admission,
            ledger._admission.limits,
            ledger._ports,
            tuple(ledger._port_functions),
            tuple(ledger._anchors.items()),
            op.life._copy_budget,
            op.life._copy_budget._limits,
            op.life._policy,
            op.life._measure,
            op.life._footprint,
            op.life._sampler,
            op.life._registry._limits,
        )
        if (
            any(a is not b for a, b in zip(authorities[:3], op.authorities[:3]))
            or any(a is not b for a, b in zip(authorities[5:], op.authorities[5:]))
            or any(a is not b for a, b in zip(authorities[3], op.authorities[3]))
            or (
                len(authorities[4]) != len(op.authorities[4])
                or any(
                    a[0] != b[0] or a[1] is not b[1]
                    for a, b in zip(authorities[4], op.authorities[4])
                )
            )
        ):
            raise ValueError("final original replay/graph/copy/lifecycle authorities changed")
        if (
            ledger._original_admission() is not ledger._admission
            or ledger._ports is not ledger._original_ports
            or any(a is not b for a, b in zip(vars(ledger._ports).values(), ledger._port_functions))
            or tuple(_row_stamp(row) for row in op.canonical) != op.original_row_stamps
            or tuple((key, id(value)) for key, value in vars(original._candidate).items())
            != op.learner_stamp
            or len(op.model.__dict__["_replay_memory"]) != len(op.originals)
            or any(a is not b for a, b in zip(op.model.__dict__["_replay_memory"], op.originals))
        ):
            raise ValueError(
                "final original row/array/source/label/receipt or admission identities changed"
            )
        expected = dict(
            runtime=original,
            actor=original._actor,
            learner=original._candidate,
            model=op.model,
            inbox=original._inbox,
            budget=original._budget,
            clock=original._inbox._clock,
            budget_policy=original._budget.budget,
            owner_policy=owner._limits,
        )
        if any(ledger._anchors[name]() is not value for name, value in expected.items()):
            raise ValueError("final original runtime/model/budget/clock/policy anchor changed")
        if (
            owner._shared._runtime is not original
            or controller._shared is not owner._shared
            or original._retired
            or original._stopped
            or prepared._retired
            or prepared._stopped
            or pending.owner is not original
            or pending.revision != original._revision
            or pending is not controller._pending.get(op.token)
            or op.checkpoint.pending() is not pending
            or pending.view is not op.checkpoint.view()
            or pending.view.state is not op.checkpoint.state()
            or pending.view.inbox is not op.checkpoint.cursor()
            or controller._build is not op.builder
            or pending.build is not op.builder
            or controller._digest is not op.probe
            or pending.state_probe is not op.probe
            or controller._policy is not op.policy
            or pending.policy is not op.policy
            or id(controller._models) != op.models_identity
            or controller._attempts != op.attempts + 1
            or len(controller._models) != op.position + 1
            or controller._models[op.position] is not prepared._candidate
            or controller._attempt_limit != op.attempt_limit
            or controller._limit != op.pending_limit
            or any(a is not b for a, b in zip(controller._models, op.initial_models))
            or controller._payload_ready is not True
            or original._payload_ready is not True
            or prepared._payload_ready is not True
            or prepared._actor is not original._actor
            or pending.actor is not original._actor
            or prepared._budget is not original._budget
            or pending.budget is not original._budget
            or pending.wall_clock is not original._budget.clock
            or pending.sampler is not original._budget.process_rss_sampler
            or pending.progress is not original._budget.progress
            or prepared._inbox._clock is not original._inbox._clock
            or pending.event_clock is not original._inbox._clock
            or prepared._payload_lineage is not ledger._lineage
            or original._payload_lineage is not ledger._lineage
            or prepared._candidate_version != ledger._version
            or original._candidate_version != ledger._version
            or prepared._base_actor_version != original._base_actor_version
            or pending.sharing_gate is not owner._shared._sharing
            or pending.resource is not owner._shared._sharing._resource
            or owner._lifecycle is not op.life
            or original._actor._payload_registry is not op.life._registry
            or op.life._registry._lifecycle is not op.life
            or ledger._lifecycle() is not op.life
            or original._inbox._draining
            or original._inbox._historical_completed_updates is not None
            or prepared._inbox._draining
            or prepared._inbox._historical_completed_updates is not None
            or prepared._inbox._budget is not original._budget
            or prepared._inbox._learner is not prepared._candidate
            or prepared._inbox._learner_version != ledger._version
            or prepared._revision != original._revision
            or prepared._attempted_ids != original._attempted_ids
            or prepared._consolidations != original._consolidations
            or prepared._consolidation_limit != original._consolidation_limit
            or prepared._inbox._last_time != cursor.last_tick
            or prepared._inbox._capacity != cursor.capacity
            or prepared._inbox._stopped != cursor.stopped
            or prepared._inbox._erased != original._inbox._erased
            or original._budget.updates_completed != cursor.completed_updates
            or _trusted_actor_stamp(original._actor)
            != (pending.view.actor_version, pending.view.actor_generation)
            or original._budget.started_at != ledger._budget_origin
            or original._budget.last_clock < ledger._last_budget_clock
            or original._inbox._clock._time < max(ledger._last_tick, pending.view.event_tick)
            or op.life._failed
            or op.life._retention_fault
            or op.life._retention_driver is not None
            and op.life._retention_driver._state in ("stopping", "stopped", "exhausted", "failed")
        ):
            raise ValueError(
                "final publication original authority/history/ready identities changed"
            )
        gate, sharing = owner._shared._sharing, pending.view.sharing
        if (
            gate._checkpointing is not True
            or gate._serving != sharing.active_serving_requests
            or gate._training != sharing.training_active
            or (gate._paused or gate._retention_hold is not None) != sharing.paused
            or gate._admitted != sharing.admitted_updates
            or tuple(sorted(gate._deferrals.items())) != sharing.deferrals
            or gate._limits is not op.life._sharing_limits
        ):
            raise ValueError("final original sharing quota/pause/deferral history changed")
        for inbox in (original._inbox, prepared._inbox):
            if inbox._event_ids != {label.event_id for label in inbox._labels.values()} | {
                item.event_id for item in inbox._erased.values() if item.event_id is not None
            }:
                raise ValueError(
                    "final original/materialized label and tombstone event indexes changed"
                )
            if not _bound(
                inbox._registration_guard, owner, ManagedExperienceOwner._require_registration
            ) or not _bound(inbox._training_guard, owner, ManagedExperienceOwner._eligible):
                raise ValueError("final original inbox consent guards changed")
            guard = ledger._copy_guard() if ledger._copy_guard is not None else None
            if guard is None:
                if inbox._payload_copy_guard is not None:
                    raise ValueError("final payload copy guard changed")
            elif not (
                inbox._payload_copy_guard is guard
                or _bound(
                    inbox._payload_copy_guard,
                    getattr(guard, "__self__", None),
                    getattr(guard, "__func__", None),
                )
            ):
                raise ValueError("final original payload copy guard changed")
        for number, kind, holder in live:
            entry = op.life._registry._holders.get(number)
            if (
                entry is None
                or entry[0] != kind
                or entry[1]() is not holder
                or holder._payload_ready is not True
            ):
                raise ValueError("final enrolled holder identity changed")
        if not any(
            number == op.enrollment and kind == "checkpoint" and holder is controller
            for number, kind, holder in live
        ):
            raise ValueError("final checkpoint enrollment changed")
        native = tuple(model.__dict__["_replay_memory"])
        if len(native) != len(rows) or any(
            rows.get(id(snapshot)) is None or rows[id(snapshot)].snapshot() is not snapshot
            for snapshot in native
        ):
            raise ValueError("final actual restored native row identities changed")
        for pair in pairs:
            source, label, receipt = pair.source(), pair.label(), pair.receipt()
            old = pair.origin
            key = old.data.key
            if (
                source is None
                or label is None
                or receipt is None
                or prepared._inbox._experiences.get(key) is not source
                or prepared._inbox._labels.get(key) is not label
                or prepared._inbox._applied.get(key) is not receipt
                or source.features is not pair.features()
                or label.targets is not pair.targets()
                or original._inbox._experiences.get(key) is not old.source()
                or original._inbox._labels.get(key) is not old.label()
                or original._inbox._applied.get(key) is not old.receipt()
                or owner._catalog.get(key) is not old.declaration()
                or key in owner._revoked_keys
                or key in original._inbox._erased
                or old.declaration().provenance.subject_id in owner._opted_out
                or not old.declaration().consent.training
                or not old.declaration().consent.replay
                or old.declaration().retention != "replay"
                or source.role != "train"
                or label.role != "train"
                or not source.permissions.training
                or not source.permissions.replay
                or source.model_version != old.data.actor_version
                or label.model_version != old.data.actor_version
                or receipt.actor_version != old.data.actor_version
                or receipt.learner_version != ledger._version
                or receipt.update_number != old.data.update_number
                or receipt.event_id != old.data.event_id
                or source.observed_at != old.data.observed_at
                or label.arrived_at != old.data.arrived_at
                or op.life._clock._time < source.observed_at
                or op.life._clock._time - source.observed_at
                > ledger._admission.limits.max_age_ticks
            ):
                raise ValueError(
                    "final original/materialized consent or receipt identities changed"
                )
            if op.life._expired(key, op.life._clock._time, op.life._last_seconds):
                raise ValueError("final original payload retention expired")
        erased = op.inbox_erasures[id(cursor)]
        untrained = op.inbox_untrained[id(cursor)]
        if erased or untrained:
            require_inbox_partition(
                cursor,
                tuple(pair.receipt() for pair in pairs),
                erased,
                op.history_content_limits,
                op.pair_capacity,
                untrained=untrained,
            )
            if (
                tuple(
                    erased_inbox_origin_stamp(record, op.history_content_limits)
                    for record in erased
                )
                != op.materialized_erased_stamp
            ):
                raise ValueError("final actual copied erased metadata changed")
            for record in erased:
                key = record.data.key
                if op.history is None:
                    raise ValueError("final erased history lacks original enrollment")
                admitted = op.history._erased_records.get(key)
                if admitted is None:
                    raise ValueError("final erased history lacks its original admitted witness")
                require_same_erased_inbox_history(admitted, record, op.history_content_limits)
                if (
                    original._inbox._applied.get(key) is not admitted.receipt()
                    or original._inbox._erased.get(key) is not admitted.tombstone()
                    or prepared._inbox._applied.get(key) is not record.receipt()
                    or prepared._inbox._erased.get(key) is not record.tombstone()
                    or key not in owner._revoked_keys
                    or key in prepared._inbox._experiences
                    or key in prepared._inbox._labels
                    or record.data.learner_version != prepared._candidate_version
                    or record.data.update_number > prepared._budget.updates_completed
                ):
                    raise ValueError("final erased receipt/tombstone/revocation identities changed")
            _require_untrained_stamp(op.materialized_untrained_stamp, op.pair_capacity)
            if (
                tuple(
                    untrained_inbox_origin_stamp(record, op.history_content_limits)
                    for record in untrained
                )
                != op.materialized_untrained_stamp
            ):
                raise ValueError("final actual copied untrained metadata changed")
            for record in untrained:
                key = record.data.key
                if op.history is None:
                    raise ValueError("final untrained history lacks original enrollment")
                admitted = op.history._untrained_records.get(key)
                if admitted is None:
                    raise ValueError("final untrained history lacks its original admitted witness")
                require_same_untrained_inbox_history(admitted, record, op.history_content_limits)
                if (
                    original._inbox._erased.get(key) is not admitted.tombstone()
                    or prepared._inbox._erased.get(key) is not record.tombstone()
                    or key not in owner._revoked_keys
                    or key in original._inbox._applied
                    or key in prepared._inbox._applied
                    or key in original._inbox._experiences
                    or key in original._inbox._labels
                    or key in prepared._inbox._experiences
                    or key in prepared._inbox._labels
                    or record.data.learner_version != prepared._candidate_version
                    or record.data.completed_updates > prepared._budget.updates_completed
                    or record.data.erased_at > cursor.last_tick
                ):
                    raise ValueError("final untrained tombstone/revocation/work identities changed")
        if len(prepared._inbox._erased) != len(erased) + len(untrained):
            raise ValueError("final erased inbox has untracked tombstones")
        if (
            len(prepared._inbox._experiences) != len(pairs)
            or len(prepared._inbox._labels) != len(pairs)
            or len(prepared._inbox._applied) != len(pairs) + len(erased)
        ):
            raise ValueError("final materialized inbox has untracked payloads")
        ledger._admission.accounting(ledger._live_records())
        ledger._require_copy_budget(owner)
