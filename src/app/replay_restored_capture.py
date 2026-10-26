"""Opt-in paid weak publication history for complete restored replay capture.

The trusted handoff prepares metadata before its plain commit. The capture
reader only checks those original weak identities and already admitted rows;
content equality never creates provenance. This first branch accepts unchanged
published current/retired pairs, not further work, chained handoffs, restoration
authority, erasure, persistence, or physical heap/RSS accounting.
"""

from contextlib import contextmanager
from dataclasses import dataclass, fields
import json
from threading import get_ident
from typing import Any, Iterator, Callable
from weakref import ReferenceType

from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.experience_inbox import ExperienceInbox
from src.app.managed_composite_sources import SCHEMAS
from src.app.managed_replay_checkpoints import (
    ManagedReplayCheckpoints,
    _history_value,
    _limit_values,
    _metadata_fields,
    _row_stamp,
    _budget_history,
    _inbox_references,
)
from src.app.managed_inbox_origins import _InboxOrigin, inbox_origin_stamp
from src.app.checkpoint_erased_inbox import (
    require_same_erased_inbox_history,
    require_erased_inbox_records,
)
from src.app.checkpoint_untrained_inbox import (
    require_same_untrained_inbox_history,
    require_untrained_inbox_records,
)
from src.app.erased_inbox_origins import erased_inbox_origin_stamp
from src.app.untrained_inbox_origins import untrained_inbox_origin_stamp
from src.app.managed_replay_origins import _Row, _weak
from src.core.replay_origin import RECORD_OVERHEAD_BYTES


@dataclass(frozen=True)
class _Publication:
    format_version: int
    references: tuple[tuple[str, ReferenceType[Any] | None], ...]
    old_rows: tuple[_Row, ...]
    current_rows: tuple[_Row, ...]
    inbox_pairs: tuple[Any, ...]
    old_erased: tuple[Any, ...]
    current_erased: tuple[Any, ...]
    old_untrained: tuple[Any, ...]
    current_untrained: tuple[Any, ...]
    metadata: tuple
    charged_bytes: int


def initialize_capture_history(manager, ledger, enabled) -> None:
    if type(enabled) is not bool:
        raise ValueError("retain_capture_history requires an exact boolean")
    capacity = 0
    if enabled:
        life = ledger._original_owner()._lifecycle
        if life is None:
            raise ValueError("retained publication history requires original bounded lifecycle")
        maximum = life._policy.holders.max_lifetime_enrollments
        if type(maximum) is not int or maximum <= 0:
            raise ValueError("retained publication history requires original holder capacity")
        capacity = min(maximum, manager._content_limits.max_nodes)
    manager._retain_capture_history = manager._original_retain_capture_history = enabled
    manager._publication_capacity = manager._original_publication_capacity = capacity
    manager._publications = {}
    manager._publication_sealed = {}
    manager._publication_storage = manager._publications, manager._publication_sealed


def require_publication_storage(manager) -> dict[int, _Publication]:
    if (
        type(manager._retain_capture_history) is not bool
        or manager._retain_capture_history is not manager._original_retain_capture_history
    ):
        raise ValueError("original publication history opt-in changed")
    capacity = manager._publication_capacity
    if (
        type(capacity) is not int
        or capacity != manager._original_publication_capacity
        or not 0 <= capacity <= 4096
        or (manager._retain_capture_history and capacity == 0)
        or (not manager._retain_capture_history and capacity != 0)
    ):
        raise ValueError("original publication history capacity changed")
    storage = manager._publication_storage
    if (
        type(storage) is not tuple
        or len(storage) != 2
        or storage[0] is not manager._publications
        or storage[1] is not manager._publication_sealed
    ):
        raise ValueError("original publication history storage changed")
    records, sealed = storage
    if (
        type(records) is not dict
        or type(sealed) is not dict
        or len(records) > capacity
        or records.keys() != sealed.keys()
    ):
        raise ValueError("original publication history exceeds its sealed bounded inventory")
    for key, record in records.items():
        if (
            type(key) is not int
            or key <= 0
            or type(record) is not _Publication
            or vars(record).keys() != {field.name for field in fields(_Publication)}
            or sealed[key] is not record
        ):
            raise ValueError("original publication history record/seal changed")
        if (
            type(record.format_version) is not int
            or record.format_version != 1
            or type(record.charged_bytes) is not int
            or record.charged_bytes < RECORD_OVERHEAD_BYTES
            or type(record.metadata) is not tuple
        ):
            raise ValueError("original publication history metadata format changed")
        if type(record.references) is not tuple or len(record.references) > 64:
            raise ValueError("publication weak reference inventory exceeds fixed bounds")
        for entry in record.references:
            if (
                type(entry) is not tuple
                or len(entry) != 2
                or type(entry[0]) is not str
                or entry[1] is not None
                and type(entry[1]) is not ReferenceType
            ):
                raise ValueError("publication requires exact named original weak references")
        for name in (
            "old_rows",
            "current_rows",
            "inbox_pairs",
            "old_erased",
            "current_erased",
            "old_untrained",
            "current_untrained",
        ):
            value = getattr(record, name)
            if type(value) is not tuple or len(value) > manager._content_limits.max_nodes:
                raise ValueError("publication retained row/inbox witnesses exceed original bounds")
    return records


def _objects(manager, owner, old, current, controller) -> dict[str, Any]:
    _schema(old, ActorShadowRuntime)
    _schema(current, ActorShadowRuntime)
    _schema(controller, CandidateCheckpointController)
    ledger, life = manager._original_ledger(), owner._lifecycle
    result = dict(
        owner=owner,
        ledger=ledger,
        life=life,
        shared=owner._shared,
        registry=life._registry,
        actor=current._actor,
        budget=current._budget,
        clock=current._inbox._clock,
        controller=controller,
        old_runtime=old,
        current_runtime=current,
        old_learner=old._candidate,
        current_learner=current._candidate,
        old_model=ledger._ports.model_reference(old._candidate),
        current_model=ledger._ports.model_reference(current._candidate),
        old_inbox=old._inbox,
        current_inbox=current._inbox,
        builder=controller._build,
        policy=controller._policy,
        probe=controller._digest,
        admission=ledger._admission,
        origin_limits=ledger._admission.limits,
        ports=ledger._ports,
        copy_budget=life._copy_budget,
        copy_limits=life._copy_budget._limits,
        retention_policy=life._policy,
        registry_limits=life._registry._limits,
        sharing=owner._shared._sharing,
        sharing_limits=owner._shared._sharing._limits,
        wall_clock=current._budget.clock,
        progress=current._budget.progress,
        sampler=life._sampler,
        resource=owner._shared._sharing._resource,
        history=ledger._history(),
        builder_source=manager._copies._builder_source,
    )
    for name, value in vars(ledger._ports).items():
        result["port." + name] = value
    result.update(
        graph_rows=manager._graph_ports.rows, graph_bytes=manager._graph_ports.payload_bytes
    )
    gates = dict(
        ledger=ledger._gate,
        owner=owner._gate,
        controller=controller._gate,
        old_runtime=old._write_gate,
        current_runtime=current._write_gate,
        actor=current._actor._read_gate,
        registry=life._registry._gate,
        time=life._time_gate,
        copy=life._copy_budget._gate,
        sampler=None if life._sampler is None else life._sampler._lock,
        sharing=owner._shared._sharing._gate,
        manager=manager._gate,
        copies=manager._copies._gate,
    )
    result.update(("gate." + name, value) for name, value in gates.items())
    return result


def _schema(value, expected) -> dict[str, Any]:
    if type(value) is not expected:
        raise ValueError("published runtime/inbox/controller requires its exact original type")
    return _metadata_fields(value, tuple(SCHEMAS[expected]))


def _inbox_metadata(manager, inbox, completed, retired):
    data = _schema(inbox, ExperienceInbox)
    result: list[tuple[Any, ...]] = []
    for name, value in data.items():
        if name == "_historical_completed_updates":
            result.append((name, completed if retired else None))
        elif name in ("_experiences", "_labels", "_applied", "_erased", "_event_ids"):
            if (
                type(value) is not (set if name == "_event_ids" else dict)
                or len(value) > inbox._capacity
            ):
                raise ValueError("published inbox history exceeds original bounded inventory")
            result.append((name, id(value), manager._stamp(value)))
        elif name in (
            "_clock",
            "_budget",
            "_learner",
            "_registration_guard",
            "_training_guard",
            "_payload_copy_guard",
        ):
            result.append((name, id(value)))
        else:
            result.append((name, value))
    return tuple(result)


def _runtime_metadata(runtime, retired):
    result = []
    for name, value in _schema(runtime, ActorShadowRuntime).items():
        if name == "_retired":
            result.append((name, retired))
        elif name in ("_attempted_ids", "_consolidations"):
            result.append((name, tuple(sorted(value)) if type(value) is set else tuple(value)))
        elif name in (
            "_actor",
            "_budget",
            "_write_gate",
            "_payload_lineage",
            "_candidate",
            "_inbox",
        ):
            result.append((name, id(value)))
        else:
            result.append((name, value))
    return tuple(result)


def _learner_metadata(learner, maximum):
    data = _metadata_fields(learner)
    if len(data) > maximum:
        raise ValueError("published learner field inventory exceeds original metadata bounds")
    # Match the existing operation's complete learner stamp: scalar policy
    # replacements and added/removed fields must invalidate publication proof.
    return tuple((name, id(value)) for name, value in data.items())


def _enrollments(objects):
    registry = objects["registry"]
    result = []
    for name, kind in (
        ("old_runtime", "candidate"),
        ("current_runtime", "candidate"),
        ("controller", "checkpoint"),
    ):
        matches = [
            number
            for number, (role, reference) in registry._holders.items()
            if role == kind and reference() is objects[name]
        ]
        if len(matches) != 1:
            raise ValueError("published original holder lacks its exact unique enrollment")
        result.append((name, matches[0]))
    return tuple(result)


def _proof(
    manager, objects, old_rows, current_rows, partitions, *, maximum=None, retired_completed=None
):
    ledger, owner, life = objects["ledger"], objects["owner"], objects["life"]
    old, current, controller = (
        objects[name] for name in ("old_runtime", "current_runtime", "controller")
    )
    row_limit = min(ledger._window_limits.max_retained_snapshots, 4096)
    if any(type(rows) is not tuple or len(rows) > row_limit for rows in (old_rows, current_rows)):
        raise ValueError("published replay row metadata exceeds original limits")
    _schema(controller, CandidateCheckpointController)
    if (
        type(controller._models) is not list
        or len(controller._models) > controller._attempt_limit
        or type(controller._pending) is not dict
        or len(controller._pending) > controller._limit
    ):
        raise ValueError("published checkpoint preparation history exceeds original limits")
    if (
        type(life._registry._holders) is not dict
        or len(life._registry._holders) > manager._publication_capacity
    ):
        raise ValueError("published holder inventory exceeds original capacity")
    completed = current._budget.updates_completed
    budget = {
        key: value
        for key, value in vars(current._budget).items()
        if key not in ("clock", "process_rss_sampler", "progress")
    }
    budget["progress_state"] = (
        None if current._budget.progress is None else vars(current._budget.progress)
    )
    sharing = owner._shared._sharing
    raw = (
        (
            "references",
            tuple(
                (name, id(value) if value is not None else None) for name, value in objects.items()
            ),
        ),
        ("enrollments", _enrollments(objects)),
        (
            "rows",
            tuple(_row_stamp(row) for row in old_rows),
            tuple(_row_stamp(row) for row in current_rows),
        ),
        ("inbox_origins", _partition_metadata(manager, partitions, current._inbox._capacity)),
        (
            "native",
            manager._stamp(vars(objects["old_model"])),
            manager._stamp(vars(objects["current_model"])),
        ),
        ("old_runtime", _runtime_metadata(old, True)),
        ("current_runtime", _runtime_metadata(current, False)),
        (
            "learners",
            _learner_metadata(old._candidate, manager._content_limits.max_nodes),
            _learner_metadata(current._candidate, manager._content_limits.max_nodes),
        ),
        (
            "old_inbox",
            _inbox_metadata(
                manager,
                old._inbox,
                completed if retired_completed is None else retired_completed,
                True,
            ),
        ),
        ("current_inbox", _inbox_metadata(manager, current._inbox, completed, False)),
        (
            "controller",
            id(controller._models),
            tuple(id(model) for model in controller._models),
            controller._attempts,
            controller._attempt_limit,
            controller._limit,
            id(controller._pending),
        ),
        (
            "registry",
            id(life._registry._holders),
            life._registry._total,
            tuple(
                (number, kind, id(reference()))
                for number, (kind, reference) in life._registry._holders.items()
            ),
        ),
        (
            "budget",
            _budget_history(budget, max_metadata_bytes=manager._content_limits.max_metadata_bytes),
        ),
        (
            "limits",
            _limit_values(
                ledger, owner, life, max_metadata_bytes=manager._content_limits.max_metadata_bytes
            ),
        ),
        ("clock", current._inbox._clock._time),
        (
            "sharing",
            id(sharing._limits),
            sharing._serving,
            sharing._admitted,
            sharing._training,
            sharing._paused,
            tuple(sorted(sharing._deferrals.items())),
        ),
    )
    return _history_value(
        raw,
        max_metadata_bytes=min(ledger._admission.limits.max_metadata_bytes, 131072)
        if maximum is None
        else maximum,
    )


def _partition_metadata(manager, partitions, maximum):
    pairs, old_erased, current_erased, old_untrained, current_untrained = partitions
    _inbox_references(pairs, maximum, manager._content_limits)
    for records in (old_erased, current_erased):
        require_erased_inbox_records(records, manager._content_limits, maximum)
    for records in (old_untrained, current_untrained):
        require_untrained_inbox_records(records, manager._content_limits, maximum)
    return (
        tuple(
            (
                id(pair),
                tuple(
                    id(reference())
                    for reference in (
                        pair.source,
                        pair.label,
                        pair.receipt,
                        pair.features,
                        pair.targets,
                    )
                ),
                _row_stamp(pair.origin)
                if type(pair.origin) is _Row
                else inbox_origin_stamp(pair.origin, manager._content_limits),
            )
            for pair in pairs
        ),
        tuple(
            tuple(erased_inbox_origin_stamp(record, manager._content_limits) for record in records)
            for records in (old_erased, current_erased)
        ),
        tuple(
            tuple(
                untrained_inbox_origin_stamp(record, manager._content_limits) for record in records
            )
            for records in (old_untrained, current_untrained)
        ),
    )


def _encoded_size(metadata, maximum):
    # _history_value outputs only tagged tuples and scalars. Validate that
    # retained shape before JSON can traverse a tampered collection subclass.
    stack = [(metadata, 0)]
    nodes = 0
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > 32768 or depth > 64:
            raise ValueError("publication normalized metadata exceeds structural bounds")
        if type(value) is tuple:
            if len(value) > 32768 - nodes - len(stack):
                raise ValueError("publication normalized metadata tuple exceeds bounds")
            stack.extend((child, depth + 1) for child in value)
        elif value is not None and type(value) not in (str, int, float, bool):
            raise ValueError("publication normalized metadata contains unsupported values")
        elif (
            type(value) is str
            and len(value) * 6 > maximum
            or type(value) is int
            and value.bit_length() > 1024
        ):
            raise ValueError("publication normalized metadata scalar exceeds bounds")
    size = 0
    encoder = json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    for chunk in encoder.iterencode(metadata):
        size += len(chunk.encode("utf8"))
        if size > maximum:
            raise ValueError("publication metadata exceeds original remaining admission allowance")
    return size


def prepare_retained_publication(manager, op, prepared, rows, live):
    records = require_publication_storage(manager)
    if not manager._retain_capture_history:
        return None
    next_records = {key: value for key, value in records.items() if _survives(value)}
    # One unchanged pair is the supported branch; never infer chained history.
    if next_records:
        raise ValueError("retained complete capture history does not support chained publication")
    remaining = (
        op.ledger._admission.limits.max_metadata_bytes
        - op.ledger._admission.accounting(op.ledger._live_records()).metadata_bytes_charged
        - RECORD_OVERHEAD_BYTES
    )
    if remaining <= 0:
        raise ValueError("publication metadata original admission allowance exhausted")
    objects = _objects(manager, op.owner, op.runtime, prepared, op.controller)
    old_rows, current_rows = tuple(op.canonical), tuple(rows.values())
    cursor, pairs = op.materialized
    partitions = (
        pairs,
        op.erased_history,
        op.inbox_erasures[id(cursor)],
        op.untrained_history,
        op.inbox_untrained[id(cursor)],
    )
    metadata = _proof(
        manager, objects, old_rows, current_rows, partitions, maximum=min(remaining, 131072)
    )
    charge = _encoded_size(metadata, remaining) + RECORD_OVERHEAD_BYTES
    references = tuple(
        (name, None if value is None else _weak(value)) for name, value in objects.items()
    )
    record = _Publication(1, references, old_rows, current_rows, *partitions, metadata, charge)
    if len(next_records) >= manager._publication_capacity:
        raise ValueError("original publication holder history capacity exhausted")
    next_records[id(objects["old_model"])] = record
    storage = next_records, dict(next_records)
    op.ledger._admission._charge(0, 0, charge, op.ledger._live_records())
    return manager, storage


def _survives(record):
    return all(reference is None or reference() is not None for _, reference in record.references)


def require_prepared_publication(value, manager) -> None:
    if (
        type(value) is not tuple
        or len(value) != 2
        or value[0] is not manager
        or type(manager) is not ManagedReplayCheckpoints
        or manager._retain_capture_history is not True
    ):
        raise ValueError("publication transition requires the original opted-in manager")
    storage = value[1]
    if (
        type(storage) is not tuple
        or len(storage) != 2
        or any(type(part) is not dict for part in storage)
        or len(storage[0]) > manager._publication_capacity
        or storage[0].keys() != storage[1].keys()
    ):
        raise ValueError("publication transition requires bounded prepared sealed storage")
    for key, record in storage[0].items():
        if (
            type(key) is not int
            or type(record) is not _Publication
            or storage[1][key] is not record
            or record.charged_bytes
            > manager._original_ledger()._admission.accounting().metadata_bytes_charged
        ):
            raise ValueError("publication transition lacks its original paid record")


def _live(record):
    if type(record.references) is not tuple or len(record.references) > 64:
        raise ValueError("publication weak reference schema changed")
    result = {}
    for entry in record.references:
        if (
            type(entry) is not tuple
            or len(entry) != 2
            or type(entry[0]) is not str
            or entry[0] in result
            or entry[1] is not None
            and type(entry[1]) is not ReferenceType
        ):
            raise ValueError("publication requires exact named original weak references")
        value = None if entry[1] is None else entry[1]()
        if entry[1] is not None and value is None:
            raise ValueError("publication original retained reference expired")
        result[entry[0]] = value
    return result


def _roles(objects, roles, old_rows, current_rows):
    enrollment = dict(_enrollments(objects))
    for prefix, rows in (("old", old_rows), ("current", current_rows)):
        model, learner = objects[prefix + "_model"], objects[prefix + "_learner"]
        appearances = roles.get(id(model), ())
        if not appearances:
            raise ValueError("published replay model lost its original candidate role")
        candidate_seen = checkpoint_seen = False
        for current, retained, holder, collection, item in appearances:
            candidate_role = (
                holder.kind == "candidate" and holder.enrollment == enrollment[prefix + "_runtime"]
            )
            checkpoint_role = (
                prefix == "current"
                and holder.kind == "checkpoint"
                and holder.enrollment == enrollment["controller"]
            )
            allowed = candidate_role or checkpoint_role
            candidate_seen = candidate_seen or candidate_role
            checkpoint_seen = checkpoint_seen or checkpoint_role
            if (
                current is not model
                or collection != "models"
                or item is not learner
                or not allowed
                or len(retained) != len(rows)
                or any(row is not record.snapshot() for row, record in zip(retained, rows))
            ):
                raise ValueError(
                    "published replay model appeared through an unqualified holder role"
                )
        if not candidate_seen or prefix == "current" and not checkpoint_seen:
            raise ValueError(
                "published replay inventory omitted an original candidate/checkpoint role"
            )


def _verify_partitions(manager, objects, partitions):
    pairs, old_erased, current_erased, old_untrained, current_untrained = partitions
    ledger, owner, old, current = (
        objects[name] for name in ("ledger", "owner", "old_runtime", "current_runtime")
    )
    _partition_metadata(manager, partitions, current._inbox._capacity)
    now = ledger._read_tick(current)
    for pair in pairs:
        origin = pair.origin
        if type(origin) is _Row:
            if origin.snapshot is None:
                raise ValueError("published old inbox original replay snapshot expired")
            ledger._verify_row(owner, old, origin.snapshot(), origin, now, lifecycle_leased=True)
        elif type(origin) is _InboxOrigin:
            declaration = ledger._source(
                owner, old, origin.source(), origin.label(), now, lifecycle_leased=True
            )
            if declaration is not origin.declaration():
                raise ValueError("published old inbox original declaration changed")
            ledger._receipt(old, origin)
            inbox_origin_stamp(origin, manager._content_limits)
        else:
            raise ValueError("published old inbox lost its original admitted pair origin")
    manager._verify_inbox(
        current._inbox._borrow_cursor(),
        pairs,
        current_erased,
        untrained=current_untrained,
        content_limits=manager._content_limits,
        maximum=current._inbox._capacity,
    )
    if len(old_erased) != len(current_erased) or len(old_untrained) != len(current_untrained):
        raise ValueError("published inbox erased/untrained partitions changed")
    for before, after in zip(old_erased, current_erased):
        require_same_erased_inbox_history(before, after, manager._content_limits)
        key = before.data.key
        if (
            old._inbox._applied.get(key) is not before.receipt()
            or old._inbox._erased.get(key) is not before.tombstone()
            or current._inbox._applied.get(key) is not after.receipt()
            or current._inbox._erased.get(key) is not after.tombstone()
            or key not in owner._revoked_keys
        ):
            raise ValueError("published erased inbox original receipt/tombstone changed")
    for before, after in zip(old_untrained, current_untrained):
        require_same_untrained_inbox_history(before, after, manager._content_limits)
        key = before.data.key
        if (
            old._inbox._erased.get(key) is not before.tombstone()
            or current._inbox._erased.get(key) is not after.tombstone()
            or key not in owner._revoked_keys
            or key in old._inbox._applied
            or key in current._inbox._applied
        ):
            raise ValueError("published untrained inbox original tombstone changed")
    if (
        len(old._inbox._experiences) != len(pairs)
        or len(old._inbox._labels) != len(pairs)
        or len(old._inbox._applied) != len(pairs) + len(old_erased)
        or len(old._inbox._erased) != len(old_erased) + len(old_untrained)
    ):
        raise ValueError("published old inbox has untracked retained records")


def _verify(manager, record, owner, ledger, runtime, roles):
    from src.app.replay_publication_updates import verify_current_success

    if verify_current_success(owner, ledger, manager, record, runtime, roles):
        return
    retained = _live(record)
    if (
        retained.get("owner") is not owner
        or retained.get("ledger") is not ledger
        or retained.get("current_runtime") is not runtime
        or retained.get("old_runtime") is None
        or retained.get("controller") is None
    ):
        raise ValueError("publication does not belong to the original current owner/ledger")
    objects = _objects(manager, owner, retained["old_runtime"], runtime, retained["controller"])
    if objects.keys() != retained.keys() or any(
        objects[name] is not value for name, value in retained.items()
    ):
        raise ValueError("publication original learner/model/policy/port/gate bindings changed")
    old, controller = objects["old_runtime"], objects["controller"]
    if (
        old._retired is not True
        or type(old._inbox._historical_completed_updates) is not int
        or old._inbox._historical_completed_updates != runtime._budget.updates_completed
        or runtime._retired is not False
        or runtime._inbox._historical_completed_updates is not None
        or controller._pending
    ):
        raise ValueError("published old/current retirement or consumed checkpoint history changed")
    if ledger._rows.keys() != {id(row.snapshot()) for row in record.current_rows} or any(
        ledger._rows[id(row.snapshot())] is not row for row in record.current_rows
    ):
        raise ValueError("published rebound current row ledger changed")
    if manager._copies._builder_source(controller._build) is not old._candidate:
        raise ValueError(
            "published checkpoint builder no longer references its original old learner"
        )
    for prefix, rows in (("old", record.old_rows), ("current", record.current_rows)):
        source = objects[prefix + "_model"]
        actual = ledger._inventory(source)
        if len(actual) != len(rows) or any(
            snapshot is not row.snapshot() for snapshot, row in zip(actual, rows)
        ):
            raise ValueError("published native replay inventory differs from actual original rows")
        now = ledger._read_tick(runtime)
        for snapshot, row in zip(actual, rows):
            ledger._verify_row(
                owner, objects[prefix + "_runtime"], snapshot, row, now, lifecycle_leased=True
            )
    _roles(objects, roles, record.old_rows, record.current_rows)
    partitions = (
        record.inbox_pairs,
        record.old_erased,
        record.current_erased,
        record.old_untrained,
        record.current_untrained,
    )
    _verify_partitions(manager, objects, partitions)
    metadata = _proof(manager, objects, record.old_rows, record.current_rows, partitions)
    if (
        metadata != record.metadata
        or _encoded_size(metadata, ledger._admission.limits.max_metadata_bytes)
        + RECORD_OVERHEAD_BYTES
        != record.charged_bytes
    ):
        raise ValueError("published complete native/inbox/history metadata changed")


@contextmanager
def lease_restored_replay_capture(owner, ledger, manager) -> Iterator[Callable[..., set[int]]]:
    """Borrow the manager gate already held by the enclosing snapshot reader."""
    require_publication_storage(manager)
    active, thread, checks = True, get_ident(), 0
    roots = [owner, ledger, manager, manager._publication_storage, manager._gate]
    try:

        def verify(entries, roles, runtime):
            nonlocal checks
            if not active or get_ident() != thread:
                raise ValueError("restored replay capture reader expired or crossed threads")
            checks += 1
            if checks > 8:
                raise ValueError("restored replay capture check allowance exhausted")
            owner, ledger, manager, storage, gate = roots
            if (
                manager._publication_storage is not storage
                or manager._gate is not gate
                or not gate.locked()
                or manager._original_ledger() is not ledger
            ):
                raise ValueError("restored replay original manager storage/gate/ledger changed")
            current = require_publication_storage(manager)
            qualified = set()
            for key, record in current.items():
                _verify(manager, record, owner, ledger, runtime, roles)
                if key != id(_live(record)["old_model"]):
                    raise ValueError("published replay original model identity key changed")
                qualified.add(key)
            requested = {id(model): rows for model, rows in entries}
            for key in qualified:
                if key not in requested:
                    raise ValueError(
                        "published retired replay model omitted from complete inventory"
                    )
                if len(requested[key]) != len(current[key].old_rows) or any(
                    witness.snapshot is None or row is not witness.snapshot()
                    for row, witness in zip(requested[key], current[key].old_rows)
                ):
                    raise ValueError("published retired replay complete inventory rows changed")
            return qualified

        yield verify
    finally:
        active = False
        roots.clear()
