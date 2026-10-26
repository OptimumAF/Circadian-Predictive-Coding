"""Seal successive admitted single-update polls without rewriting retired history.

Only a publication with both original history enrollments installs this bridge.
It borrows the original nonblocking gates and observes the existing poll; it
neither trains nor creates replay origins. Metadata is paid to the original
admission before publication. These local weak witnesses grant no restore or
persistence authority and do not support chained publications or erasure.
"""

from contextlib import contextmanager
from dataclasses import dataclass, fields
from threading import get_ident
from typing import Any, Iterator
from weakref import ReferenceType

from src.app.managed_inbox_origins import ManagedInboxOrigins, _InboxOrigin, inbox_origin_stamp
from src.app.managed_replay_checkpoints import _budget_history, _history_value, _row_stamp
from src.app.managed_replay_origins import _Row, _weak
from src.app.replay_restored_capture import (
    _encoded_size,
    _inbox_metadata,
    _learner_metadata,
    _live,
    _objects,
    _proof,
    _roles,
)
from src.core.replay_origin import RECORD_OVERHEAD_BYTES


@dataclass(frozen=True)
class _UpdateBinding:
    manager: ReferenceType[Any]
    publication: ReferenceType[Any]
    runtime: ReferenceType[Any]
    learner: ReferenceType[Any]
    model: ReferenceType[Any]
    history: ReferenceType[Any]
    rows: tuple[Any, ...]
    history_records: tuple[Any, ...]
    initial_updates: int
    native_content: str
    learner_content: tuple
    inbox_content: tuple
    metadata: tuple
    charged_bytes: int


@dataclass(frozen=True)
class _CurrentSuccess:
    receipt: ReferenceType[Any]
    rows: tuple[Any, ...]
    history_records: tuple[Any, ...]
    completed_updates: int
    native_content: str
    learner_content: tuple
    inbox_content: tuple
    metadata: tuple
    charged_bytes: int


@dataclass(frozen=True)
class _ZeroWorkPoll:
    previous_success: ReferenceType[Any]
    sharing: ReferenceType[Any]
    before: tuple
    after: tuple
    deferred_reason: str | None
    metadata: tuple
    charged_bytes: int


def _schema(value, expected):
    if type(value) is not expected or vars(value).keys() != {f.name for f in fields(expected)}:
        raise ValueError("current publication update requires its exact sealed record schema")


def _partitions(publication):
    return (
        publication.inbox_pairs,
        publication.old_erased,
        publication.current_erased,
        publication.old_untrained,
        publication.current_untrained,
    )


def _sections(metadata):
    # This selects sections of the existing tagged history format, not a new
    # digest scheme. Current-only state may advance; retired sections may not.
    if type(metadata) is not tuple or len(metadata) != 2 or metadata[0] != "tuple":
        raise ValueError("original publication normalized metadata changed")
    result = []
    for section in metadata[1]:
        if type(section) is not tuple or len(section) != 2 or section[0] != "tuple":
            raise ValueError("original publication section schema changed")
        values = section[1]
        name = values[0]
        if type(name) is not tuple or len(name) != 2 or name[0] != "str":
            raise ValueError("original publication section tag changed")
        if name[1] in ("rows", "native"):
            result.append((name, values[1]))
        elif name[1] == "current_runtime":
            # Original source/label/poll admission advances revision. All other
            # runtime fields remain bound, including consolidation history and
            # limits; training cannot legitimize an unrelated structural edit.
            runtime_fields = values[1]
            if (
                type(runtime_fields) is not tuple
                or len(runtime_fields) != 2
                or runtime_fields[0] != "tuple"
            ):
                raise ValueError("original current runtime metadata schema changed")
            result.append(
                (
                    name,
                    tuple(
                        field for field in runtime_fields[1] if field[1][0] != ("str", "_revision")
                    ),
                )
            )
        elif name[1] not in ("current_inbox", "budget", "clock", "sharing"):
            result.append(section)
    return tuple(result)


def _section(metadata, name):
    return next(part for part in metadata[1] if part[1][0] == ("str", name))


def _success_proof(proof):
    # Pause/resume and the transient training gate are real runtime operations,
    # not completed work. Keep all cumulative sharing counters and quotas.
    result = []
    for section in proof[1]:
        values = section[1]
        if values[0] == ("str", "sharing"):
            section = ("tuple", values[:4] + values[6:])
        result.append(section)
    return "tuple", tuple(result)


def _pure_proof(manager, objects, publication, rows, completed):
    return _proof(
        manager,
        objects,
        publication.old_rows,
        rows,
        _partitions(publication),
        retired_completed=completed,
    )


def _paid(ledger, raw, *, proof=None):
    remaining = (
        ledger._admission.limits.max_metadata_bytes
        - ledger._admission.accounting(ledger._live_records()).metadata_bytes_charged
        - RECORD_OVERHEAD_BYTES
    )
    if remaining <= 0:
        raise ValueError("current update original metadata allowance exhausted")
    metadata = _history_value(raw, max_metadata_bytes=min(remaining, 131072))
    if proof is not None:
        # The existing proof is already bounded tagged metadata. Do not expand
        # its tags through a second normalization traversal.
        metadata = ("tuple", (metadata, _success_proof(proof)))
    charge = _encoded_size(metadata, remaining) + RECORD_OVERHEAD_BYTES
    ledger._admission._charge(0, 0, charge, ledger._live_records())
    return metadata, charge


def _history_content(manager, records):
    if type(records) is not tuple or len(records) > manager._content_limits.max_nodes:
        raise ValueError("current publication inbox history exceeds original bounds")
    return tuple(
        (id(record), inbox_origin_stamp(record, manager._content_limits)) for record in records
    )


def _contents(manager, runtime, model):
    inbox = tuple(
        field
        for field in _inbox_metadata(
            manager, runtime._inbox, runtime._budget.updates_completed, False
        )
        if field[0] != "_draining"
    )
    return (
        manager._stamp(vars(model)),
        _learner_metadata(runtime._candidate, manager._content_limits.max_nodes),
        inbox,
    )


def _work_content(manager, runtime):
    budget = runtime._budget
    state = {
        name: value
        for name, value in vars(budget).items()
        if name not in ("clock", "process_rss_sampler", "progress")
    }
    state["progress_state"] = None if budget.progress is None else vars(budget.progress)
    return _budget_history(state, max_metadata_bytes=manager._content_limits.max_metadata_bytes)


def prepare_update_binding(manager, op, prepared, retained_publication, inbox_state):
    """Prepare paid baseline before the existing final publication barrier."""
    if retained_publication is None or op.history is None:
        return None
    if op.ledger._history() is not op.history or op.ledger._inbox_history_birth() is not op.history:
        raise ValueError("current update binding requires original birth inbox history")
    if inbox_state is None or any(
        _partitions(record)[i]
        for record in retained_publication[1][0].values()
        for i in range(1, 5)
    ):
        # Preserve the previously supported unchanged-publication branch.
        # Continued erased/untrained histories need their own original seam.
        return None
    records = retained_publication[1][0]
    if len(records) != 1 or manager._update_binding is not None:
        raise ValueError("current update continuation requires one original publication")
    publication = next(iter(records.values()))
    native, learner, inbox = _contents(manager, prepared, op.restored_model)
    history_records = tuple(inbox_state[0].values())
    raw = (
        "publication_update_baseline",
        id(publication),
        id(prepared),
        prepared._budget.updates_completed,
        native,
        learner,
        inbox,
        _history_content(manager, history_records),
        tuple(_row_stamp(row) for row in publication.current_rows),
    )
    metadata, charge = _paid(op.ledger, raw)
    binding = _UpdateBinding(
        _weak(manager),
        _weak(publication),
        _weak(prepared),
        _weak(prepared._candidate),
        _weak(op.restored_model),
        _weak(op.history),
        publication.current_rows,
        history_records,
        prepared._budget.updates_completed,
        native,
        learner,
        inbox,
        metadata,
        charge,
    )
    return manager, binding


def require_prepared_binding(value, manager, ledger):
    if type(value) is not tuple or len(value) != 2 or value[0] is not manager:
        raise ValueError("current update transition requires its original manager")
    binding = value[1]
    _schema(binding, _UpdateBinding)
    for name in ("manager", "publication", "runtime", "learner", "model", "history"):
        if type(getattr(binding, name)) is not ReferenceType:
            raise ValueError("current update transition requires original weak references")
    _witnesses(binding.rows, _Row, manager)
    _witnesses(binding.history_records, _InboxOrigin, manager)
    if (
        binding.manager() is not manager
        or binding.history() is not ledger._history()
        or binding.publication() is None
        or binding.runtime() is None
        or type(binding.initial_updates) is not int
        or binding.initial_updates < 0
        or _encoded_size(binding.metadata, ledger._admission.limits.max_metadata_bytes)
        + RECORD_OVERHEAD_BYTES
        != binding.charged_bytes
    ):
        raise ValueError("current update transition lacks original paid baseline")


def _witnesses(records, expected, manager):
    if (
        type(records) is not tuple
        or len(records) > manager._content_limits.max_nodes
        or any(type(record) is not expected for record in records)
    ):
        raise ValueError("current update requires bounded exact original witness tuples")


def _binding(ledger):
    binding = ledger._capture_update_binding
    if binding is not ledger._original_capture_update_binding:
        raise ValueError("original current update binding changed")
    if binding is None:
        return None
    _schema(binding, _UpdateBinding)
    for name in ("manager", "publication", "runtime", "learner", "model", "history"):
        reference = getattr(binding, name)
        if type(reference) is not ReferenceType or reference() is None:
            raise ValueError("original current update weak binding expired or changed")
    manager = binding.manager()
    if (
        manager._update_binding is not binding
        or manager._original_ledger() is not ledger
        or ledger._history() is not binding.history()
        or ledger._original_owner()._shared._runtime is not binding.runtime()
        or binding.runtime()._candidate is not binding.learner()
        or manager._publications.get(id(_live(binding.publication())["old_model"]))
        is not binding.publication()
    ):
        raise ValueError("original current update publication/owner bindings changed")
    require_prepared_binding((manager, binding), manager, ledger)
    return binding


def _success(manager):
    current = manager._current_success
    storage = manager._current_success_storage
    if (
        type(storage) is not tuple
        or len(storage) != 2
        or storage[0] is not current
        or storage[1] is not manager._current_success_sealed
        or current is not manager._current_success_sealed
    ):
        raise ValueError("original current success sealed storage changed")
    if current is not None:
        _schema(current, _CurrentSuccess)
        _witnesses(current.rows, _Row, manager)
        _witnesses(current.history_records, _InboxOrigin, manager)
        if (
            type(current.receipt) is not ReferenceType
            or current.receipt() is None
            or type(current.completed_updates) is not int
            or type(current.native_content) is not str
            or any(
                type(value) is not tuple
                for value in (
                    current.learner_content,
                    current.inbox_content,
                    current.metadata,
                )
            )
        ):
            raise ValueError("current success requires its original receipt and native work count")
    return current


def _objects_bound(binding, ledger):
    manager, publication = binding.manager(), binding.publication()
    live = _live(publication)
    objects = _objects(
        manager,
        ledger._original_owner(),
        live["old_runtime"],
        binding.runtime(),
        live["controller"],
    )
    if objects.keys() != live.keys() or any(
        objects[name] is not value for name, value in live.items()
    ):
        raise ValueError("current update original publication bindings changed")
    return objects


def _immutable(manager, binding, objects, rows):
    publication = binding.publication()
    old, current = objects["old_runtime"], objects["current_runtime"]
    if (
        old._retired is not True
        or old._inbox._historical_completed_updates != binding.initial_updates
        or current._retired is not False
        or current._inbox._historical_completed_updates is not None
        or objects["controller"]._pending
    ):
        raise ValueError("original retired/current publication history changed")
    proof = _pure_proof(manager, objects, publication, rows, binding.initial_updates)
    if _sections(proof) != _sections(publication.metadata):
        raise ValueError("immutable retired native/inbox/policy/publication history changed")
    return proof


def _retired_origins(binding, ledger, objects, *, lifecycle_leased=False):
    manager, old = binding.manager(), objects["old_runtime"]
    owner, now = ledger._original_owner(), ledger._read_tick(binding.runtime())
    for pair in binding.publication().inbox_pairs:
        origin = pair.origin
        declaration = ledger._source(
            owner, old, origin.source(), origin.label(), now, lifecycle_leased=lifecycle_leased
        )
        if declaration is not origin.declaration():
            raise ValueError("retired original inbox declaration changed")
        ledger._receipt(old, origin)
        if type(origin) is _InboxOrigin:
            inbox_origin_stamp(origin, manager._content_limits)


def _prefix(manager, binding, previous=None):
    history = binding.history()
    history._state()
    records = binding.history_records if previous is None else previous.history_records
    for record in records:
        source = record.source()
        if source is None or history._records.get(id(source)) is not record:
            raise ValueError("original committed inbox history prefix changed")
    if _history_content(manager, records) != _history_content(
        manager, tuple(history._records[id(record.source())] for record in records)
    ):
        raise ValueError("original committed inbox witness changed")


def _complete_inbox(manager, runtime, records):
    """Use the existing complete partition check, not known-row sampling."""
    _witnesses(records, _InboxOrigin, manager)
    inbox = runtime._inbox
    manager._verify_inbox(inbox._borrow_cursor(), records)
    if inbox._erased or inbox._event_ids != {record.label().event_id for record in records}:
        raise ValueError("current success has untracked erased or label-event membership")


def _started_inbox(binding, origin, previous):
    # The original observer has just enrolled THIS pending pair. Other pending
    # ingress must not be silently promoted to complete committed history by it.
    inbox = binding.runtime()._inbox
    originals = binding.history_records if previous is None else previous.history_records
    sources = {record.data.key: record.source() for record in originals}
    labels = {record.data.key: record.label() for record in originals}
    receipts = {record.data.key: record.receipt() for record in originals}
    sources[origin.source.key] = origin.source
    labels[origin.source.key] = origin.label
    for actual, expected in (
        (inbox._experiences, sources),
        (inbox._labels, labels),
        (inbox._applied, receipts),
    ):
        if actual.keys() != expected.keys() or any(
            actual[key] is not value for key, value in expected.items()
        ):
            raise ValueError("current update has an unwitnessed additional inbox member")
    if inbox._erased or inbox._event_ids != {label.event_id for label in labels.values()}:
        raise ValueError("current update has untracked erased or label-event membership")


def _stored_proof(success):
    metadata = success.metadata
    if (
        type(metadata) is not tuple
        or len(metadata) != 2
        or metadata[0] != "tuple"
        or type(metadata[1]) is not tuple
        or len(metadata[1]) != 2
    ):
        raise ValueError("previous current success metadata schema changed")
    return metadata[1][1]


def _history_metadata(binding, success):
    return _history_value(
        (
            "publication_current_success",
            id(binding),
            id(success.receipt()),
            success.completed_updates,
            _history_content(binding.manager(), success.history_records),
        ),
        max_metadata_bytes=131072,
    )


def _prior_inbox(manager, runtime, previous):
    """Keep the whole committed prefix while admitting new pending ingress."""
    inbox = runtime._inbox
    actual = {
        field[0]: field
        for field in _inbox_metadata(manager, inbox, runtime._budget.updates_completed, False)
        if field[0] != "_draining"
    }
    for field in previous.inbox_content:
        name = field[0]
        current = actual.get(name)
        if name in ("_experiences", "_labels", "_applied", "_erased", "_event_ids"):
            valid = current is not None and current[:2] == field[:2]
        elif name == "_last_time":
            valid = current is not None and type(current[1]) is int and current[1] >= field[1]
        else:
            valid = current == field
        if not valid:
            raise ValueError("previous current inbox identity/policy/time changed")
    if len(actual) != len(previous.inbox_content):
        raise ValueError("previous current inbox field schema changed")
    receipts = {record.data.key: record.receipt() for record in previous.history_records}
    if inbox._applied.keys() != receipts.keys() or any(
        inbox._applied[key] is not value for key, value in receipts.items()
    ):
        raise ValueError("previous current committed receipt membership changed")
    for record in previous.history_records:
        if (
            inbox._experiences.get(record.data.key) is not record.source()
            or inbox._labels.get(record.data.key) is not record.label()
        ):
            raise ValueError("previous current committed source/label prefix changed")
    if inbox._erased or inbox._event_ids != {label.event_id for label in inbox._labels.values()}:
        raise ValueError("previous current inbox has untracked erased or event membership")


def _prior_inbox_section(manager, runtime, previous):
    # Reconstruct the PAID prior section from its separately retained complete
    # inbox fields. New ingress may change current buffers, not past evidence.
    fields_by_name = {field[0]: field for field in previous.inbox_content}
    fields_by_name["_draining"] = ("_draining", False)
    names = tuple(
        field[0]
        for field in _inbox_metadata(
            manager, runtime._inbox, runtime._budget.updates_completed, False
        )
    )
    if fields_by_name.keys() != set(names):
        raise ValueError("previous current paid inbox field schema changed")
    return _history_value(
        ("current_inbox", tuple(fields_by_name[name] for name in names)), max_metadata_bytes=131072
    )


def _monotone_position(proof, stored):
    actual_clock, prior_clock = _section(proof, "clock")[1][1], _section(stored, "clock")[1][1]
    if (
        actual_clock[0] != "int"
        or type(actual_clock[1]) is not int
        or actual_clock[1] < prior_clock[1]
    ):
        raise ValueError("previous current logical clock was rewound")
    actual_revision = next(
        field[1][1]
        for field in _section(proof, "current_runtime")[1][1][1]
        if field[1][0] == ("str", "_revision")
    )
    prior_revision = next(
        field[1][1]
        for field in _section(stored, "current_runtime")[1][1][1]
        if field[1][0] == ("str", "_revision")
    )
    if (
        actual_revision[0] != "int"
        or type(actual_revision[1]) is not int
        or actual_revision[1] < prior_revision[1]
    ):
        raise ValueError("previous current owner revision was rewound")


def _share_position(sharing):
    reasons = ("paused", "serving_active", "training_active", "work_quota", "resource_unavailable")
    deferrals = sharing._deferrals
    if (
        type(deferrals) is not dict
        or len(deferrals) > len(reasons)
        or any(
            type(key) is not str or key not in reasons or type(value) is not int or value <= 0
            for key, value in deferrals.items()
        )
    ):
        raise ValueError("original sharing deferral inventory changed")
    return (
        id(sharing._limits),
        sharing._serving,
        sharing._admitted,
        tuple(sorted(deferrals.items())),
    )


def _deferral_delta(before, after, reason):
    reasons = ("paused", "serving_active", "training_active", "work_quota", "resource_unavailable")
    if (
        type(before) is not tuple
        or type(after) is not tuple
        or len(before) != 4
        or len(after) != 4
        or before[:3] != after[:3]
        or type(before[3]) is not tuple
        or type(after[3]) is not tuple
        or reason is not None
        and (type(reason) is not str or reason not in reasons)
    ):
        raise ValueError("zero-work poll changed sharing admissions or its original reason")
    for position in (before, after):
        if (
            any(type(value) is not int or value < 0 for value in position[:3])
            or len(position[3]) > len(reasons)
            or any(
                type(entry) is not tuple
                or len(entry) != 2
                or type(entry[0]) is not str
                or entry[0] not in reasons
                or type(entry[1]) is not int
                or entry[1] <= 0
                for entry in position[3]
            )
            or tuple(sorted(dict(position[3]).items())) != position[3]
        ):
            raise ValueError("zero-work sharing position exceeds its exact bounded schema")
    expected = dict(before[3])
    if reason is not None:
        expected[reason] = expected.get(reason, 0) + 1
    if tuple(sorted(expected.items())) != after[3]:
        raise ValueError("zero-work poll differs from its single actual deferral")


def _zero_metadata(binding, previous, sharing, before, after, reason):
    return (
        "publication_zero_work_poll",
        id(binding),
        id(previous),
        id(sharing),
        previous.completed_updates,
        before,
        after,
        reason,
    )


def _zero_poll(binding, previous):
    manager = binding.manager()
    current, storage = manager._zero_work_poll, manager._zero_work_poll_storage
    if (
        type(storage) is not tuple
        or len(storage) != 2
        or storage[0] is not current
        or storage[1] is not manager._zero_work_poll_sealed
        or current is not manager._zero_work_poll_sealed
    ):
        raise ValueError("original zero-work observation sealed storage changed")
    if current is None:
        return None
    _schema(current, _ZeroWorkPoll)
    sharing = binding.manager()._original_ledger()._original_owner()._shared._sharing
    if (
        previous is None
        or type(current.previous_success) is not ReferenceType
        or current.previous_success() is not previous
        or type(current.sharing) is not ReferenceType
        or current.sharing() is not sharing
    ):
        raise ValueError("zero-work observation lost original success/sharing bindings")
    _deferral_delta(current.before, current.after, current.deferred_reason)
    raw = _zero_metadata(
        binding, previous, sharing, current.before, current.after, current.deferred_reason
    )
    stored = _stored_proof(current)
    if _sections(stored) != _sections(binding.publication().metadata) or _sections(
        _stored_proof(previous)
    ) != _sections(binding.publication().metadata):
        raise ValueError("zero-work immutable publication metadata changed")
    for name in ("rows", "native", "budget"):
        if _section(stored, name) != _section(_stored_proof(previous), name):
            raise ValueError("zero-work original row/native/work metadata changed")
    if _section(stored, "sharing") != _history_value(
        ("sharing", *current.after), max_metadata_bytes=131072
    ):
        raise ValueError("zero-work paid sharing position changed")
    if (
        _section(_stored_proof(previous), "current_inbox")
        != _prior_inbox_section(binding.manager(), binding.runtime(), previous)
        or _history_metadata(binding, previous) != previous.metadata[1][0]
        or _encoded_size(
            previous.metadata, manager._original_ledger()._admission.limits.max_metadata_bytes
        )
        + RECORD_OVERHEAD_BYTES
        != previous.charged_bytes
    ):
        raise ValueError("zero-work original completed inbox/history/payment changed")
    if (
        _history_value(raw, max_metadata_bytes=131072) != current.metadata[1][0]
        or _encoded_size(
            current.metadata, manager._original_ledger()._admission.limits.max_metadata_bytes
        )
        + RECORD_OVERHEAD_BYTES
        != current.charged_bytes
    ):
        raise ValueError("zero-work observation original paid metadata changed")
    return current


def _sharing_position(proof, stored, *, started, observation):
    actual, expected = _section(_success_proof(proof), "sharing"), _section(stored, "sharing")
    if observation is not None:
        expected = _history_value(("sharing", *observation.after), max_metadata_bytes=131072)
    if started:
        values = expected[1]
        admitted = values[3]
        if admitted[0] != "int" or type(admitted[1]) is not int:
            raise ValueError("previous current sharing admission metadata changed")
        expected = ("tuple", values[:3] + (("int", admitted[1] + 1),) + values[4:])
    if actual != expected:
        raise ValueError("previous current cumulative sharing admission changed")


def _latest_baseline(binding, ledger, objects, previous, *, started):
    manager, runtime = binding.manager(), binding.runtime()
    if (
        runtime._budget.updates_completed != previous.completed_updates
        or previous.completed_updates <= binding.initial_updates
    ):
        raise ValueError("previous current success work count changed")
    native, learner, _ = _contents(manager, runtime, binding.model())
    if native != previous.native_content or learner != previous.learner_content:
        raise ValueError("previous current native or complete learner policy changed")
    _prefix(manager, binding, previous)
    _prior_inbox(manager, runtime, previous)
    if ledger._rows.keys() != {id(row.snapshot()) for row in previous.rows} or any(
        ledger._rows[id(row.snapshot())] is not row for row in previous.rows
    ):
        raise ValueError("previous current original row membership changed")
    proof = _immutable(manager, binding, objects, previous.rows)
    stored = _stored_proof(previous)
    if _sections(stored) != _sections(binding.publication().metadata):
        raise ValueError("previous current paid immutable publication metadata changed")
    for name in ("rows", "native", "budget"):
        if _section(proof, name) != _section(stored, name):
            raise ValueError("previous current row/work metadata changed")
    if _section(stored, "current_inbox") != _prior_inbox_section(manager, runtime, previous):
        raise ValueError("previous current paid inbox metadata changed")
    _monotone_position(proof, stored)
    _sharing_position(proof, stored, started=started, observation=_zero_poll(binding, previous))
    if (
        _history_metadata(binding, previous) != previous.metadata[1][0]
        or _encoded_size(previous.metadata, ledger._admission.limits.max_metadata_bytes)
        + RECORD_OVERHEAD_BYTES
        != previous.charged_bytes
    ):
        raise ValueError("previous current paid complete history metadata changed")


def _baseline(binding, ledger, objects, previous, *, started=False):
    manager, runtime = binding.manager(), binding.runtime()
    if _success(manager) is not previous:
        raise ValueError("previous current success changed during its original poll")
    if previous is not None:
        _latest_baseline(binding, ledger, objects, previous, started=started)
        return
    if runtime._budget.updates_completed != binding.initial_updates:
        raise ValueError("original current baseline work count changed")
    native, learner, _ = _contents(manager, runtime, binding.model())
    if native != binding.native_content or learner != binding.learner_content:
        raise ValueError("prior current native or complete learner policy changed")
    _prefix(manager, binding)
    proof = _immutable(manager, binding, objects, binding.rows)
    if _section(proof, "budget") != _section(binding.publication().metadata, "budget"):
        raise ValueError("prior current original cumulative budget changed")
    raw = (
        "publication_update_baseline",
        id(binding.publication()),
        id(runtime),
        binding.initial_updates,
        binding.native_content,
        binding.learner_content,
        binding.inbox_content,
        _history_content(manager, binding.history_records),
        tuple(_row_stamp(row) for row in binding.rows),
    )
    if _history_value(raw, max_metadata_bytes=131072) != binding.metadata:
        raise ValueError("original current baseline row/inbox metadata changed")


class _UpdateAccess:
    def __init__(self, ledger, binding, previous):
        self.ledger, self.binding = ledger, binding
        self.previous = previous
        # Existing frozen field values are immutable metadata/weak witnesses.
        # Pin their identities while native eviction may legitimately expire a
        # prior row; never restamp that evicted native row after the update.
        self.previous_fields = () if previous is None else tuple(vars(previous).items())
        self.observation = _zero_poll(binding, previous)
        self.observation_fields = (
            () if self.observation is None else tuple(vars(self.observation).items())
        )
        runtime = binding.runtime()
        self.poll_contents = _contents(binding.manager(), runtime, binding.model())
        self.poll_work = _work_content(binding.manager(), runtime)
        self.poll_revision = runtime._revision
        self.sharing_before = _share_position(ledger._original_owner()._shared._sharing)
        self.thread, self.active = get_ident(), True
        self.provisional: Any = None

    def _require(self):
        if not self.active or get_ident() != self.thread:
            raise ValueError("current update observation expired or crossed threads")
        if _success(self.binding.manager()) is not self.previous:
            raise ValueError("previous current success changed during its original poll")
        if self.previous is not None and any(
            getattr(self.previous, name) is not value for name, value in self.previous_fields
        ):
            raise ValueError("previous current success fields changed during its original poll")
        if (
            _zero_poll(self.binding, self.previous) is not self.observation
            or self.observation is not None
            and any(
                getattr(self.observation, name) is not value
                for name, value in self.observation_fields
            )
        ):
            raise ValueError("previous zero-work observation changed during its original poll")

    def _completed(self, read):
        self._require()
        binding = self.binding
        origin = read()
        runtime, manager = binding.runtime(), binding.manager()
        if (
            self.provisional is not None
            or origin.learner is not binding.learner()
            or origin.receipt is None
            or runtime._inbox._applied.get(origin.source.key) is not origin.receipt
            or origin.completed_updates
            != (
                binding.initial_updates
                if self.previous is None
                else self.previous.completed_updates
            )
            + 1
        ):
            raise ValueError("current success requires one actual original completion")
        self.provisional = (
            _weak(origin.receipt),
            origin.completed_updates,
            _contents(manager, runtime, binding.model()),
            _work_content(manager, runtime),
        )

    def observe(self, stage, read):
        self._require()
        if stage == "completed":
            # The native result is frozen BEFORE the base observer calls any
            # opaque model-reference port. A later mutation cannot become proof.
            self._completed(read)
        self.ledger._update(stage, read)
        self._require()
        if stage == "started":
            objects = _objects_bound(self.binding, self.ledger)
            self._require()
            _baseline(self.binding, self.ledger, objects, self.previous, started=True)
            _started_inbox(self.binding, read(), self.previous)
            self._require()
        elif stage not in ("completed",):
            self.provisional = None

    def finish(self, runtime, poll):
        self._require()
        if not poll.updates:
            if self.previous is not None:
                self._finish_empty(runtime, poll)
            return
        binding, ledger = self.binding, self.ledger
        manager = binding.manager()
        if self.provisional is None or len(poll.updates) != 1 or runtime is not binding.runtime():
            raise ValueError("current success lacks its original single completed poll")
        receipt, completed, contents, work_content = self.provisional
        if poll.updates[0] is not receipt() or runtime._budget.updates_completed != completed:
            raise ValueError("current success original receipt/work changed")
        objects = _objects_bound(binding, ledger)
        records = ManagedInboxOrigins.verify(
            binding.history(), ledger, ledger._original_owner(), runtime, ledger._read_tick(runtime)
        )
        _complete_inbox(manager, runtime, records)
        _retired_origins(binding, ledger, objects)
        rows = tuple(ledger._rows.values())
        proof = _immutable(manager, binding, objects, rows)
        native, learner, inbox = contents
        if (
            _contents(manager, runtime, binding.model()) != contents
            or _work_content(manager, runtime) != work_content
        ):
            raise ValueError("current native/inbox/policy changed after first completion")
        history_content = _history_content(manager, records)
        self._require()
        metadata, charge = _paid(
            ledger,
            (
                "publication_current_success",
                id(binding),
                id(receipt()),
                completed,
                history_content,
            ),
            proof=proof,
        )
        # Charging is trusted local accounting. Recheck the complete pure state
        # after every opaque observation and before the single seal publication.
        if (
            _contents(manager, runtime, binding.model()) != contents
            or _work_content(manager, runtime) != work_content
            or _history_content(manager, binding.history()._state()) != history_content
            or _pure_proof(manager, objects, binding.publication(), rows, binding.initial_updates)
            != proof
            or _success(manager) is not self.previous
        ):
            raise ValueError("current success changed before final pure seal")
        _complete_inbox(manager, runtime, records)
        self._require()
        success = _CurrentSuccess(
            receipt, rows, records, completed, native, learner, inbox, metadata, charge
        )
        manager._current_success = manager._current_success_sealed = success
        manager._current_success_storage = success, success
        manager._zero_work_poll = manager._zero_work_poll_sealed = None
        manager._zero_work_poll_storage = (None, None)

    def _empty_contents(self, runtime):
        manager = self.binding.manager()
        actual = _contents(manager, runtime, self.binding.model())
        before = self.poll_contents
        stable_before = tuple(field for field in before[2] if field[0] != "_last_time")
        stable_after = tuple(field for field in actual[2] if field[0] != "_last_time")
        before_time = next(field[1] for field in before[2] if field[0] == "_last_time")
        after_time = next(field[1] for field in actual[2] if field[0] == "_last_time")
        if (
            actual[:2] != before[:2]
            or stable_before != stable_after
            or type(after_time) is not int
            or after_time < before_time
            or runtime._revision != self.poll_revision + 1
            or _work_content(manager, runtime) != self.poll_work
        ):
            raise ValueError("zero-work poll changed native/inbox/receipt/work state")
        return actual

    def _finish_empty(self, runtime, poll):
        binding, ledger, previous = self.binding, self.ledger, self.previous
        manager = binding.manager()
        objects = _objects_bound(binding, ledger)
        _retired_origins(binding, ledger, objects)
        contents = self._empty_contents(runtime)
        proof = _immutable(manager, binding, objects, previous.rows)
        for name in ("rows", "native", "budget"):
            if _section(proof, name) != _section(_stored_proof(previous), name):
                raise ValueError("zero-work poll changed original row/native/work evidence")
        history_metadata = _history_metadata(binding, previous)
        if history_metadata != previous.metadata[1][0]:
            raise ValueError("zero-work poll changed original committed history")
        sharing = ledger._original_owner()._shared._sharing
        after = _share_position(sharing)
        _deferral_delta(self.sharing_before, after, poll.deferred_reason)
        self._require()
        metadata, charge = _paid(
            ledger,
            _zero_metadata(
                binding, previous, sharing, self.sharing_before, after, poll.deferred_reason
            ),
            proof=proof,
        )
        if (
            self._empty_contents(runtime) != contents
            or _share_position(sharing) != after
            or _pure_proof(
                manager, objects, binding.publication(), previous.rows, binding.initial_updates
            )
            != proof
            or _history_metadata(binding, previous) != history_metadata
        ):
            raise ValueError("zero-work observation changed before final pure seal")
        self._require()
        observation = _ZeroWorkPoll(
            _weak(previous),
            _weak(sharing),
            self.sharing_before,
            after,
            poll.deferred_reason,
            metadata,
            charge,
        )
        manager._zero_work_poll = manager._zero_work_poll_sealed = observation
        manager._zero_work_poll_storage = observation, observation

    def close(self):
        self.active = False
        self.ledger = self.binding = self.provisional = self.previous = None
        self.previous_fields = ()
        self.observation = self.poll_contents = self.poll_work = None
        self.observation_fields = self.sharing_before = ()


@contextmanager
def lease_publication_update(ledger) -> Iterator[Any]:
    binding = _binding(ledger)
    if binding is None:
        yield None
        return
    manager = binding.manager()
    # Existing nonblocking gates: partial acquisition unwinds automatically.
    with manager._copies._exclusive(), manager._exclusive():
        if _binding(ledger) is not binding:
            raise ValueError("original publication update binding changed during admission")
        previous = _success(manager)
        access = _UpdateAccess(ledger, binding, previous)
        try:
            objects = _objects_bound(binding, ledger)
            access._require()
            _baseline(binding, ledger, objects, previous)
            yield access
        finally:
            access.close()


def verify_current_success(owner, ledger, manager, publication, runtime, roles):
    success = _success(manager)
    if success is None:
        return False
    binding = _binding(ledger)
    if (
        binding is None
        or binding.manager() is not manager
        or binding.publication() is not publication
        or runtime is not binding.runtime()
    ):
        raise ValueError("current success lost original publication/update binding")
    objects = _objects_bound(binding, ledger)
    if owner is not objects["owner"] or success.completed_updates <= binding.initial_updates:
        raise ValueError("current success original owner/work changed")
    if runtime._budget.updates_completed != success.completed_updates:
        raise ValueError("current success original completed work changed")
    history = ManagedInboxOrigins.verify(
        binding.history(), ledger, owner, runtime, ledger._read_tick(runtime), lifecycle_leased=True
    )
    _complete_inbox(manager, runtime, history)
    _retired_origins(binding, ledger, objects, lifecycle_leased=True)
    if len(history) != len(success.history_records) or any(
        a is not b for a, b in zip(history, success.history_records)
    ):
        raise ValueError("current success complete inbox history changed")
    if ledger._rows.keys() != {id(row.snapshot()) for row in success.rows} or any(
        ledger._rows[id(row.snapshot())] is not row for row in success.rows
    ):
        raise ValueError("current success original row membership changed")
    _roles(objects, roles, publication.old_rows, success.rows)
    now = ledger._read_tick(runtime)
    for prefix, rows in (("old", publication.old_rows), ("current", success.rows)):
        retained = ledger._inventory(objects[prefix + "_model"])
        if len(retained) != len(rows) or any(a is not b.snapshot() for a, b in zip(retained, rows)):
            raise ValueError("current success native retained inventory changed")
        for snapshot, row in zip(retained, rows):
            ledger._verify_row(
                owner, objects[prefix + "_runtime"], snapshot, row, now, lifecycle_leased=True
            )
    # All opaque source/row/model ports have now returned; stamps below are pure.
    proof = _immutable(manager, binding, objects, success.rows)
    observation = _zero_poll(binding, success)
    contents = success.native_content, success.learner_content, success.inbox_content
    raw = (
        "publication_current_success",
        id(binding),
        id(success.receipt()),
        success.completed_updates,
        _history_content(manager, history),
    )
    metadata = ("tuple", (_history_value(raw, max_metadata_bytes=131072), _success_proof(proof)))
    actual_contents = _contents(manager, runtime, binding.model())
    if (
        _encoded_size(success.metadata, ledger._admission.limits.max_metadata_bytes)
        + RECORD_OVERHEAD_BYTES
        != success.charged_bytes
    ):
        raise ValueError("current success original metadata payment changed")
    if observation is not None:
        # The original completion is unchanged. A paid actual zero-work poll
        # separately binds its later revision/time/sharing observation. Pending
        # items have already been refused by complete inbox coverage above.
        if (
            actual_contents[:2] != contents[:2]
            or metadata[1][0] != success.metadata[1][0]
            or _success_proof(proof) != _stored_proof(observation)
        ):
            raise ValueError("current success differs from its paid zero-work observation")
        return True
    if (
        actual_contents != contents
        or metadata != success.metadata
        or _encoded_size(success.metadata, ledger._admission.limits.max_metadata_bytes)
        + RECORD_OVERHEAD_BYTES
        != success.charged_bytes
    ):
        raise ValueError("current success complete native/inbox/work seal changed")
    return True
