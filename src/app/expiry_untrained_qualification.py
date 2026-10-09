"""Observe, reprove and prepay current original untrained expiry metadata.

Original managed ingress and held cleanup leases are prerequisites. Temporary
weak seals describe the current owned arrivals, not historical registration.
Prepared persistent maps are returned without publication or payload removal.
The caller separately proves expiry authority, native cleanup and final release.
Ordinary TTL/source-age access guards and the trained-only public audit remain
unchanged; this module supplies no live access, receipt or completed work.
"""

from typing import Any, cast
from weakref import ReferenceType, ref

from src.app.actor_shadow import ActorShadowRuntime
from src.app.erased_inbox_origins import _counter, _schema
from src.app.expiry_inbox_qualification import (
    _preflight_history,
    _preflight_records,
    _require_admission,
    _require_committed_record,
    _require_key,
    _require_original_bindings,
    _require_owner_metadata,
    _require_ready_time,
)
from src.app.managed_inbox_origins import ManagedInboxOrigins, _digest
from src.app.untrained_inbox_erasure import _Arrival, _arrival_stamp
from src.app.untrained_inbox_origins import (
    prepare_untrained_inbox_origin,
    untrained_inbox_origin_stamp,
)
from src.core.checkpoint_content import checkpoint_content_stamp
from src.core.data_erasure import ErasedExperience
from src.core.data_lifecycle import LifecycleDeclaration, LifecycleLimits
from src.core.replay_origin import ReplayOriginAdmission
from src.core.untrained_inbox_origin import UntrainedInboxOriginData


def _bindings(history: Any, ledger: Any, owner: Any, runtime: Any) -> tuple:
    return (
        *(id(value) for value in (history, ledger, owner, runtime)),
        id(ledger._anchors),
        tuple((key, id(value)) for key, value in ledger._anchors.items()),
        *(
            id(value)
            for value in (
                ledger._owner,
                ledger._inbox_history_birth,
                ledger._original_inbox_origins,
                ledger._lifecycle,
                ledger._admission,
                ledger._admission.limits,
                owner._catalog,
                owner._revoked_keys,
                owner._opted_out,
                owner._limits,
                history._content_limits,
                runtime._inbox,
                runtime._inbox._clock,
                owner._lifecycle,
                runtime._inbox._experiences,
                runtime._inbox._labels,
                runtime._inbox._applied,
                runtime._inbox._erased,
            )
        ),
        runtime._inbox._clock._time,
        runtime._budget.updates_completed,
        ledger._version,
        runtime._candidate_version,
        runtime._base_actor_version,
    )


def _current_arrivals(history: Any, owner: Any, runtime: Any) -> tuple:
    inbox = runtime._inbox
    # The caller has already checked every map/key/value header without hashes.
    # Keep original capacity proof cheap until the combined aggregate is bound.
    if (
        type(history._inbox_capacity) is not int
        or type(inbox._capacity) is not int
        or inbox._capacity != history._inbox_capacity
    ):
        raise ValueError("expiry original inbox capacity changed")
    keys = inbox._experiences.keys() | inbox._labels.keys()
    trained = {record.data.key for record in history._records.values()}
    if len(keys) > history._capacity or not trained <= keys:
        raise ValueError("expiry arrival inventory lacks complete original trained lineage")
    result = []
    for key in sorted(keys - trained):
        _require_key(key, history._content_limits.max_metadata_bytes)
        declaration = owner._catalog.get(key)
        _schema(declaration, LifecycleDeclaration)
        result.append((key, inbox._experiences.get(key), inbox._labels.get(key), declaration))
    return tuple(result)


def _prove_current(
    history: Any, ledger: Any, owner: Any, runtime: Any, now: int, *, revoked: bool = False
) -> tuple:
    if type(runtime) is not ActorShadowRuntime:
        raise ValueError("expiry arrivals require the exact original runtime")
    life = _require_original_bindings(history, ledger, owner, runtime)
    _require_ready_time(ledger, runtime, life, now)
    _require_owner_metadata(owner, history)
    # Cheap trained schemas must precede deriving keys from their scalar data.
    _preflight_records(history, runtime)
    current = _current_arrivals(history, owner, runtime)
    borrowed = tuple((source, label, declaration) for _, source, label, declaration in current)
    records = _preflight_history(history, owner, runtime, borrowed)
    ManagedInboxOrigins._maps(history, runtime)
    _require_admission(history, ledger)
    ManagedInboxOrigins._verify_erased(history, owner, runtime)
    ManagedInboxOrigins._verify_untrained(history, owner, runtime)
    trained = {record.data.key for record in records}
    untrained = {key for key, *_ in current}
    inbox = runtime._inbox
    if (
        inbox._applied.keys() != trained | history._erased_records.keys()
        or untrained & (inbox._applied.keys() | inbox._erased.keys())
        or runtime._budget.updates_completed != ledger._last_work
    ):
        raise ValueError("expiry arrivals require complete original applied/work partitions")
    allowed = frozenset(trained | untrained) if revoked else frozenset()
    if revoked and not allowed <= owner._revoked_keys:
        raise ValueError("expiry preparation requires actual original cleanup revocation")
    for record in records:
        _require_committed_record(history, owner, runtime, record, now, allowed)
    for key, source, label, declaration in current:
        _arrival_stamp(source, label, declaration, history._content_limits)
        LifecycleLimits.require_supported(owner._limits, declaration)
        if (
            declaration.key != key
            or (key in owner._revoked_keys and not revoked)
            or declaration.provenance.subject_id in owner._opted_out
        ):
            raise ValueError("expiry arrival original declaration/consent changed")
        for value, tick in (
            (source, None if source is None else source.observed_at),
            (label, None if label is None else label.arrived_at),
        ):
            if value is not None and (
                value.key != key
                or value.model_version != runtime._base_actor_version
                or cast(int, tick) > now
            ):
                raise ValueError("expiry arrival original version/chronology changed")
    return current


def _arrival_seal(arrival: _Arrival) -> tuple:
    if type(arrival) is not _Arrival:
        raise ValueError("expiry arrival requires its exact original temporary weak proof")
    _counter(arrival.tick)
    _counter(arrival.work)
    _require_key(arrival.key, 128 * 1024)
    _digest(arrival.stamp)
    if type(arrival.declaration) is not ReferenceType:
        raise ValueError("expiry arrival requires the original weak declaration")
    for reference in (arrival.source, arrival.label, arrival.declaration):
        if reference is not None and type(reference) is not ReferenceType:
            raise ValueError("expiry arrival requires exact original weak references")
    return (
        id(arrival),
        arrival.key,
        arrival.stamp,
        arrival.tick,
        arrival.work,
        *(id(value) for value in (arrival.source, arrival.label, arrival.declaration)),
        *(
            id(None if value is None else value())
            for value in (
                arrival.source,
                arrival.label,
                arrival.declaration,
            )
        ),
    )


def observe_expiry_arrivals(history, ledger, owner, runtime, now: int) -> tuple:
    """Seal current trusted original managed arrivals without renewing live access."""
    current = _prove_current(history, ledger, owner, runtime, now)
    before = ManagedInboxOrigins.state_stamp(history)
    arrivals = tuple(
        _Arrival(
            key,
            None if source is None else ref(source),
            None if label is None else ref(label),
            ref(declaration),
            _arrival_stamp(source, label, declaration, history._content_limits),
            now,
            runtime._budget.updates_completed,
        )
        for key, source, label, declaration in current
    )
    proof = (
        before,
        arrivals,
        tuple(_arrival_seal(item) for item in arrivals),
        _bindings(history, ledger, owner, runtime),
    )
    require_expiry_arrivals_unchanged(history, ledger, owner, runtime, now, proof)
    return proof


def _require_scalar_proof(value: Any, history: Any) -> None:
    limits = history._content_limits
    pending = [(value, 0)]
    count = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > limits.max_nodes or depth > limits.max_depth:
            raise ValueError("expiry original saved proof exceeds scalar bounds")
        if type(item) is tuple:
            if count + len(pending) + len(item) > limits.max_nodes:
                raise ValueError("expiry original saved proof exceeds scalar bounds")
            pending.extend((child, depth + 1) for child in item)
        elif item is not None and not any(type(item) is kind for kind in (bool, int, float, str)):
            raise ValueError("expiry original saved proof requires exact scalar tuples")
        elif type(item) is str and len(item) > limits.max_metadata_bytes:
            raise ValueError("expiry original saved proof exceeds scalar bounds")
        elif type(item) is int and item.bit_length() > 64:
            raise ValueError("expiry original saved proof exceeds scalar bounds")


def _require_unchanged(history, ledger, owner, runtime, now, before, *, revoked=False) -> tuple:
    _require_original_bindings(history, ledger, owner, runtime)
    if type(before) is not tuple or len(before) != 4:
        raise ValueError("expiry arrivals require their original immutable proof tuple")
    state, arrivals, seals, bindings = before
    if (
        type(arrivals) is not tuple
        or len(arrivals) > history._capacity
        or type(seals) is not tuple
        or len(seals) != len(arrivals)
    ):
        raise ValueError("expiry original arrival proof inventory changed")
    # Reject foreign nested comparison operands before any tuple equality can
    # invoke their callbacks. This graph contains only saved scalar metadata.
    saved = state, seals, bindings
    _require_scalar_proof(saved, history)
    checkpoint_content_stamp(saved, history._content_limits)
    # Why: compare immutable saved scalar seals BEFORE trusting mutable frozen
    # arrival fields. A callback cannot refresh an arrival's own content stamp.
    if tuple(_arrival_seal(item) for item in arrivals) != seals:
        raise ValueError("expiry original arrival weak/scalar proof changed")
    current = _prove_current(history, ledger, owner, runtime, now, revoked=revoked)
    if _bindings(history, ledger, owner, runtime) != bindings:
        raise ValueError("expiry original arrival bindings/time/work changed")
    if ManagedInboxOrigins.state_stamp(history) != state or len(current) != len(arrivals):
        raise ValueError("expiry original trained/arrival history changed")
    for arrival, (key, source, label, declaration) in zip(arrivals, current):
        if (
            arrival.key != key
            or arrival.tick != now
            or arrival.work != runtime._budget.updates_completed
            or (None if arrival.source is None else arrival.source()) is not source
            or (None if arrival.label is None else arrival.label()) is not label
            or arrival.declaration() is not declaration
            or _arrival_stamp(source, label, declaration, history._content_limits) != arrival.stamp
        ):
            raise ValueError("expiry original arrival identity/content changed after callback")
    return arrivals


def require_expiry_arrivals_unchanged(history, ledger, owner, runtime, now, before) -> tuple:
    """Reprove the original borrowed objects after opaque calls, with no charge."""
    return _require_unchanged(history, ledger, owner, runtime, now, before)


def _persistent_data(history, runtime, arrivals, prepared, now) -> tuple:
    inbox = runtime._inbox
    keys = inbox._experiences.keys() | inbox._labels.keys()
    if type(prepared) is not dict or len(prepared) > history._capacity:
        raise ValueError("expiry preparation requires bounded exact tombstone storage")
    for key, tombstone in prepared.items():
        _require_key(key, history._content_limits.max_metadata_bytes)
        _schema(tombstone, ErasedExperience)
    checkpoint_content_stamp(tuple(prepared.values()), history._content_limits)
    if prepared.keys() != inbox._erased.keys() | keys or any(
        prepared[key] is not value for key, value in inbox._erased.items()
    ):
        raise ValueError("expiry original prepared tombstone inventory changed")
    for key in keys:
        source, label = inbox._experiences.get(key), inbox._labels.get(key)
        tombstone = prepared[key]
        ErasedExperience.__post_init__(tombstone)
        if (
            tombstone.reason != "expired"
            or tombstone.erased_at != now
            or tombstone.key != key
            or tombstone.actor_version != runtime._base_actor_version
            or tombstone.observed_at != (None if source is None else source.observed_at)
            or tombstone.event_id != (None if label is None else label.event_id)
            or tombstone.arrived_at != (None if label is None else label.arrived_at)
        ):
            raise ValueError("expiry original prepared tombstone contents changed")
    result = []
    for arrival in arrivals:
        declaration: Any = arrival.declaration()
        tombstone = prepared[arrival.key]
        result.append(
            UntrainedInboxOriginData(
                arrival.key,
                runtime._base_actor_version,
                runtime._candidate_version,
                declaration.provenance.subject_id,
                declaration.provenance.source_id,
                tombstone.observed_at,
                tombstone.event_id,
                tombstone.arrived_at,
                now,
                "expired",
                arrival.work,
            )
        )
    return tuple(result)


def prepare_expiry_untrained_erasure(
    history, ledger, owner, runtime, now, before, prepared, offset: int = 0
) -> tuple:
    """Pay all persistent witnesses first; return maps without publishing them."""
    arrivals = _require_unchanged(history, ledger, owner, runtime, now, before, revoked=True)
    _counter(offset)
    admission = ledger._admission
    # Fixed reserve_untrained still calls self.accounting/self._charge. Reject
    # instance shadows so payment cannot dispatch an opaque replacement port.
    if (
        type(admission) is not ReplayOriginAdmission
        or len(vars(admission)) != 4
        or any(type(key) is not str or len(key) > 64 for key in vars(admission))
        or vars(admission).keys() != {"limits", "_original_limits", "_progress", "_minimum"}
    ):
        raise ValueError("expiry requires original callback-free admission operations")
    if offset > admission.limits.max_live_records:
        raise ValueError("expiry preparation offset exceeds original live-record capacity")
    data = _persistent_data(history, runtime, arrivals, prepared, now)
    # Why: stamp the original dict itself. Rebuilding a nested inventory tuple
    # would include fresh wrapper identities and make an unchanged proof flaky.
    prepared_stamp = checkpoint_content_stamp(
        (id(prepared), prepared),
        history._content_limits,
    )
    live = (
        len(ledger._rows)
        + ledger._copy_slots
        + len(history._records)
        + len(history._erased_records)
        + len(history._untrained_records)
    )
    for index, item in enumerate(data):
        ReplayOriginAdmission.reserve_untrained(admission, item, live + offset + index)
    _require_unchanged(history, ledger, owner, runtime, now, before, revoked=True)
    records, sealed = history._untrained_records.copy(), history._untrained_sealed.copy()
    for arrival, item in zip(arrivals, data):
        if arrival.key in records:
            raise ValueError("expiry cannot overwrite original untrained witness lineage")
        record = prepare_untrained_inbox_origin(
            item, prepared[arrival.key], history._content_limits
        )
        untrained_inbox_origin_stamp(record, history._content_limits)
        if record.data is not item or record.tombstone() is not prepared[arrival.key]:
            raise ValueError("expiry persistent witness original identities changed")
        records[arrival.key] = sealed[arrival.key] = record
    _require_unchanged(history, ledger, owner, runtime, now, before, revoked=True)
    _persistent_data(history, runtime, arrivals, prepared, now)
    if (
        checkpoint_content_stamp(
            (id(prepared), prepared),
            history._content_limits,
        )
        != prepared_stamp
    ):
        raise ValueError("expiry prepared tombstones changed during persistent allocation")
    for arrival in arrivals:
        untrained_inbox_origin_stamp(records[arrival.key], history._content_limits)
    return records, sealed
