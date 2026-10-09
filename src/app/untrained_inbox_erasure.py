"""Observe current original untrained arrivals before trusted payload removal.

Inputs are the original birth-enrolled history, owner and inbox. Temporary
weak arrival seals bridge original consent checks and the actual commit; outputs
are paid payload-free witnesses and prepared maps. No registration-time claim,
applied work, copied lineage, cleanup or renewed authority is supplied here.
"""

from dataclasses import dataclass
from typing import Any
from weakref import ReferenceType, ref

import numpy as np

from src.app.erased_inbox_origins import _counter, _identifier, _schema
from src.app.untrained_inbox_origins import prepare_untrained_inbox_origin
from src.core.checkpoint_content import checkpoint_content_stamp
from src.core.data_lifecycle import LifecycleDeclaration
from src.core.experience import Experience, ExperiencePermissions, LabelArrival
from src.core.inbox_origin import inbox_numeric_stamp
from src.core.untrained_inbox_origin import UntrainedInboxOriginData


@dataclass(frozen=True, slots=True)
class _Arrival:
    key: tuple[str, str]
    source: ReferenceType[Any] | None
    label: ReferenceType[Any] | None
    declaration: ReferenceType[Any]
    stamp: str
    tick: int
    work: int


def _arrival_stamp(source, label, declaration, limits) -> str:
    from src.app.managed_inbox_origins import _declaration_values

    declaration_values = _declaration_values(declaration, limits)
    if (
        not declaration.consent.training
        or not declaration.consent.replay
        or declaration.retention != "replay"
    ):
        raise ValueError("untrained arrival requires original training/replay declaration")
    if source is None and label is None:
        raise ValueError("untrained erasure requires an actual original arrival")
    if source is not None:
        _schema(source, Experience)
        _schema(source.permissions, ExperiencePermissions)
        ExperiencePermissions.__post_init__(source.permissions)
        _counter(source.observed_at)
        for value in (*source.key, source.model_version):
            _identifier(value, limits.max_metadata_bytes)
        checkpoint_content_stamp(
            tuple(value for name, value in vars(source).items() if name != "features"), limits
        )
        if (
            type(source.role) is not str
            or source.role != "train"
            or not source.permissions.training
            or not source.permissions.replay
        ):
            raise ValueError("untrained source requires original train/replay permission")
    if label is not None:
        _schema(label, LabelArrival)
        _counter(label.arrived_at)
        for value in (*label.key, label.event_id, label.model_version):
            _identifier(value, limits.max_metadata_bytes)
        checkpoint_content_stamp(
            tuple(value for name, value in vars(label).items() if name != "targets"), limits
        )
        if type(label.role) is not str or label.role != "train":
            raise ValueError("untrained label requires original train role")
    # Bound both payloads together before any full content traversal.
    payloads = (() if source is None else (source.features,)) + (
        () if label is None else (label.targets,)
    )
    if any(type(value) is not np.ndarray for value in payloads):
        raise ValueError("untrained original payloads require exact numeric arrays")
    if sum(value.nbytes for value in payloads) > limits.max_array_bytes:
        raise ValueError("original untrained paired payload exceeds original byte limit")
    for value in payloads:
        inbox_numeric_stamp(value, limits)
    checkpoint_content_stamp((source, label), limits)
    if source is not None:
        Experience.__post_init__(source)
    if label is not None:
        LabelArrival.__post_init__(label)
    if (
        source is not None
        and label is not None
        and (
            source.key != label.key
            or source.model_version != label.model_version
            or source.observed_at > label.arrived_at
        )
    ):
        raise ValueError("untrained original pair key/version/chronology changed")
    # The root wrapper's identity is excluded by the content-stamp contract.
    # Flatten temporary scalar tuples so only original nested identities persist.
    return checkpoint_content_stamp(
        (
            id(source),
            id(label),
            id(declaration),
            *declaration_values,
            source,
            label,
            *(id(value) for value in payloads),
        ),
        limits,
    )


def _require_arrival(history, owner, runtime, arrival: _Arrival) -> tuple:
    if type(arrival) is not _Arrival:
        raise ValueError("untrained erasure requires its exact original arrival seal")
    _counter(arrival.tick)
    _counter(arrival.work)
    if type(arrival.key) is not tuple or len(arrival.key) != 2:
        raise ValueError("untrained arrival requires an exact paired key")
    for part in arrival.key:
        _identifier(part, history._content_limits.max_metadata_bytes)
    if type(arrival.declaration) is not ReferenceType or any(
        value is not None and type(value) is not ReferenceType
        for value in (arrival.source, arrival.label)
    ):
        raise ValueError("untrained erasure requires exact weak original references")
    source = None if arrival.source is None else arrival.source()
    label = None if arrival.label is None else arrival.label()
    declaration = arrival.declaration()
    if (
        declaration is None
        or (arrival.source is not None and source is None)
        or (arrival.label is not None and label is None)
        or runtime._inbox._experiences.get(arrival.key) is not source
        or runtime._inbox._labels.get(arrival.key) is not label
        or owner._catalog.get(arrival.key) is not declaration
        or arrival.key in runtime._inbox._applied
        or arrival.key in runtime._inbox._erased
        or arrival.work != runtime._budget.updates_completed
    ):
        raise ValueError("untrained original arrival/declaration/work identities changed")
    stamp = _arrival_stamp(source, label, declaration, history._content_limits)
    if declaration.key != arrival.key:
        raise ValueError("untrained original declaration key changed")
    if type(arrival.stamp) is not str or stamp != arrival.stamp:
        raise ValueError("untrained original arrival contents changed before erasure")
    for value, time in (
        (source, None if source is None else source.observed_at),
        (label, None if label is None else label.arrived_at),
    ):
        if value is not None:
            if type(time) is not int:
                raise ValueError("untrained original arrival requires an exact time")
            if (
                value.key != arrival.key
                or value.model_version != runtime._base_actor_version
                or time > arrival.tick
                or arrival.tick - time > history._limits().max_age_ticks
            ):
                raise ValueError("untrained original arrival version/age changed")
    return source, label, declaration


def observe_untrained_arrivals(history, ledger, owner, runtime, trained_keys, now) -> tuple:
    """Prove consent before revocation; retain temporary weak seals only."""
    inbox = runtime._inbox
    _counter(now)
    _counter(runtime._budget.updates_completed)
    if (
        type(runtime._stopped) is not bool
        or type(runtime._inbox._stopped) is not bool
        or runtime._stopped
        or runtime._inbox._stopped
        or ledger._pending is not None
    ):
        raise ValueError("untrained erasure refuses ambiguous stopped/pending work")
    keys = inbox._experiences.keys() | inbox._labels.keys()
    if len(keys) > history._capacity:
        raise ValueError("untrained original arrival union exceeds original capacity")
    if not trained_keys <= keys or any(key in inbox._applied for key in keys - trained_keys):
        raise ValueError("untrained erasure cannot replace missing trained lineage")
    result = []
    for key in sorted(keys - trained_keys):
        declaration = owner._require_live(key)
        _schema(declaration, LifecycleDeclaration)
        source, label = inbox._experiences.get(key), inbox._labels.get(key)
        stamp = _arrival_stamp(source, label, declaration, history._content_limits)
        result.append(
            _Arrival(
                key,
                None if source is None else ref(source),
                None if label is None else ref(label),
                ref(declaration),
                stamp,
                now,
                runtime._budget.updates_completed,
            )
        )
    # Earlier arrivals are rechecked after every opaque original access check.
    for arrival in result:
        _require_arrival(history, owner, runtime, arrival)
    if (
        set(trained_keys) | {item.key for item in result}
        != inbox._experiences.keys() | inbox._labels.keys()
    ):
        raise ValueError("original arrival inventory changed during qualification")
    return tuple(result)


def prepare_untrained_erasure(history, ledger, owner, runtime, arrivals, prepared, offset) -> tuple:
    """Reprove after native callbacks, pay persistent metadata, prebuild maps."""
    if type(arrivals) is not tuple or len(arrivals) > history._capacity:
        raise ValueError("untrained erasure requires bounded original arrival tuple")
    records, sealed = history._untrained_records.copy(), history._untrained_sealed.copy()
    for index, arrival in enumerate(arrivals):
        source, label, declaration = _require_arrival(history, owner, runtime, arrival)
        if arrival.key not in owner._revoked_keys or arrival.key in records:
            raise ValueError("untrained erasure lacks actual original revocation")
        tombstone = prepared[arrival.key]
        data = UntrainedInboxOriginData(
            arrival.key,
            runtime._base_actor_version,
            runtime._candidate_version,
            declaration.provenance.subject_id,
            declaration.provenance.source_id,
            None if source is None else source.observed_at,
            None if label is None else label.event_id,
            None if label is None else label.arrived_at,
            tombstone.erased_at,
            tombstone.reason,
            arrival.work,
        )
        ledger._admission.reserve_untrained(data, ledger._live_records() + offset + index)
        record = prepare_untrained_inbox_origin(data, tombstone, history._content_limits)
        records[arrival.key] = sealed[arrival.key] = record
    return records, sealed
