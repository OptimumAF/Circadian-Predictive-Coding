"""Validated complete inbox cursor metadata with opaque training payloads.

This format is a detached observation, not a native/budget restore capability.
Validate metadata before copying payloads. IO and native ownership stay outside.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Generic, TypeVar

from src.core.experience import (
    AppliedExperience,
    Experience,
    LabelArrival,
    SampleKey,
    require_identifier,
    require_tick,
)
from src.core.learner_ports import TrainingDiagnostic
from src.core.data_erasure import ErasedExperience

Features = TypeVar("Features")
Targets = TypeVar("Targets")


@dataclass(frozen=True)
class InboxCursor(Generic[Features, Targets]):
    format_version: int
    learner_version: str
    capacity: int
    experiences: tuple[Experience[Features], ...]
    labels: tuple[LabelArrival[Targets], ...]
    applied: tuple[AppliedExperience, ...]
    last_tick: int
    stopped: bool
    completed_updates: int
    erased: tuple[ErasedExperience, ...] = ()

    def __post_init__(self) -> None:
        validate_inbox_cursor(self)


def validate_inbox_cursor(cursor: object) -> None:
    if (
        type(cursor) is not InboxCursor
        or type(cursor.format_version) is not int
        or cursor.format_version not in (1, 2)
    ):
        raise ValueError("inbox cursor requires supported exact format version 1 or 2")
    if type(cursor.erased) is not tuple or (cursor.format_version == 1) != (not cursor.erased):
        raise ValueError("format 2 requires tombstones; format 1 has no erased history")
    require_identifier(cursor.learner_version, "cursor learner_version")
    for name in ("capacity", "last_tick", "completed_updates"):
        require_tick(getattr(cursor, name), "cursor " + name)
    if cursor.capacity == 0 or type(cursor.stopped) is not bool:
        raise ValueError("cursor capacity must be positive and stopped an exact boolean")
    if any(type(rows) is not tuple for rows in (cursor.experiences, cursor.labels, cursor.applied)):
        raise ValueError("cursor histories must be immutable tuples")
    sources, labels = _validate_events(cursor)
    erased = _validate_erased(cursor, sources, labels)
    if len(sources.keys() | labels.keys() | erased.keys()) > cursor.capacity:
        raise ValueError("cursor identity capacity exceeded")
    for key in sources.keys() & labels.keys():
        source, label = sources[key], labels[key]
        if source.model_version != label.model_version or label.arrived_at < source.observed_at:
            raise ValueError("cursor source/label version or arrival pair is inconsistent")
    _validate_applied(cursor, sources, labels, erased)


def _validate_erased(cursor, sources, labels):
    erased: dict[SampleKey, ErasedExperience] = {}
    events = {label.event_id for label in labels.values()}
    for item in cursor.erased:
        if type(item) is not ErasedExperience:
            raise ValueError("erased cursor requires exact payload-free metadata")
        replace(item)
        if item.key in erased or item.key in sources or item.key in labels:
            raise ValueError("duplicate or live-overlapping erased identity")
        if item.erased_at > cursor.last_tick:
            raise ValueError("erasure time cannot exceed the observed cursor")
        if item.event_id is not None:
            if item.event_id in events:
                raise ValueError("duplicate erased label event identity")
            events.add(item.event_id)
        erased[item.key] = item
    return erased


def _validate_events(cursor: InboxCursor[Features, Targets]):
    sources: dict[SampleKey, Experience[Features]] = {}
    labels: dict[SampleKey, LabelArrival[Targets]] = {}
    event_ids: set[str] = set()
    for source in cursor.experiences:
        if type(source) is not Experience:
            raise ValueError("cursor source must be an exact Experience")
        replace(source)  # revalidate every declared metadata field without payload copy
        replace(source.permissions)
        if source.role != "train" or not source.permissions.training:
            raise ValueError("cursor sources require train-role training permission")
        if source.key in sources:
            raise ValueError("duplicate cursor source identity")
        sources[source.key] = source
    for label in cursor.labels:
        if type(label) is not LabelArrival:
            raise ValueError("cursor label must be an exact LabelArrival")
        replace(label)
        if label.role != "train":
            raise ValueError("cursor labels require the train role")
        if label.key in labels or label.event_id in event_ids:
            raise ValueError("duplicate cursor label/event identity")
        labels[label.key] = label
        event_ids.add(label.event_id)
    return sources, labels


def _validate_applied(cursor: InboxCursor[Features, Targets], sources, labels, erased) -> None:
    seen = set()
    last_applied_tick = 0
    for number, receipt in enumerate(cursor.applied, 1):
        if type(receipt) is not AppliedExperience:
            raise ValueError("cursor receipt must be an exact AppliedExperience")
        for name in ("sample_id", "episode_id", "event_id", "actor_version", "learner_version"):
            require_identifier(getattr(receipt, name), "cursor receipt " + name)
        for name in ("observed_at", "arrived_at", "applied_at", "update_number"):
            require_tick(getattr(receipt, name), "cursor receipt " + name)
        key = receipt.episode_id, receipt.sample_id
        if key in seen or (key not in erased and (key not in sources or key not in labels)):
            raise ValueError("duplicate or unreferenced applied cursor identity")
        seen.add(key)
        if key in erased:
            item = erased[key]
            if (
                item.observed_at is None
                or item.arrived_at is None
                or receipt.applied_at > item.erased_at
            ):
                raise ValueError("applied erased identity requires a complete consumed pair")
            version, event, observed, arrived = (
                item.actor_version,
                item.event_id,
                item.observed_at,
                item.arrived_at,
            )
        else:
            source, label = sources[key], labels[key]
            version, event, observed, arrived = (
                source.model_version,
                label.event_id,
                source.observed_at,
                label.arrived_at,
            )
        expected = (
            version,
            cursor.learner_version,
            event,
            observed,
            arrived,
            number,
        )
        actual = (
            receipt.actor_version,
            receipt.learner_version,
            receipt.event_id,
            receipt.observed_at,
            receipt.arrived_at,
            receipt.update_number,
        )
        if actual != expected or not receipt.arrived_at <= receipt.applied_at <= cursor.last_tick:
            raise ValueError("applied cursor version/event/time/update numbering is inconsistent")
        if receipt.applied_at < last_applied_tick:
            raise ValueError("applied cursor chronology moved backwards")
        last_applied_tick = receipt.applied_at
        if type(receipt.diagnostic) is not TrainingDiagnostic:
            raise ValueError("applied cursor requires a native TrainingDiagnostic")
        replace(receipt.diagnostic)
    if cursor.completed_updates < len(cursor.applied) or (
        not cursor.stopped and cursor.completed_updates != len(cursor.applied)
    ):
        raise ValueError("completed cursor work requires an exclusive consistent ledger")
