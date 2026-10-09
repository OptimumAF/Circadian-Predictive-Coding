"""Bounded trusted-local delivery around the existing native learner step.

Train-role source and label events may arrive in either transport order. Only
eligible pairs update; payloads are detached and retained IDs prevent retries.
No actor service, role scoring, persistence or physical/scientific seal lives here.
"""

from __future__ import annotations

from copy import deepcopy
from src.core.native_graph_copy import copy_graph
from dataclasses import replace
from contextlib import nullcontext
from typing import Callable, ContextManager, Generic, TypeVar

from src.app.learner_step import update_learner
from src.app.native_update_origin import NativeUpdateAccess
from src.app.inbox_erasure_observation import prepare_inbox_erasure, commit_inbox_erasure
from src.core.native_update_origin import NativeUpdateObserver, NativeUpdateOrigin
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionStopped
from src.core.experience import (
    AppliedExperience,
    EventClock,
    Experience,
    LabelArrival,
    SampleKey,
    require_identifier,
    require_tick,
)
from src.core.learner_ports import NativeLearner, TrainingDiagnostic
from src.core.inbox_cursor import InboxCursor, validate_inbox_cursor
from src.core.data_erasure import ErasedExperience, ErasureReason

Features = TypeVar("Features")
Targets = TypeVar("Targets")
Prediction = TypeVar("Prediction")
State = TypeVar("State")


class ExperienceInbox(Generic[Features, Targets, Prediction, State]):
    def __init__(
        self,
        learner: NativeLearner[Features, Targets, Prediction, State],
        *,
        clock: EventClock,
        budget: ToyBudgetSession,
        learner_version: str,
        max_experiences: int = 1024,
    ) -> None:
        require_identifier(learner_version, "learner_version")
        if type(max_experiences) is not int or max_experiences <= 0:
            raise ValueError("max_experiences must be a positive integer")
        if not callable(getattr(clock, "now", None)) or type(budget) is not ToyBudgetSession:
            raise ValueError("experience inbox requires an event clock and ToyBudgetSession")
        self._learner, self._clock, self._budget = learner, clock, budget
        self._learner_version, self._capacity = learner_version, max_experiences
        self._experiences: dict[SampleKey, Experience[Features]] = {}
        self._labels: dict[SampleKey, LabelArrival[Targets]] = {}
        self._event_ids: set[str] = set()
        self._applied: dict[SampleKey, AppliedExperience] = {}
        self._erased: dict[SampleKey, ErasedExperience] = {}
        self._historical_completed_updates: int | None = None
        self._draining, self._stopped, self._last_time = False, False, -1
        self._registration_guard: Callable[[object], None] | None = None
        self._training_guard: (
            Callable[[Experience[Features], LabelArrival[Targets]], bool] | None
        ) = None
        self._payload_copy_guard: (
            Callable[[Experience[Features], LabelArrival[Targets]], None] | None
        ) = None
        self._read_clock()

    @property
    def applied_updates(self) -> tuple[AppliedExperience, ...]:
        return tuple(self._applied.values())

    @property
    def stopped(self) -> bool:
        """Expose uncertain native failure to an enclosing candidate owner."""
        return self._stopped

    def capture_cursor(self) -> InboxCursor[Features, Targets]:
        """Capture all history from a quiescent owner, including stopped state.

        Why: metadata validation precedes payload copying, so corrupt/held-out
        records cannot use checkpoint capture to access forbidden data. Native
        state and budget/clock restoration require the outer owner handoff.
        """
        if self._draining:
            raise ValueError("inbox cursor capture requires a quiescent drain")
        self._require_history_indexes()
        cursor = InboxCursor(
            2 if self._erased else 1,
            self._learner_version,
            self._capacity,
            tuple(self._experiences[k] for k in sorted(self._experiences)),
            tuple(self._labels[k] for k in sorted(self._labels)),
            tuple(self._applied.values()),
            self._last_time,
            self._stopped,
            self._completed_updates(),
            tuple(self._erased[k] for k in sorted(self._erased)),
        )
        return copy_graph(self, "inbox_capture", cursor)

    def _require_history_indexes(self) -> None:
        if (
            any(k != value.key for k, value in self._experiences.items())
            or any(k != value.key for k, value in self._labels.items())
            or any(k != value.key for k, value in self._erased.items())
            or any(k != (value.episode_id, value.sample_id) for k, value in self._applied.items())
            or self._event_ids
            != {value.event_id for value in self._labels.values()}
            | {value.event_id for value in self._erased.values() if value.event_id is not None}
        ):
            raise ValueError("inbox cursor identity indexes are inconsistent")

    def _materialize_cursor(self, learner, cursor):
        """Internal owned handoff only; no native calls or fresh budget/clock."""
        validate_inbox_cursor(cursor)
        if cursor.completed_updates != self._budget.updates_completed:
            raise ValueError("cursor work differs from the original cumulative budget")
        owned = copy_graph(self, "inbox_materialize", cursor)
        prepared = object.__new__(type(self))
        prepared._learner, prepared._clock, prepared._budget = learner, self._clock, self._budget
        prepared._learner_version, prepared._capacity = owned.learner_version, owned.capacity
        prepared._experiences = {s.key: s for s in owned.experiences}
        prepared._labels = {label.key: label for label in owned.labels}
        prepared._event_ids = {label.event_id for label in owned.labels}
        prepared._applied = {(r.episode_id, r.sample_id): r for r in owned.applied}
        prepared._erased = {item.key: item for item in owned.erased}
        prepared._event_ids.update(
            item.event_id for item in owned.erased if item.event_id is not None
        )
        prepared._last_time, prepared._stopped = owned.last_tick, owned.stopped
        prepared._draining = False
        prepared._historical_completed_updates = None
        # Why: restored payload history must retain the original live consent authority.
        prepared._registration_guard = self._registration_guard
        prepared._training_guard = self._training_guard
        prepared._payload_copy_guard = self._payload_copy_guard
        return prepared

    def _require_open(self) -> None:
        if self._historical_completed_updates is not None:
            raise ValueError("experience inbox is retired after owner handoff")
        if self._stopped:
            raise ValueError("experience inbox is stopped after an uncertain learner failure")
        if self._draining:
            raise ValueError("nested experience drain or registration is unsupported")

    def _require_capacity(self, key: SampleKey) -> None:
        identities = self._experiences.keys() | self._labels.keys() | self._erased.keys()
        if key not in identities and len(identities) >= self._capacity:
            raise ValueError("experience inbox identity capacity exceeded")

    @staticmethod
    def _require_pair(source: Experience[Features], label: LabelArrival[Targets]) -> None:
        if source.role != label.role or source.model_version != label.model_version:
            raise ValueError("label role or actor version differs from its experience")
        if label.arrived_at < source.observed_at:
            raise ValueError("label cannot arrive before source observation")

    def record_experience(self, source: Experience[Features]) -> None:
        self._require_open()
        if self._registration_guard is not None:
            self._registration_guard(source)
        if (
            type(source) is not Experience
            or source.role != "train"
            or not source.permissions.training
        ):
            raise ValueError("experience inbox requires train-role training permission")
        key = source.key
        if key in self._experiences or key in self._erased:
            raise ValueError(f"duplicate experience identity: {key}")
        self._require_capacity(key)
        if key in self._labels:
            self._require_pair(source, self._labels[key])
        self._experiences[key] = deepcopy(source)

    def record_label(self, label: LabelArrival[Targets]) -> None:
        self._require_open()
        if self._registration_guard is not None:
            self._registration_guard(label)
        if type(label) is not LabelArrival or label.role != "train":
            raise ValueError("label must have the train role before payload access")
        key = label.key
        if key in self._labels or key in self._erased or label.event_id in self._event_ids:
            raise ValueError(f"duplicate label or event identity: {key}, {label.event_id}")
        self._require_capacity(key)
        if key in self._experiences:
            self._require_pair(self._experiences[key], label)
        owned = deepcopy(label)
        self._labels[key] = owned
        self._event_ids.add(owned.event_id)

    def _erase_payloads(
        self, keys: tuple[SampleKey, ...], *, reason: ErasureReason
    ) -> tuple[ErasedExperience, ...]:
        """Internal primitive under the outer candidate's exclusive lease.

        Prepare and validate payload-free history before dropping references.
        Tombstones retain identity capacity, event IDs and applied receipts.
        This neither erases native/checkpoint copies nor grants deletion authority.
        """
        if self._draining:
            raise ValueError("payload erasure requires a quiescent inbox")
        self._require_history_indexes()
        self._require_erasure_keys(keys, reason)
        now = self._read_clock()
        prepared = self._prepare_erased_history(keys, now, reason)
        self._commit_erased_history(keys, prepared)
        return tuple(prepared[key] for key in keys)

    def _commit_erased_history(self, keys, prepared) -> None:
        transition = prepare_inbox_erasure(self, keys, prepared)
        for key in keys:
            self._experiences.pop(key, None)
            self._labels.pop(key, None)
        self._erased = prepared
        if transition is not None:
            commit_inbox_erasure(transition)

    def _completed_updates(self) -> int:
        return (
            self._budget.updates_completed
            if self._historical_completed_updates is None
            else self._historical_completed_updates
        )

    def _retire_ledger(self) -> None:
        if self._draining or self._historical_completed_updates is not None:
            raise ValueError("inbox ledger retirement requires a quiescent current owner")
        # Preserve the historical observation; never change the cumulative budget.
        self._historical_completed_updates = self._budget.updates_completed

    def _require_erasure_keys(self, keys: tuple[SampleKey, ...], reason: ErasureReason) -> None:
        if (
            type(keys) is not tuple
            or not keys
            or type(reason) is not str
            or reason not in ("deleted", "expired", "opt_out")
        ):
            raise ValueError("erasure requires immutable keys and a supported reason")
        for key in keys:
            if type(key) is not tuple or len(key) != 2:
                raise ValueError("erasure requires episode/sample identities")
            for value in key:
                require_identifier(value, "erased identity")
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate erasure identity")
        known = self._experiences.keys() | self._labels.keys() | self._erased.keys()
        if any(key not in known for key in keys):
            raise ValueError("unknown erasure identity")

    def _prepare_erased_history(
        self, keys: tuple[SampleKey, ...], now: int, reason: ErasureReason
    ) -> dict[SampleKey, ErasedExperience]:
        prepared = self._erased.copy()
        for key in keys:
            if key not in prepared:
                prepared[key] = self._erasure_metadata(key, now, reason)
        # Validate the resulting whole history without copying any payload.
        InboxCursor(
            2,
            self._learner_version,
            self._capacity,
            tuple(source for key, source in self._experiences.items() if key not in prepared),
            tuple(label for key, label in self._labels.items() if key not in prepared),
            tuple(self._applied.values()),
            now,
            self._stopped,
            self._completed_updates(),
            tuple(prepared.values()),
        )
        return prepared

    def _erasure_metadata(
        self, key: SampleKey, now: int, reason: ErasureReason
    ) -> ErasedExperience:
        source, label = self._experiences.get(key), self._labels.get(key)
        if source is not None:
            replace(source)
            replace(source.permissions)
            if source.role != "train" or not source.permissions.training:
                raise ValueError("erasure source metadata must remain train-authorized")
        if label is not None:
            replace(label)
            if label.role != "train":
                raise ValueError("erasure label metadata must retain train role")
        if source is not None and label is not None:
            self._require_pair(source, label)
        if source is None and label is None:
            raise ValueError("erasure requires existing source or label metadata")
        if source is not None:
            version = source.model_version
        else:
            if label is None:
                raise ValueError("erasure requires existing label metadata")
            version = label.model_version
        return ErasedExperience(
            key,
            version,
            None if source is None else source.observed_at,
            None if label is None else label.event_id,
            None if label is None else label.arrived_at,
            now,
            reason,
        )

    def _read_clock(self) -> int:
        now = self._clock.now()
        require_tick(now, "event clock")
        if now < self._last_time:
            raise ValueError("injected event clock moved backwards")
        self._last_time = now
        return now

    def _ready(self, now: int) -> list[SampleKey]:
        ready = [
            key
            for key in self._experiences.keys() & self._labels.keys()
            if key not in self._applied
            and self._labels[key].arrived_at <= now
            and self._experiences[key].observed_at <= now
            and self._training_allowed(key)
        ]
        return sorted(
            ready,
            key=lambda key: (
                self._labels[key].arrived_at,
                self._experiences[key].observed_at,
                *key,
            ),
        )

    def _training_allowed(self, key: SampleKey) -> bool:
        if self._training_guard is None:
            return True
        allowed = self._training_guard(self._experiences[key], self._labels[key])
        if type(allowed) is not bool:
            raise ValueError("permanent training admission must return an exact boolean")
        return allowed

    def _apply(
        self,
        key: SampleKey,
        now: int,
        native_observer: NativeUpdateObserver[Features, Targets] | None = None,
    ) -> AppliedExperience:
        source, label = self._experiences[key], self._labels[key]
        started = False
        access: NativeUpdateAccess[Features, Targets] | None = None

        def entering() -> None:
            nonlocal started
            started = True
            if access is not None:
                access.notify("started")

        def completed(diagnostic: TrainingDiagnostic) -> None:
            # Why: commit identity before the existing post-update resource
            # check; a late stop must never make the completed label retryable.
            self._applied[key] = AppliedExperience(
                source.sample_id,
                source.episode_id,
                label.event_id,
                source.model_version,
                self._learner_version,
                source.observed_at,
                label.arrived_at,
                now,
                self._budget.updates_completed,
                diagnostic,
            )
            if access is not None:
                access.notify("completed")

        if self._payload_copy_guard is not None:
            self._payload_copy_guard(source, label)
        features, targets = deepcopy((source.features, label.targets))
        if native_observer is not None:

            def read_origin() -> NativeUpdateOrigin[Features, Targets]:
                return NativeUpdateOrigin(
                    self._learner,
                    source,
                    label,
                    features,
                    targets,
                    self._learner_version,
                    self._applied.get(key),
                    self._budget.updates_completed,
                )

            access = NativeUpdateAccess(native_observer, read_origin)
        try:
            update_learner(
                self._learner,
                features,
                targets,
                self._budget,
                on_started=entering,
                on_completed=completed,
            )
        except ToyExecutionStopped as error:
            if started and key not in self._applied:
                self._stopped = True
            if access is not None:
                access.failed(
                    "committed_failure"
                    if key in self._applied
                    else "uncertain"
                    if started
                    else "refused",
                    error,
                )
            raise
        except BaseException as error:
            self._stopped = True
            if access is not None:
                access.failed(
                    "committed_failure"
                    if key in self._applied
                    else "uncertain"
                    if started
                    else "refused",
                    error,
                )
            raise
        finally:
            if access is not None:
                access.close()
        return self._applied[key]

    def drain(
        self,
        *,
        max_updates: int | None = None,
        before_each_update: Callable[[], ContextManager[bool]] | None = None,
        native_observer: NativeUpdateObserver[Features, Targets] | None = None,
    ) -> tuple[AppliedExperience, ...]:
        if max_updates is not None and (type(max_updates) is not int or max_updates <= 0):
            raise ValueError("max_updates must be a positive integer or None")
        if before_each_update is not None and not callable(before_each_update):
            raise ValueError("before_each_update must be a callable admission context")
        if native_observer is not None and not callable(native_observer):
            raise ValueError("native_observer must be callable or None")
        self._require_open()
        now = self._read_clock()
        ready = self._ready(now)
        self._draining = True
        try:
            updates = []
            for key in ready[:max_updates]:
                context = (
                    before_each_update() if before_each_update is not None else nullcontext(True)
                )
                with context as allowed:
                    if type(allowed) is not bool:
                        raise ValueError("update admission must yield an exact boolean")
                    if not allowed:
                        break
                    updates.append(self._apply(key, now, native_observer))
            return tuple(updates)
        finally:
            self._draining = False
