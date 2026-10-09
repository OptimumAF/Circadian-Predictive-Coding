"""Same-process original candidate row consent with bounded weak references.

Wraps original owner/sharing/runtime operations; records producer-to-copy identity
and final original receipts. No raw payload ownership, cleanup, native training
implementation, checkpoint/fork lineage, disk persistence or restore permission.
"""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from threading import Lock, get_ident
from typing import Any, Callable, Iterator
from weakref import ReferenceType, WeakMethod, ref

from src.app.managed_experience import ManagedExperienceOwner
from src.app.managed_inbox_origins import ManagedInboxOrigins, _InboxOrigin
from src.app.inbox_erasure_observation import observe_inbox_erasure
from src.app.payload_ownership import PayloadOwnershipBusy
from src.app.payload_copy_budget import PayloadCopyBudget
from src.app.replay_write_origin import ManagedReplayWriteAccess
from src.core.data_lifecycle import LifecycleDeclaration
from src.core.experience import AppliedExperience
from src.core.replay_origin import (
    ReplayOriginAdmission,
    ReplayOriginData,
    ReplayOriginLimits,
    ReplayOriginPorts,
    replay_origin_metadata_digest,
)
from src.core.replay_write_origin import ReplayWriteLimits
from src.core.resource_sharing import TrainingPoll


def _weak(value) -> ReferenceType[Any]:
    try:
        return ref(value)
    except TypeError as error:
        raise ValueError("row origins require weak-referenceable original objects") from error


def _bound(port, owner, function) -> bool:
    return getattr(port, "__self__", None) is owner and getattr(port, "__func__", None) is function


@dataclass
class _Row:
    data: ReplayOriginData
    source: ReferenceType[Any]
    label: ReferenceType[Any]
    declaration: ReferenceType[Any]
    snapshot: ReferenceType[Any] | None = None
    features: ReferenceType[Any] | None = None
    targets: ReferenceType[Any] | None = None
    receipt: ReferenceType[Any] | None = None
    verified: bool = False
    metadata_digest: str = ""


class ManagedReplayOrigins:
    def __init__(
        self,
        owner,
        ports: ReplayOriginPorts,
        limits: ReplayOriginLimits,
        window_limits: ReplayWriteLimits,
        *,
        retain_inbox_origins: bool = False,
    ) -> None:
        if (
            type(owner) is not ManagedExperienceOwner
            or type(ports) is not ReplayOriginPorts
            or type(limits) is not ReplayOriginLimits
            or type(window_limits) is not ReplayWriteLimits
            or type(retain_inbox_origins) is not bool
        ):
            raise ValueError("row origins require original managed owner and exact ports/limits")
        self._owner = _weak(owner)
        self._ports, self._window_limits = ports, window_limits
        self._original_ports, self._port_functions = ports, tuple(vars(ports).values())
        self._admission = ReplayOriginAdmission(limits)
        self._original_admission = _weak(self._admission)
        self._rows: dict[int, _Row] = {}
        self._inbox_origins: ManagedInboxOrigins | None = None
        self._original_inbox_origins: ReferenceType[Any] | None = None
        self._inbox_history_birth: ReferenceType[Any] | None = None
        self._expiry_authority_birth: tuple | None = None
        self._enrolling_inbox_origins = False
        self._copy_slots = self._minimum_copy_slots = 0
        self._pending: _Row | None = None
        self._gate = Lock()
        self._poisoned = self._busy = self._started = False
        self._active_source: ReferenceType[Any] | None = None
        self._active_label: ReferenceType[Any] | None = None
        self._last_work = self._last_tick = 0
        with owner._operation(), owner._shared._runtime._exclusive():
            runtime = owner._shared._runtime
            runtime._require_open()
            model = ports.model_reference(runtime._candidate)
            self._anchors = {
                name: _weak(value)
                for name, value in dict(
                    runtime=runtime,
                    actor=runtime._actor,
                    learner=runtime._candidate,
                    model=model,
                    inbox=runtime._inbox,
                    budget=runtime._budget,
                    clock=runtime._inbox._clock,
                    budget_policy=runtime._budget.budget,
                    owner_policy=owner._limits,
                ).items()
            }
            self._version, self._lineage = runtime._candidate_version, runtime._payload_lineage
            self._lifecycle = _weak(owner._lifecycle) if owner._lifecycle is not None else None
            copy_budget = None if owner._lifecycle is None else owner._lifecycle._copy_budget
            self._original_copy_budget = None if copy_budget is None else _weak(copy_budget)
            self._original_copy_limits = None if copy_budget is None else _weak(copy_budget._limits)
            self._copy_charge_minimum = 0 if copy_budget is None else copy_budget._charged
            guard = runtime._inbox._payload_copy_guard
            self._copy_guard = (
                (
                    WeakMethod(guard)
                    if getattr(guard, "__self__", None) is not None
                    else _weak(guard)
                )
                if guard is not None
                else None
            )
            self._budget_origin = runtime._budget.started_at
            self._last_budget_clock = runtime._budget.last_clock
            # The lineage sentinel has no raw payload; preserve original identity.
            if runtime._budget.updates_completed or self._inventory(model):
                raise ValueError(
                    "row origin enrollment requires fresh work and empty native replay"
                )
            if retain_inbox_origins:
                self._enrolling_inbox_origins = True
                try:
                    self._inbox_origins = ManagedInboxOrigins(self)
                finally:
                    self._enrolling_inbox_origins = False
                self._original_inbox_origins = _weak(self._inbox_origins)
                self._inbox_history_birth = self._original_inbox_origins
            self._require_bindings(owner)
            self._read_tick(runtime)
            if retain_inbox_origins and owner._lifecycle is not None:
                from src.app.expiry_history_birth import bind_expiry_history

                bind_expiry_history(owner._lifecycle, self)

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        acquired_gate = self._gate
        if not acquired_gate.acquire(blocking=False):
            raise PayloadOwnershipBusy("row origin ledger is busy; reentry is unsupported")
        try:
            yield
        finally:
            acquired_gate.release()

    def _original_owner(self):
        owner = self._owner()
        if owner is None:
            raise ValueError("row origin original owner expired")
        return owner

    def _require_bindings(self, owner):
        if self._poisoned or self._original_admission() is not self._admission:
            raise ValueError("row origin ledger is uncertain or admission authority changed")
        self._history()
        if self._ports is not self._original_ports or any(
            a is not b for a, b in zip(vars(self._ports).values(), self._port_functions)
        ):
            raise ValueError("row origin original reference/integrity ports changed")
        runtime = owner._shared._runtime
        expected = dict(
            runtime=runtime,
            actor=runtime._actor,
            learner=runtime._candidate,
            model=self._ports.model_reference(runtime._candidate),
            inbox=runtime._inbox,
            budget=runtime._budget,
            clock=runtime._inbox._clock,
            budget_policy=runtime._budget.budget,
            owner_policy=owner._limits,
        )
        if (
            any(self._anchors[name]() is not value for name, value in expected.items())
            or runtime._payload_lineage is not self._lineage
            or runtime._candidate_version != self._version
        ):
            raise ValueError("row origin original runtime/model/budget/clock/lineage changed")
        inbox = runtime._inbox
        if owner._lifecycle is not (self._lifecycle() if self._lifecycle is not None else None):
            raise ValueError("row origin original lifecycle changed or expired")
        guard = self._copy_guard() if self._copy_guard is not None else None
        if (
            (self._copy_guard is not None and guard is None)
            or (guard is None and inbox._payload_copy_guard is not None)
            or (
                guard is not None
                and not (
                    inbox._payload_copy_guard is guard
                    or (
                        getattr(guard, "__self__", None) is not None
                        and _bound(
                            inbox._payload_copy_guard,
                            getattr(guard, "__self__"),
                            getattr(guard, "__func__"),
                        )
                    )
                )
            )
        ):
            raise ValueError("row origin original payload copy guard changed")
        if (
            runtime._budget.started_at != self._budget_origin
            or runtime._budget.last_clock < self._last_budget_clock
        ):
            raise ValueError("row origin original budget time origin was reset or rewound")
        self._last_budget_clock = runtime._budget.last_clock
        if not _bound(
            inbox._registration_guard, owner, ManagedExperienceOwner._require_registration
        ) or not _bound(inbox._training_guard, owner, ManagedExperienceOwner._eligible):
            raise ValueError("row origin original consent guards changed")
        if (
            type(runtime._budget.updates_completed) is not int
            or runtime._budget.updates_completed < self._last_work
        ):
            raise ValueError("row origin original work was rewound")
        self._admission.accounting(self._live_records(pending=True))
        return runtime, expected["model"]

    def _read_tick(self, runtime) -> int:
        tick = runtime._inbox._read_clock()
        if tick < self._last_tick:
            raise ValueError("row origin original clock was rewound")
        self._last_tick = tick
        return tick

    def _inventory(self, model) -> tuple[object, ...]:
        retained = self._ports.retained(model, self._window_limits.max_retained_snapshots)
        if (
            type(retained) is not tuple
            or len(retained) > self._window_limits.max_retained_snapshots
            or len({id(s) for s in retained}) != len(retained)
        ):
            raise ValueError("row origin retained reference port is invalid")
        return retained

    @staticmethod
    def _capture_declaration(owner, key) -> LifecycleDeclaration:
        # Capture already holds lifecycle time ownership. Mirror original owner
        # declaration checks and use its existing leased retention accessor.
        declaration = owner._catalog.get(key)
        if declaration is None:
            raise ValueError("unknown lifecycle declaration")
        if key in owner._revoked_keys:
            raise ValueError("lifecycle identity has been revoked")
        owner._limits.require_supported(declaration)
        if declaration.provenance.subject_id in owner._opted_out:
            raise ValueError("subject has opted out")
        if owner._lifecycle is not None:
            owner._lifecycle._require_access_leased()
        return declaration

    def _source(
        self, owner, runtime, source, label, now, *, lifecycle_leased=False
    ) -> LifecycleDeclaration:
        declaration = (
            self._capture_declaration(owner, source.key)
            if lifecycle_leased
            else owner._require_live(source.key)
        )
        if (
            runtime._inbox._experiences.get(source.key) is not source
            or runtime._inbox._labels.get(source.key) is not label
            or not owner._eligible(source, label)
            or source.model_version != label.model_version
            or source.model_version != runtime._base_actor_version
        ):
            raise ValueError("row origin source/label differs from original eligible inbox")
        if (
            now < source.observed_at
            or now - source.observed_at > self._admission.limits.max_age_ticks
        ):
            raise ValueError("row origin original source age expired")
        if source.key in runtime._inbox._erased:
            raise ValueError("row origin source has an original tombstone")
        return declaration

    def _verify_row(
        self, owner, runtime, snapshot, row, now, pending=False, *, lifecycle_leased=False
    ) -> None:
        if (
            type(row) is not _Row
            or row.snapshot is None
            or row.snapshot() is not snapshot
            or (not row.verified and not pending)
        ):
            raise ValueError("row origin is untracked, foreign or provisional")
        if row.metadata_digest != replay_origin_metadata_digest(
            row.data, self._admission.limits.max_metadata_bytes
        ):
            raise ValueError("row origin metadata integrity changed")
        source, label = row.source(), row.label()
        if source is None or label is None:
            raise ValueError("row origin original source/label reference expired")
        declaration = self._source(
            owner, runtime, source, label, now, lifecycle_leased=lifecycle_leased
        )
        if (
            row.declaration() is not declaration
            or row.data.key != source.key
            or row.data.event_id != label.event_id
            or row.data.subject_id != declaration.provenance.subject_id
            or row.data.source_id != declaration.provenance.source_id
            or row.data.actor_version != source.model_version
            or row.data.learner_version != self._version
            or row.data.observed_at != source.observed_at
            or row.data.arrived_at != label.arrived_at
        ):
            raise ValueError("row origin original declaration/metadata changed")
        features, targets = self._ports.payloads(snapshot)
        if (
            row.features is None
            or row.targets is None
            or row.features() is not features
            or row.targets() is not targets
        ):
            raise ValueError("row origin actual payload references changed or expired")
        size, digest = self._ports.fingerprint(snapshot, self._admission.limits.max_payload_bytes)
        if size != row.data.payload_bytes or digest != row.data.payload_digest:
            raise ValueError("row origin already-bound payload integrity changed")
        if (
            self._source(owner, runtime, source, label, now, lifecycle_leased=lifecycle_leased)
            is not declaration
        ):
            raise ValueError("row origin consent changed during integrity observation")
        if row.verified:
            self._receipt(runtime, row)

    @staticmethod
    def _receipt(runtime, row):
        receipt = row.receipt() if row.receipt is not None else None
        if (
            type(receipt) is not AppliedExperience
            or runtime._inbox._applied.get(row.data.key) is not receipt
            or (receipt.episode_id, receipt.sample_id) != row.data.key
            or receipt.event_id != row.data.event_id
            or receipt.actor_version != row.data.actor_version
            or receipt.observed_at != row.data.observed_at
            or receipt.arrived_at != row.data.arrived_at
            or receipt.update_number != row.data.update_number
            or receipt.learner_version != row.data.learner_version
            or receipt.update_number > runtime._budget.updates_completed
        ):
            raise ValueError("row origin original committed receipt changed or expired")
        return receipt

    def _verify_inventory(
        self, owner, runtime, retained, now, pending=False, *, lifecycle_leased=False
    ) -> None:
        for snapshot in retained:
            row = self._rows.get(id(snapshot))
            if row is None:
                raise ValueError("native replay has an untracked original row")
            self._verify_row(
                owner, runtime, snapshot, row, now, pending, lifecycle_leased=lifecycle_leased
            )

    def _prune(self, retained) -> None:
        identities = {id(s): s for s in retained}
        self._rows = {
            key: row
            for key, row in self._rows.items()
            if row.snapshot is not None
            and row.snapshot() is identities.get(key)
            and key in identities
        }

    def _update(self, stage, read) -> None:
        owner = self._original_owner()
        runtime, _ = self._require_bindings(owner)
        origin = read()
        if (
            origin.learner is not runtime._candidate
            or origin.learner_version != self._version
            or origin.completed_updates != runtime._budget.updates_completed
        ):
            raise ValueError("row origin original native producer differs")
        if stage == "started":
            declaration = self._source(
                owner, runtime, origin.source, origin.label, self._read_tick(runtime)
            )
            self._admission.start(self._live_records())
            self._active_source, self._active_label = _weak(origin.source), _weak(origin.label)
            self._started = True
            history = self._history()
            if history is not None:
                ManagedInboxOrigins.started(history, self, owner, runtime, origin, declaration)
        elif stage == "completed":
            if origin.receipt is not runtime._inbox._applied.get(origin.source.key):
                raise ValueError("row origin completion has no original receipt")
            history = self._history()
            if history is not None:
                ManagedInboxOrigins.completed(history, self, owner, runtime, origin)
            for row in self._rows.values():
                if (
                    not row.verified
                    and row.source() is origin.source
                    and row.label() is origin.label
                ):
                    row.receipt = _weak(origin.receipt)
            self._last_work = origin.completed_updates
        elif stage in ("uncertain", "committed_failure"):
            self._last_work = origin.completed_updates
        self._pending = None

    def _writes(self, stage, read) -> None:
        if not self._busy or not self._started:
            raise ValueError("row origin writes require the original active managed operation")
        owner = self._original_owner()
        runtime, model = self._require_bindings(owner)
        observed = read()
        origin, write = observed.origin, observed.write
        if (
            write.model is not model
            or origin.learner is not runtime._candidate
            or self._active_source is None
            or self._active_source() is not origin.source
            or self._active_label is None
            or self._active_label() is not origin.label
            or write.features is not origin.features
            or write.targets is not origin.targets
        ):
            raise ValueError("row origin write differs from original producer/model/inputs")
        now = self._read_tick(runtime)
        declaration = self._source(owner, runtime, origin.source, origin.label, now)
        if stage == "begin":
            retained = self._inventory(model)
            if (
                write.retained is None
                or len(write.retained) != len(retained)
                or any(a is not b for a, b in zip(write.retained, retained))
            ):
                raise ValueError("row origin begin retained references differ")
            self._verify_inventory(owner, runtime, retained, now, pending=True)
            self._prune(retained)
        elif stage == "before_copy":
            self._reserve_row(origin, write, declaration)
        elif stage == "copied":
            self._bind_copy(write)
        elif stage == "retained":
            retained = self._inventory(model)
            if (
                write.retained is None
                or len(write.retained) != len(retained)
                or any(a is not b for a, b in zip(write.retained, retained))
            ):
                raise ValueError("row origin final retained references differ")
            self._verify_inventory(owner, runtime, retained, now, pending=True)
            self._prune(retained)
            self._pending = None

    def _reserve_row(self, origin, write, declaration) -> None:
        self._pending = None  # Native oversized rows have no copied snapshot; charges stay spent.
        size = self._ports.copy_bytes(
            origin.features,
            origin.targets,
            write.row_start,
            write.row_count,
            self._admission.limits.max_payload_bytes,
        )
        data = ReplayOriginData(
            origin.source.key,
            origin.label.event_id,
            origin.source.model_version,
            origin.learner_version,
            declaration.provenance.subject_id,
            declaration.provenance.source_id,
            write.row_start,
            write.row_count,
            origin.source.observed_at,
            origin.label.arrived_at,
            origin.completed_updates + 1,
            size,
            "0" * 64,
        )
        self._admission.reserve(data, self._live_records())
        self._pending = _Row(data, _weak(origin.source), _weak(origin.label), _weak(declaration))

    def _bind_copy(self, write) -> None:
        row = self._pending
        if (
            row is None
            or write.snapshot is None
            or (write.row_start, write.row_count) != (row.data.row_start, row.data.row_count)
        ):
            raise ValueError("row origin copied snapshot has no original charged range")
        size, digest = self._ports.fingerprint(
            write.snapshot, self._admission.limits.max_payload_bytes
        )
        if size != row.data.payload_bytes:
            raise ValueError("row origin native copy differs from admitted byte bound")
        features, targets = self._ports.payloads(write.snapshot)
        row.data = replace(row.data, payload_digest=digest)
        row.metadata_digest = replay_origin_metadata_digest(
            row.data, self._admission.limits.max_metadata_bytes
        )
        row.snapshot, row.features, row.targets = (
            _weak(write.snapshot),
            _weak(features),
            _weak(targets),
        )
        if id(write.snapshot) in self._rows:
            raise ValueError("row origin native copy reused an existing snapshot identity")
        self._rows[id(write.snapshot)] = row
        self._pending = None

    def train_ready(self) -> TrainingPoll:
        with self._exclusive():
            owner = self._original_owner()
            with owner._operation():
                runtime, model = self._require_bindings(owner)
                with runtime._exclusive():
                    runtime._require_open()
                    self._verify_inventory(
                        owner, runtime, self._inventory(model), self._read_tick(runtime)
                    )
                    history = self._history()
                    if history is not None:
                        ManagedInboxOrigins.prune(history, runtime)
                        ManagedInboxOrigins.verify(
                            history, self, owner, runtime, self._read_tick(runtime)
                        )
                    before = runtime._budget.updates_completed
                access = ManagedReplayWriteAccess(
                    self._ports.model_reference,
                    self._writes,
                    self._window_limits,
                    update_observer=self._update,
                )
                self._busy, self._started = True, False
                try:
                    poll = owner._shared.train_ready(native_observer=access)
                    with runtime._exclusive():
                        runtime._require_open()
                        self._require_bindings(owner)
                        retained = self._inventory(model)
                        self._verify_inventory(
                            owner, runtime, retained, self._read_tick(runtime), pending=True
                        )
                        if runtime._budget.updates_completed != before + len(poll.updates):
                            raise ValueError(
                                "row origin final original work differs from returned receipts"
                            )
                        runtime._budget.before_final()
                        self._require_bindings(owner)
                        self._verify_inventory(
                            owner, runtime, retained, self._read_tick(runtime), pending=True
                        )
                        for snapshot in retained:
                            row = self._rows[id(snapshot)]
                            if not row.verified:
                                receipt = self._receipt(runtime, row)
                                if not any(receipt is item for item in poll.updates):
                                    raise ValueError(
                                        "row origin final receipt is outside original poll"
                                    )
                                row.verified = True
                        if history is not None:
                            ManagedInboxOrigins.finish(history, self, owner, runtime, poll.updates)
                        self._prune(retained)
                    return poll
                except BaseException:
                    if self._started:
                        self._poisoned = True
                    raise
                finally:
                    access.close()
                    self._pending = None
                    self._active_source = self._active_label = None
                    self._busy = False

    def origins(self) -> tuple[ReplayOriginData, ...]:
        with self._exclusive():
            owner = self._original_owner()
            with owner._operation(), owner._shared._runtime._exclusive():
                runtime, model = self._require_bindings(owner)
                runtime._require_open()
                history = self._history()
                if history is not None:
                    ManagedInboxOrigins.prune(history, runtime)
                    ManagedInboxOrigins.verify(
                        history, self, owner, runtime, self._read_tick(runtime)
                    )
                retained = self._inventory(model)
                self._verify_inventory(owner, runtime, retained, self._read_tick(runtime))
                if history is not None:
                    ManagedInboxOrigins.verify(
                        history, self, owner, runtime, self._read_tick(runtime)
                    )
                runtime._budget.before_final()
                self._require_bindings(owner)
                self._verify_inventory(owner, runtime, retained, self._read_tick(runtime))
                if history is not None:
                    ManagedInboxOrigins.verify(
                        history, self, owner, runtime, self._read_tick(runtime)
                    )
                self._prune(retained)
                return tuple(self._rows[id(snapshot)].data for snapshot in retained)

    def accounting(self):
        with self._exclusive():
            return self._admission.accounting(self._live_records(pending=True))

    def delete(self, keys):
        """Observe original lifecycle deletion without granting cleanup authority."""
        from src.app.managed_data_lifecycle import ManagedDataLifecycle

        with self._exclusive():
            owner = self._original_owner()
            with owner._operation():
                runtime, _ = self._require_bindings(owner)
                with runtime._exclusive():
                    history = self._history()
                    if history is None or type(owner._lifecycle) is not ManagedDataLifecycle:
                        raise ValueError(
                            "observed erasure requires original birth history/lifecycle"
                        )
                    before = ManagedInboxOrigins.require_erasure_ready(
                        history, self, owner, runtime, self._read_tick(runtime)
                    )
                    life = owner._lifecycle

            # Why: lifecycle.delete reacquires its original owner/registry/sharing
            # leases. Holding those here would cause nonblocking reentry failure.
            with observe_inbox_erasure(self, owner, runtime, history, life, before):
                return ManagedDataLifecycle.delete(life, keys)

    def _live_records(self, *, pending=False) -> int:
        if (
            type(self._copy_slots) is not int
            or type(self._minimum_copy_slots) is not int
            or self._minimum_copy_slots < 0
            or not self._minimum_copy_slots
            <= self._copy_slots
            <= self._admission.limits.max_live_records
        ):
            raise ValueError("original copied row reservations were rewound or corrupted")
        # Why: GC cannot safely refund shared witness capacity. Reservations stay
        # spent even after a failed copier or an expired weak target.
        history = self._history()
        history_count = 0
        if history is not None:
            # Counting must permit tombstoned, expired weak payloads to reach
            # pruning under owner/runtime leases. It never grants access.
            ManagedInboxOrigins._authority(history, self)
            history_count = (
                len(ManagedInboxOrigins._state(history, contents=False))
                + len(ManagedInboxOrigins._erased_state(history, contents=False))
                + len(ManagedInboxOrigins._untrained_state(history, contents=False))
            )
        return (
            len(self._rows)
            + self._copy_slots
            + history_count
            + int(pending and self._pending is not None)
        )

    def _history(self) -> ManagedInboxOrigins | None:
        history, original = self._inbox_origins, self._original_inbox_origins
        # Why: replacing the current two pointers cannot turn a default-off
        # ledger into an authority that observed earlier training.
        birth = self._inbox_history_birth
        if birth is None:
            if history is not None or original is not None:
                raise ValueError("inbox history requires enrollment at original ledger birth")
        elif (
            original is not birth
            or type(birth) is not ReferenceType
            or type(history) is not ManagedInboxOrigins
            or birth() is not history
        ):
            raise ValueError("original inbox history authority changed or expired")
        return history

    def _reserve_copied_inbox(self, record: _InboxOrigin) -> None:
        history = self._history()
        if (
            history is None
            or type(record) is not _InboxOrigin
            or type(record.verified) is not bool
            or not record.verified
        ):
            raise ValueError("copied inbox history requires original committed enrollment")
        ManagedInboxOrigins.state_stamp(history)
        if not any(record is original for original in history._records.values()):
            raise ValueError("copied inbox history differs from its original admitted witness")
        self._admission.reserve_inbox(record.data, self._live_records())
        self._copy_slots += 1
        self._minimum_copy_slots = self._copy_slots

    def _reserve_copied_erased_inbox(self, record) -> None:
        from src.app.erased_inbox_origins import ErasedInboxOrigin, erased_inbox_origin_stamp
        from src.app.checkpoint_erased_inbox import require_same_erased_inbox_history

        history = self._history()
        if history is None or type(record) is not ErasedInboxOrigin:
            raise ValueError("copied erased inbox history requires original committed enrollment")
        ManagedInboxOrigins.state_stamp(history)
        erased_inbox_origin_stamp(record, history._content_limits)
        original = history._erased_records.get(record.data.key)
        if original is None or original.data is not record.data:
            raise ValueError("copied erased inbox metadata lacks its original admitted witness")
        if record is not original:
            require_same_erased_inbox_history(original, record, history._content_limits)
        # Every actual inbox-copy stage spends another permanent metadata slot;
        # no deleted payload bytes are copied and failure never refunds it.
        self._admission.reserve_inbox(original.data, self._live_records())
        self._copy_slots += 1
        self._minimum_copy_slots = self._copy_slots

    def _reserve_copied_untrained_inbox(self, record) -> None:
        from src.app.untrained_inbox_origins import (
            UntrainedInboxOrigin,
            untrained_inbox_origin_stamp,
        )
        from src.app.checkpoint_untrained_inbox import require_same_untrained_inbox_history

        history = self._history()
        if history is None or type(record) is not UntrainedInboxOrigin:
            raise ValueError("copied untrained inbox history requires original observed enrollment")
        ManagedInboxOrigins.state_stamp(history)
        untrained_inbox_origin_stamp(record, history._content_limits)
        original = history._untrained_records.get(record.data.key)
        if original is None or original.data is not record.data:
            raise ValueError("copied untrained metadata lacks its original admitted witness")
        if record is not original:
            require_same_untrained_inbox_history(original, record, history._content_limits)
        # Why: each actual metadata copy spends the same original admission and
        # another permanent slot; absence of applied work grants no refund.
        self._admission.reserve_untrained(original.data, self._live_records())
        self._copy_slots += 1
        self._minimum_copy_slots = self._copy_slots

    def _reserve_copied_row(self, row: _Row) -> None:
        if type(row) is not _Row:
            raise ValueError("copied row reservation requires its original verified ledger row")
        snapshot = row.snapshot() if type(row) is _Row and row.snapshot is not None else None
        if snapshot is None or self._rows.get(id(snapshot)) is not row or not row.verified:
            raise ValueError("copied row reservation requires its original verified ledger row")
        self._admission.reserve(row.data, self._live_records())
        self._copy_slots += 1
        self._minimum_copy_slots = self._copy_slots

    def _require_copy_budget(self, owner):
        life = owner._lifecycle
        if life is None or life._copy_budget is None:
            raise ValueError("original replay copy budget/limits changed, expired or rewound")
        budget: PayloadCopyBudget = life._copy_budget
        if (
            budget is None
            or self._original_copy_budget is None
            or self._original_copy_budget() is not budget
            or self._original_copy_limits is None
            or self._original_copy_limits() is not budget._limits
            or life._policy.owned_payload_copies is not budget._limits
            or type(budget._charged) is not int
            or not self._copy_charge_minimum
            <= budget._charged
            <= budget._limits.max_lifetime_owned_bytes
        ):
            raise ValueError("original replay copy budget/limits changed, expired or rewound")
        self._copy_charge_minimum = budget._charged
        return budget

    @contextmanager
    def _lease_capture(self, owner) -> Iterator[Callable[[], object]]:
        """Validate while capture already holds original owner/runtime leases.

        Why: public origins() would reenter those nonblocking locks; original
        capture performs the resource checks using its already leased sampler.
        Capture attempts spend the same lifetime invocation/metadata allowance.
        """
        with self._exclusive():
            if self._original_owner() is not owner:
                raise ValueError("capture requires the ledger's original owner")
            runtime, model = self._require_bindings(owner)
            self._capture_open(owner, runtime)
            self._admission.start(self._live_records())
            active, thread = True, get_ident()
            checks = 0

            def verify():
                nonlocal checks
                if not active or get_ident() != thread:
                    raise ValueError("replay capture check expired or crossed threads")
                checks += 1
                if checks > self._window_limits.max_notifications:
                    raise ValueError("replay capture check allowance exhausted")
                current_runtime, current_model = self._require_bindings(owner)
                if current_runtime is not runtime or current_model is not model:
                    raise ValueError("replay capture original runtime/model changed")
                self._capture_open(owner, runtime)
                retained = self._inventory(model)
                self._verify_inventory(
                    owner, runtime, retained, self._read_tick(runtime), lifecycle_leased=True
                )
                self._prune(retained)
                history = self._history()
                if history is not None:
                    ManagedInboxOrigins.prune(history, runtime)
                    ManagedInboxOrigins.verify(
                        history,
                        self,
                        owner,
                        runtime,
                        self._read_tick(runtime),
                        lifecycle_leased=True,
                    )
                return model

            try:
                verify()
                yield verify
            finally:
                active = False
                owner = runtime = model = None

    @staticmethod
    def _capture_open(owner, runtime) -> None:
        # The ordinary runtime guard reacquires elapsed ownership through the
        # registry. Preserve its checks using the original capture lease instead.
        if runtime._retired:
            raise ValueError("candidate owner is retired after checkpoint handoff")
        if runtime._stopped:
            raise ValueError("candidate is stopped after uncertain native mutation")
        life = runtime._actor._payload_registry._lifecycle
        if life is not owner._lifecycle:
            raise ValueError("replay capture original registry/lifecycle changed")
        if life is not None:
            life._require_access_leased()
