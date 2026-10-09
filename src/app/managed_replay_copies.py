"""Original managed checkpoint-fork admission and weak copied row witnesses.

Inputs are the original row ledger, enrolled checkpoint controller and a trusted
builder-source port. Outputs are original copied row metadata under owner leases.
No snapshot/restore lineage, full capture permission, native policy certification,
model operation implementation, cleanup, persistence or scientific authority.
"""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable
from weakref import ReferenceType

from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.managed_replay_origins import ManagedReplayOrigins, _Row, _weak
from src.app.payload_ownership import PayloadOwnershipBusy, lease_payload_lock
from src.core.native_model_copy import ModelCopyLimits, observe_model_copies


@dataclass(frozen=True)
class _Copy:
    model: ReferenceType[Any]
    controller: ReferenceType[Any]
    builder: ReferenceType[Any]
    enrollment: int
    position: int
    attempt: int
    models_identity: int
    rows: tuple[_Row, ...]
    policy: ReferenceType[Any]
    probe: ReferenceType[Any]
    attempt_limit: int


class ManagedReplayCopies:
    def __init__(self, ledger: ManagedReplayOrigins, *, builder_source: Callable[[object], object]):
        if type(ledger) is not ManagedReplayOrigins or not callable(builder_source):
            raise ValueError(
                "managed replay copies require original ledger and builder-source port"
            )
        self._ledger = _weak(ledger)
        self._builder_source = self._original_builder_source = builder_source
        self._copies: dict[int, _Copy] = {}
        self._gate = Lock()

    @contextmanager
    def _exclusive(self):
        acquired_gate = self._gate
        if not acquired_gate.acquire(blocking=False):
            raise PayloadOwnershipBusy("managed replay copy witnesses are busy")
        try:
            yield
        finally:
            acquired_gate.release()

    def _original_ledger(self):
        ledger = self._ledger()
        if ledger is None or self._builder_source is not self._original_builder_source:
            raise ValueError("original replay ledger expired or builder-source port changed")
        return ledger

    def _controller(self, owner, controller):
        life = owner._lifecycle
        if life is None or life._copy_budget is None:
            raise ValueError("managed replay copying requires original bounded payload lifecycle")
        self._original_ledger()._require_copy_budget(owner)
        if (
            type(controller) is not CandidateCheckpointController
            or controller._shared is not owner._shared
        ):
            raise ValueError("replay copy requires original checkpoint controller")
        life._require_holder("checkpoint", controller)
        matches = [
            number
            for number, (kind, reference) in life._registry._holders.items()
            if kind == "checkpoint" and reference() is controller
        ]
        if len(matches) != 1 or controller._payload_ready is not True:
            raise ValueError("checkpoint controller is not originally enrolled and ready")
        if (
            type(controller._models) is not list
            or type(controller._attempts) is not int
            or not 0 <= len(controller._models) <= controller._attempts <= controller._attempt_limit
        ):
            raise ValueError("original checkpoint model/preparation history is corrupt")
        if self._builder_source(controller._build) is not owner._shared._runtime._candidate:
            raise ValueError("checkpoint builder differs from original candidate learner")
        return life, matches[0]

    @staticmethod
    def _source(ledger, owner, *, leased=False):
        runtime, model = ledger._require_bindings(owner)
        if leased:
            ledger._capture_open(owner, runtime)
        else:
            runtime._require_open()
        retained = ledger._inventory(model)
        ledger._verify_inventory(
            owner, runtime, retained, ledger._read_tick(runtime), lifecycle_leased=leased
        )
        return runtime, model, retained

    def restore(self, controller, checkpoint):
        """Observe the existing original restore operation, including failed forks.

        The controller owns all preparation/publication behavior. Later native
        restore or handoff can invalidate these constructor-only witnesses.
        """
        with self._exclusive():
            ledger = self._original_ledger()
            owner = ledger._original_owner()
            with ledger._exclusive(), owner._operation():
                life = owner._lifecycle
                if life is None:
                    raise ValueError("managed replay copying requires installed lifecycle")
                with life._registry._lease():
                    life, enrollment = self._controller(owner, controller)
                    runtime, model, retained = self._source(ledger, owner)
                    if not retained:
                        raise ValueError(
                            "managed replay copy requires original nonempty row origins"
                        )
                    runtime._budget.before_final()
                    builder = controller._build
                    position, attempt = len(controller._models), controller._attempts + 1
                    models_identity = id(controller._models)
                    policy, probe, attempt_limit = (
                        controller._policy,
                        controller._digest,
                        controller._attempt_limit,
                    )
                observed = False
                copying = ExitStack()
                reserve = observe = None

                def observer(stage, read, lookup):
                    nonlocal observed, reserve, observe
                    if stage == "before_copy":
                        reserve, observe = self._lease_copy(life, copying)
                    original = read()
                    current_life, current_enrollment = self._controller(owner, controller)
                    current_runtime, current_model, rows = self._source(ledger, owner, leased=True)
                    if (
                        current_life is not life
                        or current_enrollment != enrollment
                        or current_runtime is not runtime
                        or current_model is not model
                        or original.source is not model
                        or controller._build is not builder
                        or controller._attempts != attempt
                        or len(controller._models) != position
                        or id(controller._models) != models_identity
                        or controller._policy is not policy
                        or controller._digest is not probe
                        or controller._attempt_limit != attempt_limit
                        or len(rows) != len(retained)
                        or any(a is not b for a, b in zip(rows, retained))
                    ):
                        raise ValueError("original checkpoint copy producer/history changed")
                    runtime._budget._before_final_leased(observe)
                    self._source(ledger, owner, leased=True)
                    source_rows = tuple(ledger._rows[id(row)] for row in rows)
                    if stage == "before_copy":
                        if observed:
                            raise ValueError("original checkpoint copy already admitted")
                        self._admit(ledger, life, runtime, source_rows, reserve)
                        self._source(ledger, owner, leased=True)
                        observed = True
                    else:
                        if not observed or original.target is None:
                            raise ValueError("checkpoint copied rows have no original admission")
                        copied = self._bind(
                            ledger, owner, runtime, rows, source_rows, original.target, lookup
                        )
                        self._copies[id(original.target)] = _Copy(
                            _weak(original.target),
                            _weak(controller),
                            _weak(builder),
                            enrollment,
                            position,
                            attempt,
                            id(controller._models),
                            copied,
                            _weak(policy),
                            _weak(probe),
                            attempt_limit,
                        )
                        copying.close()

                limits = ModelCopyLimits(1, 2, 3 * ledger._window_limits.max_retained_snapshots + 4)
                try:
                    with observe_model_copies(model, observer, limits):
                        # No wrapper publication or catch can alter the original result/error.
                        return controller.restore(checkpoint)
                finally:
                    copying.close()

    @staticmethod
    def _lease_copy(life, stack):
        stack.enter_context(lease_payload_lock(life._registry._gate, "original copy registry"))
        stack.enter_context(lease_payload_lock(life._time_gate, "original copy retention time"))
        reserve = stack.enter_context(life._copy_budget._lease())
        observe = (
            None
            if life._sampler is None
            else stack.enter_context(life._sampler._lease_observation())
        )
        life._require_access_leased()
        return reserve, observe

    @staticmethod
    def _admit(ledger, life, runtime, source_rows, reserve):
        size = sum(row.data.payload_bytes for row in source_rows)
        if life._describe(runtime._candidate).payload_bytes != size:
            raise ValueError("original native replay footprint differs from verified row bytes")
        ledger._admission.start(ledger._live_records())
        for row in source_rows:
            ledger._reserve_copied_row(row)
        reserve(size)
        ledger._require_copy_budget(ledger._original_owner())

    @staticmethod
    def _bind(ledger, owner, runtime, originals, source_rows, target, lookup):
        if lookup(ledger._ports.model_reference(runtime._candidate)) is not target:
            raise ValueError("copied model differs from original deepcopy memo")
        retained = ledger._inventory(target)
        if len(retained) != len(originals):
            raise ValueError("copied replay inventory differs from original row count")
        result = []
        now = ledger._read_tick(runtime)
        for source, old, snapshot in zip(originals, source_rows, retained):
            features, targets = ledger._ports.payloads(snapshot)
            if (
                snapshot is source
                or lookup(source) is not snapshot
                or old.features is None
                or old.targets is None
                or features is old.features()
                or targets is old.targets()
                or lookup(old.features()) is not features
                or lookup(old.targets()) is not targets
            ):
                raise ValueError("copied replay row/payload differs from original memo identities")
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
            ledger._verify_row(owner, runtime, snapshot, row, now, lifecycle_leased=True)
            result.append(row)
        return tuple(result)

    def origins(self, controller, learner):
        """Return row metadata only for the actual originally retained copied model."""
        with self._exclusive():
            ledger = self._original_ledger()
            owner = ledger._original_owner()
            with ledger._exclusive(), owner._operation():
                life = owner._lifecycle
                if life is None:
                    raise ValueError("original copied replay lifecycle expired")
                with (
                    life._registry._lease(),
                    lease_payload_lock(life._time_gate, "copied origin retention time"),
                ):
                    life, enrollment = self._controller(owner, controller)
                    runtime, _, _ = self._source(ledger, owner, leased=True)
                    model = ledger._ports.model_reference(learner)
                    copy = self._copies.get(id(model))
                    self._require_copy(copy, controller, learner, model, enrollment)
                    assert copy is not None
                    ledger._admission.start(ledger._live_records())
                    self._verify(ledger, owner, runtime, model, copy)
                    runtime._budget.before_final()
                    self._controller(owner, controller)
                    self._source(ledger, owner, leased=True)
                    self._verify(ledger, owner, runtime, model, copy)
                    return tuple(row.data for row in copy.rows)

    @staticmethod
    def _require_copy(
        copy: _Copy | None, controller: CandidateCheckpointController, learner, model, enrollment
    ):
        if (
            type(copy) is not _Copy
            or copy.model() is not model
            or copy.controller() is not controller
            or copy.builder() is not controller._build
            or copy.enrollment != enrollment
            or copy.models_identity != id(controller._models)
            or not 1 <= copy.attempt <= controller._attempts
            or not 0 <= copy.position < len(controller._models)
            or controller._models[copy.position] is not learner
            or copy.policy() is not controller._policy
            or copy.probe() is not controller._digest
            or copy.attempt_limit != controller._attempt_limit
        ):
            raise ValueError("copied replay lacks original retained checkpoint holder/history")

    @staticmethod
    def _verify(ledger, owner, runtime, model, copy):
        retained = ledger._inventory(model)
        if len(retained) != len(copy.rows):
            raise ValueError("copied replay inventory changed or restored without witnesses")
        now = ledger._read_tick(runtime)
        for snapshot, row in zip(retained, copy.rows):
            ledger._verify_row(owner, runtime, snapshot, row, now, lifecycle_leased=True)
