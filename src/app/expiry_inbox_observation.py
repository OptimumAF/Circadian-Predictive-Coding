"""Original birth-enrolled synchronous direct expiry observation.

Observation owns only a nonblocking metadata lease, weak/scalar proofs and paid
payload-free staged maps. Original lifecycle ports still decide/perform cleanup.
Observer faults cannot skip raw removal; publication follows whole cleanup and
the final pure proof. Native exceptions retain their original precedence.
"""

from contextvars import ContextVar, Token
import sys
from threading import Lock, get_ident
from typing import Any, NamedTuple
from weakref import ReferenceType, ref

from src.app.erased_inbox_origins import _counter, _schema
from src.app.expiry_cleanup_proof import observe_cleanup_graph, require_cleanup_graph
from src.app.expiry_history_birth import _require_birth_tuple, require_expiry_birth_source
from src.app.expiry_inbox_qualification import _require_key
from src.app.expiry_untrained_qualification import observe_expiry_arrivals
from src.core.data_retention import DataCleanupReport
from src.core.checkpoint_content import CheckpointContentLimits


_CURRENT: ContextVar[Any] = ContextVar("original_inbox_expiry", default=None)
_ACCESS: ContextVar[Any] = ContextVar("original_inbox_expiry_access", default=None)


class ExpiryObservationError(ValueError):
    """Mandatory cleanup succeeded, but original history could not be proven."""

    def __init__(self, code: str, report: DataCleanupReport) -> None:
        if (
            type(code) is not str
            or code not in ("expiry-observation-refused", "expiry-publication-refused")
            or type(report) is not DataCleanupReport
        ):
            raise ValueError("expiry observation error requires trusted code and cleanup report")
        self.code, self.report = code, report
        super().__init__(code)


def _anchors(life: Any, ledger: Any, runtime: Any) -> tuple:
    owner = life._owner
    result = []
    limits = ledger._inbox_origins._content_limits
    _schema(limits, CheckpointContentLimits)
    CheckpointContentLimits.__post_init__(limits)
    nodes, size = 0, 0
    # Preflight the combined scalar/text graph before allocating its projection.
    for mapping in (owner._declaration_ticks, owner._declaration_seconds):
        if type(mapping) is not dict or len(mapping) > min(
            owner._limits.max_approved_records, 4096
        ):
            raise ValueError("expiry original declaration anchors exceed their bound")
        nodes += len(mapping) * 6 + 2
        for key in mapping:
            _require_key(key, limits.max_metadata_bytes)
            size += sum(len(part) * 10 + 16 for part in key) + 64
            if nodes > limits.max_nodes or size > limits.max_metadata_bytes:
                raise ValueError(
                    "expiry original declaration anchors exceed aggregate original limits"
                )
    for mapping in (owner._declaration_ticks, owner._declaration_seconds):
        if type(mapping) is not dict or len(mapping) > min(
            owner._limits.max_approved_records, 4096
        ):
            raise ValueError("expiry original declaration anchors exceed their bound")
        values = []
        for key, value in mapping.items():
            _require_key(key, 4096)
            if value is not None:
                if type(value) is int:
                    _counter(value)
                elif type(value) is not float or not (0 <= value < float("inf")):
                    raise ValueError("expiry requires original finite declaration anchors")
            values.append((key, value))
        result.append((id(mapping), tuple(values)))
    for value in (
        life._admitted_bytes,
        runtime._budget.updates_completed,
        ledger._copy_slots,
        ledger._minimum_copy_slots,
        life._registry._total,
    ):
        _counter(value)
    copies = life._copy_budget
    if copies is not None:
        _counter(copies._charged)
    return (
        tuple(result),
        life._admitted_bytes,
        runtime._budget.updates_completed,
        ledger._copy_slots,
        ledger._minimum_copy_slots,
        life._registry._total,
        None if copies is None else copies._charged,
    )


class _ExpiryProof(NamedTuple):
    birth: tuple
    before: tuple
    graph: Any
    anchors: tuple
    now: int
    life: Any
    ledger: Any
    owner: Any
    runtime: Any
    history: Any
    gate: Any


class _ExpiryObservation:
    __slots__ = (
        "life",
        "ledger",
        "owner",
        "runtime",
        "history",
        "gate",
        "thread",
        "context",
        "access",
        "close_access",
        "closed",
        "used",
        "committed",
        "before",
        "proof",
        "staged",
        "fault",
        "active",
    )

    def __init__(self, life: Any) -> None:
        self.life = ref(life)
        self.ledger = self.owner = self.runtime = self.history = None
        self.gate = None
        self.thread = get_ident()
        self.context: Token[Any] | None = None
        self.access: Token[Any] | None = None
        self.close_access: Token[Any] | None = None
        self.closed = self.used = self.committed = False
        self.before = self.proof = self.staged = None
        self.fault = False
        self.active = False

    def _require_access(self) -> None:
        self._flags()
        if (
            self.closed
            or type(self.thread) is not int
            or self.thread != get_ident()
            or _CURRENT.get() is not self
            or self.context is None
            or self.access is None
        ):
            raise ValueError("expiry observation requires original active thread and context")
        # A copied context cannot reset a token created in the original context.
        _ACCESS.reset(self.access)
        self.access = _ACCESS.set(self)

    def _flags(self) -> None:
        if any(
            type(value) is not bool
            for value in (self.closed, self.used, self.committed, self.fault, self.active)
        ):
            raise ValueError("expiry observation requires exact original scalar flags")
        if self.proof is not None and self.active is not True:
            raise ValueError("expiry observation original ON marker changed")

    def __enter__(self):
        from src.app.managed_data_lifecycle import ManagedDataLifecycle

        caller = sys._getframe(1)
        life = self.life()
        if (
            caller.f_code is not ManagedDataLifecycle.expire.__code__
            or caller.f_locals.get("self") is not life
            or self.closed
            or self.context is not None
            or self.thread != get_ident()
        ):
            raise ValueError("expiry observation requires original direct lifecycle expiry")
        self.active = life._expiry_history_on is not False or life._expiry_history_birth is not None
        if not self.active:
            return self
        # Why: nested scopes must leave the enclosing context and lexical gate
        # untouched. The original owner/holder locks decide mandatory reentry.
        if _CURRENT.get() is not None:
            self.fault = True
            return self
        self.context = _CURRENT.set(self)
        self.close_access = _ACCESS.set(self)
        self.access = _ACCESS.set(self)
        try:
            require_expiry_birth_source(life)
            _require_birth_tuple(life._expiry_authority, bounded=True)
            history = life._expiry_history_birth()
            ledger = history._ledger()
            gate = ledger._gate
            if type(gate) is not type(Lock()) or not gate.acquire(blocking=False):
                raise ValueError("expiry original metadata ledger is busy")
            self.gate = gate
            owner, runtime = life._owner, life._shared._runtime
            self.ledger, self.owner, self.runtime, self.history = tuple(
                ref(value) for value in (ledger, owner, runtime, history)
            )
            with owner._operation(), runtime._exclusive():
                now = life._clock._time
                before = observe_expiry_arrivals(history, ledger, owner, runtime, now)
                graph = observe_cleanup_graph(life)
                anchors = _anchors(life, ledger, runtime)
                self.before = before
                self.proof = _ExpiryProof(
                    life._expiry_authority,
                    before,
                    graph,
                    anchors,
                    now,
                    self.life,
                    self.ledger,
                    self.owner,
                    self.runtime,
                    self.history,
                    gate,
                )
        except BaseException:
            # Never retain an observer exception, text, traceback or raw borrow.
            self.fault = True
        return self

    def close(self, gate, context, access, thread=None) -> None:
        from src.app.managed_data_lifecycle import ManagedDataLifecycle

        caller = sys._getframe(1)
        if (
            type(thread) is not int
            or thread != get_ident()
            or caller.f_code is not ManagedDataLifecycle.expire.__code__
            or caller.f_locals.get("observation") is not self
            or any(
                caller.f_locals.get(name) is not value
                for name, value in (
                    ("gate", gate),
                    ("context", context),
                    ("access", access),
                    ("thread", thread),
                )
            )
        ):
            raise ValueError("expiry close requires original lexical expiry thread/context")
        # Scope cleanup may never replace the genuine native/holder error. The
        # original gate is also pinned in expire's independent lexical proof.
        try:
            if access is not None:
                _ACCESS.reset(access)
        except BaseException:
            pass
        try:
            if context is not None:
                _CURRENT.reset(context)
        except BaseException:
            pass
        finally:
            if gate is not None:
                try:
                    gate.release()
                except BaseException:
                    pass
            self.closed = True
            self.ledger = self.owner = self.runtime = self.history = None
            self.before = self.proof = self.staged = self.gate = None
            self.context = self.access = self.close_access = None

    def _require_original(self, proof: Any) -> tuple:
        self._require_access()
        if (
            type(proof) is not _ExpiryProof
            or self.proof is not proof
            or self.before is not proof.before
        ):
            raise ValueError("expiry original lexical observation proof changed")
        life, ledger, owner, runtime, history = tuple(
            value()
            for value in (proof.life, proof.ledger, proof.owner, proof.runtime, proof.history)
        )
        if any(value is None for value in (life, ledger, owner, runtime, history)):
            raise ValueError("expiry original observation owner expired")
        require_expiry_birth_source(life)
        _counter(life._clock._time)
        if (
            life._expiry_authority is not proof.birth
            or self.life is not proof.life
            or any(
                current is not original
                for current, original in zip(
                    (self.ledger, self.owner, self.runtime, self.history),
                    (proof.ledger, proof.owner, proof.runtime, proof.history),
                )
            )
            or ledger._gate is not proof.gate
            or self.gate is not proof.gate
            or not proof.gate.locked()
            or life._clock._time != proof.now
            or _anchors(life, ledger, runtime) != proof.anchors
        ):
            raise ValueError("expiry original birth/authority/time/work/charges changed")
        return life, ledger, owner, runtime, history

    def before_cleanup(self, proof: Any, groups: Any) -> None:
        try:
            self._flags()
            if not self.active or self.fault:
                return
            life, _, _, _, _ = self._require_original(proof)
            require_cleanup_graph(proof.graph, life, final=False, groups=groups)
        except BaseException:
            self.fault = True

    def finish(self, proof: Any, report: DataCleanupReport, staged: Any, groups: Any) -> None:
        from src.app.expiry_inbox_transition import require_final_expiry_transition

        try:
            self._flags()
            if not self.active or self.fault:
                return
            life, ledger, owner, runtime, history = self._require_original(proof)
            if (
                not self.used
                or not self.committed
                or self.staged is None
                or self.staged is not staged
            ):
                raise ValueError("expiry original inbox removal was not observed")
            require_cleanup_graph(proof.graph, life, final=True, groups=groups)
            require_final_expiry_transition(
                self.staged, history, ledger, owner, runtime, proof.now, report
            )
            staged = self.staged
            # Only plain, prepaid payload-free assignments remain. There are no
            # native, clock, admission, capture or observer callbacks afterward.
            history._records, history._sealed, history._storage = (
                staged.live[0],
                staged.live[1],
                staged.live,
            )
            history._erased_records, history._erased_sealed, history._erased_storage = (
                staged.erased[0],
                staged.erased[1],
                staged.erased,
            )
            history._untrained_records, history._untrained_sealed, history._untrained_storage = (
                staged.untrained[0],
                staged.untrained[1],
                staged.untrained,
            )
        except BaseException:
            self.fault = True


def expiry_observation(life: Any) -> _ExpiryObservation:
    return _ExpiryObservation(life)


def _lexical_proof(scope: Any) -> Any:
    from src.app.managed_data_lifecycle import ManagedDataLifecycle

    # Why: ports can edit scope fields. Only the still-active exact original
    # direct expiry frame owns the immutable pre-port proof used for payment.
    for depth in range(1, 9):
        try:
            frame = sys._getframe(depth)
        except ValueError:
            break
        if frame.f_code is ManagedDataLifecycle.expire.__code__:
            if frame.f_locals.get("observation") is not scope:
                break
            return frame.f_locals.get("proof")
    raise ValueError("expiry preparation requires its original lexical direct-expiry proof")


def prepare_expiry_inbox_erasure(inbox: Any, keys: tuple, prepared: dict) -> Any:
    """The actual original inbox commit stages paid metadata before its pop."""
    scope = _CURRENT.get()
    if scope is None:
        return None
    from src.app.experience_inbox import ExperienceInbox
    from src.app.expiry_inbox_transition import prepay_expiry_transition

    if type(scope) is not _ExpiryObservation:
        return None
    try:
        scope._flags()
        if scope.fault:
            return None
        proof = _lexical_proof(scope)
        life, ledger, owner, runtime, history = scope._require_original(proof)
        if runtime._inbox is not inbox:
            return None
        caller = sys._getframe(1)
        if (
            scope.used
            or caller.f_code is not ExperienceInbox._commit_erased_history.__code__
            or caller.f_locals.get("self") is not inbox
        ):
            # Dispatch calls through the existing erasure observation module.
            outer = sys._getframe(2)
            if (
                scope.used
                or outer.f_code is not ExperienceInbox._commit_erased_history.__code__
                or outer.f_locals.get("self") is not inbox
            ):
                raise ValueError("expiry observation requires original single inbox commit")
        scope.used = True
        staged = prepay_expiry_transition(
            history, ledger, owner, runtime, proof.now, proof.before, keys, prepared
        )
        scope._require_original(proof)
        scope.staged = staged
        return scope
    except BaseException:
        scope.fault = True
        return None


def commit_expiry_inbox_erasure(scope: Any) -> None:
    """Record actual removal; final cleanup proof owns eventual publication."""
    if type(scope) is not _ExpiryObservation:
        return
    try:
        scope._require_access()
        if _CURRENT.get() is not scope or scope.staged is None or scope.committed:
            raise ValueError("expiry original deferred removal transition changed")
        scope.committed = True
    except BaseException:
        scope.fault = True
