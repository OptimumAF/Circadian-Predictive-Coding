"""Original birth-enrolled synchronous direct expiry observation.

Observation owns only a nonblocking metadata lease, weak/scalar proofs and paid
payload-free staged maps. Original lifecycle ports still decide/perform cleanup.
Observer faults cannot skip raw removal; publication follows whole cleanup and
the final pure proof. Native exceptions retain their original precedence.
"""

from contextvars import ContextVar, Token
import sys
from math import isfinite
from threading import Lock, get_ident
from typing import Any, NamedTuple
from weakref import ref

from src.app.erased_inbox_origins import _counter, _schema
from src.app.expiry_cleanup_proof import (
    CleanupGraphProof,
    observe_cleanup_graph,
    require_cleanup_graph,
)
from src.app.expiry_release_proof import (
    pin_release_methods,
    require_cleanup_release_dispatch,
    require_release_methods,
)
from src.app.expiry_history_birth import _require_birth_tuple, require_expiry_birth_source
from src.app.expiry_validation_proof import (
    pin_validation_types,
    require_original_metadata_dispatch,
    require_validation_types,
)
from src.app.managed_inbox_origins import ManagedInboxOrigins
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
    # Qualification imports the lifecycle; keep kernels eager and this audit
    # dependency local so either public module may be imported first.
    from src.app.expiry_inbox_qualification import _require_key

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
        self.ledger: Any = None
        self.owner: Any = None
        self.runtime: Any = None
        self.history: Any = None
        self.gate: Any = None
        self.thread = get_ident()
        self.context: Token[Any] | None = None
        self.access: Token[Any] | None = None
        self.close_access: Token[Any] | None = None
        self.closed = self.used = self.committed = False
        self.before: Any = None
        self.proof: Any = None
        self.staged: Any = None
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
        from src.app.managed_data_lifecycle import (
            ManagedDataLifecycle,
            _ORIGINAL_EXPIRY_CODE,
            _EXPIRY_GETTER_PINS,
            _original_expiry_fields,
        )
        from src.app.expiry_untrained_qualification import observe_expiry_arrivals

        caller = sys._getframe(1)
        life = self.life()
        if (
            life is None
            or type(life) is not ManagedDataLifecycle
            or caller.f_code is not _ORIGINAL_EXPIRY_CODE
            or caller.f_locals.get("self") is not life
            or self.closed
            or self.context is not None
            or self.thread != get_ident()
        ):
            raise ValueError("expiry observation requires original direct lifecycle expiry")
        try:
            fields = _original_expiry_fields(life)
            self.active = (
                fields["_expiry_history_on"] is not False
                or fields["_expiry_history_birth"] is not None
            )
        except BaseException:
            self.active = True
            mark_expiry_fault(self)
            return self
        if not self.active:
            return self
        # Why: nested scopes must leave the enclosing context and lexical gate
        # untouched. The original owner/holder locks decide mandatory reentry.
        if _CURRENT.get() is not None:
            mark_expiry_fault(self)
            return self
        self.context = _CURRENT.set(self)
        self.close_access = _ACCESS.set(self)
        self.access = _ACCESS.set(self)
        try:
            require_original_metadata_dispatch()
            require_validation_types(_EXPIRY_GETTER_PINS)
            require_validation_types(_OBSERVER_GETTER_PINS)
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
            require_cleanup_release_dispatch(life)
            require_release_methods(self, _EXPIRY_RELEASE_PINS, slotted=True)
            with owner._operation(), runtime._exclusive():
                now = life._clock._time
                before = observe_expiry_arrivals(history, ledger, owner, runtime, now)
                ManagedInboxOrigins._authority(history, ledger)
                graph = observe_cleanup_graph(life, history._content_limits)
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
            mark_expiry_fault(self)
        return self

    def close(self, gate, context, access, thread=None) -> None:
        from src.app.managed_data_lifecycle import _ORIGINAL_EXPIRY_CODE

        caller = sys._getframe(1)
        if (
            type(thread) is not int
            or thread != get_ident()
            or caller.f_code is not _ORIGINAL_EXPIRY_CODE
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
            # Original slot writes cannot invoke a late foreign __setattr__.
            _ORIGINAL_SCOPE_SLOTS["closed"].__set__(self, True)
            for name in _CLEARED_SCOPE_FIELDS:
                _ORIGINAL_SCOPE_SLOTS[name].__set__(self, None)

    def _require_original(self, proof: Any) -> tuple:
        from src.app.managed_data_lifecycle import _EXPIRY_GETTER_PINS

        require_original_metadata_dispatch()
        require_validation_types(_EXPIRY_GETTER_PINS)
        require_validation_types(_OBSERVER_GETTER_PINS)
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
        ManagedInboxOrigins._authority(history, ledger)
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
            require_validation_types(_OBSERVER_GETTER_PINS)
            require_release_methods(self, _EXPIRY_RELEASE_PINS, slotted=True)
            self._flags()
            if not self.active or self.fault:
                return
            life, ledger, _, _, history = self._require_original(proof)
            require_cleanup_release_dispatch(life)
            ManagedInboxOrigins._authority(history, ledger)
            require_cleanup_graph(
                proof.graph, life, history._content_limits, final=False, groups=groups
            )
        except BaseException:
            mark_expiry_fault(self)

    def finish(self, proof: Any, report: DataCleanupReport, staged: Any, groups: Any) -> None:
        from src.app.expiry_inbox_transition import require_final_expiry_transition
        from src.app.managed_data_lifecycle import _EXPIRY_GETTER_PINS

        leased_gate = None
        try:
            require_validation_types(_OBSERVER_GETTER_PINS)
            require_release_methods(self, _EXPIRY_RELEASE_PINS, slotted=True)
            self._flags()
            if not self.active or self.fault:
                return
            require_original_metadata_dispatch()
            require_validation_types(_EXPIRY_GETTER_PINS)
            require_validation_types(_OBSERVER_GETTER_PINS)
            life, ledger, owner, runtime, history = self._require_original(proof)
            require_cleanup_release_dispatch(life)
            if (
                not self.used
                or not self.committed
                or self.staged is None
                or self.staged is not staged
            ):
                raise ValueError("expiry original inbox removal was not observed")
            ManagedInboxOrigins._authority(history, ledger)
            require_cleanup_graph(
                proof.graph, life, history._content_limits, final=True, groups=groups
            )
            require_final_expiry_transition(
                staged, history, ledger, owner, runtime, proof.now, report
            )
            # All opaque cleanup/report calls have returned. Reprove the entire
            # graph under the ORIGINAL time gate, without another provider read.
            time_gate, timing = _lexical_time(self, proof)
            require_original_metadata_dispatch()
            require_validation_types(_EXPIRY_GETTER_PINS)
            require_validation_types(_OBSERVER_GETTER_PINS)
            require_release_methods(self, _EXPIRY_RELEASE_PINS, slotted=True)
            self._flags()
            if (
                self.active is not True
                or self.fault is not False
                or self.used is not True
                or self.committed is not True
            ):
                raise ValueError("expiry final original observation status changed")
            _require_time(life, runtime, proof, time_gate, timing)
            if not time_gate.acquire(blocking=False):
                raise ValueError("expiry original final time gate is busy")
            leased_gate = time_gate
            require_original_metadata_dispatch()
            require_validation_types(_EXPIRY_GETTER_PINS)
            require_validation_types(_OBSERVER_GETTER_PINS)
            require_release_methods(self, _EXPIRY_RELEASE_PINS, slotted=True)
            if self.staged is not staged or _lexical_proof(self) is not proof:
                raise ValueError("expiry original lexical staged proof changed")
            self._require_original(proof)
            require_cleanup_release_dispatch(life)
            ManagedInboxOrigins._authority(history, ledger)
            require_cleanup_graph(
                proof.graph, life, history._content_limits, final=True, groups=groups
            )
            require_final_expiry_transition(
                staged, history, ledger, owner, runtime, proof.now, report, pure_final=True
            )
            _require_time(life, runtime, proof, time_gate, timing)
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
            mark_expiry_fault(self)
        finally:
            if leased_gate is not None:
                leased_gate.release()


def expiry_observation(life: Any) -> _ExpiryObservation:
    return _ExpiryObservation(life)


def _lexical_proof(scope: Any) -> Any:
    from src.app.managed_data_lifecycle import _ORIGINAL_EXPIRY_CODE

    # Why: ports can edit scope fields. Only the still-active exact original
    # direct expiry frame owns the immutable pre-port proof used for payment.
    for depth in range(1, 9):
        try:
            frame = sys._getframe(depth)
        except ValueError:
            break
        if frame.f_code is _ORIGINAL_EXPIRY_CODE:
            if frame.f_locals.get("observation") is not scope:
                break
            return frame.f_locals.get("proof")
    raise ValueError("expiry preparation requires its original lexical direct-expiry proof")


def _lexical_time(scope: Any, proof: Any) -> tuple:
    from src.app.managed_data_lifecycle import _ORIGINAL_EXPIRY_CODE

    for depth in range(1, 9):
        try:
            frame = sys._getframe(depth)
        except ValueError:
            break
        if frame.f_code is _ORIGINAL_EXPIRY_CODE:
            if frame.f_locals.get("observation") is scope and frame.f_locals.get("proof") is proof:
                return frame.f_locals.get("time_gate"), frame.f_locals.get("timing")
            break
    raise ValueError("expiry final time proof requires original lexical expiry")


def _require_time(life: Any, runtime: Any, proof: Any, gate: Any, timing: Any) -> None:
    from src.app.managed_data_lifecycle import _EXPIRY_TIME_PINS

    require_release_methods(life, _EXPIRY_TIME_PINS)
    if (
        type(gate) is not type(Lock())
        or life._time_gate is not gate
        or id(gate) != proof.birth[1][11]
    ):
        raise ValueError("expiry original time gate changed")
    if (
        type(timing) is not tuple
        or len(timing) != 3
        or type(timing[0]) is not int
        or timing[0] != id(gate)
    ):
        raise ValueError("expiry original scalar timing token changed")
    now, seconds = timing[1:]
    _counter(now)
    if now != proof.now or type(life._last_tick) is not int or life._last_tick != now:
        raise ValueError("expiry accepted original due tick changed")
    if seconds is None:
        if life._policy.max_retention_seconds is not None or life._last_seconds is not None:
            raise ValueError("expiry original untimed scalar state changed")
    elif (
        type(seconds) is not float
        or not isfinite(seconds)
        or seconds < 0
        or type(life._last_seconds) is not float
        or life._last_seconds != seconds
    ):
        raise ValueError("expiry accepted original elapsed scalar changed")
    if (
        life._auxiliary_started_at is not None
        or life._failed is not False
        or life._retention_fault is not False
        or runtime._stopped is not False
    ):
        raise ValueError("expiry final original cleared healthy state changed")


def _has_enclosing_expiry_scope(scope: Any) -> bool:
    from src.app.managed_data_lifecycle import _ORIGINAL_EXPIRY_CODE

    for depth in range(1, 9):
        try:
            frame = sys._getframe(depth)
        except ValueError:
            break
        if frame.f_code is _ORIGINAL_EXPIRY_CODE:
            return frame.f_locals.get("observation") is not scope
    return False


def observe_expiry_time_dispatch(life: Any) -> bool:
    """Refusal is sticky locally while original raw dispatch still executes."""
    scope = _CURRENT.get()
    if scope is None:
        return False
    if type(scope) is not _ExpiryObservation:
        return True
    # Why: a refused nested expiry owns a different observation. Its local
    # refusal must not mark the enclosing scope before original raw reentry.
    if _has_enclosing_expiry_scope(scope):
        return True
    from src.app.managed_data_lifecycle import _EXPIRY_TIME_PINS, _EXPIRY_GETTER_PINS

    try:
        require_original_metadata_dispatch()
        require_validation_types(_EXPIRY_GETTER_PINS)
        require_validation_types(_OBSERVER_GETTER_PINS)
        require_release_methods(life, _EXPIRY_TIME_PINS)
        proof = _lexical_proof(scope)
        if (
            type(proof) is not _ExpiryProof
            or type(life._time_gate) is not type(Lock())
            or id(life._time_gate) != proof.birth[1][11]
        ):
            raise ValueError("expiry original time dispatch/gate changed")
        return scope.fault is not False
    except BaseException:
        mark_expiry_fault(scope)
        return True


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
        require_validation_types(_OBSERVER_GETTER_PINS)
        require_release_methods(scope, _EXPIRY_RELEASE_PINS, slotted=True)
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
        mark_expiry_fault(scope)
        return None


def commit_expiry_inbox_erasure(scope: Any) -> None:
    """Record actual removal; final cleanup proof owns eventual publication."""
    if type(scope) is not _ExpiryObservation:
        return
    try:
        require_validation_types(_OBSERVER_GETTER_PINS)
        require_release_methods(scope, _EXPIRY_RELEASE_PINS, slotted=True)
        scope._require_access()
        if _CURRENT.get() is not scope or scope.staged is None or scope.committed:
            raise ValueError("expiry original deferred removal transition changed")
        scope.committed = True
    except BaseException:
        mark_expiry_fault(scope)


_ORIGINAL_SCOPE_SLOTS = {
    name: _ExpiryObservation.__dict__[name] for name in _ExpiryObservation.__slots__
}
_CLEARED_SCOPE_FIELDS = (
    "ledger",
    "owner",
    "runtime",
    "history",
    "before",
    "proof",
    "staged",
    "gate",
    "context",
    "access",
    "close_access",
)


def mark_expiry_fault(scope: Any) -> None:
    """A refusal cannot invoke a foreign setter or retain its exception."""
    if type(scope) is _ExpiryObservation:
        _ORIGINAL_SCOPE_SLOTS["fault"].__set__(scope, True)


_EXPIRY_RELEASE_PINS = pin_release_methods(
    _ExpiryObservation,
    (
        "__init__",
        "__enter__",
        "close",
        "_flags",
        "_require_access",
        "_require_original",
        "before_cleanup",
        "finish",
    ),
)
_ORIGINAL_EXPIRY_ENTER = _ExpiryObservation.__enter__
_ORIGINAL_EXPIRY_CLOSE = _ExpiryObservation.close
_ORIGINAL_EXPIRY_BEFORE = _ExpiryObservation.before_cleanup
_ORIGINAL_EXPIRY_FINISH = _ExpiryObservation.finish
_ORIGINAL_EXPIRY_FACTORY = expiry_observation
_OBSERVER_GETTER_PINS = pin_validation_types(
    (_ExpiryObservation, _ExpiryProof, CleanupGraphProof), getters_only=True
)
