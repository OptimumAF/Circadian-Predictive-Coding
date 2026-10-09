"""Trusted synchronous observation of an actual original inbox erasure.

The original ledger scopes its existing lifecycle deletion. Preparation proves
and pays metadata before payload removal; publication uses exact plain maps.
No persisted inbox fields, consent authority, payload ownership or cleanup.
"""

from contextvars import ContextVar, Token
from dataclasses import dataclass
import sys
from threading import get_ident
from typing import Any


_CURRENT: ContextVar[Any] = ContextVar("original_inbox_erasure", default=None)
_ACCESS: ContextVar[Any] = ContextVar("original_inbox_erasure_access", default=None)


class _ErasureScope:
    def __init__(self, ledger, owner, runtime, history, life, before):
        self.ledger, self.owner, self.runtime = ledger, owner, runtime
        self.history, self.life, self.before = history, life, before
        self.inbox = runtime._inbox
        self.work = runtime._budget.updates_completed
        self.thread = get_ident()
        self.context: Token[Any] | None = None
        self.access: Token[Any] | None = None
        self.closed = self.used = self.inflight = self.committed = False

    def require(self):
        if self.closed or self.thread != get_ident() or _CURRENT.get() is not self:
            raise ValueError("inbox erasure requires its original live thread/scope")
        if self.access is None:
            raise ValueError("inbox erasure scope is not entered")
        try:
            _ACCESS.reset(self.access)
        except ValueError as error:
            raise ValueError("inbox erasure requires its original context") from error
        self.access = _ACCESS.set(self)

    def __enter__(self):
        from src.app.managed_replay_origins import ManagedReplayOrigins

        caller = sys._getframe(1)
        if (
            self.thread != get_ident()
            or self.closed
            or self.context is not None
            or _CURRENT.get() is not None
            or type(self.ledger) is not ManagedReplayOrigins
            or caller.f_code is not ManagedReplayOrigins.delete.__code__
            or caller.f_locals.get("self") is not self.ledger
        ):
            raise ValueError("inbox erasure enrollment requires original ledger deletion")
        self.context = _CURRENT.set(self)
        self.access = _ACCESS.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        from src.app.managed_replay_origins import ManagedReplayOrigins

        caller = sys._getframe(1)
        if (
            caller.f_code is not ManagedReplayOrigins.delete.__code__
            or caller.f_locals.get("self") is not self.ledger
        ):
            raise ValueError("inbox erasure close requires original ledger deletion")
        self.require()
        if self.inflight and exc_type is None:
            raise ValueError("inbox erasure cannot close during preparation/publication")
        if self.context is None or self.access is None:
            raise ValueError("inbox erasure close requires its original scope")
        _CURRENT.reset(self.context)
        _ACCESS.reset(self.access)
        self.closed = True
        self.ledger = self.owner = self.runtime = self.inbox = None
        self.history = self.life = self.before = None
        self.context = self.access = None


def observe_inbox_erasure(ledger, owner, runtime, history, life, before):
    return _ErasureScope(ledger, owner, runtime, history, life, before)


@dataclass(frozen=True, slots=True)
class _ErasureTransition:
    scope: Any
    history: Any
    live: tuple
    erased: tuple
    untrained: tuple


def prepare_inbox_erasure(inbox, keys, prepared) -> _ErasureTransition | None:
    """Only the fixed original inbox commit may prepare a transition."""
    scope = _CURRENT.get()
    if scope is None:
        from src.app.expiry_inbox_observation import prepare_expiry_inbox_erasure

        return prepare_expiry_inbox_erasure(inbox, keys, prepared)
    if type(scope) is not _ErasureScope:
        raise ValueError("inbox erasure requires its exact original scope")
    if scope.inbox is not inbox:
        return None
    from src.app.experience_inbox import ExperienceInbox
    from src.app.managed_replay_origins import ManagedReplayOrigins
    from src.app.managed_inbox_origins import ManagedInboxOrigins

    scope.require()
    caller = sys._getframe(1)
    if (
        type(inbox) is not ExperienceInbox
        or scope.used
        or caller.f_code is not ExperienceInbox._commit_erased_history.__code__
        or caller.f_locals.get("self") is not inbox
    ):
        raise ValueError("inbox erasure requires its original single commit")
    scope.used = scope.inflight = True
    try:
        current, _ = ManagedReplayOrigins._require_bindings(scope.ledger, scope.owner)
        if (
            current is not scope.runtime
            or current._inbox is not inbox
            or scope.owner._lifecycle is not scope.life
            or scope.ledger._history() is not scope.history
            or current._budget.updates_completed != scope.work
        ):
            raise ValueError("inbox erasure original owner/runtime/lifecycle/history changed")
        live, erased, untrained = ManagedInboxOrigins.prepare_erasure(
            scope.history, scope.ledger, scope.owner, current, keys, prepared, scope.before
        )
        scope.require()
        return _ErasureTransition(
            scope,
            scope.history,
            live,
            erased,
            untrained,
        )
    except BaseException:
        scope.inflight = False
        raise


def commit_inbox_erasure(transition: _ErasureTransition) -> None:
    """Trusted plain assignments only; validation finished before removal."""
    if type(transition) is not _ErasureTransition:
        from src.app.expiry_inbox_observation import commit_expiry_inbox_erasure

        commit_expiry_inbox_erasure(transition)
        return
    transition.history._records = transition.live[0]
    transition.history._sealed = transition.live[1]
    transition.history._storage = transition.live
    transition.history._erased_records = transition.erased[0]
    transition.history._erased_sealed = transition.erased[1]
    transition.history._erased_storage = transition.erased
    transition.history._untrained_records = transition.untrained[0]
    transition.history._untrained_sealed = transition.untrained[1]
    transition.history._untrained_storage = transition.untrained
    transition.scope.committed = True
    transition.scope.inflight = False
