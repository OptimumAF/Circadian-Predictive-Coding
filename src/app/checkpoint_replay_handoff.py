"""Trusted replay coordinator lease at the original checkpoint publication point.

The coordinator validates and prepares weak records before publication. The
controller commits only plain ledger assignments after its original publication.
No arbitrary post-publication callbacks, copier, probes or authority replacement.
"""

from contextvars import ContextVar, Token
from dataclasses import dataclass
from threading import get_ident
from typing import Any


@dataclass(frozen=True)
class _ReplayTransition:
    ledger: Any
    anchors: dict[str, Any]
    rows: dict[int, Any]
    inbox_state: tuple[dict[int, Any], dict[int, Any]] | None = None
    erased_inbox_state: tuple[dict[tuple[str, str], Any], dict[tuple[str, str], Any]] | None = None
    untrained_inbox_state: tuple[dict[tuple[str, str], Any], dict[tuple[str, str], Any]] | None = (
        None
    )


_CURRENT: ContextVar[Any] = ContextVar("original_checkpoint_replay_handoff", default=None)
_ACCESS: ContextVar[Any] = ContextVar("original_checkpoint_replay_handoff_access", default=None)


class _HandoffScope:
    def __init__(self, controller, coordinator):
        self.controller, self.coordinator = controller, coordinator
        self.thread = get_ident()
        self.used = self.closed = self.inflight = False
        self.context: Token[Any] | None = None
        self.access: Token[Any] | None = None

    def require(self):
        if self.closed or get_ident() != self.thread or _CURRENT.get() is not self:
            raise ValueError("checkpoint handoff requires its original live thread/scope")
        if self.access is None:
            raise ValueError("checkpoint handoff scope is not entered")
        try:
            _ACCESS.reset(self.access)
        except ValueError as error:
            raise ValueError("checkpoint handoff requires its original context") from error
        self.access = _ACCESS.set(self)

    def __enter__(self):
        if get_ident() != self.thread:
            raise ValueError("checkpoint handoff enter requires its original thread")
        from src.app.managed_replay_checkpoints import ManagedReplayCheckpoints

        if self.closed or self.context is not None or _CURRENT.get() is not None:
            raise ValueError("checkpoint handoff scope cannot nest or reopen")
        if type(self.coordinator) is not ManagedReplayCheckpoints:
            raise ValueError("checkpoint handoff requires the exact trusted replay coordinator")
        self.context = _CURRENT.set(self)
        self.access = _ACCESS.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.require()
        if self.inflight:
            raise ValueError("checkpoint handoff cannot close during active publication")
        if self.context is None or self.access is None:
            raise ValueError("checkpoint handoff close requires its original scope")
        _CURRENT.reset(self.context)
        _ACCESS.reset(self.access)
        self.closed = True
        self.controller = self.coordinator = None
        self.context = self.access = None


def lease_replay_handoff(controller, coordinator):
    return _HandoffScope(controller, coordinator)


class _PublicationLease:
    def __init__(self, controller, original, prepared, pending):
        self._arguments: Any = controller, original, prepared, pending
        self._scope: Any = None
        self._lease: Any = None
        self._thread = get_ident()
        self._entered = self._closed = False

    def __enter__(self):
        if get_ident() != self._thread or self._entered or self._closed:
            raise ValueError("checkpoint publication lease cannot cross threads or reopen")
        self._entered = True
        scope = _CURRENT.get()
        if scope is None:
            return None
        from src.app.managed_replay_checkpoints import ManagedReplayCheckpoints
        from src.app.managed_replay_origins import ManagedReplayOrigins

        scope.require()
        if self._arguments[0] is not scope.controller or scope.used:
            raise ValueError("checkpoint publication differs from original single handoff attempt")
        scope.used = scope.inflight = True
        self._scope = scope
        entered = False
        try:
            # Invoke the exact implementation, never an instance-supplied exit callback.
            self._lease = ManagedReplayCheckpoints._publication_lease(
                scope.coordinator, *self._arguments
            )
            transition = self._lease.__enter__()
            entered = True
            scope.require()
            if (
                type(transition) is not _ReplayTransition
                or type(transition.ledger) is not ManagedReplayOrigins
                or type(transition.anchors) is not dict
                or type(transition.rows) is not dict
                or (
                    transition.inbox_state is not None
                    and (
                        type(transition.inbox_state) is not tuple
                        or len(transition.inbox_state) != 2
                        or any(type(value) is not dict for value in transition.inbox_state)
                        or transition.ledger._history() is None
                    )
                )
                or (
                    transition.erased_inbox_state is not None
                    and (
                        type(transition.erased_inbox_state) is not tuple
                        or len(transition.erased_inbox_state) != 2
                        or any(type(value) is not dict for value in transition.erased_inbox_state)
                        or transition.inbox_state is None
                        or transition.ledger._history() is None
                    )
                )
                or (
                    transition.untrained_inbox_state is not None
                    and (
                        type(transition.untrained_inbox_state) is not tuple
                        or len(transition.untrained_inbox_state) != 2
                        or any(
                            type(value) is not dict for value in transition.untrained_inbox_state
                        )
                        or transition.inbox_state is None
                        or transition.ledger._history() is None
                    )
                )
            ):
                raise ValueError("checkpoint handoff requires a prepared exact replay transition")
            return transition
        except BaseException as error:
            try:
                if entered:
                    self._lease.__exit__(type(error), error, error.__traceback__)
            finally:
                self._clear()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        if get_ident() != self._thread or not self._entered or self._closed:
            raise ValueError("checkpoint publication close requires its original live thread/lease")
        if self._scope is not None:
            # Why: a generator context manager cannot refuse foreign closure
            # before its finally body runs. Validate before resuming that body.
            self._scope.require()
        try:
            if self._lease is not None:
                return self._lease.__exit__(exc_type, exc_value, traceback)
            return False
        finally:
            self._clear()

    def _clear(self):
        if self._scope is not None:
            self._scope.inflight = False
        self._closed = True
        self._scope = self._lease = self._arguments = None


def checkpoint_publication_lease(controller, original, prepared, pending):
    return _PublicationLease(controller, original, prepared, pending)


def commit_replay_transition(transition: _ReplayTransition) -> None:
    """Trusted pure assignments only; all validation precedes retirement."""
    transition.ledger._anchors = transition.anchors
    transition.ledger._rows = transition.rows
    if transition.inbox_state is not None:
        transition.ledger._inbox_origins._records = transition.inbox_state[0]
        transition.ledger._inbox_origins._sealed = transition.inbox_state[1]
        transition.ledger._inbox_origins._storage = transition.inbox_state
    if transition.erased_inbox_state is not None:
        transition.ledger._inbox_origins._erased_records = transition.erased_inbox_state[0]
        transition.ledger._inbox_origins._erased_sealed = transition.erased_inbox_state[1]
        transition.ledger._inbox_origins._erased_storage = transition.erased_inbox_state
    if transition.untrained_inbox_state is not None:
        transition.ledger._inbox_origins._untrained_records = transition.untrained_inbox_state[0]
        transition.ledger._inbox_origins._untrained_sealed = transition.untrained_inbox_state[1]
        transition.ledger._inbox_origins._untrained_storage = transition.untrained_inbox_state
