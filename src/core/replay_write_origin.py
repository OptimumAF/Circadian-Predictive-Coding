"""Bounded synchronous replay-write references; no retained provenance ledger.

An explicit local window binds a model and actual native input identities.
Reports carry row ranges/new snapshot/final retained references without copying
arrays. Caller-retained references require owned retention; no restore authority.
"""

from contextvars import ContextVar, Token
from dataclasses import dataclass
from threading import get_ident
from typing import Callable, Literal, Sequence

ReplayWriteStage = Literal["begin", "before_copy", "copied", "retained"]
ReplayWriteMode = Literal["batch", "rows"]


@dataclass(frozen=True)
class ReplayWriteLimits:
    max_input_rows: int
    max_retained_snapshots: int
    max_notifications: int

    def __post_init__(self) -> None:
        if any(type(n) is not int or n <= 0 for n in vars(self).values()):
            raise ValueError("replay observation limits must be positive integers")


@dataclass(frozen=True)
class ReplayWrite:
    model: object
    features: object
    targets: object
    mode: ReplayWriteMode
    row_start: int
    row_count: int
    snapshot: object | None
    retained: tuple[object, ...] | None


ReplayWriteObserver = Callable[[ReplayWriteStage, Callable[[], ReplayWrite]], None]


class ReplayWriteWindow:
    """Invocation-local authority; never attach this to native model state."""

    def __init__(self, model, features, targets, observer, limits: ReplayWriteLimits) -> None:
        if type(limits) is not ReplayWriteLimits or not callable(observer):
            raise ValueError("replay window requires exact limits and a callable observer")
        self._source: tuple[object, object, object] | None = (model, features, targets)
        self._observer: ReplayWriteObserver | None = observer
        self._limits = limits
        self._thread = get_ident()
        self._record: ReplayWrite | None = None
        self._rows = 0
        self._mode: ReplayWriteMode = "batch"
        self._range = (0, 0)
        self._notifications = 0
        self._begun = self._finished = False

    def _require_live(self) -> None:
        if self._source is None or get_ident() != self._thread or _CURRENT.get() is not self:
            raise ValueError("replay window is expired or outside its original thread/context")
        if self._record is not None:
            raise ValueError("replay write observer cannot reenter its window")

    def read(self) -> ReplayWrite:
        if self._record is None or get_ident() != self._thread or _CURRENT.get() is not self:
            raise ValueError("replay write access requires its synchronous original callback")
        return self._record

    def _notify(self, stage, start, count, snapshot=None, retained=None) -> None:
        self._require_live()
        if self._notifications >= self._limits.max_notifications:
            raise ValueError("replay observation notification limit exhausted")
        self._notifications += 1
        assert self._source is not None and self._observer is not None
        self._record = ReplayWrite(*self._source, self._mode, start, count, snapshot, retained)
        try:
            self._observer(stage, self.read)
        finally:
            self._record = None

    def _retained(self, retained: Sequence[object]) -> tuple[object, ...]:
        if len(retained) > self._limits.max_retained_snapshots:
            raise ValueError("replay retained reference limit exceeded")
        return tuple(retained)

    def begin(self, model, features, targets, retained, rows: int, mode: ReplayWriteMode) -> None:
        self._require_live()
        if (
            self._begun
            or self._source is None
            or any(
                actual is not expected
                for actual, expected in zip((model, features, targets), self._source)
            )
        ):
            raise ValueError("replay write differs from original model/inputs or already began")
        if type(rows) is not int or not 0 < rows <= self._limits.max_input_rows:
            raise ValueError("replay input row limit exceeded or invalid")
        if mode not in ("batch", "rows") or type(mode) is not str:
            raise ValueError("replay write mode is invalid")
        self._mode, self._rows, self._begun = mode, rows, True
        self._notify("begin", 0, rows, retained=self._retained(retained))

    def before_copy(self, start: int, count: int) -> None:
        self._require_live()
        if (
            not self._begun
            or self._finished
            or type(start) is not int
            or type(count) is not int
            or start < 0
            or count <= 0
            or start + count > self._rows
            or (self._mode == "rows" and count != 1)
            or (self._mode == "batch" and (start, count) != (0, self._rows))
        ):
            raise ValueError("replay copy range is invalid or outside its original input")
        self._range = start, count
        self._notify("before_copy", start, count)

    def copied(self, snapshot: object) -> None:
        self._require_live()
        if self._range[1] == 0 or self._finished or snapshot is None:
            raise ValueError("replay copied snapshot requires an original pending range")
        self._notify("copied", *self._range, snapshot=snapshot)
        self._range = (0, 0)

    def finish(self, retained: Sequence[object]) -> None:
        self._require_live()
        if not self._begun or self._finished:
            raise ValueError("replay window must finish its original write once")
        self._finished = True
        self._notify("retained", 0, self._rows, retained=self._retained(retained))

    def require_close(self) -> None:
        if get_ident() != self._thread or self._record is not None:
            raise ValueError("replay window close requires original quiescent thread")

    def close(self) -> None:
        self.require_close()
        self._source = self._observer = self._record = None


_CURRENT: ContextVar[ReplayWriteWindow | None] = ContextVar("original_replay_write", default=None)


class ReplayWriteScope:
    """Explicit token owner; refused close cannot destroy a suspended generator."""

    def __init__(self, model, features, targets, observer, limits) -> None:
        self._window = ReplayWriteWindow(model, features, targets, observer, limits)
        self._token: Token[ReplayWriteWindow | None] | None = None
        self._used = False

    def __enter__(self) -> ReplayWriteWindow:
        if self._used or _CURRENT.get() is not None:
            raise ValueError("replay write windows cannot nest or reopen")
        self._window.require_close()
        self._used = True
        self._token = _CURRENT.set(self._window)
        return self._window

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._token is None:
            return
        self._window.require_close()
        # Reset first: a foreign context must not clear the original window.
        _CURRENT.reset(self._token)
        self._token = None
        self._window.close()


def observe_replay_writes(
    model: object,
    features: object,
    targets: object,
    observer: ReplayWriteObserver,
    limits: ReplayWriteLimits,
) -> ReplayWriteScope:
    return ReplayWriteScope(model, features, targets, observer, limits)


def begin_replay_write(model, features, targets, retained, rows: int, mode: ReplayWriteMode):
    window = _CURRENT.get()
    if window is not None:
        window.begin(model, features, targets, retained, rows, mode)
    return window
