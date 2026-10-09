"""Synchronous original model-copy observations with bounded memo lookups.

Inputs are an explicit original source, observer and local limits. Outputs are
borrowed references at the actual copier boundary. No copying, consent, holder
enrollment, payload accounting, persistence or restore authority lives here.
Trusted observers must separately account any references they retain.
"""

from contextvars import ContextVar, Token
from dataclasses import dataclass
from threading import get_ident
from typing import Callable, Literal

ModelCopyStage = Literal["before_copy", "copied"]


@dataclass(frozen=True)
class ModelCopyLimits:
    max_copies: int
    max_notifications: int
    max_reads: int

    def __post_init__(self) -> None:
        if any(type(n) is not int or not 0 < n < 2**63 for n in vars(self).values()):
            raise ValueError("model copy limits require positive bounded integers")


@dataclass(frozen=True)
class ModelCopy:
    source: object
    target: object | None


ModelCopyObserver = Callable[
    [ModelCopyStage, Callable[[], ModelCopy], Callable[[object], object]], None
]


class ModelCopyWindow:
    def __init__(self, source, observer, limits: ModelCopyLimits, *, _channel=None) -> None:
        if source is None or not callable(observer) or type(limits) is not ModelCopyLimits:
            raise ValueError("model copy window requires source, observer and exact limits")
        self._source: object | None = source
        self._channel: _CopyChannel = _MODEL_CHANNEL if _channel is None else _channel
        self._observer: ModelCopyObserver | None = observer
        self._limits = limits
        self._thread = get_ident()
        self._context_token: Token[object | None] | None = None
        self._copies = self._notifications = self._reads = 0
        self._callback = self._inflight = self._completed = self._failed = False

    def _require_live(self) -> None:
        if (
            self._source is None
            or get_ident() != self._thread
            or self._channel.current.get() is not self
        ):
            raise ValueError("model copy window expired or outside original thread/context")
        if self._callback:
            raise ValueError("model copy observer cannot reenter its window")
        self._require_context()

    def _require_context(self) -> None:
        # Token reset checks context identity; inherited values alone cannot.
        if self._context_token is None:
            raise ValueError("model copy requires its original context")
        try:
            self._channel.access.reset(self._context_token)
        except ValueError as error:
            raise ValueError("model copy requires its original context") from error
        self._context_token = self._channel.access.set(self)

    def _notify(self, stage: ModelCopyStage, target=None, memo=None) -> None:
        self._require_live()
        if self._notifications >= self._limits.max_notifications:
            raise ValueError("model copy notification limit exhausted")
        self._notifications += 1
        source = self._source
        active = self._callback = True

        def require_read():
            if not active or get_ident() != self._thread or self._channel.current.get() is not self:
                raise ValueError("model copy reader requires its synchronous original callback")
            self._require_context()
            if self._reads >= self._limits.max_reads:
                raise ValueError("model copy read limit exhausted")
            self._reads += 1

        def read() -> ModelCopy:
            require_read()
            assert source is not None
            return ModelCopy(source, target)

        def lookup(original: object) -> object:
            require_read()
            if memo is None or id(original) not in memo:
                raise ValueError("original identity has no observed deepcopy memo entry")
            return memo[id(original)]

        try:
            assert self._observer is not None
            self._observer(stage, read, lookup)
        except BaseException:
            self._failed = True
            raise
        finally:
            active = self._callback = False
            # Saved readers must neither revive nor keep the copied graph alive.
            source = target = memo = None

    def begin(self, source) -> None:
        self._require_live()
        if self._failed or self._inflight or source is not self._source:
            raise ValueError("model copy source differs, operation failed or copy already active")
        if self._copies >= self._limits.max_copies:
            raise ValueError("model copy attempt limit exhausted")
        if self._notifications + 2 > self._limits.max_notifications:
            raise ValueError("model copy notification limit exhausted before allocation")
        self._copies += 1
        self._inflight, self._completed = True, False
        try:
            self._notify("before_copy")
        except BaseException:
            self._inflight = False
            raise

    def copied(self, target, memo) -> None:
        self._require_live()
        if (
            not self._inflight
            or self._completed
            or target is self._source
            or target is None
            or type(memo) is not dict
            or memo.get(id(self._source)) is not target
        ):
            raise ValueError("model copy requires actual distinct target and original memo")
        self._notify("copied", target, memo)
        self._completed = True

    def end(self) -> None:
        self._require_live()
        if not self._inflight:
            raise ValueError("model copy has no active operation")
        if not self._completed:
            self._failed = True
        self._inflight = False

    def require_close(self) -> None:
        if get_ident() != self._thread or self._callback or self._inflight:
            raise ValueError("model copy close requires original quiescent thread")
        if self._context_token is not None:
            self._require_context()

    def close(self) -> None:
        self.require_close()
        if self._context_token is not None:
            self._channel.access.reset(self._context_token)
            self._context_token = None
        self._source = self._observer = None


class _CopyChannel:
    """Keep independent copy boundaries on independent context stacks."""

    def __init__(self, name: str) -> None:
        self.current: ContextVar[ModelCopyWindow | None] = ContextVar(name, default=None)
        self.access: ContextVar[object | None] = ContextVar(name + "_context", default=None)


_MODEL_CHANNEL = _CopyChannel("original_model_copy")


class ModelCopyScope:
    def __init__(self, source, observer, limits, *, _channel=None) -> None:
        self._channel: _CopyChannel = _MODEL_CHANNEL if _channel is None else _channel
        self._window = ModelCopyWindow(source, observer, limits, _channel=self._channel)
        self._token: Token[ModelCopyWindow | None] | None = None
        self._used = False

    def __enter__(self) -> ModelCopyWindow:
        if self._used or self._channel.current.get() is not None:
            raise ValueError("model copy windows cannot nest or reopen")
        self._window.require_close()
        self._used = True
        self._token = self._channel.current.set(self._window)
        self._window._context_token = self._channel.access.set(self._window)
        return self._window

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._token is None:
            return
        self._window.require_close()
        self._channel.current.reset(self._token)
        self._token = None
        self._window.close()


def observe_model_copies(source, observer: ModelCopyObserver, limits: ModelCopyLimits):
    return ModelCopyScope(source, observer, limits)


def begin_model_copy(source) -> ModelCopyWindow | None:
    window = _MODEL_CHANNEL.current.get()
    if window is not None:
        window.begin(source)
    return window
