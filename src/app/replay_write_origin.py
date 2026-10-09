"""Bind invocation-local replay writes to the original managed update producer.

Consumes b3a callbacks and a native model reference port. No raw row retention,
consent certification, persistence, checkpoint materialization or model operation.
"""

from dataclasses import dataclass
from threading import get_ident
from typing import Any, Callable, ContextManager

from src.core.native_update_origin import (
    NativeUpdateOrigin,
    NativeUpdateObserver,
    NativeUpdateStage,
)
from src.core.replay_write_origin import (
    ReplayWrite,
    ReplayWriteLimits,
    ReplayWriteObserver,
    ReplayWriteStage,
    observe_replay_writes,
)


@dataclass(frozen=True)
class ManagedReplayWrite:
    origin: NativeUpdateOrigin[Any, Any]
    write: ReplayWrite


ManagedReplayWriteObserver = Callable[[ReplayWriteStage, Callable[[], ManagedReplayWrite]], None]


class ManagedReplayWriteAccess:
    def __init__(
        self,
        model_reference: Callable[[object], object],
        observer: ManagedReplayWriteObserver,
        limits: ReplayWriteLimits,
        *,
        update_observer: NativeUpdateObserver[Any, Any] | None = None,
    ) -> None:
        if (
            not callable(model_reference)
            or not callable(observer)
            or type(limits) is not ReplayWriteLimits
        ):
            raise ValueError("managed replay access requires model/observer ports and exact limits")
        if update_observer is not None and not callable(update_observer):
            raise ValueError("managed replay update observer must be callable or None")
        self._model_reference, self._observer, self._limits = model_reference, observer, limits
        self._update_observer = update_observer
        self._scope: ContextManager[Any] | None = None
        self._thread: int | None = None

    def __call__(
        self, stage: NativeUpdateStage, read_origin: Callable[[], NativeUpdateOrigin]
    ) -> None:
        if stage == "started":
            if self._scope is not None:
                raise ValueError("managed replay access already owns an invocation")
            origin = read_origin()
            model = self._model_reference(origin.learner)

            def writes(
                write_stage: ReplayWriteStage, read_write: Callable[[], ReplayWrite]
            ) -> None:
                # The core read still enforces callback lifetime and original thread.
                self._observer(write_stage, lambda: ManagedReplayWrite(origin, read_write()))

            observer: ReplayWriteObserver = writes
            scope = observe_replay_writes(
                model, origin.features, origin.targets, observer, self._limits
            )
            scope.__enter__()
            self._scope = scope
            self._thread = get_ident()
        else:
            self.close()
        try:
            if self._update_observer is not None:
                self._update_observer(stage, read_origin)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._scope is not None and self._thread != get_ident():
            raise ValueError("managed replay access close requires its original thread")
        scope = self._scope
        if scope is not None:
            scope.__exit__(None, None, None)
            self._scope = None
            self._thread = None
