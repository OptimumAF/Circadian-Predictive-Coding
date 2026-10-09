"""Expiring synchronous access to original update references, without copies.

The inbox supplies original receipt/count reads. This module neither stores
replay rows nor validates consent or native state; their original owners do so.
"""

from threading import get_ident
from typing import Callable, Generic, TypeVar

from src.core.native_update_origin import (
    NativeUpdateObserver,
    NativeUpdateOrigin,
    NativeUpdateStage,
)

Features = TypeVar("Features")
Targets = TypeVar("Targets")


class NativeUpdateAccess(Generic[Features, Targets]):
    def __init__(
        self,
        observer: NativeUpdateObserver[Features, Targets],
        read_origin: Callable[[], NativeUpdateOrigin[Features, Targets]],
    ) -> None:
        self._observer: NativeUpdateObserver[Features, Targets] | None = observer
        self._read_origin: Callable[[], NativeUpdateOrigin[Features, Targets]] | None = read_origin
        self._thread = get_ident()
        self._notifying = False

    def read(self) -> NativeUpdateOrigin[Features, Targets]:
        if not self._notifying or get_ident() != self._thread or self._read_origin is None:
            raise ValueError("native update access requires its original synchronous observer")
        return self._read_origin()

    def notify(self, stage: NativeUpdateStage) -> None:
        if self._observer is None or self._notifying or get_ident() != self._thread:
            raise ValueError("native update observer is closed, reentrant or foreign-thread")
        self._notifying = True
        try:
            self._observer(stage, self.read)
        finally:
            self._notifying = False

    def failed(self, stage: NativeUpdateStage, primary: BaseException) -> None:
        try:
            self.notify(stage)
        except BaseException as secondary:
            # Why: a diagnostic observer must not hide the native/resource failure.
            raise BaseExceptionGroup("native update and observer failed", [primary, secondary])

    def close(self) -> None:
        self._notifying = False
        self._observer = None
        self._read_origin = None
