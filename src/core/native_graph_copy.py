"""Bounded synchronous observations across an actual sequence of graph copies.

Inputs: trusted observer, fixed limits, and actual producer/kind/source at each
copy site. Outputs: borrowed identities and the actual copier memo. Original
producer/source authorization, payload admission, retention and publication are
caller responsibilities. Observations alone grant no provenance or permission.
"""

from copy import deepcopy
from typing import Callable, Literal, TypeVar

from src.core.native_model_copy import (
    ModelCopy,
    ModelCopyLimits,
    ModelCopyScope,
    ModelCopyStage,
    ModelCopyWindow,
    _CopyChannel,
)

GraphCopyKind = Literal[
    "native_snapshot",
    "native_restore",
    "checkpoint_capture",
    "checkpoint_build_state",
    "checkpoint_restore_state",
    "inbox_capture",
    "inbox_materialize",
]
GraphCopyObserver = Callable[
    [object, GraphCopyKind, ModelCopyStage, Callable[[], ModelCopy], Callable[[object], object]],
    None,
]
_KINDS = (
    "native_snapshot",
    "native_restore",
    "checkpoint_capture",
    "checkpoint_build_state",
    "checkpoint_restore_state",
    "inbox_capture",
    "inbox_materialize",
)
_CHANNEL = _CopyChannel("original_graph_copy_sequence")
Value = TypeVar("Value")


class GraphCopyScope:
    def __init__(self, observer: GraphCopyObserver, limits: ModelCopyLimits):
        if not callable(observer):
            raise ValueError("graph sequence requires a trusted observer")
        self._observer: GraphCopyObserver | None = observer
        self._anchor: object | None = observer
        self._event: tuple[object, GraphCopyKind] | None = None
        self._window: ModelCopyWindow | None = None
        self._scope = ModelCopyScope(observer, self._notify, limits, _channel=_CHANNEL)

    def __enter__(self):
        self._window = self._scope.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Delegate context/thread validation before releasing original roots.
        if self._window is None:
            raise ValueError("graph sequence close requires its original live scope")
        self._scope.__exit__(exc_type, exc_value, traceback)
        self._observer = self._anchor = self._event = None

    def _notify(self, stage, read, lookup):
        if self._event is None or self._observer is None:
            raise ValueError("graph sequence has no original synchronous copy event")
        producer, kind = self._event
        self._observer(producer, kind, stage, read, lookup)

    def _copy(self, producer, kind: GraphCopyKind | None, source, copier):
        window = self._window
        if window is None:
            raise ValueError("graph sequence is outside its original scope")
        window._require_live()
        if self._event is not None:
            raise ValueError("graph sequence cannot reenter an active copy")
        if (
            producer is None
            or kind is None
            or type(kind) is not str
            or kind not in _KINDS
            or source is None
        ):
            raise ValueError("graph copy requires actual producer, supported kind and source")
        self._event = producer, kind
        # Why this: one window's counters span all actual roots. Selecting a root
        # neither creates a new allowance nor relaxes its thread/context checks.
        window._source = source
        started = False
        try:
            window.begin(source)
            started = True
            memo: dict[int, object] = {}
            target = copier(memo)
            window.copied(target, memo)
            return target
        finally:
            try:
                if started:
                    window.end()
            finally:
                window._source = self._anchor
                self._event = None


def observe_graph_copies(observer: GraphCopyObserver, limits: ModelCopyLimits) -> GraphCopyScope:
    return GraphCopyScope(observer, limits)


def trace_graph_copy(
    producer: object | None,
    kind: GraphCopyKind | None,
    source: Value,
    copier: Callable[[dict[int, object] | None], Value],
) -> Value:
    window = _CHANNEL.current.get()
    if window is None:
        return copier(None)
    callback = window._observer
    scope = getattr(callback, "__self__", None)
    if (
        type(scope) is not GraphCopyScope
        or getattr(callback, "__func__", None) is not GraphCopyScope._notify
    ):
        raise ValueError("graph sequence original dispatch changed")
    return scope._copy(producer, kind, source, copier)


def copy_graph(producer: object, kind: GraphCopyKind, source: Value) -> Value:
    def copier(memo):
        return deepcopy(source) if memo is None else deepcopy(source, memo)

    return trace_graph_copy(producer, kind, source, copier)
