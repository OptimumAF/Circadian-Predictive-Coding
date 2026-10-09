"""Observe actual native snapshot/restore dictionary copies, without authority.

Inputs: original state dictionary, synchronous observer and fixed copy limits.
Outputs: borrowed source/target references and actual copier memo lookups. The
caller owns admission/accounting and any retained references. This module does
not validate a snapshot, publish restored state or grant replay consent.
"""

from copy import deepcopy
from typing import Any

from src.core.native_model_copy import (
    ModelCopyLimits,
    ModelCopyObserver,
    ModelCopyScope,
    _CopyChannel,
)
from src.core.native_graph_copy import GraphCopyKind, trace_graph_copy

# Why this: a checkpoint preparation can fork and snapshot in the same call.
# Their roots differ; each channel keeps its original independent limits/context.
_STATE_CHANNEL = _CopyChannel("original_native_state_copy")


def observe_state_copies(
    source: dict[str, Any], observer: ModelCopyObserver, limits: ModelCopyLimits
) -> ModelCopyScope:
    if type(source) is not dict:
        raise ValueError("native state observation requires an original dictionary")
    return ModelCopyScope(source, observer, limits, _channel=_STATE_CHANNEL)


def copy_native_state(
    source: dict[str, Any], *, producer: object | None = None, kind: GraphCopyKind | None = None
) -> dict[str, Any]:
    return trace_graph_copy(producer, kind, source, lambda memo: _copy_state(source, memo))


def _copy_state(source, memo):
    window = _STATE_CHANNEL.current.get()
    if window is None:
        return deepcopy(source) if memo is None else deepcopy(source, memo)
    window.begin(source)
    try:
        if memo is None:
            memo = {}
        copied = deepcopy(source, memo)
        window.copied(copied, memo)
        return copied
    finally:
        window.end()
