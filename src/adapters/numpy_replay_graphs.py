"""Bounded borrowed replay inventory for exact native state and snapshot roots.

Rows retain their original identity. Payload bytes count replay arrays only;
these ports do not copy arrays, inspect model weights, or grant authority.
Callers supply already identity-bound roots and retain provenance and consent.
"""

from collections import deque
from typing import cast

from src.adapters.numpy_learners import Array
from src.adapters.numpy_replay_origins import replay_payload_references
from src.core.circadian_predictive_coding import CircadianNetworkSnapshot


def _require_bound(value: int, name: str) -> None:
    if type(value) is not int or not 0 < value < 2**63:
        raise ValueError(f"{name} must be an exact positive int below 2**63")


def _replay_memory(root: object, max_rows: int) -> deque[object]:
    _require_bound(max_rows, "max_rows")
    if type(root) is dict:
        state = cast(dict[str, object], root)
    elif type(root) is CircadianNetworkSnapshot:
        try:
            state = root.state
        except AttributeError as error:
            raise ValueError("native network snapshot must contain state") from error
        if type(state) is not dict:
            raise ValueError("native network snapshot state must be an exact dict")
    else:
        raise ValueError("replay graph root must be an exact native state dict or snapshot")
    memory = state.get("_replay_memory")
    if type(memory) is not deque:
        raise ValueError("replay graph state must contain an exact native replay deque")
    # Why this: reject an oversized inventory before reading any payload row.
    if len(memory) > max_rows:
        raise ValueError("native replay exceeds replay graph row bound")
    return cast(deque[object], memory)


def _payload_references(row: object) -> tuple[Array, Array]:
    try:
        return replay_payload_references(row)
    except AttributeError as error:
        raise ValueError("native replay snapshot must contain replay payload arrays") from error


def replay_graph_rows(root: object, max_rows: int) -> tuple[object, ...]:
    """Return original validated rows from a bounded exact native replay deque."""
    memory = _replay_memory(root, max_rows)
    rows: list[object] = []
    for row in memory:
        _payload_references(row)
        rows.append(row)
    return tuple(rows)


def replay_graph_payload_bytes(root: object, max_rows: int, max_bytes: int) -> int:
    """Count replay array bytes, refusing excess before accumulating each array."""
    _require_bound(max_bytes, "max_bytes")
    memory = _replay_memory(root, max_rows)
    total = 0
    for row in memory:
        # Why this: charge aliases per row, conservatively estimating a copy.
        for array in _payload_references(row):
            size = array.nbytes
            if size > max_bytes - total:
                raise ValueError("native replay exceeds replay graph payload byte bound")
            total += size
    return total
