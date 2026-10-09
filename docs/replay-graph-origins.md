# Borrowed replay inventories for native state roots

`src/core/replay_graph_origin.py` defines the inner `ReplayGraphPorts` contract.
`src/adapters/numpy_replay_graphs.py` implements it for an exact native state
dictionary or exact `CircadianNetworkSnapshot`. Both require an exact deque of
`ReplaySnapshot` records and supported real numeric, aligned 2D arrays.

The row result borrows the original records. It never copies payload arrays.
Positive row and byte limits are exact integers below `2**63`; booleans and
integer subclasses are refused. An oversized deque is refused before inspecting
its records. Empty replay returns an empty tuple and zero replay bytes.

Byte counts sum input and target array `nbytes` per retained row. Shared payloads
are charged again for each row, conservatively. These are replay payload bytes,
not complete native model bytes, heap usage, or process RSS. Callers separately
account weights, inbox payloads and other owned content.

Why this: checkpoint copies expose dictionaries and snapshot roots rather than
the original model object accepted by the existing inventory adapter. A borrowed
inventory lets the application check the actual copier memo before retaining
weak witnesses, without creating an additional model or copy.

```python
from collections import deque

from src.adapters.numpy_replay_graphs import replay_graph_payload_bytes, replay_graph_rows
from src.core.replay_graph_origin import ReplayGraphPorts

ports = ReplayGraphPorts(replay_graph_rows, replay_graph_payload_bytes)

def inspect_identity_bound_state(state: object):
    rows = ports.rows(state, 16)
    size = ports.payload_bytes(state, 16, 4096)
    return rows, size

# Empty schema fixture only; this carries no producer or restore authority.
assert inspect_identity_bound_state({"_replay_memory": deque()}) == ((), 0)
```

In production, the application must establish the exact original producer/root
identity before calling these ports. Matching shape, values, keys or digests
cannot establish that identity. Admission, weak ownership, original receipts,
consent, revocation, tombstones, monotone quotas and checkpoint publication remain
the application coordinator's responsibilities. This module grants no capture,
restore, handoff or scientific permission.

Next extension: bind actual copied state rows and copied inbox source/label/receipt
chains to the original ledger. After the final opaque probes, validate under a
trusted publication lease at the existing actor-guarded publication body. Prepare
weak anchors before retirement and commit only trusted assignments afterward.
Never renew the original ledger or copy budget to make the handoff succeed.
