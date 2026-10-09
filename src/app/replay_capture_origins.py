"""Mandatory bounded replay admission inside an original composite lease.

Transient native inventories contain references only. No copy, model operation,
lock reentry, durable origin, retained-holder lineage issuance or restore authority.
"""

from contextlib import contextmanager
from threading import get_ident
from typing import Callable, Iterator
from src.app.managed_replay_origins import ManagedReplayOrigins


@contextmanager
def lease_replay_capture(owner, groups, limits, inventory, ledger) -> Iterator[Callable[[], None]]:
    if not callable(inventory):
        raise ValueError("complete capture requires a bounded replay inventory port")
    if ledger is not None and type(ledger) is not ManagedReplayOrigins:
        raise ValueError("complete capture requires the exact original replay ledger")

    active, thread, checks = True, get_ident(), 0

    def require(model):
        nonlocal checks
        if not active or get_ident() != thread:
            raise ValueError("replay inventory check expired or crossed threads")
        checks += 1
        if checks > 8:
            raise ValueError("replay inventory check allowance exhausted")
        entries = inventory(groups, limits)
        if type(entries) is not tuple or len(entries) > limits.max_nodes:
            raise ValueError("replay inventory exceeds original capture bounds")
        seen = set()
        for entry in entries:
            if type(entry) is not tuple or len(entry) != 2 or type(entry[1]) is not tuple:
                raise ValueError("replay inventory requires exact model/row reference tuples")
            original, rows = entry
            if id(original) in seen or len(rows) > limits.max_nodes:
                raise ValueError("replay inventory identity/count is invalid")
            seen.add(id(original))
            if rows and original is not model:
                raise ValueError(
                    "nonempty replay requires original ledger or qualified holder lineage"
                )
        return entries

    try:
        if ledger is None:
            require(None)
            # Even empty buffers can change in a trusted projection/copy callback.
            yield lambda: require(None)
        else:
            with ledger._lease_capture(owner) as verify:

                def check():
                    require(verify())

                check()
                yield check
    finally:
        active = False
        groups = None
