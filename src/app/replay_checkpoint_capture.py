"""Read original issued checkpoint witnesses during an existing capture lease.

Inputs are the already leased original owner/ledger and its checkpoint manager.
The reader qualifies actual issued snapshot references without native methods,
copying, public manager operations, admission, restoration or publication. Weak
records remain the source of provenance; current keys alone never establish it.
"""

from contextlib import ExitStack, contextmanager
from dataclasses import fields
from threading import get_ident
from typing import Any, Callable, Iterator

from src.app.checkpoint_erased_inbox import require_same_erased_inbox_history
from src.app.checkpoint_untrained_inbox import require_same_untrained_inbox_history
from src.app.managed_replay_checkpoints import (
    ManagedReplayCheckpoints,
    _Checkpoint,
    _Operation,
    _checkpoint_references,
    _view_history,
)
from src.core.managed_composite_state import CompositeCaptureLimits
from src.app.replay_restored_capture import lease_restored_replay_capture


def _witnesses(manager: ManagedReplayCheckpoints, maximum: int) -> dict[int, Any]:
    records = manager._checkpoints
    if type(records) is not dict or len(records) > maximum:
        raise ValueError("issued replay checkpoint inventory exceeds exact capture bounds")
    result = {}
    for key, witness in records.items():
        if (
            type(key) is not int
            or type(witness) is not _Checkpoint
            or vars(witness).keys() != {field.name for field in fields(_Checkpoint)}
        ):
            raise ValueError("issued replay checkpoint witness schema changed")
        _checkpoint_references(witness, maximum)
        for name in ("rows", "inbox", "erased_inbox", "untrained_inbox"):
            value = getattr(witness, name)
            if type(value) is not tuple or len(value) > maximum:
                raise ValueError("issued replay checkpoint rows/history exceed capture bounds")
        for name in ("enrollment", "attempts", "position", "attempt_limit", "pending_limit"):
            value = getattr(witness, name)
            if type(value) is not int or value < (0 if name in ("attempts", "position") else 1):
                raise ValueError("issued replay checkpoint history requires exact integer counters")
        token, state = witness.token(), witness.state()
        # Expired records cannot qualify, but remain bounded original manager data.
        if token is None or state is None:
            continue
        if key != id(token) or id(state) in result:
            raise ValueError("issued replay checkpoint identity inventory changed or is ambiguous")
        result[id(state)] = witness
    return result


def _operation(manager, owner, ledger, copies, runtime, witness, maximum):
    controller = witness.controller()
    life, enrollment = copies._controller(owner, controller)
    if (
        type(controller._models) is not list
        or len(controller._models) > maximum
        or type(controller._pending) is not dict
        or len(controller._pending) > maximum
    ):
        raise ValueError("issued replay controller history exceeds capture bounds")
    current, model, rows = copies._source(ledger, owner, leased=True)
    if current is not runtime or len(rows) > maximum:
        raise ValueError("issued replay checkpoint original runtime/row inventory changed")
    op = _Operation(ledger, owner, controller, life, enrollment, runtime, model, rows)
    try:
        op.token, op.checkpoint = witness.token(), witness
        op.original_content = manager._stamp(model.__dict__)
        op.copy_gate, op.manager_gate = copies._gate, manager._gate
        # Reuse one borrowed cursor: _original_inbox records its identity in
        # bounded operation maps, so fresh wrappers would accumulate each round.
        op.original_cursor = runtime._inbox._borrow_cursor()
        op.original_inbox_content = manager._stamp(op.original_cursor)
        return op
    except BaseException:
        op.close()
        raise


def _roles(witness, model, rows, roles):
    appearances = roles.get(id(model), ())
    if not appearances:
        raise ValueError("issued replay snapshot has no original checkpoint snapshots role")
    for current, retained, holder, collection, item in appearances:
        if (
            current is not model
            or collection != "snapshots"
            or holder.kind != "checkpoint"
            or holder.enrollment != witness.enrollment
            or item is not witness.view()
            or len(retained) != len(rows)
            or any(a is not b for a, b in zip(retained, rows))
        ):
            raise ValueError("issued replay snapshot appeared through an unqualified holder role")


def _inbox(manager, op, witness):
    # Read original admitted origins, not provenance reconstructed from cursor keys.
    original = manager._original_inbox(op, op.original_cursor)
    if len(original) != len(witness.inbox) or {id(pair.origin) for pair in original} != {
        id(pair.origin) for pair in witness.inbox
    }:
        raise ValueError("issued replay inbox differs from original admitted pair witnesses")
    cursor = witness.cursor()
    manager._verify_inbox(
        cursor,
        witness.inbox,
        witness.erased_inbox,
        untrained=witness.untrained_inbox,
        content_limits=op.history_content_limits,
        maximum=op.pair_capacity,
    )
    if len(witness.erased_inbox) != len(op.erased_history) or len(witness.untrained_inbox) != len(
        op.untrained_history
    ):
        raise ValueError("issued replay inbox partition differs from original admitted history")
    for copied, original in zip(witness.erased_inbox, op.erased_history):
        require_same_erased_inbox_history(original, copied, op.history_content_limits)
    for copied, original in zip(witness.untrained_inbox, op.untrained_history):
        require_same_untrained_inbox_history(original, copied, op.history_content_limits)


def _verify(manager, op, witness, model, rows, roles):
    manager._current(op)
    pending = op.controller._require(op.token)
    manager._require_checkpoint(op, pending)
    if pending.owner is not op.runtime or pending.revision != op.runtime._revision:
        raise ValueError(
            "issued replay checkpoint no longer belongs to its unchanged original runtime"
        )
    if _view_history(pending.view, max_metadata_bytes=op.metadata_limit) != witness.view_history:
        raise ValueError("issued replay checkpoint complete view history changed")
    manager._pure_budget(op, pending)
    manager._prove_current(op)
    retained = manager._verify_rows(op, model, witness.rows)
    if len(rows) != len(retained) or any(a is not b for a, b in zip(rows, retained)):
        raise ValueError(
            "issued replay snapshot row inventory differs from original graph witnesses"
        )
    _roles(witness, model, rows, roles)
    _inbox(manager, op, witness)
    manager._current(op)
    manager._prove_current(op)


@contextmanager
def lease_replay_checkpoint_capture(
    owner: Any,
    ledger: Any,
    copies: Any,
    manager: ManagedReplayCheckpoints,
    limits: CompositeCaptureLimits,
) -> Iterator[Callable[..., set[int]]]:
    """Yield a bounded known-state dispatcher; the enclosing reader owns rounds."""
    if (
        type(manager) is not ManagedReplayCheckpoints
        or manager._original_ledger() is not ledger
        or manager._copies is not copies
    ):
        raise ValueError(
            "issued replay capture requires the original checkpoint manager/ledger/copies"
        )
    _witnesses(manager, limits.max_nodes)
    active, thread, checks = False, get_ident(), 0
    roots: list[Any] = [owner, ledger, copies, manager, limits, manager._gate, manager._checkpoints]
    operations: dict[int, Any] = {}
    original_membership = (
        tuple((key, id(value)) for key, value in manager._checkpoints.items())
        if type(manager._checkpoints) is dict and len(manager._checkpoints) <= limits.max_nodes
        else None
    )
    try:
        with manager._exclusive(), ExitStack() as stack:
            restored_check = stack.enter_context(
                lease_restored_replay_capture(owner, ledger, manager)
            )
            active = True

            def verify(entries, roles, runtime) -> set[int]:
                nonlocal checks
                if not active or get_ident() != thread:
                    raise ValueError("issued replay checkpoint reader expired or crossed threads")
                checks += 1
                if checks > 8:
                    raise ValueError("issued replay checkpoint check allowance exhausted")
                owner, ledger, copies, manager, limits, gate, records = roots
                if (
                    manager._gate is not gate
                    or manager._original_ledger() is not ledger
                    or manager._copies is not copies
                    or manager._active is not None
                    or manager._checkpoints is not records
                ):
                    raise ValueError(
                        "issued replay original checkpoint authority/operation changed"
                    )
                known = _witnesses(manager, limits.max_nodes)
                if original_membership != tuple(
                    (key, id(value)) for key, value in manager._checkpoints.items()
                ):
                    raise ValueError("issued replay original checkpoint witness membership changed")
                qualified = restored_check(entries, roles, runtime)
                for model, rows in entries:
                    witness = known.get(id(model))
                    if witness is None:
                        continue
                    if witness.state() is not model:
                        raise ValueError("issued replay checkpoint state identity changed")
                    if id(model) not in operations:
                        operations[id(model)] = _operation(
                            manager, owner, ledger, copies, runtime, witness, limits.max_nodes
                        )
                    op = operations[id(model)]
                    if op.checkpoint is not witness or op.runtime is not runtime:
                        raise ValueError(
                            "issued replay original checkpoint witness/runtime changed"
                        )
                    _verify(manager, op, witness, model, rows, roles)
                    qualified.add(id(model))
                return qualified

            yield verify
    finally:
        active = False
        for op in operations.values():
            op.close()
        operations.clear()
        roots.clear()
