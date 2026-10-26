"""Lease-aware read-only admission of original retained checkpoint replay copies.

The original ledger lease has already paid its one capture admission. This
bridge holds the existing nonblocking copied-witness gate and checks current
original roles/weak witnesses without entering public origins(), native methods,
owner gates, new admissions, or restore/publication authority. Observed checkpoint
constructor copies qualify; the optional checkpoint reader separately qualifies
original issued snapshots. Other holder roles remain unsupported.
"""

from contextlib import ExitStack, contextmanager
from dataclasses import fields
from threading import get_ident
from typing import Any, Callable, Iterator
from weakref import ReferenceType

from src.app.managed_replay_copies import ManagedReplayCopies, _Copy
from src.app.managed_replay_origins import ManagedReplayOrigins
from src.app.replay_checkpoint_capture import lease_replay_checkpoint_capture
from src.core.managed_composite_state import CompositeCaptureLimits
from src.core.payload_ownership import OwnedPayloadGroup, PayloadHolderMetadata, PayloadReferences


def _entries(value: Any, limit: int) -> tuple[Any, ...]:
    if type(value) is not tuple or len(value) > limit:
        raise ValueError("retained replay inventory exceeds exact entry bound")
    seen = set()
    rows = 0
    for entry in value:
        if type(entry) is not tuple or len(entry) != 2 or type(entry[1]) is not tuple:
            raise ValueError("retained replay requires exact model/row reference tuples")
        if id(entry[0]) in seen:
            raise ValueError("duplicate retained replay model identity")
        seen.add(id(entry[0]))
        rows += len(entry[1])
        if rows > limit:
            raise ValueError("retained replay rows exceed aggregate bound")
    return value


def _role_inventory(
    groups: Any, limits: CompositeCaptureLimits, inventory: Callable[..., Any]
) -> dict[int, list[Any]]:
    """Retain role information that an aggregate deduplicating inventory loses."""
    if type(groups) is not tuple or len(groups) > limits.records.max_records:
        raise ValueError("retained replay groups exceed original capture bound")
    roles: dict[int, list[Any]] = {}
    count = entries = rows = 0
    enrollments = set()
    for group in groups:
        if type(group) is not OwnedPayloadGroup or vars(group).keys() != {"holder", "references"}:
            raise ValueError("retained replay requires exact owned payload groups")
        holder, references = group.holder, group.references
        if type(holder) is not PayloadHolderMetadata or vars(holder).keys() != {
            "enrollment",
            "kind",
            "ready",
        }:
            raise ValueError("retained replay holder metadata fields changed")
        PayloadHolderMetadata.__post_init__(holder)
        if not holder.ready or holder.enrollment in enrollments:
            raise ValueError("retained replay holder is not uniquely enrolled and ready")
        enrollments.add(holder.enrollment)
        if type(references) is not PayloadReferences or vars(references).keys() != {
            "models",
            "inboxes",
            "snapshots",
            "auxiliary",
        }:
            raise ValueError("retained replay payload reference schema changed")
        for name in ("models", "inboxes", "snapshots", "auxiliary"):
            items = getattr(references, name)
            if type(items) is not tuple:
                raise ValueError("retained replay references require exact tuples")
            count += len(items)
            if count > limits.max_nodes:
                raise ValueError("retained replay referenced items exceed aggregate bound")
            if name not in ("models", "snapshots"):
                continue
            for item in items:
                singleton = OwnedPayloadGroup(holder, PayloadReferences(**{name: (item,)}))
                found = _entries(inventory((singleton,), limits), limits.max_nodes)
                entries += len(found)
                rows += sum(len(entry[1]) for entry in found)
                if entries > limits.max_nodes or rows > limits.max_nodes:
                    raise ValueError("retained replay role inventory exceeds aggregate bound")
                for model, retained in found:
                    roles.setdefault(id(model), []).append((model, retained, holder, name, item))
    return roles


def _same_rows(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return len(left) == len(right) and all(a is b for a, b in zip(left, right))


def _witness(copies: ManagedReplayCopies, model: object, limit: int) -> Any:
    if type(copies._copies) is not dict or len(copies._copies) > limit:
        raise ValueError("retained replay witness inventory changed")
    witness = copies._copies.get(id(model))
    if type(witness) is not _Copy or vars(witness).keys() != {
        field.name for field in fields(_Copy)
    }:
        raise ValueError("retained replay lacks an exact original copied witness")
    for name in ("model", "controller", "builder", "policy", "probe"):
        if type(getattr(witness, name)) is not ReferenceType:
            raise ValueError("retained replay witness lost its original weak reference")
    for name in ("enrollment", "position", "attempt", "models_identity", "attempt_limit"):
        value = getattr(witness, name)
        if type(value) is not int or value < (0 if name == "position" else 1):
            raise ValueError("retained replay witness requires exact original integer history")
    return witness


def _qualify(
    owner: Any,
    ledger: ManagedReplayOrigins,
    copies: ManagedReplayCopies,
    model: object,
    rows: tuple[Any, ...],
    roles: dict[int, list[Any]],
    runtime: Any,
    limits: CompositeCaptureLimits,
) -> None:
    witness = _witness(copies, model, limits.max_nodes)
    if type(witness.rows) is not tuple or len(witness.rows) > limits.max_nodes:
        raise ValueError("retained replay witness row bound changed")
    controller = witness.controller()
    _, enrollment = copies._controller(owner, controller)
    if (
        type(controller._attempt_limit) is not int
        or controller._attempt_limit < 1
        or len(controller._models) > limits.max_nodes
    ):
        raise ValueError("retained replay controller history exceeds exact bounds")
    # Pin the actual recipient while the native copy is observed, before policy
    # callbacks can replace the controller's wrapper around that same model.
    learner_reference = getattr(witness, "learner", None)
    if type(learner_reference) is not ReferenceType or learner_reference() is None:
        raise ValueError("retained replay lacks its original copied learner witness")
    learner = learner_reference()
    if ledger._ports.model_reference(learner) is not model:
        raise ValueError("retained replay original learner now references another model")
    copies._require_copy(witness, controller, learner, model, enrollment)
    original = owner._lifecycle._registry._holders.get(enrollment)
    if original is None or original[0] != "checkpoint" or original[1]() is not controller:
        raise ValueError("retained replay controller original enrollment changed")
    appearances = roles.get(id(model), ())
    if not appearances:
        raise ValueError("retained replay model has no original checkpoint role")
    for current, retained, holder, collection, item in appearances:
        if (
            current is not model
            or collection != "models"
            or holder.kind != "checkpoint"
            or holder.enrollment != enrollment
            or item is not learner
            or not _same_rows(retained, rows)
        ):
            raise ValueError("retained replay copy appeared in an unqualified holder role")
    retained = ledger._inventory(model)
    if not _same_rows(retained, rows):
        raise ValueError("retained replay inventory differs from originally witnessed rows")
    copies._verify(ledger, owner, runtime, model, witness)


@contextmanager
def lease_replay_copy_capture(
    owner: Any,
    groups: Any,
    limits: CompositeCaptureLimits,
    inventory: Callable[..., Any],
    ledger: ManagedReplayOrigins,
    copies: ManagedReplayCopies,
    checkpoints=None,
) -> Iterator[Callable[[tuple[Any, ...]], None]]:
    """Yield at most eight whole-inventory checks under the existing ledger lease."""
    if (
        type(copies) is not ManagedReplayCopies
        or type(ledger) is not ManagedReplayOrigins
        or not callable(inventory)
    ):
        raise ValueError(
            "retained replay capture requires exact original copies/ledger and inventory"
        )
    if copies._original_ledger() is not ledger or ledger._original_owner() is not owner:
        raise ValueError("retained replay copies belong to another original ledger/owner")
    active, thread, checks = False, get_ident(), 0
    gate = copies._gate
    # Escaped checks retain only this list, which is emptied on every exit.
    roots: list[Any] = [owner, groups, limits, inventory, ledger, copies, gate]
    try:
        with copies._exclusive(), ExitStack() as stack:
            checkpoint_check = (
                None
                if checkpoints is None
                else stack.enter_context(
                    lease_replay_checkpoint_capture(owner, ledger, copies, checkpoints, limits)
                )
            )
            active = True

            def verify(entries: tuple[Any, ...]) -> None:
                nonlocal checks
                if not active or get_ident() != thread:
                    raise ValueError("retained replay capture check expired or crossed threads")
                checks += 1
                if checks > 8:
                    raise ValueError("retained replay capture check allowance exhausted")
                owner, groups, limits, inventory, ledger, copies, gate = roots
                if copies._gate is not gate or copies._original_ledger() is not ledger:
                    raise ValueError("retained replay original witness gate/ledger changed")
                requested = _entries(entries, limits.max_nodes)
                if any(not rows for _, rows in requested):
                    raise ValueError("retained replay qualification requires nonempty foreign rows")
                runtime, source, original_rows = copies._source(ledger, owner, leased=True)
                roles = _role_inventory(groups, limits, inventory)
                issued = (
                    set()
                    if checkpoint_check is None
                    else checkpoint_check(requested, roles, runtime)
                )
                foreign = {id(model): (model, rows) for model, rows in requested}
                for appearances in roles.values():
                    for model, rows, _, _, _ in appearances:
                        if rows and model is source:
                            if not _same_rows(rows, original_rows):
                                raise ValueError("original replay role inventory changed")
                        elif rows and id(model) not in foreign:
                            raise ValueError(
                                "foreign replay role was omitted from complete inventory"
                            )
                for model, rows in requested:
                    if model is source:
                        raise ValueError(
                            "retained copy verifier cannot replace original ledger checks"
                        )
                    if id(model) not in issued:
                        _qualify(owner, ledger, copies, model, rows, roles, runtime, limits)

            yield verify
    finally:
        active = False
        roots.clear()
