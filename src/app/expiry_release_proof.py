"""Finite original lease dispatch proof, before any cleanup context enters.

Inputs are supported lifecycle roots and their bounded weak holder registry.
The result is refusal or a callback-free descriptor/code attestation, never a
lease token. This module acquires no locks, calls no ports and retains no raw
payloads. Interpreter integrity and defining-module pin constants are trusted.
"""

from threading import Lock
from types import FunctionType, GetSetDescriptorType
from typing import Any
from weakref import ReferenceType

_LOCK_TYPE = type(Lock())


def pin_release_methods(kind: type, names: tuple[str, ...]) -> tuple:
    """Defining modules pin their own functions before constructor callbacks."""
    result = []
    for name in names:
        method = type.__getattribute__(kind, "__dict__")[name]
        wrapped = method.__dict__.get("__wrapped__")
        result.append(
            (name, method, method.__code__, wrapped, None if wrapped is None else wrapped.__code__)
        )
    return tuple(result)


def _fields(value: Any) -> dict:
    kind = type(value)
    if (
        type(kind) is not type
        or type.__getattribute__(kind, "__getattribute__") is not object.__getattribute__
    ):
        raise ValueError("expiry release requires plain original attributes")
    descriptor = next(
        (base.__dict__["__dict__"] for base in kind.__mro__ if "__dict__" in base.__dict__), None
    )
    if type(descriptor) is not GetSetDescriptorType:
        raise ValueError("expiry release requires original instance dictionary")
    fields = object.__getattribute__(value, "__dict__")
    if (
        type(fields) is not dict
        or len(fields) > 512
        or any(type(key) is not str or len(key) > 128 for key in fields)
    ):
        raise ValueError("expiry release instance fields exceed original bound")
    return fields


def require_release_methods(value: Any, pins: tuple, *, slotted: bool = False) -> None:
    fields = None if slotted else _fields(value)
    kind = type(value)
    for name, method, code, wrapped, wrapped_code in pins:
        if fields is not None and name in fields:
            raise ValueError("expiry release instance dispatch was shadowed")
        descriptor = next(
            (base.__dict__[name] for base in kind.__mro__ if name in base.__dict__), None
        )
        if type(descriptor) is FunctionType:
            _require_function_fields(descriptor)
        if (
            descriptor is not method
            or type(descriptor) is not FunctionType
            or descriptor.__code__ is not code
            or descriptor.__dict__.get("__wrapped__") is not wrapped
        ):
            raise ValueError("expiry release original class dispatch changed")
        if wrapped is not None:
            closure = descriptor.__closure__
            if (
                wrapped.__code__ is not wrapped_code
                or closure is None
                or len(closure) != 1
                or closure[0].cell_contents is not wrapped
            ):
                raise ValueError("expiry release original generator changed")


def _require_function_fields(function: Any) -> None:
    # Why: a foreign hash-colliding key can invoke __eq__ during dict.get.
    if type(function) is not FunctionType:
        raise ValueError("expiry release requires original function kernel")
    fields = function.__dict__
    if (
        type(fields) is not dict
        or len(fields) > 128
        or any(type(key) is not str or len(key) > 128 for key in fields)
    ):
        raise ValueError("expiry release function fields require exact bounded strings")


def require_cleanup_release_dispatch(life: Any) -> None:
    # Why: final restored attributes cannot identify a queued opaque __exit__.
    # Check every original dispatch BEFORE the first context can enqueue one.
    from src.app import actor_shadow as actor, candidate_checkpoint as checkpoint
    from src.app import managed_experience as experience, payload_ownership as ownership
    from src.app import resource_sharing as sharing, serving_promotion as promotion
    from src.app import managed_data_lifecycle as lifecycle

    if type(life) is not lifecycle.ManagedDataLifecycle:
        raise ValueError("expiry release requires original lifecycle")
    require_release_methods(life, lifecycle._EXPIRY_RELEASE_PINS)
    if lifecycle.ExitStack is not lifecycle._EXPIRY_EXIT_STACK:
        raise ValueError("expiry release lifecycle stack binding changed")
    fields = _fields(life)
    owner, registry = fields["_owner"], fields["_registry"]
    shared = _fields(owner)["_shared"]
    gate = _fields(shared)["_sharing"]
    for value, kind, pins in (
        (owner, experience.ManagedExperienceOwner, experience._EXPIRY_RELEASE_PINS),
        (registry, ownership.PayloadOwnershipRegistry, ownership._EXPIRY_RELEASE_PINS),
        (gate, sharing.ServingPriorityGate, sharing._EXPIRY_RELEASE_PINS),
    ):
        if type(value) is not kind:
            raise ValueError("expiry release root kind changed")
        require_release_methods(value, pins)
        if type(_fields(value)["_gate"]) is not _LOCK_TYPE:
            raise ValueError("expiry release requires original lock primitive")
    state = _fields(gate)
    if (
        any(type(state[name]) is not bool for name in ("_checkpointing", "_training", "_paused"))
        or type(state["_serving"]) is not int
        or state["_serving"] < 0
    ):
        raise ValueError("expiry release requires exact sharing protocol scalars")
    entries = _fields(registry)["_holders"]
    from src.core.data_retention import DataRetentionPolicy
    from src.core.payload_ownership import PayloadOwnershipLimits

    policy = fields["_policy"]
    if type(policy) is not DataRetentionPolicy:
        raise ValueError("expiry release requires original policy kind")
    limits = _fields(policy)["holders"]
    if type(limits) is not PayloadOwnershipLimits:
        raise ValueError("expiry release requires original holder limits")
    maximum = _fields(limits)["max_live_holders"]
    if type(maximum) is not int or maximum < 1:
        raise ValueError("expiry release requires original holder limit scalar")
    maximum = min(maximum, 4096)
    if type(entries) is not dict or len(entries) > maximum:
        raise ValueError("expiry release holder inventory exceeds original bound")
    for number, entry in entries.items():
        if (
            type(number) is not int
            or number < 1
            or type(entry) is not tuple
            or len(entry) != 2
            or type(entry[0]) is not str
            or entry[0] not in ("actor", "candidate", "checkpoint", "promotion")
            or type(entry[1]) is not ReferenceType
        ):
            raise ValueError("expiry release requires closed weak holder entries")
        holder = entry[1]()
        if holder is None:
            continue
        kind = type(holder)
        if entry[0] == "actor" and kind in (actor.StableActor, promotion.PromotableActor):
            require_release_methods(
                holder,
                actor._EXPIRY_ACTOR_RELEASE_PINS
                if kind is actor.StableActor
                else actor._EXPIRY_PROMOTABLE_LEASE_PINS,
            )
            if kind is promotion.PromotableActor:
                require_release_methods(holder, promotion._EXPIRY_ACTOR_RELEASE_PINS)
        elif entry[0] == "candidate" and kind is actor.ActorShadowRuntime:
            require_release_methods(holder, actor._EXPIRY_CANDIDATE_RELEASE_PINS)
        elif entry[0] == "checkpoint" and kind is checkpoint.CandidateCheckpointController:
            require_release_methods(holder, checkpoint._EXPIRY_RELEASE_PINS)
        elif entry[0] == "promotion" and kind is promotion.ServingPromotionController:
            require_release_methods(holder, promotion._EXPIRY_RELEASE_PINS)
        else:
            raise ValueError("expiry release requires original holder kind")
        if _fields(holder).get("_payload_ready") is not True:
            raise ValueError("expiry release holder is not originally ready")
    # These methods delegate the lock helper through defining-module bindings.
    for module in (actor, ownership):
        if module.__dict__["lease_payload_lock"] is not ownership._EXPIRY_LOCK_HELPER:
            raise ValueError("expiry release lock helper dispatch changed")
    if ownership._EXPIRY_LOCK_HELPER.__code__ is not ownership._EXPIRY_LOCK_CODE:
        raise ValueError("expiry release lock helper code changed")
    generator: FunctionType = ownership._EXPIRY_LOCK_GENERATOR
    if type(generator) is not FunctionType:
        raise ValueError("expiry release requires original function kernel")
    _require_function_fields(ownership._EXPIRY_LOCK_HELPER)
    if (
        ownership._EXPIRY_LOCK_HELPER.__dict__.get("__wrapped__") is not generator
        or generator.__code__ is not ownership._EXPIRY_LOCK_GENERATOR_CODE
        or ownership.ExitStack is not ownership._EXPIRY_EXIT_STACK
    ):
        raise ValueError("expiry release lock helper kernel or stack binding changed")
    closure = ownership._EXPIRY_LOCK_HELPER.__closure__
    if (
        closure is None
        or len(closure) != 1
        or closure[0].cell_contents is not ownership._EXPIRY_LOCK_GENERATOR
    ):
        raise ValueError("expiry release lock helper generator closure changed")
