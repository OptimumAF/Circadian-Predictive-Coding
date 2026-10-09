"""Finite original metadata dispatch attestation for synchronous expiry.

Pins contain defining-class metadata only, never graph instances or payloads.
Checks read raw type dictionaries without invoking descriptors or validators.
The defining modules, their helper globals and interpreter/stdlib kernels are
trusted. This proves supported public class dispatch, not process integrity.
"""

import copyreg
from dataclasses import Field
from types import FunctionType
from typing import Any

from src.app.actor_shadow import ActorShadowRuntime, StableActor
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.erased_inbox_origins import ErasedInboxOrigin
from src.app.experience_inbox import ExperienceInbox
from src.app.managed_experience import ManagedExperienceOwner
from src.app.managed_inbox_origins import ManagedInboxOrigins, _InboxOrigin
from src.app.managed_replay_origins import ManagedReplayOrigins, _Row
from src.app.payload_copy_budget import PayloadCopyBudget
from src.app.payload_ownership import PayloadOwnershipRegistry
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.serving_promotion import PromotableActor, ServingPromotionController, _Bundle, _Slot
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.app.untrained_inbox_origins import UntrainedInboxOrigin
from src.core.checkpoint_content import CheckpointContentLimits
from src.core.data_erasure import ErasedExperience
from src.core.data_lifecycle import (
    DataConsent,
    DataProvenance,
    LifecycleDeclaration,
    LifecycleLimits,
)
from src.core.data_retention import DataCleanupReport, DataRetentionPolicy
from src.core.experience import AppliedExperience, LogicalClock
from src.core.inbox_origin import InboxOriginData
from src.core.learner_ports import TrainingDiagnostic
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import (
    OwnedPayloadGroup,
    PayloadHolderMetadata,
    PayloadOwnershipLimits,
    PayloadReferences,
)
from src.core.replay_origin import (
    ReplayOriginAccounting,
    ReplayOriginAdmission,
    ReplayOriginData,
    ReplayOriginLimits,
)
from src.core.resource_sharing import SharingLimits
from src.core.untrained_inbox_origin import UntrainedInboxOriginData


# The trusted stdlib helper is private and absent from copyreg's type stub.
_slotnames = copyreg.__dict__["_slotnames"]


def _function_pin(function: Any) -> tuple | None:
    if type(function) is not FunctionType:
        return None
    keywords = function.__kwdefaults__
    return (
        function,
        function.__code__,
        function.__defaults__,
        keywords,
        () if keywords is None else tuple(keywords.items()),
        ()
        if function.__closure__ is None
        else tuple(cell.cell_contents for cell in function.__closure__),
    )


def pin_validation_types(kinds: tuple[type, ...], *, getters_only: bool = False) -> tuple:
    """Defining modules freeze their finite class closure before callbacks."""
    result = []
    for kind in kinds:
        bases = []
        for base in type.__getattribute__(kind, "__mro__"):
            # Why: deepcopy lazily installs this stdlib cache. Create it before
            # the original namespace freeze, never normalize during reproof.
            _slotnames(base)
            namespace = type.__getattribute__(base, "__dict__")
            slot_cache = namespace.get("__slotnames__")
            if slot_cache is not None and (
                type(slot_cache) is not list
                or len(slot_cache) > 128
                or any(type(name) is not str or len(name) > 128 for name in slot_cache)
            ):
                raise ValueError("expiry original slot-name cache requires bounded strings")
            slots = None if slot_cache is None else tuple(slot_cache)
            members = []
            for name, value in namespace.items():
                # Report validation is intentionally opaque, outside the gate.
                ignored = (
                    getters_only
                    and type(value) is FunctionType
                    and name not in ("__getattribute__", "__setattr__")
                ) or (base is DataCleanupReport and name == "__post_init__")
                functions: tuple[tuple | None, ...] = ()
                if not ignored:
                    if type(value) is FunctionType:
                        functions = (_function_pin(value),)
                    elif type(value) in (classmethod, staticmethod):
                        functions = (_function_pin(value.__func__),)
                    elif type(value) is property:
                        functions = tuple(
                            _function_pin(f) for f in (value.fget, value.fset, value.fdel)
                        )
                members.append((name, None if ignored else value, ignored, functions))
            fields = namespace.get("__dataclass_fields__")
            field_pins = (
                ()
                if fields is None
                else tuple(
                    (
                        name,
                        field,
                        tuple(
                            (slot, object.__getattribute__(field, slot)) for slot in Field.__slots__
                        ),
                    )
                    for name, field in fields.items()
                )
            )
            bases.append((base, tuple(members), fields, field_pins, slots))
        result.append((kind, type.__getattribute__(kind, "__mro__"), tuple(bases)))
    return tuple(result)


def _require_function(pin: Any) -> None:
    if pin is None:
        return
    function, code, defaults, keywords, items, cells = pin
    if (
        function.__code__ is not code
        or function.__defaults__ is not defaults
        or function.__kwdefaults__ is not keywords
    ):
        raise ValueError("expiry original validator function changed")
    if keywords is not None and (
        type(keywords) is not dict
        or len(keywords) != len(items)
        or any(type(name) is not str or len(name) > 128 for name in keywords)
        or any(name not in keywords or keywords[name] is not value for name, value in items)
    ):
        raise ValueError("expiry original validator keyword defaults changed")
    closure = function.__closure__
    if (0 if closure is None else len(closure)) != len(cells) or (
        closure is not None
        and any(cell.cell_contents is not value for cell, value in zip(closure, cells))
    ):
        raise ValueError("expiry original validator closure changed")


def require_validation_types(pins: tuple) -> None:
    """Reject changed dispatch/getters before ordinary metadata field reads."""
    for kind, mro, bases in pins:
        if type(kind) is not type or type.__getattribute__(kind, "__mro__") is not mro:
            raise ValueError("expiry original validation type hierarchy changed")
        for base, members, fields, field_pins, slots in bases:
            namespace = type.__getattribute__(base, "__dict__")
            if len(namespace) != len(members) or any(
                type(name) is not str or len(name) > 128 for name in namespace
            ):
                raise ValueError("expiry original validator namespace changed")
            for name, original, ignored, functions in members:
                if name not in namespace or (not ignored and namespace[name] is not original):
                    raise ValueError("expiry original validator/getter changed")
                for function in functions:
                    _require_function(function)
            if slots is not None:
                slot_cache = namespace["__slotnames__"]
                if (
                    type(slot_cache) is not list
                    or len(slot_cache) != len(slots)
                    or any(type(name) is not str or len(name) > 128 for name in slot_cache)
                    or tuple(slot_cache) != slots
                ):
                    raise ValueError("expiry original slot-name cache changed")
            if fields is not None:
                if (
                    type(fields) is not dict
                    or len(fields) != len(field_pins)
                    or any(type(name) is not str or len(name) > 128 for name in fields)
                ):
                    raise ValueError("expiry original dataclass field grammar changed")
                if any(name != original[0] for name, original in zip(fields, field_pins)):
                    raise ValueError("expiry original dataclass field order changed")
                for name, field, slots in field_pins:
                    if fields.get(name) is not field or type(field) is not Field:
                        raise ValueError("expiry original dataclass field identity changed")
                    if any(
                        object.__getattribute__(field, slot) is not value for slot, value in slots
                    ):
                        raise ValueError("expiry original dataclass field metadata changed")


_VALIDATION_PINS = pin_validation_types(
    (
        DataRetentionPolicy,
        CheckpointContentLimits,
        ReplayOriginLimits,
        LifecycleLimits,
        DataProvenance,
        DataConsent,
        LifecycleDeclaration,
        InboxOriginData,
        ReplayOriginData,
        ErasedExperience,
        ReplayOriginAdmission,
        ReplayOriginAccounting,
        UntrainedInboxOriginData,
        AppliedExperience,
        TrainingDiagnostic,
        DataCleanupReport,
        PayloadCopyLimits,
        PayloadOwnershipLimits,
        SharingLimits,
        ToyExecutionBudget,
        OwnedPayloadGroup,
        PayloadHolderMetadata,
        PayloadReferences,
        LogicalClock,
        ManagedInboxOrigins,
        ManagedReplayOrigins,
        _InboxOrigin,
        _Row,
        ErasedInboxOrigin,
        UntrainedInboxOrigin,
        ActorShadowRuntime,
        StableActor,
        PromotableActor,
        CandidateCheckpointController,
        ServingPromotionController,
        _Bundle,
        _Slot,
        ExperienceInbox,
        ManagedExperienceOwner,
        PayloadCopyBudget,
        PayloadOwnershipRegistry,
        ResourceSharedRuntime,
        ServingPriorityGate,
        ToyBudgetSession,
    )
)


def require_original_metadata_dispatch() -> None:
    require_validation_types(_VALIDATION_PINS)
