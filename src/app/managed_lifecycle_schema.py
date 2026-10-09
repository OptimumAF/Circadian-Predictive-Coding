"""Complete source-field contract and detached record preparation.

Enumerates every currently supported lifecycle/owner/driver/registry/copy field.
Inputs are exact original owners or validated typed records. This module does not
acquire a coherent live capture,invoke ports,serialize authority or restore state.
"""

from copy import deepcopy

from src.app.expiry_history_birth import EPHEMERAL_LIFECYCLE_FIELDS, require_expiry_birth_source

from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_experience import ManagedExperienceOwner
from src.app.payload_copy_budget import PayloadCopyBudget
from src.app.payload_ownership import PayloadOwnershipRegistry
from src.app.retention_expiry import RetentionExpiryDriver
from src.core.managed_lifecycle_state import (
    DRIVER_REFERENCES,
    LIFECYCLE_REFERENCES,
    OWNER_REFERENCES,
    LifecycleCaptureLimits,
    ManagedLifecycleCapture,
)
from src.core.managed_lifecycle_validation import validate_lifecycle_capture

OWNER_STATE_FIELDS = {
    "_catalog": "owner.catalog",
    "_limits": "owner.limits",
    "_opted_out": "owner.opted_out",
    "_revoked_keys": "owner.revoked_keys",
    "_declaration_ticks": "owner.declaration_ticks",
    "_declaration_seconds": "owner.declaration_seconds",
}
LIFECYCLE_STATE_FIELDS = {
    "_policy": "lifecycle.policy",
    "_admitted_bytes": "lifecycle.admitted_bytes",
    "_last_tick": "lifecycle.last_tick",
    "_failed": "lifecycle.failed",
    "_last_seconds": "lifecycle.last_seconds",
    "_auxiliary_started_at": "lifecycle.auxiliary_started_at",
    "_retention_fault": "lifecycle.retention_fault",
}
DRIVER_STATE_FIELDS = {
    "_limits": "driver.limits",
    "_state": "driver.state",
    "_polls": "driver.polls",
    "_purges": "driver.purges",
    "_cleanup_attempts": "driver.cleanup_attempts",
    "_created_at": "driver.created_at",
    "_held": "driver.held",
    "_pending": "driver.pending",
    "_error_type": "driver.error_type",
}
REGISTRY_STATE_FIELDS = {
    "_limits": "registry.limits",
    "_total": "registry.total_enrollments",
    "_holders": "registry.holders",
}
COPY_STATE_FIELDS = {"_limits": "copy.limits", "_charged": "copy.charged_bytes"}
SOURCE_FIELDS = {
    ManagedExperienceOwner: frozenset(OWNER_STATE_FIELDS) | set(OWNER_REFERENCES),
    ManagedDataLifecycle: frozenset(LIFECYCLE_STATE_FIELDS)
    | set(LIFECYCLE_REFERENCES)
    | EPHEMERAL_LIFECYCLE_FIELDS,
    RetentionExpiryDriver: frozenset(DRIVER_STATE_FIELDS) | set(DRIVER_REFERENCES),
    PayloadOwnershipRegistry: frozenset(REGISTRY_STATE_FIELDS) | {"_gate", "_lifecycle"},
    PayloadCopyBudget: frozenset(COPY_STATE_FIELDS) | {"_gate"},
}


def require_lifecycle_source_schema(owner, lifecycle, registry, *, copy=None, driver=None) -> None:
    """Refuse unknown complete source schemas without enumerating or mutating ports."""
    for value, kind in (
        (owner, ManagedExperienceOwner),
        (lifecycle, ManagedDataLifecycle),
        (registry, PayloadOwnershipRegistry),
    ):
        if (
            type(value) is not kind
            or any(type(name) is not str or len(name) > 64 for name in vars(value))
            or vars(value).keys() != SOURCE_FIELDS[kind]
        ):
            raise ValueError(
                "lifecycle source differs from complete supported private field schema"
            )
    for value, optional_kind in ((copy, PayloadCopyBudget), (driver, RetentionExpiryDriver)):
        if value is not None and (
            type(value) is not optional_kind
            or any(type(name) is not str or len(name) > 64 for name in vars(value))
            or vars(value).keys() != SOURCE_FIELDS[optional_kind]
        ):
            raise ValueError("optional lifecycle source differs from complete supported schema")
    if (
        owner._lifecycle is not lifecycle
        or lifecycle._owner is not owner
        or lifecycle._registry is not registry
        or registry._lifecycle is not lifecycle
        or lifecycle._copy_budget is not copy
        or lifecycle._retention_driver is not driver
    ):
        raise ValueError(
            "lifecycle source roots differ from original complete authority relationships"
        )
    require_expiry_birth_source(lifecycle)


def detach_lifecycle_record(
    capture: ManagedLifecycleCapture, limits: LifecycleCaptureLimits
) -> ManagedLifecycleCapture:
    """Detach validated metadata as one graph;retain all original live references.

    Caller must first obtain a coherent original-owner capture. This preparation
    does not provide that lease or prove that the supplied records came from it.
    """
    validate_lifecycle_capture(capture, limits)
    detached = ManagedLifecycleCapture(deepcopy(capture.metadata), capture.authority)
    validate_lifecycle_capture(detached, limits)
    return detached
