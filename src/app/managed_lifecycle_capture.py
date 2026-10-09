"""Read complete lifecycle metadata under original nonblocking owner leases.

Inputs: an exact installed owner and independent record/string limits. Output:
detached validated metadata plus original authority references. No clock reads,
measurement/model callbacks, weak-entry pruning, cleanup, byte encoding or restore.
"""

from contextlib import ExitStack, contextmanager
from typing import Callable, Iterator
from weakref import ReferenceType

from src.app.managed_experience import ManagedExperienceOwner
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_lifecycle_schema import (
    detach_lifecycle_record,
    require_lifecycle_source_schema,
    SOURCE_FIELDS,
)
from src.app.payload_ownership import PayloadOwnershipBusy, lease_payload_lock
from src.core.data_lifecycle import LifecycleDeclaration
from src.core.managed_lifecycle_state import (
    DRIVER_REFERENCES,
    LIFECYCLE_REFERENCES,
    OWNER_REFERENCES,
    AuthorityReference,
    CopyBudgetState,
    LifecycleAccountingState,
    LifecycleCaptureLimits,
    LifecycleMetadata,
    ManagedLifecycleCapture,
    ManagedOwnerState,
    OwnershipEnrollment,
    OwnershipRegistryState,
    RetentionDriverRecord,
)
from src.core.managed_lifecycle_validation import (
    require_lifecycle_capture_limits,
    require_state_record,
)
from src.core.payload_ownership import require_ownership_kind


def _preflight_collections(owner, registry, limits) -> None:
    collections = (
        (owner._catalog, dict),
        (owner._opted_out, set),
        (owner._revoked_keys, set),
        (owner._declaration_ticks, dict),
        (owner._declaration_seconds, dict),
        (registry._holders, dict),
    )
    if any(type(value) is not kind for value, kind in collections):
        raise ValueError("unsupported original lifecycle history container")
    if sum(len(value) for value, _ in collections) > limits.max_records:
        raise ValueError("aggregate original lifecycle history exceeds capture bound")
    for subject in owner._opted_out:
        if type(subject) is not str or len(subject) > limits.max_identifier_bytes:
            raise ValueError("unsupported or oversized original subject")
    for keys in (
        owner._catalog,
        owner._revoked_keys,
        owner._declaration_ticks,
        owner._declaration_seconds,
    ):
        for key in keys:
            if (
                type(key) is not tuple
                or len(key) != 2
                or any(
                    type(value) is not str or len(value) > limits.max_identifier_bytes
                    for value in key
                )
            ):
                raise ValueError("unsupported or oversized original lifecycle key")
    for key, declaration in owner._catalog.items():
        require_state_record(declaration, LifecycleDeclaration)
        if (
            type(declaration.key) is not tuple
            or len(declaration.key) != 2
            or any(type(value) is not str for value in declaration.key)
        ):
            raise ValueError("unsupported original declaration key")
        if declaration.key != key:
            raise ValueError("original catalog key differs from its declaration")


def _lease_holders(lifecycle, stack):
    registry = lifecycle._registry
    retained = []
    live = []
    for number, entry in registry._holders.items():
        if type(entry) is not tuple or len(entry) != 2 or type(entry[1]) is not ReferenceType:
            raise ValueError("unsupported original weak holder enrollment")
        kind, reference = entry
        require_ownership_kind(kind)
        holder = reference()
        if holder is not None:
            lifecycle._require_holder(kind, holder)
            if holder._payload_ready is not True:
                raise ValueError("cannot lease incomplete original holder")
            try:
                stack.enter_context(holder._payload_exclusive())
            except ValueError as error:
                raise PayloadOwnershipBusy(str(error)) from error
            lifecycle._require_holder(kind, holder)
            live.append(holder)  # Keep weak holders alive throughout detachment.
        retained.append(OwnershipEnrollment(number, kind, None if holder is None else True))
    if not any(holder is lifecycle._actor for holder in live) or not any(
        holder is lifecycle._shared._runtime for holder in live
    ):
        raise ValueError("original actor and current runtime enrollment are required")
    return tuple(retained), live


def _require_original_relationships(lifecycle) -> None:
    from src.app.expiry_history_birth import require_expiry_capture_source

    require_expiry_capture_source(lifecycle)
    runtime = lifecycle._shared._runtime
    if (
        runtime._payload_lineage is not lifecycle._lineage
        or runtime._inbox._clock is not lifecycle._clock
        or runtime.actor is not lifecycle._actor
        or runtime._budget is not lifecycle._budget
        or lifecycle._actor._payload_registry is not lifecycle._registry
        or lifecycle._shared._sharing is not lifecycle._sharing
        or lifecycle._sharing._resource is not lifecycle._resource
        or lifecycle._budget.clock is not lifecycle._wall_clock
        or lifecycle._budget.progress is not lifecycle._progress
        or lifecycle._budget.process_rss_sampler is not lifecycle._sampler
        or lifecycle._budget.budget is not lifecycle._budget_policy
        or lifecycle._sharing._limits is not lifecycle._sharing_limits
    ):
        raise ValueError("original lifecycle lineage/clock/budget/sharing authority changed")


def _driver_record(driver):
    if driver is None:
        return None
    return RetentionDriverRecord(
        driver._limits,
        driver._state,
        driver._polls,
        driver._purges,
        driver._cleanup_attempts,
        driver._created_at,
        driver._held,
        driver._pending,
        driver._error_type,
        driver._stop.is_set(),
        driver._wake.is_set(),
        driver._purged.is_set(),
        driver._thread is not None,
        driver._thread is not None and driver._thread.is_alive(),
    )


def _metadata(owner, lifecycle, holders) -> LifecycleMetadata:
    registry, copy = lifecycle._registry, lifecycle._copy_budget
    return LifecycleMetadata(
        1,
        ManagedOwnerState(
            owner._limits,
            tuple(owner._catalog.values()),
            tuple(sorted(owner._opted_out)),
            tuple(sorted(owner._revoked_keys)),
            tuple(owner._declaration_ticks.items()),
            tuple(owner._declaration_seconds.items()),
        ),
        LifecycleAccountingState(
            lifecycle._policy,
            lifecycle._admitted_bytes,
            lifecycle._last_tick,
            lifecycle._failed,
            lifecycle._last_seconds,
            lifecycle._auxiliary_started_at,
            lifecycle._retention_fault,
        ),
        OwnershipRegistryState(registry._limits, registry._total, holders),
        None if copy is None else CopyBudgetState(copy._limits, copy._charged),
        _driver_record(lifecycle._retention_driver),
    )


def _authority(owner, lifecycle) -> tuple[AuthorityReference, ...]:
    registry, copy, driver = (
        lifecycle._registry,
        lifecycle._copy_budget,
        lifecycle._retention_driver,
    )
    refs = {
        "root.owner": owner,
        "root.lifecycle": lifecycle,
        "root.registry": registry,
        "root.copy": copy,
        "root.driver": driver,
    }
    for name, value, fields in (
        ("owner", owner, OWNER_REFERENCES),
        ("lifecycle", lifecycle, LIFECYCLE_REFERENCES),
        ("driver", driver, DRIVER_REFERENCES),
    ):
        for field in fields:
            refs[name + "." + field] = None if value is None else getattr(value, field)
    for field in ("_gate", "_lifecycle", "_holders"):
        refs["registry." + field] = getattr(registry, field)
    refs["copy._gate"] = None if copy is None else copy._gate
    refs["sharing._retention_hold"] = lifecycle._sharing._retention_hold
    refs["sharing._gate"] = lifecycle._sharing._gate
    return tuple(AuthorityReference(path, value) for path, value in sorted(refs.items()))


@contextmanager
def _lease_lifecycle_sources(
    owner: ManagedExperienceOwner, *, limits: LifecycleCaptureLimits
) -> Iterator[
    tuple[ManagedDataLifecycle, tuple[OwnershipEnrollment, ...], Callable[[int], None] | None]
]:
    """Lease original sources before observation, sizing or attempted copies.

    Original operation leases prohibit reentrant callback capture. State lock is
    separate so stop/wake remain responsive while normal cleanup callbacks run.
    Thread liveness is a point observation, not ownership of Python's thread OS.
    """
    require_lifecycle_capture_limits(limits)
    if (
        type(owner) is not ManagedExperienceOwner
        or any(type(name) is not str or len(name) > 64 for name in vars(owner))
        or vars(owner).keys() != SOURCE_FIELDS[ManagedExperienceOwner]
        or type(owner._lifecycle) is not ManagedDataLifecycle
        or any(type(name) is not str or len(name) > 64 for name in vars(owner._lifecycle))
        or vars(owner._lifecycle).keys() != SOURCE_FIELDS[ManagedDataLifecycle]
    ):
        raise ValueError("capture requires an original installed lifecycle owner")
    lifecycle = owner._lifecycle
    registry, copy, driver = (
        lifecycle._registry,
        lifecycle._copy_budget,
        lifecycle._retention_driver,
    )
    require_lifecycle_source_schema(owner, lifecycle, registry, copy=copy, driver=driver)
    with ExitStack() as stack:
        if driver is not None:
            stack.enter_context(lease_payload_lock(driver._operation_gate, "retention driver"))
            stack.enter_context(lease_payload_lock(driver._state_gate, "retention state"))
        stack.enter_context(owner._operation())
        stack.enter_context(lease_payload_lock(registry._gate, "ownership registry"))
        _preflight_collections(owner, registry, limits)
        holders, live = _lease_holders(lifecycle, stack)
        stack.enter_context(lease_payload_lock(lifecycle._time_gate, "retention time"))
        reserve = None if copy is None else stack.enter_context(copy._lease())
        stack.enter_context(lease_payload_lock(lifecycle._sharing._gate, "sharing"))
        require_lifecycle_source_schema(owner, lifecycle, registry, copy=copy, driver=driver)
        _require_original_relationships(lifecycle)
        yield lifecycle, holders, reserve
        assert live  # Keep every originally live weak holder pinned through the copy.


@contextmanager
def _lease_managed_lifecycle(
    owner: ManagedExperienceOwner, *, limits: LifecycleCaptureLimits
) -> Iterator[ManagedLifecycleCapture]:
    """Observe records inside the original common source interval, without charge."""
    with _lease_lifecycle_sources(owner, limits=limits) as (lifecycle, holders, _):
        yield ManagedLifecycleCapture(
            _metadata(owner, lifecycle, holders), _authority(owner, lifecycle)
        )


def capture_managed_lifecycle(
    owner: ManagedExperienceOwner, *, limits: LifecycleCaptureLimits
) -> ManagedLifecycleCapture:
    """Detach complete metadata during the original common nonblocking lease."""
    with _lease_managed_lifecycle(owner, limits=limits) as capture:
        return detach_lifecycle_record(capture, limits)
