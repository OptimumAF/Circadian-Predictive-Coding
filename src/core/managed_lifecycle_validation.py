"""Strict complete lifecycle state validation without invoking live authority.

Checks independently supplied bounds, complete immutable native record fields,
cross-record policies/counters/epochs and original reference identities. Matching
references do not prove live-owner provenance or an atomic capture.
"""

from dataclasses import fields
from math import isfinite

from src.core.data_lifecycle import (
    DataConsent,
    DataProvenance,
    LifecycleDeclaration,
    LifecycleLimits,
)
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import require_identifier
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits, require_ownership_kind
from src.core.retention_driver import RetentionDriverLimits
from src.core.managed_lifecycle_state import (
    AUTHORITY_PATHS,
    DRIVER_REFERENCES,
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


def require_state_record(value, kind) -> None:
    if type(value) is not kind or vars(value).keys() != {field.name for field in fields(kind)}:
        raise ValueError("lifecycle state differs from complete typed record schema")


def _counter(value) -> None:
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("lifecycle counter requires a bounded nonnegative exact integer")


def _seconds(value) -> None:
    if type(value) not in (int, float) or not isfinite(value) or value < 0:
        raise ValueError("lifecycle epoch requires nonnegative finite original seconds")


def _flag(value) -> None:
    if type(value) is not bool:
        raise ValueError("lifecycle state flag requires exact boolean")


def _string(value, limits, *, identifier=True) -> None:
    if (
        type(value) is not str
        or not value
        or len(value) > limits.max_identifier_bytes
        or len(value.encode("utf8")) > limits.max_identifier_bytes
    ):
        raise ValueError("lifecycle string exceeds original UTF8 bound")
    if identifier:
        require_identifier(value, "lifecycle identity")


def _key(value, limits) -> None:
    if type(value) is not tuple or len(value) != 2:
        raise ValueError("lifecycle record requires immutable episode/sample key")
    for item in value:
        _string(item, limits)


def _policy(policy, kind) -> None:
    require_state_record(policy, kind)
    kind.__post_init__(policy)
    for value in vars(policy).values():
        if type(value) is int:
            _counter(value)


def require_lifecycle_capture_limits(limits) -> None:
    require_state_record(limits, LifecycleCaptureLimits)
    _counter(limits.max_records)
    _counter(limits.max_identifier_bytes)
    if limits.max_identifier_bytes == 0:
        raise ValueError("lifecycle string capacity must be positive")


def _collections(metadata, limits) -> None:
    owner = metadata.owner
    items = (
        owner.catalog,
        owner.opted_out,
        owner.revoked_keys,
        owner.declaration_ticks,
        owner.declaration_seconds,
        metadata.registry.holders,
    )
    if any(type(value) is not tuple for value in items):
        raise ValueError("lifecycle histories require immutable complete tuples")
    if sum(len(value) for value in items) > limits.max_records:
        raise ValueError("aggregate lifecycle metadata exceeds original record bound")


def _catalog(owner, limits):
    _policy(owner.limits, LifecycleLimits)
    keys, subjects = set(), set()
    if len(owner.catalog) > min(owner.limits.max_approved_records, owner.limits.max_replay_records):
        raise ValueError("lifecycle catalog exceeds original lifetime quota")
    for item in owner.catalog:
        require_state_record(item, LifecycleDeclaration)
        require_state_record(item.provenance, DataProvenance)
        require_state_record(item.consent, DataConsent)
        _key(item.key, limits)
        for value in (item.provenance.source_id, item.provenance.subject_id):
            _string(value, limits)
        DataProvenance.__post_init__(item.provenance)
        DataConsent.__post_init__(item.consent)
        LifecycleDeclaration.__post_init__(item)
        owner.limits.require_supported(item)
        if item.key in keys:
            raise ValueError("duplicate lifecycle declaration identity")
        keys.add(item.key)
        subjects.add(item.provenance.subject_id)
    for value in owner.opted_out:
        _string(value, limits)
    for value in owner.revoked_keys:
        _key(value, limits)
    if (
        owner.opted_out != tuple(sorted(set(owner.opted_out)))
        or not set(owner.opted_out) <= subjects
        or owner.revoked_keys != tuple(sorted(set(owner.revoked_keys)))
        or not set(owner.revoked_keys) <= keys
    ):
        raise ValueError("lifecycle opt-out/revocation identities differ from original catalog")
    return keys


def _anchors(entries, keys, limits, *, seconds):
    result = {}
    for entry in entries:
        if type(entry) is not tuple or len(entry) != 2:
            raise ValueError("lifecycle declaration anchor requires immutable key/value pair")
        key, value = entry
        _key(key, limits)
        (_seconds if seconds else _counter)(value)
        if key in result:
            raise ValueError("duplicate lifecycle declaration anchor")
        result[key] = value
    if result.keys() != keys:
        raise ValueError("lifecycle declaration anchors omit or invent original catalog keys")
    return result


def _accounting(metadata, limits, keys) -> None:
    state = metadata.lifecycle
    _policy(state.policy, DataRetentionPolicy)
    _policy(state.policy.holders, PayloadOwnershipLimits)
    if state.policy.owned_payload_copies is not None:
        _policy(state.policy.owned_payload_copies, PayloadCopyLimits)
    for value in (state.admitted_bytes, state.last_tick):
        _counter(value)
    for value in (state.failed, state.retention_fault):
        _flag(value)
    if state.admitted_bytes > state.policy.max_lifetime_ingress_bytes:
        raise ValueError("lifecycle consumed ingress exceeds original allowance")
    ticks = _anchors(metadata.owner.declaration_ticks, keys, limits, seconds=False)
    if any(value > state.last_tick for value in ticks.values()):
        raise ValueError("lifecycle declaration tick is ahead of original observation")
    _elapsed_accounting(metadata, limits, keys)


def _elapsed_accounting(metadata, limits, keys) -> None:
    state = metadata.lifecycle
    if state.policy.max_retention_seconds is None:
        if (
            metadata.owner.declaration_seconds
            or state.last_seconds is not None
            or state.auxiliary_started_at is not None
        ):
            raise ValueError("unconfigured elapsed policy cannot invent clock epochs")
    else:
        if state.last_seconds is None:
            raise ValueError("configured elapsed policy requires its original last seconds")
        _seconds(state.last_seconds)
        seconds = _anchors(metadata.owner.declaration_seconds, keys, limits, seconds=True)
        if any(value > state.last_seconds for value in seconds.values()):
            raise ValueError("lifecycle declaration seconds are ahead of original observation")
        if state.auxiliary_started_at is not None:
            _seconds(state.auxiliary_started_at)
            if state.auxiliary_started_at > state.last_seconds:
                raise ValueError("auxiliary epoch is ahead of original observation")


def _ownership(metadata) -> None:
    state = metadata.registry
    _policy(state.limits, PayloadOwnershipLimits)
    _counter(state.total_enrollments)
    previous, live = 0, 0
    for item in state.holders:
        require_state_record(item, OwnershipEnrollment)
        _counter(item.enrollment)
        require_ownership_kind(item.kind)
        if not previous < item.enrollment <= state.total_enrollments:
            raise ValueError("retained ownership enrollment order/consumed numbering differs")
        if item.ready is not None:
            _flag(item.ready)
            live += 1
        previous = item.enrollment
    if (
        live > state.limits.max_live_holders
        or state.total_enrollments > state.limits.max_lifetime_enrollments
        or state.limits is not metadata.lifecycle.policy.holders
    ):
        raise ValueError("ownership differs from original shared holder policy/allowance")
    _copy_accounting(metadata)


def _copy_accounting(metadata) -> None:
    original = metadata.lifecycle.policy.owned_payload_copies
    if original is None:
        if metadata.copy is not None:
            raise ValueError("unconfigured copy policy cannot invent a copy budget")
    else:
        require_state_record(metadata.copy, CopyBudgetState)
        _policy(metadata.copy.limits, PayloadCopyLimits)
        _counter(metadata.copy.charged_bytes)
        if (
            metadata.copy.limits is not original
            or not metadata.lifecycle.admitted_bytes
            <= metadata.copy.charged_bytes
            <= original.max_lifetime_owned_bytes
        ):
            raise ValueError("copy charges or original shared policy differ")


def _driver(metadata, limits) -> None:
    state = metadata.driver
    if state is None:
        return
    require_state_record(state, RetentionDriverRecord)
    _policy(state.limits, RetentionDriverLimits)
    if metadata.lifecycle.policy.max_retention_seconds is None:
        raise ValueError("retention driver requires original elapsed policy")
    if type(state.state) is not str or state.state not in (
        "ready",
        "running",
        "stopping",
        "stopped",
        "exhausted",
        "failed",
    ):
        raise ValueError("unsupported original retention driver state")
    for value in (state.polls, state.purges, state.cleanup_attempts):
        _counter(value)
    _seconds(state.created_at)
    for value in (
        state.held,
        state.pending,
        state.stop_set,
        state.wake_set,
        state.purged_set,
        state.thread_present,
        state.thread_alive,
    ):
        _flag(value)
    if (
        state.polls > state.limits.max_polls
        or state.cleanup_attempts > state.limits.max_polls + 2
        or state.purges > state.cleanup_attempts
        or (state.held and not state.pending)
        or (state.thread_alive and not state.thread_present)
    ):
        raise ValueError("retention driver counters/held/thread observations are inconsistent")
    if state.error_type is not None:
        _string(state.error_type, limits)


def validate_lifecycle_metadata(
    metadata: LifecycleMetadata, limits: LifecycleCaptureLimits
) -> None:
    try:
        require_lifecycle_capture_limits(limits)
        require_state_record(metadata, LifecycleMetadata)
        if type(metadata.format_version) is not int or metadata.format_version != 1:
            raise ValueError("unsupported complete lifecycle metadata version")
        for value, kind in (
            (metadata.owner, ManagedOwnerState),
            (metadata.lifecycle, LifecycleAccountingState),
            (metadata.registry, OwnershipRegistryState),
        ):
            require_state_record(value, kind)
        _collections(metadata, limits)
        keys = _catalog(metadata.owner, limits)
        _accounting(metadata, limits, keys)
        _ownership(metadata)
        _driver(metadata, limits)
    except (
        TypeError,
        AttributeError,
        KeyError,
        UnicodeError,
        OverflowError,
        RecursionError,
    ) as error:
        raise ValueError("unsupported or corrupt complete lifecycle metadata") from error


def _references(capture):
    if type(capture.authority) is not tuple or len(capture.authority) != len(AUTHORITY_PATHS):
        raise ValueError("complete original lifecycle authority references are required")
    result = {}
    for item in capture.authority:
        require_state_record(item, AuthorityReference)
        if type(item.path) is not str or item.path not in AUTHORITY_PATHS or item.path in result:
            raise ValueError("unknown or duplicate original authority reference")
        result[item.path] = item.value
    return result


def _identity(references, left, right) -> None:
    if references[left] is not references[right]:
        raise ValueError("original lifecycle authority relationship changed")


def _root_authority(refs) -> None:
    for name in ("owner", "lifecycle", "registry"):
        if refs["root." + name] is None:
            raise ValueError("configured lifecycle requires original nonempty roots")
    for left, right in (
        ("owner._lifecycle", "root.lifecycle"),
        ("registry._lifecycle", "root.lifecycle"),
        ("lifecycle._owner", "root.owner"),
        ("lifecycle._registry", "root.registry"),
        ("lifecycle._copy_budget", "root.copy"),
        ("lifecycle._retention_driver", "root.driver"),
        ("owner._shared", "lifecycle._shared"),
    ):
        _identity(refs, left, right)
    if refs["owner._issuing"] is not None:
        raise ValueError("quiescent capture cannot retain an in-flight managed issue")
    for path in (
        "owner._gate",
        "owner._shared",
        "lifecycle._actor",
        "lifecycle._budget",
        "lifecycle._budget_policy",
        "lifecycle._clock",
        "lifecycle._lineage",
        "lifecycle._shared",
        "lifecycle._sharing",
        "lifecycle._sharing_limits",
        "lifecycle._time_gate",
        "registry._gate",
        "registry._holders",
        "sharing._gate",
    ):
        if refs[path] is None:
            raise ValueError("required original live lifecycle authority is missing")


def _copy_authority(metadata, refs) -> None:
    if (metadata.copy is None) != (refs["root.copy"] is None):
        raise ValueError("original copy authority presence differs from metadata")
    if metadata.copy is None and refs["copy._gate"] is not None:
        raise ValueError("unconfigured copy budget cannot invent a lock")
    if metadata.copy is not None and refs["copy._gate"] is None:
        raise ValueError("configured copy budget requires original lock")


def _driver_authority(metadata, refs) -> None:
    if (metadata.driver is None) != (refs["root.driver"] is None):
        raise ValueError("original driver authority presence differs from metadata")
    if metadata.driver is None:
        if any(refs["driver." + name] is not None for name in DRIVER_REFERENCES):
            raise ValueError("unconfigured driver cannot invent live authority")
    else:
        _identity(refs, "driver._lifecycle", "root.lifecycle")
        for name in ("_operation_gate", "_state_gate", "_token", "_stop", "_wake", "_purged"):
            if refs["driver." + name] is None:
                raise ValueError("configured driver requires original token,events and lock")
        if (refs["driver._thread"] is not None) != metadata.driver.thread_present:
            raise ValueError("original driver thread reference differs from observation")
        if metadata.driver.held:
            _identity(refs, "driver._token", "sharing._retention_hold")


def _port_authority(metadata, refs) -> None:
    for name in ("_measure", "_footprint", "_erase", "_wall_clock", "_resource"):
        if not callable(refs["lifecycle." + name]):
            raise ValueError("original lifecycle port is missing")
    if metadata.lifecycle.policy.max_retention_seconds is not None and not callable(
        refs["lifecycle._auxiliary_bytes"]
    ):
        raise ValueError("elapsed retention requires original auxiliary measurement port")
    if metadata.copy is not None:
        for name in (
            "_auxiliary_bytes",
            "_checkpoint_bytes",
            "_growth_bytes",
            "_prepare_bytes",
            "_prediction_bytes",
        ):
            if not callable(refs["lifecycle." + name]):
                raise ValueError("original configured copy measurement port is missing")


def validate_lifecycle_capture(
    capture: ManagedLifecycleCapture, limits: LifecycleCaptureLimits
) -> None:
    require_state_record(capture, ManagedLifecycleCapture)
    validate_lifecycle_metadata(capture.metadata, limits)
    refs = _references(capture)
    _root_authority(refs)
    _copy_authority(capture.metadata, refs)
    _driver_authority(capture.metadata, refs)
    _port_authority(capture.metadata, refs)
