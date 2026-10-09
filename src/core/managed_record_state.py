"""Complete coupled records and original references, without source provenance.

Inputs are bounded immutable consolidation/lifecycle records and runtime scalars.
Validation checks their relationships without reading live objects or invoking
ports. Coherent live capture belongs to app; bytes and restore do not belong here.
"""

from dataclasses import dataclass

from src.core.actor_ports import AppliedConsolidation
from src.core.consolidation_cursor import ConsolidationCursor, require_consolidation_counter
from src.core.experience import require_identifier
from src.core.learner_ports import TrainingDiagnostic
from src.core.managed_lifecycle_state import (
    AUTHORITY_PATHS,
    AuthorityReference,
    LifecycleCaptureLimits,
    LifecycleMetadata,
    ManagedOwnerState,
    ManagedLifecycleCapture,
    OwnershipRegistryState,
)
from src.core.managed_lifecycle_validation import (
    require_lifecycle_capture_limits,
    require_state_record,
    validate_lifecycle_capture,
)

RUNTIME_REFERENCE_PATHS = frozenset(
    {
        "runtime.root",
        "runtime.actor",
        "runtime.candidate",
        "runtime.inbox",
        "runtime.budget",
        "runtime.lineage",
        "runtime.gate",
        "inbox.learner",
        "inbox.clock",
        "inbox.budget",
        "shared.runtime",
    }
)
RECORD_AUTHORITY_PATHS = AUTHORITY_PATHS | RUNTIME_REFERENCE_PATHS


@dataclass(frozen=True)
class RuntimeRecordObservation:
    base_actor_version: str
    serving_actor_version: str
    learner_version: str
    revision: int
    consolidation_limit: int
    stopped: bool
    retired: bool
    payload_ready: bool
    budget_updates: int
    inbox_completed_updates: int
    inbox_last_tick: int
    inbox_stopped: bool
    enrollment: int


@dataclass(frozen=True)
class ManagedRecordMetadata:
    format_version: int
    owner: RuntimeRecordObservation
    consolidation: ConsolidationCursor
    lifecycle: LifecycleMetadata


@dataclass(frozen=True, eq=False)
class ManagedRecordCapture:
    metadata: ManagedRecordMetadata
    authority: tuple[AuthorityReference, ...]


def _string(value, limits) -> None:
    if (
        type(value) is not str
        or not value
        or len(value) > limits.max_identifier_bytes
        or len(value.encode("utf8")) > limits.max_identifier_bytes
    ):
        raise ValueError("paired record identifier exceeds original UTF8 capacity")
    require_identifier(value, "paired record identity")


def _observation(owner, limits) -> None:
    require_state_record(owner, RuntimeRecordObservation)
    for value in (owner.base_actor_version, owner.serving_actor_version, owner.learner_version):
        _string(value, limits)
    for value in (
        owner.revision,
        owner.consolidation_limit,
        owner.budget_updates,
        owner.inbox_completed_updates,
        owner.enrollment,
    ):
        require_consolidation_counter(value)
    if type(owner.inbox_last_tick) is not int or not -1 <= owner.inbox_last_tick < 2**63:
        raise ValueError("paired inbox observation requires a bounded original tick")
    if any(
        type(value) is not bool
        for value in (
            owner.stopped,
            owner.retired,
            owner.payload_ready,
            owner.inbox_stopped,
        )
    ):
        raise ValueError("paired runtime flags require exact booleans")
    if owner.enrollment == 0 or owner.budget_updates < owner.inbox_completed_updates:
        raise ValueError("paired enrollment or consumed budget differs")


def _aggregate(metadata, limits) -> None:
    life, cursor = metadata.lifecycle, metadata.consolidation
    require_state_record(life, LifecycleMetadata)
    require_state_record(life.owner, ManagedOwnerState)
    require_state_record(life.registry, OwnershipRegistryState)
    require_state_record(cursor, ConsolidationCursor)
    collections = [
        life.owner.catalog,
        life.owner.opted_out,
        life.owner.revoked_keys,
        life.owner.declaration_ticks,
        life.owner.declaration_seconds,
        life.registry.holders,
        cursor.attempted_ids,
        cursor.consolidations,
    ]
    if any(type(value) is not tuple for value in collections):
        raise ValueError("paired records require complete immutable histories")
    if 1 + sum(map(len, collections)) > limits.max_records:
        raise ValueError("paired aggregate history exceeds original capture capacity")
    for value in (cursor.actor_version, cursor.learner_version, *cursor.attempted_ids):
        _string(value, limits)
    for receipt in cursor.consolidations:
        require_state_record(receipt, AppliedConsolidation)
        require_state_record(receipt.diagnostic, TrainingDiagnostic)
        _string(receipt.event_id, limits)
        _string(receipt.actor_version, limits)
        _string(receipt.learner_version, limits)
        _string(receipt.diagnostic.definition, limits)
    ConsolidationCursor.__post_init__(cursor)


def _relationships(metadata) -> None:
    owner, cursor, life = metadata.owner, metadata.consolidation, metadata.lifecycle
    if (
        owner.base_actor_version != cursor.actor_version
        or owner.learner_version != cursor.learner_version
        or any(
            getattr(owner, name) != getattr(cursor, name)
            for name in (
                "revision",
                "consolidation_limit",
                "stopped",
                "retired",
                "payload_ready",
            )
        )
    ):
        raise ValueError("paired original runtime and consolidation observations differ")
    enrollment = [item for item in life.registry.holders if item.enrollment == owner.enrollment]
    if (
        len(enrollment) != 1
        or enrollment[0].kind != "candidate"
        or enrollment[0].ready is not owner.payload_ready
        or ((life.lifecycle.failed or owner.inbox_stopped) and not cursor.stopped)
    ):
        raise ValueError("paired holder or uncertain-stop observations differ")
    # Serving may have advanced since the candidate base. Independent observed
    # inbox/lifecycle ticks may lag each other; matching clock authority is below.


def _references(capture):
    if type(capture.authority) is not tuple or len(capture.authority) != len(
        RECORD_AUTHORITY_PATHS
    ):
        raise ValueError("paired capture requires every original reference slot")
    refs = {}
    for item in capture.authority:
        require_state_record(item, AuthorityReference)
        if (
            type(item.path) is not str
            or item.path not in RECORD_AUTHORITY_PATHS
            or item.path in refs
        ):
            raise ValueError("unsupported or duplicate paired authority reference")
        refs[item.path] = item.value
    if any(refs[path] is None for path in RUNTIME_REFERENCE_PATHS):
        raise ValueError("paired runtime requires original nonempty authority")
    for left, right in (
        ("runtime.actor", "lifecycle._actor"),
        ("runtime.budget", "lifecycle._budget"),
        ("runtime.lineage", "lifecycle._lineage"),
        ("inbox.clock", "lifecycle._clock"),
        ("inbox.budget", "lifecycle._budget"),
        ("inbox.learner", "runtime.candidate"),
        ("shared.runtime", "runtime.root"),
    ):
        if refs[left] is not refs[right]:
            raise ValueError("paired original authority relationship changed")
    return tuple(item for item in capture.authority if item.path in AUTHORITY_PATHS)


def validate_managed_record_capture(
    capture: ManagedRecordCapture, limits: LifecycleCaptureLimits
) -> None:
    """Validate full records before detachment; never probe live reference values.

    Structural agreement alone does not prove a live capture interval. Retain the
    original capture independently before future byte decoding or owner handoff.
    """
    require_lifecycle_capture_limits(limits)
    require_state_record(capture, ManagedRecordCapture)
    metadata = capture.metadata
    require_state_record(metadata, ManagedRecordMetadata)
    if type(metadata.format_version) is not int or metadata.format_version != 1:
        raise ValueError("unsupported paired record version")
    _observation(metadata.owner, limits)
    try:
        _aggregate(metadata, limits)
    except (AttributeError, TypeError, UnicodeError) as error:
        raise ValueError("unsupported complete paired history") from error
    validate_lifecycle_capture(
        ManagedLifecycleCapture(metadata.lifecycle, _references(capture)), limits
    )
    _relationships(metadata)
