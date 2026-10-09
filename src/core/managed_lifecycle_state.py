"""Complete lifecycle records and retained original authority references.

Metadata may be detached as one immutable graph. Authority values remain the
original objects; they are not portable, copyable or restore authorization.
Validation lives in managed_lifecycle_validation. No clocks, locks or IO here.
"""

from dataclasses import dataclass
from typing import Literal

from src.core.data_lifecycle import LifecycleDeclaration, LifecycleLimits
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import SampleKey
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import OwnershipKind, PayloadOwnershipLimits
from src.core.retention_driver import RetentionDriverLimits, RetentionDriverState

OWNER_REFERENCES = ("_shared", "_issuing", "_gate", "_lifecycle")
LIFECYCLE_REFERENCES = (
    "_actor",
    "_auxiliary_bytes",
    "_budget",
    "_budget_policy",
    "_checkpoint_bytes",
    "_clock",
    "_copy_budget",
    "_erase",
    "_footprint",
    "_growth_bytes",
    "_lineage",
    "_measure",
    "_owner",
    "_prediction_bytes",
    "_prepare_bytes",
    "_progress",
    "_registry",
    "_resource",
    "_retention_driver",
    "_sampler",
    "_shared",
    "_sharing",
    "_sharing_limits",
    "_time_gate",
    "_wall_clock",
)
DRIVER_REFERENCES = (
    "_lifecycle",
    "_operation_gate",
    "_state_gate",
    "_token",
    "_stop",
    "_wake",
    "_purged",
    "_thread",
)
AUTHORITY_PATHS = frozenset(
    ["root." + name for name in ("owner", "lifecycle", "driver", "registry", "copy")]
    + ["owner." + name for name in OWNER_REFERENCES]
    + ["lifecycle." + name for name in LIFECYCLE_REFERENCES]
    + ["driver." + name for name in DRIVER_REFERENCES]
    + [
        "registry._gate",
        "registry._lifecycle",
        "registry._holders",
        "copy._gate",
        "sharing._retention_hold",
        "sharing._gate",
    ]
)


@dataclass(frozen=True)
class LifecycleCaptureLimits:
    max_records: int
    max_identifier_bytes: int


@dataclass(frozen=True)
class ManagedOwnerState:
    limits: LifecycleLimits
    catalog: tuple[LifecycleDeclaration, ...]
    opted_out: tuple[str, ...]
    revoked_keys: tuple[SampleKey, ...]
    declaration_ticks: tuple[tuple[SampleKey, int], ...]
    declaration_seconds: tuple[tuple[SampleKey, float], ...]


@dataclass(frozen=True)
class LifecycleAccountingState:
    policy: DataRetentionPolicy
    admitted_bytes: int
    last_tick: int
    failed: bool
    last_seconds: float | None
    auxiliary_started_at: float | None
    retention_fault: bool


@dataclass(frozen=True)
class RetentionDriverRecord:
    limits: RetentionDriverLimits
    state: RetentionDriverState
    polls: int
    purges: int
    cleanup_attempts: int
    created_at: float
    held: bool
    pending: bool
    error_type: str | None
    stop_set: bool
    wake_set: bool
    purged_set: bool
    thread_present: bool
    thread_alive: bool


@dataclass(frozen=True)
class OwnershipEnrollment:
    enrollment: int
    kind: OwnershipKind
    ready: bool | None  # None means the actual retained weak reference is dead.


@dataclass(frozen=True)
class OwnershipRegistryState:
    limits: PayloadOwnershipLimits
    total_enrollments: int
    holders: tuple[OwnershipEnrollment, ...]


@dataclass(frozen=True)
class CopyBudgetState:
    limits: PayloadCopyLimits
    charged_bytes: int


@dataclass(frozen=True)
class LifecycleMetadata:
    format_version: Literal[1]
    owner: ManagedOwnerState
    lifecycle: LifecycleAccountingState
    registry: OwnershipRegistryState
    copy: CopyBudgetState | None
    driver: RetentionDriverRecord | None


@dataclass(frozen=True, eq=False)
class AuthorityReference:
    path: str
    value: object


@dataclass(frozen=True, eq=False)
class ManagedLifecycleCapture:
    metadata: LifecycleMetadata
    authority: tuple[AuthorityReference, ...]
