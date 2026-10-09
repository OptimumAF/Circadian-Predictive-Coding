"""Independently supplied complete lifecycle codec policy; no wire authority.

Inputs are original owner/retention/driver policies and independent capture bounds.
Validation uses the existing complete native rules. No live port, IO or renewal.
"""

from dataclasses import dataclass
from src.core.data_lifecycle import LifecycleLimits
from src.core.data_retention import DataRetentionPolicy
from src.core.retention_driver import RetentionDriverLimits
from src.core.managed_lifecycle_state import (
    CopyBudgetState,
    LifecycleAccountingState,
    LifecycleCaptureLimits,
    LifecycleMetadata,
    ManagedOwnerState,
    OwnershipRegistryState,
    RetentionDriverRecord,
)
from src.core.managed_lifecycle_validation import (
    require_state_record,
    validate_lifecycle_metadata,
)


@dataclass(frozen=True)
class LifecycleCodecPolicy:
    owner_limits: LifecycleLimits
    lifecycle_policy: DataRetentionPolicy
    driver_limits: RetentionDriverLimits | None
    capture_limits: LifecycleCaptureLimits

    def __post_init__(self) -> None:
        require_state_record(self, LifecycleCodecPolicy)
        require_state_record(self.lifecycle_policy, DataRetentionPolicy)
        policy = self.lifecycle_policy
        # Validate known original policies, not records materialized from wire data.
        template = LifecycleMetadata(
            1,
            ManagedOwnerState(self.owner_limits, (), (), (), (), ()),
            LifecycleAccountingState(
                policy,
                0,
                0,
                False,
                0.0 if policy.max_retention_seconds is not None else None,
                None,
                False,
            ),
            OwnershipRegistryState(policy.holders, 0, ()),
            None
            if policy.owned_payload_copies is None
            else CopyBudgetState(policy.owned_payload_copies, 0),
            None
            if self.driver_limits is None
            else RetentionDriverRecord(
                self.driver_limits,
                "ready",
                0,
                0,
                0,
                0.0,
                False,
                False,
                None,
                False,
                False,
                False,
                False,
                False,
            ),
        )
        validate_lifecycle_metadata(template, self.capture_limits)
