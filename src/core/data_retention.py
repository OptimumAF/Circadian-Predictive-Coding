"""Declared local retention limits and payload-free cleanup observations.

No IO, copying, clock scheduling, native implementation or unlearning lives here.
"""

from dataclasses import dataclass
from math import isfinite

from src.core.data_erasure import ErasureReason
from src.core.experience import SampleKey, require_identifier, require_tick
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.payload_bytes import PayloadCopyLimits


@dataclass(frozen=True)
class DataRetentionPolicy:
    max_lifetime_ingress_bytes: int
    max_retention_ticks: int
    holders: PayloadOwnershipLimits
    max_checkpoint_pending: int = 4
    max_checkpoint_preparations: int = 4
    max_promotion_pending: int = 4
    owned_payload_copies: PayloadCopyLimits | None = None
    max_retention_seconds: float | None = None

    def __post_init__(self) -> None:
        require_tick(self.max_lifetime_ingress_bytes, "max_lifetime_ingress_bytes")
        for value in (
            self.max_retention_ticks,
            self.max_checkpoint_pending,
            self.max_checkpoint_preparations,
            self.max_promotion_pending,
        ):
            require_tick(value, "retention/copy capacity")
            if value == 0:
                raise ValueError("retention/copy capacities must be positive")
        if type(self.holders) is not PayloadOwnershipLimits:
            raise ValueError("retention policy requires typed holder limits")
        if (
            self.owned_payload_copies is not None
            and type(self.owned_payload_copies) is not PayloadCopyLimits
        ):
            raise ValueError("retention policy requires typed owned payload copy limits")
        if self.max_retention_seconds is not None and (
            type(self.max_retention_seconds) not in (int, float)
            or not isfinite(self.max_retention_seconds)
            or self.max_retention_seconds <= 0
        ):
            raise ValueError("elapsed retention requires positive finite seconds")


@dataclass(frozen=True)
class DataCleanupReport:
    reason: ErasureReason
    requested_keys: tuple[SampleKey, ...]
    revoked_keys: tuple[SampleKey, ...]
    model_snapshots_erased: int
    inboxes_cleared: int
    checkpoints_invalidated: int
    promotions_invalidated: int

    def __post_init__(self) -> None:
        if type(self.reason) is not str or self.reason not in ("deleted", "expired", "opt_out"):
            raise ValueError("unsupported cleanup reason")
        for keys in (self.requested_keys, self.revoked_keys):
            if type(keys) is not tuple:
                raise ValueError("cleanup identities must be ordered immutable unique keys")
            for key in keys:
                if type(key) is not tuple or len(key) != 2:
                    raise ValueError("cleanup requires episode/sample identities")
                for value in key:
                    require_identifier(value, "cleanup identity")
            if keys != tuple(sorted(set(keys))):
                raise ValueError("cleanup identities must be ordered immutable unique keys")
        for value in (
            self.model_snapshots_erased,
            self.inboxes_cleared,
            self.checkpoints_invalidated,
            self.promotions_invalidated,
        ):
            require_tick(value, "cleanup count")
