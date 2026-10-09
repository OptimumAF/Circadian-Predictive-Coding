"""Immutable byte allowances and payload-free retained-copy observations.

Bytes describe supported retained payload arrays, not parameters, Python overhead
or process memory. No payload inspection, locking, copying or IO lives here.
"""

from dataclasses import dataclass
from src.core.experience import require_tick


@dataclass(frozen=True)
class PayloadCopyLimits:
    max_lifetime_owned_bytes: int

    def __post_init__(self) -> None:
        require_tick(self.max_lifetime_owned_bytes, "max_lifetime_owned_bytes")


@dataclass(frozen=True)
class PayloadByteSnapshot:
    limits: PayloadCopyLimits
    charged_bytes: int
    observed_retained_bytes: int

    def __post_init__(self) -> None:
        if type(self.limits) is not PayloadCopyLimits:
            raise ValueError("payload byte snapshot requires typed limits")
        require_tick(self.charged_bytes, "charged_bytes")
        require_tick(self.observed_retained_bytes, "observed_retained_bytes")
        if (
            not self.observed_retained_bytes
            <= self.charged_bytes
            <= self.limits.max_lifetime_owned_bytes
        ):
            raise ValueError("retained payload bytes exceed the original charged allowance")
