"""Complete retained source records and inward capture ports, without ownership IO.

Records observe supported original state. Authority paths are references to live
ports/owners, never serialized authority or instructions to construct a new owner.
"""

from dataclasses import dataclass
from typing import Callable

from src.core.experience import require_tick
from src.core.managed_lifecycle_state import AuthorityReference, LifecycleCaptureLimits
from src.core.managed_record_state import ManagedRecordMetadata
from src.core.managed_lifecycle_validation import require_lifecycle_capture_limits


@dataclass(frozen=True)
class CompositeCaptureLimits:
    records: LifecycleCaptureLimits
    max_nodes: int
    max_depth: int
    max_array_bytes: int
    max_dimension: int

    def __post_init__(self) -> None:
        if type(self.records) is not LifecycleCaptureLimits:
            raise ValueError("composite requires original typed record limits")
        require_lifecycle_capture_limits(self.records)
        for name in ("max_nodes", "max_depth", "max_array_bytes", "max_dimension"):
            value = getattr(self, name)
            require_tick(value, name)
            if value == 0:
                raise ValueError("composite capture bounds must be positive")


@dataclass(frozen=True)
class AuthorityPath:
    path: str
    present: bool


@dataclass(frozen=True)
class SourceRecord:
    kind: str
    fields: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class NativeProjection:
    state: SourceRecord
    authority: tuple[AuthorityReference, ...]


@dataclass(frozen=True)
class ManagedCompositeState:
    format_version: int
    records: ManagedRecordMetadata
    holders: tuple[tuple[int, SourceRecord], ...]
    sharing: SourceRecord
    budget: SourceRecord
    clock: SourceRecord
    retained_payload_bytes: int


@dataclass(frozen=True, eq=False)
class ManagedCompositeCapture:
    state: ManagedCompositeState
    authority: tuple[AuthorityReference, ...]


NativeProjectionPort = Callable[[object, str, CompositeCaptureLimits], NativeProjection]
CompositePreflightPort = Callable[[ManagedCompositeState, CompositeCaptureLimits], None]
CompositeCopyPort = Callable[
    [ManagedCompositeState, CompositeCaptureLimits, dict[int, object]], ManagedCompositeState
]
