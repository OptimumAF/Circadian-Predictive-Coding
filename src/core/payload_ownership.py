"""Metadata and reference ports for trusted retained payload ownership.

No payload copying, native calls, locking, deletion, byte accounting or graph
certification lives here. References are only usable under the outer owner lease.
"""

from dataclasses import dataclass
from typing import ContextManager, Literal, Protocol

from src.core.experience import require_tick

OwnershipKind = Literal["actor", "candidate", "checkpoint", "promotion"]


def require_ownership_kind(kind: OwnershipKind) -> None:
    if type(kind) is not str or kind not in ("actor", "candidate", "checkpoint", "promotion"):
        raise ValueError("unsupported payload holder kind")


@dataclass(frozen=True)
class PayloadOwnershipLimits:
    max_live_holders: int
    max_lifetime_enrollments: int

    def __post_init__(self) -> None:
        require_tick(self.max_live_holders, "max_live_holders")
        require_tick(self.max_lifetime_enrollments, "max_lifetime_enrollments")


@dataclass(frozen=True)
class PayloadReferences:
    models: tuple[object, ...] = ()
    inboxes: tuple[object, ...] = ()
    snapshots: tuple[object, ...] = ()
    auxiliary: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if any(
            type(items) is not tuple
            for items in (self.models, self.inboxes, self.snapshots, self.auxiliary)
        ):
            raise ValueError("payload reference collections must be immutable tuples")


class PayloadHolder(Protocol):
    _payload_ready: bool

    def _payload_exclusive(self) -> ContextManager[None]: ...
    def _payload_references(self) -> PayloadReferences: ...


@dataclass(frozen=True)
class PayloadHolderMetadata:
    enrollment: int
    kind: OwnershipKind
    ready: bool

    def __post_init__(self) -> None:
        require_tick(self.enrollment, "enrollment")
        require_ownership_kind(self.kind)
        if self.enrollment == 0 or type(self.ready) is not bool:
            raise ValueError("holder requires positive enrollment and exact ready flag")


@dataclass(frozen=True)
class PayloadOwnershipSnapshot:
    holders: tuple[PayloadHolderMetadata, ...]
    total_enrollments: int
    limits: PayloadOwnershipLimits | None

    def __post_init__(self) -> None:
        require_tick(self.total_enrollments, "total_enrollments")
        if type(self.holders) is not tuple or any(
            type(h) is not PayloadHolderMetadata for h in self.holders
        ):
            raise ValueError("ownership snapshot requires immutable holder metadata")
        numbers = [h.enrollment for h in self.holders]
        if numbers != sorted(set(numbers)) or any(n > self.total_enrollments for n in numbers):
            raise ValueError("ownership enrollments must be unique, ordered and consumed")
        if self.limits is not None:
            if type(self.limits) is not PayloadOwnershipLimits:
                raise ValueError("ownership snapshot requires typed limits")
            if (
                len(self.holders) > self.limits.max_live_holders
                or self.total_enrollments > self.limits.max_lifetime_enrollments
            ):
                raise ValueError("ownership snapshot exceeds its declared holder limits")


@dataclass(frozen=True)
class OwnedPayloadGroup:
    holder: PayloadHolderMetadata
    references: PayloadReferences
