"""Payload-free consumed identities and explicit owned replay erasure counts.

No payload access, IO, ownership coordination, time policy or unlearning lives here.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

from src.core.experience import SampleKey, require_identifier, require_tick

ErasureReason = Literal["deleted", "expired", "opt_out"]


@dataclass(frozen=True)
class ErasedExperience:
    key: SampleKey
    actor_version: str
    observed_at: int | None
    event_id: str | None
    arrived_at: int | None
    erased_at: int
    reason: ErasureReason

    def __post_init__(self) -> None:
        if type(self.key) is not tuple or len(self.key) != 2:
            raise ValueError("erased identity requires an episode/sample tuple")
        for value in self.key:
            require_identifier(value, "erased identity")
        require_identifier(self.actor_version, "erased actor_version")
        require_tick(self.erased_at, "erased_at")
        for name in ("observed_at", "arrived_at"):
            value = getattr(self, name)
            if value is not None:
                require_tick(value, name)
        if self.observed_at is None and self.arrived_at is None:
            raise ValueError("erased identity requires source or label metadata")
        if (self.event_id is None) != (self.arrived_at is None):
            raise ValueError("erased label requires both event identity and arrival time")
        if self.event_id is not None:
            require_identifier(self.event_id, "erased event_id")
        if (
            self.observed_at is not None
            and self.arrived_at is not None
            and self.arrived_at < self.observed_at
        ):
            raise ValueError("erased label cannot arrive before observation")
        if type(self.reason) is not str or self.reason not in ("deleted", "expired", "opt_out"):
            raise ValueError("unknown erasure reason")


@dataclass(frozen=True)
class ReplayPayloadErasure:
    snapshots: int
    examples: int
    payload_bytes: int

    def __post_init__(self) -> None:
        for value in (self.snapshots, self.examples, self.payload_bytes):
            require_tick(value, "replay erasure count")
        if self.snapshots == 0:
            if self.examples or self.payload_bytes:
                raise ValueError("empty erasure requires zero examples and bytes")
        elif self.examples < self.snapshots or self.payload_bytes == 0:
            raise ValueError("erased snapshots require nonempty examples and payloads")


class ReplayPayloadOwner(Protocol):
    def replay_payload_footprint(self) -> ReplayPayloadErasure:
        """Observe current owned replay array counts without copying payloads."""
        ...

    def erase_replay_payloads(self) -> ReplayPayloadErasure:
        """Drop owned replay payload references without resetting learned state."""
        ...
