"""Bounded local replay-origin metadata and monotone admission accounting.

No payload storage, model operation, consent attestation, IO or restore authority.
Accounting charges exact canonical UTF-8 metadata plus 1024 units per fixed
record/invocation; these units are not a physical heap or process-RSS measurement.
"""

from dataclasses import asdict, dataclass, replace
import json
from hashlib import sha256
from typing import Callable

from src.core.experience import SampleKey, require_identifier, require_tick
from src.core.inbox_origin import InboxOriginData, inbox_origin_metadata
from src.core.untrained_inbox_origin import (
    UntrainedInboxOriginData,
    untrained_inbox_origin_metadata,
)

RECORD_OVERHEAD_BYTES = 1024


def replay_origin_metadata_digest(data, maximum: int) -> str:
    if type(data) is not ReplayOriginData:
        raise ValueError("row origin metadata type is corrupt")
    if type(data.key) is not tuple or len(data.key) != 2:
        raise ValueError("row origin metadata key is corrupt")
    if any(
        isinstance(value, str) and len(value) * 6 > maximum
        for value in (*vars(data).values(), *data.key)
    ):
        raise ValueError("row origin metadata string bound exceeded")
    data.__post_init__()
    encoded = json.dumps(
        asdict(data), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf8")
    if len(encoded) > maximum:
        raise ValueError("row origin metadata bound exceeded")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ReplayOriginPorts:
    model_reference: Callable[[object], object]
    retained: Callable[[object, int], tuple[object, ...]]
    copy_bytes: Callable[[object, object, int, int, int], int]
    payloads: Callable[[object], tuple[object, object]]
    fingerprint: Callable[[object, int], tuple[int, str]]

    def __post_init__(self) -> None:
        if any(not callable(port) for port in vars(self).values()):
            raise ValueError("replay origin ports must be callable")


@dataclass(frozen=True)
class ReplayOriginLimits:
    max_live_records: int
    max_records_created: int
    max_invocations: int
    max_metadata_bytes: int
    max_age_ticks: int
    max_payload_bytes: int

    def __post_init__(self) -> None:
        if any(type(n) is not int or not 0 < n < 2**63 for n in vars(self).values()):
            raise ValueError("replay origin limits require positive bounded integers")


@dataclass(frozen=True)
class ReplayOriginData:
    key: SampleKey
    event_id: str
    actor_version: str
    learner_version: str
    subject_id: str
    source_id: str
    row_start: int
    row_count: int
    observed_at: int
    arrived_at: int
    update_number: int
    payload_bytes: int
    payload_digest: str

    def __post_init__(self) -> None:
        if type(self.key) is not tuple or len(self.key) != 2:
            raise ValueError("row origin requires an original sample key")
        for value in (
            *self.key,
            self.event_id,
            self.actor_version,
            self.learner_version,
            self.subject_id,
            self.source_id,
        ):
            require_identifier(value, "row origin identifier")
        for name in [
            "row_start",
            "row_count",
            "observed_at",
            "arrived_at",
            "update_number",
            "payload_bytes",
        ]:
            require_tick(getattr(self, name), name)
        if self.row_count == 0 or self.update_number == 0 or self.payload_bytes == 0:
            raise ValueError("row origin counts/bytes must be positive")
        if (
            type(self.payload_digest) is not str
            or len(self.payload_digest) != 64
            or any(c not in "0123456789abcdef" for c in self.payload_digest)
        ):
            raise ValueError("row origin requires a SHA256 integrity digest")


@dataclass(frozen=True)
class ReplayOriginAccounting:
    records_created: int = 0
    invocations_started: int = 0
    metadata_bytes_charged: int = 0
    live_records: int = 0


class ReplayOriginAdmission:
    def __init__(self, limits: ReplayOriginLimits) -> None:
        if type(limits) is not ReplayOriginLimits:
            raise ValueError("replay origin admission requires exact limits")
        self.limits = limits
        self._original_limits = limits
        self._progress = ReplayOriginAccounting()
        self._minimum = (0, 0, 0)

    def accounting(self, live_records: int = 0) -> ReplayOriginAccounting:
        p, limit = self._progress, self.limits
        if limit is not self._original_limits:
            raise ValueError("replay origin limits were replaced or renewed")
        if type(p) is not ReplayOriginAccounting:
            raise ValueError("replay origin accounting is corrupt")
        values = (p.records_created, p.invocations_started, p.metadata_bytes_charged)
        ceilings = (limit.max_records_created, limit.max_invocations, limit.max_metadata_bytes)
        if type(p) is not ReplayOriginAccounting or any(
            type(n) is not int or not low <= n <= high
            for n, low, high in zip(values, self._minimum, ceilings)
        ):
            raise ValueError("replay origin accounting is corrupt or rewound")
        if type(live_records) is not int or not 0 <= live_records <= limit.max_live_records:
            raise ValueError("replay origin live record limit exceeded")
        return replace(p, live_records=live_records)

    def _charge(self, records: int, invocations: int, size: int, live: int) -> None:
        p = self.accounting(live)
        updated = (
            p.records_created + records,
            p.invocations_started + invocations,
            p.metadata_bytes_charged + size,
        )
        if any(
            n > cap
            for n, cap in zip(
                updated,
                (
                    self.limits.max_records_created,
                    self.limits.max_invocations,
                    self.limits.max_metadata_bytes,
                ),
            )
        ):
            raise ValueError("replay origin cumulative record/invocation/metadata limit exhausted")
        self._progress = ReplayOriginAccounting(*updated)
        self._minimum = updated

    def start(self, live: int) -> None:
        self._charge(0, 1, RECORD_OVERHEAD_BYTES, live)

    def reserve(self, data: ReplayOriginData, live: int) -> None:
        if type(data) is not ReplayOriginData or data.payload_bytes > self.limits.max_payload_bytes:
            raise ValueError("replay origin data or payload byte bound is invalid")
        remaining = self.limits.max_metadata_bytes - self.accounting(live).metadata_bytes_charged
        # Bound string rendering before JSON allocation; identifiers already live
        # in original records and are never used to infer payload identity.
        strings = (
            *data.key,
            data.event_id,
            data.actor_version,
            data.learner_version,
            data.subject_id,
            data.source_id,
        )
        if any(len(value) * 6 > remaining for value in strings):
            raise ValueError("replay origin identifier exceeds remaining metadata bound")
        encoded = json.dumps(
            asdict(data), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf8")
        self._charge(1, 0, len(encoded) + RECORD_OVERHEAD_BYTES, live + 1)

    def reserve_inbox(self, data: InboxOriginData, live: int) -> None:
        """Charge a separately observed pair under the same original admission."""
        remaining = (
            self.limits.max_metadata_bytes
            - self.accounting(live).metadata_bytes_charged
            - RECORD_OVERHEAD_BYTES
        )
        encoded = inbox_origin_metadata(data, remaining)
        if data.payload_bytes > self.limits.max_payload_bytes:
            raise ValueError("inbox origin payload byte bound is invalid")
        self._charge(1, 0, len(encoded) + RECORD_OVERHEAD_BYTES, live + 1)

    def reserve_untrained(self, data: UntrainedInboxOriginData, live: int) -> None:
        """Pay for observed untrained metadata without creating applied work."""
        remaining = (
            self.limits.max_metadata_bytes
            - self.accounting(live).metadata_bytes_charged
            - RECORD_OVERHEAD_BYTES
        )
        encoded = untrained_inbox_origin_metadata(data, remaining)
        self._charge(1, 0, len(encoded) + RECORD_OVERHEAD_BYTES, live + 1)
