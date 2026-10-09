"""Bounded scalar metadata for an originally observed untrained erasure.

The work counter is a cumulative boundary, never an applied receipt. Validation
grants no arrival provenance, consent, enrollment or replay authority.
"""

from dataclasses import asdict, dataclass
import json

from src.core.data_erasure import ErasureReason
from src.core.experience import require_identifier


_FIELDS = (
    "key",
    "actor_version",
    "learner_version",
    "subject_id",
    "source_id",
    "observed_at",
    "event_id",
    "arrived_at",
    "erased_at",
    "reason",
    "completed_updates",
)


@dataclass(frozen=True)
class UntrainedInboxOriginData:
    key: tuple[str, str]
    actor_version: str
    learner_version: str
    subject_id: str
    source_id: str
    observed_at: int | None
    event_id: str | None
    arrived_at: int | None
    erased_at: int
    reason: ErasureReason
    completed_updates: int

    def __post_init__(self) -> None:
        _require_data(self, 128 * 1024)


def _require_data(data: UntrainedInboxOriginData, maximum: int) -> None:
    if type(data) is not UntrainedInboxOriginData:
        raise ValueError("untrained inbox origin requires exact scalar data")
    values = vars(data)
    if type(values) is not dict or len(values) != len(_FIELDS):
        raise ValueError("untrained inbox origin metadata field count changed")
    if any(type(key) is not str or len(key) > 64 for key in values):
        raise ValueError("untrained inbox origin field keys require bounded exact strings")
    if values.keys() != set(_FIELDS):
        raise ValueError("untrained inbox origin metadata schema changed")
    if type(data.key) is not tuple or len(data.key) != 2:
        raise ValueError("untrained inbox origin requires an exact paired key")
    strings: tuple[str, ...] = (
        *data.key,
        data.actor_version,
        data.learner_version,
        data.subject_id,
        data.source_id,
        data.reason,
    )
    if data.event_id is not None:
        strings += (data.event_id,)
    if any(type(value) is not str for value in strings):
        raise ValueError("untrained inbox origin identities require exact strings")
    # Bound escaped aggregate text before normalization or JSON allocation.
    if sum(len(value) * 6 for value in strings) > maximum:
        raise ValueError("untrained inbox origin identities exceed metadata capacity")
    for value in strings:
        require_identifier(value, "untrained inbox origin identity")
    for counter in (data.erased_at, data.completed_updates):
        if type(counter) is not int or not 0 <= counter < 2**63:
            raise ValueError("untrained inbox origin counters require bounded exact integers")
    for optional_counter in (data.observed_at, data.arrived_at):
        if optional_counter is not None:
            if type(optional_counter) is not int or not 0 <= optional_counter < 2**63:
                raise ValueError("untrained inbox origin times require bounded exact integers")
            if optional_counter > data.erased_at:
                raise ValueError("untrained inbox arrival cannot follow erasure")
    if data.observed_at is None and data.arrived_at is None:
        raise ValueError("untrained inbox origin requires at least one original arrival")
    if (data.event_id is None) != (data.arrived_at is None):
        raise ValueError("untrained inbox label requires both event and arrival time")
    if (
        data.observed_at is not None
        and data.arrived_at is not None
        and data.observed_at > data.arrived_at
    ):
        raise ValueError("untrained inbox label cannot precede observation")
    if data.reason not in ("deleted", "expired", "opt_out"):
        raise ValueError("untrained inbox origin erasure reason changed")


def untrained_inbox_origin_metadata(data: UntrainedInboxOriginData, maximum: int) -> bytes:
    """Encode exact scalar metadata within the caller's original aggregate cap."""
    if type(maximum) is not int or not 0 < maximum < 2**63:
        raise ValueError("untrained inbox metadata requires a positive exact bound")
    _require_data(data, maximum)
    encoded = json.dumps(
        asdict(data), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf8")
    if len(encoded) > maximum:
        raise ValueError("untrained inbox metadata exceeds its original bound")
    return encoded
