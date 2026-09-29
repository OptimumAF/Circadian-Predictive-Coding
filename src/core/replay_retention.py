"""Pure eviction choices and immutable audit facts for opt-in NumPy replay.

Inputs are distinct labeled-content IDs in retention order, a policy name,
and an optional predeclared seed. Outputs are one ID to evict and an immutable
exposure summary. This module does not train, select replay, or save checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable


@dataclass(frozen=True)
class ReplayRetentionPolicy:
    """Select bounded retained rows without changing replay sampling rules."""

    name: str
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.name not in {"content_hash", "recent_fifo", "seeded_reservoir"}:
            raise ValueError("unknown replay retention policy")
        if self.name == "seeded_reservoir":
            if type(self.seed) is not int or not 0 <= self.seed < 2**64:
                raise ValueError("replay reservoir seed must be a 64-bit nonnegative integer")
        elif self.seed is not None:
            raise ValueError("replay retention policy seed is only valid for seeded reservoir")

    def eviction_id(self, retained_ids: Iterable[str]) -> str:
        """Choose the item discarded when either declared cap is exceeded."""
        ids = tuple(retained_ids)
        if not ids:
            raise ValueError("cannot evict from empty replay retention")
        if self.name == "recent_fifo":
            return ids[0]
        if self.name == "content_hash":
            return max(ids)
        return max(ids, key=lambda content_id: (self._reservoir_rank(content_id), content_id))

    def _reservoir_rank(self, content_id: str) -> bytes:
        assert self.seed is not None
        # Why this: a seeded rank gives each distinct ID one stable draw, so
        # repeated epochs need no unbounded seen-ID ledger or process RNG.
        digest = sha256(b"numpy_replay_bottom_k_v1")
        digest.update(self.seed.to_bytes(8, "little"))
        digest.update(bytes.fromhex(content_id))
        return digest.digest()


DEFAULT_REPLAY_RETENTION_POLICY = ReplayRetentionPolicy("content_hash")


@dataclass(frozen=True)
class ReplayExposureSnapshot:
    """Observed duplicate and successfully applied replay IDs for an opt-in policy."""

    observed_ids: tuple[str, ...]
    duplicate_ids: tuple[str, ...]
    duplicate_occurrences: int
    exposed_ids: tuple[str, ...]
    replay_updates: int

    def __post_init__(self) -> None:
        for name in ("observed_ids", "duplicate_ids", "exposed_ids"):
            ids = getattr(self, name)
            if (
                type(ids) is not tuple
                or ids != tuple(sorted(set(ids)))
                or any(
                    type(sample_id) is not str
                    or len(sample_id) != 64
                    or any(character not in "0123456789abcdef" for character in sample_id)
                    for sample_id in ids
                )
            ):
                raise ValueError(f"replay {name} must contain sorted distinct content IDs")
        if not set(self.duplicate_ids).issubset(self.observed_ids):
            raise ValueError("replay duplicate IDs must have been observed")
        if not set(self.exposed_ids).issubset(self.observed_ids):
            raise ValueError("replay-exposed IDs must have been observed")
        if (
            type(self.duplicate_occurrences) is not int
            or self.duplicate_occurrences < len(self.duplicate_ids)
            or type(self.replay_updates) is not int
            or self.replay_updates < len(self.exposed_ids)
        ):
            raise ValueError("replay exposure counts are inconsistent")
