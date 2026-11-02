"""Bounded content selection for explicit TRAIN nearest-memory retention.

Inputs are retained/incoming content IDs and an existing retention rule. Output
indices refer to old rows followed by incoming rows. No arrays, role authority,
learning, process RNG, future labels or unbounded seen-ID ledger lives here.
"""

from dataclasses import dataclass, field
from typing import TypeAlias

from src.core.replay_retention import ReplayRetentionPolicy, seeded_content_rank


@dataclass(frozen=True)
class RecentPriorityRetentionPolicy:
    """Nearest-only recent window plus stable priority; no replay admission."""

    seed: int
    name: str = field(default="recent_priority", init=False)

    def __post_init__(self):
        ReplayRetentionPolicy("seeded_reservoir", self.seed)

    def _reservoir_rank(self, content_id: str) -> bytes:
        return seeded_content_rank(self.seed, content_id)


NearestRetentionPolicy: TypeAlias = ReplayRetentionPolicy | RecentPriorityRetentionPolicy


def retention_policy(value: NearestRetentionPolicy) -> NearestRetentionPolicy:
    if type(value) is RecentPriorityRetentionPolicy:
        if type(value.name) is not str or value.name != "recent_priority":
            raise ValueError("recent-priority policy name changed")
        return RecentPriorityRetentionPolicy(value.seed)
    if (
        type(value) is not ReplayRetentionPolicy
        or type(value.name) is not str
        or value.name
        not in (
            "recent_fifo",
            "seeded_reservoir",
        )
    ):
        raise ValueError("nearest retention requires exact recent_fifo or seeded_reservoir")
    return ReplayRetentionPolicy(value.name, value.seed)


def capacity(value: int) -> int:
    if type(value) is not int or not 1 <= value <= 256:
        raise ValueError("nearest retained capacity must be a native integer in1..256")
    return value


def rank_evaluations_per_discard(maximum: int, policy: NearestRetentionPolicy) -> int:
    maximum, policy = capacity(maximum), retention_policy(policy)
    if isinstance(policy, RecentPriorityRetentionPolicy):
        if maximum < 2:
            raise ValueError("recent-priority memory requires capacity2..256")
        return maximum + 1 - maximum // 2
    return maximum + 1 if policy.name == "seeded_reservoir" else 0


def content_ids(value: tuple[str, ...], maximum: int) -> tuple[str, ...]:
    if (
        type(value) is not tuple
        or len(value) > maximum
        or any(
            type(item) is not str
            or len(item) != 64
            or any(c not in "0123456789abcdef" for c in item)
            for item in value
        )
    ):
        raise ValueError("nearest content IDs require a bounded tuple of SHA256 hex strings")
    return value


@dataclass(frozen=True)
class RetainedSelection:
    indices: tuple[int, ...]
    ids: tuple[str, ...]
    duplicate_rows: int
    discarded_rows: int
    selection_candidates: int
    reservoir_rank_evaluations: int


def select_retained_rows(
    prior: tuple[str, ...],
    incoming: tuple[str, ...],
    maximum: int,
    policy: NearestRetentionPolicy,
) -> RetainedSelection:
    maximum, policy = capacity(maximum), retention_policy(policy)
    rank_width = rank_evaluations_per_discard(maximum, policy)
    recent = maximum // 2 if isinstance(policy, RecentPriorityRetentionPolicy) else 0
    prior, incoming = content_ids(prior, maximum), content_ids(incoming, 256)
    if len(set(prior)) != len(prior) or not incoming:
        raise ValueError("retained IDs must be distinct and incoming IDs nonempty")
    bank = dict(zip(prior, range(len(prior)), strict=True))
    duplicates = discarded = candidates = ranks = 0
    for offset, row_id in enumerate(incoming, len(prior)):
        candidates += len(bank) + 1
        if row_id in bank:
            duplicates += 1
            if not recent:
                continue
            # A repeated current context must become recent, even with a poor rank.
            del bank[row_id]
        bank[row_id] = offset
        if len(bank) > maximum:
            ranks += rank_width
            if isinstance(policy, RecentPriorityRetentionPolicy):
                eligible = tuple(bank)[:-recent]
                evicted = max(eligible, key=lambda key: (policy._reservoir_rank(key), key))
            else:
                evicted = policy.eviction_id(tuple(bank))
            del bank[evicted]
            discarded += 1
    return RetainedSelection(
        tuple(bank.values()), tuple(bank), duplicates, discarded, candidates, ranks
    )
