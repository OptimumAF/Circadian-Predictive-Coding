"""Owned serving observations and local transaction receipts; no native handles.

Cache keys are configured full feature SHA256 identities. Metadata is observable
audit context only; it cannot configure prediction, action decoding or guards.
"""

from dataclasses import dataclass
from typing import Generic, TypeVar

from src.core.experience import require_tick
from src.core.promotion_guard import PromotionGuardReport

Prediction = TypeVar("Prediction")
State = TypeVar("State")


@dataclass(frozen=True)
class ServingConfiguration:
    cache_ttl_ticks: int
    max_cache_entries: int

    def __post_init__(self) -> None:
        require_tick(self.cache_ttl_ticks, "cache_ttl_ticks")
        require_tick(self.max_cache_entries, "max_cache_entries")
        if self.cache_ttl_ticks == 0 or self.max_cache_entries == 0:
            raise ValueError("cache TTL and entry capacity must be positive")


@dataclass(frozen=True)
class CachedPrediction(Generic[Prediction]):
    expires_at: int
    prediction: Prediction


@dataclass(frozen=True)
class ServingSnapshot(Generic[State, Prediction]):
    generation: int
    actor_version: str
    model_state: State
    configuration: ServingConfiguration
    cache: dict[str, CachedPrediction[Prediction]]
    metadata: dict[str, object]
    last_cache_tick: int


@dataclass(frozen=True)
class ServingPrediction(Generic[Prediction]):
    generation: int
    actor_version: str
    prediction: Prediction
    metadata: dict[str, object]
    cache_hit: bool


@dataclass(frozen=True, eq=False)
class PreparedPromotion:
    """Identity-checked local handle; constructing a copy grants no permission."""

    actor_generation: int
    report: PromotionGuardReport


@dataclass(frozen=True, eq=False)
class PromotionReceipt:
    generation: int
    previous_version: str
    actor_version: str
