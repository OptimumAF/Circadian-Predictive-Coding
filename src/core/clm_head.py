"""Trusted frozen-encoder declarations and bounded native CLM head settings.

These records bind features to a declared encoder recipe; they neither attest
encoder provenance nor load weights, authorize data release or perform learning.
"""

from dataclasses import dataclass, fields
from math import isfinite
from typing import Any


def _exact(value, kind):
    if type(value) is not kind or set(vars(value)) != {f.name for f in fields(kind)}:
        raise ValueError(f"expected exact complete {kind.__name__}")
    value.__post_init__()


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")


@dataclass(frozen=True)
class FrozenEncoderIdentity:
    model_id: str
    revision: str
    tokenizer_revision: str
    pooling: str
    recipe: str
    embedding_dim: int

    def __post_init__(self):
        for name in ("model_id", "revision", "tokenizer_revision", "pooling", "recipe"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or not value
                or value != value.strip()
                or len(value.encode("utf8")) > 512
                or any(ord(char) < 32 for char in value)
            ):
                raise ValueError(f"{name} requires bounded native text")
        _integer(self.embedding_dim, "embedding_dim", 1, 65536)


@dataclass(frozen=True)
class FrozenEmbeddingBatch:
    identity: FrozenEncoderIdentity
    states: Any
    actions: Any
    role: str = "train"

    def __post_init__(self):
        _exact(self.identity, FrozenEncoderIdentity)
        if type(self.role) is not str or self.role != "train":
            raise ValueError("frozen embedding learning admits TRAIN only")


@dataclass(frozen=True)
class FrozenChoiceBatch:
    """Frozen query/candidate features for readout, never a paired TRAIN batch."""

    identity: FrozenEncoderIdentity
    states: Any
    actions: Any
    role: str = "outer_selection"

    def __post_init__(self):
        _exact(self.identity, FrozenEncoderIdentity)
        if type(self.role) is not str or self.role not in ("train", "outer_selection"):
            raise ValueError("choice readout admits TRAIN or outer_selection only")


@dataclass(frozen=True)
class CLMTargets:
    group_codes: tuple[int, ...]

    def __post_init__(self):
        if type(self.group_codes) is not tuple or not 2 <= len(self.group_codes) <= 64:
            raise ValueError("group_codes requires an immutable tuple of2..64 codes")
        for code in self.group_codes:
            _integer(code, "group code", 0, 2**53 - 1)


@dataclass(frozen=True)
class CLMHeadPolicy:
    projection_dim: int
    max_updates: int
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    max_rows: int = 64
    max_parameters: int = 32000000
    max_restores: int = 8

    def __post_init__(self):
        for name, low, high in (
            ("projection_dim", 1, 65536),
            ("max_updates", 1, 1000000),
            ("max_rows", 2, 64),
            ("max_parameters", 1, 32000000),
            ("max_restores", 0, 8),
        ):
            _integer(getattr(self, name), name, low, high)
        for name in ("learning_rate", "weight_decay"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative native number")
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive")
