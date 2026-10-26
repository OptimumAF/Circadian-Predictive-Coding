"""Explicit TRAIN catalog bindings for directional CLM supervision.

Features remain opaque here. These records validate declared catalog order and
indices; they do not encode text, authorize release or perform native learning.
"""

from dataclasses import dataclass
from typing import Any
from unicodedata import category

from src.core.clm_head import FrozenEncoderIdentity, _exact, _integer

CATALOG_OBJECTIVE = "clm_full_catalog_cross_entropy_v1"


def _candidate_ids(value):
    if type(value) is not tuple or not 2 <= len(value) <= 64:
        raise ValueError("candidate_ids requires an immutable tuple of 2..64 IDs")
    for candidate in value:
        if (
            type(candidate) is not str
            or not candidate
            or candidate != candidate.strip()
            or len(candidate.encode("utf8")) > 128
            or any(category(char) == "Cc" for char in candidate)
        ):
            raise ValueError("catalog candidate IDs require bounded native text without controls")
    if len(set(value)) != len(value):
        raise ValueError("catalog candidate IDs must be unique")


@dataclass(frozen=True)
class FrozenCatalogTrainingBatch:
    identity: FrozenEncoderIdentity
    states: Any
    actions: Any
    candidate_ids: tuple[str, ...]
    role: str = "train"

    def __post_init__(self):
        _exact(self.identity, FrozenEncoderIdentity)
        _candidate_ids(self.candidate_ids)
        if type(self.role) is not str or self.role != "train":
            raise ValueError("catalog learning admits TRAIN only")


@dataclass(frozen=True)
class CLMCatalogTargets:
    candidate_ids: tuple[str, ...]
    candidate_indices: tuple[int, ...]

    def __post_init__(self):
        _candidate_ids(self.candidate_ids)
        if type(self.candidate_indices) is not tuple or not 2 <= len(self.candidate_indices) <= 64:
            raise ValueError("catalog targets require an immutable tuple of 2..64 indices")
        for index in self.candidate_indices:
            _integer(index, "candidate index", 0, len(self.candidate_ids) - 1)
