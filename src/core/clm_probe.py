"""Explicit frozen-feature probe roles; no TRAIN conversion or label authority.

These declarations bind features to an encoder identity. They neither encode,
score, grant held-out access nor make a disposable probe a deployable learner.
"""

from dataclasses import dataclass
from typing import Any

from src.core.clm_head import FrozenEncoderIdentity, _exact


@dataclass(frozen=True)
class FrozenProbeEmbeddingBatch:
    identity: FrozenEncoderIdentity
    states: Any
    actions: Any
    role: str = "probe_train"

    def __post_init__(self):
        _exact(self.identity, FrozenEncoderIdentity)
        if type(self.role) is not str or self.role != "probe_train":
            raise ValueError("probe updates require explicit probe_train features")


@dataclass(frozen=True)
class FrozenProbeChoiceBatch:
    identity: FrozenEncoderIdentity
    states: Any
    actions: Any
    role: str = "probe_test"

    def __post_init__(self):
        _exact(self.identity, FrozenEncoderIdentity)
        if type(self.role) is not str or self.role not in (
            "train",
            "probe_train",
            "probe_test",
            "outer_selection",
        ):
            raise ValueError(
                "probe readouts admit train, probe_train, probe_test or outer_selection only"
            )
