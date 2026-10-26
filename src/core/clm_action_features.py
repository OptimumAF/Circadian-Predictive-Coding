"""Bounded raw public-catalog features from one frozen encoder declaration.

No label, head projection, tensor implementation or provenance certification.
The concrete predictor validates/copies opaque feature storage before reuse.
"""

from dataclasses import dataclass
from typing import Any

from src.core.clm_head import FrozenEncoderIdentity, _exact
from src.core.typed_decision import DecisionCandidate


@dataclass(frozen=True)
class FrozenActionFeatures:
    identity: FrozenEncoderIdentity
    candidates: tuple[DecisionCandidate, ...]
    features: Any

    def __post_init__(self):
        _exact(self.identity, FrozenEncoderIdentity)
        if type(self.candidates) is not tuple or not 2 <= len(self.candidates) <= 64:
            raise ValueError("frozen actions require an immutable tuple of2..64candidates")
        for candidate in self.candidates:
            _exact(candidate, DecisionCandidate)
        ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(set(ids)) != len(ids):
            raise ValueError("frozen action candidate IDs must be unique")
