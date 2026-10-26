"""Explicit fixed-vocabulary query features for ordinary linear routing.

Owns bounded immutable feature/class schema and pure projection. No vocabulary
fitting, labels, data roles, learner, scheduling, permission or IO lives here.
"""

from dataclasses import dataclass
import re

import numpy as np

from src.core.coupled_energy import Array
from src.core.typed_decision import DecisionCandidate, DecisionRequest, _text


@dataclass(frozen=True)
class DecisionFeatureSpec:
    vocabulary: tuple[str, ...]
    catalog: tuple[DecisionCandidate, ...]

    def __post_init__(self) -> None:
        words = self.vocabulary
        if type(words) is not tuple or not 1 <= len(words) <= 128:
            raise ValueError("vocabulary must be an immutable tuple of 1..128 tokens")
        for word in words:
            _text(word, "vocabulary token", 64)
            if re.findall(r"\w+", word.casefold()) != [word]:
                raise ValueError("vocabulary requires canonical casefolded word tokens")
        if tuple(sorted(set(words))) != words:
            raise ValueError("vocabulary tokens must be unique and sorted")
        if type(self.catalog) is not tuple or not 3 <= len(self.catalog) <= 8:
            raise ValueError("linear categorical catalog must contain 3..8 candidates")
        # Why this: two-class AttributionStream batches have binary semantics.
        DecisionRequest("catalog-validation", "catalog", self.catalog).__post_init__()
        if tuple(sorted(self.catalog, key=lambda item: item.candidate_id)) != self.catalog:
            raise ValueError("catalog must be sorted by candidate ID")


def encode_decision(request: DecisionRequest, spec: DecisionFeatureSpec) -> tuple[Array, int]:
    """Return owned binary token presence and the number of parsed query tokens."""
    if type(spec) is not DecisionFeatureSpec or type(request) is not DecisionRequest:
        raise ValueError("expected exact DecisionFeatureSpec and DecisionRequest")
    spec.__post_init__()
    request.__post_init__()
    catalog = {item.candidate_id: item for item in spec.catalog}
    if any(catalog.get(item.candidate_id) != item for item in request.allowed_candidates):
        raise ValueError("allowed candidates must match the fixed catalog IDs/descriptions")
    words = re.findall(r"\w+", request.query.casefold())
    present = set(words)
    features = np.array([[float(word in present) for word in spec.vocabulary]], dtype=np.float64)
    return features, len(words)
