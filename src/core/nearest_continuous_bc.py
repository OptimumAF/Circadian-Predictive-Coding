"""Bounded nearest-row physical action retrieval in frozen feature coordinates.

Owns finite squared-distance math, earliest-row ties and explicit work counts.
No fitting, roles, learner state, simulator, persistence or selection authority.
"""

from dataclasses import dataclass

import numpy as np

from src.core import continuous_bc as bc
from src.core.continuous_demonstration import ContinuousActionSpec


@dataclass(frozen=True)
class NearestPrediction:
    actions: bc.Array
    selected_indices: tuple[int, ...]
    distance_candidates: int
    distance_feature_values: int


def nearest_continuous_actions(
    query: bc.Array,
    prototype_features: bc.Array,
    prototype_actions: bc.Array,
    spec: ContinuousActionSpec,
) -> NearestPrediction:
    owned_spec = bc._spec(spec)
    values = bc._features(query, owned_spec)
    prototypes = bc._features(prototype_features, owned_spec)
    actions = bc._targets(prototype_actions, len(prototypes), owned_spec)
    selected = []
    # Why: one (P,D) temporary avoids a (B,P,D) tensor at the bounded maxima.
    for row in values:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            differences = prototypes - row
            distances = np.sum(differences * differences, axis=1)
        if not np.isfinite(distances).all():
            raise FloatingPointError("nearest BC distances must be finite and representable")
        selected.append(int(np.argmin(distances)))
    candidates = len(values) * len(prototypes)
    return NearestPrediction(
        actions[selected].copy(),
        tuple(selected),
        candidates,
        candidates * owned_spec.observation_dim,
    )
