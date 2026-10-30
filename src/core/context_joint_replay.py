"""Bounded current+retained context packet algebra; no role/replay authority.

The application must admit original TRAIN retention and private probe ownership.
This module validates public routing/targets and composes one solver input, with
an equal-row current-only control. It performs no fitting, selection, IO or RNG.
"""

from dataclasses import dataclass

import numpy as np

from src.core.context_perceptron import ContextPerceptronPolicy, route_context_features
from src.core.categorical_linear_state import LinearFitPolicy
from src.core.categorical_probability_scores import validate_class_targets


@dataclass(frozen=True, slots=True)
class JointContextReplayPolicy:
    feature_dim: int
    num_classes: int
    context_columns: tuple[int, int]
    control: str = "joint"

    def __post_init__(self):
        ContextPerceptronPolicy(self.feature_dim, self.num_classes, self.context_columns)
        LinearFitPolicy(self.feature_dim - 2, self.num_classes)
        if type(self.control) is not str or self.control not in ("joint", "current_only"):
            raise ValueError("joint replay control must be joint or current_only")


def packet_context(policy, features, targets):
    if type(policy) is not JointContextReplayPolicy:
        raise ValueError("expected exact JointContextReplayPolicy")
    policy.__post_init__()
    index, _ = route_context_features(
        features,
        ContextPerceptronPolicy(policy.feature_dim, policy.num_classes, policy.context_columns),
    )
    validate_class_targets(targets, len(features), policy.num_classes)
    return index


def compose_joint_packet(policy, current, retained):
    """Keep all declared exposures; refuse conflicting labels before any fit."""
    index = packet_context(policy, *current)
    if type(retained) is not tuple or not retained:
        raise ValueError("joint packet requires prior same-context packets")
    rows = len(current[0]) + sum(len(x) for x, _ in retained)
    if rows > 128:
        raise ValueError("joint packet exceeds the existing 128-row solver bound")
    for x, y in retained:
        if packet_context(policy, x, y) != index:
            raise ValueError("joint packet contexts must match current public context")
    features = np.concatenate((current[0], *(x for x, _ in retained)))
    targets = np.concatenate((current[1], *(y for _, y in retained)))
    for i in range(rows):
        for j in range(i):
            if np.array_equal(features[i], features[j]) and targets[i, 0] != targets[j, 0]:
                raise ValueError("joint packet labels conflict on identical public features")
    if policy.control == "current_only":
        repeats, extra = divmod(rows, len(current[0]))
        if extra:
            raise ValueError("current-only control requires equal complete-packet row exposure")
        return np.tile(current[0], (repeats, 1)), np.tile(current[1], (repeats, 1))
    return features, targets
