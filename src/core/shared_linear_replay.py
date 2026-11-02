"""Bounded shared-head packet algebra; no roles, fitting or replay authority.

The caller supplies admitted current/retained arrays and owns all costs/consent.
Both controls validate the same retained information. Returned arrays are owned.
"""

from dataclasses import dataclass

import numpy as np

from src.core.categorical_linear_fit import _features
from src.core.categorical_probability_scores import validate_class_targets
from src.core.categorical_linear_state import LinearFitPolicy


@dataclass(frozen=True, slots=True)
class SharedJointReplayPolicy:
    """Explicit shared-head geometry/control, without routed-context authority."""

    feature_dim: int
    num_classes: int
    control: str = "joint"

    def __post_init__(self):
        if type(self) is not SharedJointReplayPolicy:
            raise ValueError("shared joint replay requires an exact policy")
        LinearFitPolicy(self.feature_dim, self.num_classes)
        if type(self.control) is not str or self.control not in ("joint", "current_only"):
            raise ValueError("shared joint replay control must be joint or current_only")


def compose_shared_packet(current, retained, num_classes, *, control, current_once=False):
    if type(current) is not tuple or len(current) != 2:
        raise ValueError("shared replay requires a current feature/target pair")
    if type(control) is not str or control not in ("joint", "current_only"):
        raise ValueError("shared replay control must be joint or current_only")
    if type(current_once) is not bool:
        raise ValueError("current-once composition requires an exact bool")
    if type(retained) is not tuple or not 1 <= len(retained) <= 32:
        raise ValueError("shared replay requires bounded retained packets")
    if any(type(packet) is not tuple or len(packet) != 2 for packet in retained):
        raise ValueError("retained packets must be feature/target pairs")
    width = _features(current[0]).shape[1] - 1
    packets = (current, *retained)
    for features, targets in packets:
        _features(features, width)
        validate_class_targets(targets, len(features), num_classes)
    rows = sum(len(features) for features, _ in packets)
    if rows > 128:
        raise ValueError("shared replay exceeds the existing128-row solver bound")
    features = np.concatenate([x for x, _ in packets])
    targets = np.concatenate([y for _, y in packets])
    for i in range(rows):
        for j in range(i):
            if np.array_equal(features[i], features[j]) and targets[i, 0] != targets[j, 0]:
                raise ValueError("shared replay labels conflict on identical public features")
    if current_once:
        matches = [
            i
            for i, (x, y) in enumerate(retained)
            if np.array_equal(x, current[0]) and np.array_equal(y, current[1])
        ]
        if len(retained) != 2 or len(matches) != 1:
            raise ValueError("current-once requires exactly one whole match in two TRAIN packets")
        # Why: validate ALL original information before removing only this duplicate.
        packets = (current, retained[1 - matches[0]])
        features = np.concatenate([x for x, _ in packets])
        targets = np.concatenate([y for _, y in packets])
        rows = len(features)
    if control == "current_only":
        repeats, extra = divmod(rows, len(current[0]))
        if extra:
            raise ValueError("current-only control requires equal complete current packets")
        return np.tile(current[0], (repeats, 1)), np.tile(current[1], (repeats, 1))
    return features, targets
