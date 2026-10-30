"""Complete bounded current-packet linear learner state and pure prediction.

Only the last fitted coefficients/evidence and cumulative committed counts persist.
No fit, row bank, role/clock/budget authority, RNG, IO or scheduling lives here.
"""

from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np

from src.core.categorical_linear_fit import (
    CategoricalLinearFit,
    _features,
    predict_categorical_linear,
    predict_categorical_linear_logits,
)
from src.core.categorical_probability_scores import validate_class_count
from src.core.coupled_energy import Array

MAX_COUNT = 2**53 - 1


@dataclass(frozen=True, slots=True)
class LinearFitPolicy:
    feature_dim: int
    num_classes: int

    def __post_init__(self) -> None:
        if (
            type(self) is not LinearFitPolicy
            or type(self.feature_dim) is not int
            or not 1 <= self.feature_dim <= 296
        ):
            raise ValueError("linear policy requires exact bounded1..296 feature geometry")
        validate_class_count(self.num_classes)


@dataclass(frozen=True, slots=True)
class LinearFitSnapshot:
    policy: LinearFitPolicy
    model: CategoricalLinearFit | None
    completed_updates: int
    training_examples: int

    def __post_init__(self) -> None:
        if type(self) is not LinearFitSnapshot or type(self.policy) is not LinearFitPolicy:
            raise ValueError("expected exact complete linear snapshot and policy")
        self.policy.__post_init__()
        updates, rows = self.completed_updates, self.training_examples
        if any(type(value) is not int or not 0 <= value <= MAX_COUNT for value in (updates, rows)):
            raise ValueError("linear snapshot counts must be native integers in0..2**53-1")
        if self.model is None:
            if updates or rows:
                raise ValueError("unfitted linear state requires zero committed counts")
            return
        if type(self.model) is not CategoricalLinearFit:
            raise ValueError("linear snapshot requires exact fitted evidence or None")
        self.model.__post_init__()
        if (
            self.model.feature_dim != self.policy.feature_dim
            or self.model.num_classes != self.policy.num_classes
            or updates < 1
            or not self.model.fit_rows + updates - 1
            <= rows
            <= self.model.fit_rows + 128 * (updates - 1)
        ):
            raise ValueError("linear snapshot model geometry/counts differ from its policy/history")


def copy_linear_policy(policy: LinearFitPolicy) -> LinearFitPolicy:
    if type(policy) is not LinearFitPolicy:
        raise ValueError("expected exact LinearFitPolicy")
    return replace(policy)


def copy_linear_snapshot(snapshot: LinearFitSnapshot) -> LinearFitSnapshot:
    if type(snapshot) is not LinearFitSnapshot:
        raise ValueError("expected exact LinearFitSnapshot")
    snapshot.__post_init__()
    return deepcopy(snapshot)


def linear_state_logits(snapshot: LinearFitSnapshot, features: Array) -> Array:
    state = copy_linear_snapshot(snapshot)
    if state.model is None:
        _features(features, state.policy.feature_dim)
        # Unfitted is explicit; do not manufacture a fitted-row record for zeros.
        return np.zeros((len(features), state.policy.num_classes), dtype=np.float64)
    return predict_categorical_linear_logits(state.model, features)


def predict_linear_state(snapshot: LinearFitSnapshot, features: Array) -> Array:
    state = copy_linear_snapshot(snapshot)
    if state.model is None:
        _features(features, state.policy.feature_dim)
        scores = np.full(
            (len(features), state.policy.num_classes),
            1 / state.policy.num_classes,
            dtype=np.float64,
        )
    else:
        scores = predict_categorical_linear(state.model, features)
    # Preserve the existing binary learner port; categorical streams receive K.
    return scores[:, 1:2].copy() if state.policy.num_classes == 2 else scores
