"""Two bounded LS heads selected only by explicit public context features.

Reuses the established context projection, not perceptron learning. Owns complete
head state/counters and pure reads; no labels for routing, row bank or IO.
"""

from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np

from src.core.context_perceptron import ContextPerceptronPolicy, _context_rows
from src.core.categorical_linear_fit import predict_categorical_linear
from src.core.categorical_linear_state import (
    MAX_COUNT,
    LinearFitPolicy,
    LinearFitSnapshot,
    copy_linear_snapshot,
)
from src.core.coupled_energy import Array


@dataclass(frozen=True, slots=True)
class ContextLinearFitPolicy:
    feature_dim: int
    num_classes: int
    context_columns: tuple[int, int]

    def __post_init__(self):
        routing_policy(self)


def routing_policy(policy: ContextLinearFitPolicy) -> ContextPerceptronPolicy:
    if type(policy) is not ContextLinearFitPolicy:
        raise ValueError("expected exact ContextLinearFitPolicy")
    return ContextPerceptronPolicy(policy.feature_dim, policy.num_classes, policy.context_columns)


@dataclass(frozen=True, slots=True)
class ContextLinearFitSnapshot:
    policy: ContextLinearFitPolicy
    heads: tuple[LinearFitSnapshot, LinearFitSnapshot]
    completed_updates: int
    training_examples: int

    def __post_init__(self):
        if type(self) is not ContextLinearFitSnapshot:
            raise ValueError("expected exact context linear snapshot")
        routing_policy(self.policy)
        if type(self.heads) is not tuple or len(self.heads) != 2:
            raise ValueError("context linear snapshot requires exactly two complete heads")
        expected = LinearFitPolicy(self.policy.feature_dim - 2, self.policy.num_classes)
        for head in self.heads:
            if copy_linear_snapshot(head).policy != expected:
                raise ValueError("context linear head differs from projected policy")
        for name in ("completed_updates", "training_examples"):
            value = getattr(self, name)
            if (
                type(value) is not int
                or not 0 <= value <= MAX_COUNT
                or value != sum(getattr(h, name) for h in self.heads)
            ):
                raise ValueError("context linear counters must equal bounded complete head sums")


def copy_context_linear_state(state: ContextLinearFitSnapshot) -> ContextLinearFitSnapshot:
    if type(state) is not ContextLinearFitSnapshot:
        raise ValueError("expected exact context linear snapshot")
    state.__post_init__()
    return deepcopy(state)


def create_context_linear_state(policy: ContextLinearFitPolicy) -> ContextLinearFitSnapshot:
    routing_policy(policy)
    owned = replace(policy)
    heads = tuple(
        LinearFitSnapshot(LinearFitPolicy(policy.feature_dim - 2, policy.num_classes), None, 0, 0)
        for _ in range(2)
    )
    return ContextLinearFitSnapshot(owned, (heads[0], heads[1]), 0, 0)


def predict_context_linear(state: ContextLinearFitSnapshot, features: Array) -> Array:
    owned = copy_context_linear_state(state)
    indices, projected = _context_rows(features, routing_policy(owned.policy))
    result = np.full(
        (len(features), owned.policy.num_classes), 1 / owned.policy.num_classes, dtype=np.float64
    )
    for index, head in enumerate(owned.heads):
        selected = indices == index
        if selected.any() and head.model is not None:
            result[selected] = predict_categorical_linear(head.model, projected[selected])
    return result[:, 1:2].copy() if owned.policy.num_classes == 2 else result
