"""Two owned conventional perceptrons selected by explicit public context.

Pure bounded state, homogeneous training and per-row readonly prediction.
No context discovery, labels for routing, data roles, stream, budgets or IO.
"""

from dataclasses import dataclass, fields

import numpy as np

from src.core.categorical_perceptron import (
    PerceptronPolicy,
    PerceptronSnapshot,
    _features,
    copy_perceptron_snapshot,
    create_perceptron_snapshot,
    predict_perceptron,
    update_perceptron,
)
from src.core.coupled_energy import Array

_MAX_COUNT = 2**53 - 1


def _exact(value, kind):
    if type(value) is not kind or set(vars(value)) != {f.name for f in fields(kind)}:
        raise ValueError(f"expected exact complete {kind.__name__}")


@dataclass(frozen=True)
class ContextPerceptronPolicy:
    feature_dim: int
    num_classes: int
    context_columns: tuple[int, int]

    def __post_init__(self):
        columns = self.context_columns
        if (
            type(self.feature_dim) is not int
            # Two routing columns plus the original perceptron's bounded296 inputs.
            or not 3 <= self.feature_dim <= 298
            or type(columns) is not tuple
            or len(columns) != 2
            or any(type(i) is not int or not 0 <= i < self.feature_dim for i in columns)
            or columns[0] >= columns[1]
        ):
            raise ValueError(
                "context policy requires two ordered distinct columns in bounded input"
            )
        PerceptronPolicy(self.feature_dim - 2, self.num_classes)


def copy_context_policy(policy: ContextPerceptronPolicy) -> ContextPerceptronPolicy:
    _exact(policy, ContextPerceptronPolicy)
    return ContextPerceptronPolicy(policy.feature_dim, policy.num_classes, policy.context_columns)


@dataclass(frozen=True)
class ContextPerceptronSnapshot:
    policy: ContextPerceptronPolicy
    heads: tuple[PerceptronSnapshot, PerceptronSnapshot]
    completed_updates: int
    training_examples: int
    mistake_updates: int


def copy_context_snapshot(snapshot: ContextPerceptronSnapshot) -> ContextPerceptronSnapshot:
    _exact(snapshot, ContextPerceptronSnapshot)
    policy = copy_context_policy(snapshot.policy)
    if type(snapshot.heads) is not tuple or len(snapshot.heads) != 2:
        raise ValueError("context snapshot requires exactly two complete heads")
    heads = (
        copy_perceptron_snapshot(snapshot.heads[0]),
        copy_perceptron_snapshot(snapshot.heads[1]),
    )
    expected = PerceptronPolicy(policy.feature_dim - 2, policy.num_classes)
    if any(head.policy != expected for head in heads):
        raise ValueError("context head policy differs from fixed input/class schema")
    for name in ("completed_updates", "training_examples", "mistake_updates"):
        value = getattr(snapshot, name)
        if (
            type(value) is not int
            or not 0 <= value <= _MAX_COUNT
            or value != sum(getattr(head, name) for head in heads)
        ):
            raise ValueError("context counters must equal complete bounded head sums")
    return ContextPerceptronSnapshot(
        policy,
        heads,
        snapshot.completed_updates,
        snapshot.training_examples,
        snapshot.mistake_updates,
    )


def create_context_snapshot(policy: ContextPerceptronPolicy) -> ContextPerceptronSnapshot:
    policy = copy_context_policy(policy)
    head = PerceptronPolicy(policy.feature_dim - 2, policy.num_classes)
    return ContextPerceptronSnapshot(
        policy, (create_perceptron_snapshot(head), create_perceptron_snapshot(head)), 0, 0, 0
    )


def _context_rows(features: Array, policy: ContextPerceptronPolicy) -> tuple[Array, Array]:
    policy = copy_context_policy(policy)
    augmented = _features(features, policy.feature_dim)
    contexts = augmented[:, policy.context_columns]
    if not np.isin(contexts, (0.0, 1.0)).all() or not (contexts.sum(axis=1) == 1).all():
        raise ValueError("context features require exactly one binary public context")
    indices = np.argmax(contexts, axis=1)
    return indices, np.delete(augmented[:, :-1], policy.context_columns, axis=1)


def route_context_features(features: Array, policy: ContextPerceptronPolicy) -> tuple[int, Array]:
    """Require exactly one public context per row and one context per TRAIN batch."""
    indices, routed = _context_rows(features, policy)
    if not (indices == indices[0]).all():
        raise ValueError("context batches must be homogeneous; mixed contexts refused")
    return int(indices[0]), routed


def update_context_perceptron(
    snapshot: ContextPerceptronSnapshot, features: Array, targets: Array
) -> ContextPerceptronSnapshot:
    """Commit one original head pass atomically; inactive head remains unchanged."""
    before = copy_context_snapshot(snapshot)
    index, routed = route_context_features(features, before.policy)
    if (
        before.completed_updates == _MAX_COUNT
        or before.training_examples + len(routed) > _MAX_COUNT
    ):
        raise ValueError("context counter overflow")
    heads = list(before.heads)
    heads[index] = update_perceptron(heads[index], routed, targets)
    # Why: routing isolates both coefficients and bias, preventing other-context
    # updates from silently changing an acquired head. Capacity cost is explicit.
    return copy_context_snapshot(
        ContextPerceptronSnapshot(
            before.policy,
            (heads[0], heads[1]),
            before.completed_updates + 1,
            before.training_examples + len(routed),
            sum(h.mistake_updates for h in heads),
        )
    )


def predict_context_perceptron(snapshot: ContextPerceptronSnapshot, features: Array) -> Array:
    """Score mixed readonly guards in original row order without changing either head."""
    owned = copy_context_snapshot(snapshot)
    indices, routed = _context_rows(features, owned.policy)
    result = np.empty((len(routed), owned.policy.num_classes), dtype=np.float64)
    for index, head in enumerate(owned.heads):
        selected = indices == index
        if selected.any():
            result[selected] = predict_perceptron(head, routed[selected])
    return result
