"""A bounded, deterministic multiclass perceptron with complete owned state.

Inputs are exact float64 feature/label batches and immutable policy snapshots.
Outputs are new snapshots or softmax probabilities for existing scoring tools.
The unit mistake rule optimizes neither CE nor a replay/selection objective;
budgets, data roles, stream scheduling, projections, and IO belong to callers.
"""

from dataclasses import dataclass, fields, replace
from math import isfinite
from typing import Any

import numpy as np

from src.core.categorical_probability_scores import (
    validate_class_count,
    validate_class_probabilities,
    validate_class_targets,
)
from src.core.coupled_energy import Array

PERCEPTRON_MISTAKES_DIAGNOSTIC = "categorical_perceptron_mistakes_v1"
_MAX_COUNT = 2**53 - 1


def _record(value: Any, expected: type) -> None:
    if type(value) is not expected or set(vars(value)) != {
        field.name for field in fields(expected)
    }:
        raise ValueError(f"an exact complete {expected.__name__} is required")


@dataclass(frozen=True)
class PerceptronPolicy:
    feature_dim: int
    num_classes: int

    def __post_init__(self) -> None:
        if type(self.feature_dim) is not int or not 1 <= self.feature_dim <= 296:
            raise ValueError("perceptron feature_dim must be a native integer in 1..296")
        validate_class_count(self.num_classes)


@dataclass(frozen=True)
class PerceptronSnapshot:
    policy: PerceptronPolicy
    coefficients: tuple[tuple[float, ...], ...]
    completed_updates: int
    training_examples: int
    mistake_updates: int


def copy_perceptron_policy(policy: PerceptronPolicy) -> PerceptronPolicy:
    """Revalidate frozen-record tampering and own the policy record."""
    _record(policy, PerceptronPolicy)
    return replace(policy)


def copy_perceptron_snapshot(snapshot: PerceptronSnapshot) -> PerceptronSnapshot:
    """Validate a complete state and detach its policy and coefficient tuples."""
    _record(snapshot, PerceptronSnapshot)
    policy = copy_perceptron_policy(snapshot.policy)
    updates, rows, mistakes = (
        snapshot.completed_updates,
        snapshot.training_examples,
        snapshot.mistake_updates,
    )
    if any(
        type(value) is not int or not 0 <= value <= _MAX_COUNT
        for value in (updates, rows, mistakes)
    ):
        raise ValueError("perceptron counters must be bounded nonnegative native integers")
    if not updates <= rows <= 128 * updates or mistakes > rows:
        raise ValueError("perceptron completed passes, training rows, and mistakes disagree")
    coefficients = snapshot.coefficients
    if type(coefficients) is not tuple or len(coefficients) != policy.feature_dim + 1:
        raise ValueError("perceptron coefficients require an exact (D+1, K) tuple")
    owned = []
    for row in coefficients:
        if (
            type(row) is not tuple
            or len(row) != policy.num_classes
            or any(
                type(value) is not float or not isfinite(value) or abs(value) > mistakes
                for value in row
            )
        ):
            raise ValueError(
                "perceptron coefficient rows require finite native floats bounded by mistakes"
            )
        owned.append(tuple(value for value in row))
    return PerceptronSnapshot(policy, tuple(owned), updates, rows, mistakes)


def create_perceptron_snapshot(policy: PerceptronPolicy) -> PerceptronSnapshot:
    """Create the frozen zero state without consuming any RNG."""
    owned = copy_perceptron_policy(policy)
    coefficients = tuple(
        tuple(0.0 for _ in range(owned.num_classes)) for _ in range(owned.feature_dim + 1)
    )
    return PerceptronSnapshot(owned, coefficients, 0, 0, 0)


def _features(features: Array, feature_dim: int) -> Array:
    if (
        type(features) is not np.ndarray
        or features.dtype != np.dtype(np.float64)
        or features.ndim != 2
        or features.shape[1] != feature_dim
        or not 1 <= features.shape[0] <= 128
        or not np.isfinite(features).all()
        or not ((features >= -1) & (features <= 1)).all()
    ):
        raise ValueError(
            f"perceptron features require finite exact float64 Bx{feature_dim} in [-1, 1], B in 1..128"
        )
    # Why: an owned bounded batch prevents later caller mutation from changing
    # the supplied-order pass or its returned probabilities.
    return np.column_stack((features.copy(), np.ones(features.shape[0], dtype=np.float64)))


def update_perceptron(
    snapshot: PerceptronSnapshot, features: Array, targets: Array
) -> PerceptronSnapshot:
    """Return one complete supplied-order pass, with first-argmax tie breaking."""
    owned = copy_perceptron_snapshot(snapshot)
    augmented = _features(features, owned.policy.feature_dim)
    rows = len(augmented)
    validate_class_targets(targets, rows, owned.policy.num_classes)
    labels = targets[:, 0].astype(np.int64, copy=True)
    updates, examples = owned.completed_updates + 1, owned.training_examples + rows
    if updates > _MAX_COUNT or examples > _MAX_COUNT:
        raise ValueError("perceptron native counter overflow")
    coefficients = np.array(owned.coefficients, dtype=np.float64)
    mistakes = owned.mistake_updates
    for row, label in zip(augmented, labels, strict=True):
        predicted = int(np.argmax(row @ coefficients))
        if predicted != label:
            if mistakes == _MAX_COUNT:
                raise ValueError("perceptron mistake counter overflow")
            coefficients[:, label] += row
            coefficients[:, predicted] -= row
            mistakes += 1
    result = PerceptronSnapshot(
        owned.policy,
        tuple(tuple(float(value) for value in row) for row in coefficients),
        updates,
        examples,
        mistakes,
    )
    return copy_perceptron_snapshot(result)


def predict_perceptron(snapshot: PerceptronSnapshot, features: Array) -> Array:
    """Return stable softmax scores; probabilities are not the update objective."""
    owned = copy_perceptron_snapshot(snapshot)
    augmented = _features(features, owned.policy.feature_dim)
    logits = augmented @ np.array(owned.coefficients, dtype=np.float64)
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    validate_class_probabilities(probabilities, owned.policy.num_classes)
    return probabilities
