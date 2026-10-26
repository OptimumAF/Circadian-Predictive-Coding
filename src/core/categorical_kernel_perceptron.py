"""A bounded multiclass perceptron using the fixed quadratic polynomial kernel.

Inputs are exact float64 batches and complete immutable snapshots; outputs are
new snapshots or softmax scoring probabilities. One pass follows supplied row
order with no RNG, optimizer, averaging, or eviction. Data roles, projection,
budgets, stream scheduling, and scientific comparisons belong to callers.
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
from src.core.learner_ports import TrainingUpdateFailed

KERNEL_PERCEPTRON_MISTAKES_DIAGNOSTIC = "categorical_polynomial_kernel_perceptron_mistakes_v1"
_MAX_COUNT = 2**53 - 1


def _record(value: Any, expected: type) -> None:
    if type(value) is not expected or set(vars(value)) != {
        field.name for field in fields(expected)
    }:
        raise ValueError(f"an exact complete {expected.__name__} is required")


@dataclass(frozen=True)
class KernelPerceptronPolicy:
    feature_dim: int
    num_classes: int
    max_prototypes: int = 128

    def __post_init__(self) -> None:
        if type(self.feature_dim) is not int or not 1 <= self.feature_dim <= 296:
            raise ValueError("kernel perceptron feature_dim must be a native integer in 1..296")
        validate_class_count(self.num_classes)
        if type(self.max_prototypes) is not int or not 1 <= self.max_prototypes <= 128:
            raise ValueError("kernel perceptron max_prototypes must be a native integer in 1..128")


@dataclass(frozen=True)
class KernelPerceptronSnapshot:
    policy: KernelPerceptronPolicy
    prototypes: tuple[tuple[float, ...], ...]
    coefficients: tuple[tuple[int, ...], ...]
    completed_updates: int
    training_examples: int
    mistake_updates: int
    training_kernel_evaluations: int


def copy_kernel_perceptron_policy(
    policy: KernelPerceptronPolicy,
) -> KernelPerceptronPolicy:
    """Revalidate record tampering and own the fixed bounded policy."""
    _record(policy, KernelPerceptronPolicy)
    return replace(policy)


def _counters(snapshot: KernelPerceptronSnapshot) -> tuple[int, int, int, int]:
    values = (
        snapshot.completed_updates,
        snapshot.training_examples,
        snapshot.mistake_updates,
        snapshot.training_kernel_evaluations,
    )
    if any(type(value) is not int or not 0 <= value <= _MAX_COUNT for value in values):
        raise ValueError("kernel perceptron counters require bounded nonnegative native integers")
    updates, rows, mistakes, evaluations = values
    if not updates <= rows <= 128 * updates or mistakes > rows:
        raise ValueError("kernel perceptron passes, training rows, and mistakes disagree")
    if evaluations > snapshot.policy.max_prototypes * rows:
        raise ValueError("kernel evaluations exceed the bounded prototype count per training row")
    return values


def _prototypes(
    prototypes: tuple[tuple[float, ...], ...], policy: KernelPerceptronPolicy, mistakes: int
) -> tuple[tuple[float, ...], ...]:
    if type(prototypes) is not tuple or len(prototypes) > min(policy.max_prototypes, mistakes):
        raise ValueError(
            "kernel prototypes require an exact tuple bounded by capacity and mistakes"
        )
    owned = []
    for row in prototypes:
        if (
            type(row) is not tuple
            or len(row) != policy.feature_dim
            or any(
                type(value) is not float or not isfinite(value) or abs(value) > 1 for value in row
            )
        ):
            raise ValueError("kernel prototypes require finite native float rows in [-1, 1]")
        owned.append(tuple(value for value in row))
    if len(set(owned)) != len(owned):
        raise ValueError("kernel prototypes require unique numerical feature keys")
    return tuple(owned)


def _coefficients(
    coefficients: tuple[tuple[int, ...], ...], count: int, classes: int, mistakes: int
) -> tuple[tuple[int, ...], ...]:
    if type(coefficients) is not tuple or len(coefficients) != count:
        raise ValueError("kernel coefficients require one exact tuple per prototype")
    owned = []
    for row in coefficients:
        if (
            type(row) is not tuple
            or len(row) != classes
            or any(type(value) is not int or abs(value) > mistakes for value in row)
            or sum(row) != 0
        ):
            raise ValueError(
                "kernel coefficients require bounded native integers with row sum zero"
            )
        owned.append(tuple(value for value in row))
    if sum(abs(value) for row in owned for value in row) > 2 * mistakes:
        raise ValueError("kernel coefficient total magnitude exceeds twice the mistake count")
    return tuple(owned)


def copy_kernel_perceptron_snapshot(
    snapshot: KernelPerceptronSnapshot,
) -> KernelPerceptronSnapshot:
    """Validate and own the complete dictionary, coefficients, policy, and work."""
    _record(snapshot, KernelPerceptronSnapshot)
    policy = copy_kernel_perceptron_policy(snapshot.policy)
    updates, rows, mistakes, evaluations = _counters(snapshot)
    prototypes = _prototypes(snapshot.prototypes, policy, mistakes)
    coefficients = _coefficients(
        snapshot.coefficients, len(prototypes), policy.num_classes, mistakes
    )
    return KernelPerceptronSnapshot(
        policy, prototypes, coefficients, updates, rows, mistakes, evaluations
    )


def create_kernel_perceptron_snapshot(
    policy: KernelPerceptronPolicy,
) -> KernelPerceptronSnapshot:
    """Create the empty dictionary without consuming RNG or reserving entries."""
    return KernelPerceptronSnapshot(copy_kernel_perceptron_policy(policy), (), (), 0, 0, 0, 0)


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
            f"kernel features require finite exact float64 Bx{feature_dim} in [-1, 1], B in 1..128"
        )
    return features.copy()


def _logits(
    row: Array,
    prototypes: list[tuple[float, ...]] | tuple[tuple[float, ...], ...],
    coefficients: list[list[int]] | tuple[tuple[int, ...], ...],
    classes: int,
) -> Array:
    if not prototypes:
        return np.zeros(classes, dtype=np.float64)
    kernel = (1.0 + np.asarray(prototypes, dtype=np.float64) @ row / len(row)) ** 2
    return kernel @ np.asarray(coefficients, dtype=np.float64)


class KernelPerceptronCapacityExceeded(TrainingUpdateFailed):
    """A classified new mistake exceeded capacity; the whole pass is uncommitted.

    Work includes the refused row's prediction and its kernel pairs. Uncommitted
    mistakes count locally applied updates before refusal, excluding that row.
    No dictionary or coefficients are returned as authority in the failure.
    """

    def __init__(
        self,
        *,
        attempted_training_rows: int,
        training_rows_evaluated: int,
        kernel_evaluations_completed: int,
        mistake_updates_uncommitted: int,
        prototype_count_before: int,
        prototype_count_proposed: int,
    ) -> None:
        super().__init__(
            "kernel perceptron new mistaken feature key exceeds prototype capacity",
            KERNEL_PERCEPTRON_MISTAKES_DIAGNOSTIC,
            {
                "attempted_passes": 1,
                "attempted_training_rows": attempted_training_rows,
                "training_rows_evaluated": training_rows_evaluated,
                "kernel_evaluations_completed": kernel_evaluations_completed,
                "mistake_updates_uncommitted": mistake_updates_uncommitted,
                "prototype_count_before": prototype_count_before,
                "prototype_count_proposed": prototype_count_proposed,
                "native_pass_committed": False,
            },
        )


def update_kernel_perceptron(
    snapshot: KernelPerceptronSnapshot, features: Array, targets: Array
) -> KernelPerceptronSnapshot:
    """Return one ordered unit-mistake pass or refuse it atomically at capacity."""
    owned = copy_kernel_perceptron_snapshot(snapshot)
    batch = _features(features, owned.policy.feature_dim)
    rows = len(batch)
    validate_class_targets(targets, rows, owned.policy.num_classes)
    labels = targets[:, 0].astype(np.int64, copy=True)
    updates, examples = owned.completed_updates + 1, owned.training_examples + rows
    # Why: counter overflow is an admission failure, before any kernel work.
    maximum_pairs = min(owned.policy.max_prototypes, len(owned.prototypes) + rows) * rows
    if (
        updates > _MAX_COUNT
        or examples > _MAX_COUNT
        or owned.training_kernel_evaluations + maximum_pairs > _MAX_COUNT
    ):
        raise ValueError("kernel perceptron native counter overflow")
    prototypes = list(owned.prototypes)
    coefficients = [list(row) for row in owned.coefficients]
    indices = {prototype: index for index, prototype in enumerate(prototypes)}
    mistakes, pairs = 0, 0
    for evaluated, (row, label) in enumerate(zip(batch, labels, strict=True), start=1):
        pairs += len(prototypes)
        predicted = int(np.argmax(_logits(row, prototypes, coefficients, owned.policy.num_classes)))
        if predicted == label:
            continue
        key = tuple(float(value) for value in row)
        if key not in indices:
            if len(prototypes) == owned.policy.max_prototypes:
                raise KernelPerceptronCapacityExceeded(
                    attempted_training_rows=rows,
                    training_rows_evaluated=evaluated,
                    kernel_evaluations_completed=pairs,
                    mistake_updates_uncommitted=mistakes,
                    prototype_count_before=len(owned.prototypes),
                    prototype_count_proposed=len(prototypes) + 1,
                )
            indices[key] = len(prototypes)
            prototypes.append(key)
            coefficients.append([0] * owned.policy.num_classes)
        coefficient = coefficients[indices[key]]
        coefficient[int(label)] += 1
        coefficient[predicted] -= 1
        mistakes += 1
    return copy_kernel_perceptron_snapshot(
        KernelPerceptronSnapshot(
            owned.policy,
            tuple(prototypes),
            tuple(tuple(row) for row in coefficients),
            updates,
            examples,
            owned.mistake_updates + mistakes,
            owned.training_kernel_evaluations + pairs,
        )
    )


def predict_kernel_perceptron(snapshot: KernelPerceptronSnapshot, features: Array) -> Array:
    """Return stable softmax scores without changing state or native work counts."""
    owned = copy_kernel_perceptron_snapshot(snapshot)
    batch = _features(features, owned.policy.feature_dim)
    logits = np.stack(
        [
            _logits(row, owned.prototypes, owned.coefficients, owned.policy.num_classes)
            for row in batch
        ]
    )
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    validate_class_probabilities(probabilities, owned.policy.num_classes)
    return probabilities
