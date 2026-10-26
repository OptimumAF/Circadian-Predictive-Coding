"""Bounded decision/class detail from already returned reserved predictions.

Inputs are float64 probabilities and labels. Outputs contain only detached Python
primitives, with absent support explicitly unknown. No learner, scoring admission,
model ranking, normalization, RNG, or IO lives here.
"""

from typing import Any

import numpy as np

from src.core.categorical_probability_scores import (
    validate_class_count,
    validate_class_probabilities,
    validate_class_targets,
)
from src.core.coupled_energy import Array


def _envelope(predictions: Array, targets: Array, num_classes: int) -> tuple[int, int]:
    count = validate_class_count(num_classes)
    width = 1 if count == 2 else count
    if (
        type(predictions) is not np.ndarray
        or type(targets) is not np.ndarray
        or predictions.dtype != np.dtype(np.float64)
        or targets.dtype != np.dtype(np.float64)
        or predictions.ndim != 2
        or targets.ndim != 2
        or not 1 <= predictions.shape[0] <= 128
        or predictions.shape[1] != width
        or targets.shape != (predictions.shape[0], 1)
    ):
        raise ValueError(
            f"diagnostics require exact float64 predictions (B, {width}) and targets (B, 1), B1..128"
        )
    return predictions.shape[0], count


def _classifications(predictions: Array, targets: Array, count: int) -> tuple[Array, Array, Array]:
    if count == 2:
        if (
            not np.isfinite(predictions).all()
            or not np.isfinite(targets).all()
            or np.any((predictions < 0) | (predictions > 1))
            or np.any((targets < 0) | (targets > 1))
        ):
            raise ValueError(
                "binary diagnostics require finite probabilities and targets in [0, 1]"
            )
        # Why: preserve original binary clipping and log1p semantics, including
        # soft targets and inclusive threshold ties; no two-class argmax repair.
        clipped = np.clip(predictions[:, 0], 1e-12, 1 - 1e-12)
        labels = targets[:, 0]
        losses = -(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
        return (labels >= 0.5).astype(float), (predictions[:, 0] >= 0.5).astype(float), losses
    validate_class_probabilities(predictions, count)
    validate_class_targets(targets, predictions.shape[0], count)
    true_classes = targets[:, 0]
    selected = predictions[np.arange(len(targets)), true_classes.astype(np.int64)]
    losses = -np.log(np.clip(selected, 1e-12, 1.0))
    return true_classes, np.argmax(predictions, axis=1).astype(float), losses


def _counts(
    targets: Array, true_classes: Array, predicted_classes: Array, losses: Array, count: int
) -> dict[str, Any]:
    class_counts = [0] * count
    predicted_counts = [0] * count
    confusion = [[0] * count for _ in range(count)]
    loss_sums = [0.0] * count
    decisions = []
    for index in range(len(targets)):
        target_class, predicted_class = int(true_classes[index]), int(predicted_classes[index])
        loss = float(losses[index])
        class_counts[target_class] += 1
        predicted_counts[predicted_class] += 1
        confusion[target_class][predicted_class] += 1
        loss_sums[target_class] += loss
        decisions.append(
            {
                "index": index,
                "target": float(targets[index, 0]),
                "target_class": target_class,
                "predicted_class": predicted_class,
                "loss": loss,
                "correct": target_class == predicted_class,
            }
        )
    return {
        "class_counts": class_counts,
        "predicted_class_counts": predicted_counts,
        "confusion": confusion,
        "per_class": [
            {
                "class_index": index,
                "support": support,
                "predicted_count": predicted_counts[index],
                "correct": confusion[index][index],
                "accuracy": confusion[index][index] / support if support else None,
                "mean_loss": loss_sums[index] / support if support else None,
            }
            for index, support in enumerate(class_counts)
        ],
        "decisions": decisions,
    }


def decision_class_diagnostics(
    predictions: Array, targets: Array, num_classes: int
) -> dict[str, Any]:
    """Describe bounded decisions without adding an aggregate selection metric."""
    rows, count = _envelope(predictions, targets, num_classes)
    true_classes, predicted_classes, losses = _classifications(predictions, targets, count)
    return {
        "num_classes": count,
        "rows": rows,
        "loss_definition": "binary_cross_entropy" if count == 2 else "categorical_cross_entropy",
        "target_class_rule": "threshold_ge_half" if count == 2 else "hard_class_index",
        **_counts(targets, true_classes, predicted_classes, losses, count),
    }
