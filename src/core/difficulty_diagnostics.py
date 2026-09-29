"""Pure train-batch diagnostics for a fixed modulation comparison.

Inputs are current feedforward probabilities and observed binary labels.
Outputs describe that batch only; they never choose an update or inspect
decision/final roles. The model's relaxed-state scale is separate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainBatchSignals:
    mean_absolute_error: float
    clipped_error: float
    cross_entropy: float


def measure_train_batch(probabilities: np.ndarray, targets: np.ndarray) -> TrainBatchSignals:
    """Measure BCE and the predeclared 0.5 per-row absolute-error clip."""
    if (
        not isinstance(probabilities, np.ndarray)
        or not isinstance(targets, np.ndarray)
        or probabilities.shape != targets.shape
        or probabilities.ndim != 2
        or probabilities.shape[1] != 1
        or probabilities.shape[0] == 0
        or not np.all(np.isfinite(probabilities))
        or not np.all((probabilities >= 0.0) & (probabilities <= 1.0))
        or not np.all(np.isin(targets, (0.0, 1.0)))
    ):
        raise ValueError("train diagnostics require finite binary labels and probabilities")
    error = np.abs(probabilities - targets)
    clipped = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    bce = -targets * np.log(clipped) - (1.0 - targets) * np.log(1.0 - clipped)
    return TrainBatchSignals(
        mean_absolute_error=float(np.mean(error)),
        clipped_error=float(np.mean(np.minimum(error, 0.5))),
        cross_entropy=float(np.mean(bce)),
    )
