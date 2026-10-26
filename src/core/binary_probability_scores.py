"""Pure common binary probability metrics, independent of stream orchestration.

Callers own target/data-role validation and scoring admission. This preserves
the existing clipping/threshold convention without importing a stream runner.
No training, selection, model access, role release or resource accounting occurs.
"""

import numpy as np

from src.core.coupled_energy import Array


def probability_scores(predictions: Array, targets: Array) -> dict[str, float]:
    if (
        type(predictions) is not np.ndarray
        or predictions.dtype != np.float64
        or predictions.ndim != 2
        or min(predictions.shape) < 1
        or not np.isfinite(predictions).all()
    ):
        raise ValueError("heldout predictions must be a nonempty finite float64 matrix")
    if predictions.shape != targets.shape or np.any((predictions < 0) | (predictions > 1)):
        raise ValueError("heldout predictions must have shape (batch, 1) and values in [0, 1]")
    clipped = np.clip(predictions, 1e-12, 1 - 1e-12)
    bce = -float(np.mean(targets * np.log(clipped) + (1 - targets) * np.log1p(-clipped)))
    accuracy = float(np.mean((predictions >= 0.5) == (targets >= 0.5)))
    return {"bce": bce, "accuracy": accuracy}
