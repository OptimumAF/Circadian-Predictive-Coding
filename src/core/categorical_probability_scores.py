"""Validate categorical probabilities and compute final classification scores.

Inputs are owned-by-caller float64 probability/label arrays and a declared action
count. Outputs are scalar cross-entropy and accuracy. No normalization, learner,
selection, RNG, or IO is performed; binary metric semantics remain separate.
"""

import numpy as np
from numpy.typing import NDArray


def validate_class_count(num_classes: int) -> int:
    """Return a declared native action count, rejecting bools and coercion."""
    if type(num_classes) is not int or not 2 <= num_classes <= 16:
        raise ValueError("num_classes must be a native integer in 2..16")
    return num_classes


def validate_class_targets(targets: NDArray[np.float64], rows: int, num_classes: int) -> None:
    """Require one finite hard class index for each positive batch row."""
    count = validate_class_count(num_classes)
    if type(rows) is not int or rows < 1:
        raise ValueError("rows must be a positive native integer")
    if (
        type(targets) is not np.ndarray
        or targets.dtype != np.dtype(np.float64)
        or targets.shape != (rows, 1)
        or not np.all(np.isfinite(targets))
        or np.any(targets != np.floor(targets))
        or np.any(targets < 0)
        or np.any(targets >= count)
    ):
        raise ValueError(
            f"targets must be finite float64 hard class indices of shape ({rows}, 1) in 0..{count - 1}"
        )


def validate_class_probabilities(predictions: NDArray[np.float64], num_classes: int) -> None:
    """Validate without targets or repair, including at live policy boundaries.

    The unchanged absolute tolerance admits native float32 softmax roundoff.
    """
    count = validate_class_count(num_classes)
    if (
        type(predictions) is not np.ndarray
        or predictions.dtype != np.dtype(np.float64)
        or predictions.ndim != 2
        or predictions.shape[0] < 1
        or predictions.shape[1] != count
        or not np.all(np.isfinite(predictions))
        or np.any(predictions < 0)
        or np.any(predictions > 1)
        or not np.allclose(predictions.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
    ):
        raise ValueError(
            f"predictions must be finite float64 probabilities of shape (B, {count}) "
            "with positive B and row sums within absolute 1e-6 of one"
        )


def probability_scores(
    predictions: NDArray[np.float64], targets: NDArray[np.float64], num_classes: int
) -> dict[str, float]:
    """Score without repairing probabilities; argmax ties choose the first class."""
    validate_class_probabilities(predictions, num_classes)
    count = validate_class_count(num_classes)
    rows = predictions.shape[0]
    validate_class_targets(targets, rows, count)
    indices = targets[:, 0].astype(np.int64)
    true_probabilities = predictions[np.arange(rows), indices]
    return {
        "ce": float(-np.mean(np.log(np.clip(true_probabilities, 1e-12, 1.0)))),
        "accuracy": float(np.mean(np.argmax(predictions, axis=1) == indices)),
    }
