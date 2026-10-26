"""Describe target-versus-competitor probability gaps from already paid rows.

Inputs are exact float64 categorical probabilities and hard labels. Outputs are
detached JSON primitives for every row and class, including absent support.
This reducer has no role admission, learner, selection, RNG, normalization or IO.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.core.categorical_probability_scores import (
    validate_class_probabilities,
    validate_class_targets,
)

DEFINITION = "categorical_target_probability_margin_v1"


def _summary(margins: NDArray[np.float64], correct: NDArray[np.bool_]) -> dict[str, Any]:
    return dict(
        mean_margin=float(np.mean(margins)) if len(margins) else None,
        min_margin=float(np.min(margins)) if len(margins) else None,
        max_margin=float(np.max(margins)) if len(margins) else None,
        positive_rows=int(np.count_nonzero(margins > 0)),
        zero_rows=int(np.count_nonzero(margins == 0)),
        negative_rows=int(np.count_nonzero(margins < 0)),
        argmax_correct_rows=int(np.count_nonzero(correct)),
    )


def probability_margins(
    predictions: NDArray[np.float64], targets: NDArray[np.float64], num_classes: int
) -> dict[str, Any]:
    """Keep raw signed gaps; a zero gap is a tie, not necessarily an error.

    Why: CE/accuracy can hide which rare action decisions remain low confidence.
    These gaps describe predictions; they do not diagnose representational or
    optimization causes, and do not replace the task's frozen primary metrics.
    """
    validate_class_probabilities(predictions, num_classes)
    rows = predictions.shape[0]
    validate_class_targets(targets, rows, num_classes)
    indices = targets[:, 0].astype(np.int64)
    competitors = predictions.copy()
    competitors[np.arange(rows), indices] = -np.inf
    margins = predictions[np.arange(rows), indices] - np.max(competitors, axis=1)
    correct = np.argmax(predictions, axis=1) == indices
    return dict(
        definition=DEFINITION,
        num_classes=num_classes,
        rows=rows,
        target_margins=margins.tolist(),
        **_summary(margins, correct),
        per_class=[
            dict(
                class_index=index,
                rows=int(np.count_nonzero(indices == index)),
                **_summary(margins[indices == index], correct[indices == index]),
            )
            for index in range(num_classes)
        ],
    )
