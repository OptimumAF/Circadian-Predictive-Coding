"""Frequency bounds for predictions that depend only on an exact input row.

Inputs are finite float64 feature rows and equal-weight hard categorical labels.
Outputs are owned scalar bounds on deterministic row accuracy and normalized
simplex CE. No role admission, model fitting, evaluation selection, RNG or IO.
"""

from dataclasses import dataclass
from math import fsum, log

import numpy as np
from numpy.typing import NDArray

from src.core.categorical_probability_scores import validate_class_count, validate_class_targets


@dataclass(frozen=True, slots=True)
class CategoricalInputBounds:
    rows: int
    feature_dim: int
    num_classes: int
    unique_input_keys: int
    conflicting_input_keys: int
    conflicting_rows: int
    max_correct_rows: int
    min_incorrect_rows: int
    row_accuracy_upper_bound: float
    normalized_ce_lower_bound: float


def categorical_input_bounds(
    features: NDArray[np.float64], targets: NDArray[np.float64], num_classes: int
) -> CategoricalInputBounds:
    """Count exact equal-input label conflicts without tolerance or normalization.

    Why: equal inputs must share predictions, so each key's modal count bounds
    correct rows; its empirical label entropy minimizes equal-row normalized CE.
    Float equality merges signed zeros and preserves every other distinct finite
    coordinate. These bounds do not establish finite-model attainability, hidden
    capacity, convergence, episode success or generalization. The CE floor assumes
    an exactly normalized simplex, not the score validator's1e-6 roundoff allowance.
    Every positive empirical frequency is at least1/128, above scoring's1e-12 clip.
    """
    count = validate_class_count(num_classes)
    if (
        type(features) is not np.ndarray
        or features.dtype != np.dtype(np.float64)
        or features.ndim != 2
        or not 1 <= features.shape[0] <= 128
        or not 1 <= features.shape[1] <= 296
        or not np.all(np.isfinite(features))
        or np.any(features < -1)
        or np.any(features > 1)
    ):
        raise ValueError("features must be finite float64 matrices (B1..128, D1..296) in[-1,1]")
    rows, width = features.shape
    validate_class_targets(targets, rows, count)
    groups: dict[tuple[float, ...], list[int]] = {}
    for row, target in zip(features, targets[:, 0], strict=True):
        counts = groups.setdefault(tuple(float(value) for value in row), [0] * count)
        counts[int(target)] += 1
    max_correct = sum(max(counts) for counts in groups.values())
    conflicts = [counts for counts in groups.values() if sum(value > 0 for value in counts) > 1]
    entropy_sum = fsum(
        -frequency * log(frequency / sum(counts))
        for counts in groups.values()
        for frequency in counts
        if frequency
    )
    return CategoricalInputBounds(
        rows=rows,
        feature_dim=width,
        num_classes=count,
        unique_input_keys=len(groups),
        conflicting_input_keys=len(conflicts),
        conflicting_rows=sum(sum(counts) for counts in conflicts),
        max_correct_rows=max_correct,
        min_incorrect_rows=rows - max_correct,
        row_accuracy_upper_bound=max_correct / rows,
        normalized_ce_lower_bound=entropy_sum / rows,
    )
