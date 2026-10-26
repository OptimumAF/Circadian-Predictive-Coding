"""Bounded descriptive shift signals with a frozen reference and persistence.

Signals summarize batch-weighted arrived data, not independent statistical tests
or evidence of a cause. No model, training policy, clock, IO or RNG belongs here.
"""

from dataclasses import dataclass, replace
from math import isfinite

import numpy as np

from src.core.categorical_probability_scores import (
    validate_class_count,
    validate_class_probabilities,
    validate_class_targets,
)
from src.core.coupled_energy import Array


@dataclass(frozen=True)
class ShiftMetrics:
    rows: int
    uncertainty: float
    feature_mean: tuple[float, ...]
    residual: float


@dataclass(frozen=True)
class ShiftPolicy:
    reference_batches: int = 3
    window_batches: int = 2
    persistence_batches: int = 2
    uncertainty_threshold: float = 0.05
    feature_threshold: float = 0.25
    residual_threshold: float = 0.05

    def __post_init__(self) -> None:
        for count in (self.reference_batches, self.window_batches, self.persistence_batches):
            if type(count) is not int or not 1 <= count <= 32:
                raise ValueError("shift policy batch counts must be native integers in 1..32")
        for threshold, upper in (
            (self.uncertainty_threshold, 1),
            (self.feature_threshold, 2),
            (self.residual_threshold, 1),
        ):
            _number(threshold, "shift threshold", 0, upper, positive=True)


@dataclass(frozen=True)
class ShiftDecision:
    batch_index: int
    phase: str
    uncertainty_change: float | None
    feature_drift: float | None
    residual_change: float | None
    streaks: tuple[int, int, int]
    alarms: tuple[bool, bool, bool]


def _number(value: float, name: str, lower: float, upper: float, *, positive: bool = False) -> None:
    if type(value) not in {int, float} or not lower <= value <= upper or not isfinite(value):
        raise ValueError(f"{name} must be a finite native number in [{lower}, {upper}]")
    if positive and value == lower:
        raise ValueError(f"{name} must be greater than {lower}")


def _validate_feature_shape(features: Array) -> None:
    if (
        type(features) is not np.ndarray
        or features.dtype != np.float64
        or features.ndim != 2
        or not 1 <= features.shape[0] <= 128
        or not 1 <= features.shape[1] <= 16
    ):
        raise ValueError("shift features must be a float64 matrix within 128 rows and 16 columns")


def _validate_arrays(features: Array, predictions: Array, targets: Array) -> None:
    _validate_feature_shape(features)
    for values in (predictions, targets):
        if (
            type(values) is not np.ndarray
            or values.dtype != np.float64
            or values.shape != (features.shape[0], 1)
        ):
            raise ValueError(
                "shift predictions and targets must be float64 matrices of shape (B, 1)"
            )
    # The complete metadata envelope precedes allocation of value-check arrays.
    for values, lower, upper in ((features, -1, 1), (predictions, 0, 1), (targets, 0, 1)):
        if not np.isfinite(values).all() or np.any((values < lower) | (values > upper)):
            raise ValueError("shift arrays must be finite within their declared value domains")


def summarize_shift_batch(features: Array, predictions: Array, targets: Array) -> ShiftMetrics:
    """Return normalized binary entropy, raw feature means and mean absolute residual.

    Entropy is exactly zero for probabilities 0 and 1. Features are bounded in
    [-1,1]; probabilities/targets in [0,1]. Returned records retain no arrays.
    """
    _validate_arrays(features, predictions, targets)
    interior = predictions[(predictions > 0) & (predictions < 1)]
    entropy = -interior * np.log(interior) - (1 - interior) * np.log1p(-interior)
    uncertainty = float(np.sum(entropy) / len(features) / np.log(2.0))
    # Roundoff at maximal entropy can exceed one; preserve the mathematical range.
    uncertainty = min(1.0, max(0.0, uncertainty))
    return ShiftMetrics(
        rows=len(features),
        uncertainty=uncertainty,
        feature_mean=tuple(float(value) for value in np.mean(features, axis=0)),
        residual=float(np.mean(np.abs(predictions - targets))),
    )


def summarize_categorical_shift_batch(
    features: Array, predictions: Array, targets: Array, num_classes: int
) -> ShiftMetrics:
    """Summarize arrived hard labels with H(p)/log(K) and mean(1-p_true).

    Reuse the existing probability tolerance without renormalizing rows. This
    separate entry point preserves binary soft-target semantics. It has no role
    or clock authority; callers must admit labels before supplying them here.
    """
    _validate_feature_shape(features)
    count = validate_class_count(num_classes)
    rows = len(features)
    # Why: bound all array shapes before the shared validators allocate checks.
    if (
        type(predictions) is not np.ndarray
        or predictions.dtype != np.float64
        or predictions.shape != (rows, count)
        or type(targets) is not np.ndarray
        or targets.dtype != np.float64
        or targets.shape != (rows, 1)
    ):
        raise ValueError(
            "categorical shift requires bounded float64 (B,K) probabilities and (B,1) targets"
        )
    if not np.isfinite(features).all() or np.any((features < -1) | (features > 1)):
        raise ValueError("shift features must be finite in [-1,1]")
    validate_class_probabilities(predictions, count)
    validate_class_targets(targets, rows, count)
    positive = predictions[predictions > 0]
    entropy = float(-np.sum(positive * np.log(positive)) / rows / np.log(count))
    indices = targets[:, 0].astype(np.int64)
    return ShiftMetrics(
        rows=rows,
        uncertainty=min(1.0, max(0.0, entropy)),
        feature_mean=tuple(float(value) for value in np.mean(features, axis=0)),
        residual=float(np.mean(1.0 - predictions[np.arange(rows), indices])),
    )


def _validated_metrics(metrics: ShiftMetrics) -> ShiftMetrics:
    if type(metrics) is not ShiftMetrics:
        raise ValueError("shift detector requires an exact ShiftMetrics record")
    if type(metrics.rows) is not int or not 1 <= metrics.rows <= 128:
        raise ValueError("shift metric rows must be a native integer in 1..128")
    if type(metrics.feature_mean) is not tuple or not 1 <= len(metrics.feature_mean) <= 16:
        raise ValueError("shift feature means must be a tuple with 1..16 columns")
    _number(metrics.uncertainty, "uncertainty", 0, 1)
    _number(metrics.residual, "residual", 0, 1)
    for value in metrics.feature_mean:
        _number(value, "feature mean", -1, 1)
    return ShiftMetrics(
        metrics.rows,
        float(metrics.uncertainty),
        tuple(map(float, metrics.feature_mean)),
        float(metrics.residual),
    )


def _average(metrics: tuple[ShiftMetrics, ...]) -> tuple[float, tuple[float, ...], float]:
    count = len(metrics)
    return (
        sum(item.uncertainty for item in metrics) / count,
        tuple(
            sum(item.feature_mean[column] for item in metrics) / count
            for column in range(len(metrics[0].feature_mean))
        ),
        sum(item.residual for item in metrics) / count,
    )


class ShiftDetector:
    """Keep only the first reference batches and a bounded post-reference window.

    Every batch has equal weight regardless of its row count. Channel order is
    uncertainty, feature drift, residual; only strictly greater scores extend a
    streak. Consecutive overlapping windows are descriptive, not independent.
    """

    def __init__(self, policy: ShiftPolicy = ShiftPolicy()) -> None:
        if type(policy) is not ShiftPolicy:
            raise ValueError("shift detector requires an exact ShiftPolicy")
        self._policy = replace(policy)
        self._reference: tuple[ShiftMetrics, ...] = ()
        self._window: tuple[ShiftMetrics, ...] = ()
        self._width: int | None = None
        self._count = 0
        self._streaks = (0, 0, 0)

    def observe(self, metrics: ShiftMetrics) -> ShiftDecision:
        metrics = _validated_metrics(metrics)
        if self._count >= 512:
            raise ValueError("shift detector supports at most 512 observations")
        if self._width is not None and len(metrics.feature_mean) != self._width:
            raise ValueError("shift metric feature width must remain constant")
        reference, window = self._reference, self._window
        if len(reference) < self._policy.reference_batches:
            reference += (metrics,)
            decision = ShiftDecision(
                self._count + 1, "reference", None, None, None, (0, 0, 0), (False, False, False)
            )
        else:
            window = (*window, metrics)[-self._policy.window_batches :]
            decision = self._decide(reference, window)
        # No state changes precede complete input validation and decision math.
        self._reference, self._window = reference, window
        self._width = len(metrics.feature_mean)
        self._count += 1
        self._streaks = decision.streaks
        return decision

    def _decide(
        self, reference: tuple[ShiftMetrics, ...], window: tuple[ShiftMetrics, ...]
    ) -> ShiftDecision:
        if len(window) < self._policy.window_batches:
            return ShiftDecision(
                self._count + 1, "window", None, None, None, (0, 0, 0), (False, False, False)
            )
        old_uncertainty, old_features, old_residual = _average(reference)
        uncertainty, features, residual = _average(window)
        scores = (
            uncertainty - old_uncertainty,
            sum(abs(a - b) for a, b in zip(features, old_features)) / len(features),
            residual - old_residual,
        )
        thresholds = (
            self._policy.uncertainty_threshold,
            self._policy.feature_threshold,
            self._policy.residual_threshold,
        )
        streaks = tuple(
            old + 1 if score > threshold else 0
            for old, score, threshold in zip(self._streaks, scores, thresholds)
        )
        fixed_streaks = (streaks[0], streaks[1], streaks[2])
        persistence = self._policy.persistence_batches
        alarms = (streaks[0] >= persistence, streaks[1] >= persistence, streaks[2] >= persistence)
        return ShiftDecision(self._count + 1, "monitor", *scores, fixed_streaks, alarms)
