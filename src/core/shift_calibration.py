"""Fit descriptive shift thresholds from one declared stationary training prefix.

Only prefix scores enter the fit. Algebraic reachability is reported without
retuning thresholds to force a reachable detector. This module neither verifies
stationarity nor establishes false-positive control, model identity or authority.
No IO, clock, RNG, model execution or training-policy feedback belongs here.
"""

from dataclasses import dataclass, replace
from math import isfinite

from src.core.shift_signals import (
    ShiftDecision,
    ShiftDetector,
    ShiftMetrics,
    ShiftPolicy,
    _average,
    _validated_metrics,
)


@dataclass(frozen=True)
class ShiftCalibration:
    prefix_batches: int = 6
    margin: float = 1e-6

    def __post_init__(self) -> None:
        if type(self.prefix_batches) is not int or not 1 <= self.prefix_batches <= 32:
            raise ValueError("calibration prefix_batches must be a native integer in 1..32")
        if type(self.margin) not in {int, float} or not 0 < self.margin <= 1:
            raise ValueError("calibration margin must be a finite native number in (0, 1]")
        if not isfinite(self.margin):
            raise ValueError("calibration margin must be finite")


@dataclass(frozen=True)
class CalibrationResult:
    policy: ShiftPolicy
    reference_uncertainty: float
    reference_feature_mean: tuple[float, ...]
    reference_residual: float
    score_ceilings: tuple[float, float, float]
    reachable: tuple[bool, bool, bool]
    calibration_batches: int
    calibration_rows: int
    calibration_windows: int
    replayed_observations: int


def _configurations(
    policy: ShiftPolicy, calibration: ShiftCalibration
) -> tuple[ShiftPolicy, ShiftCalibration]:
    if type(policy) is not ShiftPolicy or type(calibration) is not ShiftCalibration:
        raise ValueError("calibration requires exact ShiftPolicy and ShiftCalibration records")
    policy, calibration = replace(policy), replace(calibration)
    if calibration.prefix_batches < policy.reference_batches + policy.window_batches:
        raise ValueError(
            "calibration prefix must include the reference and a full following window"
        )
    return policy, calibration


def _fit(
    policy: ShiftPolicy,
    calibration: ShiftCalibration,
    prefix: tuple[ShiftMetrics, ...],
    maxima: tuple[float, float, float],
    windows: int,
) -> tuple[ShiftDetector, CalibrationResult]:
    thresholds = (
        min(1.0, maxima[0] + calibration.margin),
        min(2.0, maxima[1] + calibration.margin),
        min(1.0, maxima[2] + calibration.margin),
    )
    fitted = replace(
        policy,
        uncertainty_threshold=thresholds[0],
        feature_threshold=thresholds[1],
        residual_threshold=thresholds[2],
    )
    uncertainty, features, residual = _average(prefix[: policy.reference_batches])
    ceilings = (
        1.0 - uncertainty,
        sum(1.0 + abs(value) for value in features) / len(features),
        1.0 - residual,
    )
    reachable = tuple(threshold < ceiling for threshold, ceiling in zip(thresholds, ceilings))
    result = CalibrationResult(
        policy=replace(fitted),
        reference_uncertainty=uncertainty,
        reference_feature_mean=features,
        reference_residual=residual,
        score_ceilings=ceilings,
        reachable=(reachable[0], reachable[1], reachable[2]),
        calibration_batches=len(prefix),
        calibration_rows=sum(metrics.rows for metrics in prefix),
        calibration_windows=windows,
        replayed_observations=len(prefix),
    )
    # Why replay: retain the existing detector's exact reference/window/count
    # semantics. Strict fitted thresholds leave all prefix streaks at zero.
    detector = ShiftDetector(fitted)
    for metrics in prefix:
        detector.observe(metrics)
    return detector, result


class CalibratedShiftDetector:
    """Calibrate once on a bounded prefix, then monitor without rebasing.

    Prefix outputs expose scores but suppress alarms and streaks. All thresholds
    are maximum nonnegative full-window prefix scores plus the fixed margin,
    capped only by channel domains (1,2,1), never by observed reference headroom.
    Ceilings are algebraic limits, not guaranteed achievable distributions.
    """

    def __init__(
        self,
        policy: ShiftPolicy = ShiftPolicy(),
        calibration: ShiftCalibration = ShiftCalibration(),
    ) -> None:
        self._policy, self._calibration = _configurations(policy, calibration)
        self._detector = ShiftDetector(self._policy)
        self._prefix: tuple[ShiftMetrics, ...] = ()
        self._maxima = (0.0, 0.0, 0.0)
        self._windows = 0
        self._count = 0
        self._result: CalibrationResult | None = None

    @property
    def calibration_result(self) -> CalibrationResult | None:
        """Return detached frozen data; callers cannot alter the fitted policy."""
        _configurations(self._policy, self._calibration)
        if self._result is None:
            return None
        return replace(self._result, policy=replace(self._result.policy))

    def observe(self, metrics: ShiftMetrics) -> ShiftDecision:
        policy, calibration = _configurations(self._policy, self._calibration)
        metrics = _validated_metrics(metrics)
        if self._count >= 512:
            raise ValueError("calibrated shift detector supports at most 512 observations")
        decision = self._detector.observe(metrics)
        if self._result is not None:
            self._count += 1
            return decision
        prefix = (*self._prefix, metrics)
        maxima, windows = self._maxima, self._windows
        if decision.phase == "monitor":
            assert decision.uncertainty_change is not None
            assert decision.feature_drift is not None
            assert decision.residual_change is not None
            maxima = (
                max(maxima[0], decision.uncertainty_change),
                max(maxima[1], decision.feature_drift),
                max(maxima[2], decision.residual_change),
            )
            windows += 1
        if len(prefix) == calibration.prefix_batches:
            detector, result = _fit(policy, calibration, prefix, maxima, windows)
            self._detector, self._result = detector, result
            self._prefix = ()
        else:
            self._prefix = prefix
        self._maxima, self._windows = maxima, windows
        self._count += 1
        return replace(
            decision, phase="calibration", streaks=(0, 0, 0), alarms=(False, False, False)
        )
