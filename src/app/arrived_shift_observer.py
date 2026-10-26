"""Observe arrived batches retrospectively using copied previous parameters.

The callback runs after update k with parameters from k-1. It never receives
final evaluation data, trains a model, or feeds signals back into training. The
report retains bounded scalar rows, including failures and unreached slots.
The default two-class mode preserves binary soft targets. Explicit K>=3 uses
bounded categorical probabilities and hard labels; reports name those metrics.
Arrival authority belongs to the existing stream callback, not arbitrary calls
to this observer. Delegates own/count any actual model prediction work.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from math import isfinite
from time import monotonic
from typing import Any, Callable

import numpy as np

from src.app.coupled_attribution import AttributionBatch, _copy_batch, _validate_batch
from src.core.coupled_energy import Array
from src.core.categorical_probability_scores import validate_class_count
from src.core.shift_calibration import CalibratedShiftDetector, ShiftCalibration
from src.core.shift_incidents import ShiftIncidentTracker
from src.core.shift_signals import (
    ShiftDetector,
    ShiftPolicy,
    summarize_categorical_shift_batch,
    summarize_shift_batch,
)


def _time() -> float:
    value = monotonic()
    if type(value) not in {int, float} or not isfinite(value):
        raise ValueError("shift observer time must be a finite native number")
    return float(value)


class ArrivedShiftObserver:
    def __init__(
        self,
        predict_state: Callable[[Any, Array], Array],
        *,
        policy: ShiftPolicy = ShiftPolicy(),
        max_updates: int = 31,
        calibration: ShiftCalibration | None = None,
        calibration_model_id: str | None = None,
        incidents: bool = False,
        num_classes: int = 2,
    ) -> None:
        if not callable(predict_state):
            raise ValueError("predict_state must be callable")
        if type(max_updates) is not int or not 1 <= max_updates <= 31:
            raise ValueError("shift observer max_updates must be a native integer in 1..31")
        if type(incidents) is not bool:
            raise ValueError("shift observer incidents mode must be boolean")
        self._num_classes = validate_class_count(num_classes)
        self._incidents = ShiftIncidentTracker() if incidents else None
        self._calibration: ShiftCalibration | None = None
        self._calibration_model_id = calibration_model_id
        self._detector: ShiftDetector | CalibratedShiftDetector
        if calibration is None:
            if calibration_model_id is not None:
                raise ValueError("calibration model identity requires calibration")
            self._detector = ShiftDetector(policy)
        else:
            if type(calibration) is not ShiftCalibration:
                raise ValueError("calibration requires an exact ShiftCalibration")
            calibration = replace(calibration)
            if (
                type(calibration_model_id) is not str
                or not calibration_model_id.strip()
                or len(calibration_model_id.encode("utf8")) > 128
            ):
                raise ValueError("calibration model identity must be nonempty and <=128 UTF8 bytes")
            if calibration.prefix_batches >= max_updates:
                raise ValueError("calibration prefix must leave at least one monitoring update")
            self._detector = CalibratedShiftDetector(policy=policy, calibration=calibration)
            self._calibration = calibration
        self._predict_state = predict_state
        self._max_updates = max_updates
        self._previous: Any = None
        self._completed = 0
        self._width: int | None = None
        self._seen_rows: set[str] = set()
        self._primed = False
        self._stopped = False
        self._reason: str | None = None
        self._initial_seconds: float | None = None
        self._total_seconds = 0.0
        self._last_time: float | None = None
        self._attempted = 0
        self._predicted = 0
        self._prediction_rows_attempted = 0
        self._prediction_rows_completed = 0
        self._rows: list[dict[str, Any]] = [
            dict(
                updates=k,
                prediction_state_updates=k - 1,
                status="not_run",
                reason="observation not reached",
                metrics=None,
                decision=None,
                elapsed_seconds=None,
            )
            for k in range(1, max_updates + 1)
        ]
        if calibration is not None:
            for row in self._rows:
                row["arrived_row_ids"] = None
        if incidents:
            for row in self._rows:
                row["incident"] = None

    def _require_call(self, updates: int, arrived: AttributionBatch | None) -> None:
        if self._stopped:
            raise ValueError("shift observer is stopped")
        expected = self._completed + 1 if self._primed else 0
        if type(updates) is not int or updates != expected or updates > self._max_updates:
            raise ValueError(
                "shift observer requires consecutive native update counts starting at zero"
            )
        if not self._primed:
            if arrived is not None:
                raise ValueError("initial shift observation must not contain an arrived batch")
        elif (
            type(arrived) is not AttributionBatch
            or type(arrived.role) is not str
            or arrived.role != "train"
        ):
            raise ValueError("shift observer accepts only exact arrived training batches")

    def _validate_arrived(self, batch: AttributionBatch) -> set[str]:
        if (
            type(batch.features) is not np.ndarray
            or batch.features.dtype != np.float64
            or batch.features.ndim != 2
            or not 1 <= batch.features.shape[0] <= 128
            or not 1 <= batch.features.shape[1] <= 16
            or type(batch.targets) is not np.ndarray
            or batch.targets.dtype != np.float64
            or batch.targets.shape != (batch.features.shape[0], 1)
        ):
            raise ValueError(
                "arrived shift data must be float64 matrices <=128x16 with targets (B, 1)"
            )
        if self._width is not None and batch.features.shape[1] != self._width:
            raise ValueError("arrived shift feature width must remain constant")
        if batch.num_classes != self._num_classes:
            raise ValueError("arrived shift class count differs from the declared observer")
        seen = self._seen_rows.copy()
        _validate_batch(batch, "train", seen)
        if np.any((batch.features < -1) | (batch.features > 1)):
            raise ValueError("arrived shift features must be in [-1, 1]")
        return seen

    def _finish_time(self, started: float) -> float:
        ended = _time()
        if ended < started or not isfinite(ended - started):
            raise ValueError("shift observer time must be nondecreasing with finite durations")
        seconds = ended - started
        total = self._total_seconds + seconds
        if not isfinite(total):
            raise ValueError("shift observer total duration must remain finite")
        self._last_time = ended
        self._total_seconds = total
        return seconds

    def __call__(self, updates: int, state: Any, arrived: AttributionBatch | None) -> None:
        self._require_call(updates, arrived)
        started = _time()
        if self._last_time is not None and started < self._last_time:
            raise ValueError("shift observer time must be nondecreasing")
        seen = None if arrived is None else self._validate_arrived(arrived)
        row = None if arrived is None else self._rows[updates - 1]
        # All public preflight has passed: subsequent failure is admitted work.
        try:
            if row is not None and self._calibration is not None:
                row["arrived_row_ids"] = tuple(arrived.row_ids) if arrived is not None else None
            current = deepcopy(state)
            if arrived is None:
                self._initial_seconds = self._finish_time(started)
                self._previous, self._primed = current, True
                return
            batch = _copy_batch(arrived)
            previous = deepcopy(self._previous)
            features = batch.features.copy()
            self._attempted += 1
            self._prediction_rows_attempted += len(features)
            predictions = self._predict_state(previous, features)
            self._predicted += 1
            self._prediction_rows_completed += len(features)
            metrics = (
                summarize_shift_batch(batch.features, predictions, batch.targets)
                if self._num_classes == 2
                else summarize_categorical_shift_batch(
                    batch.features, predictions, batch.targets, self._num_classes
                )
            )
            assert row is not None
            row["metrics"] = asdict(metrics)
            decision = self._detector.observe(metrics)
            row["decision"] = asdict(decision)
            if self._incidents is not None:
                row["incident"] = asdict(self._incidents.observe(decision))
            row["elapsed_seconds"] = self._finish_time(started)
            row.update(status="complete", reason=None)
            self._previous, self._completed = current, updates
            assert seen is not None
            self._seen_rows = seen
            self._width = batch.features.shape[1]
        except Exception as error:
            self._stopped = True
            self._reason = f"{type(error).__name__}: {error}"
            try:
                seconds = self._finish_time(started)
            except Exception:
                seconds = None
            if row is None:
                self._initial_seconds = seconds
            else:
                row.update(status="failed", reason=self._reason, elapsed_seconds=seconds)
            raise

    def report(self) -> dict[str, Any]:
        """Return detached rows; total_seconds sums known measured durations only."""
        report: dict[str, Any] = dict(
            primed=self._primed,
            stopped=self._stopped,
            reason=self._reason,
            initial_seconds=self._initial_seconds,
            total_seconds=self._total_seconds,
            prediction_calls_attempted=self._attempted,
            prediction_calls_completed=self._predicted,
            rows=self._rows,
        )
        if self._num_classes > 2:
            report["prediction_schema"] = dict(
                num_classes=self._num_classes,
                uncertainty="mean_entropy_divided_by_log_num_classes",
                residual="mean_one_minus_true_class_probability",
                prediction_rows_attempted=self._prediction_rows_attempted,
                prediction_rows_completed=self._prediction_rows_completed,
            )
        if self._calibration is not None:
            assert isinstance(self._detector, CalibratedShiftDetector)
            result = self._detector.calibration_result
            report["calibration"] = dict(
                configuration=asdict(self._calibration),
                model_id=self._calibration_model_id,
                identity_kind="caller_declared_run_and_parameter_update_indices",
                result=None if result is None else asdict(result),
            )
        if self._incidents is not None:
            report["incident_semantics"] = dict(
                channels=("uncertainty", "feature", "residual"),
                onset="first alarm after a non-alarm; start index links continuation and clear",
                cleared="first non-alarm after an active incident; no additional cooldown",
                feedback=False,
            )
        return deepcopy(report)
