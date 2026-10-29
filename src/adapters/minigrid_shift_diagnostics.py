"""Readonly arrived MiniGrid signals: full native predictions, eight public summaries.

Consumes one admitted TRAIN packet/current detached candidate per arrival. Emits
bounded scalar diagnostics/costs only; no policy, fitting, data release or IO.
"""

from copy import deepcopy
from dataclasses import asdict
from math import isfinite
from time import monotonic

import numpy as np

from src.adapters.minigrid_context_linear_learner import (
    ContextLinearLearner,
    copy_context_linear_training_snapshot,
)
from src.app.coupled_attribution import AttributionBatch, _same_value, _validate_batch
from src.core.actor_ports import CandidateState
from src.core.shift_signals import ShiftDetector, ShiftPolicy, summarize_categorical_shift_batch
from src.core.shift_incidents import ShiftIncidentTracker


def _validate_arrivals(max_arrivals):
    if type(max_arrivals) is not int or not 1 <= max_arrivals <= 64:
        raise ValueError("shift diagnostics max_arrivals must be an integer in [1,64]")


def describe(*, max_arrivals=6):
    _validate_arrivals(max_arrivals)
    return dict(
        definition="minigrid_arrived_native_shift_v1",
        arrivals=max_arrivals,
        rows_per_arrival=9,
        prediction_features=296,
        summary_features=8,
        num_classes=7,
        feature_summary="initial/current frame: public image channel means (object,color,state), direction",
        prediction_clock="current candidate before this arrival's reference decision/fit",
        policy=asdict(ShiftPolicy()),
        feedback=False,
        scope="newly arrived TRAIN rows only, not joint replay/probe/final; descriptive signals, no deployability claim",
    )


def public_feature_summary(features):
    if (
        type(features) is not np.ndarray
        or features.dtype != np.float64
        or features.ndim != 2
        or not 1 <= len(features) <= 128
        or features.shape[1] != 296
        or not np.isfinite(features).all()
        or np.any(np.abs(features) > 1)
    ):
        raise ValueError(
            "MiniGrid shift inputs require finite float64 public history (B,296) in [-1,1]"
        )
    # Why: retain the existing detector's <=16-column contract without choosing
    # coordinates from labels/context names or changing native prediction inputs.
    result = np.empty((len(features), 8), dtype=np.float64)
    for frame in range(2):
        start = frame * 148
        result[:, frame * 4 : frame * 4 + 3] = (
            features[:, start : start + 147].reshape(-1, 49, 3).mean(axis=1)
        )
        result[:, frame * 4 + 3] = features[:, start + 147]
    return result


class MiniGridShiftDiagnostics:
    def __init__(self, *, max_arrivals=6):
        # Why: finite longer schedules need enough arrivals after reference
        # collection; this changes storage capacity, never detector thresholds.
        _validate_arrivals(max_arrivals)
        self._reader = ContextLinearLearner()
        self._detector = ShiftDetector()
        self._incidents = ShiftIncidentTracker()
        self._rows = [
            dict(arrival_index=i, status="not_run", reason="not reached")
            for i in range(max_arrivals)
        ]
        self._completed = self._attempts = self._returns = self._summaries = 0
        self._summary_returns = 0
        self._array_bytes = self._peak_numeric_bytes = 0
        self._seconds = 0.0
        self._last_time = None
        self._stopped = False

    def observe(self, tick, index, batch, candidate):
        if (
            self._stopped
            or type(index) is not int
            or index != self._completed
            or index >= len(self._rows)
            or type(tick) is not int
            or tick != index + 1
            or type(batch) is not AttributionBatch
            or batch.role != "train"
            or batch.num_classes != 7
            or type(batch.features) is not np.ndarray
            or type(batch.targets) is not np.ndarray
            or batch.features.shape != (9, 296)
            or batch.targets.shape != (9, 1)
            or type(candidate) is not CandidateState
        ):
            raise ValueError(
                "shift diagnostics require the next delay-one TRAIN arrival/current candidate"
            )
        _validate_batch(batch, "train", set())
        state = copy_context_linear_training_snapshot(candidate.state)
        if (
            state.failed_work is not None
            or state.completed_updates != candidate.wake_updates
            or not 0 <= state.completed_updates <= index
            or candidate.consolidation_attempts != 0
            or candidate.consolidations_completed != 0
        ):
            raise ValueError("shift diagnostics require an unfailed current native wake checkpoint")
        row = self._rows[index]
        started = monotonic()
        if not isfinite(started) or (self._last_time is not None and started < self._last_time):
            self._stopped = True
            row.update(status="failed", reason="invalid diagnostic start time", seconds=None)
            raise ValueError("shift diagnostic clock must be finite and nondecreasing")
        row.update(
            tick=tick,
            candidate_updates=state.completed_updates,
            candidate_version=candidate.learner_version,
            row_ids=batch.row_ids,
            status="running",
            reason=None,
        )
        succeeded = False
        try:
            x, y = batch.features.copy(), batch.targets.copy()
            self._array_bytes += x.nbytes + y.nbytes
            saved_x = x.copy()
            self._array_bytes += saved_x.nbytes
            self._reader.restore_state(state)
            numeric = sum(
                8 * (sum(len(r) for r in h.coefficients) + len(h.singular_values) + 1)
                for h in state.heads
                if h is not None
            )
            self._peak_numeric_bytes = max(self._peak_numeric_bytes, numeric)
            self._attempts += 1
            predictions = self._reader.predict(x)
            self._returns += 1
            if type(predictions) is np.ndarray:
                self._array_bytes += predictions.nbytes
            if not _same_value(x, saved_x) or self._reader.snapshot_state() != state:
                raise ValueError("shift prediction changed its input or readonly reader")
            summary = public_feature_summary(x)
            self._array_bytes += summary.nbytes
            self._summaries += 1
            metrics = summarize_categorical_shift_batch(summary, predictions, y, 7)
            self._summary_returns += 1
            decision = self._detector.observe(metrics)
            row.update(
                metrics=asdict(metrics),
                decision=asdict(decision),
                incident=asdict(self._incidents.observe(decision)),
                status="complete",
            )
            succeeded = True
        except Exception as error:
            self._stopped = True
            row.update(status="failed", reason=f"{type(error).__name__}: {error}"[:512])
            raise
        finally:
            ended = monotonic()
            if (
                not isfinite(ended)
                or ended < started
                or not isfinite(self._seconds + ended - started)
            ):
                self._stopped = True
                row.update(status="failed", reason="invalid diagnostic elapsed time", seconds=None)
                raise ValueError("invalid diagnostic elapsed time")
            row["seconds"] = ended - started
            self._seconds += ended - started
            self._last_time = ended
            if succeeded:
                self._completed += 1

    def report(self):
        return deepcopy(
            dict(
                protocol=describe(max_arrivals=len(self._rows)),
                rows=self._rows,
                completed_arrivals=self._completed,
                stopped=self._stopped,
                prediction_calls_attempted=self._attempts,
                prediction_calls_returned=self._returns,
                prediction_rows_attempted=9 * self._attempts,
                prediction_rows_returned=9 * self._returns,
                summary_calls_attempted=self._summaries,
                summary_calls_completed=self._summary_returns,
                unreturned_prediction_unknown=self._attempts != self._returns,
                explicit_array_bytes_created=self._array_bytes,
                peak_scorer_numeric_bytes=self._peak_numeric_bytes,
                memory_scope=(
                    "explicit input/guard/summary/returned prediction arrays; coefficient/singular/error float64-equivalent peak; Python/state/mean temporaries/native projection/wrapper copies in RSS, not exhaustive allocation; one reader/"
                    + ("six" if len(self._rows) == 6 else str(len(self._rows)))
                    + " scalar slots"
                ),
                seconds=self._seconds,
                feedback=False,
            )
        )
