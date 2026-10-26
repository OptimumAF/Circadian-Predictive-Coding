"""Observe delayed scores for the fixed recurring or stationary sensor demo.

Inputs are ordered retained prearrival pixels/probabilities and arrived TRAIN
labels. Outputs are calibrated descriptive shifts, false alarms and recovery
summaries. No model execution, data release, training feedback or IO occurs.
"""

from copy import deepcopy
from dataclasses import asdict
from typing import Any

from src.core.binary_probability_scores import probability_scores
from src.core.coupled_energy import Array
from src.core.sensor_recovery import summarize_phase_recovery
from src.core.shift_calibration import CalibratedShiftDetector, ShiftCalibration
from src.core.shift_incidents import ShiftIncidentTracker
from src.core.shift_signals import ShiftPolicy, summarize_shift_batch

CHANNELS = ("uncertainty", "feature", "residual")


def monitoring_configuration(*, stationary: bool = False) -> dict[str, Any]:
    if type(stationary) is not bool:
        raise ValueError("stationary must be an exact boolean")
    result = dict(
        policy=asdict(ShiftPolicy(reference_batches=2, window_batches=2, persistence_batches=2)),
        calibration=asdict(ShiftCalibration(prefix_batches=4)),
        calibration_scope="first four stationary-sensor A batches only; model may learn",
        observation_time="arrived labels; predictions and pixels retained from prearrival",
        channels=list(CHANNELS),
        false_alarm_scope="exclude calibration; count return-A rolling-window carryover",
        recovery=dict(
            accuracy_threshold=0.75,
            recovery_batches=2,
            label_delay=2,
            rows_per_batch=4,
            labels="phase labels arrived strictly before confirmation prediction",
        ),
        feedback=False,
    )
    if stationary:
        result.update(
            stationary=True,
            phase_schedule=["A"] * 12,
            false_alarm_scope="exclude calibration; stationary A feature ticks4..11 only",
        )
    return result


class SensorMonitoring:
    """Bounded observer for this demo; schedule truth scores diagnostics only."""

    def __init__(self, *, stationary: bool = False) -> None:
        if type(stationary) is not bool:
            raise ValueError("stationary must be an exact boolean")
        self._stationary = stationary
        self._detector = CalibratedShiftDetector(
            ShiftPolicy(reference_batches=2, window_batches=2, persistence_batches=2),
            ShiftCalibration(prefix_batches=4),
        )
        self._incidents = ShiftIncidentTracker()
        self._rows: list[dict[str, Any]] = []
        self._failed = False

    def observe(
        self, feature_tick: int, features: Array, probabilities: Array, targets: Array
    ) -> dict[str, Any]:
        if self._failed:
            raise ValueError("sensor monitoring is stopped after an uncertain observation")
        if type(feature_tick) is not int or feature_tick != len(self._rows) or feature_tick >= 12:
            raise ValueError("sensor monitoring requires consecutive feature ticks0..11")
        # Validate all arrays and scores before any stateful detector call.
        metrics = summarize_shift_batch(features, probabilities, targets)
        if features.shape != (4, 16):
            raise ValueError("sensor monitoring requires four frames of16pixels")
        scores = probability_scores(probabilities, targets)
        try:
            decision = self._detector.observe(metrics)
            incident = self._incidents.observe(decision)
            row = dict(
                feature_tick=feature_tick,
                label_tick=feature_tick + 2,
                metrics=asdict(metrics),
                decision=asdict(decision),
                incident=asdict(incident),
                accuracy=scores["accuracy"],
            )
            self._rows.append(row)
            return deepcopy(row)
        except BaseException:
            self._failed = True
            raise

    def _false_alarms(self, start: int, stop: int) -> dict[str, Any]:
        rows = [row for row in self._rows[start:stop] if row["decision"]["phase"] == "monitor"]
        alarms = [sum(row["decision"]["alarms"][c] for row in rows) for c in range(3)]
        onsets = [sum(row["incident"]["states"][c] == "onset" for row in rows) for c in range(3)]
        return dict(
            monitoring_batches=len(rows),
            alarm_batches=alarms,
            incident_onsets=onsets,
            rates=[count / len(rows) if rows else None for count in alarms],
        )

    def _recovery(self, start: int) -> dict[str, Any]:
        if self._stationary:
            # Why: an all-A control never presents either recovery phase;
            # observed good predictions cannot establish recovery from B.
            return dict(
                status="not_applicable",
                first_deficit_tick=None,
                confirmation_tick=None,
                labels_to_recovery=None,
                labels_available_before_last_prediction=0,
                observed_batches=0,
                planned_batches=0,
                horizon_complete=False,
            )
        accuracies = tuple(row["accuracy"] for row in self._rows[start : start + 4])
        if not accuracies:
            summary: dict[str, Any] = dict(
                status="not_run",
                first_deficit_tick=None,
                confirmation_tick=None,
                labels_to_recovery=None,
                labels_available_before_last_prediction=0,
            )
        else:
            summary = asdict(
                summarize_phase_recovery(
                    accuracies,
                    start_tick=start,
                    label_delay=2,
                    rows_per_batch=4,
                    accuracy_threshold=0.75,
                    recovery_batches=2,
                )
            )
        return dict(
            **summary,
            observed_batches=len(accuracies),
            planned_batches=4,
            horizon_complete=len(accuracies) == 4,
        )

    def report(self) -> dict[str, Any]:
        result = self._detector.calibration_result
        if self._stationary:
            configuration = monitoring_configuration(stationary=True)
            false_alarms = dict(
                pre_shift_A=self._false_alarms(0, 0),
                post_return_A=self._false_alarms(0, 0),
                stationary_A=self._false_alarms(4, 12),
            )
        else:
            configuration = monitoring_configuration()
            false_alarms = dict(
                pre_shift_A=self._false_alarms(0, 4), post_return_A=self._false_alarms(8, 12)
            )
        return deepcopy(
            dict(
                configuration=configuration,
                complete=len(self._rows) == 12 and not self._failed,
                stopped=self._failed,
                calibration=None if result is None else asdict(result),
                rows=self._rows,
                false_alarms=false_alarms,
                recovery=dict(B=self._recovery(4), A_return=self._recovery(8)),
                metrics_batches_completed=len(self._rows),
                native_forward_calls=0,
                diagnostic_scope="descriptive training stream; no statistical false-positive or causal recovery claim",
            )
        )
