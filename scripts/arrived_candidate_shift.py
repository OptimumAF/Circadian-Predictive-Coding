"""Finite arrived-before-wake categorical observations for the existing schedule.

Scores each newly arrived ordinary TRAIN packet once on a detached current
candidate, including while learning waits. Reuses existing metric/detector laws;
owns no training, final/probe source, FIFO selection, publication or new budget.
Arrival authority belongs to the calling delayed schedule, not arbitrary callers.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from time import monotonic
from typing import Any

import numpy as np

from src.app.coupled_attribution import AttributionBatch, _copy_batch, _same_value, _validate_batch
from src.core.experience import require_identifier
from src.core.shift_calibration import CalibratedShiftDetector, ShiftCalibration
from src.core.reference_controller import ArrivedShiftFeedback
from src.core.shift_incidents import ShiftIncidentTracker
from src.core.shift_signals import ShiftDetector, ShiftPolicy, summarize_categorical_shift_batch


def _calibration_controls(calibration, model_id, prefix_rows):
    if calibration is None:
        if model_id is not None or prefix_rows is not None:
            raise ValueError(
                "calibration configuration, model label and prefix must be provided together"
            )
        return None, None
    if type(calibration) is not ShiftCalibration:
        raise ValueError("calibration requires exact ShiftCalibration")
    calibration = replace(calibration)
    if type(model_id) is not str or not model_id.strip() or len(model_id.encode("utf-8")) > 128:
        raise ValueError("calibration model label must be nonempty and <=128 UTF8 bytes")
    if (
        type(prefix_rows) is not tuple
        or len(prefix_rows) != calibration.prefix_batches
        or not 1 <= len(prefix_rows) < 8
    ):
        raise ValueError("declared calibration prefix must leave a monitoring batch within8")
    seen = set()
    for row in prefix_rows:
        if type(row) is not tuple or not row:
            raise ValueError("calibration prefix requires nonempty immutable TRAIN row-ID tuples")
        for identifier in row:
            require_identifier(identifier, "calibration TRAIN row ID")
            if len(identifier.encode("utf-8")) > 128:
                raise ValueError("calibration TRAIN row ID exceeds128 UTF8 bytes")
            if identifier in seen:
                raise ValueError("calibration prefix row IDs must be unique")
            seen.add(identifier)
    if len(seen) >= 48:
        raise ValueError("calibration prefix must leave monitoring rows within48")
    # Reuse the unchanged core geometry and threshold formula, including failures.
    CalibratedShiftDetector(calibration=calibration)
    return calibration, tuple(tuple(row) for row in prefix_rows)


class ArrivedCandidateShift:
    def __init__(
        self, predict, *, calibration=None, calibration_model_id=None, calibration_prefix_rows=None
    ):
        if not callable(predict):
            raise ValueError("arrived shift prediction must be callable")
        self._predict = predict
        self._calibration, self._prefix_rows = _calibration_controls(
            calibration, calibration_model_id, calibration_prefix_rows
        )
        self._calibration_birth = deepcopy(
            (self._calibration, self._prefix_rows, calibration_model_id)
        )
        self._calibration_owners = (id(self._calibration), id(self._prefix_rows), id(predict))
        self._calibration_model_id = calibration_model_id
        self._reference_state: Any = None
        self._reference_pin: Any = None
        self._reference_owner = None
        self._reference_binding = None
        self._fitted_calibration = None
        self._detector = (
            ShiftDetector()
            if self._calibration is None
            else CalibratedShiftDetector(calibration=self._calibration)
        )
        self._detector_owner = self._detector
        self._incidents = ShiftIncidentTracker()
        self._arrived = 0
        self._seen: set[str] = set()
        self._width: int | None = None
        self._pin: Any = None
        self._binding: tuple[str, str, int] | None = None
        self._stopped = False
        self._paid: dict[str, Any] = dict(
            definition="arrived_candidate_categorical_shift_v1",
            policy=asdict(ShiftPolicy()),
            attempts=0,
            returns=0,
            rows_attempted=0,
            rows_returned=0,
            input_bytes=0,
            seconds=0.0,
            records=[],
            stopped=False,
            max_batches=8,
            max_rows=48,
            feedback="any current alarm requests wake1; FIFO unchanged",
        )

    @property
    def report(self):
        report = deepcopy(self._paid)
        calibration, prefix, model_id = self._calibration_birth
        if calibration is not None:
            report["definition"] = "arrived_frozen_reference_calibrated_shift_v1"
            report["calibration"] = dict(
                configuration=asdict(calibration),
                model_id=model_id,
                declared_stationary_train_prefix_rows=prefix,
                stationarity="caller declaration, not statistically verified",
                identity_kind="caller label plus owned initial diagnostic state and observed runtime origin",
                reference_binding=self._reference_binding,
                retained_native_state_copies=0 if self._reference_owner is None else 2,
                storage_scope="one fixed reference plus one detached value pin; no state history; excludes transient/live copies",
                result=None
                if self._fitted_calibration is None
                else asdict(self._fitted_calibration),
            )
        return deepcopy(report)

    def require_paid(self, report):
        if not _same_value(report, self.report):
            raise ValueError("original arrived shift cost report changed")

    def _require_calibration(self):
        if self._calibration_birth[0] is None:
            return
        owners = (id(self._calibration), id(self._prefix_rows), id(self._predict))
        if (
            owners != self._calibration_owners
            or not _same_value(
                (self._calibration, self._prefix_rows, self._calibration_model_id),
                self._calibration_birth,
            )
            or self._detector is not self._detector_owner
            or self._reference_owner is not None
            and (
                id(self._reference_state) != self._reference_owner
                or not _same_value(self._reference_state, self._reference_pin)
            )
        ):
            self._stopped = self._paid["stopped"] = True
            raise ValueError("original calibration controls, callback or fixed reference changed")

    def _prediction_state(self, snapshot, row):
        if self._calibration is None:
            return snapshot.state
        self._require_calibration()
        if self._reference_owner is None:
            self._reference_state = deepcopy(snapshot.state)
            self._reference_pin = deepcopy(snapshot.state)
            self._reference_owner = id(self._reference_state)
            self._reference_binding = (
                snapshot.actor_version,
                snapshot.learner_version,
                snapshot.wake_updates,
                snapshot.state.completed_updates,
            )
        row["prediction_model"] = "owned_frozen_initial_diagnostic_reference"
        row["prediction_native_updates"] = self._reference_pin.completed_updates
        return deepcopy(self._reference_state)

    def require_current(self, runtime):
        if self._stopped:
            raise ValueError("arrived shift reader is stopped")
        if self._pin is not None and not _same_value(runtime.candidate_snapshot(), self._pin):
            self._stopped = self._paid["stopped"] = True
            raise ValueError("arrived shift candidate changed before admission")

    def read(
        self,
        runtime,
        batches,
        *,
        arrived,
        completed,
        tick,
        label_delay,
        actor_version,
        require_input,
    ):
        if self._stopped:
            raise ValueError("arrived shift reader is stopped")
        self._require_calibration()
        if (
            any(type(v) is not int or v < 0 for v in (arrived, completed, tick))
            or type(label_delay) is not int
            or label_delay not in (1, 2)
            or not self._arrived <= arrived <= min(8, self._arrived + 1)
            or completed > arrived
        ):
            raise ValueError("arrived shift requires consecutive bounded causal arrival counts")
        # No new label means no packet, candidate copy or repeated model call.
        self._pin = None
        if arrived == self._arrived:
            return None
        index = arrived - 1
        if index + label_delay > tick:
            raise ValueError("shift labels have not arrived")
        batch = batches[index]
        self._validate(batch)
        if (
            self._calibration is not None
            and index < len(self._prefix_rows)
            and batch.row_ids != self._prefix_rows[index]
        ):
            self._stopped = self._paid["stopped"] = True
            raise ValueError("arrived TRAIN rows differ from predeclared calibration prefix")
        require_input()
        snapshot = runtime.candidate_snapshot()
        if snapshot.actor_version != actor_version or snapshot.wake_updates != completed:
            raise ValueError("shift feedback belongs to a stale actor or wake count")
        native_updates = snapshot.state.completed_updates
        if type(native_updates) is not int or native_updates < completed:
            raise ValueError("shift native update clock is invalid")
        binding = (snapshot.actor_version, snapshot.learner_version, native_updates - completed)
        if self._binding is not None and binding != self._binding:
            if self._calibration is not None:
                self._stopped = self._paid["stopped"] = True
            raise ValueError("original shift actor/candidate/native origin changed")
        self._binding = binding
        self._pin = deepcopy(snapshot)
        owned = _copy_batch(batch)
        row: dict[str, Any] = dict(
            index=index,
            label_observed_at=index,
            label_arrived_at=index + label_delay,
            read_at=tick,
            actor_version=actor_version,
            candidate_version=snapshot.learner_version,
            wake_updates=completed,
            native_updates=snapshot.state.completed_updates,
            row_ids=batch.row_ids,
            rows=len(batch.features),
            complete=False,
        )
        self._paid["records"].append(row)
        self._paid["attempts"] += 1
        self._paid["rows_attempted"] += len(batch.features)
        self._paid["input_bytes"] += batch.features.nbytes + batch.targets.nbytes
        started = monotonic()
        try:
            prediction_state = self._prediction_state(snapshot, row)
            prediction_pin = deepcopy(prediction_state) if self._calibration is not None else None
            query_features = owned.features.copy()
            prediction = self._predict(prediction_state, query_features)
            self._paid["returns"] += 1
            # Normal returns remain charged even if probability validation fails.
            self._paid["rows_returned"] += len(batch.features)
            require_input()
            self._require_calibration()
            if self._calibration is not None and (
                not _same_value(prediction_state, prediction_pin)
                or not _same_value(query_features, owned.features)
            ):
                raise ValueError("calibration prediction mutated its detached reference input")
            if not _same_value(snapshot, self._pin) or not _same_value(batch, owned):
                raise ValueError("shift prediction changed detached state or TRAIN data")
            self.require_current(runtime)
            metrics = summarize_categorical_shift_batch(
                owned.features, prediction, owned.targets, 6
            )
            try:
                decision = self._detector.observe(metrics)
            finally:
                if self._calibration is not None:
                    assert isinstance(self._detector, CalibratedShiftDetector)
                    # A lost scalar return or later incident fault cannot hide
                    # an owned completed fit/prefix replay; row success is separate.
                    self._fitted_calibration = self._detector.calibration_result
            incident = self._incidents.observe(decision)
            result = ArrivedShiftFeedback(
                "train",
                f"batch-{index}",
                actor_version,
                tick,
                tick,
                snapshot.learner_version,
                completed,
                arrived,
                decision.alarms,
            )
            row.update(
                metrics=asdict(metrics),
                decision=asdict(decision),
                incident=asdict(incident),
                complete=True,
            )
            self._seen.update(batch.row_ids)
            self._width = batch.features.shape[1]
            self._arrived = arrived
            return result
        except Exception as error:
            self._stopped = self._paid["stopped"] = True
            row["error"] = f"{type(error).__name__}: {error}"[:512]
            raise
        finally:
            row["seconds"] = monotonic() - started
            self._paid["seconds"] += row["seconds"]

    def _validate(self, batch):
        if (
            type(batch) is not AttributionBatch
            or batch.role != "train"
            or batch.num_classes != 6
            or type(batch.features) is not np.ndarray
            or batch.features.dtype != np.float64
            or batch.features.ndim != 2
            or not 1 <= batch.features.shape[0] <= 128
            or not 1 <= batch.features.shape[1] <= 16
            or self._width is not None
            and self._width != batch.features.shape[1]
        ):
            raise ValueError("arrived shift requires ordinary TRAIN float64 K6 packets <=128x16")
        if self._paid["rows_attempted"] + len(batch.features) > 48:
            raise ValueError("original arrived shift row allowance exhausted")
        _validate_batch(batch, "train", set(self._seen))
