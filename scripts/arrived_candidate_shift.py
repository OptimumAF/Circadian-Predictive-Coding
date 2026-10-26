"""Finite arrived-before-wake categorical observations for the existing schedule.

Scores each newly arrived ordinary TRAIN packet once on a detached current
candidate, including while learning waits. Reuses existing metric/detector laws;
owns no training, final/probe source, FIFO selection, publication or new budget.
Arrival authority belongs to the calling delayed schedule, not arbitrary callers.
"""

from copy import deepcopy
from dataclasses import asdict
from time import monotonic
from typing import Any

import numpy as np

from src.app.coupled_attribution import AttributionBatch, _copy_batch, _same_value, _validate_batch
from src.core.reference_controller import ArrivedShiftFeedback
from src.core.shift_incidents import ShiftIncidentTracker
from src.core.shift_signals import ShiftDetector, ShiftPolicy, summarize_categorical_shift_batch


class ArrivedCandidateShift:
    def __init__(self, predict):
        if not callable(predict):
            raise ValueError("arrived shift prediction must be callable")
        self._predict = predict
        self._detector = ShiftDetector()
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
        return deepcopy(self._paid)

    def require_paid(self, report):
        if not _same_value(report, self._paid):
            raise ValueError("original arrived shift cost report changed")

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
        require_input()
        snapshot = runtime.candidate_snapshot()
        if snapshot.actor_version != actor_version or snapshot.wake_updates != completed:
            raise ValueError("shift feedback belongs to a stale actor or wake count")
        native_updates = snapshot.state.completed_updates
        if type(native_updates) is not int or native_updates < completed:
            raise ValueError("shift native update clock is invalid")
        binding = (snapshot.actor_version, snapshot.learner_version, native_updates - completed)
        if self._binding is not None and binding != self._binding:
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
            prediction = self._predict(snapshot.state, owned.features.copy())
            self._paid["returns"] += 1
            # Normal returns remain charged even if probability validation fails.
            self._paid["rows_returned"] += len(batch.features)
            require_input()
            if not _same_value(snapshot, self._pin) or not _same_value(batch, owned):
                raise ValueError("shift prediction changed detached state or TRAIN data")
            self.require_current(runtime)
            metrics = summarize_categorical_shift_batch(
                owned.features, prediction, owned.targets, 6
            )
            decision = self._detector.observe(metrics)
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
