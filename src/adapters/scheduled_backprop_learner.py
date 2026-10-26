"""Fixed four-arrival finetuning for the small delayed sensor stream.

Inputs are arrived float64 (4, 16) features and (4, 1) binary targets. Each
four-batch window receives one ordinary matched BP mean-BCE update. Pending
data, fixed rate and native state survive owned forks/restores. This adapter
does not release labels, authorize training, schedule clocks or promote actors.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.coupled_energy import Array, _matrix
from src.core.learner_ports import TrainingDiagnostic
from src.core.matched_backprop import (
    MatchedBackpropNetwork,
    MatchedBackpropSnapshot,
    _learning_rate,
)

BUFFERED_DIAGNOSTIC_ID = "scheduled_bp_buffered_no_optimizer_v1"
_BATCH_BYTES = 4 * (16 + 1) * 8


@dataclass(frozen=True)
class ScheduledBackpropSnapshot:
    """Complete local continuation state; counters confer no runtime authority."""

    format_version: int
    native: MatchedBackpropSnapshot
    learning_rate: float
    arrived_batches: int
    optimizer_steps_attempted: int
    optimizer_steps_completed: int
    buffered_batches: tuple[tuple[Array, Array], ...]
    peak_buffer_bytes: int
    stopped: bool


def _copy_batch(features: Array, targets: Array) -> tuple[Array, Array]:
    _matrix(features, "features")
    _matrix(targets, "targets")
    if features.shape != (4, 16) or targets.shape != (4, 1):
        raise ValueError("scheduled BP requires (4, 16) features and (4, 1) targets")
    if np.any((targets < 0) | (targets > 1)):
        raise ValueError("scheduled BP targets must lie in [0, 1]")
    return features.copy(), targets.copy()


def _validate_counters(state: ScheduledBackpropSnapshot) -> None:
    integers = (
        state.format_version,
        state.arrived_batches,
        state.optimizer_steps_attempted,
        state.optimizer_steps_completed,
        state.peak_buffer_bytes,
    )
    if any(type(value) is not int for value in integers) or type(state.stopped) is not bool:
        raise ValueError("scheduled BP snapshot requires exact native counters and stopped flag")
    arrivals, attempts, completed = integers[1:4]
    if state.format_version != 1 or not 0 <= arrivals <= 12:
        raise ValueError("unsupported scheduled BP snapshot format or arrival count")
    if not 0 <= completed <= attempts <= 3 or attempts != arrivals // 4:
        raise ValueError("scheduled BP optimizer counters disagree with arrivals")
    if state.peak_buffer_bytes != _BATCH_BYTES * min(arrivals, 4):
        raise ValueError("scheduled BP peak buffer bytes disagree with arrivals")
    if type(state.buffered_batches) is not tuple:
        raise ValueError("scheduled BP buffered batches must be an exact tuple")
    if state.stopped:
        valid = arrivals > 0 and arrivals % 4 == 0 and completed == attempts - 1
        pending = 4
    else:
        valid = completed == attempts
        pending = arrivals % 4
    if not valid or len(state.buffered_batches) != pending:
        raise ValueError("scheduled BP pending data or failure counters are inconsistent")


def _copy_pending(state: ScheduledBackpropSnapshot) -> tuple[tuple[Array, Array], ...]:
    parameters = state.native.parameters
    arrays = [
        *parameters.hidden_weights,
        *parameters.hidden_biases,
        parameters.output_weight,
        parameters.output_bias,
    ]
    copied = []
    for batch in state.buffered_batches:
        if type(batch) is not tuple or len(batch) != 2:
            raise ValueError("scheduled BP pending batch must be an exact feature/target pair")
        copied.append(_copy_batch(*batch))
        arrays.extend(batch)
    for index, array in enumerate(arrays):
        if any(np.shares_memory(array, other) for other in arrays[:index]):
            raise ValueError("scheduled BP snapshot arrays must not alias")
    return tuple(copied)


class ScheduledBackpropLearner:
    """Bounded 12-arrival / three-step finetuning with permanent failed attempts."""

    def __init__(self, model: MatchedBackpropNetwork, *, learning_rate: float) -> None:
        if type(model) is not MatchedBackpropNetwork:
            raise ValueError("scheduled BP requires an exact native matched BP model")
        if model.input_dim != 16 or model.hidden_dims != (8,):
            raise ValueError("scheduled BP requires the fixed 16→8→1 sensor topology")
        self._learning_rate = _learning_rate(learning_rate)
        self._model = MatchedBackpropNetwork(model.snapshot_parameters())
        self._arrived_batches = 0
        self._attempted = 0
        self._completed = 0
        self._buffer: tuple[tuple[Array, Array], ...] = ()
        self._peak_buffer_bytes = 0
        self._stopped = False

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        if self._stopped:
            raise RuntimeError("scheduled BP stopped after a failed optimizer attempt")
        if self._arrived_batches >= 12:
            raise ValueError("scheduled BP twelve-arrival allowance exhausted")
        pending = (*self._buffer, _copy_batch(features, targets))
        # Allocate and own the whole update before admitting the next arrival.
        if len(pending) == 4:
            combined_features = np.concatenate([batch[0] for batch in pending])
            combined_targets = np.concatenate([batch[1] for batch in pending])
        self._buffer = pending
        self._arrived_batches += 1
        self._peak_buffer_bytes = max(self._peak_buffer_bytes, len(pending) * _BATCH_BYTES)
        if len(pending) < 4:
            # Why: a buffered wake is not a BCE evaluation or an optimizer step.
            return TrainingDiagnostic(BUFFERED_DIAGNOSTIC_ID, 0.0)
        self._attempted += 1
        self._stopped = True
        result = self._model.train_epoch(combined_features, combined_targets, self._learning_rate)
        diagnostic = TrainingDiagnostic(result.loss_definition, result.loss)
        self._completed += 1
        self._buffer = ()
        self._stopped = False
        return diagnostic

    def predict(self, features: Array) -> Array:
        _matrix(features, "features")
        if features.shape not in ((4, 16), (32, 16)):
            raise ValueError("scheduled BP prediction requires (4, 16) or final (32, 16) features")
        return self._model.predict_proba(features.copy()).copy()

    def snapshot_state(self) -> ScheduledBackpropSnapshot:
        return ScheduledBackpropSnapshot(
            1,
            self._model.snapshot_state(),
            self._learning_rate,
            self._arrived_batches,
            self._attempted,
            self._completed,
            tuple((features.copy(), targets.copy()) for features, targets in self._buffer),
            self._peak_buffer_bytes,
            self._stopped,
        )

    def restore_state(self, snapshot: ScheduledBackpropSnapshot) -> None:
        if self._stopped:
            raise RuntimeError("cannot restore over a permanently failed scheduled BP instance")
        if type(snapshot) is not ScheduledBackpropSnapshot:
            raise ValueError("scheduled BP requires its exact full snapshot")
        _validate_counters(snapshot)
        if _learning_rate(snapshot.learning_rate) != self._learning_rate:
            raise ValueError("scheduled BP snapshot learning rate differs from learner")
        candidate = MatchedBackpropNetwork(self._model.snapshot_parameters())
        candidate.restore_state(snapshot.native)
        pending = _copy_pending(snapshot)
        # Publish only after every tensor, policy and continuation counter passes.
        self._model = candidate
        self._arrived_batches = snapshot.arrived_batches
        self._attempted = snapshot.optimizer_steps_attempted
        self._completed = snapshot.optimizer_steps_completed
        self._buffer = pending
        self._peak_buffer_bytes = snapshot.peak_buffer_bytes
        self._stopped = snapshot.stopped

    def fork(self) -> "ScheduledBackpropLearner":
        candidate = ScheduledBackpropLearner(self._model, learning_rate=self._learning_rate)
        candidate.restore_state(self.snapshot_state())
        return candidate

    def report(self) -> dict[str, Any]:
        return dict(
            arrived_batches=self._arrived_batches,
            optimizer_steps_attempted=self._attempted,
            optimizer_steps_completed=self._completed,
            training_rows_attempted=16 * self._attempted,
            training_rows_completed=16 * self._completed,
            buffered_batches=len(self._buffer),
            buffer_bytes=len(self._buffer) * _BATCH_BYTES,
            peak_buffer_bytes=self._peak_buffer_bytes,
            stopped=self._stopped,
        )
