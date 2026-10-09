"""Owned NumPy learners implementing the generic native learner port.

Inputs are configured existing models and labeled arrays; outputs are native
diagnostics, feedforward predictions and detached in-memory model snapshots.
No new learning equations, final-role access, disk checkpoint or actor service.
"""

from copy import deepcopy
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.core.backprop_mlp import BackpropMLP, NUMPY_BACKPROP_LOSS_ID
from src.core.circadian_predictive_coding import (
    CircadianNetworkSnapshot,
    CircadianPredictiveCodingNetwork,
)
from src.core.dimension_validation import require_positive_integer_dimension
from src.core.learner_ports import TrainingDiagnostic
from src.core.data_erasure import ReplayPayloadErasure
from src.core.training_validation import validate_positive_finite_learning_rate
from src.core.native_model_copy import begin_model_copy

Array = NDArray[np.float64]


@dataclass(frozen=True)
class BackpropSnapshot:
    """Trusted local full model state; training policy remains adapter-owned."""

    format_version: int
    input_dim: int
    hidden_dims: tuple[int, ...]
    state: dict[str, Any]


def _validate_backprop_state(candidate: BackpropMLP, expected: BackpropMLP) -> None:
    """Validate topology, aliases, traffic and parameters before replacing state."""
    try:
        if (
            candidate.__dict__.keys() != expected.__dict__.keys()
            or type(candidate.input_dim) is not int
            or candidate.input_dim != expected.input_dim
            or type(candidate.hidden_dims) is not tuple
            or candidate.hidden_dims != expected.hidden_dims
            or any(type(width) is not int or width <= 0 for width in candidate.hidden_dims)
            or type(candidate._traffic_steps) is not int
            or candidate._traffic_steps < 0
            or type(candidate._hidden_weights) is not list
            or type(candidate._hidden_biases) is not list
            or type(candidate._traffic_sums) is not list
            or len(candidate._hidden_weights) != len(candidate.hidden_dims)
            or len(candidate._hidden_biases) != len(candidate.hidden_dims)
            or len(candidate._traffic_sums) != len(candidate.hidden_dims)
            or candidate.weight_input_hidden is not candidate._hidden_weights[0]
            or candidate.bias_hidden is not candidate._hidden_biases[0]
        ):
            raise ValueError("incompatible backprop snapshot fields or aliases")
        widths = (candidate.input_dim, *candidate.hidden_dims)
        arrays: list[tuple[Array, tuple[int, ...]]] = [
            (candidate.weight_hidden_output, (widths[-1], 1)),
            (candidate.bias_output, (1, 1)),
        ]
        for index, width in enumerate(candidate.hidden_dims):
            arrays.extend(
                (
                    (candidate._hidden_weights[index], (widths[index], width)),
                    (candidate._hidden_biases[index], (1, width)),
                    (candidate._traffic_sums[index], (width,)),
                )
            )
        if any(
            type(values) is not np.ndarray
            or values.dtype != np.float64
            or values.shape != shape
            or not np.all(np.isfinite(values))
            for values, shape in arrays
        ):
            raise ValueError("incompatible backprop snapshot arrays")
        if any(np.any(traffic < 0) for traffic in candidate._traffic_sums):
            raise ValueError("incompatible backprop snapshot traffic")
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError("incompatible backprop snapshot state") from error


class BackpropLearner:
    """Own a detached ordinary-gradient model without changing its native loss."""

    def __init__(self, model: BackpropMLP, *, learning_rate: float) -> None:
        if type(model) is not BackpropMLP:
            raise ValueError("backprop learner requires the native BackpropMLP")
        validate_positive_finite_learning_rate(learning_rate)
        _validate_backprop_state(model, model)
        self._model = deepcopy(model)
        self._learning_rate = learning_rate

    def fork(self) -> "BackpropLearner":
        """Copy complete native state and preserve this adapter's training policy."""
        return BackpropLearner(self._model, learning_rate=self._learning_rate)

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        result = self._model.train_epoch(features, targets, self._learning_rate)
        return TrainingDiagnostic(NUMPY_BACKPROP_LOSS_ID, result.loss)

    def predict(self, features: Array) -> Array:
        return self._model.predict_proba(features)

    def erase_replay_payloads(self) -> ReplayPayloadErasure:
        """Backprop stores no raw replay batches; parameters are not unlearned."""
        return ReplayPayloadErasure(0, 0, 0)

    def replay_payload_footprint(self) -> ReplayPayloadErasure:
        return ReplayPayloadErasure(0, 0, 0)

    def snapshot_state(self) -> BackpropSnapshot:
        # Copy the entire graph once so legacy aliases stay attached to their
        # corresponding hidden arrays and future model fields are not omitted.
        return BackpropSnapshot(
            1, self._model.input_dim, self._model.hidden_dims, deepcopy(self._model.__dict__)
        )

    def restore_state(self, snapshot: BackpropSnapshot) -> None:
        if (
            type(snapshot) is not BackpropSnapshot
            or type(snapshot.format_version) is not int
            or snapshot.format_version != 1
            or type(snapshot.input_dim) is not int
            or snapshot.input_dim != self._model.input_dim
            or snapshot.hidden_dims != self._model.hidden_dims
            or type(snapshot.state) is not dict
        ):
            raise ValueError("incompatible backprop snapshot metadata")
        candidate = object.__new__(BackpropMLP)
        candidate.__dict__.update(deepcopy(snapshot.state))
        _validate_backprop_state(candidate, self._model)
        self._model = candidate


def _copy_circadian_model(model: CircadianPredictiveCodingNetwork):
    # Why: observe the real memo at allocation, without adding native state fields.
    window = begin_model_copy(model)
    if window is None:
        return deepcopy(model)
    try:
        memo: dict[int, Any] = {}
        copied = deepcopy(model, memo)
        window.copied(copied, memo)
        return copied
    finally:
        window.end()


class CircadianLearner:
    """Reuse CPC's complete native model-state contract and energy definition."""

    def __init__(
        self,
        model: CircadianPredictiveCodingNetwork,
        *,
        learning_rate: float,
        inference_steps: int,
        inference_learning_rate: float,
    ) -> None:
        if type(model) is not CircadianPredictiveCodingNetwork:
            raise ValueError("circadian learner requires the native CPC network")
        validate_positive_finite_learning_rate(learning_rate)
        validate_positive_finite_learning_rate(inference_learning_rate)
        require_positive_integer_dimension(inference_steps, "inference_steps")
        self._model = _copy_circadian_model(model)
        self._learning_rate = learning_rate
        self._inference_steps = inference_steps
        self._inference_learning_rate = inference_learning_rate

    def fork(self) -> "CircadianLearner":
        """Reuse the owned constructor, preserving all model fields and policy."""
        return CircadianLearner(
            self._model,
            learning_rate=self._learning_rate,
            inference_steps=self._inference_steps,
            inference_learning_rate=self._inference_learning_rate,
        )

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        result = self._model.train_epoch(
            features,
            targets,
            self._learning_rate,
            self._inference_steps,
            self._inference_learning_rate,
        )
        return TrainingDiagnostic(result.energy_definition, result.energy)

    def predict(self, features: Array) -> Array:
        return self._model.predict_proba(features)

    def erase_replay_payloads(self) -> ReplayPayloadErasure:
        """Erase the owned model's complete raw replay buffer, not caller copies."""
        return self._model.erase_replay_payloads()

    def replay_payload_footprint(self) -> ReplayPayloadErasure:
        return self._model.replay_payload_footprint()

    def snapshot_state(self) -> CircadianNetworkSnapshot:
        return self._model.snapshot_state()

    def restore_state(self, snapshot: CircadianNetworkSnapshot) -> None:
        self._model.restore_state(snapshot)


def _managed_payload_bytes(value: object) -> int:
    if (
        type(value) is not np.ndarray
        or not np.issubdtype(value.dtype, np.number)
        or np.issubdtype(value.dtype, np.complexfloating)
    ):
        raise ValueError("managed payloads require supported real numeric NumPy arrays")
    return value.nbytes


def _managed_footprint(model: object) -> ReplayPayloadErasure:
    if type(model) not in (BackpropLearner, CircadianLearner):
        raise ValueError("managed cleanup requires a supported exact NumPy learner")
    assert isinstance(model, (BackpropLearner, CircadianLearner))
    return model.replay_payload_footprint()


def _managed_erase(model: object) -> ReplayPayloadErasure:
    if type(model) not in (BackpropLearner, CircadianLearner):
        raise ValueError("managed cleanup requires a supported exact NumPy learner")
    assert isinstance(model, (BackpropLearner, CircadianLearner))
    return model.erase_replay_payloads()


def make_managed_data_lifecycle(owner, *, policy):
    """Compose exact NumPy ports outside the application layer.

    Why: the coordinator must not depend on an outer learner adapter. These
    trusted ports bound array ingress and erase owned native replay buffers.
    """
    from src.app.managed_data_lifecycle import ManagedDataLifecycle

    return ManagedDataLifecycle(
        owner,
        policy=policy,
        measure_payload_bytes=_managed_payload_bytes,
        native_footprint=_managed_footprint,
        native_erase=_managed_erase,
        measure_auxiliary_bytes=_managed_auxiliary_bytes,
        measure_checkpoint_bytes=_managed_checkpoint_bytes,
        native_growth_bytes=_managed_growth_bytes,
        prepare_model_bytes=_managed_preparation_bytes,
        prediction_cache_bytes=_managed_prediction_bytes,
    )


def _managed_auxiliary_bytes(value: object) -> int:
    """Read only exact bounded containers; never invoke opaque object hooks.

    Count aliases once per independent copied graph. Reject cycles before copy;
    bounds protect measurement itself. This is array storage, not Python/RSS.
    """
    from src.core.serving_ports import CachedPrediction

    stack = [(value, 0, False)]
    seen: set[int] = set()
    active: set[int] = set()
    size = nodes = 0
    while stack:
        item, depth, leaving = stack.pop()
        identity = id(item)
        if leaving:
            active.remove(identity)
            continue
        nodes += 1
        if nodes > 4096 or depth > 32:
            raise ValueError("supported payload graph exceeds measurement bounds")
        if item is None or type(item) in (str, int, float, bool):
            continue
        if identity in active:
            raise ValueError("cyclic payload graphs are unsupported")
        if identity in seen:
            continue
        seen.add(identity)
        if type(item) is np.ndarray:
            if not np.issubdtype(item.dtype, np.number) and item.dtype != np.bool_:
                raise ValueError("object/non-numeric array graph unsupported")
            size += item.nbytes
            continue
        if type(item) is dict:
            if len(item) > 4096 - nodes:
                raise ValueError("supported payload graph exceeds measurement bounds")
            if any(type(key) is not str for key in item):
                raise ValueError("payload graph keys must be strings")
            children = tuple(item.values())
        elif type(item) in (list, tuple):
            assert isinstance(item, (list, tuple))
            if len(item) > 4096 - nodes:
                raise ValueError("supported payload graph exceeds measurement bounds")
            children = tuple(item)
        elif type(item) is CachedPrediction:
            children = (item.prediction,)
        else:
            raise ValueError("unsupported opaque payload graph")
        active.add(identity)
        stack.append((item, depth, True))
        stack.extend((v, depth + 1, False) for v in children)
    return size


def _managed_snapshot_bytes(snapshot) -> int:
    from src.core.circadian_predictive_coding import ReplaySnapshot

    if type(snapshot) is BackpropSnapshot:
        return 0
    if type(snapshot) is not CircadianNetworkSnapshot:
        raise ValueError("unsupported retained native snapshot")
    memory = snapshot.state.get("_replay_memory")
    if type(memory) is not deque or any(type(s) is not ReplaySnapshot for s in memory):
        raise ValueError("unsupported replay snapshot graph")
    return sum(
        _managed_payload_bytes(s.input_batch) + _managed_payload_bytes(s.target_batch)
        for s in memory
    )


def _managed_checkpoint_bytes(view) -> int:
    from src.app.candidate_checkpoint import CandidateCheckpointView

    if type(view) is not CandidateCheckpointView:
        raise ValueError("unsupported retained checkpoint view")
    return (
        _managed_snapshot_bytes(view.state)
        + sum(_managed_payload_bytes(s.features) for s in view.inbox.experiences)
        + sum(_managed_payload_bytes(t.targets) for t in view.inbox.labels)
    )


def _managed_growth_bytes(model, features, targets) -> int:
    _managed_footprint(model)
    size = _managed_payload_bytes(features) + _managed_payload_bytes(targets)
    return size if type(model) is CircadianLearner else 0


class ManagedNumpyBuilder:
    """Inspectable exact NumPy source fork for bounded owned preparations.

    The controller/evaluator restores its supplied complete state. A caller must
    keep this trusted source unchanged during preparation; no opaque callbacks.
    """

    def __init__(self, source: BackpropLearner | CircadianLearner) -> None:
        _managed_footprint(source)
        self._source = source

    def __call__(self, state):
        return self._source.fork()


def _managed_preparation_bytes(builder, state) -> int:
    if type(builder) is not ManagedNumpyBuilder:
        raise ValueError("owned byte policy requires supported inspectable NumPy builder")
    return _managed_footprint(builder._source).payload_bytes + _managed_snapshot_bytes(state)


def _managed_prediction_bytes(model, features) -> int:
    _managed_footprint(model)
    _managed_payload_bytes(features)
    if features.ndim != 2 or features.shape[1] != model._model.input_dim:
        raise ValueError("prediction input differs from native topology")
    return features.shape[0] * np.dtype(np.float64).itemsize
