"""Metadata-only retained raw-copy measurements for trusted coupled ports.

These ports count supported raw array payloads, not learned parameters, Python
objects, caller copies or RSS. They borrow quiescent metadata and never scan,
copy, predict, snapshot or train. Native value validation remains in the learner
and codec. The inspectable builder restores complete policy through the existing
callbacks; ownership, reservations and cleanup remain in ManagedDataLifecycle.
"""

from dataclasses import replace
from math import isfinite

import numpy as np

from src.adapters.coupled_checkpoint_callbacks import CoupledCheckpointCallbacks
from src.adapters.coupled_data_lifecycle import (
    measure_coupled_payload_bytes,
    coupled_replay_footprint,
    erase_coupled_replay_payloads,
)
from src.adapters.coupled_learners import CoupledLearner
from src.app.candidate_checkpoint import CandidateCheckpointView, _validate_view
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_experience import ManagedExperienceOwner
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_predictive_coding import CoupledSnapshot, CoupledWakeState
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import require_tick
from src.core.inbox_cursor import InboxCursor
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.serving_ports import CachedPrediction


def _fields(value: object, expected: set[str]) -> None:
    if set(vars(value)) != expected:
        raise ValueError("unsupported coupled measurement record fields")


def measure_coupled_auxiliary_bytes(value: object) -> int:
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
            raise ValueError("coupled payload graph exceeds measurement bounds")
        if item is None or type(item) in (str, int, float, bool):
            continue
        if identity in active:
            raise ValueError("cyclic coupled payload graph is unsupported")
        if identity in seen:
            continue
        seen.add(identity)
        if type(item) is np.ndarray:
            size += measure_coupled_payload_bytes(item)
            continue
        if type(item) is dict:
            if len(item) > 4096 - nodes or any(type(key) is not str for key in item):
                raise ValueError("coupled payload dictionaries require bounded string keys")
            children = tuple(item.values())
        elif type(item) in (list, tuple):
            assert isinstance(item, (list, tuple))
            if len(item) > 4096 - nodes:
                raise ValueError("coupled payload graph exceeds measurement bounds")
            children = tuple(item)
        elif type(item) is CachedPrediction:
            _fields(item, {"expires_at", "prediction"})
            require_tick(item.expires_at, "cached prediction expiry")
            children = (item.prediction,)
        else:
            raise ValueError("unsupported opaque coupled payload graph")
        active.add(identity)
        stack.append((item, depth, True))
        stack.extend((child, depth + 1, False) for child in children)
    return size


def _matrix_metadata(value: object, shape: tuple[int, ...] | None = None) -> None:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.float64
        or value.ndim != (2 if shape is None else len(shape))
        or min(value.shape) < 1
        or (shape is not None and value.shape != shape)
    ):
        raise ValueError("coupled native state requires correctly shaped float64 arrays")


def _snapshot_metadata(state: object) -> None:
    if type(state) is not CoupledSnapshot:
        raise ValueError("coupled preparation requires an exact CoupledSnapshot")
    _fields(state, {"format_version", "parameters", "precisions", "wake_state"})
    if type(state.format_version) is not int or state.format_version != 1:
        raise ValueError("coupled native snapshot version must be exactly1")
    parameters = state.parameters
    if type(parameters) is not CoupledParameters:
        raise ValueError("coupled native snapshot requires exact CoupledParameters")
    _fields(parameters, {"hidden_weights", "hidden_biases", "output_weight", "output_bias"})
    weights, biases = parameters.hidden_weights, parameters.hidden_biases
    if (
        type(weights) is not tuple
        or not 1 <= len(weights) <= 2046
        or type(biases) is not tuple
        or len(biases) != len(weights)
    ):
        raise ValueError("coupled native array collections exceed structural measurement bounds")
    # At most4096 known array fields, independently of their borrowed storage.
    previous = 0
    for weight, bias in zip(weights, biases):
        _matrix_metadata(weight)
        if previous and weight.shape[0] != previous:
            raise ValueError("coupled native weights have incompatible adjacent widths")
        previous = weight.shape[1]
        _matrix_metadata(bias, (1, previous))
    _matrix_metadata(parameters.output_weight, (previous, 1))
    _matrix_metadata(parameters.output_bias, (1, 1))
    precisions = state.precisions
    if (
        type(precisions) is not tuple
        or len(precisions) != len(weights)
        or any(
            type(beta) not in (int, float) or not 1e-6 <= beta <= 1e6 or not isfinite(beta)
            for beta in precisions
        )
    ):
        raise ValueError("coupled native precisions must match hidden topology")
    if state.wake_state is not None:
        _wake_metadata(state.wake_state, previous)


def _wake_metadata(wake: object, width: int) -> None:
    if type(wake) is not CoupledWakeState:
        raise ValueError("coupled native wake state must be exact CoupledWakeState")
    _fields(
        wake,
        {
            "chemical_decay",
            "chemical_buildup_rate",
            "plasticity_sensitivity",
            "min_plasticity",
            "chemical_state",
            "plasticity_state",
            "wake_updates",
        },
    )
    CoupledCircadianConfig(
        wake.chemical_decay,
        wake.chemical_buildup_rate,
        wake.plasticity_sensitivity,
        wake.min_plasticity,
    )
    require_tick(wake.wake_updates, "coupled wake updates")
    _matrix_metadata(wake.chemical_state, (width,))
    _matrix_metadata(wake.plasticity_state, (width,))


def measure_coupled_checkpoint_bytes(view: object) -> int:
    if type(view) is not CandidateCheckpointView:
        raise ValueError("coupled checkpoint measurement requires an exact checkpoint view")
    _fields(
        view,
        {
            "actor_version",
            "actor_generation",
            "learner_version",
            "state",
            "inbox",
            "attempted_ids",
            "consolidations",
            "consolidation_limit",
            "stopped",
            "budget_state",
            "sharing",
            "event_tick",
        },
    )
    if type(view.inbox) is not InboxCursor:
        raise ValueError("coupled checkpoint requires an exact inbox cursor")
    for items in (
        view.inbox.experiences,
        view.inbox.labels,
        view.inbox.applied,
        view.inbox.erased,
        view.attempted_ids,
        view.consolidations,
    ):
        if type(items) is not tuple or len(items) > 4096:
            raise ValueError("coupled checkpoint histories exceed measurement bounds")
    _validate_view(view)
    _snapshot_metadata(view.state)
    return sum(measure_coupled_payload_bytes(s.features) for s in view.inbox.experiences) + sum(
        measure_coupled_payload_bytes(t.targets) for t in view.inbox.labels
    )


def coupled_native_growth_bytes(learner: object, features: object, targets: object) -> int:
    # Why: sizing accepts real numeric storage; native task/dtype/value validation
    # is separate and must not require a scan merely to reserve a copy allowance.
    if type(learner) is not CoupledLearner:
        raise ValueError("coupled growth measurement requires an exact CoupledLearner")
    measure_coupled_payload_bytes(features)
    measure_coupled_payload_bytes(targets)
    return 0


class ManagedCoupledBuilder:
    """Exact inspectable full-policy builder; only an actual call counts a build."""

    def __init__(self, callbacks: CoupledCheckpointCallbacks) -> None:
        if type(callbacks) is not CoupledCheckpointCallbacks:
            raise ValueError("managed coupled builder requires exact checkpoint callbacks")
        self._callbacks = callbacks
        self._build_calls = 0

    @property
    def build_calls(self) -> int:
        return self._build_calls

    def __call__(self, state: CoupledSnapshot) -> CoupledLearner:
        self._build_calls += 1
        return self._callbacks.build(state)


def coupled_preparation_bytes(builder: object, state: object) -> int:
    if type(builder) is not ManagedCoupledBuilder:
        raise ValueError("coupled copy policy requires an exact ManagedCoupledBuilder")
    if type(builder._callbacks) is not CoupledCheckpointCallbacks:
        raise ValueError("managed coupled builder callbacks changed")
    _snapshot_metadata(state)
    return 0


def coupled_prediction_cache_bytes(learner: object, features: object) -> int:
    if type(learner) is not CoupledLearner:
        raise ValueError("coupled prediction measurement requires an exact CoupledLearner")
    measure_coupled_payload_bytes(features)
    assert isinstance(features, np.ndarray)
    input_dim, hidden_dims = learner._model.input_dim, learner._model.hidden_dims
    if (
        type(input_dim) is not int
        or input_dim < 1
        or type(hidden_dims) is not tuple
        or not hidden_dims
        or len(hidden_dims) > 2046
        or any(type(width) is not int or width < 1 for width in hidden_dims)
        or features.ndim != 2
        or features.shape[0] < 1
        or features.shape[1] != input_dim
    ):
        raise ValueError("coupled prediction input differs from supported native topology")
    return features.shape[0] * np.dtype(np.float64).itemsize


def make_coupled_copy_lifecycle(
    owner: ManagedExperienceOwner, *, policy: DataRetentionPolicy
) -> ManagedDataLifecycle:
    if type(policy) is not DataRetentionPolicy:
        raise ValueError("coupled copy lifecycle requires an exact DataRetentionPolicy")
    if policy.max_retention_seconds is not None:
        raise ValueError("coupled copy lifecycle does not support elapsed-retention policy")
    if type(policy.owned_payload_copies) is not PayloadCopyLimits:
        raise ValueError("coupled copy lifecycle requires exact owned payload copy limits")
    if type(policy.holders) is not PayloadOwnershipLimits:
        raise ValueError("coupled copy lifecycle requires exact holder limits")
    replace(
        policy,
        holders=replace(policy.holders),
        owned_payload_copies=replace(policy.owned_payload_copies),
    )
    return ManagedDataLifecycle(
        owner,
        policy=policy,
        measure_payload_bytes=measure_coupled_payload_bytes,
        native_footprint=coupled_replay_footprint,
        native_erase=erase_coupled_replay_payloads,
        measure_auxiliary_bytes=measure_coupled_auxiliary_bytes,
        measure_checkpoint_bytes=measure_coupled_checkpoint_bytes,
        native_growth_bytes=coupled_native_growth_bytes,
        prepare_model_bytes=coupled_preparation_bytes,
        prediction_cache_bytes=coupled_prediction_cache_bytes,
    )
