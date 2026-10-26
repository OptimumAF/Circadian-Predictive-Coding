"""Summarize bounded tanh features and supplied native unit histories.

Activation diagnostics infer neither importance nor neuron age; effective rank
uses singular values, rather than their squares. Unit summaries preserve supplied
importance/age observations without provenance or causal plasticity claims.
Neither interface mutates caller arrays or invents missing model history.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.core.coupled_energy import Array

UnitIds = NDArray[np.int64]


@dataclass(frozen=True)
class ActivationHealth:
    layer_index: int
    rows: int
    units: int
    active_fraction: float
    dead_fraction: float
    saturated_fraction: float
    mean_absolute_activity: float
    min_mean_absolute_activity: float
    max_mean_absolute_activity: float
    unit_mean_absolute_activity: tuple[float, ...]
    feature_effective_rank: float


def _validate_activations(activations: tuple[Array, ...]) -> None:
    if type(activations) is not tuple or not 1 <= len(activations) <= 8:
        raise ValueError("activations must be a tuple containing 1..8 layers")
    cells = 0
    for index, values in enumerate(activations):
        if (
            type(values) is not np.ndarray
            or values.dtype != np.float64
            or values.ndim != 2
            or not 1 <= values.shape[0] <= 1024
            or not 1 <= values.shape[1] <= 128
        ):
            raise ValueError(f"activation layer {index} must be a float64 matrix <=1024x128")
        cells += values.size
        if cells > 65536:
            raise ValueError("activation matrices exceed 65536 aggregate cells")
        if not np.isfinite(values).all() or np.any((values < -1) | (values > 1)):
            raise ValueError(f"activation layer {index} must contain finite tanh values [-1, 1]")


def _effective_rank(values: Array) -> float:
    centered = values - np.mean(values, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    if not np.isfinite(singular).all():
        raise ValueError("activation singular values must be finite")
    cutoff = np.finfo(np.float64).eps * max(values.shape) * max(1.0, float(singular.max()))
    positive = singular[singular > cutoff]
    if not len(positive):
        return 0.0
    probabilities = positive / positive.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def diagnose_activations(activations: tuple[Array, ...]) -> tuple[ActivationHealth, ...]:
    """Report activity and centered singular-value entropy rank for each layer.

    Dead units have mean absolute activity <=1e-3; active is the complement.
    Saturation counts entries with absolute activation >=.99, so it overlaps
    activity. Rank discards s <= eps*max(shape)*max(1,max(s)), then computes
    exp(entropy(s/sum(s))); a numerically zero feature matrix has rank zero.
    All layers are validated before the first SVD. Exactly one SVD is used per
    layer, within the fixed layer/shape/aggregate-cell envelope.
    """
    _validate_activations(activations)
    result = []
    for index, values in enumerate(activations):
        absolute = np.abs(values)
        activity = np.mean(absolute, axis=0)
        dead = float(np.mean(activity <= 1e-3))
        result.append(
            ActivationHealth(
                layer_index=index,
                rows=values.shape[0],
                units=values.shape[1],
                active_fraction=1.0 - dead,
                dead_fraction=dead,
                saturated_fraction=float(np.mean(absolute >= 0.99)),
                mean_absolute_activity=float(np.mean(activity)),
                min_mean_absolute_activity=float(np.min(activity)),
                max_mean_absolute_activity=float(np.max(activity)),
                unit_mean_absolute_activity=tuple(float(value) for value in activity),
                feature_effective_rank=_effective_rank(values),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class VectorSummary:
    count: int
    minimum: float
    mean: float
    maximum: float


@dataclass(frozen=True)
class UnitHistory:
    neuron_id: int
    parent_id: int | None
    importance_ema: float
    age_updates: float


@dataclass(frozen=True)
class UnitHistoryDiagnostics:
    units: tuple[UnitHistory, ...]
    importance: VectorSummary
    age: VectorSummary


def _validate_unit_vectors(
    neuron_ids: UnitIds, parent_ids: UnitIds, importance_ema: Array, neuron_age: Array
) -> None:
    vectors = (neuron_ids, parent_ids, importance_ema, neuron_age)
    for values, dtype in zip(vectors, (np.int64, np.int64, np.float64, np.float64)):
        if (
            type(values) is not np.ndarray
            or values.dtype != dtype
            or values.ndim != 1
            or not 1 <= len(values) <= 4096
        ):
            raise ValueError(
                "unit history requires int64 IDs and float64 vectors of length 1..4096"
            )
    if any(values.shape != neuron_ids.shape for values in vectors[1:]):
        raise ValueError("unit history vectors must have aligned lengths")
    for values in (importance_ema, neuron_age):
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("native importance and age must be finite and nonnegative")


def _summarize_nonnegative(values: Array) -> VectorSummary:
    maximum = float(np.max(values))
    # Why this: summing several valid near-float64-max observations can overflow.
    mean = 0.0 if maximum == 0.0 else maximum * float(np.mean(values / maximum))
    return VectorSummary(len(values), float(np.min(values)), mean, maximum)


def summarize_unit_history(
    neuron_ids: UnitIds,
    parent_ids: UnitIds,
    importance_ema: Array,
    neuron_age: Array,
    *,
    next_neuron_id: int,
) -> UnitHistoryDiagnostics:
    """Detach observed native histories while preserving their unit order.

    Parent IDs may name pruned units, but must precede their child's stable ID.
    Age remains the native float observation, with no integer or wall-time
    interpretation. This function establishes neither provenance nor importance
    for models that do not actually store these fields.
    """
    _validate_unit_vectors(neuron_ids, parent_ids, importance_ema, neuron_age)
    if type(next_neuron_id) is not int or next_neuron_id <= 0:
        raise ValueError("next_neuron_id must be a positive native integer")
    ids = tuple(int(value) for value in neuron_ids)
    parents = tuple(int(value) for value in parent_ids)
    if len(set(ids)) != len(ids) or any(value < 0 or value >= next_neuron_id for value in ids):
        raise ValueError("neuron IDs must be unique nonnegative integers below next_neuron_id")
    if any(parent < -1 or parent >= child for parent, child in zip(parents, ids)):
        raise ValueError("parent IDs must be -1 or nonnegative IDs preceding the child")
    units = tuple(
        UnitHistory(neuron_id, None if parent == -1 else parent, float(importance), float(age))
        for neuron_id, parent, importance, age in zip(ids, parents, importance_ema, neuron_age)
    )
    return UnitHistoryDiagnostics(
        units, _summarize_nonnegative(importance_ema), _summarize_nonnegative(neuron_age)
    )
