"""Owned one-unit prune transfer for exact native coupled PC/CPC models.

Inputs are quiescent native models and explicit hidden coordinates. Outputs own
complete parameters/wake state and copied coordinate lineage. This bounded
utility changes neither native constructor domains nor defaults. It provides no
damage decision, optimizer state, runtime authority, prediction/energy guarantee,
or future-wake trajectory guarantee. No training or inference is performed.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_predictive_coding import (
    CoupledPredictiveCodingNetwork,
    CoupledSnapshot,
    CoupledWakeState,
)

MAX_HIDDEN_LAYERS = 16
MAX_PARAMETER_ENTRIES = 1_000_000


def _geometry(
    input_dim: int, hidden_dims: tuple[int, ...], precisions: tuple[float, ...]
) -> tuple[int, tuple[int, ...], tuple[float, ...]]:
    if type(input_dim) is not int or input_dim < 1:
        raise ValueError("input_dim must be a native positive integer")
    if type(hidden_dims) is not tuple or not 1 <= len(hidden_dims) <= MAX_HIDDEN_LAYERS:
        raise ValueError("hidden_dims must be an exact tuple with 1..16 layers")
    if any(type(width) is not int or width < 1 for width in hidden_dims):
        raise ValueError("hidden widths must be native positive integers")
    previous, entries = input_dim, 1  # Include the scalar output bias.
    for width in hidden_dims:
        entries += previous * width + width
        if entries > MAX_PARAMETER_ENTRIES:
            raise ValueError("coupled pruning exceeds the 1000000 parameter-entry local limit")
        previous = width
    if entries + previous > MAX_PARAMETER_ENTRIES:
        raise ValueError("coupled pruning exceeds the 1000000 parameter-entry local limit")
    if type(precisions) is not tuple or len(precisions) != len(hidden_dims):
        raise ValueError("precisions must be an exact tuple matching hidden depth")
    if any(
        type(value) not in (float, int) or not 1e-6 <= value <= 1e6 or not isfinite(value)
        for value in precisions
    ):
        raise ValueError("precisions must be native finite numbers in [1e-6, 1e6]")
    return (
        input_dim,
        tuple(width for width in hidden_dims),
        tuple(float(value) for value in precisions),
    )


def _coordinates(hidden_dims: tuple[int, ...], layer_index: int, unit_index: int) -> None:
    if type(layer_index) is not int or not 0 <= layer_index < len(hidden_dims):
        raise ValueError("layer_index must be a native integer within hidden depth")
    width = hidden_dims[layer_index]
    if width < 2:
        raise ValueError("the pruned hidden layer must have width at least 2")
    if type(unit_index) is not int or not 0 <= unit_index < width:
        raise ValueError("unit_index must be a native integer within the pruned layer")


@dataclass(frozen=True)
class CoupledPruneLineage:
    """Copied topology/coordinate metadata; no data, budget or serving authority."""

    format_version: int
    family: str
    input_dim: int
    source_hidden_dims: tuple[int, ...]
    precisions: tuple[float, ...]
    layer_index: int
    unit_index: int

    def __post_init__(self) -> None:
        input_dim, hidden_dims, precisions = _validate_lineage(self)
        object.__setattr__(self, "input_dim", input_dim)
        object.__setattr__(self, "source_hidden_dims", hidden_dims)
        object.__setattr__(self, "precisions", precisions)

    @property
    def target_hidden_dims(self) -> tuple[int, ...]:
        _, hidden_dims, _ = _validate_lineage(self)
        return tuple(
            width - 1 if layer == self.layer_index else width
            for layer, width in enumerate(hidden_dims)
        )


def _validate_lineage(
    lineage: CoupledPruneLineage,
) -> tuple[int, tuple[int, ...], tuple[float, ...]]:
    if (
        type(lineage) is not CoupledPruneLineage
        or type(lineage.format_version) is not int
        or lineage.format_version != 1
    ):
        raise ValueError("unsupported coupled pruning lineage format")
    if type(lineage.family) is not str or lineage.family not in ("pc", "cpc"):
        raise ValueError("coupled pruning lineage family must be pc or cpc")
    geometry = _geometry(lineage.input_dim, lineage.source_hidden_dims, lineage.precisions)
    _coordinates(geometry[1], lineage.layer_index, lineage.unit_index)
    return geometry


@dataclass(frozen=True)
class CoupledPruneChange:
    model: CoupledPredictiveCodingNetwork
    lineage: CoupledPruneLineage


def _model_geometry(
    model: CoupledPredictiveCodingNetwork,
) -> tuple[str, int, tuple[int, ...], tuple[float, ...]]:
    if type(model) is CoupledPredictiveCodingNetwork:
        family = "pc"
    elif type(model) is CoupledCircadianPredictiveCodingNetwork:
        family = "cpc"
    else:
        raise ValueError("model must be an exact native coupled PC or CPC model")
    input_dim, hidden_dims, precisions = _geometry(
        model.input_dim, model.hidden_dims, model.precisions
    )
    return family, input_dim, hidden_dims, precisions


def restore_coupled_model(
    family: str,
    input_dim: int,
    hidden_dims: tuple[int, ...],
    precisions: tuple[float, ...],
    snapshot: CoupledSnapshot,
) -> CoupledPredictiveCodingNetwork:
    """Reconstruct bounded native geometry and validate its complete snapshot."""
    _geometry(input_dim, hidden_dims, precisions)
    if (
        type(snapshot) is not CoupledSnapshot
        or type(snapshot.format_version) is not int
        or snapshot.format_version != 1
    ):
        raise ValueError("coupled pruning requires a full format-1 snapshot")
    if family == "pc":
        model = CoupledPredictiveCodingNetwork(
            input_dim, hidden_dims, seed=0, precisions=precisions
        )
    elif family == "cpc":
        wake = snapshot.wake_state
        if type(wake) is not CoupledWakeState:
            raise ValueError("CPC transfer requires complete native wake state")
        # Rebuild the policy from the captured values; never share source.config.
        config = CoupledCircadianConfig(
            wake.chemical_decay,
            wake.chemical_buildup_rate,
            wake.plasticity_sensitivity,
            wake.min_plasticity,
        )
        model = CoupledCircadianPredictiveCodingNetwork(
            input_dim, hidden_dims, seed=0, precisions=precisions, config=config
        )
    else:
        raise ValueError("unsupported native coupled family")
    model.restore_state(snapshot)
    return model


def copy_coupled_model(model: CoupledPredictiveCodingNetwork) -> CoupledPredictiveCodingNetwork:
    """Copy complete compatible native state after bounded geometry preflight."""
    family, input_dim, hidden_dims, precisions = _model_geometry(model)
    snapshot = model.snapshot_state()
    return restore_coupled_model(family, input_dim, hidden_dims, precisions, snapshot)


def _pruned_parameters(
    parameters: CoupledParameters, layer_index: int, unit_index: int
) -> CoupledParameters:
    weights = [weight.copy() for weight in parameters.hidden_weights]
    biases = [bias.copy() for bias in parameters.hidden_biases]
    weights[layer_index] = np.delete(weights[layer_index], unit_index, axis=1)
    biases[layer_index] = np.delete(biases[layer_index], unit_index, axis=1)
    output_weight = parameters.output_weight.copy()
    if layer_index + 1 < len(weights):
        weights[layer_index + 1] = np.delete(weights[layer_index + 1], unit_index, axis=0)
    else:
        output_weight = np.delete(output_weight, unit_index, axis=0)
    return CoupledParameters(
        tuple(weights), tuple(biases), output_weight, parameters.output_bias.copy()
    )


def _pruned_wake(
    wake: CoupledWakeState | None, layer_index: int, unit_index: int, depth: int
) -> CoupledWakeState | None:
    if wake is None:
        return None
    chemicals, gates = wake.chemical_state.copy(), wake.plasticity_state.copy()
    if layer_index == depth - 1:
        chemicals, gates = np.delete(chemicals, unit_index), np.delete(gates, unit_index)
    return CoupledWakeState(
        wake.chemical_decay,
        wake.chemical_buildup_rate,
        wake.plasticity_sensitivity,
        wake.min_plasticity,
        chemicals,
        gates,
        wake.wake_updates,
    )


def prune_coupled_width(
    model: CoupledPredictiveCodingNetwork, layer_index: int, unit_index: int
) -> CoupledPruneChange:
    """Prune one unit, preserving every retained tensor and associated wake value.

    No general prediction, energy, or future-wake preservation is promised.
    Native models have no optimizer/momentum state to transfer. Source arrays
    remain unchanged even when geometry, state, or target restore is refused.
    """
    family, input_dim, hidden_dims, precisions = _model_geometry(model)
    _coordinates(hidden_dims, layer_index, unit_index)
    lineage = CoupledPruneLineage(
        1, family, input_dim, hidden_dims, precisions, layer_index, unit_index
    )
    target_dims = lineage.target_hidden_dims
    _geometry(input_dim, target_dims, precisions)
    # A complete native restore validates the source before any tensor selection.
    source = copy_coupled_model(model)
    captured = source.snapshot_state()
    parameters = _pruned_parameters(captured.parameters, layer_index, unit_index)
    wake = _pruned_wake(captured.wake_state, layer_index, unit_index, len(hidden_dims))
    snapshot = CoupledSnapshot(1, parameters, precisions, wake)
    target = restore_coupled_model(family, input_dim, target_dims, precisions, snapshot)
    return CoupledPruneChange(target, lineage)
