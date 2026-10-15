"""Owned uniform-width pruning in raw coupled PC/CPC coordinates.

One coordinate is removed from every hidden layer. Selection happens once in
effective coordinates, then retained weights AND biases are re-encoded under the
new width's factors. Complete candidate validation and a strict representation
roundtrip guard precede return. This guard bounds re-encoding error only; pruning
itself can change predictions/energy, and future raw SGD factors change too.
No optimizer, runtime authority, training, inference, or energy evaluation lives here.
"""

from dataclasses import dataclass, replace

import numpy as np

from src.core.coupled_energy import Array, CoupledParameters
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    CoupledScalingState,
    _copy_config,
    _validated_model,
    effective_coupled_parameters,
)
from src.core.coupled_predictive_coding import CoupledWakeState, _parameter_arrays

MAX_EFFECTIVE_RELATIVE_ERROR = float(8 * np.finfo(np.float64).eps)


def _removed(config: CoupledScalingConfig, removed_units: tuple[int, ...]) -> tuple[int, ...]:
    if config.width < 2:
        raise ValueError("raw coupled pruning requires a source width of at least 2")
    if type(removed_units) is not tuple or len(removed_units) != config.hidden_layers:
        raise ValueError(
            "removed_units must be an exact tuple with one coordinate per hidden layer"
        )
    if any(type(index) is not int or not 0 <= index < config.width for index in removed_units):
        raise ValueError("removed unit coordinates must be native integers within the source width")
    return tuple(index for index in removed_units)


@dataclass(frozen=True)
class CoupledScalingPruneLineage:
    """Owned coordinate/configuration metadata; no model or serving authority."""

    format_version: int
    source_config: CoupledScalingConfig
    removed_units: tuple[int, ...]

    def __post_init__(self) -> None:
        source, removed = _validate_lineage(self)
        object.__setattr__(self, "source_config", source)
        object.__setattr__(self, "removed_units", removed)

    @property
    def target_config(self) -> CoupledScalingConfig:
        source, _ = _validate_lineage(self)
        return replace(source, width=source.width - 1)

    @property
    def retained_indices(self) -> tuple[tuple[int, ...], ...]:
        source, removed = _validate_lineage(self)
        return tuple(
            tuple(index for index in range(source.width) if index != unit) for unit in removed
        )


def _validate_lineage(
    lineage: CoupledScalingPruneLineage,
) -> tuple[CoupledScalingConfig, tuple[int, ...]]:
    if (
        type(lineage) is not CoupledScalingPruneLineage
        or type(lineage.format_version) is not int
        or lineage.format_version != 1
    ):
        raise ValueError("unsupported raw coupled pruning lineage format")
    source = _copy_config(lineage.source_config)
    removed = _removed(source, lineage.removed_units)
    # Existing config validation includes depth and total weights/biases caps;
    # also validate the changed factor domain before any selector allocation.
    replace(source, width=source.width - 1)
    return source, removed


@dataclass(frozen=True)
class CoupledScalingPruneChange:
    state: CoupledScalingState
    lineage: CoupledScalingPruneLineage
    max_effective_relative_error: float


def _select_parameters(
    parameters: CoupledParameters, retained: tuple[tuple[int, ...], ...]
) -> CoupledParameters:
    weights = []
    for layer, weight in enumerate(parameters.hidden_weights):
        selected = np.take(weight, retained[layer], axis=1)
        if layer:
            selected = np.take(selected, retained[layer - 1], axis=0)
        weights.append(selected)
    return CoupledParameters(
        tuple(weights),
        tuple(
            np.take(bias, keep, axis=1) for bias, keep in zip(parameters.hidden_biases, retained)
        ),
        np.take(parameters.output_weight, retained[-1], axis=0),
        parameters.output_bias.copy(),
    )


def _divide(values: Array, scale: float) -> Array:
    # Direct division avoids a nonrepresentable reciprocal even when the desired
    # quotient is representable. Later reconstruction checks every retained entry.
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        raw = values / scale
    if not np.isfinite(raw).all():
        raise ValueError("raw coupled pruning produced nonfinite re-encoded parameters")
    if np.any((values != 0) & (raw == 0)):
        raise ValueError("raw coupled pruning lost a nonzero value during re-encoding")
    return raw


def _encode(config: CoupledScalingConfig, selected: CoupledParameters) -> CoupledParameters:
    scales, bias = config.forward_scales, config.bias_scale
    return CoupledParameters(
        tuple(_divide(value, scale) for value, scale in zip(selected.hidden_weights, scales[:-1])),
        tuple(_divide(value, bias) for value in selected.hidden_biases),
        _divide(selected.output_weight, scales[-1]),
        _divide(selected.output_bias, bias),
    )


def _selected_wake(
    wake: CoupledWakeState | None, retained_last: tuple[int, ...]
) -> CoupledWakeState | None:
    if wake is None:
        return None
    return CoupledWakeState(
        wake.chemical_decay,
        wake.chemical_buildup_rate,
        wake.plasticity_sensitivity,
        wake.min_plasticity,
        np.take(wake.chemical_state, retained_last),
        np.take(wake.plasticity_state, retained_last),
        wake.wake_updates,
    )


def _relative_error(expected: Array, reconstructed: Array) -> float:
    nonzero = expected != 0
    if np.any((~nonzero) & (reconstructed != 0)):
        raise ValueError("raw coupled pruning roundtrip changed an exact zero")
    if np.any(nonzero & (reconstructed == 0)):
        raise ValueError("raw coupled pruning roundtrip lost a nonzero effective value")
    if np.any(nonzero & (np.signbit(expected) != np.signbit(reconstructed))):
        raise ValueError("raw coupled pruning roundtrip changed a nonzero sign")
    if not np.any(nonzero):
        return 0.0
    # Equal-sign subtraction cannot overflow. Relative division may overflow only
    # for an actual incompatible error, which is refused rather than truncated.
    with np.errstate(over="ignore", invalid="raise", divide="raise", under="ignore"):
        relative = np.abs((reconstructed[nonzero] - expected[nonzero]) / expected[nonzero])
    error = float(np.max(relative))
    if not np.isfinite(error) or error > MAX_EFFECTIVE_RELATIVE_ERROR:
        raise ValueError(
            "raw coupled pruning exceeds the 8-epsilon effective roundtrip error bound"
        )
    return error


def prune_coupled_scaling_width(
    initial: CoupledScalingState, removed_units: tuple[int, ...]
) -> CoupledScalingPruneChange:
    """Prune one unit per hidden layer as one owned uniform-width replacement.

    Borrowed input arrays must be quiescent. Retained effective tensors pass a
    representation-only relative bound of 8*float64 epsilon with absolute bound
    zero; signed-zero signs are not preserved by contract. The source remains
    untouched on every refusal. No intermediate nonuniform state is published.
    """
    if type(initial) is not CoupledScalingState:
        raise ValueError("initial must be an exact CoupledScalingState")
    source = _copy_config(initial.config)
    lineage = CoupledScalingPruneLineage(1, source, removed_units)
    target = lineage.target_config
    # Full native/raw/wake validation; compute the old effective parameters once.
    _, effective, _ = _validated_model(initial)
    retained = lineage.retained_indices
    selected = _select_parameters(effective, retained)
    raw = _encode(target, selected)
    wake = _selected_wake(initial.wake_state, retained[-1])
    candidate = CoupledScalingState(target, raw, wake)
    reconstructed = effective_coupled_parameters(candidate)
    error = max(
        _relative_error(expected, actual)
        for expected, actual in zip(_parameter_arrays(selected), _parameter_arrays(reconstructed))
    )
    return CoupledScalingPruneChange(candidate, lineage, error)
