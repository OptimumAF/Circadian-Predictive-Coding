"""Owned two-way uniform replication and exact-pair merge for raw coupled PC.

Why uniform replication: halving each scalar hidden precision compensates doubled
residual entries. On repeated latents, cloning incoming columns/biases and halving
middle/output contributions preserves represented logits and energy in exact
arithmetic. Every cloned latent gradient is halved, so a fixed inference trajectory
requires doubled latent step size. Guarded stops can differ with rounding and
tolerances. This is not a raw-bitwise, SGD-trajectory, learning-benefit, runtime
history, or authority guarantee. Borrowed input arrays must remain quiescent.
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
from src.core.coupled_scaling_pruning import (
    MAX_EFFECTIVE_RELATIVE_ERROR as MAX_EFFECTIVE_RELATIVE_ERROR,
    _encode,
    _relative_error,
)


def _target_config(source: CoupledScalingConfig) -> CoupledScalingConfig:
    assert source.precisions is not None
    halves = tuple(value / 2 for value in source.precisions)
    if any(
        value == 0 or value * 2 != original for value, original in zip(halves, source.precisions)
    ):
        raise ValueError("replicated hidden precisions must halve exactly and reversibly")
    return replace(source, width=2 * source.width, precisions=halves)


@dataclass(frozen=True)
class CoupledScalingSplitLineage:
    """Owned source configuration metadata; not a prior raw checkpoint or authority."""

    format_version: int
    source_config: CoupledScalingConfig

    def __post_init__(self) -> None:
        source, _ = _validate_lineage(self)
        object.__setattr__(self, "source_config", source)

    @property
    def target_config(self) -> CoupledScalingConfig:
        return _validate_lineage(self)[1]

    @property
    def replication_factor(self) -> int:
        _validate_lineage(self)
        return 2


def _validate_lineage(
    lineage: CoupledScalingSplitLineage,
) -> tuple[CoupledScalingConfig, CoupledScalingConfig]:
    if (
        type(lineage) is not CoupledScalingSplitLineage
        or type(lineage.format_version) is not int
        or lineage.format_version != 1
    ):
        raise ValueError("unsupported raw coupled split lineage format")
    source = _copy_config(lineage.source_config)
    # The existing config boundary validates the expanded parameter cap and
    # precision range before any expanded tensor/coordinate allocation.
    return source, _target_config(source)


@dataclass(frozen=True)
class CoupledScalingWidthChange:
    state: CoupledScalingState
    lineage: CoupledScalingSplitLineage
    max_effective_relative_error: float


def _half_exact(values: Array) -> Array:
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        half = values / 2.0
        restored = half * 2.0
    if not np.isfinite(half).all() or not np.array_equal(restored, values):
        raise ValueError("effective middle/output halving loses nonzero or subnormal information")
    return half


def _double_exact(values: Array) -> Array:
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        doubled = values * 2.0
    if not np.isfinite(doubled).all() or np.any((values != 0) & (doubled == 0)):
        raise ValueError("effective middle/output doubling is not representable")
    return doubled


def _replicated_parameters(parameters: CoupledParameters) -> CoupledParameters:
    weights = []
    for layer, weight in enumerate(parameters.hidden_weights):
        branch = weight if layer == 0 else _half_exact(weight)
        if layer:
            branch = np.repeat(branch, 2, axis=0)
        weights.append(np.repeat(branch, 2, axis=1))
    # Halve before expansion and check exact reversibility, rather than allowing
    # the later representation guard to compare against an already rounded loss.
    output = np.repeat(_half_exact(parameters.output_weight), 2, axis=0)
    return CoupledParameters(
        tuple(weights),
        tuple(np.repeat(bias, 2, axis=1) for bias in parameters.hidden_biases),
        output,
        parameters.output_bias.copy(),
    )


def _wake(wake: CoupledWakeState | None, *, split: bool) -> CoupledWakeState | None:
    if wake is None:
        return None
    chemicals = np.repeat(wake.chemical_state, 2) if split else wake.chemical_state[::2].copy()
    gates = np.repeat(wake.plasticity_state, 2) if split else wake.plasticity_state[::2].copy()
    return CoupledWakeState(
        wake.chemical_decay,
        wake.chemical_buildup_rate,
        wake.plasticity_sensitivity,
        wake.min_plasticity,
        chemicals,
        gates,
        wake.wake_updates,
    )


def _require_pairs(values: Array, axis: int, name: str) -> None:
    left, right = (values[::2], values[1::2]) if axis == 0 else (values[:, ::2], values[:, 1::2])
    if not np.array_equal(left, right):
        raise ValueError(f"merge requires exactly equal adjacent raw {name} pairs")


def _validate_pairs(initial: CoupledScalingState) -> None:
    parameters = initial.raw_parameters
    for layer, weight in enumerate(parameters.hidden_weights):
        _require_pairs(weight, 1, f"hidden weight {layer} column")
        if layer:
            _require_pairs(weight, 0, f"hidden weight {layer} row")
    for layer, bias in enumerate(parameters.hidden_biases):
        _require_pairs(bias, 1, f"hidden bias {layer}")
    _require_pairs(parameters.output_weight, 0, "output weight row")
    if initial.wake_state is not None:
        _require_pairs(initial.wake_state.chemical_state, 0, "chemical")
        _require_pairs(initial.wake_state.plasticity_state, 0, "plasticity")


def _merged_parameters(parameters: CoupledParameters) -> CoupledParameters:
    weights = tuple(
        weight[:, ::2].copy() if layer == 0 else _double_exact(weight[::2, ::2])
        for layer, weight in enumerate(parameters.hidden_weights)
    )
    return CoupledParameters(
        weights,
        tuple(bias[:, ::2].copy() for bias in parameters.hidden_biases),
        _double_exact(parameters.output_weight[::2]),
        parameters.output_bias.copy(),
    )


def _validated_candidate(
    config: CoupledScalingConfig, expected: CoupledParameters, wake: CoupledWakeState | None
) -> tuple[CoupledScalingState, float]:
    raw = _encode(config, expected)
    candidate = CoupledScalingState(config, raw, wake)
    actual = effective_coupled_parameters(candidate)
    error = max(
        _relative_error(reference, reconstructed)
        for reference, reconstructed in zip(_parameter_arrays(expected), _parameter_arrays(actual))
    )
    return candidate, error


def split_coupled_scaling_width(initial: CoupledScalingState) -> CoupledScalingWidthChange:
    """Replicate every hidden unit into adjacent pairs as one owned replacement.

    Exact reversible effective halving precedes the shared 8-epsilon relative /
    zero-absolute raw representation guard. Raw information lost in the existing
    raw-to-effective decoding is not recoverable; no raw-bitwise promise is made.
    """
    if type(initial) is not CoupledScalingState:
        raise ValueError("initial must be an exact CoupledScalingState")
    source = _copy_config(initial.config)
    lineage = CoupledScalingSplitLineage(1, source)
    target = lineage.target_config
    _, effective, _ = _validated_model(initial)
    expected = _replicated_parameters(effective)
    candidate, error = _validated_candidate(target, expected, _wake(initial.wake_state, split=True))
    return CoupledScalingWidthChange(candidate, lineage, error)


def merge_coupled_scaling_width(
    initial: CoupledScalingState, lineage: CoupledScalingSplitLineage
) -> CoupledScalingWidthChange:
    """Merge an exact full-raw pair image under its complete declared config.

    Symmetric trained states may merge; metadata does not certify provenance or
    a prior raw checkpoint. Raw middle row AND column groups, biases, outgoing
    rows, chemistry and gates must all match exactly. Signed-zero signs are not
    promised. No existing topology precision pin or runtime history is changed.
    """
    source, target = _validate_lineage(lineage)
    if type(initial) is not CoupledScalingState:
        raise ValueError("initial must be an exact CoupledScalingState")
    current = _copy_config(initial.config)
    if current != target:
        raise ValueError("merge state differs from the full declared replicated configuration")
    _, effective, _ = _validated_model(initial)
    _validate_pairs(initial)
    expected = _merged_parameters(effective)
    candidate, error = _validated_candidate(
        source, expected, _wake(initial.wake_state, split=False)
    )
    return CoupledScalingWidthChange(candidate, CoupledScalingSplitLineage(1, source), error)
