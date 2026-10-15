"""Owned isometric width splits and compatible projected merges for linear PC.

A common hidden coordinate is duplicated with factors 1/sqrt(2) in every layer.
Residual branches transform identically; their fixed identity complement remains
implicit. No dense isometry/identity, equilibrium evaluation, optimizer, traces,
or RNG is needed. Borrowed state arrays must be quiescent during an operation.
"""

from dataclasses import dataclass, replace
from math import sqrt

import numpy as np
from numpy.typing import NDArray

from src.core.linear_pc_reference import Array
from src.core.linear_pc_scaling import LinearScalingConfig, LinearScalingState, _validate_weights
from src.core.residual_linear_pc import LinearResidualConfig, LinearResidualState

Config = LinearScalingConfig | LinearResidualConfig
State = LinearScalingState | LinearResidualState
Exponent = NDArray[np.int32]
MERGE_RTOL = 1e-10
MERGE_ATOL = 1e-12
_SPLIT_FACTOR = 1 / sqrt(2)


def _copy_config(config: Config, family: str) -> Config:
    expected = LinearScalingConfig if family == "mlp" else LinearResidualConfig
    if type(config) is not expected:
        raise ValueError("configuration family differs from the declared linear family")
    return replace(config)


def _index(index: int, width: int) -> None:
    if type(index) is not int or not 0 <= index < width:
        raise ValueError("split_index must be a native integer within the source width")


@dataclass(frozen=True)
class LinearWidthLineage:
    """Coordinate lineage only; no model provenance or runtime authority."""

    format_version: int
    family: str
    source_config: Config
    split_index: int

    def __post_init__(self) -> None:
        config = _validate_lineage(self)
        object.__setattr__(self, "source_config", config)


def _validate_lineage(lineage: LinearWidthLineage) -> Config:
    if (
        type(lineage) is not LinearWidthLineage
        or type(lineage.format_version) is not int
        or lineage.format_version != 1
    ):
        raise ValueError("unsupported linear width lineage format")
    if type(lineage.family) is not str or lineage.family not in ("mlp", "residual"):
        raise ValueError("lineage family must be mlp or residual")
    config = _copy_config(lineage.source_config, lineage.family)
    _index(lineage.split_index, config.width)
    replace(config, width=config.width + 1)
    return config


@dataclass(frozen=True)
class LinearWidthChange:
    state: State
    lineage: LinearWidthLineage


def _inspect_state(state: State) -> tuple[str, Config]:
    if type(state) is LinearScalingState:
        family = "mlp"
    elif type(state) is LinearResidualState:
        family = "residual"
    else:
        raise ValueError("state must be an exact supported linear state")
    config = _copy_config(state.config, family)
    base = LinearScalingConfig(
        config.scheme, config.input_dim, config.width, config.hidden_layers, config.gamma0
    )
    _validate_weights(base, state.raw_weights)
    return family, config


def _finish_product(fraction: Array, exponent: Exponent, *factors: float | Array) -> Array:
    for factor in factors:
        factor_fraction, factor_exponent = np.frexp(factor)
        fraction = fraction * factor_fraction
        exponent = exponent + factor_exponent
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        result = np.ldexp(fraction, exponent)
    if not np.isfinite(result).all():
        raise ValueError("linear width transformation produced nonfinite weights")
    return result


def _product(value: Array, *factors: float | Array) -> Array:
    fraction, exponent = np.frexp(value)
    return _finish_product(fraction, exponent, *factors)


def _validate_effective(state: State) -> None:
    for layer, (raw, scale) in enumerate(zip(state.raw_weights, state.config.forward_scales)):
        branch = _product(raw, scale)
        if type(state) is LinearResidualState and 0 < layer < len(state.raw_weights) - 1:
            # Check complete diagonal values without allocating an identity map.
            with np.errstate(over="raise", invalid="raise"):
                diagonal = branch.diagonal() + 1.0
            if not np.isfinite(diagonal).all():
                raise ValueError("complete residual matrix became nonfinite")


def _widen_weights(
    weights: tuple[Array, ...], source: Config, target: Config, index: int
) -> tuple[Array, ...]:
    coordinates = np.append(np.arange(source.width), index)
    gains = np.ones(target.width, dtype=np.float64)
    gains[index] = gains[-1] = _SPLIT_FACTOR
    widened = []
    for layer, (raw, old_scale, new_scale) in enumerate(
        zip(weights, source.forward_scales, target.forward_scales)
    ):
        value = raw
        factors: list[float | Array] = [old_scale, 1 / new_scale]
        if layer > 0:
            value = np.take(value, coordinates, axis=0)
            factors.append(gains[:, None])
        if layer < len(weights) - 1:
            value = np.take(value, coordinates, axis=1)
            factors.append(gains[None, :])
        widened.append(_product(value, *factors))
    return tuple(widened)


def _merge_axis(
    fraction: Array, exponent: Exponent, index: int, axis: int
) -> tuple[Array, Exponent]:
    left_fraction = np.take(fraction, index, axis=axis)
    right_fraction = np.take(fraction, -1, axis=axis)
    left_exponent = np.take(exponent, index, axis=axis)
    right_exponent = np.take(exponent, -1, axis=axis)
    # frexp(0) has exponent zero, which must not dominate a nonzero subnormal
    # partner and force that partner to round before final scale compensation.
    active_left = np.where(left_fraction == 0, right_exponent, left_exponent)
    active_right = np.where(right_fraction == 0, left_exponent, right_exponent)
    common = np.maximum(active_left, active_right)
    with np.errstate(under="ignore", over="raise", invalid="raise"):
        pair = (
            np.ldexp(left_fraction, left_exponent - common)
            + np.ldexp(right_fraction, right_exponent - common)
        ) * _SPLIT_FACTOR
    merged_fraction, shift = np.frexp(pair)
    merged_exponent = common + shift
    result_fraction = np.take(fraction, np.arange(fraction.shape[axis] - 1), axis=axis)
    result_exponent = np.take(exponent, np.arange(exponent.shape[axis] - 1), axis=axis)
    position = (index, slice(None)) if axis == 0 else (slice(None), index)
    result_fraction[position] = merged_fraction
    result_exponent[position] = merged_exponent
    return result_fraction, result_exponent


def _project_weights(
    weights: tuple[Array, ...], source: Config, target: Config, index: int
) -> tuple[Array, ...]:
    projected = []
    for layer, (raw, old_scale, new_scale) in enumerate(
        zip(weights, source.forward_scales, target.forward_scales)
    ):
        fraction, exponent = np.frexp(raw)
        if layer > 0:
            fraction, exponent = _merge_axis(fraction, exponent, index, 0)
        if layer < len(weights) - 1:
            fraction, exponent = _merge_axis(fraction, exponent, index, 1)
        projected.append(_finish_product(fraction, exponent, old_scale, 1 / new_scale))
    return tuple(projected)


def _make_state(family: str, config: Config, weights: tuple[Array, ...]) -> State:
    if family == "mlp" and type(config) is LinearScalingConfig:
        state: State = LinearScalingState(config, weights)
    elif family == "residual" and type(config) is LinearResidualConfig:
        state = LinearResidualState(config, weights)
    else:
        raise ValueError("linear state and configuration family differ")
    _validate_effective(state)
    return state


def split_linear_width(state: State, index: int) -> LinearWidthChange:
    """Widen every hidden layer isometrically and return detached coordinate lineage.

    New width-dependent SGD factors apply to continuation; future training
    trajectories are not claimed to match those of the source configuration.
    """
    family, source = _inspect_state(state)
    _index(index, source.width)
    # Validate the widened allocation before gathering any enlarged matrices.
    target = replace(source, width=source.width + 1)
    _validate_effective(state)
    weights = _widen_weights(state.raw_weights, source, target, index)
    widened = _make_state(family, target, weights)
    return LinearWidthChange(widened, LinearWidthLineage(1, family, source, index))


def merge_linear_width(state: State, lineage: LinearWidthLineage) -> State:
    """Project a compatible declared split image into an owned source-width state.

    Re-expansion must match scaled branches at rtol=1e-10/atol=1e-12. This is a
    local projection tolerance, not a global function/energy error bound: small
    discarded components can be amplified by downstream matrices.
    """
    original = _validate_lineage(lineage)
    family, current = _inspect_state(state)
    expected = replace(original, width=original.width + 1)
    if family != lineage.family or type(current) is not type(expected) or current != expected:
        raise ValueError("state configuration differs from the declared widened lineage")
    _validate_effective(state)
    projected = _project_weights(state.raw_weights, current, original, lineage.split_index)
    candidate = _make_state(family, original, projected)
    reexpanded = _widen_weights(projected, original, current, lineage.split_index)
    for raw, image, scale in zip(state.raw_weights, reexpanded, current.forward_scales):
        branch, projected_branch = _product(raw, scale), _product(image, scale)
        with np.errstate(over="ignore", invalid="ignore"):
            compatible = np.allclose(projected_branch, branch, rtol=MERGE_RTOL, atol=MERGE_ATOL)
        if not compatible:
            raise ValueError("state branches contain components outside the declared split image")
    return candidate
