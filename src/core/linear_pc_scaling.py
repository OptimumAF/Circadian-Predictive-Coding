"""Explicit initialization and raw-coordinate updates for linear PC controls.

Inputs are fixed scalar-output architectures, seeds, raw float64 weights and
clamped training batches. Outputs own their arrays. Mean-field forward/rate
factors follow the pinned author formulation; the two controls are explicit
local baselines. No nonlinear learner, biases, skips, chemistry, study loop or
default parameterization changes live here.
"""

from dataclasses import dataclass
from math import frexp, isfinite, sqrt

import numpy as np

from src.core.linear_pc_reference import Array, LinearEquilibrium, evaluate_linear_equilibrium

MAX_WEIGHT_ENTRIES = 1_000_000


def _positive_number(value: float, name: str) -> float:
    if type(value) not in (float, int):
        raise ValueError(f"{name} must be a native positive finite number")
    try:
        resolved = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    if not isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return resolved


@dataclass(frozen=True)
class LinearScalingConfig:
    scheme: str
    input_dim: int
    width: int
    hidden_layers: int
    gamma0: float = 1.0

    def __post_init__(self) -> None:
        _configuration_values(self)

    @property
    def forward_scales(self) -> tuple[float, ...]:
        return _configuration_values(self)[0]

    @property
    def initial_stddevs(self) -> tuple[float, ...]:
        return _configuration_values(self)[1]

    @property
    def raw_learning_rate_factor(self) -> float:
        return _configuration_values(self)[2]

    @property
    def effective_learning_rate_factors(self) -> tuple[float, ...]:
        scales, _, factor = _configuration_values(self)
        return tuple(factor * scale * scale for scale in scales)


def _configuration_values(
    config: LinearScalingConfig,
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    if type(config) is not LinearScalingConfig:
        raise ValueError("config must be an exact LinearScalingConfig")
    if type(config.scheme) is not str or config.scheme not in (
        "legacy_gaussian",
        "fan_in",
        "mean_field",
    ):
        raise ValueError("scheme must be legacy_gaussian, fan_in, or mean_field")
    for name in ("input_dim", "width", "hidden_layers"):
        value = getattr(config, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a native positive integer")
    dim, width, hidden = config.input_dim, config.width, config.hidden_layers
    entries = dim * width + (hidden - 1) * width * width + width
    if entries > MAX_WEIGHT_ENTRIES:
        raise ValueError("architecture exceeds the 1000000 raw-weight-entry local allocation limit")
    gamma = _positive_number(config.gamma0, "gamma0")
    if config.scheme != "mean_field" and gamma != 1:
        raise ValueError("control schemes require gamma0 equal to 1")
    depth = hidden + 1
    if config.scheme == "legacy_gaussian":
        scales, stddevs, factor = (1.0,) * depth, (0.5,) * depth, 1.0
    elif config.scheme == "fan_in":
        scales = (1.0,) * depth
        stddevs, factor = (1 / sqrt(dim),) + (1 / sqrt(width),) * hidden, 1.0
    else:
        factor = gamma * gamma * width
        if factor == 0.0:
            # Preserve a representable final factor when gamma² alone falls
            # below the float64 subnormal range. Ordinary settings retain the
            # direct author expression, including its normal rounding behavior.
            gamma_width = gamma * sqrt(width)
            factor = gamma_width * gamma_width
        factor = _positive_number(factor, "raw learning-rate factor")
        scales = (1 / sqrt(dim),) + (1 / sqrt(width),) * (hidden - 1) + (1 / (gamma * width),)
        stddevs = (1.0,) * depth
    for scale in scales:
        _positive_number(scale, "forward scale")
        _positive_number(factor * scale * scale, "effective learning-rate factor")
    return scales, stddevs, factor


def _shapes(config: LinearScalingConfig) -> tuple[tuple[int, int], ...]:
    return (
        ((config.input_dim, config.width),)
        + ((config.width, config.width),) * (config.hidden_layers - 1)
        + ((config.width, 1),)
    )


def _validate_weights(config: LinearScalingConfig, weights: tuple[Array, ...]) -> None:
    _configuration_values(config)
    if type(weights) is not tuple or len(weights) != config.hidden_layers + 1:
        raise ValueError("raw_weights must be an exact tuple matching configured depth")
    for index, (weight, shape) in enumerate(zip(weights, _shapes(config))):
        if (
            type(weight) is not np.ndarray
            or weight.dtype != np.float64
            or weight.shape != shape
            or not np.isfinite(weight).all()
        ):
            raise ValueError(f"raw weight {index} must be a finite float64 matrix of shape {shape}")


@dataclass(frozen=True)
class LinearScalingState:
    """Owned mutable arrays in a frozen record; borrowed sources must be quiescent."""

    config: LinearScalingConfig
    raw_weights: tuple[Array, ...]

    def __post_init__(self) -> None:
        _validate_weights(self.config, self.raw_weights)
        object.__setattr__(self, "raw_weights", tuple(weight.copy() for weight in self.raw_weights))


@dataclass(frozen=True)
class LinearScalingStep:
    """Replacement raw state and PREUPDATE equilibrium in effective coordinates."""

    state: LinearScalingState
    evaluation: LinearEquilibrium


def initialize_linear_scaling(config: LinearScalingConfig, seed: int) -> LinearScalingState:
    """Use identical standard-normal draws across schemes at the same shape/seed."""
    _, stddevs, _ = _configuration_values(config)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a native nonnegative integer")
    rng = np.random.default_rng(seed)
    weights = tuple(
        rng.standard_normal(shape) * stddev for shape, stddev in zip(_shapes(config), stddevs)
    )
    return LinearScalingState(config, weights)


def effective_linear_weights(state: LinearScalingState) -> tuple[Array, ...]:
    """Revalidate owned snapshot arrays and return detached effective weights."""
    if type(state) is not LinearScalingState:
        raise ValueError("state must be an exact LinearScalingState")
    _validate_weights(state.config, state.raw_weights)
    with np.errstate(over="raise", invalid="raise"):
        weights = tuple(
            scale * weight for scale, weight in zip(state.config.forward_scales, state.raw_weights)
        )
    if any(not np.isfinite(weight).all() for weight in weights):
        raise ValueError("effective linear weights became nonfinite")
    return weights


def _scaled_delta(gradient: Array, *factors: float) -> Array:
    # Why: forming rate*raw_factor first can underflow even when the complete
    # raw update is representable. Binary exponents delay rounding to the final
    # product, without requiring an extended-precision platform or dependency.
    fraction, exponent = np.frexp(gradient)
    for factor in factors:
        factor_fraction, factor_exponent = frexp(factor)
        fraction = fraction * factor_fraction
        exponent = exponent + factor_exponent
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        return np.ldexp(fraction, exponent)


def step_linear_pc(
    state: LinearScalingState, features: Array, targets: Array, learning_rate: float
) -> LinearScalingStep:
    """Return a simultaneous raw-coordinate step and its preupdate equilibrium.

    The chain rule gives grad_raw=a_l*grad_effective. Raw GD uses the configured
    rate multiplier; its effective-coordinate rate is multiplier*a_l². No input
    state is changed. Invalid candidate arithmetic raises ValueError or
    FloatingPointError before returning a replacement state.
    """
    rate = _positive_number(learning_rate, "learning_rate")
    effective = effective_linear_weights(state)
    evaluation = evaluate_linear_equilibrium(effective, features, targets)
    scales = state.config.forward_scales
    factor = state.config.raw_learning_rate_factor
    with np.errstate(over="raise", invalid="raise"):
        raw = tuple(
            weight - _scaled_delta(gradient, rate, factor, scale)
            for weight, gradient, scale in zip(
                state.raw_weights, evaluation.weight_gradients, scales
            )
        )
    candidate = LinearScalingState(state.config, raw)
    candidate_effective = effective_linear_weights(candidate)
    # Finite tensors alone do not guarantee a finite equilibrium or gradient.
    evaluate_linear_equilibrium(candidate_effective, features, targets)
    return LinearScalingStep(candidate, evaluation)
