"""Fixed-identity residual linear PC with explicit raw-coordinate scaling.

Inputs are validated scalar-output configurations, seeds, raw float64 snapshots
and clamped batches. Outputs own their arrays. Middle effective matrices are
I+a_l W_l; identities are fixed and only raw branches receive gradient updates.
The mean-field depth factor follows paper Sec. 4, Eq. 19. Controls retain their
existing MLP scales/stddevs/rates. No bias, adaptive optimizer, nonlinear learner,
serialization, tuning, or empirical study is implemented in this module.
"""

from dataclasses import dataclass
from math import sqrt

import numpy as np

from src.core.linear_pc_reference import Array, LinearEquilibrium, evaluate_linear_equilibrium
from src.core.linear_pc_scaling import (
    LinearScalingConfig,
    _positive_number,
    _scaled_delta,
    _validate_weights,
    initialize_linear_scaling,
)


@dataclass(frozen=True)
class LinearResidualConfig:
    scheme: str
    input_dim: int
    width: int
    hidden_layers: int
    gamma0: float = 1.0

    def __post_init__(self) -> None:
        _configuration_values(self)

    @property
    def depth(self) -> int:
        base, _ = _configuration_values(self)
        return base.hidden_layers + 1

    @property
    def forward_scales(self) -> tuple[float, ...]:
        return _configuration_values(self)[1]

    @property
    def initial_stddevs(self) -> tuple[float, ...]:
        base, _ = _configuration_values(self)
        return base.initial_stddevs

    @property
    def raw_learning_rate_factor(self) -> float:
        base, _ = _configuration_values(self)
        return base.raw_learning_rate_factor

    @property
    def effective_learning_rate_factors(self) -> tuple[float, ...]:
        base, scales = _configuration_values(self)
        factor = base.raw_learning_rate_factor
        return tuple(factor * scale * scale for scale in scales)


def _configuration_values(
    config: LinearResidualConfig,
) -> tuple[LinearScalingConfig, tuple[float, ...]]:
    if type(config) is not LinearResidualConfig:
        raise ValueError("config must be an exact LinearResidualConfig")
    # Compose the accepted domain/allocation guard before constructing arrays.
    base = LinearScalingConfig(
        config.scheme, config.input_dim, config.width, config.hidden_layers, config.gamma0
    )
    scales = base.forward_scales
    if base.scheme == "mean_field" and base.hidden_layers > 1:
        middle = 1 / sqrt(base.width * (base.hidden_layers + 1))
        scales = (scales[0],) + (middle,) * (base.hidden_layers - 1) + (scales[-1],)
    factor = base.raw_learning_rate_factor
    for scale in scales:
        _positive_number(scale, "residual forward scale")
        _positive_number(factor * scale * scale, "residual effective learning-rate factor")
    return base, scales


@dataclass(frozen=True)
class LinearResidualState:
    """Owned raw arrays in a frozen record; borrowed inputs must be quiescent."""

    config: LinearResidualConfig
    raw_weights: tuple[Array, ...]

    def __post_init__(self) -> None:
        base, _ = _configuration_values(self.config)
        _validate_weights(base, self.raw_weights)
        object.__setattr__(self, "raw_weights", tuple(weight.copy() for weight in self.raw_weights))


@dataclass(frozen=True)
class LinearResidualStep:
    """Replacement raw state and PREUPDATE equilibrium in effective coordinates."""

    state: LinearResidualState
    evaluation: LinearEquilibrium


def initialize_residual_linear(config: LinearResidualConfig, seed: int) -> LinearResidualState:
    """Reuse exactly the existing MLP raw draws/stddevs for matching shape/seed."""
    base, _ = _configuration_values(config)
    initialized = initialize_linear_scaling(base, seed)
    return LinearResidualState(config, initialized.raw_weights)


def effective_residual_weights(state: LinearResidualState) -> tuple[Array, ...]:
    """Return detached complete matrices, including fixed middle identities."""
    if type(state) is not LinearResidualState:
        raise ValueError("state must be an exact LinearResidualState")
    base, scales = _configuration_values(state.config)
    _validate_weights(base, state.raw_weights)
    weights = []
    with np.errstate(over="raise", invalid="raise"):
        for index, (scale, raw) in enumerate(zip(scales, state.raw_weights)):
            effective = scale * raw
            if 0 < index < len(state.raw_weights) - 1:
                effective = np.eye(base.width, dtype=np.float64) + effective
            if not np.isfinite(effective).all():
                raise ValueError("effective residual linear weights became nonfinite")
            weights.append(effective)
    return tuple(weights)


def step_residual_linear_pc(
    state: LinearResidualState, X: Array, y: Array, learning_rate: float
) -> LinearResidualStep:
    """Return simultaneous raw SGD and the complete preupdate PC equilibrium.

    The equilibrium kernel sees complete effective matrices, including identity
    branches, in both predictions and the weight-dependent rescaling. Its local
    effective gradient maps to raw coordinates by multiplication with a_l; the
    fixed identity receives no update. Invalid candidate arithmetic raises
    ValueError or FloatingPointError without changing the input state.
    """
    rate = _positive_number(learning_rate, "learning_rate")
    effective = effective_residual_weights(state)
    evaluation = evaluate_linear_equilibrium(effective, X, y)
    base, scales = _configuration_values(state.config)
    factor = base.raw_learning_rate_factor
    with np.errstate(over="raise", invalid="raise"):
        raw = tuple(
            weight - _scaled_delta(gradient, rate, factor, scale)
            for weight, gradient, scale in zip(
                state.raw_weights, evaluation.weight_gradients, scales
            )
        )
    candidate = LinearResidualState(state.config, raw)
    # Finite raw tensors alone cannot establish a finite full equilibrium.
    candidate_effective = effective_residual_weights(candidate)
    evaluate_linear_equilibrium(candidate_effective, X, y)
    return LinearResidualStep(candidate, evaluation)
