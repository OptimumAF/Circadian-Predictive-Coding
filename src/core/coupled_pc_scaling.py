"""Opt-in raw-coordinate scaling for the existing nonlinear coupled objective.

Weight factors reuse linear controls, but applying them to tanh/BCE is an
explicit local extension, not a published nonlinear scaling theorem. Latents,
precisions and the native energy are unchanged. Bias scale c=1/sqrt(stored r)
targets effective bias SGD factor one; its actual float64 factor is reported.
Raw snapshots contain complete wake state; effective native snapshots cannot
substitute for them. No optimizer, runtime authority or scientific trial lives here.
"""

from dataclasses import dataclass
from math import frexp, ldexp, sqrt

import numpy as np

from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_energy import (
    Array,
    CoupledEvaluation,
    CoupledParameters,
    _validate_parameters,
    evaluate_coupled,
)
from src.core.coupled_inference import CoupledInferenceDiagnostics, CoupledInferencePolicy
from src.core.coupled_predictive_coding import (
    CoupledPredictiveCodingNetwork,
    CoupledSnapshot,
    CoupledWakeState,
)
from src.core.coupled_width_pruning import _geometry, restore_coupled_model
from src.core.linear_pc_scaling import LinearScalingConfig, _positive_number


def _scalar_product(*factors: float) -> float:
    fraction, exponent = 1.0, 0
    for factor in factors:
        part, shift = frexp(factor)
        fraction, exponent = fraction * part, exponent + shift
    try:
        result = ldexp(fraction, exponent)
    except OverflowError:
        raise ValueError("effective learning-rate factor is nonfinite") from None
    return _positive_number(result, "effective learning-rate factor")


@dataclass(frozen=True)
class CoupledScalingConfig:
    scheme: str
    input_dim: int
    width: int
    hidden_layers: int
    gamma0: float = 1.0
    precisions: tuple[float, ...] | None = None
    circadian: CoupledCircadianConfig | None = None

    def __post_init__(self) -> None:
        _, precisions, circadian, _ = _configuration(self)
        object.__setattr__(self, "precisions", precisions)
        object.__setattr__(self, "circadian", circadian)

    @property
    def forward_scales(self) -> tuple[float, ...]:
        return _configuration(self)[0].forward_scales

    @property
    def initial_stddevs(self) -> tuple[float, ...]:
        return _configuration(self)[0].initial_stddevs

    @property
    def raw_learning_rate_factor(self) -> float:
        return _configuration(self)[0].raw_learning_rate_factor

    @property
    def bias_scale(self) -> float:
        return _configuration(self)[3]

    @property
    def effective_learning_rate_factors(self) -> tuple[float, ...]:
        base, _, _, _ = _configuration(self)
        return tuple(
            _scalar_product(base.raw_learning_rate_factor, scale, scale)
            for scale in base.forward_scales
        )

    @property
    def effective_bias_learning_rate_factor(self) -> float:
        base, _, _, bias = _configuration(self)
        return _scalar_product(base.raw_learning_rate_factor, bias, bias)


def _configuration(
    config: CoupledScalingConfig,
) -> tuple[LinearScalingConfig, tuple[float, ...], CoupledCircadianConfig | None, float]:
    if type(config) is not CoupledScalingConfig:
        raise ValueError("config must be an exact CoupledScalingConfig")
    if type(config.hidden_layers) is not int or not 1 <= config.hidden_layers <= 16:
        raise ValueError("hidden_layers must be a native integer in 1..16")
    if type(config.width) is not int or config.width < 1:
        raise ValueError("width must be a native positive integer")
    precisions = (1.0,) * config.hidden_layers if config.precisions is None else config.precisions
    _, _, precisions = _geometry(
        config.input_dim, (config.width,) * config.hidden_layers, precisions
    )
    base = LinearScalingConfig(
        config.scheme, config.input_dim, config.width, config.hidden_layers, config.gamma0
    )
    circadian = config.circadian
    if circadian is not None:
        if type(circadian) is not CoupledCircadianConfig:
            raise ValueError("circadian must be an exact CoupledCircadianConfig or None")
        circadian = CoupledCircadianConfig(
            circadian.chemical_decay,
            circadian.chemical_buildup_rate,
            circadian.plasticity_sensitivity,
            circadian.min_plasticity,
        )
    bias = _positive_number(1 / sqrt(base.raw_learning_rate_factor), "bias_scale")
    _scalar_product(base.raw_learning_rate_factor, bias, bias)
    return base, precisions, circadian, bias


def _copy_config(config: CoupledScalingConfig) -> CoupledScalingConfig:
    _configuration(config)
    return CoupledScalingConfig(
        config.scheme,
        config.input_dim,
        config.width,
        config.hidden_layers,
        config.gamma0,
        config.precisions,
        config.circadian,
    )


def _copy_parameters(parameters: CoupledParameters) -> CoupledParameters:
    return CoupledParameters(
        tuple(value.copy() for value in parameters.hidden_weights),
        tuple(value.copy() for value in parameters.hidden_biases),
        parameters.output_weight.copy(),
        parameters.output_bias.copy(),
    )


def _validate_raw(config: CoupledScalingConfig, raw: CoupledParameters) -> None:
    if (
        type(raw) is not CoupledParameters
        or type(raw.hidden_weights) is not tuple
        or type(raw.hidden_biases) is not tuple
        or len(raw.hidden_weights) != config.hidden_layers
        or len(raw.hidden_biases) != config.hidden_layers
    ):
        raise ValueError("raw parameter topology differs from the uniform scaling configuration")
    widths = (config.input_dim,) + (config.width,) * config.hidden_layers
    values = (*raw.hidden_weights, *raw.hidden_biases, raw.output_weight, raw.output_bias)
    shapes = (
        *zip(widths, widths[1:]),
        *((1, config.width),) * config.hidden_layers,
        (config.width, 1),
        (1, 1),
    )
    if any(
        type(value) is not np.ndarray or value.dtype != np.float64 or value.shape != shape
        for value, shape in zip(values, shapes)
    ):
        raise ValueError("raw parameters must match the configured finite float64 matrix shapes")
    _validate_parameters(raw)


def _scaled_product(value: Array, *factors: float | Array) -> Array:
    # Gates participate before final rounding: neither a raw partial nor an
    # ungated delta must overflow before a small gate makes the applied delta usable.
    fraction, exponent = np.frexp(value)
    for factor in factors:
        part, shift = np.frexp(factor)
        fraction, exponent = fraction * part, exponent + shift
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        result = np.ldexp(fraction, exponent)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite coupled scaling product")
    return result


def _effective(config: CoupledScalingConfig, raw: CoupledParameters) -> CoupledParameters:
    scales, bias = config.forward_scales, config.bias_scale
    return CoupledParameters(
        tuple(
            _scaled_product(value, scale) for value, scale in zip(raw.hidden_weights, scales[:-1])
        ),
        tuple(_scaled_product(value, bias) for value in raw.hidden_biases),
        _scaled_product(raw.output_weight, scales[-1]),
        _scaled_product(raw.output_bias, bias),
    )


def _native_model(
    config: CoupledScalingConfig, parameters: CoupledParameters, wake: CoupledWakeState | None
) -> CoupledPredictiveCodingNetwork:
    _, precisions, circadian, _ = _configuration(config)
    if circadian is not None:
        if type(wake) is not CoupledWakeState:
            raise ValueError("circadian raw state requires complete CoupledWakeState")
        policy = CoupledCircadianConfig(
            wake.chemical_decay,
            wake.chemical_buildup_rate,
            wake.plasticity_sensitivity,
            wake.min_plasticity,
        )
        if policy != circadian:
            raise ValueError("raw wake policy differs from the scaling configuration")
    return restore_coupled_model(
        "pc" if circadian is None else "cpc",
        config.input_dim,
        (config.width,) * config.hidden_layers,
        precisions,
        CoupledSnapshot(1, parameters, precisions, wake),
    )


@dataclass(frozen=True)
class CoupledScalingState:
    config: CoupledScalingConfig
    raw_parameters: CoupledParameters
    wake_state: CoupledWakeState | None = None

    def __post_init__(self) -> None:
        config = _copy_config(self.config)
        _validate_raw(config, self.raw_parameters)
        model = _native_model(config, _effective(config, self.raw_parameters), self.wake_state)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "raw_parameters", _copy_parameters(self.raw_parameters))
        object.__setattr__(self, "wake_state", model.snapshot_state().wake_state)


def _validated_model(
    state: CoupledScalingState,
) -> tuple[CoupledScalingConfig, CoupledParameters, CoupledPredictiveCodingNetwork]:
    if type(state) is not CoupledScalingState:
        raise ValueError("state must be an exact CoupledScalingState")
    config = _copy_config(state.config)
    _validate_raw(config, state.raw_parameters)
    effective = _effective(config, state.raw_parameters)
    return config, effective, _native_model(config, effective, state.wake_state)


def copy_coupled_scaling_state(state: CoupledScalingState) -> CoupledScalingState:
    if type(state) is not CoupledScalingState:
        raise ValueError("state must be an exact CoupledScalingState")
    return CoupledScalingState(state.config, state.raw_parameters, state.wake_state)


def effective_coupled_parameters(state: CoupledScalingState) -> CoupledParameters:
    """Revalidate complete raw/wake state and return detached effective parameters."""
    _, effective, _ = _validated_model(state)
    return effective


def initialize_coupled_scaling(config: CoupledScalingConfig, seed: int) -> CoupledScalingState:
    config = _copy_config(config)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a native nonnegative integer")
    widths = (config.input_dim,) + (config.width,) * config.hidden_layers + (1,)
    rng = np.random.default_rng(seed)
    weights = tuple(
        rng.standard_normal((lower, upper)) * stddev
        for lower, upper, stddev in zip(widths, widths[1:], config.initial_stddevs)
    )
    raw = CoupledParameters(
        weights[:-1],
        tuple(np.zeros((1, config.width), dtype=np.float64) for _ in range(config.hidden_layers)),
        weights[-1],
        np.zeros((1, 1), dtype=np.float64),
    )
    policy = config.circadian
    wake = (
        None
        if policy is None
        else CoupledWakeState(
            policy.chemical_decay,
            policy.chemical_buildup_rate,
            policy.plasticity_sensitivity,
            policy.min_plasticity,
            np.zeros(config.width, dtype=np.float64),
            np.ones(config.width, dtype=np.float64),
            0,
        )
    )
    return CoupledScalingState(config, raw, wake)


@dataclass(frozen=True)
class CoupledScalingEvaluation:
    evaluation: CoupledEvaluation
    raw_gradients: CoupledParameters


def _raw_gradients(config: CoupledScalingConfig, gradients: CoupledParameters) -> CoupledParameters:
    scales, bias = config.forward_scales, config.bias_scale
    return CoupledParameters(
        tuple(
            _scaled_product(value, scale)
            for value, scale in zip(gradients.hidden_weights, scales[:-1])
        ),
        tuple(_scaled_product(value, bias) for value in gradients.hidden_biases),
        _scaled_product(gradients.output_weight, scales[-1]),
        _scaled_product(gradients.output_bias, bias),
    )


def evaluate_coupled_scaling(
    state: CoupledScalingState, X: Array, y: Array, latents: tuple[Array, ...]
) -> CoupledScalingEvaluation:
    """Return ungated true fixed-latent partials; nonrepresentable raw partials refuse."""
    config, parameters, _ = _validated_model(state)
    _, precisions, _, _ = _configuration(config)
    evaluation = evaluate_coupled(parameters, X, y, latents, precisions)
    return CoupledScalingEvaluation(
        evaluation, _raw_gradients(config, evaluation.parameter_gradients)
    )


@dataclass(frozen=True)
class CoupledScalingStep:
    state: CoupledScalingState
    evaluation: CoupledEvaluation
    latents: tuple[Array, ...]
    inference: CoupledInferenceDiagnostics | None


def _updated_raw(
    config: CoupledScalingConfig,
    raw: CoupledParameters,
    gradients: CoupledParameters,
    rate: float,
    gates: Array | None,
) -> CoupledParameters:
    scales, bias, factor = config.forward_scales, config.bias_scale, config.raw_learning_rate_factor
    incoming_gate: float | Array = 1.0 if gates is None else gates
    outgoing_gate: float | Array = 1.0 if gates is None else gates[:, None]
    with np.errstate(over="raise", invalid="raise"):
        weights = tuple(
            value
            - _scaled_product(
                gradient,
                rate,
                factor,
                scale,
                incoming_gate if layer == config.hidden_layers - 1 else 1.0,
            )
            for layer, (value, gradient, scale) in enumerate(
                zip(raw.hidden_weights, gradients.hidden_weights, scales[:-1])
            )
        )
        biases = tuple(
            value
            - _scaled_product(
                gradient,
                rate,
                factor,
                bias,
                incoming_gate if layer == config.hidden_layers - 1 else 1.0,
            )
            for layer, (value, gradient) in enumerate(
                zip(raw.hidden_biases, gradients.hidden_biases)
            )
        )
        output_weight = raw.output_weight - _scaled_product(
            gradients.output_weight, rate, factor, scales[-1], outgoing_gate
        )
        output_bias = raw.output_bias - _scaled_product(gradients.output_bias, rate, factor, bias)
    return CoupledParameters(weights, biases, output_weight, output_bias)


def step_coupled_scaling(
    state: CoupledScalingState,
    X: Array,
    y: Array,
    learning_rate: float,
    inference_steps: int,
    inference_learning_rate: float,
    *,
    inference_policy: CoupledInferencePolicy | None = None,
) -> CoupledScalingStep:
    """Apply raw SGD using native inferred latents and optional prospective CPC gates.

    Gating is a stop-gradient preconditioner, never an objective derivative.
    Updates use direct effective partials in a single stable product; no possibly
    overflowing intermediate raw partial or ungated delta is materialized.
    """
    config, _, model = _validated_model(state)
    rate, latents, evaluation, diagnostics = model._training_with_policy(
        X, y, learning_rate, inference_steps, inference_learning_rate, inference_policy
    )
    gates, wake = None, None
    if isinstance(model, CoupledCircadianPredictiveCodingNetwork):
        chemicals, gates = model._prospective_chemistry(latents)
        policy = model.config
        wake = CoupledWakeState(
            policy.chemical_decay,
            policy.chemical_buildup_rate,
            policy.plasticity_sensitivity,
            policy.min_plasticity,
            chemicals,
            gates,
            model.wake_updates + 1,
        )
    raw = _updated_raw(config, state.raw_parameters, evaluation.parameter_gradients, rate, gates)
    candidate = CoupledScalingState(config, raw, wake)
    parameters = effective_coupled_parameters(candidate)
    _, precisions, _, _ = _configuration(config)
    evaluate_coupled(parameters, X, y, latents, precisions)
    return CoupledScalingStep(
        candidate, evaluation, tuple(latent.copy() for latent in latents), diagnostics
    )
