"""Feedforward BP using the coupled raw learner's exact representation and rates.

Inputs are borrowed quiescent PC-only CoupledScalingState records and finite
float64 batches. Outputs own replacement raw state and gradient arrays. Weight
and bias factors, initialization/state schemas, and stable updates are reused
without introducing a second scaling rule. Precisions are preserved metadata;
the BP objective is ordinary feedforward BCE and does not depend on them.
No inference, chemistry, optimizer state, runtime authority, or IO lives here.
"""

from dataclasses import dataclass

from src.core.coupled_energy import Array, CoupledParameters
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    CoupledScalingState,
    _raw_gradients,
    _updated_raw,
    _validated_model,
    effective_coupled_parameters,
)
from src.core.linear_pc_scaling import _positive_number
from src.core.matched_backprop import MatchedBackpropEvaluation, evaluate_matched_backprop


@dataclass(frozen=True)
class ScaledMatchedBackpropEvaluation:
    evaluation: MatchedBackpropEvaluation
    raw_gradients: CoupledParameters


@dataclass(frozen=True)
class ScaledMatchedBackpropStep:
    state: CoupledScalingState
    evaluation: MatchedBackpropEvaluation


def _pc_parameters(state: CoupledScalingState) -> tuple[CoupledScalingConfig, CoupledParameters]:
    if type(state) is not CoupledScalingState or type(state.config) is not CoupledScalingConfig:
        raise ValueError("scaled backpropagation requires exact coupled scaling state/config")
    if state.config.circadian is not None or state.wake_state is not None:
        raise ValueError(
            "scaled backpropagation requires PC-only state without circadian policy/wake state"
        )
    config, parameters, _ = _validated_model(state)
    return config, parameters


def evaluate_scaled_matched_backprop(
    state: CoupledScalingState, X: Array, y: Array
) -> ScaledMatchedBackpropEvaluation:
    """Return full feedforward partials in effective and raw coordinates.

    Unlike PC's held-latent partials, these differentiate through every hidden
    tanh activation. A nonrepresentable raw partial raises a numerical error.
    """
    config, parameters = _pc_parameters(state)
    evaluation = evaluate_matched_backprop(parameters, X, y)
    raw_gradients = _raw_gradients(config, evaluation.parameter_gradients)
    return ScaledMatchedBackpropEvaluation(evaluation, raw_gradients)


def step_scaled_matched_backprop(
    state: CoupledScalingState, X: Array, y: Array, learning_rate: float
) -> ScaledMatchedBackpropStep:
    """Return simultaneous raw SGD and its preupdate feedforward evaluation.

    The applied delta combines effective gradients and all raw-coordinate rate
    factors before rounding; it does not materialize a possibly overflowing raw
    partial. The complete candidate's feedforward loss/gradient must be finite.
    Failure leaves the borrowed source state unchanged.
    """
    rate = _positive_number(learning_rate, "learning_rate")
    config, parameters = _pc_parameters(state)
    evaluation = evaluate_matched_backprop(parameters, X, y)
    raw = _updated_raw(
        config, state.raw_parameters, evaluation.parameter_gradients, rate, gates=None
    )
    candidate = CoupledScalingState(config, raw)
    evaluate_matched_backprop(effective_coupled_parameters(candidate), X, y)
    return ScaledMatchedBackpropStep(candidate, evaluation)
