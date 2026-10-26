"""Owned bounded continuous-PC latent inference using shared guarded dynamics.

Inputs: quiescent continuous parameters/features/targets, precision and policy.
Outputs: owned accepted free state, matching energy and stopping-work counters.
No parameter update, deployed prediction, role authority, history, RNG or IO.
"""

from dataclasses import dataclass

from src.core import continuous_pc as energy
from src.core.continuous_bc import Array, _targets
from src.core.continuous_demonstration import ContinuousActionSpec
from src.core.continuous_pc import ContinuousPCEvaluation
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_inference import (
    CoupledInferenceDiagnostics,
    CoupledInferencePolicy,
    _finite_number,
    _run_inference,
    _validate_policy,
)


@dataclass(frozen=True)
class ContinuousPCInferenceResult:
    latents: tuple[Array, ...]
    evaluation: ContinuousPCEvaluation
    diagnostics: CoupledInferenceDiagnostics


def infer_continuous_pc(
    parameters: CoupledParameters,
    features: Array,
    targets: Array,
    precisions: tuple[float, ...],
    spec: ContinuousActionSpec,
    *,
    max_steps: int,
    learning_rate: float,
    policy: CoupledInferencePolicy,
) -> ContinuousPCInferenceResult:
    """Infer target-conditioned free latents; leave all parameters unchanged.

    Uses one target-free initialization, then at most max_steps simultaneous
    proposals. Failed numerical work/increasing energy returns the last accepted
    state without retries. Invalid initial inputs and programmer errors raise.
    The returned evaluation is target-conditioned, not a deploy-time predictor.

    Kernel counters include attempted energy entries, including failed candidate
    evaluations, but exclude validation, copies and initialization. Every energy
    entry computes both latent and parameter partials. Arrays are copied at entry
    and again by each public energy call; storage retains no step history.
    """
    _validate_policy(policy)
    if type(max_steps) is not int or not 1 <= max_steps <= 10000:
        raise ValueError("max_steps must be a native integer in 1..10000")
    if policy.min_steps > max_steps:
        raise ValueError("min_steps must not exceed max_steps")
    rate = _finite_number(learning_rate, "learning_rate", positive=True)
    owned_policy = CoupledInferencePolicy(
        policy.min_steps, policy.gradient_tolerance, policy.relative_energy_tolerance
    )
    owned, values, owned_spec = energy._inputs(parameters, features, spec)
    actions = _targets(targets, len(values), owned_spec)
    beta = energy._own_precisions(precisions, len(owned.hidden_weights))
    initial = energy.feedforward_continuous_latents(owned, values, owned_spec)
    latents, evaluation, diagnostics = _run_inference(
        initial,
        lambda current: energy.evaluate_continuous_pc(
            owned, values, actions, current, beta, owned_spec
        ),
        max_steps,
        rate,
        owned_policy,
    )
    return ContinuousPCInferenceResult(latents, evaluation, diagnostics)
