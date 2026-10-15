"""Opt-in bounded latent inference on the existing coupled binary energy.

Inputs are borrowed quiescent parameters, fixed training batches and precisions.
Outputs own latent arrays and include explicit stopping-work diagnostics. Unsafe
proposals return the last accepted state without retry. Invalid initial problems
raise. No parameter updates, learner state, data-role authority or IO lives here.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from src.core.coupled_energy import (
    Array,
    CoupledEvaluation,
    CoupledParameters,
    evaluate_coupled,
    feedforward_latents,
)


def _finite_number(value: float, name: str, *, positive: bool) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a native finite number")
    try:
        resolved = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    if not isfinite(resolved) or (resolved <= 0 if positive else resolved < 0):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {bound} and finite")
    return resolved


@dataclass(frozen=True)
class CoupledInferencePolicy:
    min_steps: int = 1
    gradient_tolerance: float | None = 1e-6
    relative_energy_tolerance: float | None = None

    def __post_init__(self) -> None:
        _validate_policy(self)


def _validate_policy(policy: CoupledInferencePolicy) -> None:
    if type(policy) is not CoupledInferencePolicy:
        raise ValueError("policy must be an exact CoupledInferencePolicy")
    if type(policy.min_steps) is not int or not 1 <= policy.min_steps <= 10000:
        raise ValueError("min_steps must be a native integer in 1..10000")
    for name in ("gradient_tolerance", "relative_energy_tolerance"):
        value = getattr(policy, name)
        if value is not None:
            _finite_number(value, name, positive=False)


@dataclass(frozen=True)
class CoupledInferenceDiagnostics:
    reason: str
    attempted_steps: int
    accepted_steps: int
    energy_evaluations: int
    gradient_norm_evaluations: int
    energy_comparisons: int
    initial_energy: float
    final_energy: float
    final_gradient_inf_norm: float
    used_fallback: bool


@dataclass(frozen=True)
class CoupledInferenceResult:
    latents: tuple[Array, ...]
    evaluation: CoupledEvaluation
    diagnostics: CoupledInferenceDiagnostics


def _gradient_inf_norm(evaluation: CoupledEvaluation) -> float:
    norm = max(float(np.max(np.abs(gradient))) for gradient in evaluation.latent_gradients)
    if not isfinite(norm):
        raise FloatingPointError("nonfinite coupled latent gradient norm")
    return norm


def _convergence_reason(
    policy: CoupledInferencePolicy, accepted: int, norm: float, old_energy: float, new_energy: float
) -> str | None:
    if accepted < policy.min_steps:
        return None
    if policy.gradient_tolerance is not None and norm <= policy.gradient_tolerance:
        return "gradient_tolerance"
    if policy.relative_energy_tolerance is not None:
        relative_drop = (old_energy - new_energy) / max(1.0, abs(old_energy))
        if relative_drop <= policy.relative_energy_tolerance:
            return "relative_energy_tolerance"
    return None


def _infer_validated(
    parameters: CoupledParameters,
    features: Array,
    targets: Array,
    precisions: tuple[float, ...],
    max_steps: int,
    rate: float,
    policy: CoupledInferencePolicy,
) -> CoupledInferenceResult:
    latents = tuple(latent.copy() for latent in feedforward_latents(parameters, features))
    evaluation = evaluate_coupled(parameters, features, targets, latents, precisions)
    initial_energy = evaluation.energy
    if not isfinite(initial_energy):
        raise FloatingPointError("nonfinite initial coupled inference energy")
    norm = _gradient_inf_norm(evaluation)
    attempted, accepted, energy_calls, comparisons = 0, 0, 1, 0
    reason, fallback = "max_steps", False
    for _ in range(max_steps):
        attempted += 1
        with np.errstate(over="ignore", invalid="ignore"):
            candidate = tuple(
                latent - rate * gradient
                for latent, gradient in zip(latents, evaluation.latent_gradients)
            )
        if any(not np.isfinite(latent).all() for latent in candidate):
            reason, fallback = "nonfinite_candidate", True
            break
        energy_calls += 1
        try:
            proposed = evaluate_coupled(parameters, features, targets, candidate, precisions)
            if not isfinite(proposed.energy):
                raise FloatingPointError("nonfinite candidate coupled inference energy")
        except FloatingPointError:
            reason, fallback = "numerical_failure", True
            break
        comparisons += 1
        if proposed.energy > evaluation.energy:
            reason, fallback = "energy_increase", True
            break
        old_energy = evaluation.energy
        latents, evaluation = candidate, proposed
        accepted += 1
        norm = _gradient_inf_norm(evaluation)
        converged = _convergence_reason(policy, accepted, norm, old_energy, evaluation.energy)
        if converged is not None:
            reason = converged
            break
    diagnostics = CoupledInferenceDiagnostics(
        reason=reason,
        attempted_steps=attempted,
        accepted_steps=accepted,
        energy_evaluations=energy_calls,
        gradient_norm_evaluations=1 + accepted,
        energy_comparisons=comparisons,
        initial_energy=initial_energy,
        final_energy=evaluation.energy,
        final_gradient_inf_norm=norm,
        used_fallback=fallback,
    )
    return CoupledInferenceResult(latents, evaluation, diagnostics)


def infer_coupled(
    parameters: CoupledParameters,
    features: Array,
    targets: Array,
    precisions: tuple[float, ...],
    *,
    max_steps: int,
    learning_rate: float,
    policy: CoupledInferencePolicy,
) -> CoupledInferenceResult:
    """Run simultaneous guarded steps, with no inference retries or state mutation.

    Gradient tolerance takes precedence over relative-energy tolerance. Stopping
    checks reuse accepted evaluations; parameter-candidate validation performed by
    a learner is outside these inference-only work counts. Safety fallback may
    return fewer accepted steps than the requested minimum.
    """
    _validate_policy(policy)
    if type(max_steps) is not int or not 1 <= max_steps <= 10000:
        raise ValueError("max_steps must be a native integer in 1..10000")
    if policy.min_steps > max_steps:
        raise ValueError("min_steps must not exceed max_steps")
    rate = _finite_number(learning_rate, "learning_rate", positive=True)
    return _infer_validated(parameters, features, targets, precisions, max_steps, rate, policy)
