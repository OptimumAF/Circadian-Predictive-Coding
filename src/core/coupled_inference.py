"""Bounded binary latent inference and shared private guarded dynamics.

Inputs are borrowed quiescent parameters, fixed training batches and precisions.
Outputs own latent arrays and include explicit stopping-work diagnostics. Unsafe
proposals return the last accepted state without retry. Invalid initial problems
raise. No parameter updates, learner state, data-role authority or IO lives here.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Protocol, TypeVar

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


class _LatentEvaluation(Protocol):
    @property
    def energy(self) -> float: ...

    @property
    def latent_gradients(self) -> tuple[Array, ...]: ...


_Evaluation = TypeVar("_Evaluation", bound=_LatentEvaluation)


@dataclass
class _InferenceWork:
    """Private per-call progress; survives exceptions in an owning adapter."""

    attempted_steps: int = 0
    accepted_steps: int = 0
    energy_evaluations: int = 0
    energy_returns: int = 0
    gradient_norm_evaluations: int = 0
    energy_comparisons: int = 0


def _gradient_inf_norm(evaluation: _LatentEvaluation) -> float:
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


def _run_inference(
    initial_latents: tuple[Array, ...],
    evaluate: Callable[[tuple[Array, ...]], _Evaluation],
    max_steps: int,
    rate: float,
    policy: CoupledInferencePolicy,
    *,
    _work: _InferenceWork | None = None,
) -> tuple[tuple[Array, ...], _Evaluation, CoupledInferenceDiagnostics]:
    # Why: binary and continuous tasks require the same guarded dynamics and
    # accounting. Keep one private loop; wrappers retain likelihood validation.
    work = _InferenceWork() if _work is None else _work
    latents = tuple(latent.copy() for latent in initial_latents)
    work.energy_evaluations += 1
    evaluation = evaluate(latents)
    work.energy_returns += 1
    initial_energy = evaluation.energy
    if not isfinite(initial_energy):
        raise FloatingPointError("nonfinite initial coupled inference energy")
    work.gradient_norm_evaluations += 1
    norm = _gradient_inf_norm(evaluation)
    reason, fallback = "max_steps", False
    for _ in range(max_steps):
        work.attempted_steps += 1
        with np.errstate(over="ignore", invalid="ignore"):
            candidate = tuple(
                latent - rate * gradient
                for latent, gradient in zip(latents, evaluation.latent_gradients)
            )
        if any(not np.isfinite(latent).all() for latent in candidate):
            reason, fallback = "nonfinite_candidate", True
            break
        work.energy_evaluations += 1
        try:
            proposed = evaluate(candidate)
            work.energy_returns += 1
            if not isfinite(proposed.energy):
                raise FloatingPointError("nonfinite candidate coupled inference energy")
        except FloatingPointError:
            reason, fallback = "numerical_failure", True
            break
        work.energy_comparisons += 1
        if proposed.energy > evaluation.energy:
            reason, fallback = "energy_increase", True
            break
        old_energy = evaluation.energy
        latents, evaluation = candidate, proposed
        work.accepted_steps += 1
        work.gradient_norm_evaluations += 1
        norm = _gradient_inf_norm(evaluation)
        converged = _convergence_reason(
            policy, work.accepted_steps, norm, old_energy, evaluation.energy
        )
        if converged is not None:
            reason = converged
            break
    diagnostics = CoupledInferenceDiagnostics(
        reason=reason,
        attempted_steps=work.attempted_steps,
        accepted_steps=work.accepted_steps,
        energy_evaluations=work.energy_evaluations,
        gradient_norm_evaluations=work.gradient_norm_evaluations,
        energy_comparisons=work.energy_comparisons,
        initial_energy=initial_energy,
        final_energy=evaluation.energy,
        final_gradient_inf_norm=norm,
        used_fallback=fallback,
    )
    return latents, evaluation, diagnostics


def _infer_validated(
    parameters: CoupledParameters,
    features: Array,
    targets: Array,
    precisions: tuple[float, ...],
    max_steps: int,
    rate: float,
    policy: CoupledInferencePolicy,
) -> CoupledInferenceResult:
    latents, evaluation, diagnostics = _run_inference(
        feedforward_latents(parameters, features),
        lambda current: evaluate_coupled(parameters, features, targets, current, precisions),
        max_steps,
        rate,
        policy,
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
