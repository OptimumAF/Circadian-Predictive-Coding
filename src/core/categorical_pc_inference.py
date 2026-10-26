"""TRAIN-only bounded categorical PC proposals, without publication or clocks.

The direction is B*dE/dz for the existing weighted objective. Safety returns the
last accepted latent state without retries. Costs include rejected attempts and
final parameter work; this module neither selects weights nor reads evaluation.
"""

from dataclasses import dataclass
import math
from typing import Any

from src.core.categorical_pc_weighted import OBJECTIVE, CategoricalPCUpdate, _finish, _inputs
from src.core.coupled_inference import (
    CoupledInferencePolicy,
    _convergence_reason,
    _validate_policy,
)
from src.core.learner_ports import TrainingUpdateFailed
from src.shared.torch_runtime import require_torch


@dataclass(frozen=True)
class CategoricalInferenceDiagnostics:
    """Scalar results; gradient_norm_evaluations counts attempted norm reads.

    Energy/direction completions require finite helper results. Finalization
    completions instead count arithmetic returns before its shared finite check.
    """

    reason: str
    initial_energy: float
    final_energy: float
    final_scaled_gradient_inf_norm: float
    rows: int
    attempted_steps: int
    accepted_steps: int
    rejected_steps: int
    attempted_latent_row_iterations: int
    accepted_latent_row_iterations: int
    energy_evaluations_attempted: int
    energy_evaluations_completed: int
    latent_gradient_evaluations_attempted: int
    latent_gradient_evaluations_completed: int
    gradient_norm_evaluations: int
    energy_comparisons: int
    parameter_gradient_evaluations_attempted: int
    parameter_gradient_evaluations_completed: int
    parameter_proposals_attempted: int
    parameter_proposals_completed: int
    final_energy_evaluations_attempted: int
    final_energy_evaluations_completed: int
    legacy_diagnostic_evaluations_attempted: int
    legacy_diagnostic_evaluations_completed: int
    initial_prior_evaluations_attempted: int
    initial_prior_evaluations_completed: int
    used_fallback: bool
    failure_work_may_be_unobserved: bool

    @property
    def work(self) -> dict[str, int | bool]:
        """Return owned counters; no borrowed tensors or mutable accounting state."""
        return {name: getattr(self, name) for name in _WORK_FIELDS}


_WORK_FIELDS = tuple(
    name
    for name in CategoricalInferenceDiagnostics.__dataclass_fields__
    if name not in ("reason", "initial_energy", "final_energy", "final_scaled_gradient_inf_norm")
)


@dataclass(frozen=True)
class BoundedCategoricalPCUpdate:
    update: CategoricalPCUpdate
    diagnostics: CategoricalInferenceDiagnostics


def _finite(torch, value):
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError("nonfinite bounded categorical PC arithmetic")
    return value


def _energy(torch, state, prior, hidden_output, output_bias, targets, target_weights):
    logits = _finite(torch, state @ hidden_output + output_bias)
    row_energy = torch.nn.functional.cross_entropy(logits, targets, reduction="none") + 0.5 * (
        (state - prior).square().sum(dim=1)
    )
    energy = (target_weights * row_energy).sum() / target_weights.sum()
    return logits, float(_finite(torch, energy).item())


def _direction(torch, state, prior, logits, hidden_output, one_hot, factors, unweighted):
    error = torch.nn.functional.softmax(logits, dim=1) - one_hot
    direction = state - prior + error @ hidden_output.T
    return _finite(torch, direction if unweighted else direction * factors)


def _norm(torch, direction):
    norm = float(torch.max(torch.abs(direction)).item())
    if not math.isfinite(norm):
        raise FloatingPointError("nonfinite scaled categorical gradient norm")
    return norm


def _evaluate(torch, state, prior, parameters, targets, weights, work):
    work["energy_evaluations_attempted"] += 1
    logits, energy = _energy(torch, state, prior, parameters[2], parameters[3], targets, weights)
    work["energy_evaluations_completed"] += 1
    return logits, energy


def _gradient(torch, state, prior, logits, parameters, one_hot, factors, uniform, work):
    work["latent_gradient_evaluations_attempted"] += 1
    direction = _direction(torch, state, prior, logits, parameters[2], one_hot, factors, uniform)
    work["latent_gradient_evaluations_completed"] += 1
    work["gradient_norm_evaluations"] += 1
    return direction, _norm(torch, direction)


def _relax(torch, features, targets, parameters, weights, steps, rate, policy, work):
    work["initial_prior_evaluations_attempted"] += 1
    prior = torch.tanh(_finite(torch, features @ parameters[0] + parameters[1]))
    work["initial_prior_evaluations_completed"] += 1
    state = prior.clone()
    one_hot = torch.nn.functional.one_hot(targets, num_classes=parameters[2].shape[1]).to(
        features.dtype
    )
    target_weights = torch.tensor(weights, dtype=features.dtype)[targets]
    factors = features.shape[0] * (target_weights / target_weights.sum())[:, None]
    uniform = bool((target_weights == target_weights[0]).all().item())
    logits, energy = _evaluate(torch, state, prior, parameters, targets, target_weights, work)
    direction, norm = _gradient(
        torch, state, prior, logits, parameters, one_hot, factors, uniform, work
    )
    initial_energy = energy
    reason = "max_steps"
    for _ in range(steps):
        work["attempted_steps"] += 1
        work["attempted_latent_row_iterations"] += features.shape[0]
        candidate = state - rate * direction
        if not bool(torch.isfinite(candidate).all().item()):
            reason = "nonfinite_candidate"
            break
        try:
            logits, proposed_energy = _evaluate(
                torch,
                candidate,
                prior,
                parameters,
                targets,
                target_weights,
                work,
            )
            work["energy_comparisons"] += 1
            if proposed_energy > energy:
                reason = "energy_increase"
                break
            proposed_direction, proposed_norm = _gradient(
                torch, candidate, prior, logits, parameters, one_hot, factors, uniform, work
            )
        except FloatingPointError:
            reason = "numerical_failure"
            break
        old_energy = energy
        state, energy, direction, norm = (
            candidate,
            proposed_energy,
            proposed_direction,
            proposed_norm,
        )
        work["accepted_steps"] += 1
        work["accepted_latent_row_iterations"] += features.shape[0]
        stopped = _convergence_reason(policy, work["accepted_steps"], norm, old_energy, energy)
        if stopped is not None:
            reason = stopped
            break
    work["rejected_steps"] = work["attempted_steps"] - work["accepted_steps"]
    work["used_fallback"] = work["rejected_steps"] > 0
    return (prior, state, one_hot, target_weights, factors, uniform), (
        reason,
        initial_energy,
        energy,
        norm,
    )


def bounded_class_weighted_pc_update(
    features: Any,
    targets: Any,
    parameters: tuple[Any, Any, Any, Any],
    *,
    class_weights: tuple[float, ...],
    learning_rate: float,
    inference_steps: int,
    inference_learning_rate: float,
    role: str,
    policy: CoupledInferencePolicy,
) -> BoundedCategoricalPCUpdate:
    """Propose at last accepted z, holding z fixed for parameter differentiation.

    Norm tolerance applies to B*dE/dz. Minimum steps gates convergence only;
    safety may stop earlier. Every call still proposes one parameter update.
    Finalization counters record arithmetic returns before final finite checks.
    Failed calls carry owned work; unexpected errors conservatively flag unknown
    work. Input admission errors precede numerical work and remain ValueError.
    """
    if type(role) is not str or role != "train":
        raise ValueError("bounded PC proposals require explicit train role")
    _validate_policy(policy)
    weights = _inputs(
        features,
        targets,
        parameters,
        class_weights,
        learning_rate,
        inference_steps,
        inference_learning_rate,
        role,
    )
    if policy.min_steps > inference_steps:
        raise ValueError("min_steps must not exceed inference_steps")
    torch = require_torch()
    work: dict[str, Any] = dict.fromkeys(_WORK_FIELDS, 0)
    work.update(
        rows=int(features.shape[0]), used_fallback=False, failure_work_may_be_unobserved=False
    )
    try:
        with torch.no_grad():
            owned_features = features.detach().clone()
            owned_targets = targets.detach().clone()
            owned_parameters = tuple(value.detach().clone() for value in parameters)
            final, scalars = _relax(
                torch,
                owned_features,
                owned_targets,
                owned_parameters,
                weights,
                inference_steps,
                inference_learning_rate,
                policy,
                work,
            )
            prior, state, one_hot, target_weights, factors, uniform = final
            update = _finish(
                torch,
                owned_features,
                owned_targets,
                owned_parameters,
                prior,
                state,
                one_hot,
                target_weights,
                factors,
                uniform,
                learning_rate,
                work["accepted_steps"],
                work=work,
            )
    except Exception as error:
        work["rejected_steps"] = work["attempted_steps"] - work["accepted_steps"]
        work["failure_work_may_be_unobserved"] = not isinstance(error, FloatingPointError)
        raise TrainingUpdateFailed(str(error) or type(error).__name__, OBJECTIVE, work) from error
    reason, initial_energy, energy, norm = scalars
    return BoundedCategoricalPCUpdate(
        update,
        CategoricalInferenceDiagnostics(
            reason=reason,
            initial_energy=initial_energy,
            final_energy=energy,
            final_scaled_gradient_inf_norm=norm,
            **work,
        ),
    )
