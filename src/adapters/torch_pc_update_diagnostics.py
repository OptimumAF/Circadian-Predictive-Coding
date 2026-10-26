"""Read realized ungated PC directions from detached, bounded TRAIN snapshots.

Inputs are an encoding-bound PC/neutral snapshot and complete TRAIN episodes.
Outputs are scalar relaxation, counterfactual fit, direction geometry and paid
work. No learner update, optimizer, task selection, probe/final data, publication,
clock or orchestration is provided. Counterfactual fit is not a causal diagnosis.
"""

from copy import deepcopy
import random
from typing import Any, cast

import numpy as np

from src.adapters import torch_feature_learner as native
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureSnapshot,
    copy_invariant_torch_feature_policy,
    copy_invariant_torch_feature_snapshot,
)
from src.adapters.torch_feature_gradient_diagnostics import _exact_record, _tensor_marks
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_recurrent_gradient_diagnostics import GradientDiagnosticFailed
from src.core.categorical_pc_weighted import (
    OBJECTIVE as WEIGHTED_OBJECTIVE,
    class_weighted_pc_update,
)
from src.core.demonstration_class_weights import validate_class_weights
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    copy_episode,
)
from src.core.gradient_geometry import compare_gradient_geometry
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from src.shared.torch_runtime import require_torch

DEFINITION = "torch_invariant_ungated_pc_realized_update_v1"
OBJECTIVE = "categorical_mean_ce_plus_half_mean_row_hidden_squared_sum_v1"
WEIGHTED_DEFINITION = "torch_invariant_weighted_pc_realized_update_v1"


def _sources(snapshot: object, episodes: tuple[DemonstrationEpisode, ...]):
    _exact_record(snapshot, InvariantTorchFeatureSnapshot)
    bound = cast(InvariantTorchFeatureSnapshot, snapshot)
    policy = copy_invariant_torch_feature_policy(bound.policy)
    if policy.native_policy.method not in {"pc", "neutral_cpc"}:
        raise ValueError("realized ungated PC diagnostics require pc or neutral_cpc")
    if type(episodes) is not tuple or not 1 <= len(episodes) <= 6:
        raise ValueError("realized PC diagnostics require an exact tuple of1..6 TRAIN episodes")
    rows = 0
    for episode in episodes:
        _exact_record(episode, DemonstrationEpisode)
        if type(episode.transitions) is not tuple or not 1 <= len(episode.transitions) <= 16:
            raise ValueError("TRAIN episodes require an exact tuple of1..16 transitions")
        rows += len(episode.transitions)
        if rows > 54:
            raise ValueError("realized PC diagnostics admit at most54 TRAIN rows")
        for transition in episode.transitions:
            _exact_record(transition, DemonstrationTransition)
            if (
                type(transition.observation) is not tuple
                or len(transition.observation) != 296
                or type(transition.next_observation) is not tuple
                or len(transition.next_observation) != 296
            ):
                raise ValueError("TRAIN histories require296 current and next coordinates")
        if episode.role != "train" or episode.num_classes != 7:
            raise ValueError("realized PC diagnostics require TRAIN296/K7 episodes")
    owned = tuple(copy_episode(episode) for episode in episodes)
    if len({episode.episode_id for episode in owned}) != len(owned):
        raise ValueError("TRAIN source episode identities must be distinct")
    return owned


def _report(episodes, policy):
    return dict(
        status="running",
        reason=None,
        definition=DEFINITION,
        objective=OBJECTIVE,
        method=policy.method,
        rows=sum(len(episode.transitions) for episode in episodes),
        source_episode_ids=[episode.episode_id for episode in episodes],
        inference_steps=policy.inference_steps,
        interpretation="TRAIN counterfactual only; gradient dot delta negative denotes local descent, not causal acquisition or benefit",
        relaxation=[],
        legacy_diagnostic=None,
        prior_mean_ce=None,
        counterfactual_prior_mean_ce=None,
        prior_ce_change=None,
        gradient_delta_geometry=None,
        per_parameter_geometry=None,
        work=dict(
            snapshot_validations_attempted=0,
            snapshot_validations_completed=0,
            validation_temporary_optimizers=0,
            projection_calls_attempted=0,
            projection_calls_completed=0,
            projection_rows_attempted=0,
            projection_rows_completed=0,
            projection_array_bytes_returned=0,
            prior_forward_calls_attempted=0,
            prior_forward_calls_completed=0,
            prior_forward_rows_attempted=0,
            prior_forward_rows_completed=0,
            prior_backward_calls_attempted=0,
            prior_backward_calls_completed=0,
            prior_backward_rows_attempted=0,
            prior_backward_rows_completed=0,
            relaxation_forward_calls_attempted=0,
            relaxation_forward_calls_completed=0,
            relaxation_forward_rows_attempted=0,
            relaxation_forward_rows_completed=0,
            candidate_parameter_sets_completed=0,
            native_train_calls=0,
            optimizer_steps=0,
            published_updates=0,
            failure_work_may_be_unobserved=False,
        ),
    )


def _finite(value, name, torch):
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"nonfinite PC diagnostic {name}")
    return value


def _prior(parameters, values, targets, work, torch):
    rows = len(values)
    work["prior_forward_calls_attempted"] += 1
    work["prior_forward_rows_attempted"] += rows
    linear = _finite(
        values @ parameters["weight_feature_hidden"] + parameters["bias_hidden"],
        "linear prior",
        torch,
    )
    hidden = torch.tanh(linear)
    logits = hidden @ parameters["weight_hidden_output"] + parameters["bias_output"]
    work["prior_forward_calls_completed"] += 1
    work["prior_forward_rows_completed"] += rows
    _finite(logits, "prior logits", torch)
    loss = _finite(torch.nn.functional.cross_entropy(logits, targets), "prior CE", torch)
    return hidden, loss


def _relax(parameters, prior, targets, policy, report, torch):
    state = prior.clone()
    one_hot = torch.nn.functional.one_hot(targets, num_classes=7).to(prior.dtype)
    weight, bias = parameters["weight_hidden_output"], parameters["bias_output"]
    work, rows = report["work"], len(prior)
    for step in range(policy.inference_steps + 1):
        work["relaxation_forward_calls_attempted"] += 1
        work["relaxation_forward_rows_attempted"] += rows
        logits = state @ weight + bias
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        work["relaxation_forward_calls_completed"] += 1
        work["relaxation_forward_rows_completed"] += rows
        _finite(logits, "relaxed logits", torch)
        error, residual = probabilities - one_hot, state - prior
        gradient = _finite(residual + error @ weight.T, "latent gradient", torch)
        loss = _finite(torch.nn.functional.cross_entropy(logits, targets), "relaxed CE", torch)
        penalty = _finite(0.5 * residual.square().sum(dim=1).mean(), "hidden penalty", torch)
        report["relaxation"].append(
            dict(
                step=step,
                mean_ce=float(loss.item()),
                hidden_quadratic=float(penalty.item()),
                objective=float(_finite(loss + penalty, "objective", torch).item()),
                latent_gradient_l2=float(torch.linalg.vector_norm(gradient.double()).item()),
                latent_displacement_l2=float(torch.linalg.vector_norm(residual.double()).item()),
            )
        )
        if step < policy.inference_steps:
            state = _finite(
                state - policy.inference_learning_rate * gradient, "latent state", torch
            )
    return state, error, residual


def _measure(owned, episodes, report, torch):
    policy = owned.policy.native_policy
    work, rows = report["work"], report["rows"]
    transitions = [step for episode in episodes for step in episode.transitions]
    work["projection_calls_attempted"] += 1
    work["projection_rows_attempted"] += rows
    projected = np.array(
        [canonicalize_public_history(step.observation) for step in transitions], dtype=np.float64
    )
    work["projection_calls_completed"] += 1
    work["projection_rows_completed"] += rows
    work["projection_array_bytes_returned"] += int(projected.nbytes)
    values = native._features(projected, 296)
    targets = torch.tensor([step.action for step in transitions], dtype=torch.int64)
    parameters = {
        name: owned.native_state.head_state[name].detach().clone().requires_grad_(True)
        for name in native._PARAMETERS
    }
    with torch.enable_grad():
        hidden, loss = _prior(parameters, values, targets, work, torch)
        report["prior_mean_ce"] = float(loss.item())
        work["prior_backward_calls_attempted"] += 1
        work["prior_backward_rows_attempted"] += rows
        gradients = torch.autograd.grad(loss, tuple(parameters.values()))
        work["prior_backward_calls_completed"] += 1
        work["prior_backward_rows_completed"] += rows
    with torch.no_grad():
        state, error, residual = _relax(parameters, hidden.detach(), targets, policy, report, torch)
        report["legacy_diagnostic"] = dict(
            definition=native.TORCH_PC_ENERGY_ID,
            value=float((0.5 * (error.square().mean() + residual.square().mean())).item()),
        )
        hidden_gradient = (-residual) * (1 - hidden.detach() * hidden.detach())
        manual = (
            (values.T @ hidden_gradient) / float(rows),
            hidden_gradient.mean(dim=0, keepdim=True),
            (state.T @ error) / float(rows),
            error.mean(dim=0, keepdim=True),
        )
        # Why subtract again: native float32 assignments round before deltas exist.
        candidate = {
            name: _finite(parameters[name] - policy.learning_rate * gradient, name, torch)
            for name, gradient in zip(native._PARAMETERS, manual, strict=True)
        }
        work["candidate_parameter_sets_completed"] += 1
        first = tuple(
            _finite(gradient, "prior gradient", torch).detach().numpy().copy()
            for gradient in gradients
        )
        deltas = tuple(
            (candidate[name] - parameters[name]).detach().numpy().copy()
            for name in native._PARAMETERS
        )
        report["gradient_delta_geometry"] = compare_gradient_geometry(first, deltas)
        report["per_parameter_geometry"] = {
            name: compare_gradient_geometry((first[i],), (deltas[i],))
            for i, name in enumerate(native._PARAMETERS)
        }
        _, after = _prior(candidate, values, targets, work, torch)
        report["counterfactual_prior_mean_ce"] = float(after.item())
        report["prior_ce_change"] = float(after.item()) - report["prior_mean_ce"]


def _weighted_report(report, weights):
    report.update(
        definition=WEIGHTED_DEFINITION,
        objective=WEIGHTED_OBJECTIVE,
        class_weights=list(weights),
        geometry_objective="weighted_prior_mean_ce",
        prior_weighted_mean_ce=None,
        counterfactual_prior_weighted_mean_ce=None,
        weighted_prior_ce_change=None,
        fixed_state_weighted_energy_before=None,
        fixed_state_weighted_energy_after=None,
        fixed_state_weighted_energy_change=None,
        proposal=None,
        relaxation_observation=(
            "exact weighted native proposal port only, no intermediate trace; "
            "internal forwards are bounded by inference_steps and charged through "
            "proposal rows/nominal iterations, not observed individually on failure"
        ),
        fixed_state_energy_definition=(
            "weighted CE at the original final latent z plus half weighted row "
            "squared distance from the corresponding parameter-set prior; "
            "candidate prior is recomputed, with no second latent relaxation"
        ),
    )
    for group in (
        "proposal_calls",
        "proposal_rows",
        "proposal_nominal_latent_row_iterations",
        "fixed_state_energy_calls",
        "fixed_state_energy_rows",
    ):
        for event in ("attempted", "completed"):
            report["work"][f"{group}_{event}"] = 0


def _weighted_ce(logits, targets, weights, torch):
    loss = torch.nn.functional.cross_entropy(logits, targets)
    # Why: all observed equal weights retain the native unweighted operation order.
    if bool((weights == weights[0]).all().item()):
        return loss, loss
    rows = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    return loss, (weights * rows).sum() / weights.sum()


def _weighted_prior(parameters, values, targets, weights, work, torch):
    rows = len(values)
    work["prior_forward_calls_attempted"] += 1
    work["prior_forward_rows_attempted"] += rows
    linear = _finite(
        values @ parameters["weight_feature_hidden"] + parameters["bias_hidden"],
        "linear prior",
        torch,
    )
    hidden = torch.tanh(linear)
    logits = _finite(
        hidden @ parameters["weight_hidden_output"] + parameters["bias_output"],
        "prior logits",
        torch,
    )
    ordinary, weighted = _weighted_ce(logits, targets, weights, torch)
    _finite(ordinary, "prior CE", torch)
    _finite(weighted, "weighted prior CE", torch)
    work["prior_forward_calls_completed"] += 1
    work["prior_forward_rows_completed"] += rows
    return hidden, ordinary, weighted


def _fixed_state_energy(candidate, values, targets, weights, state, work, torch):
    rows = len(values)
    work["fixed_state_energy_calls_attempted"] += 1
    work["fixed_state_energy_rows_attempted"] += rows
    # The candidate's feedforward prior and its fixed-z output are two different
    # projections inside this one explicitly counted energy evaluation.
    linear = _finite(
        values @ candidate["weight_feature_hidden"] + candidate["bias_hidden"],
        "candidate linear prior",
        torch,
    )
    residual = state - torch.tanh(linear)
    logits = _finite(
        state @ candidate["weight_hidden_output"] + candidate["bias_output"],
        "candidate fixed-state logits",
        torch,
    )
    row = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    energy = _finite(
        (weights * (row + 0.5 * residual.square().sum(dim=1))).sum() / weights.sum(),
        "candidate fixed-state energy",
        torch,
    )
    work["fixed_state_energy_calls_completed"] += 1
    work["fixed_state_energy_rows_completed"] += rows
    return float(energy.item())


def _measure_weighted(owned, episodes, report, torch, weights):
    policy = owned.policy.native_policy
    work, rows = report["work"], report["rows"]
    transitions = [step for episode in episodes for step in episode.transitions]
    work["projection_calls_attempted"] += 1
    work["projection_rows_attempted"] += rows
    projected = np.array(
        [canonicalize_public_history(step.observation) for step in transitions], dtype=np.float64
    )
    work["projection_calls_completed"] += 1
    work["projection_rows_completed"] += rows
    work["projection_array_bytes_returned"] += int(projected.nbytes)
    values = native._features(projected, 296)
    targets = torch.tensor([step.action for step in transitions], dtype=torch.int64)
    target_weights = torch.tensor(weights, dtype=values.dtype)[targets]
    parameters = {
        name: owned.native_state.head_state[name].detach().clone().requires_grad_(True)
        for name in native._PARAMETERS
    }
    with torch.enable_grad():
        hidden, ordinary, weighted = _weighted_prior(
            parameters, values, targets, target_weights, work, torch
        )
        report["prior_mean_ce"] = float(ordinary.item())
        report["prior_weighted_mean_ce"] = float(weighted.item())
        work["prior_backward_calls_attempted"] += 1
        work["prior_backward_rows_attempted"] += rows
        gradients = torch.autograd.grad(weighted, tuple(parameters.values()))
        work["prior_backward_calls_completed"] += 1
        work["prior_backward_rows_completed"] += rows
    with torch.no_grad():
        work["proposal_calls_attempted"] += 1
        work["proposal_rows_attempted"] += rows
        work["proposal_nominal_latent_row_iterations_attempted"] += rows * policy.inference_steps
        proposal = class_weighted_pc_update(
            values,
            targets,
            tuple(parameters.values()),
            class_weights=weights,
            learning_rate=policy.learning_rate,
            inference_steps=policy.inference_steps,
            inference_learning_rate=policy.inference_learning_rate,
            role="train",
        )
        work["proposal_calls_completed"] += 1
        work["proposal_rows_completed"] += rows
        work["proposal_nominal_latent_row_iterations_completed"] += (
            proposal.nominal_latent_row_iterations
        )
        work["candidate_parameter_sets_completed"] += 1
        candidate = dict(zip(native._PARAMETERS, proposal.parameters, strict=True))
        report["fixed_state_weighted_energy_before"] = proposal.energy
        report["proposal"] = dict(
            weighted_energy=proposal.energy,
            nominal_latent_row_iterations=proposal.nominal_latent_row_iterations,
            hidden_displacement_l2=float(
                torch.linalg.vector_norm((proposal.hidden_state - hidden.detach()).double()).item()
            ),
        )
        report["legacy_diagnostic"] = dict(
            definition=native.TORCH_PC_ENERGY_ID, value=proposal.legacy_diagnostic
        )
        first = tuple(
            _finite(g, "weighted prior gradient", torch).detach().numpy().copy() for g in gradients
        )
        deltas = tuple(
            (candidate[name] - parameters[name]).detach().numpy().copy()
            for name in native._PARAMETERS
        )
        report["gradient_delta_geometry"] = compare_gradient_geometry(first, deltas)
        report["per_parameter_geometry"] = {
            name: compare_gradient_geometry((first[i],), (deltas[i],))
            for i, name in enumerate(native._PARAMETERS)
        }
        _, ordinary_after, weighted_after = _weighted_prior(
            candidate, values, targets, target_weights, work, torch
        )
        report["counterfactual_prior_mean_ce"] = float(ordinary_after.item())
        report["prior_ce_change"] = float(ordinary_after.item()) - report["prior_mean_ce"]
        report["counterfactual_prior_weighted_mean_ce"] = float(weighted_after.item())
        report["weighted_prior_ce_change"] = (
            float(weighted_after.item()) - report["prior_weighted_mean_ce"]
        )
        after = _fixed_state_energy(
            candidate, values, targets, target_weights, proposal.hidden_state, work, torch
        )
        report["fixed_state_weighted_energy_after"] = after
        report["fixed_state_weighted_energy_change"] = after - proposal.energy


def diagnose_pc_update(
    snapshot: object,
    episodes: tuple[DemonstrationEpisode, ...],
    *,
    class_weights: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Measure one ungated native candidate without constructing or stepping a learner.

    Private detached tensors implement the original formulas; the candidate only
    supplies a TRAIN readout, never training, evaluation, tuning or promotion.
    Prior CE gradient uses batch mean; latent stationarity uses the native
    unaveraged per-row gradient. A failed post-admission operation retains costs.
    Explicit weights use the exact existing weighted native proposal and weighted
    prior CE gradient. Both ordinary and weighted CE are reported; fixed-state
    energy after uses its original final z and the candidate's new prior. No
    caller weights are inferred or selected; None preserves the original report.
    """
    sources = _sources(snapshot, episodes)
    weights = None if class_weights is None else validate_class_weights(class_weights, 7)
    bound = cast(InvariantTorchFeatureSnapshot, snapshot)
    torch = require_torch()
    if torch.is_inference_mode_enabled():
        raise ValueError("realized PC diagnostics do not support torch inference_mode")
    rng = torch.get_rng_state().clone()
    numpy_rng = cast(tuple[Any, ...], deepcopy(np.random.get_state()))
    python_rng, grad_mode = random.getstate(), torch.is_grad_enabled()
    report = _report(sources, bound.policy.native_policy)
    if weights is not None:
        _weighted_report(report, weights)
    before, marks, failure = None, None, None

    def check_inputs():
        current = _sources(snapshot, episodes)
        now = cast(tuple[Any, ...], np.random.get_state())
        if (
            current != sources
            or (
                weights is not None
                and validate_class_weights(cast(tuple[float, ...], class_weights), 7) != weights
            )
            or (before is not None and not _same_state(before, snapshot))
            or (marks is not None and not _same_state(marks, _tensor_marks(snapshot, torch)))
            or not torch.equal(rng, torch.get_rng_state())
            or numpy_rng[0] != now[0]
            or not np.array_equal(numpy_rng[1], now[1])
            or numpy_rng[2:] != now[2:]
            or python_rng != random.getstate()
            or torch.is_grad_enabled() != grad_mode
            or torch.is_inference_mode_enabled()
        ):
            raise ValueError("diagnostic input, snapshot, gradient buffers, RNG or mode changed")

    try:
        report["work"]["snapshot_validations_attempted"] += 1
        owned = copy_invariant_torch_feature_snapshot(bound)
        report["work"]["snapshot_validations_completed"] += 1
        before, marks = owned, _tensor_marks(snapshot, torch)
        check_inputs()
        if weights is None:
            _measure(owned, sources, report, torch)
        else:
            _measure_weighted(owned, sources, report, torch, weights)
    except Exception as error:
        failure = error
    try:
        check_inputs()
    except Exception as error:
        failure = failure or error
    if failure is not None:
        report.update(status="incomplete", reason=f"{type(failure).__name__}: {failure}")
        report["work"]["failure_work_may_be_unobserved"] = True
        raise GradientDiagnosticFailed(report) from failure
    report["status"] = "complete"
    return report
