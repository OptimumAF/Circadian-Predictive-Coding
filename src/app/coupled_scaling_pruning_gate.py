"""Isolated train/validation damage gate for uniform-width raw PC/CPC pruning.

The complete raw fallback and accepted candidate are independently owned. Native
probability/BCE/fixed-step relaxed-energy probes use existing scoring boundaries;
no training, final-test selection, publication or trajectory guarantee lives here.
"""

from dataclasses import dataclass, replace
from math import isfinite

import numpy as np

from src.app.coupled_pruning_gate import (
    CoupledPruneDiagnostic,
    CoupledPruneLimits,
    _Score,
    _copy_probes,
    _diagnostic,
    _score,
)
from src.app.linear_scaling_measurement import ScalingBatch, _copy_batch
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.coupled_inference import CoupledInferenceDiagnostics, _finite_number
from src.core.coupled_pc_scaling import (
    CoupledScalingState,
    _validated_model,
    copy_coupled_scaling_state,
)
from src.core.coupled_scaling_pruning import (
    MAX_EFFECTIVE_RELATIVE_ERROR,
    CoupledScalingPruneChange,
    CoupledScalingPruneLineage,
    prune_coupled_scaling_width,
)


@dataclass(frozen=True)
class CoupledScalingPruneDecision:
    status: str
    reason: str | None
    state: CoupledScalingState
    lineage: CoupledScalingPruneLineage | None
    max_effective_relative_error: float | None
    diagnostics: tuple[CoupledPruneDiagnostic, ...]
    planned_probe_count: int


def _prepare(
    initial: CoupledScalingState,
    removed_units: tuple[int, ...],
    probes: tuple[ScalingBatch, ...],
    limits: CoupledPruneLimits,
    budget: ToyBudgetSession,
    steps: int,
    rate: float,
) -> tuple[
    CoupledScalingState,
    CoupledScalingPruneLineage,
    tuple[ScalingBatch, ...],
    CoupledPruneLimits,
    float,
]:
    source = copy_coupled_scaling_state(initial)
    lineage = CoupledScalingPruneLineage(1, source.config, removed_units)
    if type(limits) is not CoupledPruneLimits or type(budget) is not ToyBudgetSession:
        raise ValueError("raw gate requires exact limits and budget records")
    limits = CoupledPruneLimits(
        limits.max_prediction_change, limits.max_bce_increase, limits.max_inferred_energy_increase
    )
    if type(steps) is not int or not 1 <= steps <= 64:
        raise ValueError("inference_steps must be a native integer in 1..64")
    rate = _finite_number(rate, "inference_learning_rate", positive=True)
    _, _, model = _validated_model(source)
    return source, lineage, _copy_probes(model, probes), limits, rate


def _candidate(
    source: CoupledScalingState, lineage: CoupledScalingPruneLineage
) -> tuple[CoupledScalingState, float]:
    # Why: the proposal function may retain its arguments; fallback ownership
    # must not depend on the lifetime of those borrowed copies.
    change = prune_coupled_scaling_width(copy_coupled_scaling_state(source), lineage.removed_units)
    if (
        type(change) is not CoupledScalingPruneChange
        or type(change.lineage) is not CoupledScalingPruneLineage
        or change.lineage != lineage
    ):
        raise ValueError("raw prune returned incompatible coordinate lineage")
    candidate = copy_coupled_scaling_state(change.state)
    if candidate.config != lineage.target_config:
        raise ValueError("raw prune changed configuration beyond the selected uniform width")
    error = change.max_effective_relative_error
    if (
        type(error) not in (float, int)
        or not isfinite(error)
        or not 0 <= error <= MAX_EFFECTIVE_RELATIVE_ERROR
    ):
        raise ValueError("raw prune returned an invalid effective encoding error")
    before, after = source.wake_state, candidate.wake_state
    if before is not None:
        indices = list(lineage.retained_indices[-1])
        if (
            after is None
            or after.wake_updates != before.wake_updates
            or not np.array_equal(after.chemical_state, before.chemical_state[indices])
            or not np.array_equal(after.plasticity_state, before.plasticity_state[indices])
        ):
            raise ValueError("raw prune changed retained CPC chemistry, gates or count")
    return candidate, float(error)


def _owned_score(
    state: CoupledScalingState, probe: ScalingBatch, steps: int, rate: float
) -> _Score:
    _, _, model = _validated_model(copy_coupled_scaling_state(state))
    score = _score(model, _copy_batch(probe), steps, rate)
    if type(score) is not _Score or type(score.inference) is not CoupledInferenceDiagnostics:
        raise ValueError("raw pruning requires an exact native score and inference diagnostic")
    predictions, inference = score.probabilities, score.inference
    if (
        type(predictions) is not np.ndarray
        or predictions.dtype != np.float64
        or predictions.shape != probe.targets.shape
        or not np.isfinite(predictions).all()
        or np.any((predictions < 0) | (predictions > 1))
        or inference.reason != "max_steps"
        or type(inference.attempted_steps) is not int
        or type(inference.accepted_steps) is not int
        or inference.attempted_steps != steps
        or inference.accepted_steps != steps
        or type(inference.used_fallback) is not bool
        or inference.used_fallback
    ):
        raise ValueError("raw pruning score requires finite probabilities and complete inference")
    for name in ("energy_evaluations", "gradient_norm_evaluations", "energy_comparisons"):
        value = getattr(inference, name)
        if type(value) is not int or value < 0:
            raise ValueError("probe inference counters must be native nonnegative integers")
    for name in ("initial_energy", "final_energy", "final_gradient_inf_norm"):
        _finite_number(getattr(inference, name), name, positive=False)
    return _Score(
        predictions.copy(),
        _finite_number(score.bce, "probe BCE", positive=False),
        _finite_number(score.energy, "probe energy", positive=False),
        replace(inference),
    )


def gate_coupled_scaling_prune(
    initial: CoupledScalingState,
    removed_units: tuple[int, ...],
    probes: tuple[ScalingBatch, ...],
    *,
    limits: CoupledPruneLimits,
    budget: ToyBudgetSession,
    inference_steps: int,
    inference_learning_rate: float,
) -> CoupledScalingPruneDecision:
    """Return a detached full raw state after bounded observed damage checks.

    Invalid preflight raises. Rejection or an execution fault returns a private
    source copy, known diagnostics and no accepted lineage/encoding-error claim.
    Fixed-step guarded probe energy is not an equilibrium or global bound.
    """
    source, lineage, probes, limits, rate = _prepare(
        initial, removed_units, probes, limits, budget, inference_steps, inference_learning_rate
    )
    diagnostics: list[CoupledPruneDiagnostic] = []
    try:
        budget.before_final()
        candidate, encoding_error = _candidate(source, lineage)
        for probe in probes:
            budget.before_final()
            before = _owned_score(source, probe, inference_steps, rate)
            budget.before_final()
            after = _owned_score(candidate, probe, inference_steps, rate)
            diagnostics.append(_diagnostic(probe, before, after))
            budget.before_final()
        budget.before_final()
        if any(
            row.max_prediction_change > limits.max_prediction_change
            or row.bce_increase > limits.max_bce_increase
            or row.inferred_energy_increase > limits.max_inferred_energy_increase
            for row in diagnostics
        ):
            return CoupledScalingPruneDecision(
                "rejected",
                "observed probe damage exceeded declared limits",
                source,
                None,
                None,
                tuple(diagnostics),
                len(probes),
            )
        return CoupledScalingPruneDecision(
            "accepted", None, candidate, lineage, encoding_error, tuple(diagnostics), len(probes)
        )
    except Exception as error:
        return CoupledScalingPruneDecision(
            "failed",
            f"{type(error).__name__}: {error}",
            source,
            None,
            None,
            tuple(diagnostics),
            len(probes),
        )
