"""Assess an owned linear prune against declared training/validation probes.

This application boundary owns copies, observed damage limits and complete
replacement selection. Rejection/failure returns the retained source. No final
test data, training, threshold tuning, runtime authority or global error bound.
"""

from dataclasses import dataclass, replace
from math import isfinite

import numpy as np

from src.app.linear_scaling_measurement import ScalingBatch, _copy_batch, _validate_batch
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.linear_pc_reference import Array, LinearEquilibrium, evaluate_linear_equilibrium
from src.core.linear_pc_scaling import LinearScalingState, effective_linear_weights
from src.core.linear_width_pruning import (
    LinearPruneChange,
    LinearPruneLineage,
    prune_linear_width,
)
from src.core.residual_linear_pc import LinearResidualState, effective_residual_weights

State = LinearScalingState | LinearResidualState
MAX_PROBE_ENTRIES = 1_000_000


def _nonnegative(value: float, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a native finite nonnegative number")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    if not isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


@dataclass(frozen=True)
class PruneDamageLimits:
    max_prediction_change: float
    max_half_mse_increase: float
    max_energy_increase: float

    def __post_init__(self) -> None:
        for name in ("max_prediction_change", "max_half_mse_increase", "max_energy_increase"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))


@dataclass(frozen=True)
class PruneProbeDiagnostic:
    role: str
    row_ids: tuple[str, ...]
    source_energy: float
    candidate_energy: float
    source_half_mse: float
    candidate_half_mse: float
    max_prediction_change: float
    half_mse_increase: float
    energy_increase: float


@dataclass(frozen=True)
class PruneDecision:
    status: str
    reason: str | None
    state: State
    lineage: LinearPruneLineage | None
    diagnostics: tuple[PruneProbeDiagnostic, ...]
    planned_probe_count: int


def _copy_state(state: State) -> State:
    if type(state) is LinearScalingState:
        linear = LinearScalingState(replace(state.config), state.raw_weights)
        effective_linear_weights(linear)
        return linear
    elif type(state) is LinearResidualState:
        residual = LinearResidualState(replace(state.config), state.raw_weights)
        effective_residual_weights(residual)
        return residual
    else:
        raise ValueError("initial must be an exact supported linear state")


def _prepare(
    initial: State,
    index: int,
    probes: tuple[ScalingBatch, ...],
    limits: PruneDamageLimits,
    budget: ToyBudgetSession,
) -> tuple[State, tuple[ScalingBatch, ...], PruneDamageLimits]:
    source = _copy_state(initial)
    if type(index) is not int or source.config.width < 2 or not 0 <= index < source.config.width:
        raise ValueError("pruning requires width>=2 and a native index within that width")
    if type(limits) is not PruneDamageLimits or type(budget) is not ToyBudgetSession:
        raise ValueError("gate requires exact PruneDamageLimits and ToyBudgetSession records")
    owned_limits = PruneDamageLimits(
        limits.max_prediction_change, limits.max_half_mse_increase, limits.max_energy_increase
    )
    if type(probes) is not tuple or not 1 <= len(probes) <= 8:
        raise ValueError("probes must be a tuple containing 1..8 train/validation batches")
    seen: set[str] = set()
    arrays: list[Array] = []
    validation_seen = False
    for probe in probes:
        if (
            type(probe) is not ScalingBatch
            or type(probe.role) is not str
            or probe.role not in ("train", "validation")
        ):
            raise ValueError(
                "prune probes require declared train or validation roles; final_test is refused"
            )
        if type(probe.features) is not np.ndarray or probe.features.ndim != 2:
            raise ValueError("probe features must be an exact matrix")
        rows = len(probe.features)
        entries = rows * (
            source.config.input_dim + 1 + source.config.width * source.config.hidden_layers
        )
        if not 1 <= rows <= 1024 or entries > MAX_PROBE_ENTRIES:
            raise ValueError("probe exceeds row or input/target/hidden activation entry cap")
        _validate_batch(probe, probe.role, source.config.input_dim)
        if seen.intersection(probe.row_ids):
            raise ValueError("probe row IDs must be globally disjoint")
        seen.update(probe.row_ids)
        for array in (probe.features, probe.targets):
            if any(np.shares_memory(array, other) for other in arrays):
                raise ValueError("probe buffers must not share storage")
            arrays.append(array)
        validation_seen |= probe.role == "validation"
    if not validation_seen:
        raise ValueError("at least one validation probe is required")
    return source, tuple(_copy_batch(probe) for probe in probes), owned_limits


@dataclass(frozen=True)
class _Score:
    energy: float
    half_mse: float
    predictions: Array


def _score(state: State, probe: ScalingBatch) -> _Score:
    # Why clone per evaluation: faulty evaluation code must not alter the
    # retained fallback, the candidate, another probe, or the comparison baseline.
    working = _copy_state(state)
    weights = (
        effective_linear_weights(working)
        if isinstance(working, LinearScalingState)
        else effective_residual_weights(working)
    )
    result = evaluate_linear_equilibrium(weights, probe.features.copy(), probe.targets.copy())
    if type(result) is not LinearEquilibrium:
        raise ValueError("native evaluation must return a LinearEquilibrium")
    energy = _nonnegative(result.energy, "evaluated energy")
    half_mse = _nonnegative(result.mse, "evaluated half-MSE")
    predictions = result.predictions
    if (
        type(predictions) is not np.ndarray
        or predictions.dtype != np.float64
        or predictions.shape != (len(probe.features), 1)
        or not np.isfinite(predictions).all()
    ):
        raise ValueError("evaluated predictions must be a finite float64 (rows,1) matrix")
    # The post-evaluation resource callback may retain/mutate the evaluator's
    # result, so detach relevant outputs before returning to external code.
    return _Score(energy, half_mse, predictions.copy())


def _diagnostic(probe: ScalingBatch, source: _Score, candidate: _Score) -> PruneProbeDiagnostic:
    with np.errstate(over="raise", invalid="raise"):
        prediction_change = float(np.max(np.abs(candidate.predictions - source.predictions)))
    return PruneProbeDiagnostic(
        probe.role,
        probe.row_ids,
        source.energy,
        candidate.energy,
        source.half_mse,
        candidate.half_mse,
        _nonnegative(prediction_change, "prediction change"),
        max(0.0, candidate.half_mse - source.half_mse),
        max(0.0, candidate.energy - source.energy),
    )


def _candidate(source: State, index: int) -> tuple[State, LinearPruneLineage]:
    change = prune_linear_width(_copy_state(source), index)
    family = "mlp" if isinstance(source, LinearScalingState) else "residual"
    expected_lineage = LinearPruneLineage(1, family, source.config, index)
    if (
        type(change) is not LinearPruneChange
        or type(change.lineage) is not LinearPruneLineage
        or change.lineage != expected_lineage
    ):
        raise ValueError("native prune returned incompatible coordinate lineage")
    candidate = _copy_state(change.state)
    if type(candidate) is not type(source) or candidate.config != replace(
        source.config, width=source.config.width - 1
    ):
        raise ValueError("native prune returned an incompatible candidate configuration")
    return candidate, expected_lineage


def gate_linear_prune(
    initial: State,
    index: int,
    probes: tuple[ScalingBatch, ...],
    *,
    limits: PruneDamageLimits,
    budget: ToyBudgetSession,
) -> PruneDecision:
    """Return a complete accepted replacement, or the detached retained source.

    Every declared probe completes before acceptance or observed-limit rejection.
    Numerical/resource/candidate failures return failed with known diagnostics;
    invalid public preflight raises. Limits bound observations on these probes,
    not unseen examples, future training, or a global function/objective error.
    """
    source, probes, limits = _prepare(initial, index, probes, limits, budget)
    diagnostics: list[PruneProbeDiagnostic] = []
    try:
        budget.before_final()
        candidate, lineage = _candidate(source, index)
        for probe in probes:
            budget.before_final()
            before = _score(source, probe)
            budget.before_final()
            after = _score(candidate, probe)
            diagnostics.append(_diagnostic(probe, before, after))
            budget.before_final()
        budget.before_final()
        rejected = any(
            item.max_prediction_change > limits.max_prediction_change
            or item.half_mse_increase > limits.max_half_mse_increase
            or item.energy_increase > limits.max_energy_increase
            for item in diagnostics
        )
        if rejected:
            return PruneDecision(
                "rejected",
                "observed probe damage exceeded declared limits",
                source,
                None,
                tuple(diagnostics),
                len(probes),
            )
        return PruneDecision("accepted", None, candidate, lineage, tuple(diagnostics), len(probes))
    except Exception as error:
        return PruneDecision(
            "failed",
            f"{type(error).__name__}: {error}",
            source,
            None,
            tuple(diagnostics),
            len(probes),
        )
