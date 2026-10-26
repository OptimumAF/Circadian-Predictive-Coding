"""Select complete native PC/CPC pruning by declared bounded probe observations.

Private model/data/score copies survive rejection and evaluator/budget faults.
Relaxed energy is a fixed-step diagnostic, not an equilibrium or compute match.
No training, final-test selection, shared-holder publication or global guarantee.
"""

from dataclasses import dataclass, replace

import numpy as np

from src.core.binary_probability_scores import probability_scores as _probability_scores
from src.app.linear_scaling_measurement import ScalingBatch, _copy_batch, _validate_batch
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.coupled_circadian_predictive_coding import CoupledCircadianPredictiveCodingNetwork
from src.core.coupled_energy import COUPLED_ENERGY_ID, Array, predict_coupled
from src.core.coupled_inference import (
    CoupledInferenceDiagnostics,
    CoupledInferencePolicy,
    CoupledInferenceResult,
    _finite_number,
    infer_coupled,
)
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.coupled_width_pruning import (
    CoupledPruneChange,
    CoupledPruneLineage,
    copy_coupled_model,
    prune_coupled_width,
)


@dataclass(frozen=True)
class CoupledPruneLimits:
    max_prediction_change: float
    max_bce_increase: float
    max_inferred_energy_increase: float

    def __post_init__(self) -> None:
        for name in ("max_prediction_change", "max_bce_increase", "max_inferred_energy_increase"):
            object.__setattr__(
                self, name, _finite_number(getattr(self, name), name, positive=False)
            )


@dataclass(frozen=True)
class CoupledPruneDiagnostic:
    role: str
    row_ids: tuple[str, ...]
    source_bce: float
    candidate_bce: float
    source_inferred_energy: float
    candidate_inferred_energy: float
    max_prediction_change: float
    bce_increase: float
    inferred_energy_increase: float
    source_inference: CoupledInferenceDiagnostics
    candidate_inference: CoupledInferenceDiagnostics


@dataclass(frozen=True)
class CoupledPruneDecision:
    status: str
    reason: str | None
    model: CoupledPredictiveCodingNetwork
    lineage: CoupledPruneLineage | None
    diagnostics: tuple[CoupledPruneDiagnostic, ...]
    planned_probe_count: int


def _copy_probes(
    model: CoupledPredictiveCodingNetwork, probes: tuple[ScalingBatch, ...]
) -> tuple[ScalingBatch, ...]:
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
                "coupled pruning accepts train/validation probes; final_test is refused"
            )
        if type(probe.features) is not np.ndarray or probe.features.ndim != 2:
            raise ValueError("probe features must be an exact matrix")
        rows = len(probe.features)
        entries = rows * (model.input_dim + 1 + sum(model.hidden_dims))
        if not 1 <= rows <= 1024 or entries > 1_000_000:
            raise ValueError("probe exceeds its row or input/target/hidden activation entry cap")
        _validate_batch(probe, probe.role, model.input_dim)
        if np.any((probe.targets < 0) | (probe.targets > 1)):
            raise ValueError("coupled probe targets must be in [0,1]")
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
    return tuple(_copy_batch(probe) for probe in probes)


def _lineage(model: CoupledPredictiveCodingNetwork, layer: int, unit: int) -> CoupledPruneLineage:
    return CoupledPruneLineage(
        1,
        "cpc" if type(model) is CoupledCircadianPredictiveCodingNetwork else "pc",
        model.input_dim,
        model.hidden_dims,
        model.precisions,
        layer,
        unit,
    )


def _prepare(
    initial: CoupledPredictiveCodingNetwork,
    layer: int,
    unit: int,
    probes: tuple[ScalingBatch, ...],
    limits: CoupledPruneLimits,
    budget: ToyBudgetSession,
    steps: int,
    rate: float,
) -> tuple[
    CoupledPredictiveCodingNetwork,
    CoupledPruneLineage,
    tuple[ScalingBatch, ...],
    CoupledPruneLimits,
    float,
]:
    source = copy_coupled_model(initial)
    lineage = _lineage(source, layer, unit)
    if type(limits) is not CoupledPruneLimits or type(budget) is not ToyBudgetSession:
        raise ValueError("gate requires exact CoupledPruneLimits and ToyBudgetSession records")
    copied_limits = CoupledPruneLimits(
        limits.max_prediction_change, limits.max_bce_increase, limits.max_inferred_energy_increase
    )
    if type(steps) is not int or not 1 <= steps <= 64:
        raise ValueError("inference_steps must be a native integer in 1..64")
    resolved_rate = _finite_number(rate, "inference_learning_rate", positive=True)
    return source, lineage, _copy_probes(source, probes), copied_limits, resolved_rate


@dataclass(frozen=True)
class _Score:
    probabilities: Array
    bce: float
    energy: float
    inference: CoupledInferenceDiagnostics


def _score(
    model: CoupledPredictiveCodingNetwork, probe: ScalingBatch, steps: int, rate: float
) -> _Score:
    # Prediction and inference use independent native copies and snapshots, so
    # either operation's borrowed buffers cannot contaminate the other or fallback.
    prediction_model = copy_coupled_model(model)
    probabilities = predict_coupled(
        prediction_model.snapshot_state().parameters, probe.features.copy()
    )
    bce = _probability_scores(probabilities, probe.targets.copy())["bce"]
    probabilities = probabilities.copy()
    inference_model = copy_coupled_model(model)
    snapshot = inference_model.snapshot_state()
    result = infer_coupled(
        snapshot.parameters,
        probe.features.copy(),
        probe.targets.copy(),
        snapshot.precisions,
        max_steps=steps,
        learning_rate=rate,
        policy=CoupledInferencePolicy(
            min_steps=steps, gradient_tolerance=None, relative_energy_tolerance=None
        ),
    )
    if (
        type(result) is not CoupledInferenceResult
        or type(result.diagnostics) is not CoupledInferenceDiagnostics
    ):
        raise ValueError("inference must return an exact native result and diagnostics")
    diagnostic = result.diagnostics
    if (
        diagnostic.reason != "max_steps"
        or type(diagnostic.attempted_steps) is not int
        or type(diagnostic.accepted_steps) is not int
        or diagnostic.attempted_steps != steps
        or diagnostic.accepted_steps != steps
        or type(diagnostic.used_fallback) is not bool
        or diagnostic.used_fallback
    ):
        raise ValueError("probe inference did not complete every declared step without fallback")
    if result.evaluation.energy_definition != COUPLED_ENERGY_ID:
        raise ValueError("inferred energy definition differs from the native objective")
    energy = _finite_number(result.evaluation.energy, "inferred energy", positive=False)
    return _Score(probabilities, bce, energy, replace(diagnostic))


def _diagnostic(probe: ScalingBatch, source: _Score, candidate: _Score) -> CoupledPruneDiagnostic:
    with np.errstate(over="raise", invalid="raise"):
        prediction_change = float(np.max(np.abs(candidate.probabilities - source.probabilities)))
    return CoupledPruneDiagnostic(
        probe.role,
        probe.row_ids,
        source.bce,
        candidate.bce,
        source.energy,
        candidate.energy,
        _finite_number(prediction_change, "prediction change", positive=False),
        max(0.0, candidate.bce - source.bce),
        max(0.0, candidate.energy - source.energy),
        source.inference,
        candidate.inference,
    )


def _candidate(
    source: CoupledPredictiveCodingNetwork, lineage: CoupledPruneLineage
) -> CoupledPredictiveCodingNetwork:
    change = prune_coupled_width(
        copy_coupled_model(source), lineage.layer_index, lineage.unit_index
    )
    if (
        type(change) is not CoupledPruneChange
        or type(change.lineage) is not CoupledPruneLineage
        or change.lineage != lineage
    ):
        raise ValueError("native prune returned incompatible coordinate lineage")
    candidate = copy_coupled_model(change.model)
    if (
        type(candidate) is not type(source)
        or candidate.input_dim != source.input_dim
        or candidate.hidden_dims != lineage.target_hidden_dims
        or candidate.precisions != source.precisions
    ):
        raise ValueError("native prune returned incompatible model geometry or precision")
    if type(source) is CoupledCircadianPredictiveCodingNetwork:
        if type(candidate) is not CoupledCircadianPredictiveCodingNetwork:
            raise ValueError("native prune changed the CPC model family")
        chemicals = source.get_chemical_state()
        gates = source.get_plasticity_state()
        if lineage.layer_index == len(source.hidden_dims) - 1:
            chemicals = np.delete(chemicals, lineage.unit_index)
            gates = np.delete(gates, lineage.unit_index)
        if (
            candidate.config != source.config
            or candidate.wake_updates != source.wake_updates
            or not np.array_equal(candidate.get_chemical_state(), chemicals)
            or not np.array_equal(candidate.get_plasticity_state(), gates)
        ):
            raise ValueError("native prune changed retained CPC policy or wake state")
    # The native transfer owns chemistry validation. Copying full state again
    # detaches its returned model before subsequent callbacks can retain it.
    return candidate


def gate_coupled_prune(
    initial: CoupledPredictiveCodingNetwork,
    layer_index: int,
    unit_index: int,
    probes: tuple[ScalingBatch, ...],
    *,
    limits: CoupledPruneLimits,
    budget: ToyBudgetSession,
    inference_steps: int,
    inference_learning_rate: float,
) -> CoupledPruneDecision:
    """Return an owned complete model or retained source after all declared probes.

    Score energy is fixed-step guarded relaxed energy, not equilibrium/minimum.
    Invalid preflight raises. Rejection/failure never exposes an accepted lineage
    or candidate. Completed diagnostics remain explicit with planned probe count.
    """
    source, lineage, probes, limits, rate = _prepare(
        initial,
        layer_index,
        unit_index,
        probes,
        limits,
        budget,
        inference_steps,
        inference_learning_rate,
    )
    diagnostics: list[CoupledPruneDiagnostic] = []
    try:
        budget.before_final()
        candidate = _candidate(source, lineage)
        for probe in probes:
            budget.before_final()
            before = _score(source, probe, inference_steps, rate)
            budget.before_final()
            after = _score(candidate, probe, inference_steps, rate)
            diagnostics.append(_diagnostic(probe, before, after))
            budget.before_final()
        budget.before_final()
        rejected = any(
            item.max_prediction_change > limits.max_prediction_change
            or item.bce_increase > limits.max_bce_increase
            or item.inferred_energy_increase > limits.max_inferred_energy_increase
            for item in diagnostics
        )
        if rejected:
            return CoupledPruneDecision(
                "rejected",
                "observed probe damage exceeded declared limits",
                source,
                None,
                tuple(diagnostics),
                len(probes),
            )
        return CoupledPruneDecision(
            "accepted", None, candidate, lineage, tuple(diagnostics), len(probes)
        )
    except Exception as error:
        return CoupledPruneDecision(
            "failed",
            f"{type(error).__name__}: {error}",
            source,
            None,
            tuple(diagnostics),
            len(probes),
        )
