"""Adapt an isolated raw prune decision to existing candidate consolidation.

Own state/data before callbacks. Return a complete raw topology replacement or
a detached refusal; neither path mutates an actor or grants serving authority.
"""

from copy import deepcopy
from dataclasses import replace

import numpy as np

from src.app.coupled_pruning_gate import CoupledPruneDiagnostic, CoupledPruneLimits
from src.app.coupled_scaling_pruning_gate import (
    CoupledScalingPruneDecision,
    gate_coupled_scaling_prune,
)
from src.app.linear_scaling_measurement import ScalingBatch, _validate_batch
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.actor_ports import ConsolidatedState
from src.core.coupled_inference import CoupledInferenceDiagnostics, _finite_number
from src.core.coupled_pc_scaling import CoupledScalingState, copy_coupled_scaling_state
from src.core.coupled_predictive_coding import _parameter_arrays
from src.core.coupled_scaling_pruning import (
    MAX_EFFECTIVE_RELATIVE_ERROR,
    CoupledScalingPruneLineage,
)
from src.core.coupled_scaling_topology import (
    CoupledScalingTopologyState,
    copy_coupled_scaling_topology,
)
from src.core.learner_ports import TrainingDiagnostic


def _copy_decision(decision: CoupledScalingPruneDecision) -> CoupledScalingPruneDecision:
    if type(decision) is not CoupledScalingPruneDecision:
        raise ValueError("raw consolidation requires an exact prune decision")
    lineage = decision.lineage
    if lineage is not None:
        if type(lineage) is not CoupledScalingPruneLineage:
            raise ValueError("raw decision requires exact prune lineage")
        lineage = CoupledScalingPruneLineage(
            lineage.format_version, lineage.source_config, lineage.removed_units
        )
    return replace(
        decision,
        state=copy_coupled_scaling_state(decision.state),
        lineage=lineage,
        diagnostics=deepcopy(decision.diagnostics),
    )


class CoupledScalingPruneRefused(ValueError):
    """Rejected or failed raw prune evidence, never a successful consolidation."""

    def __init__(self, decision: CoupledScalingPruneDecision) -> None:
        if (
            type(decision) is not CoupledScalingPruneDecision
            or decision.status not in ("rejected", "failed")
            or decision.lineage is not None
            or decision.max_effective_relative_error is not None
        ):
            raise ValueError("raw prune refusal requires a rejected/failed unaccepted decision")
        self._decision = _copy_decision(decision)
        super().__init__(f"raw coupled prune {decision.status}: {decision.reason}")

    @property
    def decision(self) -> CoupledScalingPruneDecision:
        return _copy_decision(self._decision)

    def __reduce__(
        self,
    ) -> tuple[type["CoupledScalingPruneRefused"], tuple[CoupledScalingPruneDecision]]:
        return type(self), (self._decision,)


def _owned_probes(probes: tuple[ScalingBatch, ...]) -> tuple[ScalingBatch, ...]:
    # Role checks precede touching/copying payloads; geometry-specific admission
    # is repeated by the raw gate once the captured configuration is known.
    if type(probes) is not tuple or not 1 <= len(probes) <= 8:
        raise ValueError("raw consolidation probes require 1..8 train/validation batches")
    if any(
        type(p) is not ScalingBatch
        or type(p.role) is not str
        or p.role not in ("train", "validation")
        for p in probes
    ):
        raise ValueError("raw consolidation requires train/validation; final_test refused")
    arrays: list[np.ndarray] = []
    seen: set[str] = set()
    for probe in probes:
        if (
            type(probe.features) is not np.ndarray
            or probe.features.ndim != 2
            or type(probe.targets) is not np.ndarray
            or probe.targets.ndim != 2
            or not 1 <= len(probe.features) <= 1024
            or probe.features.size + probe.targets.size > 1_000_000
        ):
            raise ValueError("raw consolidation probe exceeds matrix/row/input-target bounds")
        _validate_batch(probe, probe.role, probe.features.shape[1])
        if np.any((probe.targets < 0) | (probe.targets > 1)):
            raise ValueError("raw consolidation targets must be in [0,1]")
        if seen.intersection(probe.row_ids):
            raise ValueError("raw consolidation probe row IDs must be globally disjoint")
        seen.update(probe.row_ids)
        for array in (probe.features, probe.targets):
            if any(np.shares_memory(array, prior) for prior in arrays):
                raise ValueError("raw consolidation probe buffers must not share storage")
            arrays.append(array)
    if not any(p.role == "validation" for p in probes):
        raise ValueError("raw consolidation requires a validation probe")
    return deepcopy(probes)


def _same_raw(left: CoupledScalingState, right: CoupledScalingState) -> bool:
    if left.config != right.config or not all(
        np.array_equal(a, b)
        for a, b in zip(
            _parameter_arrays(left.raw_parameters), _parameter_arrays(right.raw_parameters)
        )
    ):
        return False
    a, b = left.wake_state, right.wake_state
    if a is None or b is None:
        return a is b
    return (
        a.wake_updates == b.wake_updates
        and np.array_equal(a.chemical_state, b.chemical_state)
        and np.array_equal(a.plasticity_state, b.plasticity_state)
    )


def _validate_selected_wake(
    source: CoupledScalingState, candidate: CoupledScalingState, retained: tuple[int, ...]
) -> None:
    before, after = source.wake_state, candidate.wake_state
    if before is None:
        if after is not None:
            raise ValueError("accepted raw prune changed the model family")
        return
    keep = list(retained)
    if (
        after is None
        or after.wake_updates != before.wake_updates
        or not np.array_equal(after.chemical_state, before.chemical_state[keep])
        or not np.array_equal(after.plasticity_state, before.plasticity_state[keep])
    ):
        raise ValueError("accepted raw prune changed retained CPC history")


def _validate_inference(diagnostic: CoupledInferenceDiagnostics, energy: float, steps: int) -> None:
    if type(diagnostic) is not CoupledInferenceDiagnostics:
        raise ValueError("raw prune requires exact native inference diagnostics")
    expected = {
        "attempted_steps": steps,
        "accepted_steps": steps,
        "energy_evaluations": steps + 1,
        "gradient_norm_evaluations": steps + 1,
        "energy_comparisons": steps,
    }
    if (
        diagnostic.reason != "max_steps"
        or type(diagnostic.used_fallback) is not bool
        or diagnostic.used_fallback
        or any(
            type(getattr(diagnostic, name)) is not int or getattr(diagnostic, name) != value
            for name, value in expected.items()
        )
    ):
        raise ValueError("raw prune inference must complete the declared fixed cap")
    for name in ("initial_energy", "final_energy", "final_gradient_inf_norm"):
        _finite_number(getattr(diagnostic, name), name, positive=False)
    if diagnostic.final_energy != energy or diagnostic.final_energy > diagnostic.initial_energy:
        raise ValueError("raw prune inference energy evidence is inconsistent")


def _validate_evidence(
    decision: CoupledScalingPruneDecision,
    probes: tuple[ScalingBatch, ...],
    limits: CoupledPruneLimits,
    steps: int,
) -> None:
    if type(decision.planned_probe_count) is not int or decision.planned_probe_count != len(probes):
        raise ValueError("raw prune planned probe count differs from consolidation")
    if type(decision.diagnostics) is not tuple or len(decision.diagnostics) > len(probes):
        raise ValueError("raw prune diagnostics must be an exact completed probe prefix")
    if decision.status in ("accepted", "rejected") and len(decision.diagnostics) != len(probes):
        raise ValueError("raw prune decision requires every declared probe diagnostic")
    for diagnostic, probe in zip(decision.diagnostics, probes):
        if (
            type(diagnostic) is not CoupledPruneDiagnostic
            or diagnostic.role != probe.role
            or diagnostic.row_ids != probe.row_ids
        ):
            raise ValueError("raw prune diagnostic role/rows differ from the declared probe")
        for name in (
            "source_bce",
            "candidate_bce",
            "source_inferred_energy",
            "candidate_inferred_energy",
            "max_prediction_change",
            "bce_increase",
            "inferred_energy_increase",
        ):
            _finite_number(getattr(diagnostic, name), name, positive=False)
        # Native gate arithmetic is retained exactly; no tolerance can hide a
        # positive increase at a caller's zero damage threshold.
        if (
            diagnostic.max_prediction_change > 1
            or diagnostic.bce_increase != max(0.0, diagnostic.candidate_bce - diagnostic.source_bce)
            or diagnostic.inferred_energy_increase
            != max(0.0, diagnostic.candidate_inferred_energy - diagnostic.source_inferred_energy)
        ):
            raise ValueError("raw prune damage evidence is inconsistent")
        _validate_inference(diagnostic.source_inference, diagnostic.source_inferred_energy, steps)
        _validate_inference(
            diagnostic.candidate_inference, diagnostic.candidate_inferred_energy, steps
        )
        if decision.status == "accepted" and (
            diagnostic.max_prediction_change > limits.max_prediction_change
            or diagnostic.bce_increase > limits.max_bce_increase
            or diagnostic.inferred_energy_increase > limits.max_inferred_energy_increase
        ):
            raise ValueError("accepted raw prune diagnostics exceed the declared damage limits")


class CoupledScalingPruneConsolidator:
    """Trusted ActorShadowRuntime transform; budget accounting remains shared."""

    def __init__(
        self,
        removed_units: tuple[int, ...],
        probes: tuple[ScalingBatch, ...],
        *,
        limits: CoupledPruneLimits,
        budget: ToyBudgetSession,
        inference_steps: int,
        inference_learning_rate: float,
    ) -> None:
        if (
            type(removed_units) is not tuple
            or not 1 <= len(removed_units) <= 16
            or any(type(i) is not int or i < 0 for i in removed_units)
        ):
            raise ValueError("removed_units requires 1..16 native nonnegative coordinates")
        if type(limits) is not CoupledPruneLimits or type(budget) is not ToyBudgetSession:
            raise ValueError("raw consolidator requires exact prune limits and budget records")
        if type(inference_steps) is not int or not 1 <= inference_steps <= 64:
            raise ValueError("inference_steps must be a native integer in 1..64")
        self._rate = _finite_number(
            inference_learning_rate, "inference_learning_rate", positive=True
        )
        self._probes = _owned_probes(probes)
        self._removed = tuple(i for i in removed_units)
        self._limits = CoupledPruneLimits(
            limits.max_prediction_change,
            limits.max_bce_increase,
            limits.max_inferred_energy_increase,
        )
        self._budget, self._steps = budget, inference_steps

    def __call__(
        self, state: CoupledScalingTopologyState
    ) -> ConsolidatedState[CoupledScalingTopologyState]:
        captured = copy_coupled_scaling_topology(state)
        if len(captured.prune_lineage) >= 64:
            raise ValueError("raw topology prune lineage capacity exhausted")
        expected = CoupledScalingPruneLineage(1, captured.raw_state.config, self._removed)
        config = captured.raw_state.config
        for probe in self._probes:
            _validate_batch(probe, probe.role, config.input_dim)
            entries = len(probe.features) * (
                config.input_dim + 1 + config.width * config.hidden_layers
            )
            if entries > 1_000_000:
                raise ValueError("raw consolidation probe exceeds hidden activation entry cap")
        decision = gate_coupled_scaling_prune(
            copy_coupled_scaling_state(captured.raw_state),
            self._removed,
            deepcopy(self._probes),
            limits=replace(self._limits),
            budget=self._budget,
            inference_steps=self._steps,
            inference_learning_rate=self._rate,
        )
        decision = _copy_decision(decision)
        _validate_evidence(decision, self._probes, self._limits, self._steps)
        if decision.status != "accepted":
            if decision.status not in ("rejected", "failed") or not _same_raw(
                decision.state, captured.raw_state
            ):
                raise ValueError("refused raw prune did not retain the complete source")
            raise CoupledScalingPruneRefused(decision)
        if (
            decision.reason is not None
            or decision.lineage != expected
            or decision.state.config != expected.target_config
        ):
            raise ValueError(
                "accepted raw prune has incompatible configuration or coordinate lineage"
            )
        if decision.max_effective_relative_error is None:
            raise ValueError("accepted raw prune is missing its encoding error")
        error = _finite_number(
            decision.max_effective_relative_error, "encoding error", positive=False
        )
        if error > MAX_EFFECTIVE_RELATIVE_ERROR:
            raise ValueError("accepted raw prune exceeds the representation error bound")
        _validate_selected_wake(captured.raw_state, decision.state, expected.retained_indices[-1])
        prepared = CoupledScalingTopologyState(
            1, decision.state, (*captured.prune_lineage, decision.lineage)
        )
        return ConsolidatedState(
            prepared,
            TrainingDiagnostic(
                "coupled_scaling_prune_max_probability_change_v1",
                max(item.max_prediction_change for item in decision.diagnostics),
            ),
        )
