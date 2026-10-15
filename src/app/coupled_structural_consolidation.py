"""Trusted raw structural transforms for owned mixed-history candidates.

Pure split/compatible merge report representation error; pruning delegates to
the existing isolated observed-damage gate. Temporary prune envelopes are an
internal reuse boundary: the captured mixed history always survives. No serving
authority, untrusted-kernel verification, training or implicit rate changes.
"""

import numpy as np

from src.app.coupled_pruning_gate import CoupledPruneLimits
from src.app.coupled_scaling_prune_consolidation import CoupledScalingPruneConsolidator
from src.app.linear_scaling_measurement import ScalingBatch
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.actor_ports import ConsolidatedState
from src.core.coupled_inference import _finite_number
from src.core.coupled_pc_scaling import CoupledScalingState, copy_coupled_scaling_state
from src.core.coupled_scaling_pruning import CoupledScalingPruneLineage
from src.core.coupled_scaling_structure import (
    MAX_EFFECTIVE_RELATIVE_ERROR,
    CoupledScalingSplitLineage,
    CoupledScalingWidthChange,
    merge_coupled_scaling_width,
    split_coupled_scaling_width,
)
from src.core.coupled_scaling_topology import (
    CoupledScalingTopologyState,
    copy_coupled_scaling_topology,
)
from src.core.coupled_structural_state import (
    MAX_STRUCTURAL_TRANSITIONS,
    CoupledStructuralState,
    CoupledStructuralTransition,
    copy_coupled_structural_state,
)
from src.core.learner_ports import TrainingDiagnostic


def _capture(state: CoupledStructuralState) -> CoupledStructuralState:
    captured = copy_coupled_structural_state(state)
    if len(captured.transitions) >= MAX_STRUCTURAL_TRANSITIONS:
        raise ValueError("mixed structural transition capacity exhausted")
    return captured


def _check_wake(
    source: CoupledScalingState, target: CoupledScalingState, indices: tuple[int, ...]
) -> None:
    a, b = source.wake_state, target.wake_state
    if a is None:
        if b is not None:
            raise ValueError("structural change altered native family")
        return
    if (
        b is None
        or b.wake_updates != a.wake_updates
        or not np.array_equal(b.chemical_state, np.take(a.chemical_state, indices))
        or not np.array_equal(b.plasticity_state, np.take(a.plasticity_state, indices))
    ):
        raise ValueError("structural change altered mapped CPC history/count")


def _checked_change(
    result: CoupledScalingWidthChange,
    expected: CoupledStructuralTransition,
    source: CoupledScalingState,
) -> tuple[CoupledScalingState, float]:
    if (
        type(result) is not CoupledScalingWidthChange
        or type(result.lineage) is not CoupledScalingSplitLineage
    ):
        raise ValueError("structural kernel must return exact width change and split lineage")
    lineage = CoupledScalingSplitLineage(
        result.lineage.format_version, result.lineage.source_config
    )
    if lineage != expected.lineage:
        raise ValueError("structural kernel returned incompatible lineage")
    state = copy_coupled_scaling_state(result.state)
    if state.config != expected.target_config:
        raise ValueError("structural kernel returned incompatible full configuration")
    error = _finite_number(
        result.max_effective_relative_error, "representation error", positive=False
    )
    if error > MAX_EFFECTIVE_RELATIVE_ERROR:
        raise ValueError("structural kernel exceeds effective representation error bound")
    indices = (
        tuple(i for i in range(source.config.width) for _ in range(2))
        if expected.kind == "split"
        else tuple(range(0, source.config.width, 2))
    )
    _check_wake(source, state, indices)
    return state, error


def _publish(
    captured: CoupledStructuralState,
    transition: CoupledStructuralTransition,
    raw: CoupledScalingState,
    diagnostic: TrainingDiagnostic,
) -> ConsolidatedState[CoupledStructuralState]:
    if transition.source_config != captured.raw_state.config:
        raise ValueError("structural transition source differs from captured configuration")
    return ConsolidatedState(
        CoupledStructuralState(1, captured.origin_config, raw, (*captured.transitions, transition)),
        diagnostic,
    )


class CoupledScalingSplitConsolidator:
    """Append a validated uniform split; learner inference settings stay fixed."""

    def __init__(self, *, max_width: int | None = None) -> None:
        if max_width is not None and (type(max_width) is not int or not 0 < max_width < 2**63):
            raise ValueError("max_width must be an exact positive integer below2**63 or None")
        self._max_width = max_width

    @property
    def max_width(self) -> int | None:
        return self._max_width

    def __call__(self, state: CoupledStructuralState) -> ConsolidatedState[CoupledStructuralState]:
        captured = _capture(state)
        expected = CoupledStructuralTransition(
            "split", CoupledScalingSplitLineage(1, captured.raw_state.config)
        )
        # Why here: current-state and target metadata are validated, but expanded
        # arrays/models have not been allocated by the split kernel yet.
        if self._max_width is not None and expected.target_config.width > self._max_width:
            raise ValueError("split target width exceeds configured ceiling")
        result = split_coupled_scaling_width(copy_coupled_scaling_state(captured.raw_state))
        raw, error = _checked_change(result, expected, captured.raw_state)
        return _publish(
            captured,
            expected,
            raw,
            TrainingDiagnostic("coupled_scaling_split_effective_encoding_relative_error_v1", error),
        )


class CoupledScalingMergeConsolidator:
    """Own explicit declared split geometry; merge only its compatible target."""

    def __init__(self, lineage: CoupledScalingSplitLineage) -> None:
        if type(lineage) is not CoupledScalingSplitLineage:
            raise ValueError("merge consolidator requires an exact split lineage")
        self._lineage = CoupledScalingSplitLineage(lineage.format_version, lineage.source_config)

    def __call__(self, state: CoupledStructuralState) -> ConsolidatedState[CoupledStructuralState]:
        captured = _capture(state)
        expected = CoupledStructuralTransition("merge", self._lineage)
        if captured.raw_state.config != expected.source_config:
            raise ValueError("merge source differs from full declared split target configuration")
        # Kernel receives detached arguments; a late mutation/fault cannot change
        # this operation's fixed lineage or the caller's full mixed envelope.
        lineage = CoupledScalingSplitLineage(
            self._lineage.format_version, self._lineage.source_config
        )
        result = merge_coupled_scaling_width(
            copy_coupled_scaling_state(captured.raw_state), lineage
        )
        raw, error = _checked_change(result, expected, captured.raw_state)
        return _publish(
            captured,
            expected,
            raw,
            TrainingDiagnostic("coupled_scaling_merge_effective_encoding_relative_error_v1", error),
        )


class CoupledScalingStructuralPruneConsolidator:
    """Reuse accepted raw prune evidence while retaining every mixed transition."""

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
        self._prune = CoupledScalingPruneConsolidator(
            removed_units,
            probes,
            limits=limits,
            budget=budget,
            inference_steps=inference_steps,
            inference_learning_rate=inference_learning_rate,
        )
        self._removed = tuple(i for i in removed_units)

    def __call__(self, state: CoupledStructuralState) -> ConsolidatedState[CoupledStructuralState]:
        captured = _capture(state)
        lineage = CoupledScalingPruneLineage(1, captured.raw_state.config, self._removed)
        expected = CoupledStructuralTransition("prune", lineage)
        result = self._prune(CoupledScalingTopologyState(1, captured.raw_state))
        if type(result) is not ConsolidatedState:
            raise ValueError("prune adapter must return an exact consolidation record")
        envelope = copy_coupled_scaling_topology(result.state)
        if (
            envelope.prune_lineage != (lineage,)
            or envelope.raw_state.config != expected.target_config
        ):
            raise ValueError("prune adapter returned incompatible single-transition evidence")
        diagnostic = result.diagnostic
        if (
            type(diagnostic) is not TrainingDiagnostic
            or diagnostic.definition != "coupled_scaling_prune_max_probability_change_v1"
        ):
            raise ValueError("prune adapter returned incompatible damage diagnostic")
        value = _finite_number(diagnostic.value, "prune probability change", positive=False)
        if value > 1:
            raise ValueError("prune probability change exceeds one")
        _check_wake(captured.raw_state, envelope.raw_state, lineage.retained_indices[-1])
        return _publish(
            captured, expected, envelope.raw_state, TrainingDiagnostic(diagnostic.definition, value)
        )
