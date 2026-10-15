"""Adapt an isolated native prune decision to existing candidate consolidation.

Inputs: owned topology state and declared train/validation probes/limits/budget.
Output: complete replacement state or a detached refusal carrying gate evidence.
No actor mutation, promotion authority, training or final-data access belongs here.
"""

from copy import deepcopy
from dataclasses import replace

import numpy as np

from src.app.coupled_pruning_gate import (
    CoupledPruneDecision,
    CoupledPruneLimits,
    gate_coupled_prune,
)
from src.app.linear_scaling_measurement import ScalingBatch, _validate_batch
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.actor_ports import ConsolidatedState
from src.core.coupled_inference import _finite_number
from src.core.coupled_topology import (
    CoupledTopologyState,
    capture_coupled_topology,
    restore_coupled_topology,
)
from src.core.coupled_width_pruning import copy_coupled_model
from src.core.learner_ports import TrainingDiagnostic


def _copy_decision(decision: CoupledPruneDecision) -> CoupledPruneDecision:
    return replace(
        decision,
        model=copy_coupled_model(decision.model),
        lineage=deepcopy(decision.lineage),
        diagnostics=deepcopy(decision.diagnostics),
    )


class CoupledPruneRefused(ValueError):
    """A gate refusal is not a successful candidate consolidation or ticket."""

    def __init__(self, decision: CoupledPruneDecision) -> None:
        if type(decision) is not CoupledPruneDecision or decision.status not in (
            "rejected",
            "failed",
        ):
            raise ValueError("prune refusal requires a rejected or failed gate decision")
        self._decision = _copy_decision(decision)
        super().__init__(f"coupled prune {decision.status}: {decision.reason}")

    @property
    def decision(self) -> CoupledPruneDecision:
        return _copy_decision(self._decision)

    def __reduce__(self) -> tuple[type["CoupledPruneRefused"], tuple[CoupledPruneDecision]]:
        return type(self), (self._decision,)


class CoupledPruneConsolidator:
    """Trusted transform for ActorShadowRuntime.consolidate; no serving swap.

    The budget session remains shared accounting state. Configuration/data are
    copied; each invocation gates a fresh native model reconstructed from state.
    """

    def __init__(
        self,
        layer_index: int,
        unit_index: int,
        probes: tuple[ScalingBatch, ...],
        *,
        limits: CoupledPruneLimits,
        budget: ToyBudgetSession,
        inference_steps: int,
        inference_learning_rate: float,
    ) -> None:
        for name, value in (("layer_index", layer_index), ("unit_index", unit_index)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a native nonnegative integer")
        # Role checks precede even copying payloads. Full shape/storage validation
        # remains in the existing gate and depends on the captured model geometry.
        if type(probes) is not tuple or not 1 <= len(probes) <= 8:
            raise ValueError("probes must contain 1..8 train/validation batches")
        if any(
            type(probe) is not ScalingBatch
            or type(probe.role) is not str
            or probe.role not in ("train", "validation")
            for probe in probes
        ):
            raise ValueError("consolidation probes require train/validation; final_test refused")
        buffers: list[np.ndarray] = []
        for probe in probes:
            if (
                type(probe.features) is not np.ndarray
                or probe.features.ndim != 2
                or type(probe.targets) is not np.ndarray
                or probe.targets.ndim != 2
                or not 1 <= len(probe.features) <= 1024
                or probe.features.size + probe.targets.size > 1_000_000
            ):
                raise ValueError("probe exceeds matrix/row/input-target entry bounds")
            _validate_batch(probe, probe.role, probe.features.shape[1])
            if np.any((probe.targets < 0) | (probe.targets > 1)):
                raise ValueError("coupled probe targets must be in [0,1]")
            for array in (probe.features, probe.targets):
                if type(array) is not np.ndarray:
                    raise ValueError("probe buffers must be exact arrays")
                if any(np.shares_memory(array, prior) for prior in buffers):
                    raise ValueError("probe buffers must not share storage")
                buffers.append(array)
        if type(limits) is not CoupledPruneLimits or type(budget) is not ToyBudgetSession:
            raise ValueError("consolidator requires declared prune limits and ToyBudgetSession")
        if type(inference_steps) is not int or not 1 <= inference_steps <= 64:
            raise ValueError("inference_steps must be a native integer in 1..64")
        self._rate = _finite_number(
            inference_learning_rate, "inference_learning_rate", positive=True
        )
        self._layer, self._unit = layer_index, unit_index
        # Snapshot caller-owned records before any budget callback can mutate them.
        self._probes, self._limits = (
            deepcopy(probes),
            CoupledPruneLimits(
                limits.max_prediction_change,
                limits.max_bce_increase,
                limits.max_inferred_energy_increase,
            ),
        )
        self._budget, self._steps = budget, inference_steps

    def __call__(self, state: CoupledTopologyState) -> ConsolidatedState[CoupledTopologyState]:
        model = restore_coupled_topology(state)
        history = deepcopy(state.prune_lineage)
        if len(history) >= 64:
            raise ValueError("coupled topology pruning lineage capacity exhausted")
        decision = gate_coupled_prune(
            model,
            self._layer,
            self._unit,
            self._probes,
            limits=self._limits,
            budget=self._budget,
            inference_steps=self._steps,
            inference_learning_rate=self._rate,
        )
        if decision.status != "accepted":
            raise CoupledPruneRefused(decision)
        if decision.lineage is None:
            raise ValueError("accepted prune is missing coordinate lineage")
        prepared = capture_coupled_topology(
            decision.model, prune_lineage=(*history, decision.lineage)
        )
        # Why this diagnostic: retain one named native observation in the existing
        # receipt port; it is neither a shared training loss nor promotion utility.
        return ConsolidatedState(
            prepared,
            TrainingDiagnostic(
                "coupled_prune_max_probability_change_v1",
                max(item.max_prediction_change for item in decision.diagnostics),
            ),
        )
