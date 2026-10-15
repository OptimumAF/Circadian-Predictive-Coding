"""Opt-in coupled learner ports with validated hidden-width reconstruction.

Training policy and family/input/precision/CPC-policy bindings belong to the
adapter. Native topology snapshots replace neither those settings nor runtime
budgets or promotion authority. Exact precision binding also fixes depth. Only
this adapter opts into reconstructing changed hidden widths; the existing strict
CoupledLearner is unchanged. Single-owner replacement publishes one owned payload.
"""

from dataclasses import dataclass, replace

from src.adapters.coupled_learners import CoupledLearner, _rate
from src.core.coupled_energy import Array
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.coupled_topology import (
    CoupledTopologyState,
    capture_coupled_topology,
    restore_coupled_topology,
)
from src.core.coupled_width_pruning import CoupledPruneLineage
from src.core.learner_ports import TrainingDiagnostic


def _copy_policy(policy: CoupledInferencePolicy | None, cap: int) -> CoupledInferencePolicy | None:
    if policy is None:
        return None
    if type(policy) is not CoupledInferencePolicy:
        raise ValueError("inference_policy must be an exact CoupledInferencePolicy")
    copied = CoupledInferencePolicy(
        policy.min_steps, policy.gradient_tolerance, policy.relative_energy_tolerance
    )
    if copied.min_steps > cap:
        raise ValueError("inference policy minimum exceeds the step cap")
    return copied


@dataclass(frozen=True)
class _Binding:
    family: str
    input_dim: int
    precisions: tuple[float, ...]
    chemistry: tuple[float, ...] | None


def _binding(state: CoupledTopologyState) -> _Binding:
    wake = state.snapshot.wake_state
    chemistry = (
        None
        if wake is None
        else (
            float(wake.chemical_decay),
            float(wake.chemical_buildup_rate),
            float(wake.plasticity_sensitivity),
            float(wake.min_plasticity),
        )
    )
    return _Binding(
        state.family,
        state.input_dim,
        tuple(value for value in state.snapshot.precisions),
        chemistry,
    )


@dataclass(frozen=True)
class _Payload:
    learner: CoupledLearner
    hidden_dims: tuple[int, ...]
    prune_lineage: tuple[CoupledPruneLineage, ...]


class TopologyCoupledLearner:
    """Compose owned native wake ports with explicit geometry-aware snapshots."""

    def __init__(
        self,
        model: CoupledPredictiveCodingNetwork,
        *,
        learning_rate: float,
        inference_steps: int,
        inference_learning_rate: float,
        inference_policy: CoupledInferencePolicy | None = None,
    ) -> None:
        self._learning_rate = _rate(learning_rate, "learning_rate")
        self._inference_learning_rate = _rate(inference_learning_rate, "inference_learning_rate")
        if type(inference_steps) is not int or not 1 <= inference_steps <= 10000:
            raise ValueError("inference_steps must be a native integer in 1..10000")
        self._inference_steps = inference_steps
        self._inference_policy = _copy_policy(inference_policy, inference_steps)
        state = capture_coupled_topology(model)
        self._binding = _binding(state)
        self._payload = self._prepare_payload(state)

    def _prepare_payload(self, state: CoupledTopologyState) -> _Payload:
        model = restore_coupled_topology(state)
        canonical = capture_coupled_topology(model, prune_lineage=state.prune_lineage)
        if _binding(canonical) != self._binding:
            raise ValueError(
                "topology differs from the adapter family/input/precision/chemistry binding"
            )
        learner = CoupledLearner(
            model,
            learning_rate=self._learning_rate,
            inference_steps=self._inference_steps,
            inference_learning_rate=self._inference_learning_rate,
            inference_policy=self._inference_policy,
        )
        return _Payload(learner, canonical.hidden_dims, canonical.prune_lineage)

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        return self._payload.learner.train_batch(features, targets)

    def predict(self, features: Array) -> Array:
        return self._payload.learner.predict(features)

    def snapshot_state(self) -> CoupledTopologyState:
        payload = self._payload
        return CoupledTopologyState(
            1,
            self._binding.family,
            self._binding.input_dim,
            payload.hidden_dims,
            payload.learner.snapshot_state(),
            payload.prune_lineage,
        )

    def restore_state(self, state: CoupledTopologyState) -> None:
        """Validate complete replacement before swapping model and lineage together."""
        prepared = self._prepare_payload(state)
        self._payload = prepared

    def build_from_state(self, state: CoupledTopologyState) -> "TopologyCoupledLearner":
        """Build an owned same-policy learner under this prototype's fixed bindings."""
        prepared = self._prepare_payload(state)
        # The complete payload is already validated. Copy the prototype's fixed
        # settings directly so structural state cannot redefine adapter policy.
        built = object.__new__(TopologyCoupledLearner)
        built._learning_rate = self._learning_rate
        built._inference_steps = self._inference_steps
        built._inference_learning_rate = self._inference_learning_rate
        built._inference_policy = _copy_policy(self._inference_policy, self._inference_steps)
        built._binding = replace(self._binding)
        built._payload = prepared
        return built

    def fork(self) -> "TopologyCoupledLearner":
        return self.build_from_state(self.snapshot_state())
