"""Opt-in raw coupled learner ports permitting validated uniform-width changes.

The strict raw learner remains responsible for native training and prediction.
This adapter pins every scaling config field except width, plus its independent
training policy. A raw topology envelope replaces neither those bindings nor
runtime budgets or promotion authority. Complete private validation precedes one
owned learner/history payload swap under synchronous single-owner use.
"""

from dataclasses import dataclass, replace

from src.adapters.coupled_learners import _rate
from src.adapters.coupled_scaling_learner import CoupledScalingLearner, _copy_policy
from src.core.coupled_energy import Array
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_pc_scaling import CoupledScalingConfig, CoupledScalingState
from src.core.coupled_scaling_pruning import CoupledScalingPruneLineage
from src.core.coupled_scaling_topology import (
    CoupledScalingTopologyState,
    copy_coupled_scaling_topology,
)
from src.core.learner_ports import TrainingDiagnostic


@dataclass(frozen=True)
class _Binding:
    scheme: str
    input_dim: int
    hidden_layers: int
    gamma0: float
    precisions: tuple[float, ...]
    chemistry: tuple[float, ...] | None


def _binding(config: CoupledScalingConfig) -> _Binding:
    policy = config.circadian
    chemistry = (
        None
        if policy is None
        else (
            float(policy.chemical_decay),
            float(policy.chemical_buildup_rate),
            float(policy.plasticity_sensitivity),
            float(policy.min_plasticity),
        )
    )
    # Bind actual validated fields. An artificial width-one config could reject
    # gamma values that are valid at the source width and must never be used.
    assert config.precisions is not None
    return _Binding(
        config.scheme,
        config.input_dim,
        config.hidden_layers,
        config.gamma0,
        tuple(value for value in config.precisions),
        chemistry,
    )


@dataclass(frozen=True)
class _Payload:
    learner: CoupledScalingLearner
    prune_lineage: tuple[CoupledScalingPruneLineage, ...]


class TopologyCoupledScalingLearner:
    """Compose owned strict raw ports with bounded topology-aware restore/build."""

    def __init__(
        self,
        initial: CoupledScalingState,
        *,
        learning_rate: float,
        inference_steps: int,
        inference_learning_rate: float,
        inference_policy: CoupledInferencePolicy | None = None,
    ) -> None:
        self._learning_rate = _rate(learning_rate, "learning_rate")
        self._inference_rate = _rate(inference_learning_rate, "inference_learning_rate")
        if type(inference_steps) is not int or not 1 <= inference_steps <= 10000:
            raise ValueError("inference_steps must be a native integer in 1..10000")
        self._steps = inference_steps
        self._policy = _copy_policy(inference_policy, inference_steps)
        state = CoupledScalingTopologyState(1, initial)
        self._binding = _binding(state.raw_state.config)
        self._payload = self._prepare_payload(state)

    def _prepare_payload(self, state: CoupledScalingTopologyState) -> _Payload:
        canonical = copy_coupled_scaling_topology(state)
        if _binding(canonical.raw_state.config) != self._binding:
            raise ValueError(
                "raw topology differs from the adapter's fixed non-width configuration"
            )
        learner = CoupledScalingLearner(
            canonical.raw_state,
            learning_rate=self._learning_rate,
            inference_steps=self._steps,
            inference_learning_rate=self._inference_rate,
            inference_policy=self._policy,
        )
        return _Payload(learner, canonical.prune_lineage)

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        return self._payload.learner.train_batch(features, targets)

    def predict(self, features: Array) -> Array:
        return self._payload.learner.predict(features)

    def snapshot_state(self) -> CoupledScalingTopologyState:
        payload = self._payload
        return CoupledScalingTopologyState(
            1, payload.learner.snapshot_state(), payload.prune_lineage
        )

    def restore_state(self, state: CoupledScalingTopologyState) -> None:
        """Publish the fully validated replacement learner and history together."""
        prepared = self._prepare_payload(state)
        self._payload = prepared

    def build_from_state(
        self, state: CoupledScalingTopologyState
    ) -> "TopologyCoupledScalingLearner":
        """Build a detached learner using this prototype's settings and bindings."""
        prepared = self._prepare_payload(state)
        built = object.__new__(TopologyCoupledScalingLearner)
        built._learning_rate = self._learning_rate
        built._inference_rate = self._inference_rate
        built._steps = self._steps
        built._policy = _copy_policy(self._policy, self._steps)
        built._binding = replace(self._binding)
        built._payload = prepared
        return built

    def fork(self) -> "TopologyCoupledScalingLearner":
        return self.build_from_state(self.snapshot_state())
