"""Opt-in learner ports for a full rooted raw structural state.

The strict learner owns native learning/prediction; this wrapper owns origin and
tagged configuration metadata. Why pin the full origin: changing current width
or precision requires a complete supported chain, not relaxed restore bindings.
Training settings are independent of structural snapshots. Restore/build first
prepare a complete private learner/origin/history and publish one payload under
synchronous single-owner use. No transform, runtime budget or authority is added.
"""

from dataclasses import dataclass

from src.adapters.coupled_learners import _rate
from src.adapters.coupled_scaling_learner import CoupledScalingLearner, _copy_policy
from src.core.coupled_energy import Array
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_pc_scaling import CoupledScalingConfig, CoupledScalingState, _copy_config
from src.core.coupled_structural_state import (
    CoupledStructuralState,
    CoupledStructuralTransition,
    copy_coupled_structural_state,
)
from src.core.learner_ports import TrainingDiagnostic


@dataclass(frozen=True)
class _Payload:
    learner: CoupledScalingLearner
    origin_config: CoupledScalingConfig
    transitions: tuple[CoupledStructuralTransition, ...]


class StructuralCoupledScalingLearner:
    """Retain an owned full-origin structural chain across wakes and reconstruction."""

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
        if type(initial) is not CoupledScalingState:
            raise ValueError("initial must be an exact CoupledScalingState")
        state = CoupledStructuralState(1, initial.config, initial)
        self._origin_config = _copy_config(state.origin_config)
        self._payload = self._prepare_payload(state)

    def _prepare_payload(self, state: CoupledStructuralState) -> _Payload:
        canonical = copy_coupled_structural_state(state)
        if canonical.origin_config != self._origin_config:
            raise ValueError("structural state differs from the adapter's fixed full origin config")
        learner = CoupledScalingLearner(
            canonical.raw_state,
            learning_rate=self._learning_rate,
            inference_steps=self._steps,
            inference_learning_rate=self._inference_rate,
            inference_policy=self._policy,
        )
        return _Payload(learner, canonical.origin_config, canonical.transitions)

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        return self._payload.learner.train_batch(features, targets)

    def predict(self, features: Array) -> Array:
        return self._payload.learner.predict(features)

    def snapshot_state(self) -> CoupledStructuralState:
        payload = self._payload
        return CoupledStructuralState(
            1, payload.origin_config, payload.learner.snapshot_state(), payload.transitions
        )

    def restore_state(self, state: CoupledStructuralState) -> None:
        """Accept valid earlier histories too; publish only after complete validation."""
        prepared = self._prepare_payload(state)
        self._payload = prepared

    def build_from_state(self, state: CoupledStructuralState) -> "StructuralCoupledScalingLearner":
        """Build independent ports with the prototype's origin and fixed settings."""
        prepared = self._prepare_payload(state)
        built = object.__new__(StructuralCoupledScalingLearner)
        built._learning_rate = self._learning_rate
        built._inference_rate = self._inference_rate
        built._steps = self._steps
        built._policy = _copy_policy(self._policy, self._steps)
        built._origin_config = _copy_config(self._origin_config)
        built._payload = prepared
        return built

    def fork(self) -> "StructuralCoupledScalingLearner":
        return self.build_from_state(self.snapshot_state())
