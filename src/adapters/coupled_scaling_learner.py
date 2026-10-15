"""Opt-in owned learner ports for raw-coordinate coupled PC/CPC updates.

Inputs: bounded raw state and fixed training settings. Outputs: native energy
diagnostics, target-free probabilities and complete raw-coordinate snapshots.
No scaling-study selection, effective-only restore, topology change or IO.
"""

from src.adapters.coupled_learners import _rate
from src.core.coupled_energy import Array, predict_coupled
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_pc_scaling import (
    CoupledScalingState,
    copy_coupled_scaling_state,
    effective_coupled_parameters,
    step_coupled_scaling,
)
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


class CoupledScalingLearner:
    """Keep raw scaling config/state and fixed update policy across owned forks."""

    def __init__(
        self,
        state: CoupledScalingState,
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
        self._state = copy_coupled_scaling_state(state)

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        result = step_coupled_scaling(
            self._state,
            features,
            targets,
            self._learning_rate,
            self._steps,
            self._inference_rate,
            inference_policy=self._policy,
        )
        self._state = result.state
        return TrainingDiagnostic(result.evaluation.energy_definition, result.evaluation.energy)

    def predict(self, features: Array) -> Array:
        return predict_coupled(effective_coupled_parameters(self._state), features)

    def snapshot_state(self) -> CoupledScalingState:
        return copy_coupled_scaling_state(self._state)

    def restore_state(self, state: CoupledScalingState) -> None:
        candidate = copy_coupled_scaling_state(state)
        if candidate.config != self._state.config:
            raise ValueError("raw snapshot scaling/precision/chemistry config differs from learner")
        self._state = candidate

    def fork(self) -> "CoupledScalingLearner":
        return CoupledScalingLearner(
            self._state,
            learning_rate=self._learning_rate,
            inference_steps=self._steps,
            inference_learning_rate=self._inference_rate,
            inference_policy=self._policy,
        )
