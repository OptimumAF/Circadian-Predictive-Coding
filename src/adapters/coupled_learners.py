"""Owned native ports for coupled binary PC and CPC wake models.

Inputs: trusted quiescent native models, fixed update policy and float64 batches.
Outputs: native diagnostics, predictions and detached complete model snapshots.
No runtime cursor/budget restoration, data authorization, disk decoding or sleep.
"""

from math import isfinite

from src.core.coupled_circadian_predictive_coding import CoupledCircadianPredictiveCodingNetwork
from src.core.coupled_energy import Array
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_learner_state import CoupledUpdatePolicy, copy_update_policy
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork, CoupledSnapshot
from src.core.learner_ports import TrainingDiagnostic


def _rate(value: float, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a native positive finite number")
    try:
        resolved = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    if not isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return resolved


def _copy_model(model: CoupledPredictiveCodingNetwork) -> CoupledPredictiveCodingNetwork:
    # Why: reconstruction through the native state boundary validates adaptive
    # state too; copying only the parameter arrays would restart CPC chemistry.
    copied: CoupledPredictiveCodingNetwork
    if type(model) is CoupledCircadianPredictiveCodingNetwork:
        copied = CoupledCircadianPredictiveCodingNetwork(
            model.input_dim,
            model.hidden_dims,
            seed=0,
            precisions=model.precisions,
            config=model.config,
        )
    elif type(model) is CoupledPredictiveCodingNetwork:
        copied = CoupledPredictiveCodingNetwork(
            model.input_dim,
            model.hidden_dims,
            seed=0,
            precisions=model.precisions,
        )
    else:
        raise ValueError("coupled learner requires an exact native PC or CPC model")
    copied.restore_state(model.snapshot_state())
    return copied


class CoupledLearner:
    """Implement NativeLearner/ForkableLearner without changing the native update."""

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
        if inference_policy is not None:
            if type(inference_policy) is not CoupledInferencePolicy:
                raise ValueError("inference_policy must be an exact CoupledInferencePolicy")
            inference_policy = CoupledInferencePolicy(
                inference_policy.min_steps,
                inference_policy.gradient_tolerance,
                inference_policy.relative_energy_tolerance,
            )
            if inference_policy.min_steps > inference_steps:
                raise ValueError("inference policy minimum exceeds the step cap")
        # Why: adapter settings are owned independently of model snapshots and
        # caller policy records; forks must continue the same fixed policy.
        self._inference_policy = inference_policy
        self._model = _copy_model(model)

    def fork(self) -> "CoupledLearner":
        return CoupledLearner(
            self._model,
            learning_rate=self._learning_rate,
            inference_steps=self._inference_steps,
            inference_learning_rate=self._inference_learning_rate,
            inference_policy=self._inference_policy,
        )

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        if self._inference_policy is None:
            result = self._model.train_epoch(
                features,
                targets,
                self._learning_rate,
                self._inference_steps,
                self._inference_learning_rate,
            )
        else:
            result = self._model.train_epoch(
                features,
                targets,
                self._learning_rate,
                self._inference_steps,
                self._inference_learning_rate,
                inference_policy=self._inference_policy,
            )
        return TrainingDiagnostic(result.energy_definition, result.energy)

    def predict(self, features: Array) -> Array:
        return self._model.predict_proba(features)

    def snapshot_state(self) -> CoupledSnapshot:
        return self._model.snapshot_state()

    def checkpoint_policy(self) -> CoupledUpdatePolicy:
        """Expose detached fixed settings without invoking the native model."""
        return copy_update_policy(
            CoupledUpdatePolicy(
                self._learning_rate,
                self._inference_steps,
                self._inference_learning_rate,
                self._inference_policy,
            )
        )

    def restore_state(self, snapshot: CoupledSnapshot) -> None:
        self._model.restore_state(snapshot)
