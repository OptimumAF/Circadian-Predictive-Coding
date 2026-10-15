"""Owned raw-coordinate BP ports matched to scaled PC configuration and rates.

Raw snapshots retain exact scaling/precision metadata. Predictions have no labels;
updates use feedforward BCE without latent inference or chemical adaptation.
No data-role authorization, runtime budget, topology change or durable codec.
"""

from src.core.coupled_energy import Array, predict_coupled
from src.core.coupled_pc_scaling import (
    CoupledScalingState,
    copy_coupled_scaling_state,
    effective_coupled_parameters,
)
from src.core.learner_ports import TrainingDiagnostic
from src.core.matched_backprop import _learning_rate
from src.core.scaled_matched_backprop import step_scaled_matched_backprop


def _copy_pc_state(state: CoupledScalingState) -> CoupledScalingState:
    copied = copy_coupled_scaling_state(state)
    if copied.config.circadian is not None or copied.wake_state is not None:
        raise ValueError("scaled BP requires PC raw state without circadian policy or wake state")
    return copied


class ScaledMatchedBackpropLearner:
    """Preserve a fixed base rate and exact raw configuration on owned forks."""

    def __init__(self, state: CoupledScalingState, *, learning_rate: float) -> None:
        self._learning_rate = _learning_rate(learning_rate)
        self._state = _copy_pc_state(state)

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        result = step_scaled_matched_backprop(self._state, features, targets, self._learning_rate)
        self._state = result.state
        return TrainingDiagnostic(result.evaluation.loss_definition, result.evaluation.loss)

    def predict(self, features: Array) -> Array:
        return predict_coupled(effective_coupled_parameters(self._state), features)

    def snapshot_state(self) -> CoupledScalingState:
        return _copy_pc_state(self._state)

    def restore_state(self, state: CoupledScalingState) -> None:
        candidate = _copy_pc_state(state)
        if candidate.config != self._state.config:
            raise ValueError("raw BP snapshot configuration differs from learner")
        self._state = candidate

    def fork(self) -> "ScaledMatchedBackpropLearner":
        return ScaledMatchedBackpropLearner(self._state, learning_rate=self._learning_rate)
