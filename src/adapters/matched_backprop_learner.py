"""Owned native ports for the opt-in matched feedforward BP model.

Inputs are a trusted quiescent native model and fixed rate. Predictions, native
diagnostics and model snapshots are detached. No data role, clock, runtime budget,
durable decoding or promotion authority is supplied by a model restore.
"""

from src.core.coupled_energy import Array
from src.core.learner_ports import TrainingDiagnostic
from src.core.matched_backprop import (
    MatchedBackpropNetwork,
    MatchedBackpropSnapshot,
    _learning_rate,
)


class MatchedBackpropLearner:
    """Preserve complete native model state and fixed training policy on forks."""

    def __init__(self, model: MatchedBackpropNetwork, *, learning_rate: float) -> None:
        self._learning_rate = _learning_rate(learning_rate)
        if type(model) is not MatchedBackpropNetwork:
            raise ValueError("matched BP learner requires an exact native matched BP model")
        # This model's complete mutable state is its parameter bundle; no RNG,
        # latent optimizer or adaptive traces continue after construction.
        self._model = MatchedBackpropNetwork(model.snapshot_parameters())

    def fork(self) -> "MatchedBackpropLearner":
        return MatchedBackpropLearner(self._model, learning_rate=self._learning_rate)

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        result = self._model.train_epoch(features, targets, self._learning_rate)
        return TrainingDiagnostic(result.loss_definition, result.loss)

    def predict(self, features: Array) -> Array:
        return self._model.predict_proba(features)

    def snapshot_state(self) -> MatchedBackpropSnapshot:
        return self._model.snapshot_state()

    def restore_state(self, snapshot: MatchedBackpropSnapshot) -> None:
        self._model.restore_state(snapshot)
