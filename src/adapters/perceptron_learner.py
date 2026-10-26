"""Expose the fixed multiclass mistake rule through the NativeLearner port.

The adapter owns complete bounded snapshots and commits a whole valid pass.
It returns native mistake diagnostics and scoring probabilities. Projection,
data roles, scheduling, budgets, and scientific comparison protocols stay outside.
"""

from src.core.categorical_perceptron import (
    PERCEPTRON_MISTAKES_DIAGNOSTIC,
    PerceptronPolicy,
    PerceptronSnapshot,
    copy_perceptron_policy,
    copy_perceptron_snapshot,
    create_perceptron_snapshot,
    predict_perceptron,
    update_perceptron,
)
from src.core.coupled_energy import Array
from src.core.learner_ports import TrainingDiagnostic


class PerceptronLearner:
    """Own immutable perceptron state with atomic whole-batch replacement."""

    def __init__(self, policy: PerceptronPolicy) -> None:
        self._policy = copy_perceptron_policy(policy)
        self._state = create_perceptron_snapshot(self._policy)

    @property
    def policy(self) -> PerceptronPolicy:
        return copy_perceptron_policy(self._policy)

    @property
    def completed_updates(self) -> int:
        return self.snapshot_state().completed_updates

    @property
    def training_examples(self) -> int:
        return self.snapshot_state().training_examples

    def snapshot_state(self) -> PerceptronSnapshot:
        owned = copy_perceptron_snapshot(self._state)
        if owned.policy != self.policy:
            raise ValueError("perceptron state policy differs from its bound learner policy")
        return owned

    def restore_state(self, snapshot: PerceptronSnapshot) -> None:
        owned = copy_perceptron_snapshot(snapshot)
        if owned.policy != self.policy:
            raise ValueError("perceptron snapshot policy differs from its bound learner policy")
        self._state = owned

    def fork(self) -> "PerceptronLearner":
        child = PerceptronLearner(self.policy)
        child.restore_state(self.snapshot_state())
        return child

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        before = self.snapshot_state()
        after = update_perceptron(before, features, targets)
        diagnostic = TrainingDiagnostic(
            PERCEPTRON_MISTAKES_DIAGNOSTIC,
            float(after.mistake_updates - before.mistake_updates),
        )
        self._state = after
        return diagnostic

    def predict(self, features: Array) -> Array:
        return predict_perceptron(self.snapshot_state(), features)
