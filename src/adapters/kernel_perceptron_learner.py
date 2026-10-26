"""Expose the bounded quadratic-kernel mistake rule through NativeLearner.

This adapter owns full immutable state, committing only complete valid passes.
It returns native mistake diagnostics and scoring probabilities. Projection,
data roles, scheduling, budgets, and scientific comparisons belong to callers.
"""

from src.core.categorical_kernel_perceptron import (
    KERNEL_PERCEPTRON_MISTAKES_DIAGNOSTIC,
    KernelPerceptronPolicy,
    KernelPerceptronSnapshot,
    copy_kernel_perceptron_policy,
    copy_kernel_perceptron_snapshot,
    create_kernel_perceptron_snapshot,
    predict_kernel_perceptron,
    update_kernel_perceptron,
)
from src.core.coupled_energy import Array
from src.core.learner_ports import TrainingDiagnostic


class KernelPerceptronLearner:
    """Own a bounded dictionary with atomic whole-pass state replacement."""

    def __init__(self, policy: KernelPerceptronPolicy) -> None:
        self._policy = copy_kernel_perceptron_policy(policy)
        self._state = create_kernel_perceptron_snapshot(self._policy)

    @property
    def policy(self) -> KernelPerceptronPolicy:
        return copy_kernel_perceptron_policy(self._policy)

    @property
    def completed_updates(self) -> int:
        return self.snapshot_state().completed_updates

    @property
    def training_examples(self) -> int:
        return self.snapshot_state().training_examples

    def snapshot_state(self) -> KernelPerceptronSnapshot:
        owned = copy_kernel_perceptron_snapshot(self._state)
        if owned.policy != self.policy:
            raise ValueError("kernel perceptron state differs from its bound learner policy")
        return owned

    def restore_state(self, snapshot: KernelPerceptronSnapshot) -> None:
        owned = copy_kernel_perceptron_snapshot(snapshot)
        if owned.policy != self.policy:
            raise ValueError("kernel perceptron snapshot differs from its bound learner policy")
        self._state = owned

    def fork(self) -> "KernelPerceptronLearner":
        child = KernelPerceptronLearner(self.policy)
        child.restore_state(self.snapshot_state())
        return child

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        before = self.snapshot_state()
        after = copy_kernel_perceptron_snapshot(update_kernel_perceptron(before, features, targets))
        diagnostic = TrainingDiagnostic(
            KERNEL_PERCEPTRON_MISTAKES_DIAGNOSTIC,
            float(after.mistake_updates - before.mistake_updates),
        )
        self._state = after
        return diagnostic

    def predict(self, features: Array) -> Array:
        return predict_kernel_perceptron(self.snapshot_state(), features)
