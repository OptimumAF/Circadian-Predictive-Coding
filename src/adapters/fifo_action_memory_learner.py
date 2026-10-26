"""Owned FIFO count memory through the existing array-native learner port.

Inputs are bounded public arrays or exact FIFO snapshots. Outputs retain native
count diagnostics, detached predictions and complete owned FIFO state. Role
release, replay, budgets, IO and serving authority remain outside this adapter.
"""

from src.core.coupled_energy import Array
from src.core.fifo_action_memory import (
    FIFO_ACTION_MEMORY_COUNT_DIAGNOSTIC,
    FifoActionMemoryPolicy,
    FifoActionMemorySnapshot,
    copy_fifo_action_memory_policy,
    copy_fifo_action_memory_snapshot,
    count_fifo_action_observations,
    predict_fifo_action_memory,
)
from src.core.learner_ports import TrainingDiagnostic


class FifoActionMemoryLearner:
    """Commit one complete FIFO count update or preserve the preceding state."""

    def __init__(self, policy: FifoActionMemoryPolicy = FifoActionMemoryPolicy()) -> None:
        self._policy = copy_fifo_action_memory_policy(policy)
        self._state = FifoActionMemorySnapshot(
            copy_fifo_action_memory_policy(self._policy), (), 0, 0, 0, 0
        )

    @property
    def policy(self) -> FifoActionMemoryPolicy:
        return copy_fifo_action_memory_policy(self._policy)

    @property
    def completed_updates(self) -> int:
        return self.snapshot_state().completed_updates

    @property
    def training_examples(self) -> int:
        return self.snapshot_state().training_examples

    def snapshot_state(self) -> FifoActionMemorySnapshot:
        owned = copy_fifo_action_memory_snapshot(self._state)
        if owned.policy != self.policy:
            raise ValueError("FIFO state policy differs from its bound learner policy")
        return owned

    def restore_state(self, snapshot: FifoActionMemorySnapshot) -> None:
        owned = copy_fifo_action_memory_snapshot(snapshot)
        if owned.policy != self.policy:
            raise ValueError("FIFO snapshot policy differs from its bound learner policy")
        self._state = owned

    def fork(self) -> "FifoActionMemoryLearner":
        child = FifoActionMemoryLearner(self.policy)
        child.restore_state(self.snapshot_state())
        return child

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        before = self.snapshot_state()
        after = count_fifo_action_observations(before, features, targets)
        # Counts and evictions are storage operations, never surrogate SGD work.
        diagnostic = TrainingDiagnostic(
            FIFO_ACTION_MEMORY_COUNT_DIAGNOSTIC,
            float(after.training_examples - before.training_examples),
        )
        self._state = after
        return diagnostic

    def predict(self, features: Array) -> Array:
        return predict_fifo_action_memory(self.snapshot_state(), features)
