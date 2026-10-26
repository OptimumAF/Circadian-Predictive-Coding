"""Expose bounded exact action-count memory through the existing native port.

Inputs are arrays or complete owned snapshots. Outputs are count diagnostics,
probabilities and detached snapshots. Data-role release, actors, budgets, live
sessions, IO and scientific comparison protocols remain with their callers.
"""

from src.core.action_memory import (
    ACTION_MEMORY_COUNT_DIAGNOSTIC,
    ActionMemoryPolicy,
    ActionMemorySnapshot,
    copy_action_memory_policy,
    copy_action_memory_snapshot,
    count_action_observations,
    predict_action_memory,
)
from src.core.coupled_energy import Array
from src.core.learner_ports import TrainingDiagnostic


class ActionMemoryLearner:
    """Own a bounded immutable count table with atomic whole-batch commits."""

    def __init__(self, policy: ActionMemoryPolicy = ActionMemoryPolicy()) -> None:
        self._policy = copy_action_memory_policy(policy)
        self._state = ActionMemorySnapshot(copy_action_memory_policy(self._policy), (), 0, 0)

    @property
    def policy(self) -> ActionMemoryPolicy:
        return copy_action_memory_policy(self._policy)

    @property
    def completed_updates(self) -> int:
        return self.snapshot_state().completed_updates

    @property
    def training_examples(self) -> int:
        return self.snapshot_state().training_examples

    def snapshot_state(self) -> ActionMemorySnapshot:
        owned = copy_action_memory_snapshot(self._state)
        if owned.policy != self.policy:
            raise ValueError("memory state policy differs from its bound learner policy")
        return owned

    def restore_state(self, snapshot: ActionMemorySnapshot) -> None:
        owned = copy_action_memory_snapshot(snapshot)
        if owned.policy != self.policy:
            raise ValueError("memory snapshot policy differs from its bound learner policy")
        self._state = owned

    def fork(self) -> "ActionMemoryLearner":
        child = ActionMemoryLearner(self.policy)
        child.restore_state(self.snapshot_state())
        return child

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        before = self.snapshot_state()
        after = count_action_observations(before, features, targets)
        # Why: this learner counts exposures rather than applying gradients.
        # Report committed row increments without inventing a common loss.
        diagnostic = TrainingDiagnostic(
            ACTION_MEMORY_COUNT_DIAGNOSTIC,
            float(after.training_examples - before.training_examples),
        )
        self._state = after
        return diagnostic

    def predict(self, features: Array) -> Array:
        return predict_action_memory(self.snapshot_state(), features)
