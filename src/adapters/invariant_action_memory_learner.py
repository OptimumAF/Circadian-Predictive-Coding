"""Bind a public representation control to the existing count learner ports.

Inputs are raw296 first/current public frames, labels, or explicitly bound state.
Outputs retain native count diagnostics and independent complete snapshots.
Role release, budgets, episode collection, persistence and scientific protocols
remain with callers. No raw saved count state is silently reinterpreted here.
"""

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from src.adapters.action_memory_learner import ActionMemoryLearner
from src.adapters.fifo_action_memory_learner import FifoActionMemoryLearner
from src.core.action_memory import (
    ActionMemoryPolicy,
    ActionMemorySnapshot,
    _feature_keys,
    copy_action_memory_policy,
    copy_action_memory_snapshot,
)
from src.core.coupled_energy import Array
from src.core.fifo_action_memory import (
    FifoActionMemoryPolicy,
    FifoActionMemorySnapshot,
    copy_fifo_action_memory_policy,
    copy_fifo_action_memory_snapshot,
)
from src.core.learner_ports import TrainingDiagnostic
from src.core.minigrid_invariant_encoding import (
    INVARIANT_HISTORY_ENCODING_ID,
    canonicalize_public_history,
)

MemoryPolicy = ActionMemoryPolicy | FifoActionMemoryPolicy
MemorySnapshot = ActionMemorySnapshot | FifoActionMemorySnapshot


def _copy_native_policy(policy: MemoryPolicy) -> MemoryPolicy:
    if type(policy) is ActionMemoryPolicy:
        return copy_action_memory_policy(policy)
    if type(policy) is FifoActionMemoryPolicy:
        return copy_fifo_action_memory_policy(policy)
    raise ValueError("invariant memory requires an exact refusal or FIFO native policy")


@dataclass(frozen=True)
class InvariantMemoryPolicy:
    native_policy: MemoryPolicy = ActionMemoryPolicy(
        feature_dim=296, num_classes=7, max_payload_bytes=155136
    )
    encoding_id: str = INVARIANT_HISTORY_ENCODING_ID

    def __post_init__(self) -> None:
        if type(self.encoding_id) is not str or self.encoding_id != INVARIANT_HISTORY_ENCODING_ID:
            raise ValueError("invariant memory requires its explicit fixed encoding identity")
        owned = _copy_native_policy(self.native_policy)
        if owned.feature_dim != 296 or owned.num_classes != 7:
            raise ValueError("invariant MiniGrid memory requires296 public features/seven classes")
        object.__setattr__(self, "native_policy", owned)

    @property
    def method(self) -> str:
        return f"public_invariant_{self.native_policy.method}"

    @property
    def feature_dim(self) -> int:
        return self.native_policy.feature_dim

    @property
    def num_classes(self) -> int:
        return self.native_policy.num_classes

    @property
    def max_entries(self) -> int:
        return self.native_policy.max_entries

    @property
    def max_payload_bytes(self) -> int:
        return self.native_policy.max_payload_bytes


def _record(value: Any, kind: type[Any]) -> None:
    if type(value) is not kind or set(vars(value)) != {field.name for field in fields(kind)}:
        raise ValueError(f"invariant memory requires an exact complete {kind.__name__}")


def copy_invariant_memory_policy(policy: InvariantMemoryPolicy) -> InvariantMemoryPolicy:
    _record(policy, InvariantMemoryPolicy)
    return InvariantMemoryPolicy(policy.native_policy, policy.encoding_id)


@dataclass(frozen=True)
class InvariantMemorySnapshot:
    policy: InvariantMemoryPolicy
    native_state: MemorySnapshot

    @property
    def completed_updates(self) -> int:
        return self.native_state.completed_updates

    @property
    def training_examples(self) -> int:
        return self.native_state.training_examples


def copy_invariant_memory_snapshot(snapshot: InvariantMemorySnapshot) -> InvariantMemorySnapshot:
    _record(snapshot, InvariantMemorySnapshot)
    policy = copy_invariant_memory_policy(snapshot.policy)
    native: MemorySnapshot
    if type(policy.native_policy) is ActionMemoryPolicy:
        if type(snapshot.native_state) is not ActionMemorySnapshot:
            raise ValueError("invariant refusal policy requires its exact native snapshot")
        native = copy_action_memory_snapshot(snapshot.native_state)
    else:
        if type(snapshot.native_state) is not FifoActionMemorySnapshot:
            raise ValueError("invariant FIFO policy requires its exact native snapshot")
        native = copy_fifo_action_memory_snapshot(snapshot.native_state)
    if native.policy != policy.native_policy:
        raise ValueError("invariant snapshot native policy differs from its bound control")
    # Why: a tag alone must not relabel stored raw observation keys. The full
    # canonical key contract is checked before admitting any restore or use.
    if any(
        canonicalize_public_history(entry.features) != entry.features for entry in native.entries
    ):
        raise ValueError("invariant snapshot contains noncanonical public-history keys")
    return InvariantMemorySnapshot(policy, native)


class InvariantActionMemoryLearner:
    """Project public inputs before existing atomic count updates/predictions."""

    def __init__(self, policy: InvariantMemoryPolicy = InvariantMemoryPolicy()) -> None:
        self._policy = copy_invariant_memory_policy(policy)
        self._native: ActionMemoryLearner | FifoActionMemoryLearner
        native_policy = self._policy.native_policy
        if type(native_policy) is ActionMemoryPolicy:
            self._native = ActionMemoryLearner(native_policy)
        elif type(native_policy) is FifoActionMemoryPolicy:
            self._native = FifoActionMemoryLearner(native_policy)
        else:
            raise ValueError("invariant memory requires its declared native backend")

    @property
    def policy(self) -> InvariantMemoryPolicy:
        return copy_invariant_memory_policy(self._policy)

    @property
    def completed_updates(self) -> int:
        return self.snapshot_state().completed_updates

    @property
    def training_examples(self) -> int:
        return self.snapshot_state().training_examples

    def snapshot_state(self) -> InvariantMemorySnapshot:
        return copy_invariant_memory_snapshot(
            InvariantMemorySnapshot(self.policy, self._native.snapshot_state())
        )

    def restore_state(self, snapshot: InvariantMemorySnapshot) -> None:
        owned = copy_invariant_memory_snapshot(snapshot)
        if owned.policy != self.policy:
            raise ValueError("invariant snapshot encoding/control/cap differs from the learner")
        if (
            type(self._native) is ActionMemoryLearner
            and type(owned.native_state) is ActionMemorySnapshot
        ):
            self._native.restore_state(owned.native_state)
        elif (
            type(self._native) is FifoActionMemoryLearner
            and type(owned.native_state) is FifoActionMemorySnapshot
        ):
            self._native.restore_state(owned.native_state)
        else:
            raise ValueError("invariant native backend differs from its bound snapshot type")

    def fork(self) -> "InvariantActionMemoryLearner":
        child = InvariantActionMemoryLearner(self.policy)
        child.restore_state(self.snapshot_state())
        return child

    def _features(self, features: Array) -> Array:
        # Project every row before native mutation, retaining atomic refusal if
        # a later row is malformed. Neither labels nor metadata enter this step.
        rows = _feature_keys(features, 296)
        return np.array([canonicalize_public_history(row) for row in rows], dtype=np.float64)

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        self.snapshot_state()
        return self._native.train_batch(self._features(features), targets)

    def predict(self, features: Array) -> Array:
        self.snapshot_state()
        return self._native.predict(self._features(features))
