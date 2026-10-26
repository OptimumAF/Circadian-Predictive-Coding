"""Bind the fixed perceptron to raw MiniGrid initial/current public histories.

Projects all rows before native work. Complete owned snapshots bind the original
invariant encoding; there is no role authority, IO, tuning or stream ownership.
"""

from dataclasses import dataclass, fields

import numpy as np

from src.adapters.perceptron_learner import PerceptronLearner
from src.core.categorical_perceptron import (
    PerceptronPolicy,
    PerceptronSnapshot,
    copy_perceptron_policy,
    copy_perceptron_snapshot,
)
from src.core.coupled_energy import Array
from src.core.learner_ports import TrainingDiagnostic
from src.core.minigrid_invariant_encoding import (
    INVARIANT_HISTORY_ENCODING_ID,
    canonicalize_public_history,
)


@dataclass(frozen=True)
class InvariantPerceptronSnapshot:
    policy: PerceptronPolicy
    native_state: PerceptronSnapshot
    encoding_id: str = INVARIANT_HISTORY_ENCODING_ID

    @property
    def completed_updates(self) -> int:
        return self.native_state.completed_updates

    @property
    def training_examples(self) -> int:
        return self.native_state.training_examples


def copy_invariant_perceptron_snapshot(
    state: InvariantPerceptronSnapshot,
) -> InvariantPerceptronSnapshot:
    if (
        type(state) is not InvariantPerceptronSnapshot
        or set(vars(state)) != {field.name for field in fields(InvariantPerceptronSnapshot)}
        or type(state.encoding_id) is not str
        or state.encoding_id != INVARIANT_HISTORY_ENCODING_ID
    ):
        raise ValueError("an exact original invariant perceptron snapshot is required")
    policy = copy_perceptron_policy(state.policy)
    native = copy_perceptron_snapshot(state.native_state)
    if policy != PerceptronPolicy(296, 7) or native.policy != policy:
        raise ValueError("invariant perceptron requires its original296/K7 policy")
    return InvariantPerceptronSnapshot(policy, native)


class InvariantPerceptronLearner:
    """Project raw296 rows, then delegate the unchanged deterministic rule."""

    def __init__(self) -> None:
        self._native = PerceptronLearner(PerceptronPolicy(296, 7))

    @property
    def policy(self) -> PerceptronPolicy:
        return self._native.policy

    @property
    def completed_updates(self) -> int:
        return self.snapshot_state().completed_updates

    @property
    def training_examples(self) -> int:
        return self.snapshot_state().training_examples

    def snapshot_state(self) -> InvariantPerceptronSnapshot:
        return copy_invariant_perceptron_snapshot(
            InvariantPerceptronSnapshot(self.policy, self._native.snapshot_state())
        )

    def restore_state(self, state: InvariantPerceptronSnapshot) -> None:
        owned = copy_invariant_perceptron_snapshot(state)
        self._native.restore_state(owned.native_state)

    def fork(self) -> "InvariantPerceptronLearner":
        result = InvariantPerceptronLearner()
        result.restore_state(self.snapshot_state())
        return result

    def _features(self, features: Array) -> Array:
        if (
            type(features) is not np.ndarray
            or features.dtype != np.dtype(np.float64)
            or features.ndim != 2
            or not 1 <= features.shape[0] <= 128
            or features.shape[1] != 296
            or not np.isfinite(features).all()
        ):
            raise ValueError("invariant perceptron requires finite float64(B1..128,296) histories")
        # Every supplied row must project before any target can update the owner.
        return np.array(
            [canonicalize_public_history(tuple(float(v) for v in row)) for row in features],
            dtype=np.float64,
        )

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        self.snapshot_state()
        return self._native.train_batch(self._features(features), targets)

    def predict(self, features: Array) -> Array:
        self.snapshot_state()
        return self._native.predict(self._features(features))
