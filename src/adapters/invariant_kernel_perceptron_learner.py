"""Bind the fixed kernel_perceptron to raw MiniGrid initial/current public histories.

Projects all rows before native work. Complete owned snapshots bind the original
invariant encoding; there is no role authority, IO, tuning or stream ownership.
"""

from dataclasses import dataclass, fields

import numpy as np

from src.adapters.kernel_perceptron_learner import KernelPerceptronLearner
from src.core.categorical_kernel_perceptron import (
    KernelPerceptronPolicy,
    KernelPerceptronSnapshot,
    copy_kernel_perceptron_policy,
    copy_kernel_perceptron_snapshot,
)
from src.core.coupled_energy import Array
from src.core.learner_ports import TrainingDiagnostic
from src.core.minigrid_invariant_encoding import (
    INVARIANT_HISTORY_ENCODING_ID,
    canonicalize_public_history,
)


@dataclass(frozen=True)
class InvariantKernelPerceptronSnapshot:
    policy: KernelPerceptronPolicy
    native_state: KernelPerceptronSnapshot
    encoding_id: str = INVARIANT_HISTORY_ENCODING_ID

    @property
    def completed_updates(self) -> int:
        return self.native_state.completed_updates

    @property
    def training_examples(self) -> int:
        return self.native_state.training_examples


def copy_invariant_kernel_perceptron_snapshot(
    state: InvariantKernelPerceptronSnapshot,
) -> InvariantKernelPerceptronSnapshot:
    if (
        type(state) is not InvariantKernelPerceptronSnapshot
        or set(vars(state)) != {field.name for field in fields(InvariantKernelPerceptronSnapshot)}
        or type(state.encoding_id) is not str
        or state.encoding_id != INVARIANT_HISTORY_ENCODING_ID
    ):
        raise ValueError("an exact original invariant kernel_perceptron snapshot is required")
    policy = copy_kernel_perceptron_policy(state.policy)
    native = copy_kernel_perceptron_snapshot(state.native_state)
    if policy != KernelPerceptronPolicy(296, 7) or native.policy != policy:
        raise ValueError("invariant kernel_perceptron requires its original296/K7 policy")
    return InvariantKernelPerceptronSnapshot(policy, native)


class InvariantKernelPerceptronLearner:
    """Project raw296 rows, then delegate the unchanged deterministic rule."""

    def __init__(self) -> None:
        self._native = KernelPerceptronLearner(KernelPerceptronPolicy(296, 7))

    @property
    def policy(self) -> KernelPerceptronPolicy:
        return self._native.policy

    @property
    def completed_updates(self) -> int:
        return self.snapshot_state().completed_updates

    @property
    def training_examples(self) -> int:
        return self.snapshot_state().training_examples

    def snapshot_state(self) -> InvariantKernelPerceptronSnapshot:
        return copy_invariant_kernel_perceptron_snapshot(
            InvariantKernelPerceptronSnapshot(self.policy, self._native.snapshot_state())
        )

    def restore_state(self, state: InvariantKernelPerceptronSnapshot) -> None:
        owned = copy_invariant_kernel_perceptron_snapshot(state)
        self._native.restore_state(owned.native_state)

    def fork(self) -> "InvariantKernelPerceptronLearner":
        result = InvariantKernelPerceptronLearner()
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
            raise ValueError(
                "invariant kernel_perceptron requires finite float64(B1..128,296) histories"
            )
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
