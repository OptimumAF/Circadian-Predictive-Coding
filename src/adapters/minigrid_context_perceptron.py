"""Bind raw public296 MiniGrid input to two existing owned perceptron heads.

Complete snapshots retain the explicit encoding and all head/cumulative clocks.
Roles, delayed admission, budgets, game episodes and IO remain caller concerns.
"""

from dataclasses import dataclass, fields

import numpy as np

from src.adapters.context_perceptron import ContextPerceptronLearner
from src.core.context_perceptron import (
    ContextPerceptronPolicy,
    ContextPerceptronSnapshot,
    copy_context_policy,
    copy_context_snapshot,
)
from src.core.minigrid_context_encoding import CONTEXT_HISTORY_ENCODING_ID, context_public_history


def context_policy():
    return ContextPerceptronPolicy(298, 7, (296, 297))


@dataclass(frozen=True)
class MiniGridContextSnapshot:
    policy: ContextPerceptronPolicy
    native_state: ContextPerceptronSnapshot
    encoding_id: str = CONTEXT_HISTORY_ENCODING_ID

    @property
    def completed_updates(self):
        return self.native_state.completed_updates

    @property
    def training_examples(self):
        return self.native_state.training_examples


def copy_snapshot(state: MiniGridContextSnapshot) -> MiniGridContextSnapshot:
    if (
        type(state) is not MiniGridContextSnapshot
        or set(vars(state)) != {f.name for f in fields(MiniGridContextSnapshot)}
        or type(state.encoding_id) is not str
        or state.encoding_id != CONTEXT_HISTORY_ENCODING_ID
    ):
        raise ValueError("exact encoding-bound MiniGrid context snapshot required")
    native = copy_context_snapshot(state.native_state)
    if native.policy != context_policy() or copy_context_policy(state.policy) != native.policy:
        raise ValueError("MiniGrid context policy requires298/K7 with final two cue columns")
    return MiniGridContextSnapshot(native.policy, native)


def project_features(features):
    if (
        type(features) is not np.ndarray
        or features.dtype != np.dtype(np.float64)
        or features.ndim != 2
        or features.shape[1] != 296
        or not 1 <= len(features) <= 128
    ):
        raise ValueError("MiniGrid context input requires exact float64(B1..128,296)")
    # All rows project before native mutation; mixed TRAIN cue rows are then
    # refused by the existing homogeneous-context training contract.
    return np.array(
        [context_public_history(tuple(float(v) for v in row)) for row in features],
        dtype=np.float64,
    )


class MiniGridContextPerceptronLearner:
    def __init__(self):
        self._native = ContextPerceptronLearner(context_policy())

    @property
    def policy(self):
        return self._native.policy

    @property
    def completed_updates(self):
        return self.snapshot_state().completed_updates

    @property
    def training_examples(self):
        return self.snapshot_state().training_examples

    def snapshot_state(self):
        return copy_snapshot(MiniGridContextSnapshot(self.policy, self._native.snapshot_state()))

    def restore_state(self, state):
        self._native.restore_state(copy_snapshot(state).native_state)

    def fork(self):
        child = MiniGridContextPerceptronLearner()
        child.restore_state(self.snapshot_state())
        return child

    def train_batch(self, features, targets):
        self.snapshot_state()
        return self._native.train_batch(project_features(features), targets)

    def predict(self, features):
        self.snapshot_state()
        return self._native.predict(project_features(features))
