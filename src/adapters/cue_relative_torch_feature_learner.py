"""Bind original native feature heads to raw public cue-relative key inputs.

Inputs are raw296 initial/current public histories, hard labels or this family's
exact snapshots. Outputs preserve original native diagnostics and full owned
optimizer/gradient/chemistry/RNG state. This adapter supplies no roles, budgets,
or orchestration. Its distinct tag/types prevent accidental old-encoding restore;
they cannot authenticate the historical origin of arbitrary supplied weights.
Key-color projection is applied exactly once to raw inputs, never cached output.
"""

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from src.adapters.torch_feature_learner import (
    TorchFeatureLearner,
    TorchFeaturePolicy,
    TorchFeatureSnapshot,
    _policy,
    _snapshot,
)
from src.core.action_memory import _feature_keys
from src.core.coupled_energy import Array
from src.core.learner_ports import TrainingDiagnostic
from src.core.minigrid_cue_relative_encoding import (
    CUE_RELATIVE_HISTORY_ENCODING_ID,
    cue_relative_public_history,
)


@dataclass(frozen=True)
class CueRelativeTorchFeaturePolicy:
    native_policy: TorchFeaturePolicy
    encoding_id: str = CUE_RELATIVE_HISTORY_ENCODING_ID

    def __post_init__(self) -> None:
        if (
            type(self.encoding_id) is not str
            or self.encoding_id != CUE_RELATIVE_HISTORY_ENCODING_ID
        ):
            raise ValueError(
                "cue-relative neural policy requires its fixed public encoding identity"
            )
        owned = _policy(self.native_policy)
        if owned.feature_dim != 296 or owned.num_classes != 7:
            raise ValueError("cue-relative neural policy requires296 public features/seven classes")
        object.__setattr__(self, "native_policy", owned)

    @property
    def method(self) -> str:
        return f"public_cue_relative_{self.native_policy.method}"

    @property
    def feature_dim(self) -> int:
        return self.native_policy.feature_dim

    @property
    def hidden_dim(self) -> int:
        return self.native_policy.hidden_dim

    @property
    def num_classes(self) -> int:
        return self.native_policy.num_classes


def _record(value: Any, kind: type[Any]) -> None:
    if type(value) is not kind or set(vars(value)) != {field.name for field in fields(kind)}:
        raise ValueError(f"an exact complete {kind.__name__} is required")


def copy_cue_relative_torch_feature_policy(
    policy: CueRelativeTorchFeaturePolicy,
) -> CueRelativeTorchFeaturePolicy:
    _record(policy, CueRelativeTorchFeaturePolicy)
    return CueRelativeTorchFeaturePolicy(policy.native_policy, policy.encoding_id)


@dataclass(frozen=True)
class CueRelativeTorchFeatureSnapshot:
    policy: CueRelativeTorchFeaturePolicy
    native_state: TorchFeatureSnapshot

    @property
    def completed_updates(self) -> int:
        return self.native_state.completed_updates

    @property
    def training_examples(self) -> int:
        return self.native_state.training_examples


def copy_cue_relative_torch_feature_snapshot(
    snapshot: CueRelativeTorchFeatureSnapshot,
) -> CueRelativeTorchFeatureSnapshot:
    _record(snapshot, CueRelativeTorchFeatureSnapshot)
    policy = copy_cue_relative_torch_feature_policy(snapshot.policy)
    native = _snapshot(snapshot.native_state)
    # Only policy records are compared here. Native tensors are validated and
    # detached by their existing complete-state copier, never tensor equality.
    if native.policy != policy.native_policy:
        raise ValueError("cue-relative neural snapshot differs from its bound native policy")
    return CueRelativeTorchFeatureSnapshot(policy, native)


class CueRelativeTorchFeatureLearner:
    """Project every public row before delegating unchanged native operations."""

    def __init__(self, policy: CueRelativeTorchFeaturePolicy) -> None:
        self._policy = copy_cue_relative_torch_feature_policy(policy)
        self._native = TorchFeatureLearner(self._policy.native_policy)

    @property
    def policy(self) -> CueRelativeTorchFeaturePolicy:
        return copy_cue_relative_torch_feature_policy(self._policy)

    @property
    def completed_updates(self) -> int:
        return self.snapshot_state().completed_updates

    @property
    def training_examples(self) -> int:
        return self.snapshot_state().training_examples

    def snapshot_state(self) -> CueRelativeTorchFeatureSnapshot:
        return copy_cue_relative_torch_feature_snapshot(
            CueRelativeTorchFeatureSnapshot(self.policy, self._native.snapshot_state())
        )

    def restore_state(self, snapshot: CueRelativeTorchFeatureSnapshot) -> None:
        owned = copy_cue_relative_torch_feature_snapshot(snapshot)
        if owned.policy != self.policy:
            raise ValueError(
                "cue-relative neural snapshot encoding/native policy differs from learner"
            )
        self._native.restore_state(owned.native_state)

    def fork(self) -> "CueRelativeTorchFeatureLearner":
        child = CueRelativeTorchFeatureLearner(self.policy)
        child.restore_state(self.snapshot_state())
        return child

    def _features(self, features: Array) -> Array:
        # Why: preflight and project the complete batch before any native
        # training or prediction, so a malformed later row cannot publish a partial update.
        rows = _feature_keys(features, 296)
        return np.array([cue_relative_public_history(row) for row in rows], dtype=np.float64)

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        self.snapshot_state()
        return self._native.train_batch(self._features(features), targets)

    def predict(self, features: Array) -> Array:
        self.snapshot_state()
        return self._native.predict(self._features(features))
