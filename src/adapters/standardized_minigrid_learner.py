"""Opt-in offline TRAIN standardization around the unchanged invariant BP head.

Fit statistics from validated current TRAIN frames once. Own those frozen values
alongside complete native state. No evaluation, selection, refitting, permissions,
budgets or environment access belongs here. Raw array calls retain the ordinary
NativeLearner contract; the stream owns role/arrival admission for updates.
"""

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from src.adapters.minigrid_train_catalogue import _validate_sources
from src.adapters.torch_feature_learner import (
    TorchFeatureLearner,
    TorchFeaturePolicy,
    TorchFeatureSnapshot,
    _snapshot,
)
from src.core.action_memory import _feature_keys
from src.core.demonstration_episode import DemonstrationEpisode
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from src.core.observation_standardization import (
    ObservationStandardization,
    fit_observation_standardization,
)

ENCODING_ID = "minigrid_train_standardized_invariant_history_296_v1"


def _exact(value: Any, kind: type[Any]) -> None:
    if type(value) is not kind or set(vars(value)) != {f.name for f in fields(kind)}:
        raise ValueError(f"expected an exact complete {kind.__name__}")


@dataclass(frozen=True)
class StandardizedMiniGridPolicy:
    statistics: ObservationStandardization
    native_policy: TorchFeaturePolicy = TorchFeaturePolicy("bp", 296, 32, 17, num_classes=7)
    encoding_id: str = ENCODING_ID

    def __post_init__(self) -> None:
        _exact(self.statistics, ObservationStandardization)
        owned = ObservationStandardization(
            self.statistics.means, self.statistics.scales, self.statistics.fitted_rows
        )
        _exact(self.native_policy, TorchFeaturePolicy)
        if (
            self.encoding_id != ENCODING_ID
            or type(self.encoding_id) is not str
            or self.native_policy != TorchFeaturePolicy("bp", 296, 32, 17, num_classes=7)
            or len(owned.means) != 296
        ):
            raise ValueError(
                "standardized MiniGrid requires its fixed BP policy/296 statistics/encoding"
            )
        object.__setattr__(self, "statistics", owned)
        object.__setattr__(self, "native_policy", TorchFeaturePolicy(**vars(self.native_policy)))

    @property
    def method(self) -> str:
        return "train_standardized_invariant_bp"


def copy_policy(policy: StandardizedMiniGridPolicy) -> StandardizedMiniGridPolicy:
    _exact(policy, StandardizedMiniGridPolicy)
    return StandardizedMiniGridPolicy(policy.statistics, policy.native_policy, policy.encoding_id)


def fit_policy(episodes: tuple[DemonstrationEpisode, ...]) -> StandardizedMiniGridPolicy:
    """Use only canonical current observations, not next views, targets or roles as features."""
    witness, _ = _validate_sources(episodes, 1)
    if any(episode.role != "train" for episode in episodes):
        raise ValueError("standardization requires TRAIN sources only")
    features = np.asarray(
        [canonicalize_public_history(step.observation) for e in episodes for step in e.transitions],
        dtype=np.float64,
    )
    statistics = fit_observation_standardization(features)
    witness.check(episodes)
    return StandardizedMiniGridPolicy(statistics)


@dataclass(frozen=True)
class StandardizedMiniGridSnapshot:
    policy: StandardizedMiniGridPolicy
    native_state: TorchFeatureSnapshot

    @property
    def completed_updates(self) -> int:
        return self.native_state.completed_updates

    @property
    def training_examples(self) -> int:
        return self.native_state.training_examples


def copy_snapshot(snapshot: StandardizedMiniGridSnapshot) -> StandardizedMiniGridSnapshot:
    _exact(snapshot, StandardizedMiniGridSnapshot)
    policy = copy_policy(snapshot.policy)
    native = _snapshot(snapshot.native_state)
    if native.policy != policy.native_policy:
        raise ValueError("standardized snapshot differs from its complete native policy")
    return StandardizedMiniGridSnapshot(policy, native)


class StandardizedMiniGridLearner:
    """Apply the same frozen transform to every update and prediction."""

    def __init__(self, policy: StandardizedMiniGridPolicy) -> None:
        self._policy = copy_policy(policy)
        self._native = TorchFeatureLearner(self._policy.native_policy)

    @property
    def policy(self) -> StandardizedMiniGridPolicy:
        return copy_policy(self._policy)

    @property
    def completed_updates(self) -> int:
        return self._native.completed_updates

    @property
    def training_examples(self) -> int:
        return self._native.training_examples

    def snapshot_state(self) -> StandardizedMiniGridSnapshot:
        return copy_snapshot(
            StandardizedMiniGridSnapshot(self.policy, self._native.snapshot_state())
        )

    def restore_state(self, snapshot: StandardizedMiniGridSnapshot) -> None:
        owned = copy_snapshot(snapshot)
        if owned.policy != self.policy:
            raise ValueError("standardized restore requires the original frozen transform/policy")
        self._native.restore_state(owned.native_state)

    def fork(self) -> "StandardizedMiniGridLearner":
        child = StandardizedMiniGridLearner(self.policy)
        child.restore_state(self.snapshot_state())
        return child

    def _features(self, features: np.ndarray) -> np.ndarray:
        rows = _feature_keys(features, 296)
        canonical = np.asarray([canonicalize_public_history(row) for row in rows], dtype=np.float64)
        stats = self.policy.statistics
        # Why: reuse the original population statistics without inventing a
        # continuous-action specification for categorical observations.
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            transformed = (canonical - np.asarray(stats.means)) / np.asarray(stats.scales)
        if not np.isfinite(transformed).all():
            raise FloatingPointError("standardized MiniGrid features are not finite")
        return transformed

    def train_batch(self, features: np.ndarray, targets: np.ndarray):
        return self._native.train_batch(self._features(features), targets)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._native.predict(self._features(features))
