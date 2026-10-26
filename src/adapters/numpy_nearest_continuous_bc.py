"""Owned bounded nearest-observation BC through existing continuous data ports.

Stores frozen-normalized or explicit fixed-input prototypes and physical actions.
The fixed-input policy uses identity copies without inventing fitted statistics.
No refit or eviction.
Batch callers own release authority; episode updates require the policy's role.
No optimizer, runner, simulator, evaluation selection, checkpoint IO or promotion.
"""

from dataclasses import dataclass

import numpy as np

from src.core import continuous_bc as bc
from src.core import nearest_continuous_bc as nearest
from src.core import observation_standardization as observations
from src.core.continuous_demonstration import ContinuousActionSpec, ContinuousEpisode
from src.core.continuous_demonstration import continuous_training_pairs
from src.core.experience import require_training_role
from src.core.learner_ports import TrainingDiagnostic
from src.core.observation_standardization import ObservationStandardization

OBJECTIVE = "continuous_nearest_bc_added_pairs_v1"


@dataclass(frozen=True)
class NearestContinuousBCPolicy:
    spec: ContinuousActionSpec
    observation_standardization: ObservationStandardization | None
    max_prototypes: int = 256
    training_role: str = "train"

    def __post_init__(self) -> None:
        spec = bc._spec(self.spec)
        bc._geometry(spec)
        transform = observations._standardization(
            self.observation_standardization, spec.observation_dim
        )
        if type(self) is FixedInputNearestContinuousBCPolicy and transform is not None:
            raise ValueError("fixed-input nearest BC forbids fitted observation statistics")
        if transform is None and type(self) is not FixedInputNearestContinuousBCPolicy:
            raise ValueError("nearest BC requires explicit frozen observation statistics")
        if type(self.max_prototypes) is not int or not 1 <= self.max_prototypes <= 256:
            raise ValueError("nearest BC prototype capacity must be a native integer in1..256")
        require_training_role(self.training_role)
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "observation_standardization", transform)


@dataclass(frozen=True)
class FixedInputNearestContinuousBCPolicy(NearestContinuousBCPolicy):
    """Select already-scaled inputs explicitly; the legacy fitted mode is unchanged.

    Why: physical taskspace features have declared units and need no empirical
    fit. A distinct exact policy prevents None from weakening fitted admission.
    """

    observation_standardization: ObservationStandardization | None = None


def _policy(value: NearestContinuousBCPolicy) -> NearestContinuousBCPolicy:
    if type(value) not in (NearestContinuousBCPolicy, FixedInputNearestContinuousBCPolicy):
        raise ValueError("expected an exact fitted or fixed-input nearest BC policy")
    return type(value)(
        value.spec, value.observation_standardization, value.max_prototypes, value.training_role
    )


@dataclass(frozen=True)
class NearestContinuousBCSnapshot:
    policy: NearestContinuousBCPolicy
    prototype_features: bc.Array
    prototype_actions: bc.Array
    completed_updates: int
    training_examples: int


def _snapshot(
    value: NearestContinuousBCSnapshot, policy: NearestContinuousBCPolicy
) -> NearestContinuousBCSnapshot:
    if type(value) is not NearestContinuousBCSnapshot or _policy(value.policy) != policy:
        raise ValueError("nearest BC snapshot policy differs from its owner")
    rows = value.training_examples
    if type(rows) is not int or not 0 <= rows <= policy.max_prototypes:
        raise ValueError("nearest BC examples must be a native count within prototype capacity")
    updates = value.completed_updates
    if type(updates) is not int or not (updates == 0 if rows == 0 else 1 <= updates <= rows):
        raise ValueError("nearest BC update and prototype counters are inconsistent")
    features = bc._array(
        value.prototype_features, (rows, policy.spec.observation_dim), "prototypes"
    )
    actions = bc._targets(value.prototype_actions, rows, policy.spec)
    return NearestContinuousBCSnapshot(_policy(policy), features, actions, updates, rows)


class NumpyNearestContinuousBCLearner:
    def __init__(self, policy: NearestContinuousBCPolicy) -> None:
        self._policy = _policy(policy)
        self._state = _snapshot(
            NearestContinuousBCSnapshot(
                self._policy,
                np.empty((0, self._policy.spec.observation_dim), dtype=np.float64),
                np.empty((0, len(self._policy.spec.action_units)), dtype=np.float64),
                0,
                0,
            ),
            self._policy,
        )

    @property
    def policy(self) -> NearestContinuousBCPolicy:
        return _policy(self._policy)

    def snapshot_state(self) -> NearestContinuousBCSnapshot:
        return _snapshot(self._state, self._policy)

    def restore_state(self, snapshot: NearestContinuousBCSnapshot) -> None:
        candidate = _snapshot(snapshot, self._policy)
        self._state = candidate

    def fork(self) -> "NumpyNearestContinuousBCLearner":
        child = object.__new__(NumpyNearestContinuousBCLearner)
        child._policy = self.policy
        child._state = self.snapshot_state()
        return child

    def train_episode(self, episode: ContinuousEpisode) -> TrainingDiagnostic:
        features, targets = continuous_training_pairs(episode, role=self._policy.training_role)
        if episode.spec != self._policy.spec:
            raise ValueError("nearest BC episode spec differs from its policy")
        return self.train_batch(
            np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64)
        )

    def train_batch(self, features: bc.Array, targets: bc.Array) -> TrainingDiagnostic:
        before = self.snapshot_state()
        values = bc._features(features, before.policy.spec)
        actions = bc._targets(targets, len(values), before.policy.spec)
        if before.training_examples + len(values) > before.policy.max_prototypes:
            raise ValueError("nearest BC prototype capacity exceeded; no rows were evicted")
        normalized = observations.standardize_observations(
            values, before.policy.observation_standardization, before.policy.spec
        )
        candidate = _snapshot(
            NearestContinuousBCSnapshot(
                before.policy,
                np.concatenate((before.prototype_features, normalized)),
                np.concatenate((before.prototype_actions, actions)),
                before.completed_updates + 1,
                before.training_examples + len(values),
            ),
            self._policy,
        )
        diagnostic = TrainingDiagnostic(OBJECTIVE, len(values))
        self._state = candidate
        return diagnostic

    def predict_with_work(self, features: bc.Array) -> nearest.NearestPrediction:
        state = self.snapshot_state()
        if state.training_examples == 0:
            raise ValueError("nearest BC prediction requires released prototypes")
        values = observations.standardize_observations(
            features, state.policy.observation_standardization, state.policy.spec
        )
        return nearest.nearest_continuous_actions(
            values, state.prototype_features, state.prototype_actions, state.policy.spec
        )

    def predict(self, features: bc.Array) -> bc.Array:
        return self.predict_with_work(features).actions
