"""Explicit reward-update packets through the existing generic native port.

Inputs are owned on-policy episodes and opaque behavior snapshots. Outputs are
the existing recurrent diagnostic/state/prediction types. Collection, release
authority, clocks, budgets, replay and supervised targets remain outside here.
"""

from copy import deepcopy
from dataclasses import dataclass

from src.adapters.torch_recurrent_learner import (
    RecurrentEpisodePredictor,
    TorchRecurrentLearner,
    TorchRecurrentPolicy,
    TorchRecurrentSnapshot,
)
from src.core.legal_policy_gradient import (
    LegalPolicyGradientEpisode,
    copy_legal_policy_gradient_episode,
)
from src.core.coupled_energy import Array
from src.core.learner_ports import TrainingDiagnostic
from src.core.policy_gradient_episode import (
    PolicyGradientEpisode,
    copy_policy_gradient_episode,
)


@dataclass(frozen=True)
class RewardUpdate:
    """One detached reward episode and its complete opaque behavior checkpoint."""

    episode: PolicyGradientEpisode
    behavior_state: TorchRecurrentSnapshot

    def __post_init__(self) -> None:
        if type(self) is not RewardUpdate or set(vars(self)) != {"episode", "behavior_state"}:
            raise ValueError("reward update must be an exact complete RewardUpdate")
        object.__setattr__(self, "episode", copy_policy_gradient_episode(self.episode))
        object.__setattr__(self, "behavior_state", deepcopy(self.behavior_state))


@dataclass(frozen=True)
class LegalRewardUpdate:
    """Explicit conditioned trajectory; the old RewardUpdate record stays exact."""

    episode: LegalPolicyGradientEpisode
    behavior_state: TorchRecurrentSnapshot

    def __post_init__(self) -> None:
        if type(self) is not LegalRewardUpdate or set(vars(self)) != {"episode", "behavior_state"}:
            raise ValueError("legal reward update must be an exact complete LegalRewardUpdate")
        object.__setattr__(self, "episode", copy_legal_policy_gradient_episode(self.episode))
        object.__setattr__(self, "behavior_state", deepcopy(self.behavior_state))


class TorchRewardUpdateLearner:
    """Own the existing recurrent learner with an explicit reward-only update DTO."""

    def __init__(self, policy: TorchRecurrentPolicy) -> None:
        self._learner = TorchRecurrentLearner(policy)

    @property
    def policy(self) -> TorchRecurrentPolicy:
        return self._learner.policy

    @property
    def completed_updates(self) -> int:
        return self._learner.completed_updates

    @property
    def training_examples(self) -> int:
        return self._learner.training_examples

    def train_batch(
        self, features: RewardUpdate | LegalRewardUpdate, targets: str
    ) -> TrainingDiagnostic:
        if type(self) is not TorchRewardUpdateLearner:
            raise ValueError("reward training requires an exact TorchRewardUpdateLearner")
        if type(features) not in {RewardUpdate, LegalRewardUpdate} or set(vars(features)) != {
            "episode",
            "behavior_state",
        }:
            raise ValueError(
                "reward features must be an exact complete RewardUpdate or LegalRewardUpdate"
            )
        episode: PolicyGradientEpisode | LegalPolicyGradientEpisode
        if isinstance(features, LegalRewardUpdate):
            episode = copy_legal_policy_gradient_episode(features.episode)
            record = episode.episode
        else:
            episode = record = copy_policy_gradient_episode(features.episode)
        if type(targets) is not str or targets != record.episode_id:
            raise ValueError("reward target must be the exact matching episode release identifier")
        if record.role != "train":
            raise ValueError("only train-role policy trajectories may train rewards")
        # Why: the identifier carries no class target. Existing reward code keeps
        # the full on-policy state check and original unnormalized objective.
        return self._learner.train_reward_episode(
            episode, deepcopy(features.behavior_state), discount=1.0
        )

    def predict(self, features: Array) -> Array:
        return self._learner.predict(features)

    def start_episode(self) -> RecurrentEpisodePredictor:
        return self._learner.start_episode()

    def snapshot_state(self) -> TorchRecurrentSnapshot:
        return self._learner.snapshot_state()

    def restore_state(self, snapshot: TorchRecurrentSnapshot) -> None:
        self._learner.restore_state(snapshot)

    def fork(self) -> "TorchRewardUpdateLearner":
        if type(self) is not TorchRewardUpdateLearner:
            raise ValueError("reward forks require an exact TorchRewardUpdateLearner")
        owned = object.__new__(TorchRewardUpdateLearner)
        owned._learner = self._learner.fork()
        return owned
