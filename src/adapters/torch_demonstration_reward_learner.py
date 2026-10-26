"""Typed mean/weighted demonstration and reward updates on one recurrent owner.

Inputs distinguish demonstrations from sampled reward trajectories explicitly.
Outputs retain the existing native diagnostic, prediction and snapshot types.
No phase schedule, optimizer reset, release authority, budget or replay lives here.
"""

from copy import deepcopy
from dataclasses import dataclass, field, fields

from src.adapters.torch_recurrent_learner import (
    RecurrentEpisodePredictor,
    TorchRecurrentLearner,
    TorchRecurrentPolicy,
    TorchRecurrentSnapshot,
)
from src.adapters.torch_reward_update_learner import RewardUpdate, LegalRewardUpdate
from src.core.legal_policy_gradient import (
    copy_legal_policy_gradient_episode,
)
from src.core.coupled_energy import Array
from src.core.demonstration_class_weights import episode_class_weights
from src.core.demonstration_episode import DemonstrationEpisode, copy_episode
from src.core.learner_ports import TrainingDiagnostic
from src.core.policy_gradient_episode import PolicyGradientEpisode, copy_policy_gradient_episode


def _demonstration(episode: DemonstrationEpisode) -> DemonstrationEpisode:
    if type(episode) is not DemonstrationEpisode or set(vars(episode)) != {
        field.name for field in fields(DemonstrationEpisode)
    }:
        raise ValueError("demonstration must be an exact complete DemonstrationEpisode")
    owned = copy_episode(episode)
    if owned.role != "train":
        raise ValueError("only train-role demonstrations may train")
    return owned


@dataclass(frozen=True)
class DemonstrationUpdate:
    """One owned train-role demonstration; this metadata grants no release authority."""

    episode: DemonstrationEpisode

    def __post_init__(self) -> None:
        if type(self) is not DemonstrationUpdate or set(vars(self)) != {"episode"}:
            raise ValueError("demonstration update must be an exact complete DemonstrationUpdate")
        object.__setattr__(self, "episode", _demonstration(self.episode))


@dataclass(frozen=True)
class ClassWeightedDemonstrationUpdate:
    """One owned train episode carrying weights derived solely from its labels."""

    episode: DemonstrationEpisode
    class_weights: tuple[float, ...] = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not ClassWeightedDemonstrationUpdate or set(vars(self)) != {"episode"}:
            raise ValueError(
                "weighted update must be an exact complete ClassWeightedDemonstrationUpdate"
            )
        owned = _demonstration(self.episode)
        object.__setattr__(self, "episode", owned)
        object.__setattr__(self, "class_weights", episode_class_weights(owned))


@dataclass(frozen=True)
class MomentumOnlyUpdate:
    """Original TRAIN evidence for a distinct label-ignoring zero-loss objective.

    This record grants no release or on-policy authority. Actions, rewards and
    behavior probabilities are copied verbatim, not replaced with control data.
    """

    episode: PolicyGradientEpisode

    def __post_init__(self) -> None:
        if type(self) is not MomentumOnlyUpdate or set(vars(self)) != {"episode"}:
            raise ValueError("momentum update must be an exact complete MomentumOnlyUpdate")
        owned = copy_policy_gradient_episode(self.episode)
        if owned.role != "train":
            raise ValueError("momentum control requires an original train-role policy episode")
        object.__setattr__(self, "episode", owned)


class TorchDemonstrationRewardLearner:
    """Dispatch explicit native objectives without replacing their shared owner."""

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

    def _release(self, episode: DemonstrationEpisode | PolicyGradientEpisode, target: str) -> None:
        if type(target) is not str or target != episode.episode_id:
            raise ValueError("target must be the exact matching episode release identifier")
        if episode.role != "train":
            raise ValueError("only train-role episodes may train")
        if episode.num_classes != self.policy.num_classes:
            raise ValueError("episode class count differs from bound policy")

    def train_batch(
        self,
        features: DemonstrationUpdate
        | ClassWeightedDemonstrationUpdate
        | RewardUpdate
        | LegalRewardUpdate
        | MomentumOnlyUpdate,
        targets: str,
    ) -> TrainingDiagnostic:
        if type(self) is not TorchDemonstrationRewardLearner:
            raise ValueError("training requires an exact TorchDemonstrationRewardLearner")
        if type(features) is DemonstrationUpdate and set(vars(features)) == {"episode"}:
            demonstration = _demonstration(features.episode)
            self._release(demonstration, targets)
            return self._learner.train_episode(demonstration)
        if type(features) is ClassWeightedDemonstrationUpdate and set(vars(features)) == {
            "episode",
            "class_weights",
        }:
            demonstration = _demonstration(features.episode)
            weights = episode_class_weights(demonstration)
            if (
                type(features.class_weights) is not tuple
                or any(type(value) is not float for value in features.class_weights)
                or features.class_weights != weights
            ):
                raise ValueError("class weights differ from the released training episode labels")
            self._release(demonstration, targets)
            return self._learner.train_weighted_episode(demonstration, weights)
        if type(features) is RewardUpdate and set(vars(features)) == {"episode", "behavior_state"}:
            episode = copy_policy_gradient_episode(features.episode)
            self._release(episode, targets)
            # Why: the same native optimizer/gradients continue across objectives;
            # existing reward code retains full on-policy validation and its sum.
            return self._learner.train_reward_episode(
                episode, deepcopy(features.behavior_state), discount=1.0
            )
        if type(features) is LegalRewardUpdate and set(vars(features)) == {
            "episode",
            "behavior_state",
        }:
            legal = copy_legal_policy_gradient_episode(features.episode)
            self._release(legal.episode, targets)
            return self._learner.train_reward_episode(
                legal, deepcopy(features.behavior_state), discount=1.0
            )
        if type(features) is MomentumOnlyUpdate and set(vars(features)) == {"episode"}:
            episode = copy_policy_gradient_episode(features.episode)
            self._release(episode, targets)
            return self._learner.train_zero_loss_episode(episode)
        raise ValueError(
            "features must be an exact complete DemonstrationUpdate, ClassWeightedDemonstrationUpdate, RewardUpdate, LegalRewardUpdate or MomentumOnlyUpdate"
        )

    def predict(self, features: Array) -> Array:
        return self._learner.predict(features)

    def start_episode(self) -> RecurrentEpisodePredictor:
        return self._learner.start_episode()

    def snapshot_state(self) -> TorchRecurrentSnapshot:
        return self._learner.snapshot_state()

    def restore_state(self, snapshot: TorchRecurrentSnapshot) -> None:
        self._learner.restore_state(snapshot)

    def fork(self) -> "TorchDemonstrationRewardLearner":
        if type(self) is not TorchDemonstrationRewardLearner:
            raise ValueError("forks require an exact TorchDemonstrationRewardLearner")
        owned = object.__new__(TorchDemonstrationRewardLearner)
        owned._learner = self._learner.fork()
        return owned
