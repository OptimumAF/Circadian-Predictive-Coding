"""TRAIN reward packets for the existing two public-cue recurrent heads.

The full behavior checkpoint binds both heads; the initial public frame selects
one existing native reward update. Collection, release authority, budgets,
sampling, comparisons and evaluation remain in their existing outer components.
Why this: a two-head joint BC checkpoint cannot restore into a one-head learner.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from src.adapters import minigrid_public_cue_recurrent_learner as public
from src.adapters.torch_demonstration_reward_learner import MomentumOnlyUpdate
from src.adapters.torch_recurrent_reward import TORCH_RECURRENT_REWARD_ID
from src.adapters.torch_recurrent_zero_loss import TORCH_RECURRENT_ZERO_LOSS_ID
from src.adapters.torch_train_episode_batch import _binding as _episode_binding, _same_schedule
from src.app.coupled_attribution import _same_value
from src.core.learner_ports import TrainingDiagnostic
from src.core.policy_gradient_episode import PolicyGradientEpisode, copy_policy_gradient_episode


@dataclass(frozen=True)
class PublicCueRewardUpdate:
    """Detached TRAIN trajectory plus its complete two-head behavior checkpoint."""

    episode: PolicyGradientEpisode
    behavior_state: public.PublicCueRecurrentSnapshot

    def __post_init__(self) -> None:
        public._record(self, PublicCueRewardUpdate)
        object.__setattr__(self, "episode", copy_policy_gradient_episode(self.episode))
        object.__setattr__(self, "behavior_state", public.copy_snapshot(self.behavior_state))


def _copy_update(
    value: PublicCueRewardUpdate | MomentumOnlyUpdate,
) -> PublicCueRewardUpdate | MomentumOnlyUpdate:
    if type(value) is MomentumOnlyUpdate:
        public._record(value, MomentumOnlyUpdate)
        return MomentumOnlyUpdate(value.episode)
    public._record(value, PublicCueRewardUpdate)
    assert isinstance(value, PublicCueRewardUpdate)
    return PublicCueRewardUpdate(value.episode, value.behavior_state)


def _tree_refs(value: Any):
    """Strong references to the already validated native containers and leaves."""
    yield value
    if type(value) is dict:
        for child in value.values():
            yield from _tree_refs(child)
    elif type(value) in (tuple, list):
        for child in value:
            yield from _tree_refs(child)


def _state_refs(state: public.PublicCueRecurrentSnapshot) -> tuple[Any, ...]:
    refs = [state, state.policy, state.policy.native_policy, state.heads]
    for head in state.heads:
        refs.extend((head, head.policy))
        for value in (head.parameters, head.gradients, head.optimizer_state):
            refs.extend(_tree_refs(value))
    return tuple(refs)


def _binding(owner: public.PublicCueRecurrentLearner) -> tuple[Any, ...]:
    return (owner, owner._policy, *_state_refs(owner._state))


def _same_binding(owner: public.PublicCueRecurrentLearner, original: tuple[Any, ...]) -> bool:
    return all(a is b for a, b in zip(_binding(owner), original, strict=True))


def _packet_binding(value: PublicCueRewardUpdate | MomentumOnlyUpdate) -> tuple[Any, ...]:
    return (
        value.episode,
        value.episode.transitions,
        *(_state_refs(value.behavior_state) if isinstance(value, PublicCueRewardUpdate) else ()),
    )


class PublicCueRewardLearner:
    """Explicit reward/control wrapper; ordinary two-head BC behavior is unchanged."""

    def __init__(
        self, policy: public.PublicCueRecurrentPolicy = public.PublicCueRecurrentPolicy()
    ) -> None:
        self._learner = public.PublicCueRecurrentLearner(policy)
        self._native_owner = self._learner

    def _owner(self) -> public.PublicCueRecurrentLearner:
        if (
            type(self) is not PublicCueRewardLearner
            or set(vars(self)) != {"_learner", "_native_owner"}
            or type(self._learner) is not public.PublicCueRecurrentLearner
            or self._learner is not self._native_owner
        ):
            raise ValueError("reward port requires its exact two-head owner")
        return self._learner

    @property
    def policy(self) -> public.PublicCueRecurrentPolicy:
        return self._owner().policy

    @property
    def completed_updates(self) -> int:
        return self._owner().completed_updates

    @property
    def training_examples(self) -> int:
        return self._owner().training_examples

    def train_batch(
        self, features: PublicCueRewardUpdate | MomentumOnlyUpdate, targets: str
    ) -> TrainingDiagnostic:
        packet = _copy_update(features)
        packet_kind = type(packet)
        momentum_only = packet_kind is MomentumOnlyUpdate
        definition = TORCH_RECURRENT_ZERO_LOSS_ID if momentum_only else TORCH_RECURRENT_REWARD_ID
        frozen = deepcopy(packet)
        packet_bindings = [(v, _packet_binding(v)) for v in (features, packet)]
        episode_bindings = [(v, _episode_binding((v.episode,))) for v in (features, packet)]
        episode = packet.episode
        if type(targets) is not str or targets != episode.episode_id:
            raise ValueError("reward target must match the exact episode release identifier")
        if episode.role != "train" or episode.num_classes != 7:
            raise ValueError("two-head rewards require a complete K7 TRAIN trajectory")
        raw = public._raw(np.array([t.observation for t in episode.transitions], dtype=np.float64))
        selected = public._cue(raw[0])
        public._consistent_cues(raw, selected)
        expected_raw = raw.copy()
        owner = self._owner()
        binding = _binding(owner)
        _, before = owner._ready()
        if isinstance(packet, PublicCueRewardUpdate) and not _same_value(
            before, packet.behavior_state
        ):
            raise ValueError("full two-head behavior checkpoint is stale or differs")
        diagnostic = result = None

        def check() -> None:
            public._record(features, packet_kind)
            public._record(packet, packet_kind)
            if diagnostic is not None:
                public._record(result, TrainingDiagnostic)
            if (
                self._owner() is not owner
                or not _same_binding(owner, binding)
                or not _same_value(owner.snapshot_state(), before)
                or not _same_value(features, frozen)
                or not _same_value(packet, frozen)
                or not _same_value(raw, expected_raw)
                or any(
                    not all(a is b for a, b in zip(_packet_binding(v), pin, strict=True))
                    for v, pin in packet_bindings
                )
                or any(not _same_schedule((v.episode,), pin) for v, pin in episode_bindings)
                or not _same_binding(owner, binding)
                or self._learner is not owner
                or self._native_owner is not owner
                # A snapshot can return old values after an in-place mutation.
                # Read the actual owned state after the final snapshot callback.
                or not _same_value(owner._state, before)
                or (diagnostic is not None and not _same_value(result, diagnostic))
            ):
                raise ValueError("original two-head owner, behavior or reward packet changed")

        check()  # Refuse changed ownership before starting any native work.
        work = public._work("train", selected)
        try:
            native = public._stage(before.heads[selected], work)
            check()
            work["native_calls_attempted"], work["native_rows_attempted"] = 1, len(raw)
            if isinstance(packet, MomentumOnlyUpdate):
                # This distinct objective ignores labels/rewards/likelihoods.
                # Keep original trajectory evidence; never invent new behavior.
                result = native.train_zero_loss_episode(packet.episode)
            else:
                result = native.train_reward_episode(
                    packet.episode, deepcopy(packet.behavior_state.heads[selected]), discount=1.0
                )
            work["native_calls_returned"], work["native_rows_returned"] = 1, len(raw)
            public._record(result, TrainingDiagnostic)
            diagnostic = TrainingDiagnostic(result.definition, result.value)
            check()
            after = public.native._snapshot(native.snapshot_state())
            previous = before.heads[selected]
            if (
                after.policy != before.policy.native_policy
                or after.completed_updates != previous.completed_updates + 1
                or after.training_examples != previous.training_examples + len(raw)
                or type(result) is not TrainingDiagnostic
                or result.definition != definition
                or (momentum_only and result.value != 0.0)
            ):
                raise ValueError("native reward diagnostic or update/row clocks differ")
            heads = list(before.heads)
            heads[selected] = after
            candidate = public.copy_snapshot(
                public.PublicCueRecurrentSnapshot(before.policy, (heads[0], heads[1]))
            )
            check()
            owner._state = candidate
            return result
        except Exception as error:
            # Preserve the detached paid prefix and uncertainty, not a repaired
            # live native owner. Existing failure rules permanently block reuse.
            owner._state = replace(before, failed_work=public.FailureWork(**work))
            self._learner = self._native_owner = owner
            public._failed(before, work, error)

    def snapshot_state(self) -> public.PublicCueRecurrentSnapshot:
        return self._owner().snapshot_state()

    def restore_state(self, state: public.PublicCueRecurrentSnapshot) -> None:
        self._owner().restore_state(state)

    def fork(self) -> "PublicCueRewardLearner":
        child = object.__new__(PublicCueRewardLearner)
        child._learner = self._owner().fork()
        child._native_owner = child._learner
        return child

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._owner().predict(features)

    def start_episode(self) -> public.PublicCueRecurrentEpisode:
        return self._owner().start_episode()
