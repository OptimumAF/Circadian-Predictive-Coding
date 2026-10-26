"""Sequential array port bound to original complete paired TRAIN episodes.

Two arrays are concatenated only for the existing stream transport; native math
receives the two original records and resets hidden activation independently.
Cue order/provenance belongs to the caller, not an inference from numbered IDs.
No release, budget, environment or new loss math belongs here.
"""

from copy import deepcopy
from dataclasses import fields, replace
from operator import is_

import numpy as np

from src.adapters import torch_recurrent_learner as native
from src.adapters import torch_recurrent_episode_batch as episode_batch
from src.adapters.torch_recurrent_episode_batch import (
    EpisodePairUpdateFailed,
    TorchRecurrentEpisodeBatchLearner,
    TorchRecurrentEpisodeBatchSnapshot,
    TORCH_RECURRENT_EPISODE_PAIR_CE_ID,
)
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.core.coupled_energy import Array
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    copy_episode,
    episode_arrays,
)
from src.core.learner_ports import TrainingDiagnostic
from src.shared.torch_runtime import require_torch


def _schedule(
    episodes: tuple, policy: native.TorchRecurrentPolicy
) -> tuple[DemonstrationEpisode, ...]:
    if type(episodes) is not tuple or not 2 <= len(episodes) <= 128 or len(episodes) % 2:
        raise ValueError("paired schedule requires an even exact tuple of2..128 TRAIN episodes")
    for episode in episodes:
        if (
            type(episode) is not DemonstrationEpisode
            or set(vars(episode)) != {field.name for field in fields(DemonstrationEpisode)}
            or type(episode.transitions) is not tuple
            or not 1 <= len(episode.transitions) <= policy.max_episode_steps
            or any(
                type(step) is not DemonstrationTransition
                or set(vars(step)) != {field.name for field in fields(DemonstrationTransition)}
                for step in episode.transitions
            )
        ):
            raise ValueError("paired schedule requires exact complete episode records")
    owned = tuple(copy_episode(episode) for episode in episodes)
    if len({episode.episode_id for episode in owned}) != len(owned) or any(
        episode.role != "train"
        or episode.context != "A"
        or episode.num_classes != policy.num_classes
        or len(episode.transitions) > policy.max_episode_steps
        or len(episode.transitions[0].observation) != policy.feature_dim
        for episode in owned
    ):
        raise ValueError("paired schedule requires distinct-ID TRAIN A episodes matching policy")
    return owned


def _binding(episodes: tuple) -> tuple:
    # These leaves were validated native immutable tuples/scalars. Bind identity
    # as well as field sets, refusing replacements before any foreign serializer.
    return tuple(
        (
            episode,
            tuple(vars(episode).keys()),
            tuple(vars(episode).values()),
            tuple(
                (
                    step,
                    tuple(vars(step).keys()),
                    tuple(vars(step).values()),
                )
                for step in episode.transitions
            ),
        )
        for episode in episodes
    )


def _same_schedule(episodes: tuple, binding: tuple) -> bool:
    # Why: profiling found per-leaf generator/getattr overhead dominant. The
    # exact ordered key check makes values align; C-level is_ retains identity
    # and strong captured references without skipping a field or caching proof.
    if type(episodes) is not tuple or len(episodes) != len(binding):
        return False
    for episode, (owner, keys, values, steps) in zip(episodes, binding, strict=True):
        if (
            episode is not owner
            or tuple(vars(episode).keys()) != keys
            or not all(map(is_, vars(episode).values(), values))
        ):
            return False
        for step, (step_owner, step_keys, leaves) in zip(episode.transitions, steps, strict=True):
            if (
                step is not step_owner
                or tuple(vars(step).keys()) != step_keys
                or not all(map(is_, vars(step).values(), leaves))
            ):
                return False
    return True


class TorchTrainEpisodeBatchLearner:
    """Bind each stream step to the next two actual owned TRAIN records."""

    def __init__(
        self, policy: native.TorchRecurrentPolicy, episodes: tuple[DemonstrationEpisode, ...]
    ) -> None:
        self._configure(policy, episodes)
        self._learner = self._native_owner = TorchRecurrentEpisodeBatchLearner(self._policy)
        self._expected_state = deepcopy(self._learner.snapshot_state())
        self._check()

    def _configure(self, policy: native.TorchRecurrentPolicy, episodes: tuple) -> None:
        self._policy = native._policy(policy)
        self._policy_seal = replace(self._policy)
        self._episodes = _schedule(episodes, self._policy)
        self._schedule_owner = self._episodes
        self._schedule_binding = self._binding_owner = _binding(self._episodes)
        self._objective_id = TORCH_RECURRENT_EPISODE_PAIR_CE_ID
        self._uncertain = False

    def _metadata(self) -> None:
        if (
            type(self) is not TorchTrainEpisodeBatchLearner
            or self._uncertain
            or self._learner is not self._native_owner
            or self._episodes is not self._schedule_owner
            or self._schedule_binding is not self._binding_owner
            or not _same_schedule(self._episodes, self._schedule_binding)
            or type(self._objective_id) is not str
            or self._objective_id != TORCH_RECURRENT_EPISODE_PAIR_CE_ID
            or not _same_state(self._policy, self._policy_seal)
            or self._learner.policy != self._policy_seal
        ):
            raise ValueError(
                "paired schedule/objective/policy/native owner changed or is uncertain"
            )

    def _progress(self, state: TorchRecurrentEpisodeBatchSnapshot) -> None:
        if (
            type(state) is not TorchRecurrentEpisodeBatchSnapshot
            or state.policy != self._policy_seal
            or type(state.completed_updates) is not int
            or not 0 <= state.completed_updates <= len(self._episodes) // 2
            or type(state.training_episodes) is not int
            or state.training_episodes != 2 * state.completed_updates
            or type(state.training_examples) is not int
            or state.training_examples
            != sum(
                len(episode.transitions) for episode in self._episodes[: state.training_episodes]
            )
        ):
            raise ValueError(
                "paired native clocks must match the owned two-episode schedule prefix"
            )

    def _check(self) -> TorchRecurrentEpisodeBatchSnapshot:
        self._metadata()
        state = self._learner.snapshot_state()
        self._metadata()
        self._progress(state)
        if not _same_state(state, self._expected_state):
            raise ValueError("paired native state changed outside an admitted operation")
        return state

    @property
    def policy(self) -> native.TorchRecurrentPolicy:
        self._check()
        return replace(self._policy_seal)

    @property
    def completed_updates(self) -> int:
        return self._check().completed_updates

    @property
    def training_episodes(self) -> int:
        return self._check().training_episodes

    @property
    def training_examples(self) -> int:
        return self._check().training_examples

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        before = self._check()
        index = before.completed_updates * 2
        if index == len(self._episodes):
            raise ValueError("paired training schedule is exhausted")
        pair = tuple(copy_episode(episode) for episode in self._episodes[index : index + 2])
        arrays = [episode_arrays(episode) for episode in pair]
        expected = (
            np.concatenate([entry[0] for entry in arrays]),
            np.concatenate([entry[1] for entry in arrays]),
        )

        def check_arrays() -> None:
            if any(
                type(actual) is not np.ndarray
                or actual.dtype != np.float64
                or actual.shape != value.shape
                or not np.array_equal(actual, value)
                for actual, value in zip((features, targets), expected, strict=True)
            ):
                raise ValueError(
                    "paired arrays must exactly match the next two original TRAIN episodes"
                )

        check_arrays()
        admitted_native = self._learner
        diagnostic = admitted_native.train_episode_pair(pair)
        try:
            check_arrays()
            self._metadata()
            after = admitted_native.snapshot_state()
            self._metadata()
            self._progress(after)
            if (
                after.completed_updates != before.completed_updates + 1
                or after.training_episodes != before.training_episodes + 2
                or after.training_examples != before.training_examples + len(features)
                or diagnostic.definition != TORCH_RECURRENT_EPISODE_PAIR_CE_ID
                or diagnostic.episode_ids != tuple(episode.episode_id for episode in pair)
            ):
                raise ValueError(
                    "paired native update objective/progression differs from admitted records"
                )
            returned = TrainingDiagnostic(diagnostic.definition, diagnostic.value)
        except Exception as error:
            # Native publication already happened. Preserve actual physical and
            # published work, stop this owner, and never roll back unknown changes.
            self._uncertain = True
            raise EpisodePairUpdateFailed(
                f"paired array post-update refusal: {error}", diagnostic.work
            ) from error
        self._expected_state = deepcopy(after)
        return returned

    def snapshot_state(self) -> TorchRecurrentEpisodeBatchSnapshot:
        return self._check()

    def restore_state(self, state: TorchRecurrentEpisodeBatchSnapshot) -> None:
        self._check()
        owned = episode_batch._snapshot(state)
        self._progress(owned)
        admitted_native = self._learner
        admitted_native.restore_state(owned)
        try:
            self._metadata()
            after = admitted_native.snapshot_state()
            if not _same_state(owned, after):
                raise ValueError("paired restore differs from the requested complete state")
        except Exception:
            self._uncertain = True
            raise
        self._expected_state = deepcopy(after)

    def fork(self) -> "TorchTrainEpisodeBatchLearner":
        before = self._check()
        native_owner = self._learner.fork()
        self._check()
        if native_owner is self._learner or not _same_state(before, native_owner.snapshot_state()):
            raise ValueError("paired fork must be a distinct identical complete native owner")
        child = object.__new__(TorchTrainEpisodeBatchLearner)
        child._configure(self._policy_seal, self._episodes)
        child._learner = child._native_owner = native_owner
        child._expected_state = deepcopy(before)
        child._check()
        return child

    def predict(self, features: Array) -> Array:
        self._check()
        result = self._learner.predict(features)
        self._check()
        return result

    def start_episode(self) -> "_EpisodeBatchSession":
        self._check()
        session = _EpisodeBatchSession(self)
        self._check()
        return session


class _EpisodeBatchSession:
    """One independent bounded hidden trajectory over a frozen paired checkpoint."""

    def __init__(self, learner: TorchTrainEpisodeBatchLearner) -> None:
        self._learner = learner.fork()
        self._state = deepcopy(self._learner.snapshot_state())
        self._steps = 0
        torch = require_torch()
        self._hidden = torch.zeros(
            (1, self._state.policy.hidden_dim), dtype=torch.float32, device="cpu"
        )

    @property
    def completed_steps(self) -> int:
        return self._steps

    def snapshot_state(self) -> TorchRecurrentEpisodeBatchSnapshot:
        current = self._learner.snapshot_state()
        if not _same_state(current, self._state):
            raise ValueError("paired rollout learned/optimizer/gradient state changed")
        return current

    def predict(self, features: Array) -> Array:
        self.snapshot_state()
        policy = self._state.policy
        if type(self._steps) is not int or not 0 <= self._steps < policy.max_episode_steps:
            raise ValueError("paired rollout step bound exhausted")
        native._tensor(self._hidden, (1, policy.hidden_dim), "paired episode activation")
        values = native._features(features, policy, single=True)
        torch = require_torch()
        with torch.no_grad():
            logits, hidden = native._forward(
                self._state.parameters, values, self._hidden, check_finite=True
            )
            native._tensor(hidden, (1, policy.hidden_dim), "paired next activation")
            result = native._probabilities(logits)
        self.snapshot_state()
        self._hidden = hidden.detach().clone()
        self._steps += 1
        return result
