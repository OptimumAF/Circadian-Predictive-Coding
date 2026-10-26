"""Warm paired recurrent checkpoints at the existing isolated array probe port.

Inputs are a complete paired snapshot and original probe_train episodes. Arrays
carry the next two episodes through the app port; recurrence still resets for
each original episode. Absolute native clocks retain the warm training history.
The app owns local probe budgets, evaluation and scoring, not this adapter.
"""

from copy import deepcopy
from dataclasses import replace

import numpy as np

from src.adapters import torch_recurrent_episode_batch as episode_batch
from src.adapters import torch_recurrent_learner as native
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_recurrent_episode_batch import (
    TORCH_RECURRENT_EPISODE_PAIR_CE_ID,
    EpisodePairUpdateFailed,
    TorchRecurrentEpisodeBatchLearner,
    TorchRecurrentEpisodeBatchSnapshot,
)
from src.adapters.torch_train_episode_batch import _binding, _same_schedule
from src.core.coupled_energy import Array
from src.core.demonstration_episode import DemonstrationEpisode, copy_episode, episode_arrays
from src.core.learner_ports import TrainingDiagnostic


def _schedule(episodes: tuple, policy: native.TorchRecurrentPolicy) -> tuple:
    if type(episodes) is not tuple or not 2 <= len(episodes) <= 32 or len(episodes) % 2:
        raise ValueError("paired probe requires an even exact tuple of2..32 probe_train episodes")
    owned = tuple(
        episode
        for index in range(0, len(episodes), 2)
        for episode in episode_batch._pair(episodes[index : index + 2], policy, role="probe_train")
    )
    if len({episode.episode_id for episode in owned}) != len(owned):
        raise ValueError("paired probe episode IDs must be globally distinct")
    return owned


class TorchPairedProbeLearner:
    """An owned warm learner with a fixed, bounded disposable probe schedule."""

    def __init__(
        self,
        snapshot: TorchRecurrentEpisodeBatchSnapshot,
        episodes: tuple[DemonstrationEpisode, ...],
    ) -> None:
        episode_batch._exact(snapshot, TorchRecurrentEpisodeBatchSnapshot)
        policy = native._policy(snapshot.policy)
        # Why: reject roles/geometry before native tensor validation or allocation.
        admitted = _schedule(episodes, policy)
        owned = episode_batch._snapshot(snapshot)
        self._configure(
            policy,
            admitted,
            (owned.completed_updates, owned.training_episodes, owned.training_examples),
        )
        self._learner = self._native_owner = TorchRecurrentEpisodeBatchLearner(policy)
        self._learner.restore_state(owned)
        self._expected_state = deepcopy(owned)
        self._check()

    def _configure(self, policy: native.TorchRecurrentPolicy, episodes: tuple, base: tuple) -> None:
        self._policy = native._policy(policy)
        self._policy_seal = replace(self._policy)
        self._episodes = self._schedule_owner = _schedule(episodes, self._policy)
        self._schedule_binding = self._binding_owner = _binding(self._episodes)
        self._base = self._base_owner = base
        self._uncertain = False

    def _metadata(self) -> None:
        if (
            type(self) is not TorchPairedProbeLearner
            or self._uncertain
            or self._learner is not self._native_owner
            or self._episodes is not self._schedule_owner
            or self._schedule_binding is not self._binding_owner
            or self._base is not self._base_owner
            or not _same_schedule(self._episodes, self._schedule_binding)
            or not _same_state(self._policy, self._policy_seal)
            or self._learner.policy != self._policy_seal
        ):
            raise ValueError(
                "paired probe schedule/base/policy/native owner changed or is uncertain"
            )

    def _progress(self, state: TorchRecurrentEpisodeBatchSnapshot) -> None:
        if type(state) is not TorchRecurrentEpisodeBatchSnapshot:
            raise ValueError("paired probe requires the new paired snapshot")
        delta = state.completed_updates - self._base[0]
        if (
            state.policy != self._policy_seal
            or type(state.completed_updates) is not int
            or not 0 <= delta <= len(self._episodes) // 2
            or type(state.training_episodes) is not int
            or state.training_episodes != self._base[1] + 2 * delta
            or type(state.training_examples) is not int
            or state.training_examples
            != self._base[2]
            + sum(len(episode.transitions) for episode in self._episodes[: 2 * delta])
        ):
            raise ValueError("paired probe native clocks must match warm base plus schedule prefix")

    def _check(self) -> TorchRecurrentEpisodeBatchSnapshot:
        self._metadata()
        state = self._learner.snapshot_state()
        self._metadata()
        self._progress(state)
        if not _same_state(state, self._expected_state):
            raise ValueError("paired probe native state changed outside an admitted operation")
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
        index = 2 * (before.completed_updates - self._base[0])
        if index == len(self._episodes):
            raise ValueError("paired probe training schedule is exhausted")
        pair = tuple(copy_episode(episode) for episode in self._episodes[index : index + 2])
        arrays = [episode_arrays(episode) for episode in pair]
        expected = tuple(np.concatenate([entry[index] for entry in arrays]) for index in (0, 1))

        def check_arrays() -> None:
            if any(
                type(actual) is not np.ndarray
                or actual.dtype != np.float64
                or actual.shape != value.shape
                or not np.array_equal(actual, value)
                for actual, value in zip((features, targets), expected, strict=True)
            ):
                raise ValueError(
                    "paired probe arrays must match the next original probe_train pair"
                )

        check_arrays()
        admitted_native = self._learner
        try:
            diagnostic = admitted_native.train_probe_episode_pair(pair)
        except EpisodePairUpdateFailed:
            self._uncertain = True
            raise
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
                    "paired probe update objective/progression differs from admitted pair"
                )
            returned = TrainingDiagnostic(diagnostic.definition, diagnostic.value)
            expected_state = deepcopy(after)
        except Exception as error:
            self._uncertain = True
            work = dict(diagnostic.work)
            work["failure_work_may_be_unobserved"] = True
            raise EpisodePairUpdateFailed(
                f"paired probe post-update refusal: {error}", work
            ) from error
        self._expected_state = expected_state
        return returned

    def snapshot_state(self) -> TorchRecurrentEpisodeBatchSnapshot:
        return self._check()

    def restore_state(self, state: TorchRecurrentEpisodeBatchSnapshot) -> None:
        self._check()
        owned = episode_batch._snapshot(state)
        self._progress(owned)
        admitted_native = self._learner
        try:
            admitted_native.restore_state(owned)
            self._metadata()
            after = admitted_native.snapshot_state()
            self._metadata()
            if not _same_state(owned, after):
                raise ValueError("paired probe restore differs from requested complete state")
            expected_state = deepcopy(after)
        except Exception:
            self._uncertain = True
            raise
        self._expected_state = expected_state

    def fork(self) -> "TorchPairedProbeLearner":
        before = self._check()
        native_owner = self._learner.fork()
        self._check()
        if native_owner is self._learner or not _same_state(before, native_owner.snapshot_state()):
            raise ValueError("paired probe fork must be a distinct identical complete native owner")
        child = object.__new__(TorchPairedProbeLearner)
        child._configure(self._policy_seal, self._episodes, self._base)
        child._learner = child._native_owner = native_owner
        child._expected_state = deepcopy(before)
        child._check()
        self._check()
        return child

    def predict(self, features: Array) -> Array:
        self._check()
        result = self._learner.predict(features)
        self._check()
        return result
