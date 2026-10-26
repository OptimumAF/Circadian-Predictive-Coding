"""Opt-in two-episode mean-CE recurrent SGD with explicit episode/step clocks.

Each complete TRAIN (or explicit probe_train) episode starts at zero hidden activation. Their two mean
losses receive equal weight, including unequal lengths, in one atomic SGD step.
This separate snapshot never weakens legacy single-episode clock validation.
No stream, data collection, release authority, evaluation or selection lives here.
"""

from copy import deepcopy
from dataclasses import dataclass, fields
from math import isfinite
from typing import Any, cast

import numpy as np

from src.adapters import torch_recurrent_learner as native
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.core.coupled_energy import Array
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    copy_episode,
    episode_arrays,
)
from src.shared.torch_runtime import require_torch
from src.core.learner_ports import TrainingUpdateFailed

TORCH_RECURRENT_EPISODE_PAIR_CE_ID = "torch_recurrent_two_episode_cross_entropy_mean_v1"


@dataclass(frozen=True)
class TorchRecurrentEpisodeBatchSnapshot:
    format_version: int
    policy: native.TorchRecurrentPolicy
    parameters: dict[str, Any]
    optimizer_state: dict[str, Any]
    gradients: dict[str, Any]
    completed_updates: int
    training_episodes: int
    training_examples: int


def _exact(value: Any, kind: type) -> None:
    if type(value) is not kind or set(vars(value)) != {field.name for field in fields(kind)}:
        raise ValueError(f"an exact complete {kind.__name__} is required")


def _bridge(value: TorchRecurrentEpisodeBatchSnapshot) -> native.TorchRecurrentSnapshot:
    # Why: native validation bounds complete episodes, not optimizer steps.
    # This internal bridge is never a public/restorable legacy checkpoint.
    return native.TorchRecurrentSnapshot(
        1,
        value.policy,
        value.parameters,
        value.optimizer_state,
        value.gradients,
        value.training_episodes,
        value.training_examples,
    )


def _snapshot(value: TorchRecurrentEpisodeBatchSnapshot) -> TorchRecurrentEpisodeBatchSnapshot:
    _exact(value, TorchRecurrentEpisodeBatchSnapshot)
    if type(value.format_version) is not int or value.format_version != 1:
        raise ValueError("unsupported episode-batch snapshot version")
    policy = native._policy(value.policy)
    for name in ("completed_updates", "training_episodes", "training_examples"):
        native._count(getattr(value, name), name)
    if not (
        value.completed_updates <= value.training_episodes <= 2 * value.completed_updates
        and value.training_episodes
        <= value.training_examples
        <= policy.max_episode_steps * value.training_episodes
    ):
        raise ValueError("episode-batch optimizer/episode/row clocks disagree")
    # Validate every caller-owned tensor and optimizer field BEFORE copying.
    owned = native._snapshot(_bridge(value))
    return TorchRecurrentEpisodeBatchSnapshot(
        1,
        owned.policy,
        owned.parameters,
        owned.optimizer_state,
        owned.gradients,
        value.completed_updates,
        value.training_episodes,
        owned.training_examples,
    )


def _capture(
    parameters: dict[str, Any],
    optimizer: Any,
    policy: native.TorchRecurrentPolicy,
    updates: int,
    episodes: int,
    examples: int,
) -> TorchRecurrentEpisodeBatchSnapshot:
    owned = native._capture(parameters, optimizer, policy, episodes, examples)
    return _snapshot(
        TorchRecurrentEpisodeBatchSnapshot(
            1,
            owned.policy,
            owned.parameters,
            owned.optimizer_state,
            owned.gradients,
            updates,
            episodes,
            examples,
        )
    )


def _pair(
    episodes: tuple, policy: native.TorchRecurrentPolicy, *, role: str = "train"
) -> tuple[DemonstrationEpisode, ...]:
    if type(role) is not str or role not in ("train", "probe_train"):
        raise ValueError("paired update role must explicitly name train or probe_train")
    if type(episodes) is not tuple or len(episodes) != 2:
        raise ValueError("paired update requires an exact tuple of two complete episodes")
    result = []
    for episode in episodes:
        _exact(episode, DemonstrationEpisode)
        if type(episode.transitions) is not tuple or not (
            1 <= len(episode.transitions) <= policy.max_episode_steps
        ):
            raise ValueError("episode transitions must be a policy-bounded nonempty exact tuple")
        for transition in episode.transitions:
            _exact(transition, DemonstrationTransition)
        owned = copy_episode(episode)
        if (
            owned.role != role
            or owned.num_classes != policy.num_classes
            or len(owned.transitions) > policy.max_episode_steps
            or len(owned.transitions[0].observation) != policy.feature_dim
        ):
            raise ValueError(f"paired {role} episode geometry differs from the bound policy")
        result.append(owned)
    if result[0].episode_id == result[1].episode_id:
        raise ValueError("paired episode IDs must be distinct")
    return tuple(result)


def _work() -> dict[str, Any]:
    return dict(
        forward_calls_attempted=0,
        forward_calls_completed=0,
        forward_rows_attempted=0,
        forward_rows_completed=0,
        backward_calls_attempted=0,
        backward_calls_completed=0,
        backward_rows_attempted=0,
        backward_rows_completed=0,
        optimizer_steps_attempted=0,
        optimizer_steps=0,
        published_updates=0,
        failure_work_may_be_unobserved=False,
    )


@dataclass(frozen=True)
class EpisodePairDiagnostic:
    definition: str
    value: float
    episode_ids: tuple[str, ...]
    episode_losses: tuple[float, ...]
    work: dict[str, Any]

    def __post_init__(self) -> None:
        _exact(self, EpisodePairDiagnostic)
        if (
            self.definition != TORCH_RECURRENT_EPISODE_PAIR_CE_ID
            or type(self.value) is not float
            or not isfinite(self.value)
            or self.value < 0
            or type(self.episode_ids) is not tuple
            or len(self.episode_ids) != 2
            or any(type(value) is not str or not value for value in self.episode_ids)
            or self.episode_ids[0] == self.episode_ids[1]
            or type(self.episode_losses) is not tuple
            or len(self.episode_losses) != 2
            or any(
                type(value) is not float or not isfinite(value) or value < 0
                for value in self.episode_losses
            )
            or type(self.work) is not dict
            or self.work.keys() != _work().keys()
        ):
            raise ValueError("invalid episode-pair diagnostic")
        for key, value in self.work.items():
            if key == "failure_work_may_be_unobserved":
                if type(value) is not bool:
                    raise ValueError("diagnostic uncertainty must be a bool")
            elif type(value) is not int or value < 0:
                raise ValueError("diagnostic work counts must be nonnegative native integers")
        object.__setattr__(self, "work", deepcopy(self.work))


class EpisodePairUpdateFailed(TrainingUpdateFailed):
    """A staged update failed; work may include an unpublished physical SGD step."""

    def __init__(self, reason: str, work: dict[str, Any]) -> None:
        super().__init__(reason, TORCH_RECURRENT_EPISODE_PAIR_CE_ID, work)


class TorchRecurrentEpisodeBatchLearner:
    """Independent opt-in paired learner; legacy restore requires explicit conversion."""

    def __init__(self, policy: native.TorchRecurrentPolicy) -> None:
        owned = native.TorchRecurrentLearner(native._policy(policy)).snapshot_state()
        self._policy = owned.policy
        parameters, optimizer = native._restore(owned)
        self._payload = (parameters, optimizer, 0, 0, 0)

    @classmethod
    def from_single_episode_snapshot(
        cls, snapshot: native.TorchRecurrentSnapshot
    ) -> "TorchRecurrentEpisodeBatchLearner":
        if cls is not TorchRecurrentEpisodeBatchLearner:
            raise ValueError("legacy conversion requires the exact paired learner class")
        owned = native._snapshot(snapshot)
        learner = object.__new__(cls)
        learner._policy = owned.policy
        parameters, optimizer = native._restore(owned)
        learner._payload = (
            parameters,
            optimizer,
            owned.completed_updates,
            owned.completed_updates,
            owned.training_examples,
        )
        return learner

    @property
    def policy(self) -> native.TorchRecurrentPolicy:
        return native._policy(self._policy)

    @property
    def completed_updates(self) -> int:
        return self._payload[2]

    @property
    def training_episodes(self) -> int:
        return self._payload[3]

    @property
    def training_examples(self) -> int:
        return self._payload[4]

    def snapshot_state(self) -> TorchRecurrentEpisodeBatchSnapshot:
        return _capture(*self._payload[:2], self.policy, *self._payload[2:])

    def restore_state(self, snapshot: TorchRecurrentEpisodeBatchSnapshot) -> None:
        owned = _snapshot(snapshot)
        if owned.policy != self.policy:
            raise ValueError("episode-batch snapshot policy differs from this learner")
        parameters, optimizer = native._restore(_bridge(owned))
        self._payload = (
            parameters,
            optimizer,
            owned.completed_updates,
            owned.training_episodes,
            owned.training_examples,
        )

    def fork(self) -> "TorchRecurrentEpisodeBatchLearner":
        if type(self) is not TorchRecurrentEpisodeBatchLearner:
            raise ValueError("fork requires an exact paired learner")
        owned = self.snapshot_state()
        learner = object.__new__(TorchRecurrentEpisodeBatchLearner)
        learner._policy = owned.policy
        learner.restore_state(owned)
        return learner

    def predict(self, features: Array) -> Array:
        policy = self.policy
        values = native._features(features, policy)
        self.snapshot_state()
        torch = require_torch()
        with torch.no_grad():
            logits, _ = native._forward(
                self._payload[0],
                values,
                torch.zeros((1, policy.hidden_dim), dtype=torch.float32, device="cpu"),
                check_finite=True,
            )
            return native._probabilities(logits)

    def train_episode_pair(
        self, episodes: tuple[DemonstrationEpisode, ...]
    ) -> EpisodePairDiagnostic:
        """Update from TRAIN records only; probe roles require their explicit port."""
        return self._train_episode_pair(episodes, role="train")

    def train_probe_episode_pair(
        self, episodes: tuple[DemonstrationEpisode, ...]
    ) -> EpisodePairDiagnostic:
        """Update an owned probe fork from unchanged probe_train episode records."""
        return self._train_episode_pair(episodes, role="probe_train")

    def _train_episode_pair(
        self, episodes: tuple[DemonstrationEpisode, ...], *, role: str
    ) -> EpisodePairDiagnostic:
        if type(self) is not TorchRecurrentEpisodeBatchLearner:
            raise ValueError("paired updates require an exact paired learner")
        policy = self.policy
        pair = _pair(episodes, policy, role=role)
        torch = require_torch()
        if torch.is_inference_mode_enabled():
            raise ValueError("paired updates do not support inference_mode")
        saved = self.snapshot_state()
        owner = self._payload
        updates, episode_count = saved.completed_updates + 1, saved.training_episodes + 2
        row_count = sum(len(episode.transitions) for episode in pair)
        examples = saved.training_examples + row_count
        for value, name in (
            (updates, "completed_updates"),
            (episode_count, "training_episodes"),
            (examples, "training_examples"),
        ):
            native._count(value, name)
        work = _work()
        rng = torch.get_rng_state().clone()
        numpy_rng = cast(tuple[Any, ...], deepcopy(np.random.get_state()))
        grad_mode = torch.is_grad_enabled()

        def check_owner() -> None:
            if self._payload is not owner or not _same_state(saved, self.snapshot_state()):
                raise ValueError("original paired learner full state changed during staged work")

        try:
            parameters, optimizer = native._restore(_bridge(saved))
            optimizer.zero_grad(set_to_none=True)
            losses = []
            with torch.enable_grad():
                for episode in pair:
                    features, targets = episode_arrays(episode)
                    values = native._features(features, policy)
                    rows = len(episode.transitions)
                    work["forward_calls_attempted"] += 1
                    work["forward_rows_attempted"] += rows
                    logits, _ = native._forward(
                        parameters,
                        values,
                        torch.zeros((1, policy.hidden_dim), dtype=torch.float32, device="cpu"),
                        check_finite=True,
                    )
                    work["forward_calls_completed"] += 1
                    work["forward_rows_completed"] += rows
                    if not bool(torch.isfinite(logits).all().item()):
                        raise FloatingPointError("nonfinite paired recurrent logits")
                    loss = torch.nn.functional.cross_entropy(
                        logits, torch.from_numpy(targets[:, 0].astype(np.int64))
                    )
                    if not bool(torch.isfinite(loss).all().item()):
                        raise FloatingPointError("nonfinite episode mean CE")
                    losses.append(loss)
                mean_loss = (losses[0] + losses[1]) / 2
                if not bool(torch.isfinite(mean_loss).all().item()):
                    raise FloatingPointError("nonfinite paired mean CE")
                work["backward_calls_attempted"] += 1
                work["backward_rows_attempted"] += row_count
                mean_loss.backward()
                work["backward_calls_completed"] += 1
                work["backward_rows_completed"] += row_count
            for name, parameter in parameters.items():
                gradient = parameter.grad
                if (
                    type(gradient) is not torch.Tensor
                    or gradient.layout != torch.strided
                    or gradient.is_sparse
                ):
                    raise ValueError("all five paired gradients must be dense tensors")
                native._tensor(gradient, tuple(parameter.shape), f"paired gradient {name}")
            check_owner()
            work["optimizer_steps_attempted"] += 1
            optimizer.step()
            work["optimizer_steps"] += 1
            _capture(parameters, optimizer, policy, updates, episode_count, examples)
            check_owner()
            current_numpy = cast(tuple[Any, ...], np.random.get_state())
            if (
                not torch.equal(rng, torch.get_rng_state())
                or numpy_rng[0] != current_numpy[0]
                or not np.array_equal(numpy_rng[1], current_numpy[1])
                or numpy_rng[2:] != current_numpy[2:]
                or torch.is_grad_enabled() != grad_mode
                or torch.is_inference_mode_enabled()
                or self._payload is not owner
                or self.policy != policy
                or _pair(episodes, policy, role=role) != pair
            ):
                raise ValueError("paired update owner/input/RNG/mode changed during staged work")
            work["published_updates"] = 1
            diagnostic = EpisodePairDiagnostic(
                TORCH_RECURRENT_EPISODE_PAIR_CE_ID,
                float(mean_loss.item()),
                tuple(episode.episode_id for episode in pair),
                tuple(float(loss.item()) for loss in losses),
                work,
            )
            self._payload = (parameters, optimizer, updates, episode_count, examples)
            return diagnostic
        except Exception as error:
            work["published_updates"] = 0
            work["failure_work_may_be_unobserved"] = True
            raise EpisodePairUpdateFailed(f"{type(error).__name__}: {error}", work) from error
