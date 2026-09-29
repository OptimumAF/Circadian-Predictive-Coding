"""Plan one arrived, train-only replay stream for three NumPy methods.

Inputs are a new fixed schedule manifest and explicit phase arrival calls.
Outputs are per-sleep selected IDs, detached labeled rows, retained-memory
facts, and planned method work. This module does not train or score models,
open guard/selection/final roles, checkpoint, or rank algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from src.app import continual_arrived_benchmark as arrived
from src.app.continual_checkpoint import continual_config_digest
from src.core.circadian_predictive_coding import (
    ReplayRetentionBudget,
    ReplayRetentionSnapshot,
    replay_sample_id,
)
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.shared_replay_schedule import SharedReplayBuffer, SharedReplaySelection
from src.infra.continual_roles import PhaseDecisionRoles


MATCHED_REPLAY_SCHEDULE_PROTOCOL = "continual_matched_replay_schedule_v9"
SHARED_REPLAY_SAMPLER = "newest_retained_v1"


@dataclass(frozen=True)
class MatchedReplayScheduleManifest:
    """Bind one data/role setting, policy, and prediction-free replay budget."""

    arrived: arrived.ContinualArrivedRolesConfig
    seeds: tuple[int, ...]
    policy: ReplayRetentionPolicy
    replay_updates_per_sleep: int
    pc_replay_inference_steps: int
    sampling_policy: str = SHARED_REPLAY_SAMPLER
    protocol_id: str = MATCHED_REPLAY_SCHEDULE_PROTOCOL


@dataclass(frozen=True)
class ReplayMethodWorkPlan:
    """Planned work for one method; actual updates belong to a later runner."""

    method: str
    sample_ids: tuple[str, ...]
    planned_examples: int
    planned_optimizer_updates: int
    planned_inference_iterations: int


@dataclass(frozen=True)
class MatchedReplayBoundary:
    """One periodic sleep boundary with one selection for all methods."""

    phase: str
    epoch: int
    train_role_hash: str
    retention: ReplayRetentionSnapshot
    retained_order_ids: tuple[str, ...]
    selection: SharedReplaySelection
    method_work: tuple[ReplayMethodWorkPlan, ...]

    @property
    def selected_ids(self) -> tuple[str, ...]:
        return self.selection.sample_ids


def _validate_manifest(manifest: MatchedReplayScheduleManifest) -> None:
    if (
        type(manifest) is not MatchedReplayScheduleManifest
        or manifest.protocol_id != MATCHED_REPLAY_SCHEDULE_PROTOCOL
        or manifest.sampling_policy != SHARED_REPLAY_SAMPLER
        or type(manifest.seeds) is not tuple
        or not manifest.seeds
        or any(type(seed) is not int for seed in manifest.seeds)
        or len(set(manifest.seeds)) != len(manifest.seeds)
        or type(manifest.policy) is not ReplayRetentionPolicy
        or manifest.policy.name not in {"recent_fifo", "seeded_reservoir"}
    ):
        raise ValueError("matched replay schedule requires its fixed protocol, seeds, and policy")
    arrived._validate_arrived_config(manifest.arrived, list(manifest.seeds))
    training = manifest.arrived.training
    replay = training.circadian_config
    if (
        type(manifest.replay_updates_per_sleep) is not int
        or manifest.replay_updates_per_sleep <= 0
        or manifest.replay_updates_per_sleep != replay.replay_steps
        or type(manifest.pc_replay_inference_steps) is not int
        or manifest.pc_replay_inference_steps <= 0
    ):
        raise ValueError("matched replay schedule budgets must be positive and agree")
    if (
        replay.replay_prioritized
        or replay.replay_class_balanced
        or replay.use_adaptive_sleep_trigger
        or not replay.sleep_enable_replay
        or not training.circadian_force_sleep
    ):
        raise ValueError("matched replay requires unprioritized periodic replay for every method")


class MatchedReplayScheduleSession:
    """Advance A then B only when the caller declares one wake epoch done."""

    def __init__(self, manifest: MatchedReplayScheduleManifest, *, seed: int) -> None:
        _validate_manifest(manifest)
        if type(seed) is not int or seed not in manifest.seeds:
            raise ValueError("matched replay schedule seed is outside the manifest")
        training = manifest.arrived.training
        self._digest = continual_config_digest(manifest, manifest.seeds)
        self._seed = seed
        self.phase = "a"
        self.completed_epochs = 0
        self._roles: PhaseDecisionRoles = arrived._build_phase_a_roles(manifest.arrived, seed)
        self._train_digest = _train_content_digest(self._roles)
        self._buffer = SharedReplayBuffer(
            2,
            ReplayRetentionBudget(training.replay_max_examples, training.replay_max_bytes),
            manifest.policy,
        )

    @property
    def manifest_digest(self) -> str:
        return self._digest

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def retention(self) -> ReplayRetentionSnapshot:
        return self._buffer.retention

    def arrive_phase_b(self, manifest: MatchedReplayScheduleManifest) -> None:
        """Expose B training only after all declared A wake epochs complete."""
        self._check_identity(manifest)
        if self.phase != "a" or self.completed_epochs != manifest.arrived.training.phase_a_epochs:
            raise ValueError("matched replay Phase B cannot arrive before Phase A completes")
        phase_b_roles = arrived._build_phase_b_roles(manifest.arrived, self.seed)
        train_digest = _train_content_digest(phase_b_roles)
        self._roles = phase_b_roles
        self._train_digest = train_digest
        self.phase = "b"
        self.completed_epochs = 0

    def complete_wake_epoch(
        self, manifest: MatchedReplayScheduleManifest, *, source_role: str
    ) -> MatchedReplayBoundary | None:
        """Observe one train-only wake epoch, then plan periodic replay if due."""
        self._check_identity(manifest)
        if source_role != "train":
            raise ValueError("matched replay source role must be arrived train")
        training = manifest.arrived.training
        epoch_count = training.phase_a_epochs if self.phase == "a" else training.phase_b_epochs
        if self.completed_epochs >= epoch_count:
            raise ValueError("matched replay phase has no remaining wake epochs")
        if _train_content_digest(self._roles) != self._train_digest:
            raise ValueError("incompatible matched replay arrived train role")
        train = self._roles.train
        self._buffer.observe_train_batch(train.input, train.target)
        self.completed_epochs += 1
        interval = (
            training.circadian_sleep_interval_phase_a
            if self.phase == "a"
            else training.circadian_sleep_interval_phase_b
        )
        if self.completed_epochs % interval:
            return None
        selection = self._buffer.select_recent(manifest.replay_updates_per_sleep)
        return MatchedReplayBoundary(
            phase=self.phase,
            epoch=self.completed_epochs,
            train_role_hash=self._roles.split_hashes["train"],
            retention=self._buffer.retention,
            retained_order_ids=self._buffer.retained_order_ids,
            selection=selection,
            method_work=_method_work(manifest, selection.sample_ids),
        )

    def _check_identity(self, manifest: MatchedReplayScheduleManifest) -> None:
        _validate_manifest(manifest)
        if continual_config_digest(manifest, manifest.seeds) != self._digest:
            raise ValueError("incompatible matched replay schedule manifest")


def _method_work(
    manifest: MatchedReplayScheduleManifest, sample_ids: tuple[str, ...]
) -> tuple[ReplayMethodWorkPlan, ...]:
    training = manifest.arrived.training
    inference_steps = {
        "backprop": 0,
        "predictive_coding": manifest.pc_replay_inference_steps,
        "circadian_predictive_coding": training.circadian_config.replay_inference_steps,
    }
    count = len(sample_ids)
    return tuple(
        ReplayMethodWorkPlan(
            method=method,
            sample_ids=sample_ids,
            planned_examples=count,
            planned_optimizer_updates=count,
            planned_inference_iterations=count * inference_steps[method],
        )
        for method in training.model_order
    )


def _train_content_digest(roles: PhaseDecisionRoles) -> str:
    """Bind the ordered arrived train IDs and values without opening other roles."""
    train = roles.train
    role_ids = roles.sample_ids["train"]
    if train.input.shape[0] != len(role_ids) or train.target.shape[0] != len(role_ids):
        raise ValueError("incompatible matched replay arrived train role length")
    digest = sha256(b"matched_replay_train_role_v1")
    for index, role_id in enumerate(role_ids):
        digest.update(role_id.encode("utf-8"))
        digest.update(
            bytes.fromhex(
                replay_sample_id(train.input[index : index + 1], train.target[index : index + 1])
            )
        )
    return digest.hexdigest()
