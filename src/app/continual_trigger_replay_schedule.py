"""Plan train-only replay opportunities at every arrived wake epoch.

Inputs are the fixed v14 manifest and explicit A-to-B arrival calls. Outputs
are shared retained/selected row IDs, detached labeled rows, and potential
per-method work. This module does not train, decide or guard sleep, access
decision/final roles, score models, checkpoint, or write files.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.app import continual_arrived_benchmark as arrived
from src.app.continual_checkpoint import continual_config_digest
from src.app.continual_matched_replay_schedule import (
    ReplayMethodWorkPlan,
    SHARED_REPLAY_SAMPLER,
    _train_content_digest,
)
from src.app.continual_shift_benchmark import ContinualGlobalSealConfig
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    ReplayRetentionBudget,
    ReplayRetentionSnapshot,
)
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.shared_replay_schedule import SharedReplayBuffer, SharedReplaySelection
from src.infra.continual_roles import PhaseDecisionRoles


TRIGGER_REPLAY_OPPORTUNITIES_PROTOCOL = "continual_trigger_opportunities_v14"
TRIGGER_ARMS = ("periodic", "adaptive", "no_sleep")


@dataclass(frozen=True)
class TriggerReplayOpportunityManifest:
    """Bind the prospective source, arms, replay, guard, and capacity caps."""

    arrived: arrived.ContinualArrivedRolesConfig
    seeds: tuple[int, ...]
    policy: ReplayRetentionPolicy
    replay_updates_per_sleep: int
    pc_replay_inference_steps: int
    arms: tuple[str, ...]
    min_width: int
    max_width: int
    max_applied_splits: int
    max_applied_prunes: int
    sampling_policy: str = SHARED_REPLAY_SAMPLER
    protocol_id: str = TRIGGER_REPLAY_OPPORTUNITIES_PROTOCOL


def fixed_trigger_replay_manifest() -> TriggerReplayOpportunityManifest:
    """Return the fixed v14 protocol before any model training or scoring."""
    training = ContinualGlobalSealConfig(
        sample_count_phase_a=160,
        sample_count_phase_b=160,
        phase_b_train_fraction=0.5,
        hidden_dim=8,
        phase_a_epochs=12,
        phase_b_epochs=12,
        pc_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval_phase_a=4,
        circadian_sleep_interval_phase_b=4,
        circadian_force_sleep=True,
        circadian_config=CircadianConfig(
            sleep_mode="components",
            max_split_per_sleep=1,
            max_prune_per_sleep=1,
            homeostatic_downscale_factor=0.99,
            replay_steps=2,
            replay_memory_size=8,
            replay_inference_steps=2,
            replay_prioritized=False,
            replay_class_balanced=False,
        ),
        replay_max_examples=8,
        replay_max_bytes=192,
    )
    return TriggerReplayOpportunityManifest(
        arrived=arrived.ContinualArrivedRolesConfig(training, 0.2, 0.2, 0.0),
        seeds=(47, 53),
        policy=ReplayRetentionPolicy("recent_fifo"),
        replay_updates_per_sleep=2,
        pc_replay_inference_steps=2,
        arms=TRIGGER_ARMS,
        min_width=4,
        max_width=32,
        max_applied_splits=6,
        max_applied_prunes=6,
    )


@dataclass(frozen=True)
class TriggerReplayOpportunity:
    """Potential replay supply for one completed wake epoch, not applied work."""

    phase: str
    epoch: int
    global_epoch: int
    train_role_hash: str
    train_role_ids: tuple[str, ...]
    retention: ReplayRetentionSnapshot
    retained_order_ids: tuple[str, ...]
    selection: SharedReplaySelection
    method_work: tuple[ReplayMethodWorkPlan, ...]

    @property
    def selected_ids(self) -> tuple[str, ...]:
        return self.selection.sample_ids


def _validate_manifest(manifest: TriggerReplayOpportunityManifest) -> None:
    # Why this: a fixed research protocol must reject changed budgets before
    # any source field opens, rather than silently become a tuning surface.
    if type(manifest) is not TriggerReplayOpportunityManifest or manifest != (
        fixed_trigger_replay_manifest()
    ):
        raise ValueError("v14 trigger replay requires the fixed prospective manifest")
    arrived._validate_arrived_config(manifest.arrived, list(manifest.seeds))


class TriggerReplayScheduleSession:
    """Advance one train-only source in declared A then B wake order."""

    def __init__(self, manifest: TriggerReplayOpportunityManifest, *, seed: int) -> None:
        _validate_manifest(manifest)
        if type(seed) is not int or seed not in manifest.seeds:
            raise ValueError("v14 trigger replay seed is outside the manifest")
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

    def arrive_phase_b(self, manifest: TriggerReplayOpportunityManifest) -> None:
        """Open B training only after the complete declared A wake phase."""
        self._check_identity(manifest)
        if self.phase != "a" or self.completed_epochs != manifest.arrived.training.phase_a_epochs:
            raise ValueError("v14 trigger replay Phase B cannot arrive before Phase A completes")
        phase_b_roles = arrived._build_phase_b_roles(manifest.arrived, self.seed)
        self._train_digest = _train_content_digest(phase_b_roles)
        self._roles = phase_b_roles
        self.phase = "b"
        self.completed_epochs = 0

    def complete_wake_epoch(
        self, manifest: TriggerReplayOpportunityManifest, *, source_role: str
    ) -> TriggerReplayOpportunity:
        """Observe one arrived train batch and offer the newest retained rows."""
        self._check_identity(manifest)
        if source_role != "train":
            raise ValueError("v14 trigger replay source role must be arrived train")
        training = manifest.arrived.training
        epoch_count = training.phase_a_epochs if self.phase == "a" else training.phase_b_epochs
        if self.completed_epochs >= epoch_count:
            raise ValueError("v14 trigger replay phase has no remaining wake epochs")
        if _train_content_digest(self._roles) != self._train_digest:
            raise ValueError("incompatible v14 trigger replay arrived train role")
        train = self._roles.train
        self._buffer.observe_train_batch(train.input, train.target)
        self.completed_epochs += 1
        selection = self._buffer.select_recent(manifest.replay_updates_per_sleep)
        return TriggerReplayOpportunity(
            phase=self.phase,
            epoch=self.completed_epochs,
            global_epoch=(
                self.completed_epochs
                if self.phase == "a"
                else training.phase_a_epochs + self.completed_epochs
            ),
            train_role_hash=self._roles.split_hashes["train"],
            train_role_ids=self._roles.sample_ids["train"],
            retention=self._buffer.retention,
            retained_order_ids=self._buffer.retained_order_ids,
            selection=selection,
            method_work=_method_work(manifest, selection.sample_ids),
        )

    def _check_identity(self, manifest: TriggerReplayOpportunityManifest) -> None:
        _validate_manifest(manifest)
        if continual_config_digest(manifest, manifest.seeds) != self._digest:
            raise ValueError("incompatible v14 trigger replay schedule manifest")


def _method_work(
    manifest: TriggerReplayOpportunityManifest, sample_ids: tuple[str, ...]
) -> tuple[ReplayMethodWorkPlan, ...]:
    training = manifest.arrived.training
    steps = {
        "backprop": 0,
        "predictive_coding": manifest.pc_replay_inference_steps,
        "circadian_predictive_coding": training.circadian_config.replay_inference_steps,
    }
    count = len(sample_ids)
    return tuple(
        ReplayMethodWorkPlan(method, sample_ids, count, count, count * steps[method])
        for method in training.model_order
    )
