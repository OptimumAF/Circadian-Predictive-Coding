"""Resume completed or active v8 policy/seed trials before final-test release.

Inputs are a fixed manifest and a trusted format-9 store. Outputs are
unscored arrived seeds in manifest order. This module validates saved
development roles and replay provenance; it does not score final roles.
"""

from __future__ import annotations

from collections import Counter

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_replay_policy_comparison as comparison
from src.app import continual_shift_benchmark as base
from src.app.continual_arrived_checkpoint import ArrivedRunnerCheckpoint
from src.app.continual_replay_policy_checkpoint import (
    REPLAY_POLICY_CHECKPOINT_FORMAT,
    ReplayPolicyActiveTransaction,
    ReplayPolicyCheckpointStore,
    ReplayPolicyRunnerCheckpoint,
    ReplayPolicyUnscoredSeed,
    replay_exposure_digest,
    validate_replay_policy_checkpoint_header,
)
from src.core.circadian_predictive_coding import replay_sample_id
from src.core.replay_retention import ReplayExposureSnapshot, ReplayRetentionPolicy
from src.infra.datasets import LabeledData


class _ActivePolicyStore:
    """Wrap v6's transaction save callback in the distinct v9 envelope."""

    def __init__(
        self,
        store: ReplayPolicyCheckpointStore,
        digest: str,
        trial_index: int,
        policy_index: int,
        records: tuple[ReplayPolicyUnscoredSeed, ...],
    ) -> None:
        self.store = store
        self.digest = digest
        self.trial_index = trial_index
        self.policy_index = policy_index
        self.records = records

    def load(self) -> ArrivedRunnerCheckpoint:
        checkpoint = self.store.load()
        if checkpoint.active is None:
            raise ValueError("v8 active policy cursor is absent")
        return checkpoint.active

    def save(self, checkpoint: ArrivedRunnerCheckpoint) -> None:
        if checkpoint.config_digest != self.digest or checkpoint.unscored_seeds != ():
            raise ValueError("v8 active policy transaction identity changed")
        values = vars(checkpoint).copy()
        values["format_version"] = REPLAY_POLICY_CHECKPOINT_FORMAT
        active = ReplayPolicyActiveTransaction(**values, policy_index=self.policy_index)
        self.store.save(
            ReplayPolicyRunnerCheckpoint(
                format_version=REPLAY_POLICY_CHECKPOINT_FORMAT,
                manifest_digest=self.digest,
                next_trial_index=self.trial_index,
                unscored_seeds=self.records,
                active=active,
            )
        )


def _observed_counts(role: LabeledData, epochs: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    if epochs == 0:
        return counts
    for index in range(role.input.shape[0]):
        sample_id = replay_sample_id(role.input[index : index + 1], role.target[index : index + 1])
        counts[sample_id] += epochs
    return counts


def _validate_exposure(
    exposure: ReplayExposureSnapshot, counts: Counter[str], replay_updates: int
) -> None:
    if (
        type(exposure) is not ReplayExposureSnapshot
        or exposure.observed_ids != tuple(sorted(counts))
        or exposure.duplicate_ids
        != tuple(sorted(key for key, count in counts.items() if count > 1))
        or exposure.duplicate_occurrences != sum(count - 1 for count in counts.values())
        or exposure.replay_updates != replay_updates
        or not set(exposure.exposed_ids).issubset(counts)
    ):
        raise ValueError("incompatible v8 replay exposure or arrived provenance")


def _validate_completed_policy_record(
    record: ReplayPolicyUnscoredSeed,
    pending: arrived._PendingSeed,
    policy: ReplayRetentionPolicy,
    manifest: comparison.ReplayPolicyComparisonManifest,
) -> None:
    state = pending.state
    phase_a_exposure = state.circadian_after_a.get_replay_exposure()
    after_b_exposure = state.circadian_model.get_replay_exposure()
    if (
        state.circadian_after_a._replay_retention_policy != policy
        or state.circadian_model._replay_retention_policy != policy
        or replay_exposure_digest(phase_a_exposure, after_b_exposure) != record.exposure_digest
    ):
        raise ValueError("incompatible v8 policy or replay exposure digest")
    training = manifest.arrived.training
    phase_a_counts = _observed_counts(pending.phase_a.train, training.phase_a_epochs)
    all_counts = phase_a_counts + _observed_counts(pending.phase_b.train, training.phase_b_epochs)
    a_updates = sum(
        event.replay.applied_updates
        for event in pending.sleep_events
        if event.completed_epoch is not None and event.completed_epoch <= training.phase_a_epochs
    )
    total_updates = sum(event.replay.applied_updates for event in pending.sleep_events)
    _validate_exposure(phase_a_exposure, phase_a_counts, a_updates)
    _validate_exposure(after_b_exposure, all_counts, total_updates)


def _rehydrate_record(
    record: ReplayPolicyUnscoredSeed,
    policy: ReplayRetentionPolicy,
    manifest: comparison.ReplayPolicyComparisonManifest,
) -> arrived._PendingSeed:
    seed = manifest.seeds[record.seed_index]
    pending = arrived._rehydrate_unscored_arrived_seed(
        record.arrived, manifest.arrived, seed, policy
    )
    _validate_completed_policy_record(record, pending, policy, manifest)
    return pending


def _capture_record(
    pending: arrived._PendingSeed, policy_index: int, seed_index: int
) -> ReplayPolicyUnscoredSeed:
    return ReplayPolicyUnscoredSeed(
        policy_index=policy_index,
        seed_index=seed_index,
        arrived=arrived._capture_unscored_arrived_seed(pending),
        exposure_digest=replay_exposure_digest(
            pending.state.circadian_after_a.get_replay_exposure(),
            pending.state.circadian_model.get_replay_exposure(),
        ),
    )


def _circadian_wake_epochs(
    active: ReplayPolicyActiveTransaction,
    phase: str,
    manifest: comparison.ReplayPolicyComparisonManifest,
) -> int:
    if phase == "b" and active.phase == "a":
        return 0
    if phase == "a" and active.phase == "b":
        return manifest.arrived.training.phase_a_epochs
    local = active.phase_epoch_completed
    if (
        active.stage == "wake"
        and "circadian_predictive_coding"
        in manifest.arrived.training.model_order[: active.next_model_index]
    ):
        return local + 1
    return local


def _validate_active_policy_exposure(
    active: ReplayPolicyActiveTransaction,
    policy: ReplayRetentionPolicy,
    seed: int,
    manifest: comparison.ReplayPolicyComparisonManifest,
) -> None:
    phase_a = arrived._build_phase_a_roles(manifest.arrived, seed)
    phase_b = arrived._build_phase_b_roles(manifest.arrived, seed) if active.phase == "b" else None
    a_counts = _observed_counts(phase_a.train, _circadian_wake_epochs(active, "a", manifest))
    all_counts = a_counts.copy()
    if phase_b is not None:
        all_counts += _observed_counts(phase_b.train, _circadian_wake_epochs(active, "b", manifest))
    candidate = base._new_checkpoint_models(manifest.arrived.training, seed, policy)[1]
    assert active.active_circadian is not None
    candidate.restore_state(active.active_circadian)
    a_events = manifest.arrived.training.phase_a_epochs
    a_updates = sum(
        item.replay.applied_updates
        for item in active.active_sleep_events
        if item.completed_epoch is not None and item.completed_epoch <= a_events
    )
    total_updates = sum(item.replay.applied_updates for item in active.active_sleep_events)
    _validate_exposure(candidate.get_replay_exposure(), all_counts, total_updates)
    if phase_b is not None:
        assert active.active_state is not None
        frozen_snapshot = active.active_state.circadian_after_a
        if frozen_snapshot is None:
            raise ValueError("incompatible v8 missing frozen Phase A exposure")
        frozen = base._new_checkpoint_models(manifest.arrived.training, seed, policy)[1]
        frozen.restore_state(frozen_snapshot)
        _validate_exposure(frozen.get_replay_exposure(), a_counts, a_updates)


def run_completed_policy_checkpoints(
    manifest: comparison.ReplayPolicyComparisonManifest,
    manifest_digest: str,
    store: ReplayPolicyCheckpointStore,
    resume: bool,
) -> tuple[tuple[arrived._PendingSeed, ...], ...]:
    """Validate saved prefix, then train each remaining declared trial once."""
    checkpoint = store.load() if resume else None
    if checkpoint is not None:
        validate_replay_policy_checkpoint_header(
            checkpoint,
            manifest_digest=manifest_digest,
            policy_count=len(manifest.policies),
            seeds=manifest.seeds,
            phase_a_epochs=manifest.arrived.training.phase_a_epochs,
            phase_b_epochs=manifest.arrived.training.phase_b_epochs,
            model_order=manifest.arrived.training.model_order,
        )
    records = list(checkpoint.unscored_seeds) if checkpoint is not None else []
    pending_by_policy: list[tuple[arrived._PendingSeed, ...]] = []
    for policy_index, policy in enumerate(manifest.policies):
        pending_policy: list[arrived._PendingSeed] = []
        for seed_index, seed in enumerate(manifest.seeds):
            trial_index = policy_index * len(manifest.seeds) + seed_index
            if trial_index < len(records):
                pending_policy.append(_rehydrate_record(records[trial_index], policy, manifest))
                continue
            from src.app.continual_arrived_transactions import train_or_resume_arrived_seed

            active = (
                checkpoint.active
                if checkpoint is not None and trial_index == checkpoint.next_trial_index
                else None
            )
            if active is not None:
                _validate_active_policy_exposure(active, policy, seed, manifest)
            adapter = _ActivePolicyStore(
                store, manifest_digest, trial_index, policy_index, tuple(records)
            )
            item = train_or_resume_arrived_seed(
                manifest.arrived,
                seed,
                seed_index=seed_index,
                seeds=manifest.seeds,
                config_digest=manifest_digest,
                unscored_seeds=(),
                store=adapter,
                checkpoint=active,
                retention_policy=policy,
            )
            records.append(_capture_record(item, policy_index, seed_index))
            store.save(
                ReplayPolicyRunnerCheckpoint(
                    format_version=REPLAY_POLICY_CHECKPOINT_FORMAT,
                    manifest_digest=manifest_digest,
                    next_trial_index=trial_index + 1,
                    unscored_seeds=tuple(records),
                )
            )
            pending_policy.append(item)
        pending_by_policy.append(tuple(pending_policy))
    return tuple(pending_by_policy)
