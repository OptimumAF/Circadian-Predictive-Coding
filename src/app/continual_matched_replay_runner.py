"""Train NumPy controls on one arrived, prediction-free replay selection.

Inputs are a fixed v9 schedule manifest and one declared seed. Outputs are
unscored trained models, role/sleep audit, and actual per-method replay work.
This module does not release final tests, tune a policy, checkpoint, or rank
algorithms.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_shift_benchmark as base
from src.app.continual_matched_replay_schedule import (
    MatchedReplayBoundary,
    MatchedReplayScheduleManifest,
    MatchedReplayScheduleSession,
)
from src.core.circadian_predictive_coding import (
    CircadianPredictiveCodingNetwork,
    WAKE_ONLY_REPLAY_SIDE_EFFECT_POLICY,
    replay_sample_id,
)
from src.core.sleep_telemetry import SleepEventTelemetry
from src.infra.continual_roles import PhaseDecisionRoles


MATCHED_REPLAY_TRAINING_PROTOCOL = "continual_matched_replay_training_v9"
SIDE_EFFECT_TRAINING_PROTOCOL = "continual_replay_side_effect_training_v10"


def replay_side_effect_training_identity(schedule_digest: str, policy: str) -> tuple[str, str]:
    """Keep v9 identity exact; bind opt-in training to a distinct digest."""
    if policy == "historical":
        return MATCHED_REPLAY_TRAINING_PROTOCOL, schedule_digest
    if policy != WAKE_ONLY_REPLAY_SIDE_EFFECT_POLICY:
        raise ValueError("unsupported matched replay side-effect policy")
    digest = sha256(
        f"{SIDE_EFFECT_TRAINING_PROTOCOL}:{schedule_digest}:{policy}".encode("ascii")
    ).hexdigest()
    return SIDE_EFFECT_TRAINING_PROTOCOL, digest


@dataclass(frozen=True)
class AppliedMethodReplay:
    """Replay rows and work actually retained by one method at one sleep."""

    method: str
    sample_ids: tuple[str, ...]
    examples: int
    optimizer_updates: int
    inference_iterations: int


@dataclass(frozen=True)
class MatchedReplayAppliedBoundary:
    """Shared proposal and guard-committed work at one periodic boundary."""

    phase: str
    epoch: int
    retained_order_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    sleep_outcome: str
    applied_by_method: tuple[AppliedMethodReplay, ...]


@dataclass(frozen=True)
class MatchedReplayTrainingResult:
    """One train-only seed, before any outer-selection or final release."""

    protocol_id: str
    manifest_digest: str
    seed: int
    pending: arrived._PendingSeed
    boundaries: tuple[MatchedReplayAppliedBoundary, ...]


@dataclass
class _TrainingProgress:
    models: base.ContinualRunnerState
    circadian: CircadianPredictiveCodingNetwork
    audit: arrived._RoleAudit
    sleep_events: list[SleepEventTelemetry] = field(default_factory=list)
    boundaries: list[MatchedReplayAppliedBoundary] = field(default_factory=list)
    sleep_event_count: int = 0
    total_splits: int = 0
    total_prunes: int = 0


def run_matched_replay_training(
    manifest: MatchedReplayScheduleManifest, *, seed: int, side_effect_policy: str = "historical"
) -> MatchedReplayTrainingResult:
    """Apply shared train rows only after their circadian sleep is accepted."""
    if side_effect_policy not in {"historical", WAKE_ONLY_REPLAY_SIDE_EFFECT_POLICY}:
        raise ValueError("unsupported matched replay side-effect policy")
    session = MatchedReplayScheduleSession(manifest, seed=seed)
    protocol_id, result_digest = replay_side_effect_training_identity(
        session.manifest_digest, side_effect_policy
    )
    training = manifest.arrived.training
    models, circadian = base._new_checkpoint_models(training, seed, manifest.policy)
    if side_effect_policy == WAKE_ONLY_REPLAY_SIDE_EFFECT_POLICY:
        circadian.configure_replay_side_effect_policy(side_effect_policy)
    progress = _TrainingProgress(models, circadian, arrived._RoleAudit(seed))
    phase_a = session._roles
    progress.audit.record_arrival("a")
    _train_phase(manifest, session, phase_a, progress)
    after_a = (
        deepcopy(models.backprop_model),
        deepcopy(models.predictive_model),
        deepcopy(circadian),
    )

    session.arrive_phase_b(manifest)
    phase_b = session._roles
    progress.audit.record_arrival("b")
    _train_phase(manifest, session, phase_b, progress)
    state = base._ContinualTrainingState(
        models.backprop_model,
        models.predictive_model,
        circadian,
        *after_a,
        progress.sleep_event_count,
        progress.total_splits,
        progress.total_prunes,
        models.hidden_dim_start,
    )
    pending = arrived._PendingSeed(
        seed, state, phase_a, phase_b, progress.audit, tuple(progress.sleep_events)
    )
    return MatchedReplayTrainingResult(
        protocol_id,
        result_digest,
        seed,
        pending,
        tuple(progress.boundaries),
    )


def _train_phase(
    manifest: MatchedReplayScheduleManifest,
    session: MatchedReplayScheduleSession,
    roles: PhaseDecisionRoles,
    progress: _TrainingProgress,
) -> None:
    training = manifest.arrived.training
    phase = session.phase
    count = training.phase_a_epochs if phase == "a" else training.phase_b_epochs
    interval = (
        training.circadian_sleep_interval_phase_a
        if phase == "a"
        else training.circadian_sleep_interval_phase_b
    )
    offset = 0 if phase == "a" else training.phase_a_epochs
    for epoch in range(1, count + 1):
        for method in training.model_order:
            base._train_named_model_epoch(
                training,
                roles.train,
                progress.models.backprop_model,
                progress.models.predictive_model,
                progress.circadian,
                method,
            )
            progress.audit.record_update(phase, method, epoch)
        boundary = session.complete_wake_epoch(manifest, source_role="train")
        if boundary is not None:
            _preflight_replay(progress.circadian, boundary, manifest.replay_updates_per_sleep)
        event = _attempt_sleep(manifest, roles, progress, phase, epoch, offset, interval)
        if boundary is not None:
            progress.boundaries.append(_apply_boundary(manifest, progress, boundary, event))
        elif event.replay.applied_updates or event.replay.applied_examples:
            raise ValueError("matched replay applied outside a shared periodic boundary")


def _preflight_replay(
    circadian: CircadianPredictiveCodingNetwork,
    boundary: MatchedReplayBoundary,
    replay_count: int,
) -> None:
    if circadian.get_replay_retention() != boundary.retention:
        raise ValueError("matched replay retained IDs or caps differ before sleep")
    if circadian.get_replay_retained_order_ids() != boundary.retained_order_ids:
        raise ValueError("matched replay retained order differs before sleep")
    if circadian.preview_unprioritized_replay_ids(replay_count) != boundary.selected_ids:
        raise ValueError("matched replay selected replay IDs differ before sleep")
    actual_rows = tuple(
        replay_sample_id(inputs, targets)
        for inputs, targets in boundary.selection.training_batches()
    )
    if actual_rows != boundary.selected_ids:
        raise ValueError("matched replay selected replay IDs differ from copied rows")


def _attempt_sleep(
    manifest: MatchedReplayScheduleManifest,
    roles: PhaseDecisionRoles,
    progress: _TrainingProgress,
    phase: str,
    epoch: int,
    offset: int,
    interval: int,
) -> SleepEventTelemetry:
    training = manifest.arrived.training
    before = len(progress.sleep_events)
    clock_before = progress.circadian.get_sleep_clocks()
    progress.sleep_event_count, progress.total_splits, progress.total_prunes = (
        base._apply_scheduled_sleep(
            progress.circadian,
            interval,
            epoch,
            offset + epoch,
            training.phase_a_epochs + training.phase_b_epochs,
            training.circadian_force_sleep,
            progress.sleep_event_count,
            progress.total_splits,
            progress.total_prunes,
            guard=roles.inner_guard,
            guard_drop_tolerance=manifest.arrived.guard_drop_tolerance,
            on_guard_decision=lambda local, previous, current, performed, accepted: (
                progress.audit.record_guard(
                    phase,
                    roles.split_hashes["inner_guard"],
                    local,
                    previous,
                    current,
                    performed,
                    accepted,
                )
            ),
            on_sleep_event=progress.sleep_events.append,
            guard_role_hash=roles.split_hashes["inner_guard"],
        )
    )
    if len(progress.sleep_events) != before + 1:
        raise ValueError("matched replay sleep must produce exactly one event")
    event = progress.sleep_events[-1]
    clock_after = progress.circadian.get_sleep_clocks()
    if (
        clock_after.wake_batches != clock_before.wake_batches
        or clock_after.wake_examples != clock_before.wake_examples
        or clock_after.replay_updates - clock_before.replay_updates != event.replay.applied_updates
    ):
        raise ValueError("matched replay sleep changed wake or replay clocks unexpectedly")
    return event


def _apply_boundary(
    manifest: MatchedReplayScheduleManifest,
    progress: _TrainingProgress,
    boundary: MatchedReplayBoundary,
    event: SleepEventTelemetry,
) -> MatchedReplayAppliedBoundary:
    count = len(boundary.selected_ids)
    if event.outcome == "accepted":
        if event.replay.applied_updates != count or event.replay.applied_examples != count:
            raise ValueError("matched replay circadian applied work differs from selection")
        applied_ids = boundary.selected_ids
    elif event.outcome in {"rolled_back", "skipped"}:
        if event.replay.applied_updates or event.replay.applied_examples:
            raise ValueError("matched replay rejected sleep retained replay work")
        applied_ids = ()
    else:
        raise ValueError("matched replay periodic sleep must be accepted or rolled back")
    if progress.circadian.get_replay_retention() != boundary.retention:
        raise ValueError("matched replay sleep refilled the retained buffer")
    if progress.circadian.get_replay_retained_order_ids() != boundary.retained_order_ids:
        raise ValueError("matched replay sleep changed retained order")
    work = _apply_method_work(manifest, progress, boundary, applied_ids)
    return MatchedReplayAppliedBoundary(
        boundary.phase,
        boundary.epoch,
        boundary.retained_order_ids,
        boundary.selected_ids,
        event.outcome,
        work,
    )


def _apply_method_work(
    manifest: MatchedReplayScheduleManifest,
    progress: _TrainingProgress,
    boundary: MatchedReplayBoundary,
    applied_ids: tuple[str, ...],
) -> tuple[AppliedMethodReplay, ...]:
    replay = manifest.arrived.training.circadian_config
    inference_steps = {
        "backprop": 0,
        "predictive_coding": manifest.pc_replay_inference_steps,
        "circadian_predictive_coding": replay.replay_inference_steps,
    }
    work = []
    for method in manifest.arrived.training.model_order:
        if method != "circadian_predictive_coding" and applied_ids:
            # Why this: give each baseline fresh detached arrays, while the
            # circadian core has already committed the same selected rows.
            for inputs, targets in boundary.selection.training_batches():
                if method == "backprop":
                    progress.models.backprop_model.train_epoch(
                        inputs, targets, learning_rate=replay.replay_learning_rate
                    )
                else:
                    progress.models.predictive_model.train_epoch(
                        inputs,
                        targets,
                        learning_rate=replay.replay_learning_rate,
                        inference_steps=manifest.pc_replay_inference_steps,
                        inference_learning_rate=replay.replay_inference_learning_rate,
                    )
        count = len(applied_ids)
        work.append(
            AppliedMethodReplay(method, applied_ids, count, count, count * inference_steps[method])
        )
    return tuple(work)
