"""Train fixed v14 NumPy arms with guard-committed matched replay.

Inputs are the fixed all-opportunity schedule, seed, and trigger arm.
Outputs are unscored A/B models, typed sleep history, and actual per-method
work. This module neither releases final roles nor writes artifacts.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from hashlib import sha256

import numpy as np

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_shift_benchmark as base
from src.app.continual_matched_replay_runner import AppliedMethodReplay
from src.app.wake_diagnostic import WakeDiagnostic, diagnostic_from_update
from src.app.continual_trigger_replay_schedule import (
    TriggerReplayOpportunity,
    TriggerReplayOpportunityManifest,
    TriggerReplayScheduleSession,
    _validate_manifest,
)
from src.core.circadian_predictive_coding import (
    CircadianPredictiveCodingNetwork,
    ReplayRetentionSnapshot,
    replay_sample_id,
)
from src.core.backprop_mlp import BackpropMLP
from src.core.predictive_coding import PredictiveCodingNetwork
from src.core.sleep_telemetry import SleepEventTelemetry
from src.infra.continual_roles import PhaseDecisionRoles


TRIGGER_REPLAY_TRAINING_PROTOCOL = "continual_trigger_replay_training_v14"


@dataclass(frozen=True)
class AppliedTriggerOpportunity:
    """One offered selection and the work actually committed after guard."""

    phase: str
    epoch: int
    global_epoch: int
    train_role_hash: str
    retention: ReplayRetentionSnapshot
    retained_order_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    event: SleepEventTelemetry
    applied_by_method: tuple[AppliedMethodReplay, ...]
    width: int
    parameter_count: int
    cumulative_splits: int
    cumulative_prunes: int


@dataclass(frozen=True)
class TriggerReplayTrainingResult:
    """One complete arm/seed before outer selection or final role release."""

    protocol_id: str
    manifest_digest: str
    seed: int
    arm: str
    pending: arrived._PendingSeed
    opportunities: tuple[AppliedTriggerOpportunity, ...]
    initial_parameter_digests: tuple[tuple[str, str], ...]
    wake_work: tuple[MethodWakeWork, ...]
    wake_diagnostics: tuple[WakeDiagnostic, ...] = ()


@dataclass(frozen=True)
class MethodWakeWork:
    """Successful full-batch wake calls and their exact row/loop exposure."""

    method: str
    optimizer_updates: int
    examples: int
    inference_iterations: int
    example_inference_iterations: int


@dataclass
class _Progress:
    models: base.ContinualRunnerState
    circadian: CircadianPredictiveCodingNetwork
    audit: arrived._RoleAudit
    arm: str
    sleep_events: list[SleepEventTelemetry] = field(default_factory=list)
    opportunities: list[AppliedTriggerOpportunity] = field(default_factory=list)
    sleep_event_count: int = 0
    total_splits: int = 0
    total_prunes: int = 0
    wake_updates: dict[str, int] = field(default_factory=dict)
    wake_examples: dict[str, int] = field(default_factory=dict)
    capture_wake_diagnostics: bool = False
    wake_diagnostics: list[WakeDiagnostic] = field(default_factory=list)


def run_trigger_replay_training(
    manifest: TriggerReplayOpportunityManifest,
    *,
    seed: int,
    arm: str,
    capture_wake_diagnostics: bool = False,
) -> TriggerReplayTrainingResult:
    """Train one arm on arrived roles, keeping all final labels sealed."""
    _validate_manifest(manifest)
    if type(arm) is not str or arm not in manifest.arms:
        raise ValueError("v14 trigger replay arm is outside the manifest")
    if type(capture_wake_diagnostics) is not bool:
        raise ValueError("wake diagnostic capture flag must be boolean")
    session = TriggerReplayScheduleSession(manifest, seed=seed)
    training = _arm_training_config(manifest, arm)
    models, circadian = base._new_checkpoint_models(training, seed, manifest.policy)
    if (
        circadian._min_hidden_dim != manifest.min_width
        or circadian.max_hidden_dim != manifest.max_width
    ):
        raise ValueError("v14 trigger replay model capacity differs from manifest")
    progress = _Progress(
        models,
        circadian,
        arrived._RoleAudit(seed),
        arm,
        capture_wake_diagnostics=capture_wake_diagnostics,
    )
    initial_digests = tuple(
        (method, _parameter_digest(_model_for_method(progress, method)))
        for method in training.model_order
    )
    phase_a = session._roles
    progress.audit.record_arrival("a")
    _train_phase(manifest, training, session, phase_a, progress)
    after_a = (
        deepcopy(models.backprop_model),
        deepcopy(models.predictive_model),
        deepcopy(circadian),
    )

    session.arrive_phase_b(manifest)
    phase_b = session._roles
    progress.audit.record_arrival("b")
    _train_phase(manifest, training, session, phase_b, progress)
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
    return TriggerReplayTrainingResult(
        TRIGGER_REPLAY_TRAINING_PROTOCOL,
        session.manifest_digest,
        seed,
        arm,
        pending,
        tuple(progress.opportunities),
        initial_digests,
        _wake_work(training, progress),
        tuple(progress.wake_diagnostics),
    )


def _arm_training_config(
    manifest: TriggerReplayOpportunityManifest, arm: str
) -> base.ContinualGlobalSealConfig:
    training = manifest.arrived.training
    periodic = arm == "periodic"
    return replace(
        training,
        circadian_force_sleep=periodic,
        circadian_sleep_interval_phase_a=(
            training.circadian_sleep_interval_phase_a if periodic else 0
        ),
        circadian_sleep_interval_phase_b=(
            training.circadian_sleep_interval_phase_b if periodic else 0
        ),
        circadian_config=replace(
            training.circadian_config,
            use_adaptive_sleep_trigger=arm == "adaptive",
        ),
    )


def _train_phase(
    manifest: TriggerReplayOpportunityManifest,
    training: base.ContinualGlobalSealConfig,
    session: TriggerReplayScheduleSession,
    roles: PhaseDecisionRoles,
    progress: _Progress,
) -> None:
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
            result = base._train_named_model_epoch(
                training,
                roles.train,
                progress.models.backprop_model,
                progress.models.predictive_model,
                progress.circadian,
                method,
            )
            progress.audit.record_update(phase, method, epoch)
            if progress.capture_wake_diagnostics:
                progress.wake_diagnostics.append(
                    diagnostic_from_update(
                        result,
                        seed=progress.audit.seed,
                        arm=progress.arm,
                        phase=phase,
                        epoch=epoch,
                        global_epoch=offset + epoch,
                        method=method,
                        train_role_hash=roles.split_hashes["train"],
                    )
                )
            progress.wake_updates[method] = progress.wake_updates.get(method, 0) + 1
            progress.wake_examples[method] = progress.wake_examples.get(method, 0) + len(
                roles.train.input
            )
        offered = session.complete_wake_epoch(manifest, source_role="train")
        _preflight_replay(progress.circadian, offered)
        event = _attempt_sleep(manifest, training, roles, progress, phase, epoch, offset, interval)
        progress.opportunities.append(_apply_opportunity(manifest, progress, offered, event))


def _preflight_replay(
    circadian: CircadianPredictiveCodingNetwork, offered: TriggerReplayOpportunity
) -> None:
    if circadian.get_replay_retention() != offered.retention:
        raise ValueError("v14 trigger replay retained IDs or caps differ before sleep")
    if circadian.get_replay_retained_order_ids() != offered.retained_order_ids:
        raise ValueError("v14 trigger replay retained order differs before sleep")
    if (
        circadian.preview_unprioritized_replay_ids(len(offered.selected_ids))
        != offered.selected_ids
    ):
        raise ValueError("v14 trigger replay selected IDs differ before sleep")
    actual_rows = tuple(
        replay_sample_id(inputs, targets)
        for inputs, targets in offered.selection.training_batches()
    )
    if actual_rows != offered.selected_ids:
        raise ValueError("v14 trigger replay selected IDs differ from copied rows")


def _attempt_sleep(
    manifest: TriggerReplayOpportunityManifest,
    training: base.ContinualGlobalSealConfig,
    roles: PhaseDecisionRoles,
    progress: _Progress,
    phase: str,
    epoch: int,
    offset: int,
    interval: int,
) -> SleepEventTelemetry:
    previous = len(progress.sleep_events)
    before = progress.circadian.get_sleep_clocks()
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
            on_guard_decision=lambda local, prior, current, performed, accepted: (
                progress.audit.record_guard(
                    phase,
                    roles.split_hashes["inner_guard"],
                    local,
                    prior,
                    current,
                    performed,
                    accepted,
                )
            ),
            on_sleep_event=progress.sleep_events.append,
            guard_role_hash=roles.split_hashes["inner_guard"],
        )
    )
    if len(progress.sleep_events) != previous + 1:
        raise ValueError("v14 trigger replay decision must emit exactly one event")
    event = progress.sleep_events[-1]
    after = progress.circadian.get_sleep_clocks()
    if (
        event.completed_epoch != offset + epoch
        or event.wake_batches != after.wake_batches
        or after.wake_batches != before.wake_batches
        or after.wake_examples != before.wake_examples
        or after.replay_updates - before.replay_updates != event.replay.applied_updates
    ):
        raise ValueError("v14 trigger replay sleep changed work clocks unexpectedly")
    return event


def _apply_opportunity(
    manifest: TriggerReplayOpportunityManifest,
    progress: _Progress,
    offered: TriggerReplayOpportunity,
    event: SleepEventTelemetry,
) -> AppliedTriggerOpportunity:
    selected_count = len(offered.selected_ids)
    if event.outcome == "accepted":
        if (
            event.replay.applied_updates != selected_count
            or event.replay.applied_examples != selected_count
        ):
            raise ValueError("v14 trigger replay committed work differs from selection")
        applied_ids = offered.selected_ids
    elif event.outcome in {"rolled_back", "skipped"}:
        if event.replay.applied_updates or event.replay.applied_examples:
            raise ValueError("v14 trigger replay rejected sleep retained work")
        applied_ids = ()
    else:
        raise ValueError("v14 trigger replay sleep outcome is incompatible")
    if (
        progress.circadian.get_replay_retention() != offered.retention
        or progress.circadian.get_replay_retained_order_ids() != offered.retained_order_ids
    ):
        raise ValueError("v14 trigger replay sleep changed retained rows")
    if (
        progress.total_splits > manifest.max_applied_splits
        or progress.total_prunes > manifest.max_applied_prunes
        or not manifest.min_width <= progress.circadian.hidden_dim <= manifest.max_width
        or event.final_width != progress.circadian.hidden_dim
    ):
        raise ValueError("v14 trigger replay structural capacity exceeds manifest")
    work = _apply_method_work(manifest, progress, offered, applied_ids)
    return AppliedTriggerOpportunity(
        offered.phase,
        offered.epoch,
        offered.global_epoch,
        offered.train_role_hash,
        offered.retention,
        offered.retained_order_ids,
        offered.selected_ids,
        event,
        work,
        progress.circadian.hidden_dim,
        _parameter_count(progress.circadian),
        progress.total_splits,
        progress.total_prunes,
    )


def _apply_method_work(
    manifest: TriggerReplayOpportunityManifest,
    progress: _Progress,
    offered: TriggerReplayOpportunity,
    applied_ids: tuple[str, ...],
) -> tuple[AppliedMethodReplay, ...]:
    replay = manifest.arrived.training.circadian_config
    steps = {
        "backprop": 0,
        "predictive_coding": manifest.pc_replay_inference_steps,
        "circadian_predictive_coding": replay.replay_inference_steps,
    }
    work = []
    for method in manifest.arrived.training.model_order:
        if method != "circadian_predictive_coding" and applied_ids:
            # Why this: each baseline sees private copies only after the
            # circadian guarded event has committed those exact row IDs.
            for inputs, targets in offered.selection.training_batches():
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
        work.append(AppliedMethodReplay(method, applied_ids, count, count, count * steps[method]))
    return tuple(work)


def _parameter_count(model: CircadianPredictiveCodingNetwork) -> int:
    return int(
        sum(
            values.size
            for values in (
                model.weight_input_hidden,
                model.bias_hidden,
                model.weight_hidden_output,
                model.bias_output,
            )
        )
    )


def _model_for_method(
    progress: _Progress, method: str
) -> BackpropMLP | PredictiveCodingNetwork | CircadianPredictiveCodingNetwork:
    if method == "backprop":
        return progress.models.backprop_model
    if method == "predictive_coding":
        return progress.models.predictive_model
    return progress.circadian


def _parameter_digest(
    model: BackpropMLP | PredictiveCodingNetwork | CircadianPredictiveCodingNetwork,
) -> str:
    """Compare equal-arm initial trainable state within each NumPy method."""
    digest = sha256(b"v14_initial_parameters\0")
    for values in (
        model.weight_input_hidden,
        model.bias_hidden,
        model.weight_hidden_output,
        model.bias_output,
    ):
        canonical = np.ascontiguousarray(values, dtype="<f8")
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _wake_work(
    training: base.ContinualGlobalSealConfig, progress: _Progress
) -> tuple[MethodWakeWork, ...]:
    steps = {
        "backprop": 0,
        "predictive_coding": training.pc_inference_steps,
        "circadian_predictive_coding": training.circadian_inference_steps,
    }
    return tuple(
        MethodWakeWork(
            method,
            progress.wake_updates.get(method, 0),
            progress.wake_examples.get(method, 0),
            progress.wake_updates.get(method, 0) * steps[method],
            progress.wake_examples.get(method, 0) * steps[method],
        )
        for method in training.model_order
    )
