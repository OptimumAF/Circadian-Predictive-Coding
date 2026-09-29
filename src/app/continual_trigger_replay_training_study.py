"""Preflight all six fixed v14 guarded training trials without final roles.

Inputs are the prospective manifest and the train-only runner. Outputs are
all validated unscored trials. This module does not release final tests,
score models, tune a setting, checkpoint, or write files.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.app import continual_shift_benchmark as base
from src.app.continual_checkpoint import continual_config_digest
from src.app.continual_matched_replay_runner import AppliedMethodReplay
from src.app.continual_trigger_replay_runner import (
    AppliedTriggerOpportunity,
    TRIGGER_REPLAY_TRAINING_PROTOCOL,
    MethodWakeWork,
    TriggerReplayTrainingResult,
    _arm_training_config,
    _parameter_digest,
    run_trigger_replay_training,
)
from src.app.continual_trigger_replay_schedule import (
    TriggerReplayOpportunity,
    TriggerReplayOpportunityManifest,
    TriggerReplayScheduleSession,
    _validate_manifest,
)
from src.app.wake_diagnostic import validate_wake_diagnostic_trials
from src.infra.continual_roles import PhaseDecisionRoles
from src.core.sleep_telemetry import SleepEventTelemetry


TRIGGER_REPLAY_TRAINING_STUDY_PROTOCOL = "continual_trigger_replay_train_only_v14"


@dataclass(frozen=True)
class TriggerReplayTrainingStudy:
    """Complete six-trial train-only gate, still holding all final seals."""

    protocol_id: str
    manifest: TriggerReplayOpportunityManifest
    manifest_digest: str
    trials: tuple[TriggerReplayTrainingResult, ...]


def run_trigger_replay_training_study(
    manifest: TriggerReplayOpportunityManifest,
    *,
    capture_wake_diagnostics: bool = False,
) -> TriggerReplayTrainingStudy:
    """Train and independently validate all fixed cells before any score."""
    _validate_manifest(manifest)
    trials = tuple(
        run_trigger_replay_training(
            manifest,
            seed=seed,
            arm=arm,
            capture_wake_diagnostics=capture_wake_diagnostics,
        )
        for seed in manifest.seeds
        for arm in manifest.arms
    )
    study = TriggerReplayTrainingStudy(
        TRIGGER_REPLAY_TRAINING_STUDY_PROTOCOL,
        manifest,
        continual_config_digest(manifest, manifest.seeds),
        trials,
    )
    preflight_trigger_replay_training_study(study)
    return study


def preflight_trigger_replay_training_study(study: TriggerReplayTrainingStudy) -> None:
    """Recheck a complete unscored Cartesian immediately before final release."""
    if type(study) is not TriggerReplayTrainingStudy:
        raise ValueError("v14 train-only study type differs")
    manifest = study.manifest
    _validate_manifest(manifest)
    expected_cells = tuple((seed, arm) for seed in manifest.seeds for arm in manifest.arms)
    if (
        study.protocol_id != TRIGGER_REPLAY_TRAINING_STUDY_PROTOCOL
        or study.manifest_digest != continual_config_digest(manifest, manifest.seeds)
        or tuple((trial.seed, trial.arm) for trial in study.trials) != expected_cells
    ):
        raise ValueError("v14 train-only study identity or cells differ")
    for trial in study.trials:
        _validate_trial(manifest, trial)
    for seed in manifest.seeds:
        _validate_matched_seed(tuple(trial for trial in study.trials if trial.seed == seed))
    if any(trial.wake_diagnostics for trial in study.trials):
        validate_wake_diagnostic_trials(study.trials)


def _validate_trial(
    manifest: TriggerReplayOpportunityManifest, trial: TriggerReplayTrainingResult
) -> None:
    pending = trial.pending
    session = TriggerReplayScheduleSession(manifest, seed=trial.seed)
    if (
        trial.protocol_id != TRIGGER_REPLAY_TRAINING_PROTOCOL
        or trial.manifest_digest != session.manifest_digest
        or trial.seed != pending.seed
        or trial.arm not in manifest.arms
        or pending.phase_a.final_released
        or pending.phase_b.final_released
        or pending.phase_a.final_test is not None
        or pending.phase_b.final_test is not None
        or any(access.role == "final_test" for access in pending.audit.accesses)
        or any(
            access.role == "outer_selection"
            and access.action not in {"source_release", "label_release"}
            for access in pending.audit.accesses
        )
    ):
        raise ValueError("v14 train-only trial identity or final seal differs")
    _validate_roles(pending.phase_a, session._roles)
    _validate_initial_state(manifest, trial)
    count = manifest.arrived.training.phase_a_epochs + manifest.arrived.training.phase_b_epochs
    if len(trial.opportunities) != count or len(pending.sleep_events) != count:
        raise ValueError("v14 train-only decision opportunity count differs")

    next_index = 0
    phase_a_retention = None
    cumulative_splits = cumulative_prunes = 0
    previous_width = manifest.arrived.training.hidden_dim
    for phase in ("a", "b"):
        if phase == "b":
            session.arrive_phase_b(manifest)
            _validate_roles(pending.phase_b, session._roles)
        phase_count = (
            manifest.arrived.training.phase_a_epochs
            if phase == "a"
            else manifest.arrived.training.phase_b_epochs
        )
        for _ in range(phase_count):
            expected = session.complete_wake_epoch(manifest, source_role="train")
            actual = trial.opportunities[next_index]
            event = pending.sleep_events[next_index]
            cumulative_splits += len(event.changes.applied_split_pairs)
            cumulative_prunes += len(event.changes.applied_scheduled_prune_ids) + len(
                event.changes.applied_removed_prune_ids
            )
            _validate_opportunity(
                manifest,
                trial.arm,
                session._roles.split_hashes["inner_guard"],
                expected,
                actual,
                event,
                previous_width,
                cumulative_splits,
                cumulative_prunes,
            )
            previous_width = actual.width
            next_index += 1
        if phase == "a":
            phase_a_retention = session.retention
    _validate_final_train_state(manifest, trial, session, phase_a_retention)
    _validate_structural_lineage(manifest, trial)


def _validate_roles(actual: PhaseDecisionRoles, expected: PhaseDecisionRoles) -> None:
    if (
        actual.phase != expected.phase
        or actual.seed != expected.seed
        or any(
            actual.sample_ids[role] != expected.sample_ids[role]
            or actual.split_hashes[role] != expected.split_hashes[role]
            for role in ("train", "inner_guard", "outer_selection")
        )
    ):
        raise ValueError("v14 train-only arrived development role differs")


def _validate_initial_state(
    manifest: TriggerReplayOpportunityManifest, trial: TriggerReplayTrainingResult
) -> None:
    training = _arm_training_config(manifest, trial.arm)
    models, circadian = base._new_checkpoint_models(training, trial.seed, manifest.policy)
    expected = tuple(
        (method, _parameter_digest(model))
        for method, model in (
            ("backprop", models.backprop_model),
            ("predictive_coding", models.predictive_model),
            ("circadian_predictive_coding", circadian),
        )
    )
    if trial.initial_parameter_digests != expected:
        raise ValueError("v14 train-only initial model state differs")


def _validate_opportunity(
    manifest: TriggerReplayOpportunityManifest,
    arm: str,
    guard_hash: str,
    expected: TriggerReplayOpportunity,
    actual: AppliedTriggerOpportunity,
    event: SleepEventTelemetry,
    previous_width: int,
    cumulative_splits: int,
    cumulative_prunes: int,
) -> None:
    if not isinstance(actual, AppliedTriggerOpportunity) or not isinstance(
        event, SleepEventTelemetry
    ):
        raise ValueError("v14 train-only opportunity or event type differs")
    if (
        actual.phase != expected.phase
        or actual.epoch != expected.epoch
        or actual.global_epoch != expected.global_epoch
        or actual.train_role_hash != expected.train_role_hash
        or actual.retention != expected.retention
        or actual.retained_order_ids != expected.retained_order_ids
        or actual.selected_ids != expected.selected_ids
        or actual.event != event
        or event.completed_epoch != expected.global_epoch
        or event.wake_batches != expected.global_epoch
        or event.before_width != previous_width
        or event.final_width != actual.width
        or actual.parameter_count != 4 * actual.width + 1
        or actual.cumulative_splits != cumulative_splits
        or actual.cumulative_prunes != cumulative_prunes
        or not manifest.min_width <= actual.width <= manifest.max_width
        or actual.cumulative_splits > manifest.max_applied_splits
        or actual.cumulative_prunes > manifest.max_applied_prunes
    ):
        raise ValueError("v14 train-only opportunity, event, or capacity differs")
    if event.guard is not None and (
        event.guard.role != "inner_guard"
        or event.guard.role_hash != guard_hash
        or event.outcome not in {"accepted", "rolled_back"}
    ):
        raise ValueError("v14 train-only guard role or outcome differs")
    applied = expected.selected_ids if event.outcome == "accepted" else ()
    if event.outcome not in {"accepted", "rolled_back", "skipped"} or (
        event.replay.applied_updates != len(applied)
        or event.replay.applied_examples != len(applied)
        or (event.outcome in {"accepted", "rolled_back"} and event.guard is None)
    ):
        raise ValueError("v14 train-only guarded replay work differs")
    steps = {
        "backprop": 0,
        "predictive_coding": manifest.pc_replay_inference_steps,
        "circadian_predictive_coding": manifest.arrived.training.circadian_config.replay_inference_steps,
    }
    expected_work = tuple(
        AppliedMethodReplay(
            method, applied, len(applied), len(applied), len(applied) * steps[method]
        )
        for method in manifest.arrived.training.model_order
    )
    if actual.applied_by_method != expected_work:
        raise ValueError("v14 train-only matched baseline replay work differs")
    interval = (
        manifest.arrived.training.circadian_sleep_interval_phase_a
        if expected.phase == "a"
        else manifest.arrived.training.circadian_sleep_interval_phase_b
    )
    periodic_due = arm == "periodic" and expected.epoch % interval == 0
    if arm == "no_sleep" and (event.outcome != "skipped" or event.trigger_reason != "not_due"):
        raise ValueError("v14 no-sleep arm attempted sleep")
    if arm == "periodic" and (
        (periodic_due and (event.outcome == "skipped" or event.trigger_reason != "periodic"))
        or (not periodic_due and (event.outcome != "skipped" or event.trigger_reason != "not_due"))
    ):
        raise ValueError("v14 periodic arm decision differs")
    if arm == "adaptive" and event.trigger_reason not in {"not_due", "adaptive"}:
        raise ValueError("v14 adaptive arm decision differs")


def _validate_final_train_state(
    manifest: TriggerReplayOpportunityManifest,
    trial: TriggerReplayTrainingResult,
    session: TriggerReplayScheduleSession,
    phase_a_retention: object,
) -> None:
    pending = trial.pending
    clock = pending.state.circadian_model.get_sleep_clocks()
    exposure = pending.state.circadian_model.get_replay_exposure()
    applied_ids = {
        sample_id
        for item in trial.opportunities
        if item.event.outcome == "accepted"
        for sample_id in item.selected_ids
    }
    expected_examples = (
        len(pending.phase_a.train.input) * manifest.arrived.training.phase_a_epochs
        + len(pending.phase_b.train.input) * manifest.arrived.training.phase_b_epochs
    )
    expected_work = tuple(
        MethodWakeWork(
            method,
            len(trial.opportunities),
            expected_examples,
            len(trial.opportunities) * steps,
            expected_examples * steps,
        )
        for method, steps in (
            ("backprop", 0),
            ("predictive_coding", manifest.arrived.training.pc_inference_steps),
            ("circadian_predictive_coding", manifest.arrived.training.circadian_inference_steps),
        )
    )
    if (
        pending.state.circadian_after_a.get_replay_retention() != phase_a_retention
        or pending.state.circadian_model.get_replay_retention() != session.retention
        or clock.wake_batches != len(trial.opportunities)
        or clock.wake_examples != expected_examples
        or clock.replay_updates
        != sum(item.event.replay.applied_updates for item in trial.opportunities)
        or exposure.replay_updates != clock.replay_updates
        or set(exposure.exposed_ids) != applied_ids
        or trial.wake_work != expected_work
        or pending.state.total_splits > manifest.max_applied_splits
        or pending.state.total_prunes > manifest.max_applied_prunes
        or pending.state.total_splits != trial.opportunities[-1].cumulative_splits
        or pending.state.total_prunes != trial.opportunities[-1].cumulative_prunes
        or len(pending.audit.guard_decisions)
        != sum(item.event.outcome in {"accepted", "rolled_back"} for item in trial.opportunities)
    ):
        raise ValueError("v14 train-only retained state or work clock differs")


def _validate_matched_seed(trials: tuple[TriggerReplayTrainingResult, ...]) -> None:
    if len(trials) != 3 or tuple(trial.arm for trial in trials) != (
        "periodic",
        "adaptive",
        "no_sleep",
    ):
        raise ValueError("v14 train-only arm order differs")
    _validate_matched_seed_prefix(trials)


def _validate_matched_seed_prefix(trials: tuple[TriggerReplayTrainingResult, ...]) -> None:
    """Check matched facts even when a checkpoint ends mid-seed."""
    if not trials:
        raise ValueError("v14 train-only matched seed prefix is empty")
    reference = trials[0]
    for trial in trials[1:]:
        if (
            trial.initial_parameter_digests != reference.initial_parameter_digests
            or trial.wake_work != reference.wake_work
            or any(
                actual.train_role_hash != expected.train_role_hash
                or actual.retention != expected.retention
                or actual.retained_order_ids != expected.retained_order_ids
                or actual.selected_ids != expected.selected_ids
                for actual, expected in zip(
                    trial.opportunities, reference.opportunities, strict=True
                )
            )
        ):
            raise ValueError("v14 train-only arms have unmatched source or initial work")


def _validate_structural_lineage(
    manifest: TriggerReplayOpportunityManifest, trial: TriggerReplayTrainingResult
) -> None:
    """Reconstruct applied stable IDs and compare both saved model states."""
    active = list(range(manifest.arrived.training.hidden_dim))
    parent_ids: dict[int, int | None] = {neuron_id: None for neuron_id in active}
    next_id = len(active)
    for item in trial.opportunities:
        if item.event.before_width != len(active):
            raise ValueError("v14 structural lineage width before sleep differs")
        for parent, child in item.event.changes.applied_split_pairs:
            if parent not in active or child != next_id:
                raise ValueError("v14 structural split identity differs")
            active.append(child)
            parent_ids[child] = parent
            next_id += 1
        for removed in item.event.changes.applied_removed_prune_ids:
            if removed not in active:
                raise ValueError("v14 structural prune identity differs")
            active.remove(removed)
        if item.width != len(active):
            raise ValueError("v14 structural lineage width after sleep differs")
        if item.phase == "a" and item.epoch == manifest.arrived.training.phase_a_epochs:
            _compare_lineage(trial.pending.state.circadian_after_a, active, parent_ids, next_id)
    _compare_lineage(trial.pending.state.circadian_model, active, parent_ids, next_id)


def _compare_lineage(
    model: object, active: list[int], parent_ids: dict[int, int | None], next_id: int
) -> None:
    from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork

    if not isinstance(model, CircadianPredictiveCodingNetwork):
        raise ValueError("v14 structural saved model type differs")
    lineage = model.get_neuron_lineage()
    if (
        lineage.neuron_ids != tuple(active)
        or lineage.parent_ids != tuple(parent_ids[neuron_id] for neuron_id in active)
        or lineage.next_neuron_id != next_id
    ):
        raise ValueError("v14 structural applied IDs differ from model lineage")
