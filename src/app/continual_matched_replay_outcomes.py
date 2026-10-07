"""Freeze all matched NumPy replay trials before common final scoring.

Inputs are one fixed arrived-role setting, two declared policies, and seeds.
Outputs keep every scored policy/seed, its applied replay work, and aggregate
metrics. This module does not tune, checkpoint, perform file IO, or select a
winner.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_matched_replay_schedule as schedule
from src.app import continual_shift_benchmark as base
from src.app.continual_checkpoint import continual_config_digest
from src.app.continual_matched_replay_runner import (
    MATCHED_REPLAY_TRAINING_PROTOCOL,
    AppliedMethodReplay,
    MatchedReplayAppliedBoundary,
    MatchedReplayTrainingResult,
    run_matched_replay_training,
)
from src.core.replay_retention import ReplayExposureSnapshot, ReplayRetentionPolicy
from src.core.sleep_telemetry import SleepEventTelemetry
from src.infra.continual_roles import PhaseDecisionRoles, release_final_test


MATCHED_REPLAY_OUTCOME_PROTOCOL = "continual_matched_replay_outcomes_v9"


@dataclass(frozen=True)
class MatchedReplayOutcomeManifest:
    """Bind every policy and seed before any source is constructed."""

    arrived: arrived.ContinualArrivedRolesConfig
    seeds: tuple[int, ...]
    policies: tuple[ReplayRetentionPolicy, ReplayRetentionPolicy]
    replay_updates_per_sleep: int
    pc_replay_inference_steps: int
    sampling_policy: str = schedule.SHARED_REPLAY_SAMPLER
    protocol_id: str = MATCHED_REPLAY_OUTCOME_PROTOCOL


@dataclass(frozen=True)
class MatchedReplaySeedOutcome:
    """One fixed seed's common-role metrics and committed replay work."""

    arrived: arrived.ContinualArrivedSeedResult
    schedule_manifest_digest: str
    boundaries: tuple[MatchedReplayAppliedBoundary, ...]
    applied_work: tuple[AppliedMethodReplay, ...]
    replay_exposure: ReplayExposureSnapshot


@dataclass(frozen=True)
class MatchedReplayPolicyOutcome:
    policy: ReplayRetentionPolicy
    seeds: tuple[MatchedReplaySeedOutcome, ...]
    aggregate: base.ContinualShiftAggregate


@dataclass(frozen=True)
class MatchedReplayOutcomeResult:
    protocol_id: str
    manifest: MatchedReplayOutcomeManifest
    manifest_digest: str
    policies: tuple[MatchedReplayPolicyOutcome, ...]


def run_matched_replay_outcomes(
    manifest: MatchedReplayOutcomeManifest,
) -> MatchedReplayOutcomeResult:
    """Train and audit all four fixed trials before any final-label access."""
    _validate_manifest(manifest)
    digest = continual_config_digest(manifest, manifest.seeds)
    schedules = tuple(_schedule_manifest(manifest, policy) for policy in manifest.policies)
    trained = tuple(
        tuple(run_matched_replay_training(bound, seed=seed) for seed in manifest.seeds)
        for bound in schedules
    )
    for bound, trials in zip(schedules, trained, strict=True):
        for seed, trial in zip(manifest.seeds, trials, strict=True):
            _validate_unscored_trial(bound, seed, trial)

    # Why this: neither a failed training audit nor a missing trial may
    # release a final source. Compare all final roles before scoring any.
    released = tuple(
        tuple(
            (release_final_test(trial.pending.phase_a), release_final_test(trial.pending.phase_b))
            for trial in trials
        )
        for trials in trained
    )
    _validate_matched_final_roles(manifest.seeds, released)
    policies = tuple(
        _score_policy(manifest, policy, trials, roles)
        for policy, trials, roles in zip(manifest.policies, trained, released, strict=True)
    )
    return MatchedReplayOutcomeResult(MATCHED_REPLAY_OUTCOME_PROTOCOL, manifest, digest, policies)


def _validate_manifest(manifest: MatchedReplayOutcomeManifest) -> None:
    if (
        type(manifest) is not MatchedReplayOutcomeManifest
        or manifest.protocol_id != MATCHED_REPLAY_OUTCOME_PROTOCOL
        or manifest.sampling_policy != schedule.SHARED_REPLAY_SAMPLER
        or type(manifest.policies) is not tuple
        or len(manifest.policies) != 2
        or any(type(policy) is not ReplayRetentionPolicy for policy in manifest.policies)
        or tuple(policy.name for policy in manifest.policies) != ("recent_fifo", "seeded_reservoir")
    ):
        raise ValueError("matched outcome manifest requires ordered FIFO and reservoir policies")
    try:
        for policy in manifest.policies:
            schedule._validate_manifest(_schedule_manifest(manifest, policy))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "matched outcome manifest has invalid roles, seeds, or replay budget"
        ) from error


def _schedule_manifest(
    manifest: MatchedReplayOutcomeManifest, policy: ReplayRetentionPolicy
) -> schedule.MatchedReplayScheduleManifest:
    return schedule.MatchedReplayScheduleManifest(
        arrived=manifest.arrived,
        seeds=manifest.seeds,
        policy=policy,
        replay_updates_per_sleep=manifest.replay_updates_per_sleep,
        pc_replay_inference_steps=manifest.pc_replay_inference_steps,
        sampling_policy=manifest.sampling_policy,
    )


def _validate_unscored_trial(
    manifest: schedule.MatchedReplayScheduleManifest,
    seed: int,
    trial: MatchedReplayTrainingResult,
    *,
    training_protocol: str = MATCHED_REPLAY_TRAINING_PROTOCOL,
    training_digest: str | None = None,
) -> None:
    pending = trial.pending
    session = schedule.MatchedReplayScheduleSession(manifest, seed=seed)
    if (
        trial.protocol_id != training_protocol
        or trial.manifest_digest != (training_digest or session.manifest_digest)
        or trial.seed != seed
        or pending.seed != seed
        or pending.phase_a.final_released
        or pending.phase_b.final_released
        or pending.phase_a.final_test is not None
        or pending.phase_b.final_test is not None
        or any(event.role == "final_test" for event in pending.audit.accesses)
        or any(
            event.role == "outer_selection"
            and event.action not in {"source_release", "label_release"}
            for event in pending.audit.accesses
        )
    ):
        raise ValueError("matched replay unscored trial identity or final seal differs")
    _validate_development_roles(pending.phase_a, session._roles)
    if len(pending.sleep_events) != (
        manifest.arrived.training.phase_a_epochs + manifest.arrived.training.phase_b_epochs
    ):
        raise ValueError("matched replay sleep history length differs")

    next_boundary = 0
    phase_a_retention = None
    for phase in ("a", "b"):
        if phase == "b":
            session.arrive_phase_b(manifest)
            _validate_development_roles(pending.phase_b, session._roles)
        count = (
            manifest.arrived.training.phase_a_epochs
            if phase == "a"
            else manifest.arrived.training.phase_b_epochs
        )
        offset = 0 if phase == "a" else manifest.arrived.training.phase_a_epochs
        for epoch in range(1, count + 1):
            expected = session.complete_wake_epoch(manifest, source_role="train")
            event = pending.sleep_events[offset + epoch - 1]
            if event.completed_epoch != offset + epoch or event.wake_batches != offset + epoch:
                raise ValueError("matched replay sleep history cursor differs")
            if event.guard is not None and (
                event.guard.role != "inner_guard"
                or event.guard.role_hash != session._roles.split_hashes["inner_guard"]
            ):
                raise ValueError("matched replay guard role differs")
            if expected is None:
                if (
                    event.outcome != "skipped"
                    or event.replay.applied_updates
                    or event.replay.applied_examples
                ):
                    raise ValueError("matched replay work outside a shared boundary")
                continue
            if next_boundary >= len(trial.boundaries):
                raise ValueError("matched replay boundary count differs")
            _validate_applied_boundary(manifest, expected, trial.boundaries[next_boundary], event)
            next_boundary += 1
        if phase == "a":
            phase_a_retention = session.retention
    if next_boundary != len(trial.boundaries):
        raise ValueError("matched replay boundary count differs")
    clock = pending.state.circadian_model.get_sleep_clocks()
    exposure = pending.state.circadian_model.get_replay_exposure()
    applied_ids = {
        sample_id
        for boundary in trial.boundaries
        for work in boundary.applied_by_method
        if work.method == "circadian_predictive_coding"
        for sample_id in work.sample_ids
    }
    if (
        pending.state.circadian_after_a.get_replay_retention() != phase_a_retention
        or pending.state.circadian_model.get_replay_retention() != session.retention
        or clock.wake_batches != len(pending.sleep_events)
        or clock.wake_examples
        != (
            len(pending.phase_a.train.input) * manifest.arrived.training.phase_a_epochs
            + len(pending.phase_b.train.input) * manifest.arrived.training.phase_b_epochs
        )
        or clock.replay_updates
        != sum(event.replay.applied_updates for event in pending.sleep_events)
        or exposure.replay_updates != clock.replay_updates
        or set(exposure.exposed_ids) != applied_ids
    ):
        raise ValueError("matched replay retained state or work clock differs")


def _validate_development_roles(actual: PhaseDecisionRoles, expected: PhaseDecisionRoles) -> None:
    if (
        actual.phase != expected.phase
        or actual.seed != expected.seed
        or any(
            actual.sample_ids[role] != expected.sample_ids[role]
            or actual.split_hashes[role] != expected.split_hashes[role]
            for role in ("train", "inner_guard", "outer_selection")
        )
    ):
        raise ValueError("matched replay arrived development role differs")


def _validate_applied_boundary(
    manifest: schedule.MatchedReplayScheduleManifest,
    planned: schedule.MatchedReplayBoundary,
    actual: MatchedReplayAppliedBoundary,
    event: SleepEventTelemetry,
) -> None:
    if (
        actual.phase != planned.phase
        or actual.epoch != planned.epoch
        or actual.retained_order_ids != planned.retained_order_ids
        or actual.selected_ids != planned.selected_ids
        or actual.sleep_outcome != event.outcome
    ):
        raise ValueError("matched replay retained or selected boundary differs")
    applied_ids = planned.selected_ids if event.outcome == "accepted" else ()
    if event.outcome not in {"accepted", "rolled_back", "skipped"} or (
        event.replay.applied_updates != len(applied_ids)
        or event.replay.applied_examples != len(applied_ids)
        or (event.outcome in {"accepted", "rolled_back"} and event.guard is None)
    ):
        raise ValueError("matched replay sleep applied work differs")
    inference_steps = {
        "backprop": 0,
        "predictive_coding": manifest.pc_replay_inference_steps,
        "circadian_predictive_coding": manifest.arrived.training.circadian_config.replay_inference_steps,
    }
    expected_work = tuple(
        AppliedMethodReplay(
            method,
            applied_ids,
            len(applied_ids),
            len(applied_ids),
            len(applied_ids) * inference_steps[method],
        )
        for method in manifest.arrived.training.model_order
    )
    if actual.applied_by_method != expected_work:
        raise ValueError("matched replay applied replay work differs")


def _validate_matched_final_roles(
    seeds: tuple[int, ...],
    released: tuple[tuple[tuple[PhaseDecisionRoles, PhaseDecisionRoles], ...], ...],
) -> None:
    if len(released) != 2 or any(len(policy) != len(seeds) for policy in released):
        raise ValueError("matched final roles require every policy and seed")
    for first, second in zip(released[0], released[1], strict=True):
        for phase_a_or_b in (0, 1):
            left, right = first[phase_a_or_b], second[phase_a_or_b]
            if (
                left.seed != right.seed
                or left.sample_ids["final_test"] != right.sample_ids["final_test"]
                or left.split_hashes["final_test"] != right.split_hashes["final_test"]
            ):
                raise ValueError("matched final roles differ across policies")


def _score_policy(
    manifest: MatchedReplayOutcomeManifest,
    policy: ReplayRetentionPolicy,
    trials: tuple[MatchedReplayTrainingResult, ...],
    released: tuple[tuple[PhaseDecisionRoles, PhaseDecisionRoles], ...],
) -> MatchedReplayPolicyOutcome:
    scored = tuple(
        MatchedReplaySeedOutcome(
            _score_released_trial(manifest.arrived, trial.pending, bound_a, bound_b),
            trial.manifest_digest,
            trial.boundaries,
            _sum_applied_work(manifest, trial.boundaries),
            trial.pending.state.circadian_model.get_replay_exposure(),
        )
        for trial, (bound_a, bound_b) in zip(trials, released, strict=True)
    )
    return MatchedReplayPolicyOutcome(
        policy,
        scored,
        base.ContinualShiftAggregate(
            run_count=len(scored),
            backprop=base._aggregate_model_stats(
                [item.arrived.metrics.backprop for item in scored]
            ),
            predictive_coding=base._aggregate_model_stats(
                [item.arrived.metrics.predictive_coding for item in scored]
            ),
            circadian_predictive_coding=base._aggregate_circadian_stats(
                [item.arrived.metrics.circadian_predictive_coding for item in scored]
            ),
        ),
    )


def _score_released_trial(
    config: arrived.ContinualArrivedRolesConfig,
    pending: arrived._PendingSeed,
    bound_a: PhaseDecisionRoles,
    bound_b: PhaseDecisionRoles,
) -> arrived.ContinualArrivedSeedResult:
    pending.audit.record_final_release("a")
    pending.audit.record_final_release("b")
    assert bound_a.final_test is not None and bound_b.final_test is not None
    role_hashes = {
        f"phase_{phase}_{role}": digest
        for phase, bound in (("a", bound_a), ("b", bound_b))
        for role, digest in bound.split_hashes.items()
    }
    metrics = base._score_seed_models(
        config.training,
        pending.seed,
        pending.state,
        bound_a.final_test,
        bound_b.final_test,
        role_hashes,
        sleep_events=pending.sleep_events,
    )
    assert isinstance(metrics, base.ContinualBoundedReplaySeedResult)
    role_ids = {
        f"phase_{phase}_{role}": ids
        for phase, bound in (("a", bound_a), ("b", bound_b))
        for role, ids in bound.sample_ids.items()
    }
    return arrived.ContinualArrivedSeedResult(
        pending.seed,
        metrics,
        role_ids,
        role_hashes,
        tuple(pending.audit.accesses),
        tuple(pending.audit.guard_decisions),
        tuple(pending.audit.task_information),
    )


def _sum_applied_work(
    manifest: MatchedReplayOutcomeManifest,
    boundaries: tuple[MatchedReplayAppliedBoundary, ...],
) -> tuple[AppliedMethodReplay, ...]:
    return tuple(
        AppliedMethodReplay(
            method,
            tuple(
                sample_id
                for boundary in boundaries
                for work in boundary.applied_by_method
                if work.method == method
                for sample_id in work.sample_ids
            ),
            sum(
                work.examples
                for boundary in boundaries
                for work in boundary.applied_by_method
                if work.method == method
            ),
            sum(
                work.optimizer_updates
                for boundary in boundaries
                for work in boundary.applied_by_method
                if work.method == method
            ),
            sum(
                work.inference_iterations
                for boundary in boundaries
                for work in boundary.applied_by_method
                if work.method == method
            ),
        )
        for method in manifest.arrived.training.model_order
    )
