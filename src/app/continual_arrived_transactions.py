"""Active A/B checkpoint transactions for arrived NumPy continual runs.

Inputs are arrived development roles, a fixed run identity, optional
retention policy, and a v6-shaped active cursor. The default policy keeps
v6 behavior; v8 wraps transaction saves in its own format-9 envelope.
Outputs are unscored trained models and observed access events. This module
does not read checkpoint files or final-test fields.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_shift_benchmark as base
from src.app.circadian_checkpoint import CircadianResumePosition
from src.app.continual_arrived_checkpoint import (
    ARRIVED_CHECKPOINT_FORMAT,
    ArrivedCheckpointStore,
    ArrivedRunnerCheckpoint,
    ArrivedUnscoredSeed,
    arrived_event_digest,
    arrived_sleep_history_digest,
)
from src.app.continual_arrived_sleep_history import validate_arrived_sleep_history
from src.app.continual_checkpoint import ContinualRunnerState
from src.app.numpy_checkpoint_validation import validate_numpy_baseline_model
from src.core.circadian_predictive_coding import (
    CircadianNetworkSnapshot,
    CircadianPredictiveCodingNetwork,
    ReplayRetentionBudget,
)
from src.infra.continual_roles import PhaseDecisionRoles
from src.core.sleep_telemetry import SleepEventTelemetry
from src.core.replay_retention import ReplayRetentionPolicy


@dataclass(frozen=True)
class _TransactionContext:
    config: arrived.ContinualArrivedRolesConfig
    seed: int
    seed_index: int
    seeds: tuple[int, ...]
    config_digest: str
    unscored_seeds: tuple[ArrivedUnscoredSeed, ...]
    store: ArrivedCheckpointStore
    retention_policy: ReplayRetentionPolicy | None = None


@dataclass
class _ActiveSeed:
    state: ContinualRunnerState
    circadian: CircadianPredictiveCodingNetwork
    phase_a: PhaseDecisionRoles
    phase_b: PhaseDecisionRoles | None
    audit: arrived._RoleAudit
    sleep_events: list[SleepEventTelemetry] = field(default_factory=list)


def train_or_resume_arrived_seed(
    config: arrived.ContinualArrivedRolesConfig,
    seed: int,
    *,
    seed_index: int,
    seeds: tuple[int, ...],
    config_digest: str,
    unscored_seeds: tuple[ArrivedUnscoredSeed, ...],
    store: ArrivedCheckpointStore,
    checkpoint: ArrivedRunnerCheckpoint | None,
    retention_policy: ReplayRetentionPolicy | None = None,
) -> arrived._PendingSeed:
    """Save each model and sleep transaction, then return an unscored seed."""
    context = _TransactionContext(
        config, seed, seed_index, seeds, config_digest, unscored_seeds, store, retention_policy
    )
    active = _restore_active_seed(context, checkpoint) if checkpoint else _new_active_seed(context)
    if checkpoint is None or checkpoint.phase == "a":
        _train_phase(context, active, "a", checkpoint)
        active.state.backprop_after_a = deepcopy(active.state.backprop_model)
        active.state.predictive_after_a = deepcopy(active.state.predictive_model)
        active.state.circadian_after_a = active.circadian.snapshot_state()
        active.phase_b = arrived._build_phase_b_roles(config, seed)
        active.audit.record_arrival("b")
        _save_active(context, active, "b", 0, "after_sleep", 0)
    _train_phase(
        context, active, "b", checkpoint if checkpoint and checkpoint.phase == "b" else None
    )
    if active.phase_b is None:
        raise ValueError("v6 Phase B completion requires arrived roles")
    return arrived._PendingSeed(
        seed,
        _completed_training_state(context, active),
        active.phase_a,
        active.phase_b,
        active.audit,
        tuple(active.sleep_events),
    )


def _new_active_seed(context: _TransactionContext) -> _ActiveSeed:
    state, circadian = base._new_checkpoint_models(
        context.config.training, context.seed, context.retention_policy
    )
    phase_a = arrived._build_phase_a_roles(context.config, context.seed)
    audit = arrived._RoleAudit(context.seed)
    audit.record_arrival("a")
    return _ActiveSeed(state, circadian, phase_a, None, audit)


def _restore_active_seed(
    context: _TransactionContext, checkpoint: ArrivedRunnerCheckpoint
) -> _ActiveSeed:
    phase_a = arrived._build_phase_a_roles(context.config, context.seed)
    phase_b = (
        arrived._build_phase_b_roles(context.config, context.seed)
        if checkpoint.phase == "b"
        else None
    )
    _validate_active_checkpoint(context, checkpoint, phase_a, phase_b)
    assert checkpoint.active_state is not None
    assert checkpoint.active_circadian is not None
    state = deepcopy(checkpoint.active_state)
    circadian = base._new_checkpoint_models(
        context.config.training, context.seed, context.retention_policy
    )[1]
    circadian.restore_state(checkpoint.active_circadian)
    information = list(checkpoint.active_task_information)
    audit = arrived._RoleAudit(
        context.seed,
        accesses=list(checkpoint.active_role_accesses),
        guard_decisions=list(checkpoint.active_guard_decisions),
        task_information=information,
        _seen_updates={(item.phase, item.method) for item in information},
    )
    return _ActiveSeed(
        state, circadian, phase_a, phase_b, audit, list(checkpoint.active_sleep_events)
    )


def _train_phase(
    context: _TransactionContext,
    active: _ActiveSeed,
    phase: str,
    checkpoint: ArrivedRunnerCheckpoint | None,
) -> None:
    training = context.config.training
    roles = active.phase_a if phase == "a" else active.phase_b
    if roles is None:
        raise ValueError("v6 Phase B training requires arrived roles")
    epoch_count = training.phase_a_epochs if phase == "a" else training.phase_b_epochs
    offset = 0 if phase == "a" else training.phase_a_epochs
    interval = (
        training.circadian_sleep_interval_phase_a
        if phase == "a"
        else training.circadian_sleep_interval_phase_b
    )
    start_epoch, start_model = _resume_start(checkpoint, offset, len(training.model_order))
    for epoch in range(start_epoch, epoch_count + 1):
        model_start = start_model if epoch == start_epoch else 0
        for model_index in range(model_start, len(training.model_order)):
            method = training.model_order[model_index]
            base._train_named_model_epoch(
                training,
                roles.train,
                active.state.backprop_model,
                active.state.predictive_model,
                active.circadian,
                method,
            )
            active.audit.record_update(phase, method, epoch)
            if model_index + 1 < len(training.model_order):
                _save_active(context, active, phase, epoch - 1, "wake", model_index + 1)
        if model_start < len(training.model_order):
            _save_active(context, active, phase, epoch, "before_sleep", 0)
        try:
            (
                active.state.sleep_event_count,
                active.state.total_splits,
                active.state.total_prunes,
            ) = base._apply_scheduled_sleep(
                model=active.circadian,
                sleep_interval=interval,
                epoch_index=epoch,
                global_epoch=offset + epoch,
                total_epochs=(training.phase_a_epochs if phase == "a" else offset + epoch_count),
                force_sleep=training.circadian_force_sleep,
                sleep_event_count=active.state.sleep_event_count,
                total_splits=active.state.total_splits,
                total_prunes=active.state.total_prunes,
                guard=roles.inner_guard,
                guard_drop_tolerance=context.config.guard_drop_tolerance,
                on_guard_decision=lambda local, before, after, performed, accepted: (
                    active.audit.record_guard(
                        phase,
                        roles.split_hashes["inner_guard"],
                        local,
                        before,
                        after,
                        performed,
                        accepted,
                    )
                ),
                on_sleep_event=active.sleep_events.append,
                guard_role_hash=roles.split_hashes["inner_guard"],
            )
        except Exception:
            # Why this: keep the failed attempt while the before-sleep model
            # cursor remains retryable and no guard decision was committed.
            if (
                active.sleep_events
                and active.sleep_events[-1].completed_epoch == offset + epoch
                and active.sleep_events[-1].outcome == "error"
            ):
                _save_active(context, active, phase, epoch, "before_sleep", 0)
            raise
        _save_active(context, active, phase, epoch, "after_sleep", 0)


def _resume_start(
    checkpoint: ArrivedRunnerCheckpoint | None, offset: int, model_count: int
) -> tuple[int, int]:
    if checkpoint is None:
        return 1, 0
    local = checkpoint.phase_epoch_completed
    if checkpoint.stage == "wake":
        return local + 1, checkpoint.next_model_index
    if checkpoint.stage == "before_sleep":
        return local, model_count
    return local + 1, 0


def _save_active(
    context: _TransactionContext,
    active: _ActiveSeed,
    phase: str,
    local_epoch: int,
    stage: str,
    next_model_index: int,
) -> None:
    role_ids, role_hashes = arrived._development_identity(active.phase_a, active.phase_b)
    accesses = tuple(active.audit.accesses)
    decisions = tuple(active.audit.guard_decisions)
    information = tuple(active.audit.task_information)
    event_digest = arrived_event_digest(accesses, decisions, information)
    sleep_events = tuple(active.sleep_events)
    offset = 0 if phase == "a" else context.config.training.phase_a_epochs
    position = CircadianResumePosition(
        completed_epoch=offset + local_epoch,
        stage=stage,
        wake_batches=active.circadian.get_sleep_clocks().wake_batches,
        next_batch_index=next_model_index,
    )
    context.store.save(
        ArrivedRunnerCheckpoint(
            format_version=ARRIVED_CHECKPOINT_FORMAT,
            config_digest=context.config_digest,
            seeds=context.seeds,
            seed_index=context.seed_index,
            phase=phase,
            phase_epoch_completed=local_epoch,
            stage=stage,
            next_model_index=next_model_index,
            unscored_seeds=deepcopy(context.unscored_seeds),
            active_role_ids=role_ids,
            active_role_hashes=role_hashes,
            active_role_accesses=accesses,
            active_guard_decisions=decisions,
            active_task_information=information,
            active_event_digest=event_digest,
            active_sleep_events=sleep_events,
            active_sleep_history_digest=arrived_sleep_history_digest(sleep_events, event_digest),
            active_state=deepcopy(active.state),
            active_circadian=active.circadian.snapshot_state(),
            active_position=position,
        )
    )


def _completed_training_state(
    context: _TransactionContext, active: _ActiveSeed
) -> base._ContinualTrainingState:
    state = active.state
    if (
        state.backprop_after_a is None
        or state.predictive_after_a is None
        or state.circadian_after_a is None
    ):
        raise ValueError("v6 completed seed needs frozen Phase A models")
    circadian_after_a = base._new_checkpoint_models(
        context.config.training, context.seed, context.retention_policy
    )[1]
    circadian_after_a.restore_state(state.circadian_after_a)
    return base._ContinualTrainingState(
        backprop_model=state.backprop_model,
        predictive_model=state.predictive_model,
        circadian_model=active.circadian,
        backprop_after_a=state.backprop_after_a,
        predictive_after_a=state.predictive_after_a,
        circadian_after_a=circadian_after_a,
        sleep_event_count=state.sleep_event_count,
        total_splits=state.total_splits,
        total_prunes=state.total_prunes,
        hidden_dim_start=state.hidden_dim_start,
    )


def _validate_active_checkpoint(
    context: _TransactionContext,
    checkpoint: ArrivedRunnerCheckpoint,
    phase_a: PhaseDecisionRoles,
    phase_b: PhaseDecisionRoles | None,
) -> None:
    role_ids, role_hashes = arrived._development_identity(phase_a, phase_b)
    if checkpoint.active_role_ids != role_ids or checkpoint.active_role_hashes != role_hashes:
        raise ValueError("incompatible v6 active development roles")
    _validate_active_audit(context, checkpoint, phase_a, phase_b)
    training = context.config.training
    phases = [
        (
            "a",
            phase_a,
            0,
            _completed_phase_progress(checkpoint, "a", training.phase_a_epochs)[1],
            training.circadian_sleep_interval_phase_a,
        )
    ]
    if phase_b is not None:
        phases.append(
            (
                "b",
                phase_b,
                training.phase_a_epochs,
                _completed_phase_progress(checkpoint, "b", training.phase_b_epochs)[1],
                training.circadian_sleep_interval_phase_b,
            )
        )
    validate_arrived_sleep_history(
        checkpoint.active_sleep_events,
        checkpoint.active_guard_decisions,
        role_event_digest=checkpoint.active_event_digest,
        history_digest=checkpoint.active_sleep_history_digest,
        phases=tuple(phases),
        tolerance=context.config.guard_drop_tolerance,
        sleep_mode=training.circadian_config.sleep_mode,
        pending_epoch=(checkpoint.phase, checkpoint.phase_epoch_completed)
        if checkpoint.stage == "before_sleep"
        else None,
    )
    _validate_active_models(context, checkpoint, phase_a, phase_b)


def _validate_active_audit(
    context: _TransactionContext,
    checkpoint: ArrivedRunnerCheckpoint,
    phase_a: PhaseDecisionRoles,
    phase_b: PhaseDecisionRoles | None,
) -> None:
    if (
        not isinstance(checkpoint.active_role_accesses, tuple)
        or not all(
            isinstance(item, arrived.RoleAccessEvent) for item in checkpoint.active_role_accesses
        )
        or not isinstance(checkpoint.active_guard_decisions, tuple)
        or not all(
            isinstance(item, arrived.GuardDecision) for item in checkpoint.active_guard_decisions
        )
        or not isinstance(checkpoint.active_task_information, tuple)
        or not all(
            isinstance(item, arrived.MethodTaskInformation)
            for item in checkpoint.active_task_information
        )
    ):
        raise ValueError("incompatible v6 active event cursor")
    try:
        digest = arrived_event_digest(
            checkpoint.active_role_accesses,
            checkpoint.active_guard_decisions,
            checkpoint.active_task_information,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("incompatible v6 active event cursor") from error
    if checkpoint.active_event_digest != digest:
        raise ValueError("incompatible v6 active event digest")
    decisions: dict[tuple[str, int], arrived.GuardDecision] = {}
    for item in checkpoint.active_guard_decisions:
        if (
            item.phase not in {"a", "b"}
            or type(item.epoch) is not int
            or type(item.performed) is not bool
            or type(item.accepted) is not bool
            or type(item.restored) is not bool
            or item.restored == item.accepted
            or not np.isfinite(item.accuracy_before)
            or not np.isfinite(item.accuracy_after)
            or not 0.0 <= item.accuracy_before <= 1.0
            or not 0.0 <= item.accuracy_after <= 1.0
            or (item.phase, item.epoch) in decisions
        ):
            raise ValueError("incompatible v6 active guard cursor")
        decisions[item.phase, item.epoch] = item
    expected = arrived._RoleAudit(context.seed)
    phase_specs: list[tuple[str, PhaseDecisionRoles, int]] = [
        ("a", phase_a, context.config.training.phase_a_epochs)
    ]
    if phase_b is not None:
        phase_specs.append(("b", phase_b, context.config.training.phase_b_epochs))
    for phase, roles, epoch_count in phase_specs:
        expected.record_arrival(phase)
        train_epochs, sleep_epochs = _completed_phase_progress(checkpoint, phase, epoch_count)
        for epoch in range(1, train_epochs + 1):
            model_count = (
                checkpoint.next_model_index
                if phase == checkpoint.phase
                and checkpoint.stage == "wake"
                and epoch == train_epochs
                else len(context.config.training.model_order)
            )
            for method in context.config.training.model_order[:model_count]:
                expected.record_update(phase, method, epoch)
            decision = decisions.get((phase, epoch)) if epoch <= sleep_epochs else None
            if decision is not None:
                if decision.role_hash != roles.split_hashes["inner_guard"]:
                    raise ValueError("incompatible v6 active guard role")
                expected.record_guard(
                    phase,
                    decision.role_hash,
                    epoch,
                    decision.accuracy_before,
                    decision.accuracy_after,
                    decision.performed,
                    decision.accepted,
                )
    if (
        len(expected.guard_decisions) != len(checkpoint.active_guard_decisions)
        or tuple(expected.guard_decisions) != checkpoint.active_guard_decisions
        or tuple(expected.accesses) != checkpoint.active_role_accesses
        or tuple(expected.task_information) != checkpoint.active_task_information
    ):
        raise ValueError("incompatible v6 active event cursor")


def _completed_phase_progress(
    checkpoint: ArrivedRunnerCheckpoint, phase: str, epoch_count: int
) -> tuple[int, int]:
    if phase != checkpoint.phase:
        return epoch_count, epoch_count
    local = checkpoint.phase_epoch_completed
    if checkpoint.stage == "wake":
        return local + 1, local
    if checkpoint.stage == "before_sleep":
        return local, local - 1
    return local, local


def _validate_active_models(
    context: _TransactionContext,
    checkpoint: ArrivedRunnerCheckpoint,
    phase_a: PhaseDecisionRoles,
    phase_b: PhaseDecisionRoles | None,
) -> None:
    state = checkpoint.active_state
    snapshot = checkpoint.active_circadian
    position = checkpoint.active_position
    assert state is not None and snapshot is not None and position is not None
    training = context.config.training
    hidden_dims = training.hidden_dims or (training.hidden_dim,)
    offset = 0 if checkpoint.phase == "a" else training.phase_a_epochs
    prefix = (
        training.model_order[: checkpoint.next_model_index] if checkpoint.stage == "wake" else ()
    )
    completed = offset + checkpoint.phase_epoch_completed
    expected_steps = {method: completed + int(method in prefix) for method in training.model_order}
    validate_numpy_baseline_model(state.backprop_model, hidden_dims, expected_steps["backprop"])
    validate_numpy_baseline_model(
        state.predictive_model, hidden_dims, expected_steps["predictive_coding"]
    )
    if (
        any(
            type(value) is not int or value < 0
            for value in (
                state.sleep_event_count,
                state.total_splits,
                state.total_prunes,
                state.hidden_dim_start,
            )
        )
        or state.hidden_dim_start != hidden_dims[-1]
        or position.wake_batches != expected_steps["circadian_predictive_coding"]
        or state.sleep_event_count > len(checkpoint.active_guard_decisions)
        or state.sleep_event_count
        != sum(item.performed and item.accepted for item in checkpoint.active_guard_decisions)
    ):
        raise ValueError("incompatible v6 active model progress")
    budget = ReplayRetentionBudget(training.replay_max_examples, training.replay_max_bytes)
    phase_a_ids = base._observed_replay_ids(phase_a.train)
    observed_ids = set(phase_a_ids)
    if phase_b is not None:
        observed_ids.update(base._observed_replay_ids(phase_b.train))
    base._validate_replay_snapshot_provenance(snapshot, budget=budget, observed_ids=observed_ids)
    candidate = base._new_checkpoint_models(training, context.seed, context.retention_policy)[1]
    candidate.restore_state(snapshot)
    if candidate.get_sleep_clocks().wake_batches != position.wake_batches:
        raise ValueError("incompatible v6 active circadian wake progress")
    if checkpoint.phase == "a":
        if any(
            frozen is not None
            for frozen in (
                state.backprop_after_a,
                state.predictive_after_a,
                state.circadian_after_a,
            )
        ):
            raise ValueError("incompatible v6 premature Phase A frozen state")
        return
    if (
        state.backprop_after_a is None
        or state.predictive_after_a is None
        or not isinstance(state.circadian_after_a, CircadianNetworkSnapshot)
    ):
        raise ValueError("incompatible v6 missing Phase A frozen state")
    validate_numpy_baseline_model(state.backprop_after_a, hidden_dims, training.phase_a_epochs)
    validate_numpy_baseline_model(state.predictive_after_a, hidden_dims, training.phase_a_epochs)
    base._validate_replay_snapshot_provenance(
        state.circadian_after_a, budget=budget, observed_ids=phase_a_ids
    )
    frozen = base._new_checkpoint_models(training, context.seed, context.retention_policy)[1]
    frozen.restore_state(state.circadian_after_a)
    if frozen.get_sleep_clocks().wake_batches != training.phase_a_epochs:
        raise ValueError("incompatible v6 frozen Phase A wake progress")
