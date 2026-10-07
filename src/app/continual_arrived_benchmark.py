"""Ordinary v6 continual run with arrived decision roles and inner guard.

Inputs are one fixed v5 training configuration, four-role split budgets,
and predeclared seeds. Outputs include descriptive metrics, actual role-use
events, guarded sleep decisions, and per-method task information. The
checkpoint route resumes active A/B transactions and completed seeds;
outer setting selection remains separate work.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np

from src.app import continual_shift_benchmark as base
from src.app.continual_arrived_checkpoint import (
    ARRIVED_CHECKPOINT_FORMAT,
    ArrivedCheckpointStore,
    ArrivedRunnerCheckpoint,
    ArrivedUnscoredSeed,
    arrived_config_digest,
    arrived_event_digest,
    arrived_sleep_history_digest,
    validate_arrived_checkpoint_header,
)
from src.app.continual_arrived_sleep_history import validate_arrived_sleep_history
from src.app.continual_shift_benchmark import ContinualGlobalSealConfig
from src.app.numpy_checkpoint_validation import validate_numpy_baseline_model
from src.core.circadian_predictive_coding import (
    CircadianPredictiveCodingNetwork,
    ReplayRetentionBudget,
)
from src.core.sleep_telemetry import SleepEventTelemetry
from src.core.replay_retention import ReplayRetentionPolicy
from src.infra.continual_roles import (
    PhaseDecisionRoles,
    release_final_test,
    split_phase_decision_roles,
)
from src.infra.datasets import generate_two_cluster_dataset_with_transform


ARRIVED_ROLES_PROTOCOL = "continual_arrived_roles_v6"
_PHASE_A_SPLIT_SEED_OFFSET = 17
_PHASE_B_EXPOSURE_SEED_OFFSET = 118
_PHASE_B_SPLIT_SEED_OFFSET = 138


@dataclass(frozen=True)
class ContinualArrivedRolesConfig:
    """One fixed training setting plus disjoint decision-role budgets."""

    training: ContinualGlobalSealConfig
    inner_guard_fraction: float
    outer_selection_fraction: float
    guard_drop_tolerance: float = 0.0
    protocol_id: str = ARRIVED_ROLES_PROTOCOL


@dataclass(frozen=True)
class RoleAccessEvent:
    """An observed runner action and the role whose values it used."""

    seed: int
    phase: str
    role: str
    action: str
    event: str
    method: str | None = None
    epoch: int | None = None


@dataclass(frozen=True)
class GuardDecision:
    """One attempted circadian sleep assessed on the arrived inner role."""

    phase: str
    epoch: int
    role_hash: str
    accuracy_before: float
    accuracy_after: float
    performed: bool
    accepted: bool
    restored: bool


@dataclass(frozen=True)
class MethodTaskInformation:
    """Declared phase settings and data actually arrived at one method update."""

    method: str
    phase: str
    declared_phases: tuple[str, ...]
    arrived_phases: tuple[str, ...]
    train_role: str
    guard_role: str | None


@dataclass(frozen=True)
class ContinualArrivedSeedResult:
    seed: int
    metrics: base.ContinualBoundedReplaySeedResult
    role_ids: dict[str, tuple[str, ...]]
    role_hashes: dict[str, str]
    role_accesses: tuple[RoleAccessEvent, ...]
    guard_decisions: tuple[GuardDecision, ...]
    method_task_information: tuple[MethodTaskInformation, ...]


@dataclass(frozen=True)
class ContinualArrivedBenchmarkResult:
    protocol_id: str
    config: ContinualArrivedRolesConfig
    seeds: tuple[int, ...]
    seed_results: tuple[ContinualArrivedSeedResult, ...]
    aggregate: base.ContinualShiftAggregate


@dataclass(frozen=True)
class _DeferredPhaseSource:
    """Preserve the final source seal after reducing only B training fields."""

    train_input: np.ndarray
    train_target: np.ndarray
    source: Any = field(repr=False)

    @property
    def test_input(self) -> np.ndarray:
        return self.source.test_input

    @property
    def test_target(self) -> np.ndarray:
        return self.source.test_target


@dataclass
class _PendingSeed:
    seed: int
    state: base._ContinualTrainingState
    phase_a: PhaseDecisionRoles
    phase_b: PhaseDecisionRoles
    audit: _RoleAudit
    sleep_events: tuple[SleepEventTelemetry, ...] = ()


@dataclass
class _RoleAudit:
    """Record actions when the runner actually releases or consumes a role."""

    seed: int
    accesses: list[RoleAccessEvent] = field(default_factory=list)
    guard_decisions: list[GuardDecision] = field(default_factory=list)
    task_information: list[MethodTaskInformation] = field(default_factory=list)
    _seen_updates: set[tuple[str, str]] = field(default_factory=set)

    def record_arrival(self, phase: str) -> None:
        for role in ("train", "inner_guard", "outer_selection"):
            for action in ("source_release", "label_release"):
                self.accesses.append(
                    RoleAccessEvent(self.seed, phase, role, action, f"phase_{phase}_arrival")
                )

    def record_update(self, phase: str, method: str, epoch: int) -> None:
        self.accesses.append(
            RoleAccessEvent(self.seed, phase, "train", "train", f"phase_{phase}", method, epoch)
        )
        if (phase, method) in self._seen_updates:
            return
        self._seen_updates.add((phase, method))
        self.task_information.append(
            MethodTaskInformation(
                method=method,
                phase=phase,
                declared_phases=("a", "b"),
                arrived_phases=("a",) if phase == "a" else ("a", "b"),
                train_role=f"phase_{phase}_train",
                guard_role=f"phase_{phase}_inner_guard"
                if method == "circadian_predictive_coding"
                else None,
            )
        )

    def record_guard(
        self,
        phase: str,
        role_hash: str,
        epoch: int,
        before: float,
        after: float,
        performed: bool,
        accepted: bool,
    ) -> None:
        self.accesses.append(
            RoleAccessEvent(
                self.seed,
                phase,
                "inner_guard",
                "guard",
                f"phase_{phase}",
                "circadian_predictive_coding",
                epoch,
            )
        )
        self.guard_decisions.append(
            GuardDecision(phase, epoch, role_hash, before, after, performed, accepted, not accepted)
        )

    def record_final_release(self, phase: str) -> None:
        for action in ("source_release", "label_release"):
            self.accesses.append(
                RoleAccessEvent(self.seed, phase, "final_test", action, "global_freeze")
            )


def run_continual_arrived_benchmark(
    config: ContinualArrivedRolesConfig,
    seeds: list[int],
    *,
    checkpoint_store: ArrivedCheckpointStore | None = None,
    resume_from_checkpoint: bool = False,
    sleep_error_retries: int = 0,
) -> ContinualArrivedBenchmarkResult:
    """Train every seed before any final-source field or label is opened."""
    _validate_arrived_config(config, seeds)
    if type(sleep_error_retries) is not int or sleep_error_retries < 0:
        raise ValueError("v6 sleep_error_retries must be a nonnegative integer")
    if checkpoint_store is not None and sleep_error_retries:
        raise ValueError("checkpointed v6 sleep errors use explicit checkpoint resume")
    if resume_from_checkpoint and checkpoint_store is None:
        raise ValueError("v6 resume requires a checkpoint store")
    pending = (
        _run_completed_seed_checkpoints(config, seeds, checkpoint_store, resume_from_checkpoint)
        if checkpoint_store is not None
        else [_train_arrived_seed(config, seed, sleep_error_retries) for seed in seeds]
    )
    return _build_arrived_result(config, seeds, pending)


def _build_arrived_result(
    config: ContinualArrivedRolesConfig, seeds: list[int], pending: list[_PendingSeed]
) -> ContinualArrivedBenchmarkResult:
    if len(pending) != len(seeds):
        raise ValueError("v6 final-test release requires every configured seed")
    scored = tuple(_score_arrived_seed(config, item) for item in pending)
    return ContinualArrivedBenchmarkResult(
        protocol_id=ARRIVED_ROLES_PROTOCOL,
        config=config,
        seeds=tuple(seeds),
        seed_results=scored,
        aggregate=base.ContinualShiftAggregate(
            run_count=len(scored),
            backprop=base._aggregate_model_stats([item.metrics.backprop for item in scored]),
            predictive_coding=base._aggregate_model_stats(
                [item.metrics.predictive_coding for item in scored]
            ),
            circadian_predictive_coding=base._aggregate_circadian_stats(
                [item.metrics.circadian_predictive_coding for item in scored]
            ),
        ),
    )


def _validate_arrived_config(config: ContinualArrivedRolesConfig, seeds: list[int]) -> None:
    if (
        type(config) is not ContinualArrivedRolesConfig
        or config.protocol_id != ARRIVED_ROLES_PROTOCOL
    ):
        raise ValueError("v6 requires its matching arrived-roles config")
    if type(config.training) is not ContinualGlobalSealConfig:
        raise ValueError("v6 requires one fixed v5 training config")
    base._validate_config(config.training)
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("v6 seeds must be nonempty unique Python integers")
    if (
        any(
            not isinstance(value, Real) or isinstance(value, bool) or not np.isfinite(value)
            for value in (
                config.inner_guard_fraction,
                config.outer_selection_fraction,
                config.guard_drop_tolerance,
            )
        )
        or config.inner_guard_fraction <= 0.0
        or config.outer_selection_fraction <= 0.0
        or config.inner_guard_fraction + config.outer_selection_fraction >= 1.0
        or config.guard_drop_tolerance < 0.0
    ):
        raise ValueError("v6 role fractions or guard tolerance are invalid")


def _expected_final_count(sample_count: int, test_ratio: float) -> int:
    effective_count = 2 * (sample_count // 2)
    return effective_count - int((1.0 - test_ratio) * effective_count)


def _phase_roles(
    source: Any,
    config: ContinualArrivedRolesConfig,
    seed: int,
    phase: str,
    source_rows: tuple[int, ...] | None = None,
) -> PhaseDecisionRoles:
    sample_count = (
        config.training.sample_count_phase_a
        if phase == "a"
        else config.training.sample_count_phase_b
    )
    return split_phase_decision_roles(
        source,
        phase=phase,
        seed=seed,
        split_seed=seed
        + (_PHASE_A_SPLIT_SEED_OFFSET if phase == "a" else _PHASE_B_SPLIT_SEED_OFFSET),
        inner_guard_fraction=config.inner_guard_fraction,
        outer_selection_fraction=config.outer_selection_fraction,
        expected_final_count=_expected_final_count(sample_count, config.training.test_ratio),
        source_row_indices=source_rows,
    )


def _reduce_phase_b_source(
    source: Any, config: ContinualGlobalSealConfig, seed: int
) -> tuple[_DeferredPhaseSource, tuple[int, ...]]:
    """Apply the declared B exposure fraction while retaining source row IDs."""
    targets = source.train_target[:, 0]
    source_count = len(targets)
    count = min(source_count, max(8, int(source_count * config.phase_b_train_fraction)))
    if count == source_count:
        selected = tuple(range(source_count))
    else:
        # Why this: exposure and role assignment use separate streams so
        # changing split fractions does not change which B source rows arrive.
        rng = np.random.default_rng(seed + _PHASE_B_EXPOSURE_SEED_OFFSET)
        negative = np.flatnonzero(targets == 0.0)
        positive = np.flatnonzero(targets == 1.0)
        positive_count = min(len(positive), count // 2)
        negative_count = min(len(negative), count - positive_count)
        if positive_count + negative_count < count:
            positive_count += min(
                len(positive) - positive_count, count - positive_count - negative_count
            )
            negative_count = count - positive_count
        selected = tuple(
            int(index)
            for index in sorted(
                (
                    *rng.choice(positive, size=positive_count, replace=False),
                    *rng.choice(negative, size=negative_count, replace=False),
                )
            )
        )
    rows = np.asarray(selected, dtype=np.int64)
    return _DeferredPhaseSource(
        source.train_input[rows], source.train_target[rows], source
    ), selected


def _build_phase_a_roles(config: ContinualArrivedRolesConfig, seed: int) -> PhaseDecisionRoles:
    training = config.training
    source_a = generate_two_cluster_dataset_with_transform(
        sample_count=training.sample_count_phase_a,
        noise_scale=training.phase_a_noise_scale,
        seed=seed,
        test_ratio=training.test_ratio,
    )
    return _phase_roles(source_a, config, seed, "a")


def _build_phase_b_roles(config: ContinualArrivedRolesConfig, seed: int) -> PhaseDecisionRoles:
    training = config.training
    source_b = _generate_phase_b_source(training, seed + 101)
    reduced_b, source_rows = _reduce_phase_b_source(source_b, training, seed)
    return _phase_roles(reduced_b, config, seed, "b", source_rows)


def _train_arrived_seed(
    config: ContinualArrivedRolesConfig,
    seed: int,
    sleep_error_retries: int = 0,
    *,
    retention_policy: ReplayRetentionPolicy | None = None,
) -> _PendingSeed:
    training = config.training
    phase_a = _build_phase_a_roles(config, seed)
    audit = _RoleAudit(seed)
    sleep_events: list[SleepEventTelemetry] = []
    audit.record_arrival("a")

    state = base._train_phase_a_models(
        config=training,
        seed=seed,
        phase_a_train=phase_a.train,
        phase_a_guard=phase_a.inner_guard,
        guard_drop_tolerance=config.guard_drop_tolerance,
        on_model_update=lambda method, epoch: audit.record_update("a", method, epoch),
        on_guard_decision=lambda epoch, before, after, performed, accepted: audit.record_guard(
            "a", phase_a.split_hashes["inner_guard"], epoch, before, after, performed, accepted
        ),
        on_sleep_event=sleep_events.append,
        guard_role_hash=phase_a.split_hashes["inner_guard"],
        sleep_error_retries=sleep_error_retries,
        retention_policy=retention_policy,
    )
    phase_b = _build_phase_b_roles(config, seed)
    audit.record_arrival("b")
    state = base._train_phase_b_models(
        config=training,
        phase_b_train=phase_b.train,
        phase_b_guard=phase_b.inner_guard,
        state=state,
        guard_drop_tolerance=config.guard_drop_tolerance,
        on_model_update=lambda method, epoch: audit.record_update("b", method, epoch),
        on_guard_decision=lambda epoch, before, after, performed, accepted: audit.record_guard(
            "b", phase_b.split_hashes["inner_guard"], epoch, before, after, performed, accepted
        ),
        on_sleep_event=sleep_events.append,
        guard_role_hash=phase_b.split_hashes["inner_guard"],
        sleep_error_retries=sleep_error_retries,
    )
    return _PendingSeed(seed, state, phase_a, phase_b, audit, tuple(sleep_events))


def _generate_phase_b_source(config: ContinualGlobalSealConfig, seed: int) -> Any:
    return base._generate_phase_b_source(config, seed)


def _development_identity(
    phase_a: PhaseDecisionRoles, phase_b: PhaseDecisionRoles | None = None
) -> tuple[tuple[tuple[str, tuple[str, ...]], ...], tuple[tuple[str, str], ...]]:
    roles = ("train", "inner_guard", "outer_selection")
    arrived = (("a", phase_a),) if phase_b is None else (("a", phase_a), ("b", phase_b))
    role_ids = tuple(
        (f"phase_{phase}_{role}", bound.sample_ids[role])
        for phase, bound in arrived
        for role in roles
    )
    hashes = tuple(
        (f"phase_{phase}_{role}", bound.split_hashes[role])
        for phase, bound in arrived
        for role in roles
    )
    return role_ids, hashes


def _capture_unscored_arrived_seed(pending: _PendingSeed) -> ArrivedUnscoredSeed:
    """Detach only trained models and observed development identity."""
    role_ids, role_hashes = _development_identity(pending.phase_a, pending.phase_b)
    accesses = tuple(pending.audit.accesses)
    decisions = tuple(pending.audit.guard_decisions)
    information = tuple(pending.audit.task_information)
    event_digest = arrived_event_digest(accesses, decisions, information)
    return ArrivedUnscoredSeed(
        seed=pending.seed,
        role_ids=role_ids,
        role_hashes=role_hashes,
        state=deepcopy(pending.state),
        role_accesses=accesses,
        guard_decisions=decisions,
        method_task_information=information,
        event_digest=event_digest,
        sleep_events=pending.sleep_events,
        sleep_history_digest=arrived_sleep_history_digest(pending.sleep_events, event_digest),
    )


def _validate_completed_audit(
    record: ArrivedUnscoredSeed,
    config: ContinualArrivedRolesConfig,
    phase_a: PhaseDecisionRoles,
    phase_b: PhaseDecisionRoles,
) -> None:
    """Rebuild the legal completed cursor from actual saved guard attempts."""
    if (
        not isinstance(record.role_accesses, tuple)
        or not all(isinstance(event, RoleAccessEvent) for event in record.role_accesses)
        or not isinstance(record.guard_decisions, tuple)
        or not all(isinstance(item, GuardDecision) for item in record.guard_decisions)
        or not isinstance(record.method_task_information, tuple)
        or not all(
            isinstance(item, MethodTaskInformation) for item in record.method_task_information
        )
    ):
        raise ValueError("incompatible v6 completed event cursor")
    try:
        digest = arrived_event_digest(
            record.role_accesses, record.guard_decisions, record.method_task_information
        )
    except (TypeError, ValueError) as error:
        raise ValueError("incompatible v6 completed event cursor") from error
    if record.event_digest != digest:
        raise ValueError("incompatible v6 completed event digest")
    decisions: dict[tuple[str, int], GuardDecision] = {}
    for item in record.guard_decisions:
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
            raise ValueError("incompatible v6 completed guard decision")
        decisions[item.phase, item.epoch] = item
    expected = _RoleAudit(record.seed)
    for phase, bound, epochs in (
        ("a", phase_a, config.training.phase_a_epochs),
        ("b", phase_b, config.training.phase_b_epochs),
    ):
        expected.record_arrival(phase)
        for epoch in range(1, epochs + 1):
            for method in config.training.model_order:
                expected.record_update(phase, method, epoch)
            decision = decisions.get((phase, epoch))
            if decision is not None:
                if decision.role_hash != bound.split_hashes["inner_guard"]:
                    raise ValueError("incompatible v6 completed guard role")
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
        len(expected.guard_decisions) != len(record.guard_decisions)
        or tuple(expected.guard_decisions) != record.guard_decisions
        or tuple(expected.accesses) != record.role_accesses
        or tuple(expected.task_information) != record.method_task_information
    ):
        raise ValueError("incompatible v6 completed event cursor")


def _validate_unscored_arrived_seed(
    record: ArrivedUnscoredSeed,
    config: ContinualArrivedRolesConfig,
    seed: int,
    phase_a: PhaseDecisionRoles,
    phase_b: PhaseDecisionRoles,
    retention_policy: ReplayRetentionPolicy | None = None,
) -> None:
    """Check all saved development, replay, model, and event state pre-update."""
    expected_ids, expected_hashes = _development_identity(phase_a, phase_b)
    if (
        type(record) is not ArrivedUnscoredSeed
        or record.seed != seed
        or record.role_ids != expected_ids
        or record.role_hashes != expected_hashes
        or not isinstance(record.state, base._ContinualTrainingState)
    ):
        raise ValueError("incompatible v6 completed development roles")
    _validate_completed_audit(record, config, phase_a, phase_b)
    if vars(record).get("sleep_event_history_version") != 1:
        raise ValueError("incompatible v6 completed sleep history version")
    validate_arrived_sleep_history(
        record.sleep_events,
        record.guard_decisions,
        role_event_digest=record.event_digest,
        history_digest=record.sleep_history_digest,
        phases=(
            (
                "a",
                phase_a,
                0,
                config.training.phase_a_epochs,
                config.training.circadian_sleep_interval_phase_a,
            ),
            (
                "b",
                phase_b,
                config.training.phase_a_epochs,
                config.training.phase_b_epochs,
                config.training.circadian_sleep_interval_phase_b,
            ),
        ),
        tolerance=config.guard_drop_tolerance,
        sleep_mode=config.training.circadian_config.sleep_mode,
    )
    state = record.state
    training = config.training
    hidden_dims = training.hidden_dims or (training.hidden_dim,)
    total_epochs = training.phase_a_epochs + training.phase_b_epochs
    if (
        not isinstance(state.circadian_model, CircadianPredictiveCodingNetwork)
        or not isinstance(state.circadian_after_a, CircadianPredictiveCodingNetwork)
        or any(
            type(value) is not int or value < 0
            for value in (
                state.sleep_event_count,
                state.total_splits,
                state.total_prunes,
                state.hidden_dim_start,
            )
        )
        or state.sleep_event_count > total_epochs
        or state.hidden_dim_start != hidden_dims[-1]
    ):
        raise ValueError("incompatible v6 completed model state")
    validate_numpy_baseline_model(state.backprop_model, hidden_dims, total_epochs)
    validate_numpy_baseline_model(state.predictive_model, hidden_dims, total_epochs)
    validate_numpy_baseline_model(state.backprop_after_a, hidden_dims, training.phase_a_epochs)
    validate_numpy_baseline_model(state.predictive_after_a, hidden_dims, training.phase_a_epochs)
    budget = ReplayRetentionBudget(training.replay_max_examples, training.replay_max_bytes)
    phase_a_ids = base._observed_replay_ids(phase_a.train)
    snapshots = (
        (state.circadian_after_a.snapshot_state(), phase_a_ids, training.phase_a_epochs),
        (
            state.circadian_model.snapshot_state(),
            phase_a_ids | base._observed_replay_ids(phase_b.train),
            total_epochs,
        ),
    )
    for snapshot, observed_ids, expected_wakes in snapshots:
        base._validate_replay_snapshot_provenance(
            snapshot, budget=budget, observed_ids=observed_ids
        )
        candidate = base._new_checkpoint_models(training, seed, retention_policy)[1]
        candidate.restore_state(snapshot)
        if candidate.get_sleep_clocks().wake_batches != expected_wakes:
            raise ValueError("incompatible v6 completed wake progress")


def _rehydrate_unscored_arrived_seed(
    record: ArrivedUnscoredSeed,
    config: ContinualArrivedRolesConfig,
    seed: int,
    retention_policy: ReplayRetentionPolicy | None = None,
) -> _PendingSeed:
    phase_a = _build_phase_a_roles(config, seed)
    phase_b = _build_phase_b_roles(config, seed)
    _validate_unscored_arrived_seed(record, config, seed, phase_a, phase_b, retention_policy)
    audit = _RoleAudit(
        seed,
        accesses=list(record.role_accesses),
        guard_decisions=list(record.guard_decisions),
        task_information=list(record.method_task_information),
    )
    return _PendingSeed(seed, deepcopy(record.state), phase_a, phase_b, audit, record.sleep_events)


def _run_completed_seed_checkpoints(
    config: ContinualArrivedRolesConfig,
    seeds: list[int],
    store: ArrivedCheckpointStore,
    resume: bool,
) -> list[_PendingSeed]:
    """Resume only durable completed seeds; intra-seed recovery is b2."""
    ordered_seeds = tuple(seeds)
    config_digest = arrived_config_digest(config, ordered_seeds)
    checkpoint = store.load() if resume else None
    if checkpoint is not None:
        validate_arrived_checkpoint_header(
            checkpoint,
            config_digest=config_digest,
            seeds=ordered_seeds,
            phase_a_epochs=config.training.phase_a_epochs,
            phase_b_epochs=config.training.phase_b_epochs,
            model_order=config.training.model_order,
        )
    records = list(checkpoint.unscored_seeds) if checkpoint is not None else []
    pending: list[_PendingSeed] = []
    for index, seed in enumerate(seeds):
        if index < len(records):
            pending.append(_rehydrate_unscored_arrived_seed(records[index], config, seed))
            continue
        from src.app.continual_arrived_transactions import train_or_resume_arrived_seed

        active = (
            checkpoint
            if checkpoint is not None
            and checkpoint.phase in {"a", "b"}
            and index == checkpoint.seed_index
            else None
        )
        item = train_or_resume_arrived_seed(
            config,
            seed,
            seed_index=index,
            seeds=ordered_seeds,
            config_digest=config_digest,
            unscored_seeds=tuple(records),
            store=store,
            checkpoint=active,
        )
        records.append(_capture_unscored_arrived_seed(item))
        store.save(
            ArrivedRunnerCheckpoint(
                format_version=ARRIVED_CHECKPOINT_FORMAT,
                config_digest=config_digest,
                seeds=ordered_seeds,
                seed_index=index,
                phase="seed_complete",
                phase_epoch_completed=config.training.phase_b_epochs,
                stage="after_sleep",
                next_model_index=0,
                unscored_seeds=deepcopy(tuple(records)),
            )
        )
        pending.append(item)
    return pending


def _score_arrived_seed(
    config: ContinualArrivedRolesConfig, pending: _PendingSeed
) -> ContinualArrivedSeedResult:
    bound_a = release_final_test(pending.phase_a)
    pending.audit.record_final_release("a")
    bound_b = release_final_test(pending.phase_b)
    pending.audit.record_final_release("b")
    assert bound_a.final_test is not None and bound_b.final_test is not None
    split_hashes = {
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
        split_hashes,
        sleep_events=pending.sleep_events,
    )
    assert isinstance(metrics, base.ContinualBoundedReplaySeedResult)
    role_ids = {
        f"phase_{phase}_{role}": ids
        for phase, bound in (("a", bound_a), ("b", bound_b))
        for role, ids in bound.sample_ids.items()
    }
    return ContinualArrivedSeedResult(
        seed=pending.seed,
        metrics=metrics,
        role_ids=role_ids,
        role_hashes=split_hashes,
        role_accesses=tuple(pending.audit.accesses),
        guard_decisions=tuple(pending.audit.guard_decisions),
        method_task_information=tuple(pending.audit.task_information),
    )
