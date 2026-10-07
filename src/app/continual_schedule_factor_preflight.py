"""Train fixed-width schedule controls with guard-committed matched replay.

Inputs are a frozen manifest and arrived train/inner roles. Outputs are all
unscored decisions, role identities, replay costs and parameter facts. This
app does not read outer/final values, choose settings, or write artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from typing import Any

import numpy as np

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_shift_benchmark as base
from src.app.continual_replay_factor_pilot import (
    Model,
    PARAMETER_FIELDS,
    _circadian_config,
    _parameter_count,
    _parameter_hash,
    fixed_replay_pilot_manifest,
)
from src.app.continual_schedule_factor_validation import verify_schedule_preflight_payload
from src.app.sleep_schedule import decide_sleep_attempt
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
    ReplayRetentionBudget,
    replay_sample_id,
)
from src.core.predictive_coding import PredictiveCodingNetwork
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.shared_replay_schedule import SharedReplayBuffer, SharedReplaySelection
from src.core.sleep_telemetry import SleepEventTelemetry
from src.infra.continual_roles import PhaseDecisionRoles


PROTOCOL_ID = "continual_schedule_factor_train_only_v1"
DEVELOPMENT_SEEDS = (79, 83, 89)
CONFIRMATION_SEEDS = (199, 211, 223, 227, 229, 233, 239, 241, 251, 257)
POLICIES = ("periodic", "adaptive", "no_sleep")
METHODS = ("backprop", "pc", "neutral")
ARMS = tuple(f"{method}_{policy}" for policy in POLICIES for method in METHODS) + (
    "backprop_12_no_sleep",
    "pc_12_no_sleep",
)


@dataclass(frozen=True)
class ScheduleFactorManifest:
    source: arrived.ContinualArrivedRolesConfig
    seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    policies: tuple[str, ...]
    arms: tuple[str, ...]
    configs: dict[str, CircadianConfig]
    width: int = 8
    planned_width: int = 12
    model_seed_offset: int = 1001
    periodic_interval: int = 4
    replay_updates_per_attempt: int = 2
    memory_examples: int = 8
    memory_bytes: int = 192
    max_optimizer_updates: int = 1100
    max_process_rss_bytes: int = 256 * 1024 * 1024
    protocol_id: str = PROTOCOL_ID


@dataclass(frozen=True)
class ScheduleDecisionFacts:
    policy: str
    energy_window: tuple[float, ...]
    chemical_variance: float
    spacing_before: int
    spacing_after: int
    periodic_due: bool
    adaptive_due: bool
    attempted: bool
    force_sleep: bool
    outcome: str
    reason: str
    trigger_reason: str
    guard_role_hash: str | None
    guard_pre_accuracy: float | None
    guard_post_accuracy: float | None
    proposed_replay_ids: tuple[str, ...]
    proposed_replay_updates: int
    proposed_replay_examples: int
    applied_ids_by_method: dict[str, tuple[str, ...]]
    parameter_sha256_before: str
    parameter_sha256_after: str


@dataclass(frozen=True)
class ScheduleOpportunityFacts:
    phase: str
    epoch: int
    global_epoch: int
    train_role_hash: str
    retained_order_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    retained_examples: int
    retained_bytes: int
    decisions: tuple[ScheduleDecisionFacts, ...]


@dataclass(frozen=True)
class ScheduleMethodFacts:
    name: str
    initial_parameter_sha256: str
    after_a_parameter_sha256: str
    final_parameter_sha256: str
    width_initial: int
    width_final: int
    width_peak: int
    parameters_initial: int
    parameters_final: int
    parameters_peak: int
    wake_updates: int
    wake_presentations: int
    wake_inference_loops: int
    wake_example_inference_iterations: int
    applied_replay_updates: int
    applied_replay_presentations: int
    applied_replay_inference_loops: int
    rejected_executed_replay_updates: int
    rejected_replay_inference_loops: int
    controller_accepted_events: int
    sleep_attempts: int
    guard_evaluations: int
    retained_examples: int
    retained_array_bytes: int
    replay_memory_access: str


@dataclass(frozen=True)
class ScheduleSeedFacts:
    seed: int
    role_hashes: dict[str, str]
    role_counts: dict[str, int]
    opportunities: tuple[ScheduleOpportunityFacts, ...]
    methods: tuple[ScheduleMethodFacts, ...]
    executed_optimizer_updates: int
    final_released: bool = False


@dataclass(frozen=True)
class ScheduleFactorPreflight:
    protocol_id: str
    manifest: ScheduleFactorManifest
    seed_results: tuple[ScheduleSeedFacts, ...]
    final_released: bool = False
    outer_selection_scored: bool = False


def fixed_schedule_factor_manifest() -> ScheduleFactorManifest:
    source = fixed_replay_pilot_manifest().source
    configs = {
        policy: replace(_circadian_config(True), use_adaptive_sleep_trigger=policy == "adaptive")
        for policy in POLICIES
    }
    return ScheduleFactorManifest(
        source, DEVELOPMENT_SEEDS, CONFIRMATION_SEEDS, POLICIES, ARMS, configs
    )


def validate_schedule_factor_manifest(manifest: ScheduleFactorManifest) -> int:
    if type(manifest) is not ScheduleFactorManifest or manifest != fixed_schedule_factor_manifest():
        raise ValueError("P6.3 schedule factor requires its frozen manifest")
    if set(manifest.seeds) & set(manifest.confirmation_seeds):
        raise ValueError("P6.3 schedule development and confirmation seeds overlap")
    arrived._validate_arrived_config(manifest.source, list(manifest.seeds))
    total_epochs = manifest.source.training.phase_a_epochs + manifest.source.training.phase_b_epochs
    adaptive = manifest.configs["adaptive"]
    periodic_attempts = total_epochs // manifest.periodic_interval
    adaptive_attempts = (
        total_epochs - max(adaptive.min_epochs_between_sleep, adaptive.sleep_energy_window) + 1
    )
    adaptive_commits = total_epochs // adaptive.min_epochs_between_sleep
    replay_bound = manifest.replay_updates_per_attempt * (
        3 * periodic_attempts + adaptive_attempts + 2 * adaptive_commits
    )
    planned = len(manifest.seeds) * (len(manifest.arms) * total_epochs + replay_bound)
    if planned > manifest.max_optimizer_updates:
        raise ValueError("P6.3 schedule factor exceeds executed optimizer cap")
    return planned


def run_schedule_factor_preflight(manifest: ScheduleFactorManifest) -> ScheduleFactorPreflight:
    validate_schedule_factor_manifest(manifest)
    result = ScheduleFactorPreflight(
        PROTOCOL_ID, manifest, tuple(_run_seed(manifest, seed) for seed in manifest.seeds)
    )
    verify_schedule_preflight_payload(_json_value(result), _json_value(manifest))
    return result


def _json_value(value: ScheduleFactorManifest | ScheduleFactorPreflight) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(value), allow_nan=False))


def _new_models(manifest: ScheduleFactorManifest, seed: int) -> dict[str, Model]:
    models: dict[str, Model] = {}
    model_seed = seed + manifest.model_seed_offset
    for policy in manifest.policies:
        models[f"backprop_{policy}"] = BackpropMLP(2, manifest.width, model_seed)
        models[f"pc_{policy}"] = PredictiveCodingNetwork(2, manifest.width, model_seed)
        neutral = CircadianPredictiveCodingNetwork(
            2,
            manifest.width,
            model_seed,
            manifest.configs[policy],
            min_hidden_dim=manifest.width,
            max_hidden_dim=manifest.width,
        )
        neutral.configure_replay_retention(
            ReplayRetentionBudget(manifest.memory_examples, manifest.memory_bytes),
            policy=ReplayRetentionPolicy("recent_fifo"),
        )
        models[f"neutral_{policy}"] = neutral
    models["backprop_12_no_sleep"] = BackpropMLP(2, manifest.planned_width, model_seed)
    models["pc_12_no_sleep"] = PredictiveCodingNetwork(2, manifest.planned_width, model_seed)
    if len({_parameter_hash(models[name]) for name in manifest.arms[:9]}) != 1:
        raise ValueError("P6.3 schedule width-eight initial parameters differ")
    if _parameter_hash(models[manifest.arms[-1]]) != _parameter_hash(models[manifest.arms[-2]]):
        raise ValueError("P6.3 schedule planned-width initial parameters differ")
    return models


def _require_parity(models: dict[str, Model], manifest: ScheduleFactorManifest) -> None:
    for policy in manifest.policies:
        if any(
            not np.array_equal(
                getattr(models[f"pc_{policy}"], field), getattr(models[f"neutral_{policy}"], field)
            )
            for field in PARAMETER_FIELDS
        ):
            raise ValueError(f"P6.3 schedule neutral PC parity differs: {policy}")


def _train_wake(
    models: dict[str, Model], roles: PhaseDecisionRoles, manifest: ScheduleFactorManifest
) -> None:
    training = manifest.source.training
    for name in manifest.arms:
        model = models[name]
        if isinstance(model, BackpropMLP):
            model.train_epoch(
                roles.train.input, roles.train.target, training.backprop_learning_rate
            )
        else:
            model.train_epoch(
                roles.train.input,
                roles.train.target,
                training.pc_learning_rate,
                training.pc_inference_steps,
                training.pc_inference_learning_rate,
            )
    _require_parity(models, manifest)


def _check_supply(
    model: CircadianPredictiveCodingNetwork,
    shared: SharedReplayBuffer,
    selected: SharedReplaySelection,
) -> None:
    if (
        model.get_replay_retention() != shared.retention
        or model.get_replay_retained_order_ids() != shared.retained_order_ids
        or model.preview_unprioritized_replay_ids(len(selected.sample_ids)) != selected.sample_ids
        or tuple(replay_sample_id(x, y) for x, y in selected.training_batches())
        != selected.sample_ids
    ):
        raise ValueError("P6.3 schedule model/shared replay supply differs")


def _baseline_replay(
    models: dict[str, Model], policy: str, selected: SharedReplaySelection, config: CircadianConfig
) -> None:
    for method in ("backprop", "pc"):
        model = models[f"{method}_{policy}"]
        for inputs, targets in selected.training_batches():
            if isinstance(model, BackpropMLP):
                model.train_epoch(inputs, targets, config.replay_learning_rate)
            else:
                model.train_epoch(
                    inputs,
                    targets,
                    config.replay_learning_rate,
                    config.replay_inference_steps,
                    config.replay_inference_learning_rate,
                )


def _decide_and_apply(
    models: dict[str, Model],
    roles: PhaseDecisionRoles,
    manifest: ScheduleFactorManifest,
    policy: str,
    epoch: int,
    global_epoch: int,
    shared: SharedReplayBuffer,
    selected: SharedReplaySelection,
) -> ScheduleDecisionFacts:
    model = models[f"neutral_{policy}"]
    assert isinstance(model, CircadianPredictiveCodingNetwork)
    _check_supply(model, shared, selected)
    config = model.config
    history = tuple(model._energy_history[-config.sleep_energy_window :])
    variance = float(np.var(model.get_chemical_state()))
    before = model.get_sleep_clocks()
    parameter_before = _parameter_hash(model)
    interval = manifest.periodic_interval if policy == "periodic" else 0
    decision = decide_sleep_attempt(
        sleep_mode=config.sleep_mode,
        completed_epochs=epoch,
        interval_epochs=interval,
        adaptive_due=model.should_trigger_sleep(),
        force_periodic=policy == "periodic",
    )
    events: list[SleepEventTelemetry] = []
    base._apply_scheduled_sleep(
        model,
        interval,
        epoch,
        global_epoch,
        24,
        policy == "periodic",
        before.sleep_events,
        0,
        0,
        guard=roles.inner_guard,
        guard_drop_tolerance=manifest.source.guard_drop_tolerance,
        on_sleep_event=events.append,
        guard_role_hash=roles.split_hashes["inner_guard"],
    )
    if len(events) != 1:
        raise ValueError("P6.3 schedule expected one event per opportunity")
    event = events[0]
    committed = event.outcome == "accepted"
    attempted_rows = manifest.replay_updates_per_attempt if decision.attempted else 0
    if (
        event.outcome not in {"accepted", "rolled_back", "skipped"}
        or event.replay.proposed_updates != attempted_rows
        or event.replay.proposed_examples != attempted_rows
        or event.replay.applied_updates != (attempted_rows if committed else 0)
        or event.changes.proposed_split_pairs
        or event.changes.proposed_removed_prune_ids
        or model.hidden_dim != manifest.width
    ):
        raise ValueError("P6.3 schedule sleep proposal or fixed capacity differs")
    if not committed and _parameter_hash(model) != parameter_before:
        raise ValueError("P6.3 schedule rejected/skipped sleep changed parameters")
    if committed:
        _baseline_replay(models, policy, selected, config)
    _require_parity(models, manifest)
    _check_supply(model, shared, selected)
    after = model.get_sleep_clocks()
    if (after.wake_batches, after.wake_examples) != (
        before.wake_batches,
        before.wake_examples,
    ) or after.replay_updates - before.replay_updates != event.replay.applied_updates:
        raise ValueError("P6.3 schedule sleep altered wake or applied replay clocks")
    guard = event.guard
    return ScheduleDecisionFacts(
        policy,
        history,
        variance,
        before.wake_batches_since_sleep,
        after.wake_batches_since_sleep,
        decision.periodic_due,
        decision.adaptive_due,
        decision.attempted,
        decision.force_sleep,
        event.outcome,
        event.reason,
        event.trigger_reason,
        guard.role_hash if guard else None,
        guard.pre_accuracy if guard else None,
        guard.post_accuracy if guard else None,
        selected.sample_ids if attempted_rows else (),
        event.replay.proposed_updates,
        event.replay.proposed_examples,
        {method: selected.sample_ids if committed else () for method in METHODS},
        parameter_before,
        _parameter_hash(model),
    )


def _train_phase(
    models: dict[str, Model],
    roles: PhaseDecisionRoles,
    manifest: ScheduleFactorManifest,
    shared: SharedReplayBuffer,
) -> tuple[ScheduleOpportunityFacts, ...]:
    phase = roles.phase
    offset = 0 if phase == "a" else manifest.source.training.phase_a_epochs
    count = (
        manifest.source.training.phase_a_epochs
        if phase == "a"
        else manifest.source.training.phase_b_epochs
    )
    facts = []
    for epoch in range(1, count + 1):
        _train_wake(models, roles, manifest)
        shared.observe_train_batch(roles.train.input, roles.train.target)
        selected = shared.select_recent(manifest.replay_updates_per_attempt)
        decisions = tuple(
            _decide_and_apply(
                models, roles, manifest, policy, epoch, offset + epoch, shared, selected
            )
            for policy in manifest.policies
        )
        facts.append(
            ScheduleOpportunityFacts(
                phase,
                epoch,
                offset + epoch,
                roles.split_hashes["train"],
                shared.retained_order_ids,
                selected.sample_ids,
                shared.retention.example_count,
                shared.retention.retained_bytes,
                decisions,
            )
        )
    return tuple(facts)


def _method_facts(
    name: str,
    model: Model,
    initial: str,
    after_a: str,
    opportunities: tuple[ScheduleOpportunityFacts, ...],
    manifest: ScheduleFactorManifest,
) -> ScheduleMethodFacts:
    is_reference = name in manifest.arms[-2:]
    policy = next((p for p in manifest.policies if name in {f"{m}_{p}" for m in METHODS}), None)
    decisions = [d for row in opportunities for d in row.decisions if d.policy == policy]
    accepted = sum(d.outcome == "accepted" for d in decisions)
    attempts = sum(d.attempted for d in decisions)
    applied = accepted * manifest.replay_updates_per_attempt
    neutral = isinstance(model, CircadianPredictiveCodingNetwork)
    rejected = (
        sum(d.proposed_replay_updates for d in decisions if d.outcome == "rolled_back")
        if neutral
        else 0
    )
    width = manifest.planned_width if is_reference else manifest.width
    if _parameter_count(model) != 4 * width + 1:
        raise ValueError("P6.3 schedule final capacity changed")
    retained_examples = retained_bytes = 0
    if isinstance(model, CircadianPredictiveCodingNetwork):
        clocks = model.get_sleep_clocks()
        if (
            clocks.wake_batches != 24
            or clocks.wake_examples != 1296
            or clocks.replay_updates != applied
            or clocks.sleep_events != accepted
        ):
            raise ValueError("P6.3 schedule model work clocks differ")
        if not np.all(model.get_plasticity_state() == 1.0):
            raise ValueError("P6.3 schedule neutral plasticity changed")
        retention = model.get_replay_retention()
        retained_examples, retained_bytes = retention.example_count, retention.retained_bytes
    is_pc = not isinstance(model, BackpropMLP)
    return ScheduleMethodFacts(
        name,
        initial,
        after_a,
        _parameter_hash(model),
        width,
        width,
        width,
        4 * width + 1,
        4 * width + 1,
        4 * width + 1,
        24,
        1296,
        48 if is_pc else 0,
        2592 if is_pc else 0,
        applied,
        applied,
        2 * applied if is_pc else 0,
        rejected,
        2 * rejected,
        accepted,
        attempts if neutral else 0,
        2 * attempts if neutral else 0,
        retained_examples,
        retained_bytes,
        "none" if is_reference else "model_fifo" if neutral else "shared_fifo",
    )


def _run_seed(manifest: ScheduleFactorManifest, seed: int) -> ScheduleSeedFacts:
    roles_a = arrived._build_phase_a_roles(manifest.source, seed)
    models = _new_models(manifest, seed)
    initial = {name: _parameter_hash(model) for name, model in models.items()}
    shared = SharedReplayBuffer(
        2,
        ReplayRetentionBudget(manifest.memory_examples, manifest.memory_bytes),
        ReplayRetentionPolicy("recent_fifo"),
    )
    opportunities = _train_phase(models, roles_a, manifest, shared)
    after_a = {name: _parameter_hash(model) for name, model in models.items()}
    # Why this: every A wake/guard decision is complete before B source arrival.
    roles_b = arrived._build_phase_b_roles(manifest.source, seed)
    opportunities += _train_phase(models, roles_b, manifest, shared)
    if roles_a.final_released or roles_b.final_released:
        raise ValueError("P6.3 schedule preflight released final data")
    hashes = {
        f"{phase}_{role}": roles.split_hashes[role]
        for phase, roles in (("a", roles_a), ("b", roles_b))
        for role in ("train", "inner_guard", "outer_selection")
    }
    counts = {
        f"{phase}_{role}": len(roles.sample_ids[role])
        for phase, roles in (("a", roles_a), ("b", roles_b))
        for role in ("train", "inner_guard", "outer_selection")
    }
    methods = tuple(
        _method_facts(name, models[name], initial[name], after_a[name], opportunities, manifest)
        for name in manifest.arms
    )
    executed = sum(
        row.wake_updates + row.applied_replay_updates + row.rejected_executed_replay_updates
        for row in methods
    )
    return ScheduleSeedFacts(seed, hashes, counts, opportunities, methods, executed)
