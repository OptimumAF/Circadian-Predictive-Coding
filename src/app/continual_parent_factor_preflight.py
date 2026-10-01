"""Train explicit parent controls using arrived train/inner roles only.

Inputs are frozen counts/settings and arrived role containers. Outputs are
unscored selector, guard, topology, memory and work facts. This module owns
no artifact IO, outer/final scoring, confirmation or treatment selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

import numpy as np

from src.app import continual_arrived_benchmark as arrived
from src.app.continual_combined_factor_preflight import _event_payload, _state_hash, json_value
from src.app.continual_parent_factor_manifest import (
    GROWTH_ARMS,
    PROTOCOL_ID,
    ParentArm,
    ParentManifest,
    validate_parent_manifest,
)
from src.app.continual_parent_factor_validation import verify_parent_payload
from src.app.continual_replay_factor_pilot import Model, _parameter_count, _parameter_hash
from src.app.numpy_sleep_decisions import (
    describe_guarded_numpy_sleep_decision,
    describe_unguarded_numpy_sleep_decision,
)
from src.app.sleep_schedule import SleepAttemptDecision, decide_sleep_attempt
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import (
    CircadianPredictiveCodingNetwork,
    ReplayRetentionBudget,
    replay_sample_id,
)
from src.core.controlled_parent_selection import (
    ParentControlledCircadianNetwork,
    ParentSelectionSettings,
)
from src.core.neuron_adaptation import LayerTraffic, NeuronChangeProposal
from src.core.predictive_coding import PredictiveCodingNetwork
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.shared_replay_schedule import SharedReplayBuffer, SharedReplaySelection
from src.core.sleep_clocks import SleepEpochProgress
from src.core.sleep_telemetry import SleepEventTelemetry
from src.infra.continual_roles import PhaseDecisionRoles


@dataclass(frozen=True)
class ParentSeedFacts:
    seed: int
    role_hashes: dict[str, str]
    role_counts: dict[str, int]
    methods: tuple[dict[str, Any], ...]
    opportunities: tuple[dict[str, Any], ...]
    executed_optimizer_updates: int
    final_released: bool = False


@dataclass(frozen=True)
class ParentPreflight:
    protocol_id: str
    manifest: ParentManifest
    seed_results: tuple[ParentSeedFacts, ...]
    outer_selection_scored: bool = False
    final_released: bool = False


@dataclass
class _Work:
    initial_parameter_sha256: str
    peak_width: int
    wake_updates: int = 0
    wake_presentations: int = 0
    attempts: int = 0
    accepted: int = 0
    guard_examples: int = 0


@dataclass
class _Progress:
    manifest: ParentManifest
    models: dict[str, Model]
    work: dict[str, _Work]
    shared: SharedReplayBuffer
    opportunities: list[dict[str, Any]]


@dataclass(frozen=True)
class _CountPolicy:
    add_count: int

    def propose(self, traffic_by_layer: list[LayerTraffic]) -> list[NeuronChangeProposal]:
        return [NeuronChangeProposal("hidden", add_count=self.add_count)]


def _width(model: Model) -> int:
    return int(model.weight_hidden_output.shape[0])


def _new_progress(manifest: ParentManifest, seed: int) -> _Progress:
    models: dict[str, Model] = {}
    for arm in manifest.arms:
        model_seed = seed + manifest.model_seed_offset
        if arm.model_kind == "backprop":
            model: Model = BackpropMLP(2, arm.width, model_seed)
        elif arm.model_kind == "pc":
            model = PredictiveCodingNetwork(2, arm.width, model_seed)
        elif arm.parent_mode is None:
            model = CircadianPredictiveCodingNetwork(
                2,
                arm.width,
                model_seed,
                arm.config,
                min_hidden_dim=arm.min_width,
                max_hidden_dim=arm.max_width,
            )
        else:
            model = ParentControlledCircadianNetwork(
                2,
                arm.width,
                model_seed,
                arm.config,
                min_hidden_dim=arm.min_width,
                max_hidden_dim=arm.max_width,
                parent_selection=ParentSelectionSettings(
                    arm.parent_mode,
                    seed + manifest.selector_seed_offset,
                    manifest.initial_cursor_id,
                ),
            )
        if isinstance(model, CircadianPredictiveCodingNetwork):
            model.configure_replay_retention(
                ReplayRetentionBudget(manifest.memory_examples, manifest.memory_bytes),
                policy=ReplayRetentionPolicy("recent_fifo"),
            )
        models[arm.name] = model
    for width in (8, 13):
        if (
            len({_parameter_hash(models[arm.name]) for arm in manifest.arms if arm.width == width})
            != 1
        ):
            raise ValueError("P6.3 parent initial parameters differ within width")
    shared = SharedReplayBuffer(
        2,
        ReplayRetentionBudget(manifest.memory_examples, manifest.memory_bytes),
        ReplayRetentionPolicy("recent_fifo"),
    )
    return _Progress(
        manifest,
        models,
        {name: _Work(_parameter_hash(model), _width(model)) for name, model in models.items()},
        shared,
        [],
    )


def _require_parity(progress: _Progress) -> None:
    if _parameter_hash(progress.models["pc_off"]) != _parameter_hash(
        progress.models["neutral_off"]
    ):
        raise ValueError("P6.3 parent neutral PC parity differs")


def _train_wake(progress: _Progress, roles: PhaseDecisionRoles) -> tuple[dict[str, Any], ...]:
    training = progress.manifest.source.training
    rows = []
    for name, model in progress.models.items():
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
        work = progress.work[name]
        work.wake_updates += 1
        work.wake_presentations += len(roles.train.input)
        rows.append(
            {
                "name": name,
                "width": _width(model),
                "parameters": _parameter_count(model),
                "parameter_sha256": _parameter_hash(model),
                "state_sha256": _state_hash(model)
                if isinstance(model, CircadianPredictiveCodingNetwork)
                else None,
                "minimum_plasticity": float(np.min(model.get_plasticity_state()))
                if isinstance(model, CircadianPredictiveCodingNetwork)
                else None,
                "reward_scale": model.get_last_reward_scale()
                if isinstance(model, CircadianPredictiveCodingNetwork)
                else None,
            }
        )
    _require_parity(progress)
    return tuple(rows)


def _capture(model: ParentControlledCircadianNetwork) -> dict[str, Any]:
    return {
        "state_sha256": _state_hash(model),
        "parameter_sha256": _parameter_hash(model),
        "lineage": asdict(model.get_neuron_lineage()),
        "clocks": asdict(model.get_sleep_clocks()),
        "selector": asdict(model.get_parent_selection_state()),
    }


def _guarded_proposal(
    model: ParentControlledCircadianNetwork,
    roles: PhaseDecisionRoles,
    decision: SleepAttemptDecision,
    global_epoch: int,
    add_count: int,
    tolerance: float,
) -> tuple[SleepEventTelemetry, dict[str, Any]]:
    saved = model.snapshot_state()
    try:
        pre = model.compute_accuracy(roles.inner_guard.input, roles.inner_guard.target)
        if not isfinite(pre):
            raise ValueError("P6.3 parent inner guard accuracy must be finite before sleep")
        result = model.sleep_event(
            adaptation_policy=_CountPolicy(add_count),
            force_sleep=decision.force_sleep,
            epoch_progress=SleepEpochProgress(global_epoch, 24),
            max_hidden_width=13,
            max_replay_examples=0,
        )
        proposed = _capture(model)
        post = model.compute_accuracy(roles.inner_guard.input, roles.inner_guard.target)
        if not isfinite(post):
            raise ValueError("P6.3 parent inner guard accuracy must be finite after sleep")
        accepted = post + tolerance >= pre
        if not accepted:
            model.restore_state(saved)
        event = describe_guarded_numpy_sleep_decision(
            decision,
            result,
            completed_epoch=global_epoch,
            guard_role_hash=roles.split_hashes["inner_guard"],
            accuracy_before=pre,
            accuracy_after=post,
            tolerance=tolerance,
            guard_examples=len(roles.inner_guard.input),
            accepted=accepted,
            attempt_seconds=0.0,
            model=model,
        )
        return event, proposed
    except Exception:
        model.restore_state(saved)
        raise


def _apply_growth_sleep(
    progress: _Progress,
    roles: PhaseDecisionRoles,
    arm: ParentArm,
    epoch: int,
    global_epoch: int,
) -> dict[str, Any]:
    model = progress.models[arm.name]
    assert isinstance(model, ParentControlledCircadianNetwork)
    before = _capture(model)
    scores = tuple(float(value) for value in model._compute_split_scores())
    decision = decide_sleep_attempt(
        sleep_mode=model.config.sleep_mode,
        completed_epochs=epoch,
        interval_epochs=progress.manifest.interval,
        adaptive_due=model.should_trigger_sleep(),
        force_periodic=True,
    )
    add = int(global_epoch in progress.manifest.add_epochs)
    if decision.attempted:
        event, proposed = _guarded_proposal(
            model, roles, decision, global_epoch, add, progress.manifest.source.guard_drop_tolerance
        )
    else:
        proposed = before
        event = describe_unguarded_numpy_sleep_decision(
            model, decision, completed_epoch=global_epoch
        )
    after = _capture(model)
    if event.outcome != "accepted" and before != after:
        raise ValueError("P6.3 parent rejected/skipped sleep changed complete state")
    work = progress.work[arm.name]
    work.attempts += int(event.guard is not None)
    work.accepted += int(event.outcome == "accepted")
    work.guard_examples += event.guard.examples_scored if event.guard else 0
    work.peak_width = max(work.peak_width, event.proposed_width)
    if work.peak_width > arm.max_width:
        raise ValueError("P6.3 parent transient width cap exceeded")
    return {
        "name": arm.name,
        "requested_add_count": add,
        "split_scores": scores,
        "event": _event_payload(event),
        "before": before,
        "proposed": proposed,
        "after": after,
    }


def _train_phase(progress: _Progress, roles: PhaseDecisionRoles) -> None:
    offset = 0 if roles.phase == "a" else 12
    for epoch in range(1, 13):
        wake = _train_wake(progress, roles)
        progress.shared.observe_train_batch(roles.train.input, roles.train.target)
        selected = progress.shared.select_recent(progress.manifest.memory_examples)
        for model in progress.models.values():
            if isinstance(model, CircadianPredictiveCodingNetwork):
                _check_retained_supply(model, progress.shared, selected)
        decisions = tuple(
            _apply_growth_sleep(progress, roles, arm, epoch, offset + epoch)
            for arm in progress.manifest.arms
            if arm.name in GROWTH_ARMS
        )
        for model in progress.models.values():
            if isinstance(model, CircadianPredictiveCodingNetwork):
                _check_retained_supply(model, progress.shared, selected)
        progress.opportunities.append(
            {
                "phase": roles.phase,
                "epoch": epoch,
                "global_epoch": offset + epoch,
                "train_role_hash": roles.split_hashes["train"],
                "wake": wake,
                "retained_order_ids": progress.shared.retained_order_ids,
                "retained_examples": progress.shared.retention.example_count,
                "retained_bytes": progress.shared.retention.retained_bytes,
                "decisions": decisions,
                "after_epoch_parameter_sha256": {
                    name: _parameter_hash(model) for name, model in progress.models.items()
                },
                "after_epoch_state_sha256": {
                    name: _state_hash(model)
                    for name, model in progress.models.items()
                    if isinstance(model, CircadianPredictiveCodingNetwork)
                },
                "after_epoch_widths": {
                    name: _width(model) for name, model in progress.models.items()
                },
            }
        )


def _check_retained_supply(
    model: CircadianPredictiveCodingNetwork,
    shared: SharedReplayBuffer,
    available: SharedReplaySelection,
) -> None:
    # Why this: disabled replay has no sampling-priority treatment. Bind all
    # available row contents/order without asking a replay selector to run.
    if (
        model.get_replay_retention() != shared.retention
        or model.get_replay_retained_order_ids() != shared.retained_order_ids
        or available.sample_ids != shared.retained_order_ids
        or tuple(replay_sample_id(x, y) for x, y in available.training_batches())
        != shared.retained_order_ids
    ):
        raise ValueError("P6.3 parent model/shared retained supply differs")


def _method_facts(progress: _Progress, arm: ParentArm) -> dict[str, Any]:
    model, work = progress.models[arm.name], progress.work[arm.name]
    clocks = retention = state_hash = None
    if isinstance(model, CircadianPredictiveCodingNetwork):
        clocks = asdict(model.get_sleep_clocks())
        retention = asdict(model.get_replay_retention())
        state_hash = _state_hash(model)
    is_pc = not isinstance(model, BackpropMLP)
    return {
        "name": arm.name,
        "initial_parameter_sha256": work.initial_parameter_sha256,
        "after_a_parameter_sha256": progress.opportunities[11]["after_epoch_parameter_sha256"][
            arm.name
        ],
        "final_parameter_sha256": _parameter_hash(model),
        "after_a_state_sha256": progress.opportunities[11]["after_epoch_state_sha256"].get(
            arm.name
        ),
        "final_state_sha256": state_hash,
        "width_initial": arm.width,
        "width_final": _width(model),
        "width_peak": work.peak_width,
        "parameters_initial": 4 * arm.width + 1,
        "parameters_final": _parameter_count(model),
        "parameters_peak": 4 * work.peak_width + 1,
        "wake_updates": work.wake_updates,
        "wake_presentations": work.wake_presentations,
        "wake_inference_loops": 2 * work.wake_updates if is_pc else 0,
        "wake_example_inference_iterations": 2 * work.wake_presentations if is_pc else 0,
        "applied_replay_updates": 0,
        "rejected_executed_replay_updates": 0,
        "own_sleep_attempts": work.attempts,
        "committed_sleep_events": work.accepted,
        "guard_evaluations": 2 * work.attempts,
        "guard_examples": work.guard_examples,
        "final_clocks": clocks,
        "retention": retention,
        "final_selector": asdict(model.get_parent_selection_state())
        if isinstance(model, ParentControlledCircadianNetwork)
        else None,
    }


def _run_seed(manifest: ParentManifest, seed: int) -> ParentSeedFacts:
    phase_a = arrived._build_phase_a_roles(manifest.source, seed)
    progress = _new_progress(manifest, seed)
    _train_phase(progress, phase_a)
    phase_b = arrived._build_phase_b_roles(manifest.source, seed)
    _train_phase(progress, phase_b)
    if phase_a.final_released or phase_b.final_released:
        raise ValueError("P6.3 parent gate released final data")
    hashes = {
        f"{phase}_{role}": bound.split_hashes[role]
        for phase, bound in (("a", phase_a), ("b", phase_b))
        for role in ("train", "inner_guard", "outer_selection")
    }
    counts = {
        f"{phase}_{role}": len(bound.sample_ids[role])
        for phase, bound in (("a", phase_a), ("b", phase_b))
        for role in ("train", "inner_guard", "outer_selection")
    }
    methods = tuple(_method_facts(progress, arm) for arm in manifest.arms)
    return ParentSeedFacts(
        seed,
        hashes,
        counts,
        methods,
        tuple(progress.opportunities),
        sum(row["wake_updates"] for row in methods),
    )


def run_parent_preflight(manifest: ParentManifest) -> ParentPreflight:
    validate_parent_manifest(manifest)
    result = ParentPreflight(
        PROTOCOL_ID, manifest, tuple(_run_seed(manifest, seed) for seed in manifest.seeds)
    )
    verify_parent_payload(json_value(result), json_value(manifest))
    return result
