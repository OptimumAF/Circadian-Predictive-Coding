"""Train frozen combined/removal controls with complete rollback evidence.

Inputs are the manifest and arrived train/inner roles. Outputs are unscored
model, lineage, memory, guard and executed-work facts. This app does not
read outer/final arrays, select settings, score or write artifacts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, fields, is_dataclass
from hashlib import sha256
import json
from typing import Any

import numpy as np

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_shift_benchmark as base
from src.app.continual_combined_factor_manifest import (
    CombinedArm,
    CombinedManifest,
    DECISION_ARMS,
    PARITY_PAIRS,
    PROTOCOL_ID,
    REPLAY_CONTROLS,
    validate_combined_manifest,
)
from src.app.continual_replay_factor_pilot import Model, _parameter_count, _parameter_hash
from src.app.continual_schedule_factor_preflight import _check_supply
from src.app.continual_combined_factor_validation import verify_combined_payload
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import (
    CircadianPredictiveCodingNetwork,
    ReplayRetentionBudget,
)
from src.core.predictive_coding import PredictiveCodingNetwork
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.shared_replay_schedule import SharedReplayBuffer, SharedReplaySelection
from src.core.sleep_clocks import SleepEpochProgress
from src.core.sleep_telemetry import SleepEventTelemetry
from src.infra.continual_roles import PhaseDecisionRoles


@dataclass(frozen=True)
class CombinedSeedFacts:
    seed: int
    role_hashes: dict[str, str]
    role_counts: dict[str, int]
    methods: tuple[dict[str, Any], ...]
    opportunities: tuple[dict[str, Any], ...]
    executed_optimizer_updates: int
    final_released: bool = False


@dataclass(frozen=True)
class CombinedPreflight:
    protocol_id: str
    manifest: CombinedManifest
    seed_results: tuple[CombinedSeedFacts, ...]
    outer_selection_scored: bool = False
    final_released: bool = False


@dataclass
class _Work:
    initial_parameter_sha256: str
    peak_width: int
    wake_updates: int = 0
    applied_replay_updates: int = 0
    rejected_executed_replay_updates: int = 0
    own_sleep_attempts: int = 0
    committed_sleep_events: int = 0
    guard_evaluations: int = 0
    guard_examples: int = 0


@dataclass
class _Progress:
    manifest: CombinedManifest
    models: dict[str, Model]
    work: dict[str, _Work]
    shared: SharedReplayBuffer
    opportunities: list[dict[str, Any]]


def json_value(value: Any) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(value), allow_nan=False))


def _canonical_state(value: Any) -> Any:
    """Bind value/type of every snapshot field, including RNG and containers."""
    if isinstance(value, np.ndarray):
        return {
            "array_dtype": value.dtype.str,
            "shape": list(value.shape),
            "bytes_sha256": sha256(value.tobytes(order="C")).hexdigest(),
        }
    if isinstance(value, np.random.Generator):
        return {"generator": _canonical_state(value.bit_generator.state)}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: _canonical_state(getattr(value, field.name)) for field in fields(value)
            },
        }
    if isinstance(value, dict):
        return {
            "dict": [
                [_canonical_state(k), _canonical_state(v)]
                for k, v in sorted(value.items(), key=lambda row: str(row[0]))
            ]
        }
    if isinstance(value, deque):
        return {"deque": [_canonical_state(item) for item in value], "maxlen": value.maxlen}
    if isinstance(value, (list, tuple)):
        return {type(value).__name__: [_canonical_state(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        return {type(value).__name__: [_canonical_state(item) for item in sorted(value)]}
    if isinstance(value, np.generic):
        return {"numpy_scalar": value.dtype.str, "value": value.item()}
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"unsupported complete snapshot field type: {type(value).__name__}")


def _state_hash(model: CircadianPredictiveCodingNetwork) -> str:
    payload = json.dumps(_canonical_state(model.snapshot_state()), sort_keys=True, allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def _width(model: Model) -> int:
    return int(model.weight_hidden_output.shape[0])


def _new_models(manifest: CombinedManifest, seed: int) -> dict[str, Model]:
    models: dict[str, Model] = {}
    for arm in manifest.arms:
        model_seed = seed + manifest.model_seed_offset
        if arm.model_kind == "backprop":
            model: Model = BackpropMLP(2, arm.width, model_seed)
        elif arm.model_kind == "pc":
            model = PredictiveCodingNetwork(2, arm.width, model_seed)
        else:
            assert arm.config is not None
            model = CircadianPredictiveCodingNetwork(
                2,
                arm.width,
                model_seed,
                arm.config,
                min_hidden_dim=arm.min_width,
                max_hidden_dim=arm.max_width,
            )
            model.configure_replay_retention(
                ReplayRetentionBudget(manifest.memory_examples, manifest.memory_bytes),
                policy=ReplayRetentionPolicy("recent_fifo"),
            )
        models[arm.name] = model
    for width in (8, 14):
        if (
            len({_parameter_hash(models[arm.name]) for arm in manifest.arms if arm.width == width})
            != 1
        ):
            raise ValueError("P6.3 combined initial parameters differ within width")
    return models


def _new_progress(manifest: CombinedManifest, seed: int) -> _Progress:
    models = _new_models(manifest, seed)
    work = {name: _Work(_parameter_hash(model), _width(model)) for name, model in models.items()}
    shared = SharedReplayBuffer(
        2,
        ReplayRetentionBudget(manifest.memory_examples, manifest.memory_bytes),
        ReplayRetentionPolicy("recent_fifo"),
    )
    return _Progress(manifest, models, work, shared, [])


def _require_parity(models: dict[str, Model]) -> None:
    for left, right in PARITY_PAIRS:
        if _parameter_hash(models[left]) != _parameter_hash(models[right]):
            raise ValueError(f"P6.3 combined neutral PC parity differs: {left}/{right}")


def _train_wake(progress: _Progress, roles: PhaseDecisionRoles) -> tuple[dict[str, Any], ...]:
    training = progress.manifest.source.training
    rows = []
    for arm in progress.manifest.arms:
        model = progress.models[arm.name]
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
        progress.work[arm.name].wake_updates += 1
        progress.work[arm.name].peak_width = max(progress.work[arm.name].peak_width, _width(model))
        plasticity = reward_scale = None
        if isinstance(model, CircadianPredictiveCodingNetwork):
            plasticity = float(np.min(model.get_plasticity_state()))
            reward_scale = model.get_last_reward_scale()
        rows.append(
            {
                "name": arm.name,
                "width": _width(model),
                "parameters": _parameter_count(model),
                "parameter_sha256": _parameter_hash(model),
                "minimum_plasticity": plasticity,
                "reward_scale": reward_scale,
            }
        )
    _require_parity(progress.models)
    return tuple(rows)


def _event_payload(event: SleepEventTelemetry) -> dict[str, Any]:
    payload = asdict(event)
    # Why this: timings belong to scoped audits, not deterministic train facts.
    del payload["durations"]
    return payload


def _apply_own_sleep(
    progress: _Progress,
    roles: PhaseDecisionRoles,
    arm: CombinedArm,
    epoch: int,
    global_epoch: int,
    selected: SharedReplaySelection,
) -> dict[str, Any]:
    model = progress.models[arm.name]
    assert isinstance(model, CircadianPredictiveCodingNetwork)
    before = _state_hash(model)
    parameter_before = _parameter_hash(model)
    lineage_before = asdict(model.get_neuron_lineage())
    clocks_before = asdict(model.get_sleep_clocks())
    events: list[SleepEventTelemetry] = []
    base._apply_scheduled_sleep(
        model,
        arm.interval,
        epoch,
        global_epoch,
        24,
        bool(arm.interval),
        model.get_sleep_clocks().sleep_events,
        0,
        0,
        guard=roles.inner_guard,
        guard_drop_tolerance=progress.manifest.source.guard_drop_tolerance,
        on_sleep_event=events.append,
        guard_role_hash=roles.split_hashes["inner_guard"],
    )
    if len(events) != 1:
        raise ValueError("P6.3 combined expected one own decision per epoch")
    event = events[0]
    after = _state_hash(model)
    if event.outcome not in {"accepted", "rolled_back", "skipped"}:
        raise ValueError("P6.3 combined incomplete own sleep attempt")
    if event.outcome != "accepted" and after != before:
        raise ValueError("P6.3 combined rejected/skipped sleep changed complete state")
    work = progress.work[arm.name]
    attempted = event.guard is not None
    work.own_sleep_attempts += int(attempted)
    work.guard_evaluations += 2 * int(attempted)
    work.guard_examples += event.guard.examples_scored if event.guard else 0
    work.committed_sleep_events += int(event.outcome == "accepted")
    work.applied_replay_updates += event.replay.applied_updates
    if event.outcome == "rolled_back":
        work.rejected_executed_replay_updates += event.replay.proposed_updates
    work.peak_width = max(
        work.peak_width, event.before_width + len(event.changes.proposed_split_pairs)
    )
    if work.peak_width > arm.max_width:
        raise ValueError("P6.3 combined transient capacity cap exceeded")
    return {
        "name": arm.name,
        "event": _event_payload(event),
        "state_sha256_before": before,
        "state_sha256_after": after,
        "parameter_sha256_before": parameter_before,
        "parameter_sha256_after": _parameter_hash(model),
        "lineage_before": lineage_before,
        "lineage_after": asdict(model.get_neuron_lineage()),
        "clocks_before": clocks_before,
        "clocks_after": asdict(model.get_sleep_clocks()),
        "proposed_replay_ids": selected.sample_ids if event.replay.proposed_updates else (),
        "applied_replay_ids": selected.sample_ids if event.replay.applied_updates else (),
    }


def _apply_matched_replay(
    progress: _Progress,
    selected: SharedReplaySelection,
    global_epoch: int,
) -> dict[str, tuple[str, ...]]:
    full = progress.models["full"]
    assert isinstance(full, CircadianPredictiveCodingNetwork)
    config = full.config
    for name in REPLAY_CONTROLS:
        model = progress.models[name]
        if isinstance(model, CircadianPredictiveCodingNetwork):
            event = model.sleep_event(
                force_sleep=True, epoch_progress=SleepEpochProgress(global_epoch, 24)
            )
            if event.telemetry is None or event.telemetry.replay.applied_updates != len(
                selected.sample_ids
            ):
                raise ValueError("P6.3 combined neutral controlled replay differs")
            progress.work[name].committed_sleep_events += 1
        else:
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
        progress.work[name].applied_replay_updates += len(selected.sample_ids)
    _require_parity(progress.models)
    return {name: selected.sample_ids for name in REPLAY_CONTROLS}


def _train_phase(progress: _Progress, roles: PhaseDecisionRoles) -> None:
    offset = 0 if roles.phase == "a" else 12
    for epoch in range(1, 13):
        wake = _train_wake(progress, roles)
        progress.shared.observe_train_batch(roles.train.input, roles.train.target)
        selected = progress.shared.select_recent(2)
        decisions = []
        controlled: dict[str, tuple[str, ...]] = {name: () for name in REPLAY_CONTROLS}
        for model in progress.models.values():
            if isinstance(model, CircadianPredictiveCodingNetwork):
                _check_supply(model, progress.shared, selected)
        for arm in progress.manifest.arms:
            if arm.name not in DECISION_ARMS:
                continue
            fact = _apply_own_sleep(progress, roles, arm, epoch, offset + epoch, selected)
            decisions.append(fact)
            if arm.name == "full" and fact["event"]["outcome"] == "accepted":
                controlled = _apply_matched_replay(progress, selected, offset + epoch)
        for model in progress.models.values():
            if isinstance(model, CircadianPredictiveCodingNetwork):
                _check_supply(model, progress.shared, selected)
        progress.opportunities.append(
            {
                "phase": roles.phase,
                "epoch": epoch,
                "global_epoch": offset + epoch,
                "train_role_hash": roles.split_hashes["train"],
                "wake": wake,
                "retained_order_ids": progress.shared.retained_order_ids,
                "selected_ids": selected.sample_ids,
                "retained_examples": progress.shared.retention.example_count,
                "retained_bytes": progress.shared.retention.retained_bytes,
                "decisions": tuple(decisions),
                "applied_control_replay_ids": controlled,
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


def _method_facts(progress: _Progress, arm: CombinedArm, after_a: str) -> dict[str, Any]:
    model, work = progress.models[arm.name], progress.work[arm.name]
    clocks = retention = None
    state_hash = None
    if isinstance(model, CircadianPredictiveCodingNetwork):
        clocks = asdict(model.get_sleep_clocks())
        retention = asdict(model.get_replay_retention())
        state_hash = _state_hash(model)
    is_pc = not isinstance(model, BackpropMLP)
    if clocks is not None and (
        clocks["wake_batches"] != work.wake_updates
        or clocks["wake_examples"] != 1296
        or clocks["replay_updates"] != work.applied_replay_updates
        or clocks["sleep_events"] != work.committed_sleep_events
    ):
        raise ValueError("P6.3 combined final clocks differ from observed work")
    return {
        "name": arm.name,
        "initial_parameter_sha256": work.initial_parameter_sha256,
        "after_a_parameter_sha256": after_a,
        "final_parameter_sha256": _parameter_hash(model),
        "width_initial": arm.width,
        "width_final": _width(model),
        "width_peak": work.peak_width,
        "parameters_initial": 4 * arm.width + 1,
        "parameters_final": _parameter_count(model),
        "parameters_peak": 4 * work.peak_width + 1,
        "wake_updates": work.wake_updates,
        "wake_presentations": 1296,
        "wake_inference_loops": 2 * work.wake_updates if is_pc else 0,
        "wake_example_inference_iterations": 2592 if is_pc else 0,
        "applied_replay_updates": work.applied_replay_updates,
        "applied_replay_presentations": work.applied_replay_updates,
        "applied_replay_inference_loops": 2 * work.applied_replay_updates if is_pc else 0,
        "rejected_executed_replay_updates": work.rejected_executed_replay_updates,
        "rejected_replay_inference_loops": 2 * work.rejected_executed_replay_updates,
        "own_sleep_attempts": work.own_sleep_attempts,
        "committed_sleep_events": work.committed_sleep_events,
        "guard_evaluations": work.guard_evaluations,
        "guard_examples": work.guard_examples,
        "final_clocks": clocks,
        "retention": retention,
        "final_state_sha256": state_hash,
    }


def _finish_seed(
    progress: _Progress,
    seed: int,
    phase_a: PhaseDecisionRoles,
    phase_b: PhaseDecisionRoles,
    after_a: dict[str, str],
) -> CombinedSeedFacts:
    if phase_a.final_released or phase_b.final_released:
        raise ValueError("P6.3 combined gate released final data")
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
    methods = tuple(
        _method_facts(progress, arm, after_a[arm.name]) for arm in progress.manifest.arms
    )
    executed = sum(
        row["wake_updates"]
        + row["applied_replay_updates"]
        + row["rejected_executed_replay_updates"]
        for row in methods
    )
    return CombinedSeedFacts(seed, hashes, counts, methods, tuple(progress.opportunities), executed)


def _run_seed(manifest: CombinedManifest, seed: int) -> CombinedSeedFacts:
    phase_a = arrived._build_phase_a_roles(manifest.source, seed)
    progress = _new_progress(manifest, seed)
    _train_phase(progress, phase_a)
    after_a = {name: _parameter_hash(model) for name, model in progress.models.items()}
    phase_b = arrived._build_phase_b_roles(manifest.source, seed)
    _train_phase(progress, phase_b)
    return _finish_seed(progress, seed, phase_a, phase_b, after_a)


def run_combined_preflight(manifest: CombinedManifest) -> CombinedPreflight:
    validate_combined_manifest(manifest)
    result = CombinedPreflight(
        PROTOCOL_ID, manifest, tuple(_run_seed(manifest, seed) for seed in manifest.seeds)
    )
    verify_combined_payload(json_value(result), json_value(manifest))
    return result
