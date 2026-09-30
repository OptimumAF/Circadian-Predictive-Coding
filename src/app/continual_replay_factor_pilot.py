"""Run a fixed-width, development-only replay factor on arrived A then B.

Inputs are a frozen factor manifest and train/outer-selection roles. Outputs
are explicit two-task metrics and matched replay/work facts. This module
does not release final roles, tune settings, or write experiment artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from typing import TypeAlias

import numpy as np

from src.app import continual_arrived_benchmark as arrived
from src.app.continual_trigger_replay_schedule import fixed_trigger_replay_manifest
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
    ReplayRetentionBudget,
    replay_sample_id,
)
from src.core.continual_metrics import TWO_TASK_METRIC_CONTRACT_ID, TwoTaskAccuracy
from src.core.predictive_coding import PredictiveCodingNetwork
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.shared_replay_schedule import SharedReplayBuffer, SharedReplaySelection
from src.infra.continual_roles import PhaseDecisionRoles


PROTOCOL_ID = "continual_mechanism_replay_dev_v1"
DEVELOPMENT_SEEDS = (41, 43, 59)
CONFIRMATION_SEEDS = (101, 103, 107, 109, 113, 127, 131, 137, 139, 149)
ARMS = (
    "backprop_off",
    "backprop_on",
    "pc_off",
    "pc_on",
    "circadian_off",
    "circadian_on",
    "backprop_width12_off",
    "pc_width12_off",
)
ON_ARMS = ("backprop_on", "pc_on", "circadian_on")
MAX_OPTIMIZER_UPDATES = 720
MODEL_SEED_OFFSET = 1001
REPLAY_INTERVAL = 4
REPLAY_UPDATES_PER_BOUNDARY = 2
FIXED_WIDTH = 8
PLANNED_WIDTH = 12
PARAMETER_FIELDS = ("weight_input_hidden", "bias_hidden", "weight_hidden_output", "bias_output")
Model: TypeAlias = BackpropMLP | PredictiveCodingNetwork | CircadianPredictiveCodingNetwork


def _circadian_config(replay_enabled: bool) -> CircadianConfig:
    return replace(
        CircadianConfig.matched_pc_control(),
        sleep_mode="components",
        sleep_enable_chemical_reset=False,
        sleep_enable_replay=replay_enabled,
        sleep_enable_homeostasis=False,
        sleep_enable_split=False,
        sleep_enable_prune=False,
        replay_steps=REPLAY_UPDATES_PER_BOUNDARY,
        replay_memory_size=8,
        replay_learning_rate=0.01,
        replay_inference_steps=2,
        replay_inference_learning_rate=0.15,
        replay_prioritized=False,
        replay_class_balanced=False,
    )


@dataclass(frozen=True)
class ReplayPilotManifest:
    source: arrived.ContinualArrivedRolesConfig
    seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    arms: tuple[str, ...]
    width: int
    planned_width: int
    interval: int
    replay_updates_per_boundary: int
    max_optimizer_updates: int
    metric_contract_id: str
    protocol_id: str = PROTOCOL_ID


@dataclass(frozen=True)
class ReplayDevelopmentMetrics:
    a_after_a: float
    a_after_b: float
    b_after_b: float
    final_mean_task_accuracy: float
    signed_forgetting_a: float
    retention_ratio_a: float | None


@dataclass(frozen=True)
class ReplayMethodResult:
    method: str
    initial_parameter_sha256: str
    final_parameter_sha256: str
    development: ReplayDevelopmentMetrics
    wake_updates: int
    wake_presentations: int
    wake_inference_loops: int
    wake_example_inference_iterations: int
    replay_updates: int
    replay_presentations: int
    replay_inference_loops: int
    sleep_attempts: int
    hidden_width_start: int
    hidden_width_end: int
    parameter_count_start: int
    parameter_count_end: int


@dataclass(frozen=True)
class ReplayBoundaryResult:
    phase: str
    epoch: int
    selected_ids: tuple[str, ...]
    retained_order_ids: tuple[str, ...]
    retained_examples: int
    retained_bytes: int
    applied_ids_by_method: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class ReplaySeedResult:
    seed: int
    source_role_hashes: dict[str, str]
    source_role_counts: dict[str, int]
    boundaries: tuple[ReplayBoundaryResult, ...]
    methods: tuple[ReplayMethodResult, ...]


@dataclass(frozen=True)
class ReplayPilotResult:
    protocol_id: str
    manifest: ReplayPilotManifest
    final_released: bool
    seed_results: tuple[ReplaySeedResult, ...]


def fixed_replay_pilot_manifest() -> ReplayPilotManifest:
    historical = fixed_trigger_replay_manifest().arrived
    # Why this: retain the independently reviewed v14 source/role geometry
    # while removing every sleep effect except the prospective replay factor.
    source = replace(
        historical, training=replace(historical.training, circadian_config=_circadian_config(True))
    )
    return ReplayPilotManifest(
        source,
        DEVELOPMENT_SEEDS,
        CONFIRMATION_SEEDS,
        ARMS,
        FIXED_WIDTH,
        PLANNED_WIDTH,
        REPLAY_INTERVAL,
        REPLAY_UPDATES_PER_BOUNDARY,
        MAX_OPTIMIZER_UPDATES,
        TWO_TASK_METRIC_CONTRACT_ID,
    )


def validate_replay_pilot_manifest(manifest: ReplayPilotManifest) -> int:
    if type(manifest) is not ReplayPilotManifest or manifest != fixed_replay_pilot_manifest():
        raise ValueError("P6.3 replay factor requires its frozen manifest")
    if set(manifest.seeds) & set(manifest.confirmation_seeds):
        raise ValueError("P6.3 development and confirmation seeds overlap")
    arrived._validate_arrived_config(manifest.source, list(manifest.seeds))
    training = manifest.source.training
    if training.hidden_dim != manifest.width:
        raise ValueError("P6.3 replay factor width differs from source")
    if (
        training.phase_a_epochs % manifest.interval
        or training.phase_b_epochs % manifest.interval
        or training.circadian_sleep_interval_phase_a != manifest.interval
        or training.circadian_sleep_interval_phase_b != manifest.interval
    ):
        raise ValueError("P6.3 replay factor boundaries differ from source")
    wake = (
        len(manifest.seeds)
        * len(manifest.arms)
        * (training.phase_a_epochs + training.phase_b_epochs)
    )
    boundaries = (
        training.phase_a_epochs // manifest.interval + training.phase_b_epochs // manifest.interval
    )
    replay = len(manifest.seeds) * len(ON_ARMS) * boundaries * manifest.replay_updates_per_boundary
    planned = wake + replay
    if planned > manifest.max_optimizer_updates:
        raise ValueError("P6.3 replay factor exceeds local optimizer cap")
    return planned


def run_replay_pilot(manifest: ReplayPilotManifest) -> ReplayPilotResult:
    """Train every frozen development cell, then score outer roles only."""
    validate_replay_pilot_manifest(manifest)
    return ReplayPilotResult(
        PROTOCOL_ID,
        manifest,
        False,
        tuple(_run_seed(manifest, seed) for seed in manifest.seeds),
    )


def _parameter_hash(model: Model) -> str:
    digest = sha256(b"p63-replay-shallow-parameter-tensors-v1")
    for name in PARAMETER_FIELDS:
        array = np.ascontiguousarray(getattr(model, name), dtype="<f8")
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _parameter_count(model: Model) -> int:
    return sum(int(getattr(model, name).size) for name in PARAMETER_FIELDS)


def _models(seed: int, manifest: ReplayPilotManifest) -> dict[str, Model]:
    model_seed = seed + MODEL_SEED_OFFSET
    width = manifest.width
    models: dict[str, Model] = {
        "backprop_off": BackpropMLP(2, width, model_seed),
        "backprop_on": BackpropMLP(2, width, model_seed),
        "pc_off": PredictiveCodingNetwork(2, width, model_seed),
        "pc_on": PredictiveCodingNetwork(2, width, model_seed),
        "circadian_off": CircadianPredictiveCodingNetwork(
            2,
            width,
            model_seed,
            _circadian_config(False),
            min_hidden_dim=width,
            max_hidden_dim=width,
        ),
        "circadian_on": CircadianPredictiveCodingNetwork(
            2,
            width,
            model_seed,
            _circadian_config(True),
            min_hidden_dim=width,
            max_hidden_dim=width,
        ),
        "backprop_width12_off": BackpropMLP(2, manifest.planned_width, model_seed),
        "pc_width12_off": PredictiveCodingNetwork(2, manifest.planned_width, model_seed),
    }
    for name in ("circadian_off", "circadian_on"):
        circadian = models[name]
        assert isinstance(circadian, CircadianPredictiveCodingNetwork)
        circadian.configure_replay_retention(
            ReplayRetentionBudget(8, 192), policy=ReplayRetentionPolicy("recent_fifo")
        )
    if len({_parameter_hash(models[name]) for name in manifest.arms[:6]}) != 1:
        raise ValueError("P6.3 width-eight model initialization differs")
    if _parameter_hash(models["backprop_width12_off"]) != _parameter_hash(models["pc_width12_off"]):
        raise ValueError("P6.3 planned-width initialization differs")
    return models


def _require_neutral_parity(models: dict[str, Model]) -> None:
    for pc_name, circadian_name in (("pc_off", "circadian_off"), ("pc_on", "circadian_on")):
        if any(
            not np.array_equal(
                getattr(models[pc_name], field), getattr(models[circadian_name], field)
            )
            for field in PARAMETER_FIELDS
        ):
            raise ValueError(f"P6.3 neutral circadian parity differs: {circadian_name}")


def _train_wake(
    models: dict[str, Model], roles: PhaseDecisionRoles, manifest: ReplayPilotManifest
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
    _require_neutral_parity(models)


def _apply_replay(
    models: dict[str, Model], selection: SharedReplaySelection, global_epoch: int
) -> dict[str, tuple[str, ...]]:
    applied: dict[str, tuple[str, ...]] = {}
    for name in ("circadian_off", "circadian_on"):
        model = models[name]
        assert isinstance(model, CircadianPredictiveCodingNetwork)
        before = model.get_sleep_clocks()
        event = model.sleep_event(
            force_sleep=True,
            current_step=global_epoch,
            total_steps=24,
            max_replay_examples=2,
            max_hidden_width=8,
        )
        after = model.get_sleep_clocks()
        expected = len(selection.sample_ids) if name == "circadian_on" else 0
        if (
            not event.performed
            or event.telemetry is None
            or event.telemetry.replay.applied_updates != expected
            or event.telemetry.replay.applied_examples != expected
            or after.replay_updates - before.replay_updates != expected
            or model.hidden_dim != 8
            or event.split_indices
            or event.pruned_indices
        ):
            raise ValueError(f"P6.3 replay-only sleep work differs: {name}")
        applied[name] = selection.sample_ids if expected else ()
    circadian_on = models["circadian_on"]
    assert isinstance(circadian_on, CircadianPredictiveCodingNetwork)
    replay = circadian_on.config
    for name in ("backprop_on", "pc_on"):
        model = models[name]
        for inputs, targets in selection.training_batches():
            if isinstance(model, BackpropMLP):
                model.train_epoch(inputs, targets, replay.replay_learning_rate)
            else:
                model.train_epoch(
                    inputs,
                    targets,
                    replay.replay_learning_rate,
                    replay.replay_inference_steps,
                    replay.replay_inference_learning_rate,
                )
        applied[name] = selection.sample_ids
    for name in ("backprop_off", "pc_off", "backprop_width12_off", "pc_width12_off"):
        applied[name] = ()
    _require_neutral_parity(models)
    return applied


def _boundary(
    models: dict[str, Model],
    shared: SharedReplayBuffer,
    phase: str,
    epoch: int,
    global_epoch: int,
    manifest: ReplayPilotManifest,
) -> ReplayBoundaryResult:
    retention = shared.retention
    selected = shared.select_recent(manifest.replay_updates_per_boundary)
    if len(selected.sample_ids) != manifest.replay_updates_per_boundary:
        raise ValueError("P6.3 replay supply is shorter than planned")
    actual_ids = tuple(
        replay_sample_id(inputs, targets) for inputs, targets in selected.training_batches()
    )
    if actual_ids != selected.sample_ids:
        raise ValueError("P6.3 copied replay rows differ from selected IDs")
    for name in ("circadian_off", "circadian_on"):
        model = models[name]
        assert isinstance(model, CircadianPredictiveCodingNetwork)
        if (
            model.get_replay_retention() != retention
            or model.get_replay_retained_order_ids() != shared.retained_order_ids
            or model.preview_unprioritized_replay_ids(len(selected.sample_ids))
            != selected.sample_ids
        ):
            raise ValueError(f"P6.3 circadian memory differs from shared replay: {name}")
    applied = _apply_replay(models, selected, global_epoch)
    return ReplayBoundaryResult(
        phase,
        epoch,
        selected.sample_ids,
        shared.retained_order_ids,
        retention.example_count,
        retention.retained_bytes,
        applied,
    )


def _train_phase(
    models: dict[str, Model],
    roles: PhaseDecisionRoles,
    phase: str,
    manifest: ReplayPilotManifest,
    shared: SharedReplayBuffer,
) -> tuple[ReplayBoundaryResult, ...]:
    count = (
        manifest.source.training.phase_a_epochs
        if phase == "a"
        else manifest.source.training.phase_b_epochs
    )
    offset = 0 if phase == "a" else manifest.source.training.phase_a_epochs
    boundaries = []
    for epoch in range(1, count + 1):
        _train_wake(models, roles, manifest)
        shared.observe_train_batch(roles.train.input, roles.train.target)
        if epoch % manifest.interval == 0:
            boundaries.append(_boundary(models, shared, phase, epoch, offset + epoch, manifest))
    return tuple(boundaries)


def _accuracy(model: Model, roles: PhaseDecisionRoles) -> float:
    return float(
        np.mean(
            (model.predict_proba(roles.outer_selection.input) >= 0.5)
            == roles.outer_selection.target
        )
    )


def _method_result(
    name: str,
    model: Model,
    initial: str,
    a_after_a: float,
    phase_a: PhaseDecisionRoles,
    phase_b: PhaseDecisionRoles,
    manifest: ReplayPilotManifest,
) -> ReplayMethodResult:
    metrics = TwoTaskAccuracy(a_after_a, _accuracy(model, phase_a), _accuracy(model, phase_b))
    training = manifest.source.training
    presentations = training.phase_a_epochs * len(
        phase_a.train.input
    ) + training.phase_b_epochs * len(phase_b.train.input)
    is_pc = not isinstance(model, BackpropMLP)
    replay_count = 12 if name in ON_ARMS else 0
    width = manifest.planned_width if "width12" in name else manifest.width
    expected_parameters = 4 * width + 1
    if _parameter_count(model) != expected_parameters or (
        isinstance(model, CircadianPredictiveCodingNetwork) and model.hidden_dim != width
    ):
        raise ValueError(f"P6.3 replay factor model capacity differs: {name}")
    return ReplayMethodResult(
        name,
        initial,
        _parameter_hash(model),
        ReplayDevelopmentMetrics(
            metrics.a_after_a,
            metrics.a_after_b,
            metrics.b_after_b,
            metrics.final_mean_task_accuracy,
            metrics.signed_forgetting_a,
            metrics.retention_ratio_a,
        ),
        24,
        presentations,
        48 if is_pc else 0,
        2 * presentations if is_pc else 0,
        replay_count,
        replay_count,
        2 * replay_count if is_pc else 0,
        6 if name.startswith("circadian") else 0,
        width,
        width,
        expected_parameters,
        expected_parameters,
    )


def _run_seed(manifest: ReplayPilotManifest, seed: int) -> ReplaySeedResult:
    phase_a = arrived._build_phase_a_roles(manifest.source, seed)
    models = _models(seed, manifest)
    initial = {name: _parameter_hash(model) for name, model in models.items()}
    shared = SharedReplayBuffer(
        2, ReplayRetentionBudget(8, 192), ReplayRetentionPolicy("recent_fifo")
    )
    a_boundaries = _train_phase(models, phase_a, "a", manifest, shared)
    after_a = {name: _accuracy(model, phase_a) for name, model in models.items()}
    # Why this: the B source does not exist in the runner until all A work ends.
    phase_b = arrived._build_phase_b_roles(manifest.source, seed)
    b_boundaries = _train_phase(models, phase_b, "b", manifest, shared)
    if phase_a.final_released or phase_b.final_released:
        raise ValueError("P6.3 replay pilot released a final role")
    methods = tuple(
        _method_result(name, models[name], initial[name], after_a[name], phase_a, phase_b, manifest)
        for name in manifest.arms
    )
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
    return ReplaySeedResult(seed, hashes, counts, a_boundaries + b_boundaries, methods)
