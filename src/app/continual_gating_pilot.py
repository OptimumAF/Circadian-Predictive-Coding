"""Run the first development-only, fixed-width P6.3 mechanism pair.

Inputs are a frozen manifest and arrived development roles. Outputs are
matched work/role facts and outer-selection diagnostics. This module never
releases final roles, attempts sleep, or chooses a winning setting.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from typing import TypeAlias

import numpy as np

from src.app import continual_arrived_benchmark as arrived
from src.app.continual_shift_benchmark import ContinualGlobalSealConfig
from src.app.continual_trigger_replay_schedule import fixed_trigger_replay_manifest
from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork


PROTOCOL_ID = "continual_mechanism_gating_dev_v1"
ARMS = ("ordinary_pc", "neutral_circadian", "chemical_gating")
DEVELOPMENT_SEEDS = (41, 43, 59)
CONFIRMATION_SEEDS = (101, 103, 107, 109, 113, 127, 131, 137, 139, 149)
MAX_PLANNED_WAKE_UPDATES = 240
MODEL_SEED_OFFSET = 1001
Model: TypeAlias = PredictiveCodingNetwork | CircadianPredictiveCodingNetwork
PARAMETER_FIELDS = (
    "weight_input_hidden",
    "bias_hidden",
    "weight_hidden_output",
    "bias_output",
)


@dataclass(frozen=True)
class GatingPilotManifest:
    """Fix source roles, treatment arms, independent seeds, and work cap."""

    source: arrived.ContinualArrivedRolesConfig
    seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    arms: tuple[str, ...]
    max_planned_wake_updates: int
    protocol_id: str = PROTOCOL_ID


@dataclass(frozen=True)
class DevelopmentMetrics:
    """Explicit A/B outer-selection observations for one trained method."""

    a_after_a: float
    a_after_b: float
    b_after_b: float
    signed_forgetting: float
    final_mean_task_accuracy: float


@dataclass(frozen=True)
class GatingMethodResult:
    method: str
    initial_parameter_sha256: str
    final_parameter_sha256: str
    development: DevelopmentMetrics
    wake_updates: int
    train_presentations: int
    latent_iterations: int
    example_inference_iterations: int
    sleep_attempts: int
    replay_updates: int
    hidden_width_start: int
    hidden_width_end: int
    parameter_count_start: int
    parameter_count_end: int
    minimum_plasticity: float | None


@dataclass(frozen=True)
class GatingSeedResult:
    seed: int
    source_role_hashes: dict[str, str]
    source_role_counts: dict[str, int]
    methods: tuple[GatingMethodResult, ...]


@dataclass(frozen=True)
class GatingPilotResult:
    protocol_id: str
    manifest: GatingPilotManifest
    final_released: bool
    seed_results: tuple[GatingSeedResult, ...]


@dataclass
class _Models:
    ordinary: PredictiveCodingNetwork
    neutral: CircadianPredictiveCodingNetwork
    gating: CircadianPredictiveCodingNetwork


def fixed_gating_pilot_manifest() -> GatingPilotManifest:
    """Use the frozen v14 source geometry under a separate pilot ID."""
    return GatingPilotManifest(
        source=fixed_trigger_replay_manifest().arrived,
        seeds=DEVELOPMENT_SEEDS,
        confirmation_seeds=CONFIRMATION_SEEDS,
        arms=ARMS,
        max_planned_wake_updates=MAX_PLANNED_WAKE_UPDATES,
    )


def validate_gating_pilot_manifest(manifest: GatingPilotManifest) -> int:
    if type(manifest) is not GatingPilotManifest or manifest != fixed_gating_pilot_manifest():
        raise ValueError("P6.3 gating pilot requires its frozen manifest")
    if set(manifest.seeds) & set(manifest.confirmation_seeds):
        raise ValueError("P6.3 development and confirmation seeds overlap")
    arrived._validate_arrived_config(manifest.source, list(manifest.seeds))
    planned = (
        len(manifest.seeds)
        * len(manifest.arms)
        * (manifest.source.training.phase_a_epochs + manifest.source.training.phase_b_epochs)
    )
    if planned > manifest.max_planned_wake_updates:
        raise ValueError("P6.3 planned wake updates exceed the local cap")
    return planned


def run_gating_pilot(manifest: GatingPilotManifest) -> GatingPilotResult:
    """Train every development cell, then report its outer-role metrics."""
    validate_gating_pilot_manifest(manifest)
    results = tuple(_run_seed(manifest, seed) for seed in manifest.seeds)
    return GatingPilotResult(PROTOCOL_ID, manifest, False, results)


def _new_models(seed: int, width: int) -> _Models:
    model_seed = seed + MODEL_SEED_OFFSET
    neutral = CircadianConfig.matched_pc_control()
    models = _Models(
        ordinary=PredictiveCodingNetwork(2, width, model_seed),
        neutral=CircadianPredictiveCodingNetwork(
            2, width, model_seed, neutral, min_hidden_dim=width, max_hidden_dim=width
        ),
        gating=CircadianPredictiveCodingNetwork(
            2,
            width,
            model_seed,
            replace(neutral, min_plasticity=0.2),
            min_hidden_dim=width,
            max_hidden_dim=width,
        ),
    )
    _require_neutral_parity(models)
    if _parameter_hash(models.ordinary) != _parameter_hash(models.gating):
        raise ValueError("P6.3 gating model has unmatched initial parameters")
    return models


def _require_neutral_parity(models: _Models) -> None:
    if any(
        not np.array_equal(getattr(models.ordinary, name), getattr(models.neutral, name))
        for name in PARAMETER_FIELDS
    ):
        raise ValueError("P6.3 neutral circadian model diverged from ordinary PC")


def _parameter_hash(model: Model) -> str:
    digest = sha256(b"p63-shallow-parameter-tensors-v1")
    for name in PARAMETER_FIELDS:
        array = np.ascontiguousarray(getattr(model, name), dtype="<f8")
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _parameter_count(model: Model) -> int:
    return sum(int(getattr(model, name).size) for name in PARAMETER_FIELDS)


def _train_phase(models: _Models, inputs: np.ndarray, targets: np.ndarray, count: int) -> float:
    smallest = 1.0
    for _ in range(count):
        for model in (models.ordinary, models.neutral, models.gating):
            model.train_epoch(inputs, targets, 0.05, 2, 0.2)
        _require_neutral_parity(models)
        smallest = min(smallest, float(np.min(models.gating.get_plasticity_state())))
    return smallest


def _accuracy(model: Model, inputs: np.ndarray, targets: np.ndarray) -> float:
    value = float(np.mean((model.predict_proba(inputs) >= 0.5) == targets))
    if not np.isfinite(value):
        raise ValueError("P6.3 nonfinite development accuracy")
    return value


def _method_result(
    name: str,
    model: Model,
    initial: str,
    a_after_a: float,
    phase_a: arrived.PhaseDecisionRoles,
    phase_b: arrived.PhaseDecisionRoles,
    training: ContinualGlobalSealConfig,
    smallest_plasticity: float | None,
) -> GatingMethodResult:
    a_after_b = _accuracy(model, phase_a.outer_selection.input, phase_a.outer_selection.target)
    b_after_b = _accuracy(model, phase_b.outer_selection.input, phase_b.outer_selection.target)
    updates = training.phase_a_epochs + training.phase_b_epochs
    presentations = training.phase_a_epochs * len(
        phase_a.train.input
    ) + training.phase_b_epochs * len(phase_b.train.input)
    return GatingMethodResult(
        method=name,
        initial_parameter_sha256=initial,
        final_parameter_sha256=_parameter_hash(model),
        development=DevelopmentMetrics(
            a_after_a,
            a_after_b,
            b_after_b,
            a_after_a - a_after_b,
            (a_after_b + b_after_b) / 2.0,
        ),
        wake_updates=updates,
        train_presentations=presentations,
        latent_iterations=updates * training.pc_inference_steps,
        example_inference_iterations=2 * presentations,
        sleep_attempts=0,
        replay_updates=0,
        hidden_width_start=training.hidden_dim,
        hidden_width_end=training.hidden_dim,
        parameter_count_start=_parameter_count(model),
        parameter_count_end=_parameter_count(model),
        minimum_plasticity=smallest_plasticity,
    )


def _run_seed(manifest: GatingPilotManifest, seed: int) -> GatingSeedResult:
    source = manifest.source
    training = source.training
    phase_a = arrived._build_phase_a_roles(source, seed)
    models = _new_models(seed, training.hidden_dim)
    initial = _parameter_hash(models.ordinary)
    minimum = _train_phase(
        models, phase_a.train.input, phase_a.train.target, training.phase_a_epochs
    )
    after_a = {
        name: _accuracy(model, phase_a.outer_selection.input, phase_a.outer_selection.target)
        for name, model in (
            ("ordinary_pc", models.ordinary),
            ("neutral_circadian", models.neutral),
            ("chemical_gating", models.gating),
        )
    }
    # Why this: B may arrive only after every A update and A development score.
    phase_b = arrived._build_phase_b_roles(source, seed)
    minimum = min(
        minimum,
        _train_phase(models, phase_b.train.input, phase_b.train.target, training.phase_b_epochs),
    )
    if not minimum < 1.0:
        raise ValueError("P6.3 chemical gating did not become active")
    model_rows = (
        ("ordinary_pc", models.ordinary, None),
        ("neutral_circadian", models.neutral, 1.0),
        ("chemical_gating", models.gating, minimum),
    )
    methods = tuple(
        _method_result(name, model, initial, after_a[name], phase_a, phase_b, training, factor)
        for name, model, factor in model_rows
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
    return GatingSeedResult(seed, hashes, counts, methods)
