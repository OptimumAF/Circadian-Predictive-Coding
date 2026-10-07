"""Validate a frozen no-replay sleep-factor matrix before scoring.

Inputs are arrived A/B train and A inner-guard roles. Outputs are deterministic
role, guard, structure, capacity, and work facts. This app neither reads
outer/final values nor writes files or chooses a winning factor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from typing import TypeAlias

import numpy as np

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_shift_benchmark as base
from src.app.continual_trigger_replay_schedule import fixed_trigger_replay_manifest
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from src.core.sleep_telemetry import SleepEventTelemetry
from src.infra.continual_roles import PhaseDecisionRoles


PROTOCOL_ID = "continual_sleep_factor_train_only_v1"
DEVELOPMENT_SEEDS = (67, 71, 73)
CONFIRMATION_SEEDS = (151, 157, 163, 167, 173, 179, 181, 191, 193, 197)
ARMS = (
    "backprop_8",
    "pc_8",
    "neutral_sham",
    "structure_only",
    "homeostasis_only",
    "gating_sham",
    "gating_reset",
    "backprop_12",
    "pc_12",
)
SLEEP_ARMS = ("neutral_sham", "structure_only", "homeostasis_only", "gating_sham", "gating_reset")
GATING_ARMS = ("gating_sham", "gating_reset")
WIDTH = 8
PLANNED_WIDTH = 12
MODEL_SEED_OFFSET = 1001
EPOCHS_PER_PHASE = 12
MAX_OPTIMIZER_UPDATES = 700
PARAMETER_FIELDS = ("weight_input_hidden", "bias_hidden", "weight_hidden_output", "bias_output")
Model: TypeAlias = BackpropMLP | PredictiveCodingNetwork | CircadianPredictiveCodingNetwork


def _neutral_config() -> CircadianConfig:
    return replace(
        CircadianConfig.matched_pc_control(),
        sleep_mode="components",
        sleep_enable_chemical_reset=False,
        sleep_enable_replay=False,
        sleep_enable_homeostasis=False,
        sleep_enable_split=False,
        sleep_enable_prune=False,
        replay_steps=0,
        replay_memory_size=0,
    )


def _arm_config(name: str) -> CircadianConfig:
    neutral = _neutral_config()
    if name == "structure_only":
        return replace(
            neutral,
            sleep_enable_split=True,
            sleep_enable_prune=True,
            max_split_per_sleep=1,
            max_prune_per_sleep=1,
            split_threshold=0.0,
            prune_threshold=1.0,
            split_noise_scale=0.02,
            prune_decay_steps=1,
        )
    if name == "homeostasis_only":
        return replace(neutral, sleep_enable_homeostasis=True, homeostatic_downscale_factor=0.99)
    if name == "gating_sham":
        return replace(neutral, min_plasticity=0.2)
    if name == "gating_reset":
        return replace(
            neutral,
            min_plasticity=0.2,
            sleep_enable_chemical_reset=True,
            sleep_reset_factor=0.45,
        )
    if name == "neutral_sham":
        return neutral
    raise ValueError(f"unknown sleep-factor circadian arm: {name}")


@dataclass(frozen=True)
class SleepFactorManifest:
    source: arrived.ContinualArrivedRolesConfig
    seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    arms: tuple[str, ...]
    width: int
    planned_width: int
    epochs_per_phase: int
    guard_drop_tolerance: float
    max_optimizer_updates: int
    max_process_rss_bytes: int
    protocol_id: str = PROTOCOL_ID


@dataclass(frozen=True)
class GuardedSleepFacts:
    outcome: str
    guard_role_hash: str
    guard_pre_accuracy: float
    guard_post_accuracy: float
    guard_evaluations: int
    proposed_split_pairs: tuple[tuple[int, int], ...]
    applied_split_pairs: tuple[tuple[int, int], ...]
    proposed_removed_prune_ids: tuple[int, ...]
    applied_removed_prune_ids: tuple[int, ...]
    width_before: int
    proposed_width: int
    width_after: int
    transient_peak_width: int
    replay_updates: int
    chemical_before_mean: float
    chemical_proposed_mean: float
    chemical_final_mean: float


@dataclass(frozen=True)
class SleepArmFacts:
    name: str
    initial_parameter_sha256: str
    pre_sleep_parameter_sha256: str
    post_sleep_parameter_sha256: str
    final_parameter_sha256: str
    width_initial: int
    width_final: int
    width_peak: int
    parameters_initial: int
    parameters_final: int
    parameters_peak: int
    wake_updates: int
    wake_presentations: int
    latent_inference_loops: int
    example_inference_iterations: int
    replay_updates: int
    minimum_a_plasticity: float | None
    sleep: GuardedSleepFacts | None


@dataclass(frozen=True)
class SleepSeedFacts:
    seed: int
    role_hashes: dict[str, str]
    role_counts: dict[str, int]
    arms: tuple[SleepArmFacts, ...]
    final_released: bool


@dataclass(frozen=True)
class SleepFactorPreflight:
    protocol_id: str
    manifest: SleepFactorManifest
    seed_results: tuple[SleepSeedFacts, ...]
    final_released: bool
    outer_selection_scored: bool


def fixed_sleep_factor_manifest() -> SleepFactorManifest:
    historical = fixed_trigger_replay_manifest().arrived
    training = replace(
        historical.training,
        circadian_sleep_interval_phase_a=EPOCHS_PER_PHASE,
        circadian_sleep_interval_phase_b=EPOCHS_PER_PHASE,
    )
    source = replace(historical, training=training, guard_drop_tolerance=0.0)
    return SleepFactorManifest(
        source,
        DEVELOPMENT_SEEDS,
        CONFIRMATION_SEEDS,
        ARMS,
        WIDTH,
        PLANNED_WIDTH,
        EPOCHS_PER_PHASE,
        0.0,
        MAX_OPTIMIZER_UPDATES,
        256 * 1024 * 1024,
    )


def validate_sleep_factor_manifest(manifest: SleepFactorManifest) -> int:
    if type(manifest) is not SleepFactorManifest or manifest != fixed_sleep_factor_manifest():
        raise ValueError("P6.3 sleep factor requires its frozen manifest")
    if set(manifest.seeds) & set(manifest.confirmation_seeds):
        raise ValueError("P6.3 sleep factor development and confirmation seeds overlap")
    arrived._validate_arrived_config(manifest.source, list(manifest.seeds))
    training = manifest.source.training
    if (
        training.hidden_dim != manifest.width
        or training.phase_a_epochs != manifest.epochs_per_phase
        or training.phase_b_epochs != manifest.epochs_per_phase
        or training.circadian_sleep_interval_phase_a != manifest.epochs_per_phase
        or manifest.guard_drop_tolerance != manifest.source.guard_drop_tolerance
    ):
        raise ValueError("P6.3 sleep factor source work or guard differs")
    planned = len(manifest.seeds) * len(manifest.arms) * 2 * manifest.epochs_per_phase
    if planned > manifest.max_optimizer_updates:
        raise ValueError("P6.3 sleep factor exceeds local optimizer cap")
    return planned


def run_sleep_factor_preflight(manifest: SleepFactorManifest) -> SleepFactorPreflight:
    """Train all fixed cells and return no outer-selection or final score."""
    validate_sleep_factor_manifest(manifest)
    return SleepFactorPreflight(
        PROTOCOL_ID,
        manifest,
        tuple(_run_seed(manifest, seed) for seed in manifest.seeds),
        False,
        False,
    )


def _parameter_hash(model: Model) -> str:
    digest = sha256(b"p63-sleep-factor-shallow-parameters-v1")
    for name in PARAMETER_FIELDS:
        array = np.ascontiguousarray(getattr(model, name), dtype="<f8")
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _parameter_count(model: Model) -> int:
    return sum(int(getattr(model, name).size) for name in PARAMETER_FIELDS)


def _new_models(seed: int, manifest: SleepFactorManifest) -> dict[str, Model]:
    model_seed = seed + MODEL_SEED_OFFSET
    models: dict[str, Model] = {
        "backprop_8": BackpropMLP(2, manifest.width, model_seed),
        "pc_8": PredictiveCodingNetwork(2, manifest.width, model_seed),
        "backprop_12": BackpropMLP(2, manifest.planned_width, model_seed),
        "pc_12": PredictiveCodingNetwork(2, manifest.planned_width, model_seed),
    }
    for name in SLEEP_ARMS:
        models[name] = CircadianPredictiveCodingNetwork(
            2,
            manifest.width,
            model_seed,
            _arm_config(name),
            min_hidden_dim=manifest.width - 1 if name == "structure_only" else manifest.width,
            max_hidden_dim=manifest.width + 1 if name == "structure_only" else manifest.width,
        )
    if len({_parameter_hash(models[name]) for name in manifest.arms[:7]}) != 1:
        raise ValueError("P6.3 sleep factor width-eight initialization differs")
    if _parameter_hash(models["backprop_12"]) != _parameter_hash(models["pc_12"]):
        raise ValueError("P6.3 sleep factor planned-width initialization differs")
    return models


def _require_parameter_parity(models: dict[str, Model], left: str, right: str) -> None:
    if any(
        not np.array_equal(getattr(models[left], field), getattr(models[right], field))
        for field in PARAMETER_FIELDS
    ):
        raise ValueError(f"P6.3 sleep factor parameter parity differs: {left}/{right}")


def _train_phase(
    models: dict[str, Model],
    roles: PhaseDecisionRoles,
    manifest: SleepFactorManifest,
    *,
    before_sleep: bool,
) -> None:
    training = manifest.source.training
    for _ in range(manifest.epochs_per_phase):
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
        _require_parameter_parity(models, "pc_8", "neutral_sham")
        if before_sleep:
            _require_parameter_parity(models, "gating_sham", "gating_reset")


def _sleep_facts(event: SleepEventTelemetry, roles: PhaseDecisionRoles) -> GuardedSleepFacts:
    guard = event.guard
    if (
        guard is None
        or guard.role_hash != roles.split_hashes["inner_guard"]
        or guard.examples_scored != 2 * len(roles.inner_guard.input)
        or guard.pre_accuracy is None
        or guard.post_accuracy is None
        or event.outcome not in {"accepted", "rolled_back"}
        or event.completed_epoch != EPOCHS_PER_PHASE
        or event.replay.proposed_updates
        or event.replay.applied_updates
        or (event.outcome == "accepted" and guard.post_accuracy < guard.pre_accuracy)
        or (event.outcome == "rolled_back" and guard.post_accuracy >= guard.pre_accuracy)
    ):
        raise ValueError("P6.3 sleep factor guard, role, or replay facts differ")
    changes = event.changes
    transient = event.before_width + len(changes.proposed_split_pairs)
    return GuardedSleepFacts(
        event.outcome,
        roles.split_hashes["inner_guard"],
        guard.pre_accuracy,
        guard.post_accuracy,
        2,
        changes.proposed_split_pairs,
        changes.applied_split_pairs,
        changes.proposed_removed_prune_ids,
        changes.applied_removed_prune_ids,
        event.before_width,
        event.proposed_width,
        event.final_width,
        transient,
        event.replay.applied_updates,
        event.chemistry_before.primary.mean,
        event.chemistry_proposed.primary.mean,
        event.chemistry_final.primary.mean,
    )


def _apply_guarded_sleep(
    models: dict[str, Model], roles: PhaseDecisionRoles, manifest: SleepFactorManifest
) -> dict[str, GuardedSleepFacts]:
    facts = {}
    for name in SLEEP_ARMS:
        model = models[name]
        if not isinstance(model, CircadianPredictiveCodingNetwork):
            raise TypeError(f"P6.3 sleep factor arm is not circadian: {name}")
        parameters_before = _parameter_hash(model)
        events: list[SleepEventTelemetry] = []
        accepted_count, split_count, prune_count = base._apply_scheduled_sleep(
            model,
            manifest.epochs_per_phase,
            manifest.epochs_per_phase,
            manifest.epochs_per_phase,
            2 * manifest.epochs_per_phase,
            True,
            0,
            0,
            0,
            guard=roles.inner_guard,
            guard_drop_tolerance=manifest.guard_drop_tolerance,
            on_sleep_event=events.append,
            guard_role_hash=roles.split_hashes["inner_guard"],
        )
        if len(events) != 1:
            raise ValueError(f"P6.3 sleep factor expected one A-boundary event: {name}")
        fact = _sleep_facts(events[0], roles)
        if (
            accepted_count != int(fact.outcome == "accepted")
            or split_count != len(fact.applied_split_pairs)
            or prune_count != len(fact.applied_removed_prune_ids)
            or model.hidden_dim != fact.width_after
        ):
            raise ValueError(f"P6.3 sleep factor committed work differs: {name}")
        if name != "structure_only" and (
            fact.proposed_split_pairs
            or fact.proposed_removed_prune_ids
            or model.hidden_dim != manifest.width
        ):
            raise ValueError(f"P6.3 nonstructural arm changed topology: {name}")
        if name == "structure_only" and fact.transient_peak_width > manifest.width + 1:
            raise ValueError("P6.3 structural transient width exceeds cap")
        if name == "structure_only" and (
            len(fact.proposed_split_pairs) != 1 or len(fact.proposed_removed_prune_ids) != 1
        ):
            raise ValueError("P6.3 structural factor proposal was inactive")
        if fact.outcome == "rolled_back" and _parameter_hash(model) != parameters_before:
            raise ValueError(f"P6.3 rejected sleep changed parameters: {name}")
        if name in {"neutral_sham", "gating_sham", "gating_reset"} and (
            _parameter_hash(model) != parameters_before
        ):
            raise ValueError(f"P6.3 no-weight sleep changed parameters: {name}")
        if (
            name == "homeostasis_only"
            and fact.outcome == "accepted"
            and (_parameter_hash(model) == parameters_before)
        ):
            raise ValueError("P6.3 accepted homeostasis did not change parameters")
        if (
            name == "gating_reset"
            and fact.outcome == "accepted"
            and not (0.0 < fact.chemical_proposed_mean < fact.chemical_before_mean)
        ):
            raise ValueError("P6.3 accepted chemical reset was inactive")
        facts[name] = fact
    _require_parameter_parity(models, "pc_8", "neutral_sham")
    _require_parameter_parity(models, "gating_sham", "gating_reset")
    return facts


def _arm_facts(
    name: str,
    model: Model,
    initial: str,
    pre_sleep: str,
    post_sleep: str,
    minimum_a_plasticity: float | None,
    sleep: GuardedSleepFacts | None,
    roles_a: PhaseDecisionRoles,
    roles_b: PhaseDecisionRoles,
    manifest: SleepFactorManifest,
) -> SleepArmFacts:
    width = manifest.planned_width if name.endswith("_12") else manifest.width
    peak_width = max(width, sleep.transient_peak_width if sleep is not None else width)
    count = 4 * width + 1
    final_count = _parameter_count(model)
    if (
        final_count
        != 4 * (model.hidden_dim if isinstance(model, CircadianPredictiveCodingNetwork) else width)
        + 1
    ):
        raise ValueError(f"P6.3 sleep factor parameter count differs: {name}")
    if name != "structure_only" and (peak_width != width or final_count != count):
        raise ValueError(f"P6.3 sleep factor fixed capacity differs: {name}")
    training = manifest.source.training
    presentations = training.phase_a_epochs * len(
        roles_a.train.input
    ) + training.phase_b_epochs * len(roles_b.train.input)
    is_pc = not isinstance(model, BackpropMLP)
    if isinstance(model, CircadianPredictiveCodingNetwork):
        clocks = model.get_sleep_clocks()
        if (
            clocks.wake_batches != 2 * manifest.epochs_per_phase
            or clocks.wake_examples != presentations
            or clocks.replay_updates != 0
            or clocks.sleep_events != int(sleep is not None and sleep.outcome == "accepted")
        ):
            raise ValueError(f"P6.3 sleep factor observed work differs: {name}")
    return SleepArmFacts(
        name,
        initial,
        pre_sleep,
        post_sleep,
        _parameter_hash(model),
        width,
        model.hidden_dim if isinstance(model, CircadianPredictiveCodingNetwork) else width,
        peak_width,
        count,
        final_count,
        4 * peak_width + 1,
        2 * manifest.epochs_per_phase,
        presentations,
        2 * manifest.epochs_per_phase * training.pc_inference_steps if is_pc else 0,
        presentations * training.pc_inference_steps if is_pc else 0,
        0,
        minimum_a_plasticity,
        sleep,
    )


def _run_seed(manifest: SleepFactorManifest, seed: int) -> SleepSeedFacts:
    phase_a = arrived._build_phase_a_roles(manifest.source, seed)
    models = _new_models(seed, manifest)
    initial = {name: _parameter_hash(model) for name, model in models.items()}
    _train_phase(models, phase_a, manifest, before_sleep=True)
    pre_sleep = {name: _parameter_hash(model) for name, model in models.items()}
    minimum_a_plasticity = {
        name: float(np.min(model.get_plasticity_state()))
        for name in GATING_ARMS
        if isinstance(model := models[name], CircadianPredictiveCodingNetwork)
    }
    if any(not 0.2 <= value < 1.0 for value in minimum_a_plasticity.values()):
        raise ValueError("P6.3 conditional reset pilot chemical gate was inactive")
    sleeps = _apply_guarded_sleep(models, phase_a, manifest)
    post_sleep = {name: _parameter_hash(model) for name, model in models.items()}
    # Why this: B source construction happens only after every A wake and guard decision.
    phase_b = arrived._build_phase_b_roles(manifest.source, seed)
    _train_phase(models, phase_b, manifest, before_sleep=False)
    if phase_a.final_released or phase_b.final_released:
        raise ValueError("P6.3 sleep factor released final data")
    roles = (("a", phase_a), ("b", phase_b))
    hashes = {
        f"{phase}_{role}": bound.split_hashes[role]
        for phase, bound in roles
        for role in ("train", "inner_guard", "outer_selection")
    }
    counts = {
        f"{phase}_{role}": len(bound.sample_ids[role])
        for phase, bound in roles
        for role in ("train", "inner_guard", "outer_selection")
    }
    arms = tuple(
        _arm_facts(
            name,
            models[name],
            initial[name],
            pre_sleep[name],
            post_sleep[name],
            minimum_a_plasticity.get(name),
            sleeps.get(name),
            phase_a,
            phase_b,
            manifest,
        )
        for name in manifest.arms
    )
    return SleepSeedFacts(seed, hashes, counts, arms, False)
