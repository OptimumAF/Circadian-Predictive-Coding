"""Train one fixed v13 sleep-timing arm without opening final roles.

Inputs are pre-split phase roles and a frozen manifest. Outputs are
unscored model states, exact train-stream hashes, decisions, and work.
The existing core owns wake updates and sleep; this app owns scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from typing import Any

import numpy as np

from src.app.sleep_schedule import decide_sleep_attempt
from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.sleep_clocks import SleepEpochProgress
from src.infra.continual_roles import PhaseDecisionRoles
from src.infra.datasets import LabeledData

PROTOCOL_ID = "sleep_trigger_timing_v13"
ARMS = ("periodic", "adaptive", "no_sleep")


@dataclass(frozen=True)
class TriggerManifest:
    protocol_id: str = PROTOCOL_ID
    seeds: tuple[int, ...] = (41, 43)
    conditions: tuple[str, ...] = ("stationary_noise", "axis_shift")
    arms: tuple[str, ...] = ARMS
    hidden_dim: int = 8
    epochs_per_phase: int = 16
    periodic_interval: int = 8
    learning_rate: float = 0.05
    inference_steps: int = 2
    inference_learning_rate: float = 0.1
    train_jitter_std: float = 0.15
    inner_guard_fraction: float = 0.2
    outer_selection_fraction: float = 0.2
    final_count: int = 40


@dataclass(frozen=True)
class TriggerDecisionFacts:
    phase: str
    phase_epoch: int
    global_epoch: int
    energy: float
    energy_improvement: float | None
    chemical_variance: float
    wake_batches_since_sleep: int
    periodic_due: bool
    adaptive_due: bool
    attempted: bool
    force_sleep: bool
    performed: bool
    trigger_reason: str
    chemical_mean_before: float
    chemical_mean_after: float
    parameter_digest_before: str
    parameter_digest_after: str


@dataclass(frozen=True)
class TriggerWork:
    wake_updates: int
    wake_examples: int
    inference_loops: int
    example_inference_loops: int
    replay_updates: int
    sleep_events: int
    decision_opportunities: int
    sleep_attempts: int


@dataclass(frozen=True)
class UnscoredTriggerFacts:
    seed: int
    condition: str
    arm: str
    role_hashes: dict[str, str]
    train_batch_hashes: dict[str, tuple[str, ...]]
    initial_parameter_digest: str
    post_a_parameter_digest: str
    post_b_parameter_digest: str
    decisions: tuple[TriggerDecisionFacts, ...]
    a_work: TriggerWork
    total_work: TriggerWork
    initial_width: int
    post_a_width: int
    final_width: int
    initial_parameter_count: int
    post_a_parameter_count: int
    final_parameter_count: int


@dataclass
class TrainedTriggerTrial:
    facts: UnscoredTriggerFacts
    post_a_state: Any
    post_b_model: CircadianPredictiveCodingNetwork


def fixed_trigger_manifest() -> TriggerManifest:
    """Return the sole prospective v13 schedule without tuning options."""
    return TriggerManifest()


def make_trigger_model(
    manifest: TriggerManifest, seed: int, arm: str
) -> CircadianPredictiveCodingNetwork:
    if arm not in ARMS:
        raise ValueError("unknown v13 trigger arm")
    config = CircadianConfig(
        sleep_mode="components",
        use_adaptive_sleep_trigger=arm == "adaptive",
        use_adaptive_sleep_budget=False,
        use_reward_modulated_learning=False,
        max_split_per_sleep=0,
        max_prune_per_sleep=0,
        sleep_enable_split=False,
        sleep_enable_prune=False,
        sleep_enable_replay=False,
        sleep_enable_homeostasis=False,
        sleep_enable_chemical_reset=True,
        replay_steps=0,
        replay_memory_size=0,
    )
    return CircadianPredictiveCodingNetwork(
        2,
        manifest.hidden_dim,
        seed,
        circadian_config=config,
        min_hidden_dim=manifest.hidden_dim,
        max_hidden_dim=manifest.hidden_dim,
    )


def parameter_digest(model: CircadianPredictiveCodingNetwork) -> str:
    """Hash only trainable arrays; chemistry is the intended sleep effect."""
    digest = sha256(b"sleep_trigger_parameters_v13\0")
    for values in (
        model.weight_input_hidden,
        model.bias_hidden,
        model.weight_hidden_output,
        model.bias_output,
    ):
        canonical = np.ascontiguousarray(values, dtype="<f8")
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def parameter_count(model: CircadianPredictiveCodingNetwork) -> int:
    return int(
        sum(
            values.size
            for values in (
                model.weight_input_hidden,
                model.bias_hidden,
                model.weight_hidden_output,
                model.bias_output,
            )
        )
    )


def _hash_batch(phase: str, sample_ids: tuple[str, ...], batch: LabeledData) -> str:
    digest = sha256(f"sleep_trigger_batch_v13/{phase}\0".encode("ascii"))
    for sample_id in sample_ids:
        digest.update(sample_id.encode("ascii") + b"\0")
    for values in (batch.input, batch.target):
        canonical = np.ascontiguousarray(values, dtype="<f8")
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def build_epoch_batches(
    manifest: TriggerManifest, roles: PhaseDecisionRoles
) -> tuple[tuple[LabeledData, ...], tuple[str, ...]]:
    """Freeze an arm-independent noisy train stream from train rows only."""
    offset = 1 if roles.phase == "a" else 2
    rng = np.random.default_rng(1000 * roles.seed + offset)
    batches = tuple(
        LabeledData(
            roles.train.input
            + rng.normal(0.0, manifest.train_jitter_std, size=roles.train.input.shape),
            roles.train.target.copy(),
        )
        for _ in range(manifest.epochs_per_phase)
    )
    hashes = tuple(_hash_batch(roles.phase, roles.sample_ids["train"], batch) for batch in batches)
    return batches, hashes


def _current_work(
    model: CircadianPredictiveCodingNetwork,
    decisions: list[TriggerDecisionFacts],
    manifest: TriggerManifest,
) -> TriggerWork:
    clocks = model.get_sleep_clocks()
    return TriggerWork(
        wake_updates=clocks.wake_batches,
        wake_examples=clocks.wake_examples,
        inference_loops=clocks.wake_batches * manifest.inference_steps,
        example_inference_loops=clocks.wake_examples * manifest.inference_steps,
        replay_updates=clocks.replay_updates,
        sleep_events=clocks.sleep_events,
        decision_opportunities=len(decisions),
        sleep_attempts=sum(decision.attempted for decision in decisions),
    )


def _record_decision(
    manifest: TriggerManifest,
    model: CircadianPredictiveCodingNetwork,
    arm: str,
    phase: str,
    phase_epoch: int,
    global_epoch: int,
    energy: float,
    energy_history: list[float],
) -> TriggerDecisionFacts:
    window = model.config.sleep_energy_window
    improvement = (
        energy_history[-window] - energy_history[-1] if len(energy_history) >= window else None
    )
    chemical_before = model.get_chemical_state()
    variance = float(np.var(chemical_before))
    since_sleep = model.get_sleep_clocks().wake_batches_since_sleep
    adaptive_due = model.should_trigger_sleep()
    expected_due = (
        arm == "adaptive"
        and since_sleep >= model.config.min_epochs_between_sleep
        and improvement is not None
        and improvement <= model.config.sleep_plateau_delta
        and variance >= model.config.sleep_chemical_variance_threshold
    )
    if adaptive_due != expected_due or not isfinite(energy) or not isfinite(variance):
        raise ValueError("v13 adaptive trigger facts disagree with the core")
    decision = decide_sleep_attempt(
        sleep_mode=model.config.sleep_mode,
        completed_epochs=global_epoch,
        interval_epochs=manifest.periodic_interval if arm == "periodic" else 0,
        adaptive_due=adaptive_due,
        force_periodic=arm == "periodic",
    )
    before_digest = parameter_digest(model)
    performed = False
    reason = "not_due"
    if decision.attempted:
        result = model.sleep_event(
            force_sleep=decision.force_sleep,
            epoch_progress=SleepEpochProgress(global_epoch, 2 * manifest.epochs_per_phase),
        )
        telemetry = result.telemetry
        if telemetry is None:
            raise ValueError("v13 sleep attempt lacks typed telemetry")
        performed = result.performed
        reason = telemetry.trigger_reason
        if (
            not performed
            or telemetry.outcome != "applied"
            or telemetry.changes.applied_split_pairs
            or telemetry.changes.applied_removed_prune_ids
            or telemetry.replay.applied_updates
            or result.new_hidden_dim != manifest.hidden_dim
        ):
            raise ValueError("v13 component sleep exceeded fixed-capacity policy")
    chemical_after = model.get_chemical_state()
    after_digest = parameter_digest(model)
    if before_digest != after_digest or not np.allclose(
        chemical_after,
        chemical_before * (model.config.sleep_reset_factor if performed else 1.0),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("v13 sleep changed weights or unexpected chemistry")
    return TriggerDecisionFacts(
        phase=phase,
        phase_epoch=phase_epoch,
        global_epoch=global_epoch,
        energy=energy,
        energy_improvement=improvement,
        chemical_variance=variance,
        wake_batches_since_sleep=since_sleep,
        periodic_due=decision.periodic_due,
        adaptive_due=decision.adaptive_due,
        attempted=decision.attempted,
        force_sleep=decision.force_sleep,
        performed=performed,
        trigger_reason=reason,
        chemical_mean_before=float(np.mean(chemical_before)),
        chemical_mean_after=float(np.mean(chemical_after)),
        parameter_digest_before=before_digest,
        parameter_digest_after=after_digest,
    )


def train_trigger_trial(
    manifest: TriggerManifest,
    roles: dict[str, PhaseDecisionRoles],
    *,
    seed: int,
    condition: str,
    arm: str,
) -> TrainedTriggerTrial:
    """Train A then B with all decision opportunities, returning no score."""
    model = make_trigger_model(manifest, seed, arm)
    initial_digest = parameter_digest(model)
    initial_count = parameter_count(model)
    decisions: list[TriggerDecisionFacts] = []
    energy_history: list[float] = []
    hashes: dict[str, tuple[str, ...]] = {}
    post_a_state: Any = None
    post_a_digest = ""
    post_a_work: TriggerWork | None = None
    post_a_width = 0
    post_a_count = 0
    for phase_index, phase in enumerate(("a", "b")):
        batches, hashes[phase] = build_epoch_batches(manifest, roles[phase])
        for phase_epoch, batch in enumerate(batches, start=1):
            energy = float(
                model.train_epoch(
                    batch.input,
                    batch.target,
                    manifest.learning_rate,
                    manifest.inference_steps,
                    manifest.inference_learning_rate,
                ).energy
            )
            energy_history.append(energy)
            decisions.append(
                _record_decision(
                    manifest,
                    model,
                    arm,
                    phase,
                    phase_epoch,
                    phase_index * manifest.epochs_per_phase + phase_epoch,
                    energy,
                    energy_history,
                )
            )
        if phase == "a":
            post_a_state = model.snapshot_state()
            post_a_digest = parameter_digest(model)
            post_a_work = _current_work(model, decisions, manifest)
            post_a_width = model.hidden_dim
            post_a_count = parameter_count(model)
    if post_a_work is None:
        raise ValueError("v13 phase A did not finish")
    facts = UnscoredTriggerFacts(
        seed=seed,
        condition=condition,
        arm=arm,
        role_hashes={
            f"{phase}/{role}": phase_roles.split_hashes[role]
            for phase, phase_roles in roles.items()
            for role in ("train", "inner_guard", "outer_selection")
        },
        train_batch_hashes=hashes,
        initial_parameter_digest=initial_digest,
        post_a_parameter_digest=post_a_digest,
        post_b_parameter_digest=parameter_digest(model),
        decisions=tuple(decisions),
        a_work=post_a_work,
        total_work=_current_work(model, decisions, manifest),
        initial_width=manifest.hidden_dim,
        post_a_width=post_a_width,
        final_width=model.hidden_dim,
        initial_parameter_count=initial_count,
        post_a_parameter_count=post_a_count,
        final_parameter_count=parameter_count(model),
    )
    return TrainedTriggerTrial(facts, post_a_state, model)
