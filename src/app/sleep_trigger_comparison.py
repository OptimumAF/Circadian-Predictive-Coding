"""Preflight and globally score the fixed v13 sleep-timing controls.

The app derives roles and exact train streams, verifies all twelve
unscored trials, then releases shared final roles. It neither writes
files nor selects a trigger policy from the results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isclose, isfinite
from typing import Callable

import numpy as np

from src.app.sleep_trigger_trial import (
    PROTOCOL_ID,
    TrainedTriggerTrial,
    TriggerManifest,
    TriggerWork,
    UnscoredTriggerFacts,
    build_epoch_batches,
    fixed_trigger_manifest,
    make_trigger_model,
    parameter_count,
    parameter_digest,
    train_trigger_trial,
)
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.infra.continual_roles import (
    PhaseDecisionRoles,
    PhaseSource,
    release_final_test,
    split_phase_decision_roles,
)
from src.infra.datasets import LabeledData
from src.infra.trigger_streams import TriggerPhaseSource


@dataclass
class UnscoredTriggerStudy:
    manifest: TriggerManifest
    roles: dict[tuple[str, int], dict[str, PhaseDecisionRoles]]
    trials: tuple[TrainedTriggerTrial, ...]


@dataclass(frozen=True)
class TriggerOutcome:
    train_facts: UnscoredTriggerFacts
    final_role_hashes: dict[str, str]
    a_after_a_accuracy: float
    a_after_a_bce: float
    a_after_b_accuracy: float
    a_after_b_bce: float
    b_after_b_accuracy: float
    b_after_b_bce: float
    forgetting: float


@dataclass(frozen=True)
class TriggerComparison:
    protocol_id: str
    manifest: TriggerManifest
    manifest_digest: str
    outcomes: tuple[TriggerOutcome, ...]


def _validate_manifest(manifest: TriggerManifest) -> None:
    if type(manifest) is not TriggerManifest or manifest != fixed_trigger_manifest():
        raise ValueError("v13 trigger manifest differs from the predeclared protocol")


def manifest_digest(manifest: TriggerManifest) -> str:
    canonical = json.dumps(asdict(manifest), sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _make_roles(
    manifest: TriggerManifest,
    source_factory: Callable[[int, str, str], PhaseSource],
) -> dict[tuple[str, int], dict[str, PhaseDecisionRoles]]:
    return {
        (condition, seed): {
            phase: split_phase_decision_roles(
                source_factory(seed, phase, condition),
                phase=phase,
                seed=seed,
                split_seed=seed * 10 + (1 if phase == "a" else 2),
                inner_guard_fraction=manifest.inner_guard_fraction,
                outer_selection_fraction=manifest.outer_selection_fraction,
                expected_final_count=manifest.final_count,
            )
            for phase in ("a", "b")
        }
        for condition in manifest.conditions
        for seed in manifest.seeds
    }


def _expected_work(
    manifest: TriggerManifest,
    *,
    after_a: bool,
    performed: int,
    attempts: int,
) -> TriggerWork:
    updates = manifest.epochs_per_phase * (1 if after_a else 2)
    examples = updates * 24
    return TriggerWork(
        wake_updates=updates,
        wake_examples=examples,
        inference_loops=updates * manifest.inference_steps,
        example_inference_loops=examples * manifest.inference_steps,
        replay_updates=0,
        sleep_events=performed,
        decision_opportunities=updates,
        sleep_attempts=attempts,
    )


def _preflight_decisions(manifest: TriggerManifest, facts: UnscoredTriggerFacts) -> None:
    decisions = facts.decisions
    if len(decisions) != 2 * manifest.epochs_per_phase:
        raise ValueError("v13 decision opportunity count changed")
    last_sleep = 0
    for index, decision in enumerate(decisions, start=1):
        phase = "a" if index <= manifest.epochs_per_phase else "b"
        phase_epoch = (index - 1) % manifest.epochs_per_phase + 1
        window = decisions[max(0, index - 8) : index]
        improvement = window[0].energy - window[-1].energy if len(window) == 8 else None
        expected_adaptive = (
            facts.arm == "adaptive"
            and index - last_sleep >= 10
            and improvement is not None
            and improvement <= 1e-3
            and decision.chemical_variance >= 0.02
        )
        periodic = facts.arm == "periodic" and index % manifest.periodic_interval == 0
        attempted = periodic or expected_adaptive
        scale = 0.45 if attempted else 1.0
        improvement_mismatch = (
            decision.energy_improvement is not None
            if improvement is None
            else decision.energy_improvement is None
            or not isclose(decision.energy_improvement, improvement, rel_tol=1e-12, abs_tol=1e-12)
        )
        if (
            decision.phase != phase
            or decision.phase_epoch != phase_epoch
            or decision.global_epoch != index
            or not isfinite(decision.energy)
            or not isfinite(decision.chemical_variance)
            or decision.chemical_variance < 0.0
            or decision.wake_batches_since_sleep != index - last_sleep
            or decision.periodic_due != periodic
            or decision.adaptive_due != expected_adaptive
            or decision.attempted != attempted
            or decision.force_sleep != periodic
            or decision.performed != attempted
            or decision.trigger_reason
            != ("forced" if periodic else "adaptive" if attempted else "not_due")
            or decision.parameter_digest_before != decision.parameter_digest_after
            or not isclose(
                decision.chemical_mean_after,
                decision.chemical_mean_before * scale,
                rel_tol=1e-11,
                abs_tol=1e-12,
            )
            or improvement_mismatch
        ):
            raise ValueError("v13 trigger decision trace failed preflight")
        if attempted:
            last_sleep = index
    if (
        facts.post_a_parameter_digest
        != decisions[manifest.epochs_per_phase - 1].parameter_digest_after
        or facts.post_b_parameter_digest != decisions[-1].parameter_digest_after
    ):
        raise ValueError("v13 phase model digest does not match decisions")


def _preflight_trial(
    manifest: TriggerManifest,
    trial: TrainedTriggerTrial,
    roles: dict[str, PhaseDecisionRoles],
) -> None:
    facts = trial.facts
    expected_hashes = {
        f"{phase}/{role}": phase_roles.split_hashes[role]
        for phase, phase_roles in roles.items()
        for role in ("train", "inner_guard", "outer_selection")
    }
    expected_batches = {
        phase: build_epoch_batches(manifest, phase_roles)[1] for phase, phase_roles in roles.items()
    }
    first_half = facts.decisions[: manifest.epochs_per_phase]
    if (
        len(roles["a"].train.input) != 24
        or len(roles["b"].train.input) != 24
        or facts.role_hashes != expected_hashes
        or facts.train_batch_hashes != expected_batches
        or any(
            width != manifest.hidden_dim
            for width in (facts.initial_width, facts.post_a_width, facts.final_width)
        )
        or any(
            count != 4 * manifest.hidden_dim + 1
            for count in (
                facts.initial_parameter_count,
                facts.post_a_parameter_count,
                facts.final_parameter_count,
            )
        )
        or facts.a_work
        != _expected_work(
            manifest,
            after_a=True,
            performed=sum(decision.performed for decision in first_half),
            attempts=sum(decision.attempted for decision in first_half),
        )
        or facts.total_work
        != _expected_work(
            manifest,
            after_a=False,
            performed=sum(decision.performed for decision in facts.decisions),
            attempts=sum(decision.attempted for decision in facts.decisions),
        )
        or facts.initial_parameter_digest
        != parameter_digest(make_trigger_model(manifest, facts.seed, facts.arm))
    ):
        raise ValueError("v13 role, stream, work, or capacity preflight failed")
    _preflight_decisions(manifest, facts)
    post_b = trial.post_b_model
    if (
        post_b.hidden_dim != manifest.hidden_dim
        or parameter_count(post_b) != facts.final_parameter_count
        or parameter_digest(post_b) != facts.post_b_parameter_digest
        or post_b.get_sleep_clocks().wake_batches != facts.total_work.wake_updates
        or post_b.get_sleep_clocks().wake_examples != facts.total_work.wake_examples
        or post_b.get_sleep_clocks().sleep_events != facts.total_work.sleep_events
        or post_b.get_sleep_clocks().replay_updates != 0
    ):
        raise ValueError("v13 post-B model state failed preflight")
    post_a = make_trigger_model(manifest, facts.seed, facts.arm)
    post_a.restore_state(trial.post_a_state)
    if (
        post_a.hidden_dim != manifest.hidden_dim
        or parameter_count(post_a) != facts.post_a_parameter_count
        or parameter_digest(post_a) != facts.post_a_parameter_digest
        or post_a.get_sleep_clocks().wake_batches != facts.a_work.wake_updates
        or post_a.get_sleep_clocks().wake_examples != facts.a_work.wake_examples
        or post_a.get_sleep_clocks().sleep_events != facts.a_work.sleep_events
    ):
        raise ValueError("v13 post-A model state failed preflight")


def preflight_unscored_trigger_study(study: UnscoredTriggerStudy) -> None:
    """Check the global Cartesian train-only gate before any final read."""
    manifest = study.manifest
    _validate_manifest(manifest)
    expected = {
        (condition, seed, arm)
        for condition in manifest.conditions
        for seed in manifest.seeds
        for arm in manifest.arms
    }
    observed = [
        (trial.facts.condition, trial.facts.seed, trial.facts.arm) for trial in study.trials
    ]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("v13 factor cells do not cover the complete manifest")
    if set(study.roles) != {
        (condition, seed) for condition in manifest.conditions for seed in manifest.seeds
    }:
        raise ValueError("v13 condition/seed roles are incomplete")
    for condition_seed, phases in study.roles.items():
        if set(phases) != {"a", "b"} or any(
            role.final_released or role.final_test is not None for role in phases.values()
        ):
            raise ValueError(f"v13 {condition_seed} final role opened before preflight")
    initial_groups: dict[int, set[str]] = {}
    batch_groups: dict[tuple[str, int], set[tuple[tuple[str, ...], tuple[str, ...]]]] = {}
    a_groups: dict[tuple[int, str], set[str]] = {}
    for trial in study.trials:
        facts = trial.facts
        _preflight_trial(manifest, trial, study.roles[(facts.condition, facts.seed)])
        initial_groups.setdefault(facts.seed, set()).add(facts.initial_parameter_digest)
        batch_groups.setdefault((facts.condition, facts.seed), set()).add(
            (facts.train_batch_hashes["a"], facts.train_batch_hashes["b"])
        )
        a_groups.setdefault((facts.seed, facts.arm), set()).add(facts.post_a_parameter_digest)
    if (
        any(len(group) != 1 for group in initial_groups.values())
        or any(len(group) != 1 for group in batch_groups.values())
        or any(len(group) != 1 for group in a_groups.values())
    ):
        raise ValueError("v13 initial, arm stream, or common A state differs")


def train_unscored_trigger_study(
    manifest: TriggerManifest,
    *,
    source_factory: Callable[[int, str, str], PhaseSource] = TriggerPhaseSource,
) -> UnscoredTriggerStudy:
    """Train all arms and preflight without accessing final fields."""
    _validate_manifest(manifest)
    roles = _make_roles(manifest, source_factory)
    trials = tuple(
        train_trigger_trial(
            manifest,
            roles[(condition, seed)],
            seed=seed,
            condition=condition,
            arm=arm,
        )
        for condition in manifest.conditions
        for seed in manifest.seeds
        for arm in manifest.arms
    )
    study = UnscoredTriggerStudy(manifest, roles, trials)
    preflight_unscored_trigger_study(study)
    return study


def _score(model: CircadianPredictiveCodingNetwork, batch: LabeledData) -> tuple[float, float]:
    probabilities = np.asarray(model.predict_proba(batch.input), dtype=np.float64)
    accuracy = float(np.mean((probabilities >= 0.5) == batch.target))
    clipped = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    bce = float(
        np.mean(-batch.target * np.log(clipped) - (1.0 - batch.target) * np.log(1.0 - clipped))
    )
    return accuracy, bce


def run_trigger_comparison(
    manifest: TriggerManifest,
    *,
    source_factory: Callable[[int, str, str], PhaseSource] = TriggerPhaseSource,
) -> TriggerComparison:
    """Release all common final roles only after every trial freezes."""
    study = train_unscored_trigger_study(manifest, source_factory=source_factory)
    preflight_unscored_trigger_study(study)
    released = {
        key: {phase: release_final_test(roles) for phase, roles in phases.items()}
        for key, phases in study.roles.items()
    }
    outcomes: list[TriggerOutcome] = []
    for trial in study.trials:
        facts = trial.facts
        phases = released[(facts.condition, facts.seed)]
        final_a, final_b = phases["a"].final_test, phases["b"].final_test
        if final_a is None or final_b is None:
            raise ValueError("v13 final role release is incomplete")
        post_a = make_trigger_model(manifest, facts.seed, facts.arm)
        post_a.restore_state(trial.post_a_state)
        a_after_a, a_bce_after_a = _score(post_a, final_a)
        a_after_b, a_bce_after_b = _score(trial.post_b_model, final_a)
        b_after_b, b_bce_after_b = _score(trial.post_b_model, final_b)
        outcomes.append(
            TriggerOutcome(
                facts,
                {phase: phases[phase].split_hashes["final_test"] for phase in ("a", "b")},
                a_after_a,
                a_bce_after_a,
                a_after_b,
                a_bce_after_b,
                b_after_b,
                b_bce_after_b,
                a_after_a - a_after_b,
            )
        )
    return TriggerComparison(PROTOCOL_ID, manifest, manifest_digest(manifest), tuple(outcomes))
