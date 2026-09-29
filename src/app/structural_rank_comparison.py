"""Globally seal and score the fixed v12 structural-ranking factor study.

Inputs are the exact manifest and deferred phase sources. The output
contains every factor cell and its held-out outcome. This app neither
chooses a setting nor writes result files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Any, Callable

import numpy as np

from src.app.structural_rank_trial import (
    PROTOCOL_ID,
    RankFactors,
    RankWork,
    StructuralRankManifest,
    TrainedRankTrial,
    UnscoredRankFacts,
    fixed_structural_rank_manifest,
    make_structural_model,
    parameter_count,
    parameter_digest,
    train_structural_rank_trial,
)
from src.infra.continual_roles import (
    PhaseDecisionRoles,
    PhaseSource,
    release_final_test,
    split_phase_decision_roles,
)
from src.infra.datasets import LabeledData
from src.infra.difficulty_streams import DifficultyPhaseSource


@dataclass
class UnscoredRankStudy:
    manifest: StructuralRankManifest
    roles_by_seed: dict[int, dict[str, PhaseDecisionRoles]]
    trials: tuple[TrainedRankTrial, ...]


@dataclass(frozen=True)
class RankOutcome:
    train_facts: UnscoredRankFacts
    final_role_hashes: dict[str, str]
    a_after_a_accuracy: float
    a_after_b_accuracy: float
    b_after_b_accuracy: float
    forgetting: float


@dataclass(frozen=True)
class StructuralRankComparison:
    protocol_id: str
    manifest: StructuralRankManifest
    manifest_digest: str
    outcomes: tuple[RankOutcome, ...]


def _validate_manifest(manifest: StructuralRankManifest) -> None:
    if type(manifest) is not StructuralRankManifest or manifest != fixed_structural_rank_manifest():
        raise ValueError("v12 structural manifest differs from the predeclared protocol")


def _manifest_digest(manifest: StructuralRankManifest) -> str:
    canonical = json.dumps(asdict(manifest), sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _make_roles(
    manifest: StructuralRankManifest,
    source_factory: Callable[[int, str], PhaseSource],
) -> dict[int, dict[str, PhaseDecisionRoles]]:
    roles_by_seed: dict[int, dict[str, PhaseDecisionRoles]] = {}
    for seed in manifest.seeds:
        roles_by_seed[seed] = {}
        for phase in ("a", "b"):
            roles_by_seed[seed][phase] = split_phase_decision_roles(
                source_factory(seed, phase),
                phase=phase,
                seed=seed,
                split_seed=seed * 10 + (1 if phase == "a" else 2),
                inner_guard_fraction=manifest.inner_guard_fraction,
                outer_selection_fraction=manifest.outer_selection_fraction,
                expected_final_count=manifest.final_count,
            )
    return roles_by_seed


def _expected_work(
    manifest: StructuralRankManifest, train_count: int, *, after_a: bool
) -> RankWork:
    updates = manifest.epochs_per_phase * (1 if after_a else 2)
    examples = updates * train_count
    return RankWork(
        wake_updates=updates,
        wake_examples=examples,
        inference_loops=updates * manifest.inference_steps,
        example_inference_loops=examples * manifest.inference_steps,
        replay_updates=0,
        sleep_events=1,
    )


def _preflight_trial(
    manifest: StructuralRankManifest,
    trial: TrainedRankTrial,
    phases: dict[str, PhaseDecisionRoles],
) -> None:
    facts = trial.facts
    expected_hashes = {
        f"{phase}/{role}": phase_roles.split_hashes[role]
        for phase, phase_roles in phases.items()
        for role in ("train", "inner_guard", "outer_selection")
    }
    train_count = len(phases["a"].train.input)
    if train_count != len(phases["b"].train.input) or train_count != 24:
        raise ValueError("v12 A/B train exposure is unequal or changed")
    expected_count = 4 * manifest.hidden_dim + (1 if facts.backend == "numpy" else 2)
    # The formula above counts 2×width input weights, width hidden bias,
    # width×output-class weights, and output-class bias.
    if facts.backend == "torch_cpu":
        expected_count = 5 * manifest.hidden_dim + 2
    expected_width = manifest.hidden_dim
    if (
        facts.role_hashes != expected_hashes
        or facts.initial_width != expected_width
        or facts.post_sleep_width != expected_width
        or facts.final_width != expected_width
        or facts.initial_parameter_count != expected_count
        or facts.post_sleep_parameter_count != expected_count
        or facts.final_parameter_count != expected_count
        or facts.a_work != _expected_work(manifest, train_count, after_a=True)
        or facts.total_work != _expected_work(manifest, train_count, after_a=False)
        or len(facts.split_pairs) != manifest.split_cap
        or len(facts.removed_prune_ids) != manifest.prune_cap
        or len(facts.reward_scales) != 2 * manifest.epochs_per_phase
        or any(not isfinite(scale) or not 0.75 <= scale <= 1.5 for scale in facts.reward_scales)
        or len(set(facts.removed_prune_ids)) != len(facts.removed_prune_ids)
        or any(parent < 0 or child < manifest.hidden_dim for parent, child in facts.split_pairs)
    ):
        raise ValueError("v12 role, work, structure, or reward preflight failed")
    model = trial.post_b_model
    if (
        model.hidden_dim != expected_width
        or parameter_count(model, facts.backend) != expected_count
        or parameter_digest(model, facts.backend) != facts.post_b_parameter_digest
        or model.get_sleep_clocks().wake_batches != 2 * manifest.epochs_per_phase
        or model.get_sleep_clocks().sleep_events != 1
    ):
        raise ValueError("v12 post-B model state does not match train facts")
    after_a = make_structural_model(manifest, facts.backend, facts.seed, facts.factors)
    after_a.restore_state(trial.post_a_state)
    after_a_lineage = after_a.get_neuron_lineage()
    initial_ids = set(range(manifest.hidden_dim))
    child_ids = {child for _, child in facts.split_pairs}
    expected_ids = initial_ids | child_ids
    expected_ids.difference_update(facts.removed_prune_ids)
    parents_by_id = dict(zip(after_a_lineage.neuron_ids, after_a_lineage.parent_ids, strict=True))
    if (
        after_a.hidden_dim != expected_width
        or parameter_digest(after_a, facts.backend) != facts.post_a_parameter_digest
        or after_a.get_sleep_clocks().wake_batches != manifest.epochs_per_phase
        or after_a.get_sleep_clocks().sleep_events != 1
        or set(after_a_lineage.neuron_ids) != expected_ids
        or any(
            parent not in initial_ids
            or child != manifest.hidden_dim
            or parents_by_id.get(child) != parent
            for parent, child in facts.split_pairs
        )
        or any(neuron_id not in initial_ids | child_ids for neuron_id in facts.removed_prune_ids)
        or model.get_neuron_lineage() != after_a_lineage
    ):
        raise ValueError("v12 post-A model state does not match train facts")


def preflight_unscored_study(study: UnscoredRankStudy) -> None:
    """Reject any incomplete or unmatched trial before opening final fields."""
    manifest = study.manifest
    _validate_manifest(manifest)
    expected = {
        (seed, backend, wake, history, rank)
        for seed in manifest.seeds
        for backend in manifest.backends
        for wake in manifest.wake_modulation
        for history in manifest.reward_weight_history
        for rank in manifest.rank_importance
    }
    observed = [
        (
            trial.facts.seed,
            trial.facts.backend,
            trial.facts.factors.wake_modulation,
            trial.facts.factors.reward_weight_history,
            trial.facts.factors.rank_importance,
        )
        for trial in study.trials
    ]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("v12 trials do not cover the complete fixed factorial")
    if set(study.roles_by_seed) != set(manifest.seeds):
        raise ValueError("v12 seed roles are incomplete")
    initial_groups: dict[tuple[int, str], set[str]] = {}
    pre_sleep_groups: dict[tuple[int, str, bool], set[str]] = {}
    scale_groups: dict[tuple[int, str, bool], set[tuple[float, ...]]] = {}
    for seed, phases in study.roles_by_seed.items():
        if set(phases) != {"a", "b"} or any(
            role.final_released or role.final_test is not None for role in phases.values()
        ):
            raise ValueError(f"v12 seed {seed} final role opened before global preflight")
    for trial in study.trials:
        facts = trial.facts
        _preflight_trial(manifest, trial, study.roles_by_seed[facts.seed])
        initial_groups.setdefault((facts.seed, facts.backend), set()).add(
            facts.initial_parameter_digest
        )
        pre_sleep_groups.setdefault(
            (facts.seed, facts.backend, facts.factors.wake_modulation), set()
        ).add(facts.pre_sleep_parameter_digest)
        scale_groups.setdefault(
            (facts.seed, facts.backend, facts.factors.wake_modulation), set()
        ).add(facts.reward_scales[: manifest.epochs_per_phase])
    if (
        any(len(values) != 1 for values in initial_groups.values())
        or any(len(values) != 1 for values in pre_sleep_groups.values())
        or any(len(values) != 1 for values in scale_groups.values())
    ):
        raise ValueError("v12 factor cells have unmatched initial, pre-sleep, or scale facts")


def train_unscored_structural_rank_study(
    manifest: StructuralRankManifest,
    *,
    source_factory: Callable[[int, str], PhaseSource] = DifficultyPhaseSource,
) -> UnscoredRankStudy:
    """Train all 32 cells and preflight them without final data access."""
    _validate_manifest(manifest)
    roles_by_seed = _make_roles(manifest, source_factory)
    trials = tuple(
        train_structural_rank_trial(
            manifest,
            roles_by_seed[seed],
            seed=seed,
            backend=backend,
            factors=RankFactors(wake, history, rank),
        )
        for seed in manifest.seeds
        for backend in manifest.backends
        for wake in manifest.wake_modulation
        for history in manifest.reward_weight_history
        for rank in manifest.rank_importance
    )
    study = UnscoredRankStudy(manifest, roles_by_seed, trials)
    preflight_unscored_study(study)
    return study


def _accuracy(model: Any, backend: str, batch: LabeledData) -> float:
    if backend == "numpy":
        probabilities = model.predict_proba(batch.input)
    else:
        import torch

        with torch.no_grad():
            features = torch.as_tensor(batch.input, dtype=torch.float32)
            probabilities = torch.softmax(model.predict_logits(features), dim=1)[:, 1:2]
            probabilities = probabilities.cpu().numpy()
    return float(np.mean((probabilities >= 0.5) == batch.target))


def run_structural_rank_comparison(
    manifest: StructuralRankManifest,
    *,
    source_factory: Callable[[int, str], PhaseSource] = DifficultyPhaseSource,
) -> StructuralRankComparison:
    """Release common final roles only after all train-only cells freeze."""
    study = train_unscored_structural_rank_study(manifest, source_factory=source_factory)
    preflight_unscored_study(study)
    released = {
        seed: {phase: release_final_test(role) for phase, role in phases.items()}
        for seed, phases in study.roles_by_seed.items()
    }
    outcomes: list[RankOutcome] = []
    for trial in study.trials:
        facts = trial.facts
        final_a = released[facts.seed]["a"].final_test
        final_b = released[facts.seed]["b"].final_test
        if final_a is None or final_b is None:
            raise ValueError("v12 final role release is incomplete")
        post_a = make_structural_model(manifest, facts.backend, facts.seed, facts.factors)
        post_a.restore_state(trial.post_a_state)
        a_after_a = _accuracy(post_a, facts.backend, final_a)
        a_after_b = _accuracy(trial.post_b_model, facts.backend, final_a)
        b_after_b = _accuracy(trial.post_b_model, facts.backend, final_b)
        outcomes.append(
            RankOutcome(
                train_facts=facts,
                final_role_hashes={
                    phase: released[facts.seed][phase].split_hashes["final_test"]
                    for phase in ("a", "b")
                },
                a_after_a_accuracy=a_after_a,
                a_after_b_accuracy=a_after_b,
                b_after_b_accuracy=b_after_b,
                forgetting=a_after_a - a_after_b,
            )
        )
    return StructuralRankComparison(
        PROTOCOL_ID, manifest, _manifest_digest(manifest), tuple(outcomes)
    )
