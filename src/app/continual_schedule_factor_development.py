"""Score matched schedule policies after their complete frozen train gate.

Inputs are the c5 manifest and verified saved train-only facts. Outputs are
every outer-development cell and paired policy contrast. This app neither
reads final sources, writes artifacts nor selects settings or a winner.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_schedule_factor_preflight as preflight
from src.app.continual_schedule_factor_validation import verify_schedule_preflight_payload
from src.app.continual_sleep_factor_development import (
    PairedContrast,
    ScoredSeed,
    _contrast,
    _role_facts,
    _score_arm,
)
from src.core.continual_metrics import TWO_TASK_METRIC_CONTRACT_ID
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.shared_replay_schedule import SharedReplayBuffer
from src.infra.continual_roles import PhaseDecisionRoles


PROTOCOL_ID = "continual_schedule_factor_outer_development_v1"
REFERENCE_SHA256 = "87931c5fc3fad5bf50d5b18d07901d04b8a8e827ab4fce4429c0f1644c813c41"
POLICY_PAIRS = (("periodic", "no_sleep"), ("adaptive", "no_sleep"), ("adaptive", "periodic"))
CONTRASTS = tuple(
    (f"{method}_{left}", f"{method}_{right}")
    for method in preflight.METHODS
    for left, right in POLICY_PAIRS
)
CONTRAST_FIELDS = (
    "a_after_a",
    "a_after_b",
    "b_after_b",
    "final_mean_task_accuracy",
    "signed_forgetting_a",
)
SCORE_FIELDS = CONTRAST_FIELDS + ("retention_ratio_a",)


@dataclass(frozen=True)
class ScheduleFactorDevelopment:
    protocol_id: str
    metric_contract_id: str
    reference_sha256: str
    train_facts: preflight.ScheduleFactorPreflight
    scored_seeds: tuple[ScoredSeed, ...]
    outer_selection_scored: bool = True
    final_released: bool = False


@dataclass
class _TrainedSeed:
    facts: preflight.ScheduleSeedFacts
    roles_a: PhaseDecisionRoles
    roles_b: PhaseDecisionRoles
    models_after_a: dict[str, preflight.Model]
    models_after_b: dict[str, preflight.Model]


def run_schedule_factor_development(
    manifest: preflight.ScheduleFactorManifest, reference: dict[str, Any]
) -> ScheduleFactorDevelopment:
    """Finish every c5 cell and compare all facts before any outer access."""
    preflight.validate_schedule_factor_manifest(manifest)
    verify_schedule_preflight_payload(reference, preflight._json_value(manifest))
    trained = tuple(_train_seed(manifest, seed) for seed in manifest.seeds)
    facts = preflight.ScheduleFactorPreflight(
        preflight.PROTOCOL_ID, manifest, tuple(item.facts for item in trained)
    )
    payload = preflight._json_value(facts)
    verify_schedule_preflight_payload(payload, preflight._json_value(manifest))
    if payload != reference:
        raise ValueError("P6.3 schedule development train facts differ from complete c5 reference")
    for item in trained:
        _require_checkpoint_hashes(item)
    # Why this: a late-seed failure must precede even the first outer value.
    scored = tuple(_score_seed(item, manifest) for item in trained)
    return ScheduleFactorDevelopment(
        PROTOCOL_ID, TWO_TASK_METRIC_CONTRACT_ID, REFERENCE_SHA256, facts, scored
    )


def _train_seed(manifest: preflight.ScheduleFactorManifest, seed: int) -> _TrainedSeed:
    phase_a = arrived._build_phase_a_roles(manifest.source, seed)
    models = preflight._new_models(manifest, seed)
    initial = {name: preflight._parameter_hash(model) for name, model in models.items()}
    shared = SharedReplayBuffer(
        2,
        ReplayRetentionBudget(manifest.memory_examples, manifest.memory_bytes),
        ReplayRetentionPolicy("recent_fifo"),
    )
    opportunities = preflight._train_phase(models, phase_a, manifest, shared)
    after_a = {name: preflight._parameter_hash(model) for name, model in models.items()}
    models_after_a = deepcopy(models)
    # Why this: all eleven A models finish wake and decisions before B arrives.
    phase_b = arrived._build_phase_b_roles(manifest.source, seed)
    opportunities += preflight._train_phase(models, phase_b, manifest, shared)
    if phase_a.final_released or phase_b.final_released:
        raise ValueError("P6.3 schedule development released final data during training")
    hashes, counts = _role_facts(phase_a, phase_b)
    methods = tuple(
        preflight._method_facts(
            name, models[name], initial[name], after_a[name], opportunities, manifest
        )
        for name in manifest.arms
    )
    executed = sum(
        row.wake_updates + row.applied_replay_updates + row.rejected_executed_replay_updates
        for row in methods
    )
    facts = preflight.ScheduleSeedFacts(seed, hashes, counts, opportunities, methods, executed)
    return _TrainedSeed(facts, phase_a, phase_b, models_after_a, models)


def _require_checkpoint_hashes(item: _TrainedSeed) -> None:
    for method in item.facts.methods:
        if (
            preflight._parameter_hash(item.models_after_a[method.name])
            != method.after_a_parameter_sha256
            or preflight._parameter_hash(item.models_after_b[method.name])
            != method.final_parameter_sha256
        ):
            raise ValueError("P6.3 schedule development checkpoint parameters differ")


def _score_seed(item: _TrainedSeed, manifest: preflight.ScheduleFactorManifest) -> ScoredSeed:
    _require_checkpoint_hashes(item)
    arms = tuple(
        _score_arm(
            name, item.models_after_a[name], item.models_after_b[name], item.roles_a, item.roles_b
        )
        for name in manifest.arms
    )
    _require_checkpoint_hashes(item)
    by_name = {arm.name: arm for arm in arms}
    for policy in manifest.policies:
        if any(
            getattr(by_name[f"pc_{policy}"], field) != getattr(by_name[f"neutral_{policy}"], field)
            for field in SCORE_FIELDS
        ):
            raise ValueError(
                f"P6.3 schedule development neutral PC outcome parity differs: {policy}"
            )
    contrasts: tuple[PairedContrast, ...] = tuple(
        _contrast(by_name[left], by_name[right]) for left, right in CONTRASTS
    )
    return ScoredSeed(item.facts.seed, arms, contrasts)
