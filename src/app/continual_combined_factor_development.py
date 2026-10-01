"""Score combined/removal cells only after all c7 facts and copies match.

Inputs are the frozen c7 manifest and verified saved train facts. Outputs
are every outer-development cell and declared contrast. This module does
not read final sources, write files, change settings or select a winner.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_combined_factor_preflight as preflight
from src.app.continual_combined_factor_manifest import (
    CombinedManifest,
    FULL_ARMS,
    PARITY_PAIRS,
    fixed_combined_manifest,
    validate_combined_manifest,
)
from src.app.continual_combined_factor_validation import verify_combined_payload
from src.app.continual_sleep_factor_development import (
    PairedContrast,
    ScoredSeed,
    _contrast,
    _score_arm,
)
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.continual_metrics import TWO_TASK_METRIC_CONTRACT_ID
from src.infra.continual_roles import PhaseDecisionRoles


PROTOCOL_ID = "continual_combined_factor_outer_development_v1"
REFERENCE_SHA256 = "79a7d7e09f0ada01ff72d6e266ddee7576aca52316a0f9a8e2ab3dd402251d13"
ARMS = tuple(arm.name for arm in fixed_combined_manifest().arms)
CONTRASTS = (
    tuple(("full", name) for name in FULL_ARMS[1:])
    + tuple(("full", name) for name in ARMS if name not in FULL_ARMS)
    + (
        ("backprop_full_replay", "backprop_off"),
        ("pc_full_replay", "pc_off"),
        ("neutral_full_replay", "neutral_off"),
        ("backprop_14_off", "backprop_off"),
        ("pc_14_off", "pc_off"),
        ("periodic_structure_only", "neutral_off"),
    )
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
class CombinedFactorDevelopment:
    protocol_id: str
    metric_contract_id: str
    reference_sha256: str
    train_facts: preflight.CombinedPreflight
    scored_seeds: tuple[ScoredSeed, ...]
    outer_selection_scored: bool = True
    final_released: bool = False


@dataclass
class _TrainedSeed:
    facts: preflight.CombinedSeedFacts
    roles_a: PhaseDecisionRoles
    roles_b: PhaseDecisionRoles
    models_after_a: dict[str, preflight.Model]
    models_after_b: dict[str, preflight.Model]


def run_combined_factor_development(
    manifest: CombinedManifest, reference: dict[str, Any]
) -> CombinedFactorDevelopment:
    """Finish every c7 cell and copy check before the first outer value."""
    validate_combined_manifest(manifest)
    verify_combined_payload(reference, preflight.json_value(manifest))
    trained = tuple(_train_seed(manifest, seed) for seed in manifest.seeds)
    facts = preflight.CombinedPreflight(
        preflight.PROTOCOL_ID, manifest, tuple(item.facts for item in trained)
    )
    payload = preflight.json_value(facts)
    verify_combined_payload(payload, preflight.json_value(manifest))
    if payload != reference:
        raise ValueError("P6.3 combined development train facts differ from complete c7 reference")
    for item in trained:
        _require_checkpoint_hashes(item)
    # Why this: late-seed fact or complete-state drift must block the first score.
    scored = tuple(_score_seed(item) for item in trained)
    for item in trained:
        _require_checkpoint_hashes(item)
    return CombinedFactorDevelopment(
        PROTOCOL_ID, TWO_TASK_METRIC_CONTRACT_ID, REFERENCE_SHA256, facts, scored
    )


def _train_seed(manifest: CombinedManifest, seed: int) -> _TrainedSeed:
    phase_a = arrived._build_phase_a_roles(manifest.source, seed)
    progress = preflight._new_progress(manifest, seed)
    preflight._train_phase(progress, phase_a)
    after_a = {name: preflight._parameter_hash(model) for name, model in progress.models.items()}
    models_after_a = deepcopy(progress.models)
    # Why this: all 17 A wake/decision histories finish before B source construction.
    phase_b = arrived._build_phase_b_roles(manifest.source, seed)
    preflight._train_phase(progress, phase_b)
    facts = preflight._finish_seed(progress, seed, phase_a, phase_b, after_a)
    return _TrainedSeed(facts, phase_a, phase_b, models_after_a, progress.models)


def _require_checkpoint_hashes(item: _TrainedSeed) -> None:
    for models, epoch in (
        (item.models_after_a, item.facts.opportunities[11]),
        (item.models_after_b, item.facts.opportunities[-1]),
    ):
        if tuple(models) != ARMS:
            raise ValueError("P6.3 combined development checkpoint model cells differ")
        for name, model in models.items():
            if (
                preflight._parameter_hash(model) != epoch["after_epoch_parameter_sha256"][name]
                or preflight._width(model) != epoch["after_epoch_widths"][name]
            ):
                raise ValueError("P6.3 combined development checkpoint parameters/width differ")
            is_circadian = isinstance(model, CircadianPredictiveCodingNetwork)
            if is_circadian != (name in epoch["after_epoch_state_sha256"]):
                raise ValueError("P6.3 combined development checkpoint state kind differs")
            if isinstance(model, CircadianPredictiveCodingNetwork) and (
                preflight._state_hash(model) != epoch["after_epoch_state_sha256"][name]
            ):
                raise ValueError("P6.3 combined development complete checkpoint state differs")


def _score_seed(item: _TrainedSeed) -> ScoredSeed:
    _require_checkpoint_hashes(item)
    arms = tuple(
        _score_arm(
            name, item.models_after_a[name], item.models_after_b[name], item.roles_a, item.roles_b
        )
        for name in ARMS
    )
    _require_checkpoint_hashes(item)
    by_name = {arm.name: arm for arm in arms}
    for left, right in PARITY_PAIRS:
        if any(
            getattr(by_name[left], field) != getattr(by_name[right], field)
            for field in SCORE_FIELDS
        ):
            raise ValueError(f"P6.3 combined development neutral PC outcome parity differs: {left}")
    contrasts: tuple[PairedContrast, ...] = tuple(
        _contrast(by_name[left], by_name[right]) for left, right in CONTRASTS
    )
    return ScoredSeed(item.facts.seed, arms, contrasts)
