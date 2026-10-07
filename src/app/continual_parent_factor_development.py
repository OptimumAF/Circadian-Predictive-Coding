"""Score parent controls after the complete c9b fact and checkpoint gate.

Inputs are frozen c9b settings and verified train-only facts. Outputs retain
all outer-development cells and pairs. This module owns no artifact IO,
final source access, confirmation or treatment selection.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_parent_factor_preflight as preflight
from src.app.continual_parent_factor_manifest import (
    GROWTH_ARMS,
    ParentManifest,
    fixed_parent_manifest,
    validate_parent_manifest,
)
from src.app.continual_parent_factor_validation import verify_parent_payload
from src.app.continual_sleep_factor_development import (
    ScoredSeed,
    _contrast,
    _role_facts,
    _score_arm,
)
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.continual_metrics import TWO_TASK_METRIC_CONTRACT_ID
from src.core.controlled_parent_selection import ParentControlledCircadianNetwork
from src.infra.continual_roles import PhaseDecisionRoles


PROTOCOL_ID = "continual_parent_factor_outer_development_v1"
REFERENCE_SHA256 = "555fc2fe5dfd86981d87af2d0c15bd8ab925417d9ad748783d95cee95f5a1874"
ARMS = tuple(arm.name for arm in fixed_parent_manifest().arms)
REFERENCES = ("backprop_off", "pc_off", "neutral_off", "backprop_13_off", "pc_13_off")
CONTRASTS = (
    (
        ("usage_growth", "scheduled_growth"),
        ("usage_growth", "random_growth"),
        ("scheduled_growth", "random_growth"),
    )
    + tuple((growth, reference) for growth in GROWTH_ARMS for reference in REFERENCES)
    + (
        ("backprop_13_off", "backprop_off"),
        ("pc_13_off", "pc_off"),
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
class ParentFactorDevelopment:
    protocol_id: str
    metric_contract_id: str
    reference_sha256: str
    train_facts: preflight.ParentPreflight
    scored_seeds: tuple[ScoredSeed, ...]
    outer_selection_scored: bool = True
    final_released: bool = False


@dataclass
class _TrainedSeed:
    facts: preflight.ParentSeedFacts
    roles_a: PhaseDecisionRoles
    roles_b: PhaseDecisionRoles
    models_after_a: dict[str, preflight.Model]
    models_after_b: dict[str, preflight.Model]


def run_parent_factor_development(
    manifest: ParentManifest, reference: dict[str, Any]
) -> ParentFactorDevelopment:
    """Validate every seed and full copy before the first outer value."""
    validate_parent_manifest(manifest)
    verify_parent_payload(reference, preflight.json_value(manifest))
    trained = tuple(_train_seed(manifest, seed) for seed in manifest.seeds)
    facts = preflight.ParentPreflight(
        preflight.PROTOCOL_ID, manifest, tuple(item.facts for item in trained)
    )
    payload = preflight.json_value(facts)
    verify_parent_payload(payload, preflight.json_value(manifest))
    if payload != reference:
        raise ValueError("P6.3 parent development train facts differ from complete c9b reference")
    for item in trained:
        _require_checkpoint_hashes(item)
    # Why this: a late selector/RNG/cursor drift must block even the first score.
    scored = tuple(_score_seed(item) for item in trained)
    for item in trained:
        _require_checkpoint_hashes(item)
    return ParentFactorDevelopment(
        PROTOCOL_ID, TWO_TASK_METRIC_CONTRACT_ID, REFERENCE_SHA256, facts, scored
    )


def _train_seed(manifest: ParentManifest, seed: int) -> _TrainedSeed:
    phase_a = arrived._build_phase_a_roles(manifest.source, seed)
    progress = preflight._new_progress(manifest, seed)
    preflight._train_phase(progress, phase_a)
    models_after_a = deepcopy(progress.models)
    # Why this: all eight A histories complete before constructing any B source.
    phase_b = arrived._build_phase_b_roles(manifest.source, seed)
    preflight._train_phase(progress, phase_b)
    if phase_a.final_released or phase_b.final_released:
        raise ValueError("P6.3 parent development released final data during training")
    hashes, counts = _role_facts(phase_a, phase_b)
    methods = tuple(preflight._method_facts(progress, arm) for arm in manifest.arms)
    facts = preflight.ParentSeedFacts(
        seed,
        hashes,
        counts,
        methods,
        tuple(progress.opportunities),
        sum(row["wake_updates"] for row in methods),
    )
    return _TrainedSeed(facts, phase_a, phase_b, models_after_a, progress.models)


def _require_checkpoint_hashes(item: _TrainedSeed) -> None:
    for models, epoch in (
        (item.models_after_a, item.facts.opportunities[11]),
        (item.models_after_b, item.facts.opportunities[-1]),
    ):
        if tuple(models) != ARMS:
            raise ValueError("P6.3 parent development checkpoint model cells differ")
        for name, model in models.items():
            if (
                preflight._parameter_hash(model) != epoch["after_epoch_parameter_sha256"][name]
                or preflight._width(model) != epoch["after_epoch_widths"][name]
            ):
                raise ValueError("P6.3 parent development checkpoint parameters/width differ")
            is_circadian = isinstance(model, CircadianPredictiveCodingNetwork)
            if is_circadian != (name in epoch["after_epoch_state_sha256"]) or isinstance(
                model, ParentControlledCircadianNetwork
            ) != (name in GROWTH_ARMS):
                raise ValueError("P6.3 parent development checkpoint state kind differs")
            if isinstance(model, CircadianPredictiveCodingNetwork) and (
                preflight._state_hash(model) != epoch["after_epoch_state_sha256"][name]
            ):
                raise ValueError("P6.3 parent development complete checkpoint state differs")


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
    if any(
        getattr(by_name["pc_off"], field) != getattr(by_name["neutral_off"], field)
        for field in SCORE_FIELDS
    ):
        raise ValueError("P6.3 parent development neutral PC outcome parity differs")
    return ScoredSeed(
        item.facts.seed,
        arms,
        tuple(_contrast(by_name[left], by_name[right]) for left, right in CONTRASTS),
    )
