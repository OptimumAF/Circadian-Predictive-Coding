"""Score a frozen sleep factor only after every train-only fact matches c3.

Inputs are the fixed c3 manifest and its verified, saved train-only fact object.
Outputs are all outer-development accuracies and prespecified paired contrasts.
This app does not release final roles, write files, or choose a winning arm.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from typing import Any, Mapping

import numpy as np

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_sleep_factor_preflight as preflight
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.continual_metrics import TWO_TASK_METRIC_CONTRACT_ID, TwoTaskAccuracy
from src.infra.continual_roles import PhaseDecisionRoles


PROTOCOL_ID = "continual_sleep_factor_outer_development_v1"
CONTRASTS = (
    ("structure_only", "neutral_sham"),
    ("homeostasis_only", "neutral_sham"),
    ("gating_reset", "gating_sham"),
)
REFERENCE_SHA256 = "e3140d5a0529dc3023b90b60801d8f9caf2e15dd8edb4a85fa875f76f60d624d"


@dataclass(frozen=True)
class ScoredArm:
    name: str
    a_after_a: float
    a_after_b: float
    b_after_b: float
    final_mean_task_accuracy: float
    signed_forgetting_a: float
    retention_ratio_a: float | None


@dataclass(frozen=True)
class PairedContrast:
    left: str
    right: str
    a_after_a: float
    a_after_b: float
    b_after_b: float
    final_mean_task_accuracy: float
    signed_forgetting_a: float


@dataclass(frozen=True)
class ScoredSeed:
    seed: int
    arms: tuple[ScoredArm, ...]
    contrasts: tuple[PairedContrast, ...]


@dataclass(frozen=True)
class SleepFactorDevelopment:
    protocol_id: str
    metric_contract_id: str
    reference_sha256: str
    train_facts: preflight.SleepFactorPreflight
    scored_seeds: tuple[ScoredSeed, ...]
    outer_selection_scored: bool
    final_released: bool


@dataclass
class _TrainedSeed:
    facts: preflight.SleepSeedFacts
    roles_a: PhaseDecisionRoles
    roles_b: PhaseDecisionRoles
    models_after_a: dict[str, preflight.Model]
    models_after_b: dict[str, preflight.Model]


def run_sleep_factor_development(
    manifest: preflight.SleepFactorManifest, reference: Mapping[str, Any]
) -> SleepFactorDevelopment:
    """Train, globally compare every c3 fact, then read outer roles."""
    preflight.validate_sleep_factor_manifest(manifest)
    if (
        reference.get("protocol_id") != preflight.PROTOCOL_ID
        or reference.get("manifest") != _json_value(manifest)
        or reference.get("outer_selection_scored") is not False
        or reference.get("final_released") is not False
    ):
        raise ValueError("P6.3 scored development reference is not the sealed c3 manifest")

    trained = tuple(_train_seed(manifest, seed) for seed in manifest.seeds)
    facts = preflight.SleepFactorPreflight(
        preflight.PROTOCOL_ID,
        manifest,
        tuple(item.facts for item in trained),
        False,
        False,
    )
    if _json_value(facts) != reference:
        raise ValueError("P6.3 scored development train facts differ from complete c3 reference")

    # Why this: no outer value is touched until all seeds and arms match c3.
    scored = tuple(_score_seed(item, manifest) for item in trained)
    return SleepFactorDevelopment(
        PROTOCOL_ID,
        TWO_TASK_METRIC_CONTRACT_ID,
        REFERENCE_SHA256,
        facts,
        scored,
        True,
        False,
    )


def _train_seed(manifest: preflight.SleepFactorManifest, seed: int) -> _TrainedSeed:
    phase_a = arrived._build_phase_a_roles(manifest.source, seed)
    models = preflight._new_models(seed, manifest)
    initial = {name: preflight._parameter_hash(model) for name, model in models.items()}
    preflight._train_phase(models, phase_a, manifest, before_sleep=True)
    pre_sleep = {name: preflight._parameter_hash(model) for name, model in models.items()}
    minimum = _minimum_plasticity(models)
    sleeps = preflight._apply_guarded_sleep(models, phase_a, manifest)
    post_sleep = {name: preflight._parameter_hash(model) for name, model in models.items()}
    models_after_a = deepcopy(models)

    # Why this: the B generator is called only after A training and guarded sleep.
    phase_b = arrived._build_phase_b_roles(manifest.source, seed)
    preflight._train_phase(models, phase_b, manifest, before_sleep=False)
    if phase_a.final_released or phase_b.final_released:
        raise ValueError("P6.3 scored development released final data during training")
    hashes, counts = _role_facts(phase_a, phase_b)
    arms = tuple(
        preflight._arm_facts(
            name,
            models[name],
            initial[name],
            pre_sleep[name],
            post_sleep[name],
            minimum.get(name),
            sleeps.get(name),
            phase_a,
            phase_b,
            manifest,
        )
        for name in manifest.arms
    )
    return _TrainedSeed(
        preflight.SleepSeedFacts(seed, hashes, counts, arms, False),
        phase_a,
        phase_b,
        models_after_a,
        models,
    )


def _minimum_plasticity(models: dict[str, preflight.Model]) -> dict[str, float]:
    values = {
        name: float(np.min(model.get_plasticity_state()))
        for name in preflight.GATING_ARMS
        if isinstance(model := models[name], CircadianPredictiveCodingNetwork)
    }
    if any(not 0.2 <= value < 1.0 for value in values.values()):
        raise ValueError("P6.3 scored development chemical gate was inactive")
    return values


def _json_value(
    value: preflight.SleepFactorManifest | preflight.SleepFactorPreflight,
) -> Any:
    """Use the c3 adapter's JSON tuple/list representation for exact comparison."""
    return json.loads(json.dumps(asdict(value), allow_nan=False))


def _role_facts(
    phase_a: PhaseDecisionRoles, phase_b: PhaseDecisionRoles
) -> tuple[dict[str, str], dict[str, int]]:
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
    return hashes, counts


def _score_seed(item: _TrainedSeed, manifest: preflight.SleepFactorManifest) -> ScoredSeed:
    arms = tuple(
        _score_arm(
            name, item.models_after_a[name], item.models_after_b[name], item.roles_a, item.roles_b
        )
        for name in manifest.arms
    )
    by_name = {arm.name: arm for arm in arms}
    pc = by_name["pc_8"]
    neutral = by_name["neutral_sham"]
    if (
        pc.a_after_a,
        pc.a_after_b,
        pc.b_after_b,
        pc.final_mean_task_accuracy,
        pc.signed_forgetting_a,
        pc.retention_ratio_a,
    ) != (
        neutral.a_after_a,
        neutral.a_after_b,
        neutral.b_after_b,
        neutral.final_mean_task_accuracy,
        neutral.signed_forgetting_a,
        neutral.retention_ratio_a,
    ):
        raise ValueError("P6.3 scored development neutral PC outcome parity differs")
    return ScoredSeed(
        item.facts.seed,
        arms,
        tuple(_contrast(by_name[left], by_name[right]) for left, right in CONTRASTS),
    )


def _score_arm(
    name: str,
    after_a: preflight.Model,
    after_b: preflight.Model,
    roles_a: PhaseDecisionRoles,
    roles_b: PhaseDecisionRoles,
) -> ScoredArm:
    a_role = roles_a.outer_selection
    b_role = roles_b.outer_selection
    values = TwoTaskAccuracy(
        float(after_a.compute_accuracy(a_role.input, a_role.target)),
        float(after_b.compute_accuracy(a_role.input, a_role.target)),
        float(after_b.compute_accuracy(b_role.input, b_role.target)),
    )
    return ScoredArm(
        name,
        values.a_after_a,
        values.a_after_b,
        values.b_after_b,
        values.final_mean_task_accuracy,
        values.signed_forgetting_a,
        values.retention_ratio_a,
    )


def _contrast(left: ScoredArm, right: ScoredArm) -> PairedContrast:
    return PairedContrast(
        left.name,
        right.name,
        left.a_after_a - right.a_after_a,
        left.a_after_b - right.a_after_b,
        left.b_after_b - right.b_after_b,
        left.final_mean_task_accuracy - right.final_mean_task_accuracy,
        left.signed_forgetting_a - right.signed_forgetting_a,
    )
