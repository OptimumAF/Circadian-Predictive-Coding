"""Inspect fixed confirmation scope without data, models, scores or IO.

Inputs are six validated development configuration factories. Outputs bind
all reserved cells/roles/contrasts and conservative joint resource budgets.
Training, artifact verification and final release belong to separate gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Callable

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_combined_factor_development as combined_scores
from src.app import continual_combined_factor_manifest as combined
from src.app import continual_gating_pilot as gating
from src.app import continual_parent_factor_development as parent_scores
from src.app import continual_parent_factor_manifest as parent
from src.app import continual_replay_factor_pilot as replay
from src.app import continual_schedule_factor_development as schedule_scores
from src.app import continual_schedule_factor_preflight as schedule
from src.app import continual_sleep_factor_development as sleep_scores
from src.app import continual_sleep_factor_preflight as sleep
from src.core.continual_metrics import TWO_TASK_METRIC_CONTRACT_ID


PROTOCOL_ID = "continual_mechanism_confirmation_scope_v1"
ROLE_COUNTS = (72, 24, 24, 36, 12, 12)


@dataclass(frozen=True)
class ConfirmationFamily:
    name: str
    development_manifest_json: str
    development_manifest_sha256: str
    development_result_sha256: str
    development_source_map_sha256: str
    development_seeds: tuple[int, ...]
    seeds: tuple[int, ...]
    arms: tuple[str, ...]
    contrasts: tuple[tuple[str, str], ...]
    wake_updates: int
    maximum_optimizer_updates: int
    maximum_guarded_attempts: int
    retained_array_bytes_per_seed_before_copies: int
    expected_final_counts: tuple[int, int]


@dataclass(frozen=True)
class ConfirmationManifest:
    families: tuple[ConfirmationFamily, ...]
    protocol_id: str = PROTOCOL_ID
    metric_contract_id: str = TWO_TASK_METRIC_CONTRACT_ID
    role_counts: tuple[int, ...] = ROLE_COUNTS
    max_optimizer_updates: int = 16000
    wall_limit_seconds: int = 600
    max_process_rss_bytes: int = 512 * 1024 * 1024
    rss_interval_seconds: float = 0.005
    score_role: str = "independent_confirmation_final_test"
    global_train_gate_required: bool = True
    uncertainty_contract_required_before_scoring: bool = True


@dataclass(frozen=True)
class _FamilySpec:
    name: str
    factory: Callable[[], Any]
    validator: Callable[[Any], int]
    manifest_sha256: str
    result_sha256: str
    source_map_sha256: str
    contrasts: tuple[tuple[str, str], ...]
    guarded_attempts_per_seed: int = 0
    retained_array_bytes_per_seed: int = 0


def _specs() -> tuple[_FamilySpec, ...]:
    return (
        _FamilySpec(
            "gating",
            gating.fixed_gating_pilot_manifest,
            gating.validate_gating_pilot_manifest,
            "3aec33aba455ff3fa04b0d20993ed97f6e0aedc8e64f150fbe0040bf0a44dfe9",
            "ede5ebcd618c663c6cc34a142f81ea6fef8ead627069edc0019ac8e5dda516f7",
            "797183a7da84bab8c1cb2825dda06c5fdcf3db1fea051564f11122fd88979167",
            (("chemical_gating", "neutral_circadian"),),
        ),
        _FamilySpec(
            "replay",
            replay.fixed_replay_pilot_manifest,
            replay.validate_replay_pilot_manifest,
            "221465f5f0825c2a4c9bd5b01804008b3f6e51b5bb0549be4126ef0025f1cbed",
            "ecf3f868c96779507e30328f73070edfbdfce0e255dd76eb12bd3d707c3b1de5",
            "153f5e7fd287b4cfd6940d3367cee6275fada4e5192b9c09d5e8186c2eee8cbe",
            tuple((name, name.replace("_on", "_off")) for name in replay.ON_ARMS),
            retained_array_bytes_per_seed=576,
        ),
        _FamilySpec(
            "sleep",
            sleep.fixed_sleep_factor_manifest,
            sleep.validate_sleep_factor_manifest,
            "97c576e16b8e2307d89952b28db7948b1feb433b87d3407f9f33e451472617d1",
            "e0795b279346054d9c4e4c680e3aea103e347e3dd49960c44b93f8b642eaa9b6",
            "f618900fde912e7b7fdfecf0ebe329266e6c69604133bbd425560384235daa77",
            sleep_scores.CONTRASTS,
            5,
        ),
        _FamilySpec(
            "schedule",
            schedule.fixed_schedule_factor_manifest,
            schedule.validate_schedule_factor_manifest,
            "f8b6d60209516c58bc54e729f659c9fdd37ea1b869652b1cac548a71270ea42a",
            "776df6a47efd452a9eb33aced8fa74b1938608637109bd716333fafb2edae689",
            "3e7d8b8086ba881527f21d513015deb30743884b388b37f2eda23a79b4e29c69",
            schedule_scores.CONTRASTS,
            21,
            768,
        ),
        _FamilySpec(
            "combined",
            combined.fixed_combined_manifest,
            combined.validate_combined_manifest,
            "729bf9df8472752f51696299373555339208af7b5add1892d14154d4f21ad04a",
            "2e32c5d7b98b5798f45f5e9b7734088950eec1ef0ae5f19420a282321f3f8714",
            "ee0e2c8cb154f9aa427c0b27680841b64a0dfe3c6a90f254159114ca4ce4d6b5",
            combined_scores.CONTRASTS,
            48,
            2304,
        ),
        _FamilySpec(
            "parent",
            parent.fixed_parent_manifest,
            parent.validate_parent_manifest,
            "a7938028ed3c9279ef74a5f9a2550012927e4bb626b72672861ae64aa71497c9",
            "7f76e793eaf123b30116f57d68aa755962d63a56e4bba4b29703ae37b16e2040",
            "b0a86792167a9965e38007b0a0a3cd5a7ee3d229495df4950201198b545be55b",
            parent_scores.CONTRASTS,
            18,
            960,
        ),
    )


def _family(spec: _FamilySpec) -> ConfirmationFamily:
    original = spec.factory()
    maximum = spec.validator(original)
    encoded = json.dumps(asdict(original), sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = sha256(encoded.encode("utf-8")).hexdigest()
    if digest != spec.manifest_sha256 or maximum % len(original.seeds):
        raise ValueError(f"confirmation source manifest or additive budget changed: {spec.name}")
    seeds = original.confirmation_seeds
    arrived._validate_arrived_config(original.source, list(seeds))
    arms = tuple(getattr(arm, "name", arm) for arm in original.arms)
    if len(seeds) != 10 or len(set(seeds)) != 10 or set(seeds) & set(original.seeds):
        raise ValueError(f"confirmation reservations differ: {spec.name}")
    if any(left not in arms or right not in arms for left, right in spec.contrasts):
        raise ValueError(f"confirmation contrasts name missing cells: {spec.name}")
    training = original.source.training
    epochs = training.phase_a_epochs + training.phase_b_epochs
    final_counts = tuple(
        arrived._expected_final_count(count, training.test_ratio)
        for count in (training.sample_count_phase_a, training.sample_count_phase_b)
    )
    if epochs != 24 or final_counts != (40, 40):
        raise ValueError(f"confirmation training/final role geometry changed: {spec.name}")
    return ConfirmationFamily(
        spec.name,
        encoded,
        digest,
        spec.result_sha256,
        spec.source_map_sha256,
        original.seeds,
        seeds,
        arms,
        spec.contrasts,
        len(seeds) * len(arms) * epochs,
        maximum // len(original.seeds) * len(seeds),
        spec.guarded_attempts_per_seed * len(seeds),
        spec.retained_array_bytes_per_seed,
        (final_counts[0], final_counts[1]),
    )


def fixed_confirmation_manifest() -> ConfirmationManifest:
    return ConfirmationManifest(tuple(_family(spec) for spec in _specs()))


def confirmation_summary(manifest: ConfirmationManifest) -> dict[str, int]:
    families = manifest.families
    cells = sum(len(family.seeds) * len(family.arms) for family in families)
    return {
        "family_count": len(families),
        "family_seed_instances": sum(len(family.seeds) for family in families),
        "distinct_confirmation_seeds": len({seed for family in families for seed in family.seeds}),
        "cell_count": cells,
        "wake_updates": sum(family.wake_updates for family in families),
        "maximum_optimizer_updates": sum(family.maximum_optimizer_updates for family in families),
        "maximum_guarded_attempts": sum(family.maximum_guarded_attempts for family in families),
        "final_evaluations": 3 * cells,
        "final_examples": sum(
            len(family.seeds)
            * len(family.arms)
            * (2 * family.expected_final_counts[0] + family.expected_final_counts[1])
            for family in families
        ),
        "contrast_count": sum(len(family.seeds) * len(family.contrasts) for family in families),
    }


def validate_confirmation_manifest(manifest: ConfirmationManifest) -> dict[str, int]:
    if type(manifest) is not ConfirmationManifest or manifest != fixed_confirmation_manifest():
        raise ValueError("confirmation requires its frozen complete scope manifest")
    development = {seed for family in manifest.families for seed in family.development_seeds}
    confirm = {seed for family in manifest.families for seed in family.seeds}
    summary = confirmation_summary(manifest)
    if (
        development & confirm
        or len(confirm) != 50
        or summary["maximum_optimizer_updates"] > manifest.max_optimizer_updates
    ):
        raise ValueError("confirmation seed-role or joint optimizer budget gate failed")
    return summary
