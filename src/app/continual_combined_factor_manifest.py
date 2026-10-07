"""Freeze combined/minus-one cells and prelaunch work without opening data.

Inputs are existing reviewed source and component configurations. Outputs
are named immutable cell settings and a conservative optimizer ceiling.
This module does not construct datasets, train, score or write artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.app import continual_arrived_benchmark as arrived
from src.app.continual_replay_factor_pilot import _circadian_config, fixed_replay_pilot_manifest
from src.app.continual_sleep_factor_preflight import _arm_config
from src.app.continual_trigger_replay_schedule import fixed_trigger_replay_manifest
from src.core.circadian_predictive_coding import CircadianConfig


PROTOCOL_ID = "continual_combined_factor_train_only_v1"
DEVELOPMENT_SEEDS = (263, 269, 271)
CONFIRMATION_SEEDS = (277, 281, 283, 293, 307, 311, 313, 317, 331, 337)
FULL_ARMS = (
    "full",
    "minus_replay",
    "minus_gating",
    "minus_structure",
    "minus_schedule",
    "minus_difficulty",
    "minus_homeostasis",
    "minus_reset",
)
DECISION_ARMS = FULL_ARMS + ("periodic_structure_only",)
REPLAY_CONTROLS = ("backprop_full_replay", "pc_full_replay", "neutral_full_replay")
PARITY_PAIRS = (("pc_off", "neutral_off"), ("pc_full_replay", "neutral_full_replay"))


@dataclass(frozen=True)
class CombinedArm:
    name: str
    model_kind: str
    width: int
    min_width: int
    max_width: int
    config: CircadianConfig | None = None
    interval: int = 0
    replay_controller: str = "none"


@dataclass(frozen=True)
class CombinedManifest:
    source: arrived.ContinualArrivedRolesConfig
    arms: tuple[CombinedArm, ...]
    seeds: tuple[int, ...] = DEVELOPMENT_SEEDS
    confirmation_seeds: tuple[int, ...] = CONFIRMATION_SEEDS
    model_seed_offset: int = 1001
    memory_examples: int = 8
    memory_bytes: int = 192
    max_optimizer_updates: int = 1600
    max_process_rss_bytes: int = 256 * 1024 * 1024
    protocol_id: str = PROTOCOL_ID


def fixed_combined_manifest() -> CombinedManifest:
    full = replace(
        fixed_trigger_replay_manifest().arrived.training.circadian_config,
        use_reward_modulated_learning=True,
    )
    configs = {
        "full": full,
        "minus_replay": replace(full, sleep_enable_replay=False),
        "minus_gating": replace(full, min_plasticity=1.0),
        "minus_structure": replace(full, sleep_enable_split=False, sleep_enable_prune=False),
        "minus_schedule": replace(full, sleep_mode="disabled"),
        "minus_difficulty": replace(full, use_reward_modulated_learning=False),
        "minus_homeostasis": replace(full, sleep_enable_homeostasis=False),
        "minus_reset": replace(full, sleep_enable_chemical_reset=False),
    }
    arms = [
        CombinedArm("backprop_off", "backprop", 8, 8, 8),
        CombinedArm("pc_off", "pc", 8, 8, 8),
        CombinedArm("neutral_off", "circadian", 8, 8, 8, _circadian_config(False)),
        CombinedArm("backprop_full_replay", "backprop", 8, 8, 8, replay_controller="full"),
        CombinedArm("pc_full_replay", "pc", 8, 8, 8, replay_controller="full"),
        CombinedArm(
            "neutral_full_replay",
            "circadian",
            8,
            8,
            8,
            _circadian_config(True),
            replay_controller="full",
        ),
    ]
    arms.extend(
        CombinedArm(
            name,
            "circadian",
            8,
            8 if name == "minus_structure" else 4,
            8 if name == "minus_structure" else 14,
            configs[name],
            0 if name == "minus_schedule" else 4,
            "self" if configs[name].sleep_enable_replay and name != "minus_schedule" else "none",
        )
        for name in FULL_ARMS
    )
    structure = replace(
        _arm_config("structure_only"),
        replay_memory_size=8,
        replay_steps=2,
        replay_inference_steps=2,
        replay_prioritized=False,
        replay_class_balanced=False,
    )
    arms.extend(
        (
            CombinedArm("periodic_structure_only", "circadian", 8, 7, 9, structure, 4),
            CombinedArm("backprop_14_off", "backprop", 14, 14, 14),
            CombinedArm("pc_14_off", "pc", 14, 14, 14),
        )
    )
    return CombinedManifest(fixed_replay_pilot_manifest().source, tuple(arms))


def validate_combined_manifest(manifest: CombinedManifest) -> int:
    if type(manifest) is not CombinedManifest or manifest != fixed_combined_manifest():
        raise ValueError("P6.3 combined factor requires its frozen manifest")
    if set(manifest.seeds) & set(manifest.confirmation_seeds):
        raise ValueError("P6.3 combined seed roles overlap")
    arrived._validate_arrived_config(manifest.source, list(manifest.seeds))
    epochs = manifest.source.training.phase_a_epochs + manifest.source.training.phase_b_epochs
    wake = len(manifest.arms) * epochs
    replay = sum(
        (epochs // arm.interval) * 2
        for arm in manifest.arms
        if arm.interval and arm.config is not None and arm.config.sleep_enable_replay
    )
    consumers = len(REPLAY_CONTROLS) * (epochs // 4) * 2
    maximum = len(manifest.seeds) * (wake + replay + consumers)
    if maximum > manifest.max_optimizer_updates:
        raise ValueError("P6.3 combined executed optimizer cap exceeded before launch")
    return maximum
