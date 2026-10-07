"""Freeze parent-control cells/counts without constructing or scoring data."""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.app import continual_arrived_benchmark as arrived
from src.app.continual_replay_factor_pilot import _circadian_config, fixed_replay_pilot_manifest
from src.app.continual_sleep_factor_preflight import _arm_config
from src.core.circadian_predictive_coding import CircadianConfig
from src.core.controlled_parent_selection import SelectionMode

PROTOCOL_ID = "continual_parent_factor_train_only_v1"
DEVELOPMENT_SEEDS = (347, 349, 353)
CONFIRMATION_SEEDS = (359, 367, 373, 379, 383, 389, 397, 401, 409, 419)
GROWTH_MODES: tuple[SelectionMode, ...] = ("usage", "scheduled", "random")
GROWTH_ARMS = tuple(f"{mode}_growth" for mode in GROWTH_MODES)
ADD_EPOCHS = (4, 8, 12, 16, 20)


@dataclass(frozen=True)
class ParentArm:
    name: str
    model_kind: str
    width: int
    min_width: int
    max_width: int
    config: CircadianConfig | None = None
    parent_mode: SelectionMode | None = None


@dataclass(frozen=True)
class ParentManifest:
    source: arrived.ContinualArrivedRolesConfig
    arms: tuple[ParentArm, ...]
    seeds: tuple[int, ...] = DEVELOPMENT_SEEDS
    confirmation_seeds: tuple[int, ...] = CONFIRMATION_SEEDS
    model_seed_offset: int = 1001
    selector_seed_offset: int = 5001
    initial_cursor_id: int = 0
    interval: int = 4
    add_epochs: tuple[int, ...] = ADD_EPOCHS
    memory_examples: int = 8
    memory_bytes: int = 192
    max_optimizer_updates: int = 600
    max_process_rss_bytes: int = 256 * 1024 * 1024
    protocol_id: str = PROTOCOL_ID


def fixed_parent_manifest() -> ParentManifest:
    # Why this: hold counts and all non-parent mechanisms fixed; the final
    # phase's existing zero split budget determines the five-add ceiling.
    growth = replace(
        _arm_config("structure_only"),
        sleep_enable_prune=False,
        max_prune_per_sleep=0,
        replay_memory_size=8,
    )
    arms = (
        ParentArm("backprop_off", "backprop", 8, 8, 8),
        ParentArm("pc_off", "pc", 8, 8, 8),
        ParentArm("neutral_off", "circadian", 8, 8, 8, _circadian_config(False)),
        *(
            ParentArm(f"{mode}_growth", "circadian", 8, 8, 13, growth, mode)
            for mode in GROWTH_MODES
        ),
        ParentArm("backprop_13_off", "backprop", 13, 13, 13),
        ParentArm("pc_13_off", "pc", 13, 13, 13),
    )
    return ParentManifest(fixed_replay_pilot_manifest().source, arms)


def validate_parent_manifest(manifest: ParentManifest) -> int:
    if type(manifest) is not ParentManifest or manifest != fixed_parent_manifest():
        raise ValueError("P6.3 parent factor requires its frozen manifest")
    if set(manifest.seeds) & set(manifest.confirmation_seeds):
        raise ValueError("P6.3 parent seed roles overlap")
    arrived._validate_arrived_config(manifest.source, list(manifest.seeds))
    maximum = len(manifest.seeds) * len(manifest.arms) * 24
    if maximum > manifest.max_optimizer_updates:
        raise ValueError("P6.3 parent executed optimizer cap exceeded before launch")
    return maximum
