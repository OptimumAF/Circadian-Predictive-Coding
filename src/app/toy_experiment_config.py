"""Resolve the documented toy CLI's existing settings before training.

Inputs are environment defaults, legacy flags, and named typed overrides.
Outputs are a complete config and ordered execution record. This module
does not parse syntax, train models, inspect scores, or write files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import json
from math import isfinite
from typing import Literal, Mapping, cast

from src.app.experiment_runner import (
    TOY_LEGACY_PROTOCOL,
    TOY_MODEL_ORDER,
    TOY_VALIDATION_PROTOCOL,
    ExperimentConfig,
)
from src.app.indepth_comparison import build_indepth_trial_config
from src.config.settings import Settings
from src.core.circadian_predictive_coding import CircadianConfig

ToyPresetId = Literal["historical-toy"]
TOY_PRESET_ID: ToyPresetId = "historical-toy"
RESOLVED_TOY_CONFIG_SCHEMA = "toy_resolved_config_v1"

_INTEGER_FIELDS = frozenset(
    {"sample_count", "hidden_dim", "epoch_count", "circadian_sleep_interval", "random_seed"}
)
_FLOAT_FIELDS = frozenset({"validation_fraction", "noise_scale"})
_BOOLEAN_FIELDS = frozenset({"circadian_force_sleep", "circadian_use_policy_for_sleep"})
ALLOWED_TOY_OVERRIDES = _INTEGER_FIELDS | _FLOAT_FIELDS | _BOOLEAN_FIELDS | {"hidden_dims"}


@dataclass(frozen=True)
class ToyExperimentPreset:
    """Historical parser defaults under one explicit identity."""

    preset_id: ToyPresetId
    config: ExperimentConfig
    indepth_seeds: tuple[int, ...]
    indepth_noise_levels: tuple[float, ...]


def get_toy_experiment_preset(
    settings: Settings, preset_id: str = TOY_PRESET_ID
) -> ToyExperimentPreset:
    """Bind environment defaults without changing any inherited model setting."""
    if type(preset_id) is not str or preset_id != TOY_PRESET_ID:
        raise ValueError(f"unknown toy preset: {preset_id!r}")
    if type(settings) is not Settings:
        raise TypeError("toy preset requires environment Settings")
    config = replace(
        ExperimentConfig(circadian_config=CircadianConfig()),
        sample_count=settings.dataset_size,
        epoch_count=settings.epoch_count,
        random_seed=settings.base_seed,
    )
    return ToyExperimentPreset(TOY_PRESET_ID, config, (3, 7, 11, 19, 23), (0.6, 0.8, 1.0))


def _normalize_override(name: str, value: object) -> object:
    if name in _INTEGER_FIELDS:
        if type(value) is not int:
            raise ValueError(f"{name} override must be a Python integer")
        return value
    if name in _FLOAT_FIELDS:
        if type(value) not in {int, float} or not _is_finite(value):
            raise ValueError(f"{name} override must be a finite number")
        return float(cast(int | float, value))
    if name in _BOOLEAN_FIELDS:
        if type(value) is not bool:
            raise ValueError(f"{name} override must be a Python boolean")
        return value
    if name == "hidden_dims":
        if value is None:
            return None
        if type(value) not in {list, tuple}:
            raise ValueError("hidden_dims override must contain Python integers")
        values = cast(list[object] | tuple[object, ...], value)
        if not values or any(type(item) is not int for item in values):
            raise ValueError("hidden_dims override must contain Python integers")
        return tuple(values)
    raise ValueError(f"unknown toy override key: {name}")


def resolve_toy_overrides(
    base: ExperimentConfig, overrides: Mapping[str, object]
) -> ExperimentConfig:
    """Apply only already exposed toy settings after legacy flags."""
    validate_toy_experiment_config(base)
    if not isinstance(overrides, Mapping):
        raise TypeError("toy overrides must be a key/value mapping")
    normalized: dict[str, object] = {}
    for name, value in overrides.items():
        if type(name) is not str or name not in ALLOWED_TOY_OVERRIDES:
            raise ValueError(f"unknown toy override key: {name}")
        normalized[name] = _normalize_override(name, value)
    # Why this: model order and all three learning-rate defaults stay fixed.
    resolved = replace(base, **normalized)  # type: ignore[arg-type]
    validate_toy_experiment_config(resolved)
    return resolved


def _is_finite(value: object) -> bool:
    try:
        return isfinite(cast(int | float, value))
    except (TypeError, ValueError, OverflowError):
        return False


def validate_toy_experiment_config(config: ExperimentConfig) -> None:
    """Reject malformed exposed settings before data or model construction."""
    if type(config) is not ExperimentConfig:
        raise TypeError("toy configuration requires ExperimentConfig")
    for name in _INTEGER_FIELDS:
        if type(getattr(config, name)) is not int:
            raise ValueError(f"{name} must be a Python integer")
    if config.sample_count < 20 or config.hidden_dim <= 0 or config.epoch_count <= 0:
        raise ValueError("sample_count >= 20, hidden_dim > 0, and epoch_count > 0 are required")
    if config.circadian_sleep_interval < 0 or config.random_seed < 0:
        raise ValueError("circadian_sleep_interval and random_seed must be non-negative")
    if config.hidden_dims is not None and (
        type(config.hidden_dims) is not tuple
        or not config.hidden_dims
        or any(type(width) is not int or width <= 0 for width in config.hidden_dims)
    ):
        raise ValueError("hidden_dims must contain positive Python integers")
    adaptive_width = config.hidden_dims[-1] if config.hidden_dims is not None else config.hidden_dim
    if adaptive_width < 4:
        raise ValueError("circadian adaptive hidden width must be at least 4")
    if config.protocol_id not in {TOY_VALIDATION_PROTOCOL, TOY_LEGACY_PROTOCOL}:
        raise ValueError("unknown toy protocol_id")
    if config.model_order != TOY_MODEL_ORDER:
        raise ValueError("toy CLI model_order must remain the historical order")
    for name in _FLOAT_FIELDS | {
        "backprop_learning_rate",
        "pc_learning_rate",
        "pc_inference_learning_rate",
        "circadian_learning_rate",
        "circadian_inference_learning_rate",
    }:
        value = getattr(config, name)
        if type(value) not in {int, float} or not _is_finite(value):
            raise ValueError(f"{name} must be finite")
    if config.protocol_id == TOY_VALIDATION_PROTOCOL and not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if config.noise_scale <= 0.0:
        raise ValueError("noise_scale must be positive")
    for name in (
        "backprop_learning_rate",
        "pc_learning_rate",
        "pc_inference_learning_rate",
        "circadian_learning_rate",
        "circadian_inference_learning_rate",
    ):
        if getattr(config, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name in ("pc_inference_steps", "circadian_inference_steps"):
        value = getattr(config, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive Python integer")
    for name in _BOOLEAN_FIELDS:
        if type(getattr(config, name)) is not bool:
            raise ValueError(f"{name} must be a Python boolean")
    _validate_exposed_circadian_config(config.circadian_config)
    try:
        json.dumps(asdict(config), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("toy config must contain finite JSON values") from exc


def _validate_exposed_circadian_config(config: CircadianConfig | None) -> None:
    if type(config) is not CircadianConfig:
        raise ValueError("toy CLI requires CircadianConfig")
    for item in fields(config):
        value = getattr(config, item.name)
        default = item.default
        if type(default) is bool and type(value) is not bool:
            raise ValueError(f"{item.name} must be a Python boolean")
        if type(default) is int and type(value) is not int:
            raise ValueError(f"{item.name} must be a Python integer")
        if type(default) is float and (type(value) not in {int, float} or not _is_finite(value)):
            raise ValueError(f"{item.name} must be finite")
    for name in ("adaptive_split_percentile", "adaptive_prune_percentile"):
        if not 0 <= getattr(config, name) <= 100:
            raise ValueError(f"{name} must be between 0 and 100")
    for name in ("split_weight_norm_mix", "prune_weight_norm_mix"):
        if not 0 <= getattr(config, name) <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    for name in ("min_epochs_between_sleep", "replay_steps", "replay_memory_size"):
        if getattr(config, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if config.sleep_energy_window <= 1 or config.prune_decay_steps < 1:
        raise ValueError("sleep_energy_window > 1 and prune_decay_steps >= 1 are required")
    for name in (
        "reward_scale_min",
        "adaptive_sleep_budget_min_scale",
    ):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.replay_steps > 0 and (
        config.replay_learning_rate <= 0
        or config.replay_inference_steps <= 0
        or config.replay_inference_learning_rate <= 0
    ):
        raise ValueError("enabled replay requires positive learning rates and inference steps")
    if config.reward_scale_max < config.reward_scale_min:
        raise ValueError("reward_scale_max must be >= reward_scale_min")
    if (
        not 0
        < config.adaptive_sleep_budget_min_scale
        <= config.adaptive_sleep_budget_max_scale
        <= 1
    ):
        raise ValueError("adaptive_sleep_budget_max_scale must be between min scale and 1")
    for name in ("sleep_plateau_delta", "sleep_chemical_variance_threshold"):
        if getattr(config, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    for name in ("prune_decay_factor", "homeostatic_downscale_factor"):
        if not 0 < getattr(config, name) <= 1:
            raise ValueError(f"{name} must be in (0, 1]")


def build_resolved_toy_record(
    config: ExperimentConfig,
    mode: Literal["baseline", "indepth"],
    seeds: list[int],
    noise_levels: list[float],
    preset_id: str,
    explicit_inputs: list[str],
    overrides: Mapping[str, object],
) -> dict[str, object]:
    """Record every setting actually used, in execution order."""
    get_toy_experiment_preset(Settings(), preset_id)
    validate_toy_experiment_config(config)
    if mode not in {"baseline", "indepth"}:
        raise ValueError("unknown toy mode")
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("toy seeds must be non-negative Python integers")
    if not noise_levels or any(
        type(noise) not in {int, float} or not _is_finite(noise) or noise <= 0
        for noise in noise_levels
    ):
        raise ValueError("toy noise levels must be positive finite numbers")
    if mode == "baseline" and (
        seeds != [config.random_seed] or noise_levels != [config.noise_scale]
    ):
        raise ValueError("baseline grid must match its config")
    if any(type(token) is not str for token in explicit_inputs):
        raise ValueError("toy input tokens must be text")
    for name, value in overrides.items():
        if type(name) is not str or name not in ALLOWED_TOY_OVERRIDES:
            raise ValueError(f"unknown toy override key: {name}")
        if getattr(config, name) != _normalize_override(name, value):
            raise ValueError(f"resolved toy override {name} differs from config")
    trials = (
        [config]
        if mode == "baseline"
        else [
            build_indepth_trial_config(config, seed, noise)
            for noise in noise_levels
            for seed in seeds
        ]
    )
    record: dict[str, object] = {
        "schema_id": RESOLVED_TOY_CONFIG_SCHEMA,
        "preset": preset_id,
        "mode": mode,
        "seeds": list(seeds),
        "noise_levels": list(noise_levels),
        "explicit_inputs": list(explicit_inputs),
        "overrides": dict(overrides),
        "config": asdict(config),
        "trial_configs": [asdict(trial) for trial in trials],
    }
    try:
        json.dumps(record, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("resolved toy record must contain finite JSON values") from exc
    return record
