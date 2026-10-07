"""Resolve the documented descriptive single-run ResNet configuration.

Inputs are a named historical preset, the existing CLI settings, and typed
overrides. Outputs are a validated benchmark config and complete record.
This module does not parse syntax, load images, train, score, or write files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import json
from math import isfinite
from typing import Literal, Mapping, cast

from src.app.resnet50_benchmark import (
    ResNet50BenchmarkConfig,
    VISION_DEFAULT_MODEL_ORDER,
    VISION_UNMATCHED_REFERENCE_TRACK,
    _validate_benchmark_config,
)

SingleResnetPresetId = Literal["historical-single-unmatched"]
SINGLE_RESNET_PRESET_ID: SingleResnetPresetId = "historical-single-unmatched"
RESOLVED_SINGLE_RESNET_SCHEMA = "resnet_single_resolved_config_v1"
ALLOWED_SINGLE_RESNET_OVERRIDES = frozenset(item.name for item in fields(ResNet50BenchmarkConfig))


@dataclass(frozen=True)
class SingleResnetPreset:
    """The original one-run unmatched reference settings."""

    preset_id: SingleResnetPresetId
    config: ResNet50BenchmarkConfig


def get_single_resnet_preset(
    preset_id: str = SINGLE_RESNET_PRESET_ID,
) -> SingleResnetPreset:
    """Return the unchanged dataclass defaults under an explicit identity."""
    if type(preset_id) is not str or preset_id != SINGLE_RESNET_PRESET_ID:
        raise ValueError(f"unknown single-run ResNet preset: {preset_id!r}")
    config = ResNet50BenchmarkConfig()
    validate_single_resnet_config(config)
    return SingleResnetPreset(SINGLE_RESNET_PRESET_ID, config)


def _is_finite(value: object) -> bool:
    try:
        return isfinite(cast(int | float, value))
    except (TypeError, ValueError, OverflowError):
        return False


def _normalize_setting(name: str, value: object) -> object:
    if name not in ALLOWED_SINGLE_RESNET_OVERRIDES:
        raise ValueError(f"unknown single-run ResNet override key: {name}")
    default = getattr(ResNet50BenchmarkConfig(), name)
    if name == "target_accuracy":
        if value is None:
            return None
        if type(value) not in {int, float} or not _is_finite(value):
            raise ValueError(f"{name} must be a finite number or null")
        return float(cast(int | float, value))
    if name == "circadian_sleep_rollback_cooldown_epochs":
        if value is not None and type(value) is not int:
            raise ValueError(f"{name} must be a Python integer or null")
        return value
    if type(default) is bool:
        if type(value) is not bool:
            raise ValueError(f"{name} must be a Python boolean")
        return value
    if type(default) is int:
        if type(value) is not int:
            raise ValueError(f"{name} must be a Python integer")
        return value
    if type(default) is float:
        if type(value) not in {int, float} or not _is_finite(value):
            raise ValueError(f"{name} must be a finite number")
        return float(cast(int | float, value))
    if type(default) is str:
        if type(value) is not str or (name != "dataset_data_root" and not value.strip()):
            raise ValueError(f"{name} must be text")
        return value
    raise ValueError(f"unsupported single-run ResNet setting: {name}")


def validate_single_resnet_config(config: ResNet50BenchmarkConfig) -> None:
    """Check strict scalar types and runner constraints before Torch starts."""
    if type(config) is not ResNet50BenchmarkConfig:
        raise TypeError("single-run ResNet requires ResNet50BenchmarkConfig")
    for item in fields(config):
        _normalize_setting(item.name, getattr(config, item.name))
    _validate_benchmark_config(config)
    if config.num_classes <= 1:
        raise ValueError("num_classes must be greater than 1")
    if config.image_size < 32:
        raise ValueError("image_size must be at least 32")
    if config.dataset_name == "synthetic" and (
        config.train_samples <= 0 or config.test_samples <= 0
    ):
        raise ValueError("synthetic train_samples and test_samples must be positive")
    for name in (
        "predictive_head_hidden_dim",
        "circadian_head_hidden_dim",
        "circadian_min_hidden_dim",
        "circadian_max_hidden_dim",
        "predictive_inference_steps",
        "circadian_inference_steps",
    ):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in (
        "predictive_learning_rate",
        "predictive_inference_learning_rate",
        "circadian_learning_rate",
        "circadian_inference_learning_rate",
    ):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.backprop_learning_rate < 0:
        raise ValueError("backprop_learning_rate must be non-negative")
    try:
        json.dumps(asdict(config), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("single-run ResNet config must contain finite JSON values") from exc


def resolve_single_resnet_overrides(
    base: ResNet50BenchmarkConfig, overrides: Mapping[str, object]
) -> ResNet50BenchmarkConfig:
    """Apply strict values after legacy flags, then validate the final config."""
    if type(base) is not ResNet50BenchmarkConfig or not isinstance(overrides, Mapping):
        raise TypeError("single-run ResNet resolver requires a config and key/value mapping")
    normalized: dict[str, object] = {}
    for name, value in overrides.items():
        if type(name) is not str:
            raise ValueError(f"unknown single-run ResNet override key: {name}")
        normalized[name] = _normalize_setting(name, value)
    resolved = replace(base, **normalized)  # type: ignore[arg-type]
    validate_single_resnet_config(resolved)
    return resolved


def build_resolved_single_resnet_record(
    config: ResNet50BenchmarkConfig,
    preset_id: str,
    explicit_inputs: list[str],
    overrides: Mapping[str, object],
) -> dict[str, object]:
    """Describe the exact unmatched request without reading model scores."""
    get_single_resnet_preset(preset_id)
    validate_single_resnet_config(config)
    if any(type(token) is not str for token in explicit_inputs):
        raise ValueError("single-run ResNet input tokens must be text")
    for name, value in overrides.items():
        if type(name) is not str or name not in ALLOWED_SINGLE_RESNET_OVERRIDES:
            raise ValueError(f"unknown single-run ResNet override key: {name}")
        if getattr(config, name) != _normalize_setting(name, value):
            raise ValueError(f"resolved single-run ResNet override {name} differs from config")
    record: dict[str, object] = {
        "schema_id": RESOLVED_SINGLE_RESNET_SCHEMA,
        "preset": preset_id,
        "benchmark_track": VISION_UNMATCHED_REFERENCE_TRACK,
        "training_order": list(VISION_DEFAULT_MODEL_ORDER),
        "seed": config.seed,
        "explicit_inputs": list(explicit_inputs),
        "overrides": dict(overrides),
        "config": asdict(config),
    }
    try:
        json.dumps(record, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("resolved single-run ResNet record must be finite JSON") from exc
    return record
