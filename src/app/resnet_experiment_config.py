"""Resolve the documented descriptive multi-seed ResNet configuration.

Inputs are the named historical preset, typed exposed setting values, and
ordered seeds. Outputs are validated configs and a complete JSON record.
This module does not parse CLI syntax, load images, train, or write files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from math import isfinite
from typing import Literal, Mapping, cast

from src.app.resnet50_benchmark import (
    ResNet50BenchmarkConfig,
    _validate_benchmark_config,
)


MultiSeedResnetPresetId = Literal["historical-unmatched"]
MULTISEED_RESNET_PRESET_ID: MultiSeedResnetPresetId = "historical-unmatched"
RESOLVED_RESNET_CONFIG_SCHEMA = "resnet_multiseed_resolved_config_v1"

_POSITIVE_INTEGER_FIELDS = frozenset(
    {
        "validation_samples",
        "num_classes",
        "image_size",
        "batch_size",
        "epochs",
    }
)
_NONNEGATIVE_INTEGER_FIELDS = frozenset(
    {
        "guard_samples",
        "train_samples",
        "test_samples",
        "dataset_train_subset_size",
        "dataset_guard_subset_size",
        "dataset_test_subset_size",
        "dataset_num_workers",
        "evaluation_batches",
        "inference_batches",
        "warmup_batches",
        "seed",
    }
)
_POSITIVE_SUBSET_FIELDS = frozenset({"dataset_validation_subset_size"})
_FLOAT_FIELDS = frozenset({"dataset_noise_std", "target_accuracy"})
_BOOLEAN_FIELDS = frozenset(
    {"dataset_download", "dataset_use_augmentation", "backprop_freeze_backbone"}
)
_TEXT_FIELDS = frozenset(
    {
        "protocol_id",
        "dataset_name",
        "dataset_data_root",
        "dataset_difficulty",
        "device",
        "backbone_weights",
    }
)
ALLOWED_RESNET_OVERRIDES = (
    _POSITIVE_INTEGER_FIELDS
    | _NONNEGATIVE_INTEGER_FIELDS
    | _POSITIVE_SUBSET_FIELDS
    | _FLOAT_FIELDS
    | _BOOLEAN_FIELDS
    | _TEXT_FIELDS
) - {"seed"}


@dataclass(frozen=True)
class MultiSeedResnetPreset:
    """One named CLI default, preserving the historical unmatched track."""

    preset_id: MultiSeedResnetPresetId
    seeds: tuple[int, ...]
    config: ResNet50BenchmarkConfig
    output_prefix: str


def get_multiseed_resnet_preset(
    preset_id: str = MULTISEED_RESNET_PRESET_ID,
) -> MultiSeedResnetPreset:
    """Return the original CLI defaults, including inherited model settings."""
    if type(preset_id) is not str or preset_id != MULTISEED_RESNET_PRESET_ID:
        raise ValueError(f"unknown multi-seed ResNet preset: {preset_id!r}")
    # Why this: the older CLI already offered a descriptive unmatched track.
    # Keep its exact defaults in one typed object instead of parser literals.
    config = replace(
        ResNet50BenchmarkConfig(),
        train_samples=2500,
        test_samples=700,
        num_classes=100,
        batch_size=64,
        dataset_name="cifar100",
        dataset_difficulty="hard",
        dataset_noise_std=0.08,
        epochs=12,
        device="cuda",
        target_accuracy=None,
        inference_batches=20,
        warmup_batches=5,
        backprop_freeze_backbone=True,
        backbone_weights="imagenet",
    )
    validate_multiseed_resnet_config(config)
    return MultiSeedResnetPreset(
        MULTISEED_RESNET_PRESET_ID, (7, 13, 29), config, "benchmark_multiseed_results"
    )


def _normalize_setting(name: str, value: object) -> object:
    if name in _POSITIVE_INTEGER_FIELDS | _NONNEGATIVE_INTEGER_FIELDS | _POSITIVE_SUBSET_FIELDS:
        if type(value) is not int:
            raise ValueError(f"{name} must be a Python integer")
        return value
    if name in _FLOAT_FIELDS:
        if name == "target_accuracy" and value is None:
            return None
        if type(value) not in {int, float}:
            raise ValueError(f"{name} must be a finite number")
        numeric = cast(int | float, value)
        try:
            finite = isfinite(numeric)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{name} must be a finite number")
        return float(numeric)
    if name in _BOOLEAN_FIELDS:
        if type(value) is not bool:
            raise ValueError(f"{name} must be a Python boolean")
        return value
    if name in _TEXT_FIELDS:
        if type(value) is not str or not value.strip():
            raise ValueError(f"{name} must be nonempty text")
        return value
    raise ValueError(f"unknown multi-seed ResNet setting: {name}")


def validate_multiseed_resnet_config(config: ResNet50BenchmarkConfig) -> None:
    """Reject malformed exposed settings before a benchmark runner opens data."""
    if type(config) is not ResNet50BenchmarkConfig:
        raise TypeError("multi-seed ResNet configuration requires ResNet50BenchmarkConfig")
    for name in ALLOWED_RESNET_OVERRIDES | {"seed"}:
        value = _normalize_setting(name, getattr(config, name))
        if name in _POSITIVE_INTEGER_FIELDS | _POSITIVE_SUBSET_FIELDS:
            if cast(int, value) <= 0:
                raise ValueError(f"{name} must be positive")
        if name in _NONNEGATIVE_INTEGER_FIELDS:
            if cast(int, value) < 0:
                raise ValueError(f"{name} must be non-negative")
    if config.dataset_name == "synthetic" and (
        config.train_samples <= 0 or config.test_samples <= 0
    ):
        raise ValueError("synthetic train_samples and test_samples must be positive")
    _validate_benchmark_config(config)
    try:
        json.dumps(asdict(config), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("multi-seed ResNet config must contain finite JSON values") from exc


def resolve_multiseed_resnet_overrides(
    base: ResNet50BenchmarkConfig, overrides: Mapping[str, object]
) -> ResNet50BenchmarkConfig:
    """Apply only existing CLI setting fields after legacy flags."""
    validate_multiseed_resnet_config(base)
    if not isinstance(overrides, Mapping):
        raise TypeError("multi-seed ResNet overrides must be a key/value mapping")
    normalized: dict[str, object] = {}
    for name, value in overrides.items():
        if type(name) is not str or name not in ALLOWED_RESNET_OVERRIDES:
            raise ValueError(f"unknown multi-seed ResNet override key: {name}")
        normalized[name] = _normalize_setting(name, value)
    resolved = replace(base, **normalized)  # type: ignore[arg-type]
    validate_multiseed_resnet_config(resolved)
    return resolved


def build_resolved_multiseed_resnet_record(
    config: ResNet50BenchmarkConfig,
    seeds: tuple[int, ...],
    preset_id: str,
    explicit_inputs: list[str],
    overrides: Mapping[str, object],
) -> dict[str, object]:
    """Describe the exact base and ordered per-seed configs used by the run."""
    get_multiseed_resnet_preset(preset_id)
    validate_multiseed_resnet_config(config)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("multi-seed ResNet seeds must be non-negative Python integers")
    if any(type(token) is not str for token in explicit_inputs):
        raise ValueError("multi-seed ResNet input tokens must be text")
    for name, value in overrides.items():
        if type(name) is not str or name not in ALLOWED_RESNET_OVERRIDES:
            raise ValueError(f"unknown multi-seed ResNet override key: {name}")
        if getattr(config, name) != _normalize_setting(name, value):
            raise ValueError(f"resolved multi-seed ResNet override {name} differs from config")
    record: dict[str, object] = {
        "schema_id": RESOLVED_RESNET_CONFIG_SCHEMA,
        "preset": preset_id,
        "seeds": list(seeds),
        "explicit_inputs": list(explicit_inputs),
        "overrides": dict(overrides),
        "base_config": asdict(config),
        "trial_configs": [asdict(replace(config, seed=seed)) for seed in seeds],
    }
    try:
        json.dumps(record, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("resolved multi-seed ResNet record must be finite JSON") from exc
    return record
