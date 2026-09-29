"""Resolve explicit overrides for existing continual-shift presets.

Inputs are a typed preset-derived config and named JSON-compatible values.
Output is a validated config of the same protocol type. This module does
not parse CLI syntax, train, write results, or alter fixed v14 manifests.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from math import isfinite
from typing import Mapping, cast

from src.app.continual_shift_benchmark import ContinualShiftConfig, _validate_config


_INTEGER_FIELDS = frozenset(
    {
        "sample_count_phase_a",
        "sample_count_phase_b",
        "phase_a_epochs",
        "phase_b_epochs",
        "hidden_dim",
        "circadian_sleep_interval_phase_a",
        "circadian_sleep_interval_phase_b",
    }
)
_FLOAT_FIELDS = frozenset(
    {
        "validation_fraction",
        "phase_b_train_fraction",
        "phase_a_noise_scale",
        "phase_b_noise_scale",
        "phase_b_rotation_degrees",
        "phase_b_translation_x",
        "phase_b_translation_y",
    }
)
ALLOWED_CONTINUAL_OVERRIDES = _INTEGER_FIELDS | _FLOAT_FIELDS | {"hidden_dims"}
RESOLVED_CONTINUAL_CONFIG_SCHEMA = "continual_resolved_config_v1"


def _normalize_override(name: str, value: object) -> object:
    if name in _INTEGER_FIELDS:
        if type(value) is not int:
            raise ValueError(f"{name} override must be a Python integer")
        return value
    if name in _FLOAT_FIELDS:
        if type(value) not in {int, float}:
            raise ValueError(f"{name} override must be a finite number")
        numeric = cast(int | float, value)
        if not isfinite(numeric):
            raise ValueError(f"{name} override must be a finite number")
        return float(numeric)
    if name == "hidden_dims":
        if value is None:
            return None
        if type(value) not in {list, tuple}:
            raise ValueError("hidden_dims override must contain Python integers")
        values = cast(list[object] | tuple[object, ...], value)
        if any(type(item) is not int for item in values):
            raise ValueError("hidden_dims override must contain Python integers")
        return tuple(values)
    raise ValueError(f"unknown continual override key: {name}")


def resolve_continual_overrides(
    base: ContinualShiftConfig, overrides: Mapping[str, object]
) -> ContinualShiftConfig:
    """Apply a finite, typed allowlist after preset and legacy CLI flags."""
    if not isinstance(base, ContinualShiftConfig) or not isinstance(overrides, Mapping):
        raise TypeError("continual override resolver requires a config and key/value mapping")
    normalized: dict[str, object] = {}
    for name, value in overrides.items():
        if type(name) is not str or name not in ALLOWED_CONTINUAL_OVERRIDES:
            raise ValueError(f"unknown continual override key: {name}")
        normalized[name] = _normalize_override(name, value)
    # Why this: replace retains the exact existing protocol dataclass;
    # users cannot change the model order, baseline rates, or protocol ID.
    resolved = replace(base, **normalized)  # type: ignore[arg-type]
    _validate_config(resolved)
    return resolved


def build_resolved_continual_record(
    config: ContinualShiftConfig,
    seeds: list[int],
    preset: str,
    overrides: Mapping[str, object],
) -> dict[str, object]:
    """Describe the exact typed config and explicit inputs used by a run."""
    _validate_config(config)
    if preset not in {"baseline", "strength-case", "hardest-case"}:
        raise ValueError("unknown continual preset")
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("continual resolved seeds must be integers")
    for name, value in overrides.items():
        if type(name) is not str or name not in ALLOWED_CONTINUAL_OVERRIDES:
            raise ValueError(f"unknown continual override key: {name}")
        if getattr(config, name) != _normalize_override(name, value):
            raise ValueError(f"resolved continual override {name} differs from config")
    record: dict[str, object] = {
        "schema_id": RESOLVED_CONTINUAL_CONFIG_SCHEMA,
        "preset": preset,
        "seeds": seeds,
        "overrides": dict(overrides),
        "config": asdict(config),
    }
    try:
        json.dumps(record, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("resolved continual config must contain finite JSON values") from exc
    return record
