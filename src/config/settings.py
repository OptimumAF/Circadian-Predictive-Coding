"""Environment-backed defaults for experiment configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Global settings loaded from environment variables."""

    base_seed: int = 7
    dataset_size: int = 400
    epoch_count: int = 160


def load_settings_from_env() -> Settings:
    """Load settings from process environment with safe fallbacks."""
    return Settings(
        base_seed=_read_int_env("PC_BASE_SEED", 7),
        dataset_size=_read_int_env("PC_DATASET_SIZE", 400),
        epoch_count=_read_int_env("PC_EPOCHS", 160),
    )


def _read_int_env(key: str, default_value: int) -> int:
    raw_value = os.getenv(key)
    if raw_value is None:
        return default_value
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key} must be an integer.") from exc

