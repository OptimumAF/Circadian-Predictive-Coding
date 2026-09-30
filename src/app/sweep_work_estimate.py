"""Estimate synthetic-vision training work before any external resource opens.

Inputs are resolved candidate configs and a seed count. The output is a
conservative optimizer-call and training-example bound. This module does not
measure wall time, memory, validation work, or perform training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.app.resnet50_benchmark import ResNet50BenchmarkConfig


@dataclass(frozen=True)
class SweepWorkEstimate:
    candidate_count: int
    seed_count: int
    trial_count: int
    planned_max_training_updates: int
    planned_max_training_examples: int
    counting_rule: str = "epochs * ceil(train_examples / batch_size) per candidate and seed"


def estimate_vision_candidate_work(
    configs: Sequence[ResNet50BenchmarkConfig], *, seed_count: int
) -> SweepWorkEstimate:
    """Count every planned synthetic training batch, including short last batches."""
    if not configs:
        raise ValueError("candidate configs must be non-empty")
    if type(seed_count) is not int or seed_count <= 0:
        raise ValueError("seed_count must be a positive integer")
    updates = 0
    examples = 0
    for config in configs:
        if type(config) is not ResNet50BenchmarkConfig:
            raise ValueError("candidate must be a ResNet50BenchmarkConfig")
        if config.dataset_name != "synthetic":
            raise ValueError("sweep estimate currently supports synthetic data only")
        for field_name in ("train_samples", "batch_size", "epochs"):
            value = getattr(config, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        # Why this: the synthetic loader does not drop its final short batch.
        updates += config.epochs * (
            (config.train_samples + config.batch_size - 1) // config.batch_size
        )
        examples += config.epochs * config.train_samples
    return SweepWorkEstimate(
        len(configs),
        seed_count,
        len(configs) * seed_count,
        updates * seed_count,
        examples * seed_count,
    )


def require_planned_training_limit(estimate: SweepWorkEstimate, limit: object) -> None:
    """Refuse an oversized launch; this is not a runtime work limiter."""
    if type(limit) is not int or limit <= 0:
        raise ValueError("launch limit must be a positive integer")
    if estimate.planned_max_training_updates > limit:
        raise ValueError(
            f"estimated {estimate.planned_max_training_updates} training updates "
            f"exceeds launch limit {limit}; set --max-planned-training-updates explicitly"
        )
