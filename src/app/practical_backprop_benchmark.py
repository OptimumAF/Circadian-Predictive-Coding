"""End-to-end backprop reference on guarded vision splits.

This route takes a ResNet benchmark config, trains one unfrozen backbone and
linear head, then returns a descriptive final-test report. It does not rank
against the separate fixed-feature learning-rule track.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.app.resnet50_benchmark import (
    ModelSpeedReport,
    ResNet50BenchmarkConfig,
    VISION_END_TO_END_BACKPROP_TRACK,
    VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL,
    _build_benchmark_loaders,
    _finalize_test_report,
    _resolve_device,
    _set_seed,
    _train_backprop,
    _training_loaders,
    _validate_benchmark_config,
)
from src.shared.torch_runtime import require_torch

END_TO_END_BACKPROP_PROTOCOL = "vision_end_to_end_backprop_v1"


@dataclass(frozen=True)
class PracticalBackpropResult:
    protocol_id: str
    source_protocol_id: str
    device: str
    split_hashes: dict[str, str]
    report: ModelSpeedReport


def run_practical_backprop_benchmark(
    config: ResNet50BenchmarkConfig,
) -> PracticalBackpropResult:
    """Train a practical ResNet reference and score final test afterward."""
    _validate_benchmark_config(config)
    if config.protocol_id != VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL:
        raise ValueError("Practical route requires the guard-separated vision protocol.")
    if config.backprop_freeze_backbone:
        raise ValueError("Practical route requires a trainable backbone.")

    torch = require_torch()
    _set_seed(torch, config.seed)
    device = _resolve_device(torch, config.device)
    loaders = _build_benchmark_loaders(config)
    if "guard" not in loaders.split_hashes or loaders.guard_loader is loaders.validation_loader:
        raise ValueError("Practical route requires a distinct guard loader.")

    trained = _train_backprop(
        torch=torch, device=device, loaders=_training_loaders(loaders), config=config,
    )
    report = _finalize_test_report(
        torch=torch, device=device, outcome=trained,
        test_loader=loaders.test_loader, config=config,
        benchmark_track=VISION_END_TO_END_BACKPROP_TRACK,
    )
    return PracticalBackpropResult(
        protocol_id=END_TO_END_BACKPROP_PROTOCOL,
        source_protocol_id=config.protocol_id,
        device=str(device), split_hashes=dict(loaders.split_hashes), report=report,
    )
