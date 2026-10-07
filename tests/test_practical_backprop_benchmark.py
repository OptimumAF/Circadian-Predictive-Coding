"""Boundary and metadata gates for the end-to-end practical reference."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.app import practical_backprop_benchmark  # noqa: E402
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig  # noqa: E402


def tiny_config() -> ResNet50BenchmarkConfig:
    return ResNet50BenchmarkConfig(
        train_samples=8, guard_samples=8, validation_samples=8, test_samples=8,
        num_classes=3, image_size=32, batch_size=4, epochs=1, seed=61,
        device="cpu", target_accuracy=None, backbone_weights="none",
        backprop_freeze_backbone=False, inference_batches=1, warmup_batches=0,
    )


def test_practical_route_trains_end_to_end_and_opens_final_test_after_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_build = practical_backprop_benchmark._build_benchmark_loaders
    original_train = practical_backprop_benchmark._train_backprop
    training_done = False

    class SealedTestLoader:
        def __init__(self, original: Any) -> None:
            self.original = original

        def __iter__(self) -> Any:
            if not training_done:
                raise AssertionError("Final test opened before end-to-end training")
            return iter(self.original)

    def build_loaders(config: ResNet50BenchmarkConfig) -> Any:
        loaders = original_build(config)
        return replace(loaders, test_loader=SealedTestLoader(loaders.test_loader))

    def train(*args: Any, **kwargs: Any) -> Any:
        nonlocal training_done
        outcome = original_train(*args, **kwargs)
        training_done = True
        return outcome

    monkeypatch.setattr(practical_backprop_benchmark, "_build_benchmark_loaders", build_loaders)
    monkeypatch.setattr(practical_backprop_benchmark, "_train_backprop", train)

    result = practical_backprop_benchmark.run_practical_backprop_benchmark(tiny_config())

    assert training_done
    assert result.protocol_id == "vision_end_to_end_backprop_v1"
    assert result.source_protocol_id == "vision_guard_separated_unmatched_v2"
    assert set(result.split_hashes) == {"train", "guard", "validation", "test"}
    assert result.report.benchmark_track == "end_to_end_backprop"
    assert result.report.backbone_trainable is True
    assert result.report.backbone_pretraining == "none"
    assert result.report.head_type == "linear"
    assert result.report.total_parameters == result.report.trainable_parameters
    assert 0.0 <= result.report.test_accuracy <= 1.0


def test_practical_route_rejects_frozen_backbone() -> None:
    with pytest.raises(ValueError, match="trainable backbone"):
        practical_backprop_benchmark.run_practical_backprop_benchmark(
            replace(tiny_config(), backprop_freeze_backbone=True)
        )
