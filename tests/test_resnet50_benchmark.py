from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.app.resnet50_benchmark import ResNet50BenchmarkConfig, run_resnet50_benchmark


def test_should_run_resnet50_benchmark_and_return_three_reports() -> None:
    config = ResNet50BenchmarkConfig(
        train_samples=48,
        test_samples=24,
        num_classes=6,
        image_size=64,
        batch_size=8,
        dataset_difficulty="hard",
        dataset_noise_std=0.08,
        epochs=1,
        seed=5,
        device="cpu",
        target_accuracy=None,
        inference_batches=3,
        evaluation_batches=1,
        warmup_batches=1,
        backprop_freeze_backbone=True,
        predictive_head_hidden_dim=64,
        circadian_head_hidden_dim=64,
        circadian_min_hidden_dim=32,
        circadian_max_hidden_dim=128,
    )

    result = run_resnet50_benchmark(config)
    assert len(result.reports) == 3
    assert {report.model_name for report in result.reports} == {
        "BackpropResNet50",
        "PredictiveCodingResNet50",
        "CircadianPredictiveCodingResNet50",
    }

    for report in result.reports:
        assert report.train_seconds >= 0.0
        assert report.train_samples_per_second >= 0.0
        assert report.inference_samples_per_second >= 0.0
        assert 0.0 <= report.test_accuracy <= 1.0
        assert report.total_parameters > 0
        assert report.trainable_parameters > 0
        assert report.circadian_total_rollbacks >= 0
        assert report.final_cross_entropy is not None
        if report.model_name != "BackpropResNet50":
            assert report.final_energy is not None


def test_should_validate_new_sleep_config_values() -> None:
    config = ResNet50BenchmarkConfig(
        circadian_sleep_energy_window=1,
    )

    with pytest.raises(ValueError):
        _ = run_resnet50_benchmark(config)


def test_should_validate_rollback_metric_name() -> None:
    config = ResNet50BenchmarkConfig(circadian_sleep_rollback_metric="invalid")

    with pytest.raises(ValueError):
        _ = run_resnet50_benchmark(config)


def test_should_validate_hidden_dim_bounds() -> None:
    config = ResNet50BenchmarkConfig(
        circadian_head_hidden_dim=64,
        circadian_min_hidden_dim=96,
    )

    with pytest.raises(ValueError):
        _ = run_resnet50_benchmark(config)
