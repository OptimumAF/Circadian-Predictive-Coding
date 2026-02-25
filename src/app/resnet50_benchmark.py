"""Application workflow for ResNet-50 speed and accuracy benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np

from src.core.resnet50_variants import (
    BackpropResNet50Classifier,
    CircadianPredictiveCodingResNet50Classifier,
    PredictiveCodingResNet50Classifier,
)
from src.infra.vision_datasets import SyntheticVisionDatasetConfig, build_synthetic_vision_dataloaders
from src.shared.torch_runtime import require_torch, sync_device


@dataclass(frozen=True)
class ResNet50BenchmarkConfig:
    """Configurable benchmark settings for all three ResNet-50 variants."""

    train_samples: int = 2000
    test_samples: int = 500
    num_classes: int = 10
    image_size: int = 96
    batch_size: int = 32
    epochs: int = 8
    seed: int = 7
    device: str = "auto"
    target_accuracy: float | None = 0.99
    inference_batches: int = 50
    warmup_batches: int = 10

    backprop_learning_rate: float = 0.02
    backprop_momentum: float = 0.9
    backprop_freeze_backbone: bool = False

    predictive_head_hidden_dim: int = 256
    predictive_learning_rate: float = 0.03
    predictive_inference_steps: int = 10
    predictive_inference_learning_rate: float = 0.15

    circadian_head_hidden_dim: int = 256
    circadian_learning_rate: float = 0.03
    circadian_inference_steps: int = 10
    circadian_inference_learning_rate: float = 0.15
    circadian_sleep_interval: int = 2
    circadian_min_hidden_dim: int = 96
    circadian_max_hidden_dim: int = 1024


@dataclass(frozen=True)
class ModelSpeedReport:
    """Training and inference speed metrics for a model."""

    model_name: str
    epochs_ran: int
    final_metric_name: str
    final_metric_value: float
    test_accuracy: float
    train_seconds: float
    train_samples_per_second: float
    mean_train_step_ms: float
    inference_latency_mean_ms: float
    inference_latency_p95_ms: float
    inference_samples_per_second: float
    total_parameters: int
    trainable_parameters: int
    circadian_hidden_dim_start: int | None = None
    circadian_hidden_dim_end: int | None = None
    circadian_total_splits: int = 0
    circadian_total_prunes: int = 0


@dataclass(frozen=True)
class ResNet50BenchmarkResult:
    """Benchmark result across all model variants."""

    device: str
    config: ResNet50BenchmarkConfig
    reports: list[ModelSpeedReport]


def run_resnet50_benchmark(config: ResNet50BenchmarkConfig) -> ResNet50BenchmarkResult:
    """Benchmark all three model families on the same synthetic image task."""
    torch = require_torch()
    _set_seed(torch, config.seed)
    device = _resolve_device(torch, config.device)

    loaders = build_synthetic_vision_dataloaders(
        SyntheticVisionDatasetConfig(
            train_samples=config.train_samples,
            test_samples=config.test_samples,
            num_classes=config.num_classes,
            image_size=config.image_size,
            batch_size=config.batch_size,
            seed=config.seed,
        )
    )

    reports = [
        _benchmark_backprop(torch=torch, device=device, loaders=loaders, config=config),
        _benchmark_predictive(torch=torch, device=device, loaders=loaders, config=config),
        _benchmark_circadian(torch=torch, device=device, loaders=loaders, config=config),
    ]
    return ResNet50BenchmarkResult(device=str(device), config=config, reports=reports)


def format_resnet50_benchmark_result(result: ResNet50BenchmarkResult) -> str:
    """Create a readable benchmark report."""
    lines = [
        "ResNet-50 Speed Benchmark (Backprop vs Predictive vs Circadian)",
        "---------------------------------------------------------------",
        f"Device: {result.device}",
        (
            "Dataset: "
            f"train={result.config.train_samples}, test={result.config.test_samples}, "
            f"classes={result.config.num_classes}, image={result.config.image_size}x{result.config.image_size}"
        ),
        "",
    ]
    for report in result.reports:
        lines.extend(
            [
                f"{report.model_name}",
                (
                    f"  epochs={report.epochs_ran}, "
                    f"{report.final_metric_name}={report.final_metric_value:.4f}, "
                    f"acc={report.test_accuracy:.3f}"
                ),
                (
                    f"  training: {report.train_seconds:.2f}s total, "
                    f"{report.train_samples_per_second:.1f} samples/s, "
                    f"{report.mean_train_step_ms:.2f} ms/step"
                ),
                (
                    f"  inference: mean={report.inference_latency_mean_ms:.2f} ms, "
                    f"p95={report.inference_latency_p95_ms:.2f} ms, "
                    f"{report.inference_samples_per_second:.1f} samples/s"
                ),
                (
                    f"  params: total={report.total_parameters:,}, "
                    f"trainable={report.trainable_parameters:,}"
                ),
            ]
        )
        if report.circadian_hidden_dim_start is not None and report.circadian_hidden_dim_end is not None:
            lines.append(
                (
                    "  circadian sleep: "
                    f"hidden={report.circadian_hidden_dim_start}->{report.circadian_hidden_dim_end}, "
                    f"splits={report.circadian_total_splits}, prunes={report.circadian_total_prunes}"
                )
            )
        lines.append("")
    return "\n".join(lines).strip()


def _benchmark_backprop(
    torch: Any,
    device: Any,
    loaders: Any,
    config: ResNet50BenchmarkConfig,
) -> ModelSpeedReport:
    model = BackpropResNet50Classifier(
        num_classes=loaders.num_classes,
        device=device,
        freeze_backbone=config.backprop_freeze_backbone,
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.trainable_parameters(),
        lr=config.backprop_learning_rate,
        momentum=config.backprop_momentum,
    )

    step_times_ms: list[float] = []
    seen_samples = 0
    epochs_ran = 0
    final_loss = 0.0

    if config.backprop_freeze_backbone:
        model.backbone.eval()
    else:
        model.backbone.train()
    model.classifier.train()

    train_timer_start = perf_counter()
    for epoch in range(1, config.epochs + 1):
        for images, labels in loaders.train_loader:
            images = images.to(device)
            labels = labels.to(device)

            sync_device(torch, device)
            step_start = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_logits(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            sync_device(torch, device)

            step_times_ms.append((perf_counter() - step_start) * 1000.0)
            final_loss = float(loss.item())
            seen_samples += int(labels.shape[0])

        epochs_ran = epoch
        if _should_stop_early(
            target_accuracy=config.target_accuracy,
            accuracy=_compute_backprop_accuracy(torch, model, loaders.test_loader, device),
        ):
            break
    train_seconds = perf_counter() - train_timer_start

    model.backbone.eval()
    model.classifier.eval()
    test_accuracy = _compute_backprop_accuracy(torch, model, loaders.test_loader, device)
    inference_metrics = _benchmark_inference(
        torch=torch,
        device=device,
        loader=loaders.test_loader,
        forward_logits=lambda x: model.forward_logits(x),
        warmup_batches=config.warmup_batches,
        benchmark_batches=config.inference_batches,
    )
    return ModelSpeedReport(
        model_name="BackpropResNet50",
        epochs_ran=epochs_ran,
        final_metric_name="loss",
        final_metric_value=final_loss,
        test_accuracy=test_accuracy,
        train_seconds=train_seconds,
        train_samples_per_second=_safe_div(seen_samples, train_seconds),
        mean_train_step_ms=float(np.mean(step_times_ms)) if step_times_ms else 0.0,
        inference_latency_mean_ms=inference_metrics["latency_mean_ms"],
        inference_latency_p95_ms=inference_metrics["latency_p95_ms"],
        inference_samples_per_second=inference_metrics["samples_per_second"],
        total_parameters=model.parameter_count(),
        trainable_parameters=model.trainable_parameter_count(),
    )


def _benchmark_predictive(
    torch: Any,
    device: Any,
    loaders: Any,
    config: ResNet50BenchmarkConfig,
) -> ModelSpeedReport:
    model = PredictiveCodingResNet50Classifier(
        num_classes=loaders.num_classes,
        device=device,
        head_hidden_dim=config.predictive_head_hidden_dim,
        seed=config.seed + 11,
        freeze_backbone=True,
    )

    step_times_ms: list[float] = []
    seen_samples = 0
    epochs_ran = 0
    final_energy = 0.0

    train_timer_start = perf_counter()
    for epoch in range(1, config.epochs + 1):
        for images, labels in loaders.train_loader:
            images = images.to(device)
            labels = labels.to(device)

            sync_device(torch, device)
            step_start = perf_counter()
            energy = model.train_step(
                images=images,
                targets=labels,
                learning_rate=config.predictive_learning_rate,
                inference_steps=config.predictive_inference_steps,
                inference_learning_rate=config.predictive_inference_learning_rate,
            )
            sync_device(torch, device)

            step_times_ms.append((perf_counter() - step_start) * 1000.0)
            final_energy = float(energy)
            seen_samples += int(labels.shape[0])

        epochs_ran = epoch
        if _should_stop_early(
            target_accuracy=config.target_accuracy,
            accuracy=_compute_pc_accuracy(torch, model, loaders.test_loader, device),
        ):
            break
    train_seconds = perf_counter() - train_timer_start

    test_accuracy = _compute_pc_accuracy(torch, model, loaders.test_loader, device)
    inference_metrics = _benchmark_inference(
        torch=torch,
        device=device,
        loader=loaders.test_loader,
        forward_logits=lambda x: model.predict_logits(x),
        warmup_batches=config.warmup_batches,
        benchmark_batches=config.inference_batches,
    )
    return ModelSpeedReport(
        model_name="PredictiveCodingResNet50",
        epochs_ran=epochs_ran,
        final_metric_name="energy",
        final_metric_value=final_energy,
        test_accuracy=test_accuracy,
        train_seconds=train_seconds,
        train_samples_per_second=_safe_div(seen_samples, train_seconds),
        mean_train_step_ms=float(np.mean(step_times_ms)) if step_times_ms else 0.0,
        inference_latency_mean_ms=inference_metrics["latency_mean_ms"],
        inference_latency_p95_ms=inference_metrics["latency_p95_ms"],
        inference_samples_per_second=inference_metrics["samples_per_second"],
        total_parameters=model.parameter_count(),
        trainable_parameters=model.trainable_parameter_count(),
    )


def _benchmark_circadian(
    torch: Any,
    device: Any,
    loaders: Any,
    config: ResNet50BenchmarkConfig,
) -> ModelSpeedReport:
    model = CircadianPredictiveCodingResNet50Classifier(
        num_classes=loaders.num_classes,
        device=device,
        head_hidden_dim=config.circadian_head_hidden_dim,
        seed=config.seed + 23,
        freeze_backbone=True,
        min_hidden_dim=config.circadian_min_hidden_dim,
        max_hidden_dim=config.circadian_max_hidden_dim,
    )
    hidden_dim_start = model.head.hidden_dim
    sleep_splits = 0
    sleep_prunes = 0

    step_times_ms: list[float] = []
    seen_samples = 0
    epochs_ran = 0
    final_energy = 0.0

    train_timer_start = perf_counter()
    for epoch in range(1, config.epochs + 1):
        for images, labels in loaders.train_loader:
            images = images.to(device)
            labels = labels.to(device)

            sync_device(torch, device)
            step_start = perf_counter()
            energy = model.train_step(
                images=images,
                targets=labels,
                learning_rate=config.circadian_learning_rate,
                inference_steps=config.circadian_inference_steps,
                inference_learning_rate=config.circadian_inference_learning_rate,
            )
            sync_device(torch, device)

            step_times_ms.append((perf_counter() - step_start) * 1000.0)
            final_energy = float(energy)
            seen_samples += int(labels.shape[0])

        if config.circadian_sleep_interval > 0 and epoch % config.circadian_sleep_interval == 0:
            sleep_result = model.sleep_event()
            sleep_splits += len(sleep_result.split_indices)
            sleep_prunes += len(sleep_result.pruned_indices)

        epochs_ran = epoch
        if _should_stop_early(
            target_accuracy=config.target_accuracy,
            accuracy=_compute_pc_accuracy(torch, model, loaders.test_loader, device),
        ):
            break
    train_seconds = perf_counter() - train_timer_start

    test_accuracy = _compute_pc_accuracy(torch, model, loaders.test_loader, device)
    inference_metrics = _benchmark_inference(
        torch=torch,
        device=device,
        loader=loaders.test_loader,
        forward_logits=lambda x: model.predict_logits(x),
        warmup_batches=config.warmup_batches,
        benchmark_batches=config.inference_batches,
    )
    return ModelSpeedReport(
        model_name="CircadianPredictiveCodingResNet50",
        epochs_ran=epochs_ran,
        final_metric_name="energy",
        final_metric_value=final_energy,
        test_accuracy=test_accuracy,
        train_seconds=train_seconds,
        train_samples_per_second=_safe_div(seen_samples, train_seconds),
        mean_train_step_ms=float(np.mean(step_times_ms)) if step_times_ms else 0.0,
        inference_latency_mean_ms=inference_metrics["latency_mean_ms"],
        inference_latency_p95_ms=inference_metrics["latency_p95_ms"],
        inference_samples_per_second=inference_metrics["samples_per_second"],
        total_parameters=model.parameter_count(),
        trainable_parameters=model.trainable_parameter_count(),
        circadian_hidden_dim_start=hidden_dim_start,
        circadian_hidden_dim_end=model.head.hidden_dim,
        circadian_total_splits=sleep_splits,
        circadian_total_prunes=sleep_prunes,
    )


def _benchmark_inference(
    torch: Any,
    device: Any,
    loader: Any,
    forward_logits: Callable[[Any], Any],
    warmup_batches: int,
    benchmark_batches: int,
) -> dict[str, float]:
    if benchmark_batches <= 0:
        raise ValueError("benchmark_batches must be positive.")

    cached_batches = list(loader)
    if not cached_batches:
        raise ValueError("Inference loader is empty.")
    batch_index = 0

    def next_cached_batch() -> tuple[Any, Any]:
        nonlocal batch_index
        batch = cached_batches[batch_index % len(cached_batches)]
        batch_index += 1
        return batch

    latencies_ms: list[float] = []
    seen_samples = 0
    total_time_seconds = 0.0

    with torch.no_grad():
        for _ in range(max(warmup_batches, 0)):
            images, _ = next_cached_batch()
            images = images.to(device)
            _ = forward_logits(images)
            sync_device(torch, device)

        for _ in range(benchmark_batches):
            images, _ = next_cached_batch()
            images = images.to(device)

            sync_device(torch, device)
            start = perf_counter()
            _ = forward_logits(images)
            sync_device(torch, device)
            elapsed = perf_counter() - start

            latencies_ms.append(elapsed * 1000.0)
            total_time_seconds += elapsed
            seen_samples += int(images.shape[0])

    return {
        "latency_mean_ms": float(np.mean(latencies_ms)),
        "latency_p95_ms": float(np.percentile(latencies_ms, 95)),
        "samples_per_second": _safe_div(seen_samples, total_time_seconds),
    }


def _compute_backprop_accuracy(torch: Any, model: Any, loader: Any, device: Any) -> float:
    correct = 0
    total = 0
    model.backbone.eval()
    model.classifier.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model.forward_logits(images)
            predictions = torch.argmax(logits, dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.shape[0])
    return _safe_div(correct, total)


def _compute_pc_accuracy(torch: Any, model: Any, loader: Any, device: Any) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model.predict_logits(images)
            predictions = torch.argmax(logits, dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.shape[0])
    return _safe_div(correct, total)


def _resolve_device(torch: Any, requested_device: str) -> Any:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def _set_seed(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _should_stop_early(target_accuracy: float | None, accuracy: float) -> bool:
    if target_accuracy is None:
        return False
    return accuracy >= target_accuracy


def _safe_div(numerator: float | int, denominator: float | int) -> float:
    if float(denominator) == 0.0:
        return 0.0
    return float(numerator) / float(denominator)
