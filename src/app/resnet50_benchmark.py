"""Application workflow for ResNet-50 speed and accuracy benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np

from src.core.resnet50_variants import (
    BackpropResNet50Classifier,
    CircadianHeadConfig,
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
    dataset_difficulty: str = "medium"
    dataset_noise_std: float = 0.06
    epochs: int = 8
    seed: int = 7
    device: str = "auto"
    target_accuracy: float | None = 0.99
    evaluation_batches: int = 0
    inference_batches: int = 50
    warmup_batches: int = 10

    backprop_learning_rate: float = 0.02
    backprop_momentum: float = 0.9
    backprop_freeze_backbone: bool = False
    backbone_weights: str = "none"

    predictive_head_hidden_dim: int = 256
    predictive_learning_rate: float = 0.03
    predictive_inference_steps: int = 10
    predictive_inference_learning_rate: float = 0.15

    circadian_head_hidden_dim: int = 256
    circadian_learning_rate: float = 0.03
    circadian_inference_steps: int = 10
    circadian_inference_learning_rate: float = 0.15
    circadian_sleep_interval: int = 2
    circadian_force_sleep: bool = True
    circadian_use_adaptive_sleep_trigger: bool = False
    circadian_min_sleep_steps: int = 40
    circadian_sleep_energy_window: int = 32
    circadian_sleep_plateau_delta: float = 1e-4
    circadian_sleep_chemical_variance_threshold: float = 0.02
    circadian_enable_sleep_rollback: bool = False
    circadian_sleep_rollback_tolerance: float = 0.01
    circadian_sleep_rollback_metric: str = "cross_entropy"
    circadian_sleep_rollback_eval_batches: int = 0
    circadian_min_hidden_dim: int = 96
    circadian_max_hidden_dim: int = 1024
    circadian_chemical_decay: float = 0.995
    circadian_chemical_buildup_rate: float = 0.02
    circadian_use_saturating_chemical: bool = True
    circadian_chemical_max_value: float = 2.5
    circadian_chemical_saturation_gain: float = 1.0
    circadian_use_dual_chemical: bool = True
    circadian_dual_fast_mix: float = 0.70
    circadian_slow_chemical_decay: float = 0.999
    circadian_slow_buildup_scale: float = 0.25
    circadian_plasticity_sensitivity: float = 0.7
    circadian_use_adaptive_plasticity_sensitivity: bool = True
    circadian_plasticity_sensitivity_min: float = 0.35
    circadian_plasticity_sensitivity_max: float = 1.20
    circadian_plasticity_importance_mix: float = 0.50
    circadian_min_plasticity: float = 0.2
    circadian_use_adaptive_thresholds: bool = True
    circadian_adaptive_split_percentile: float = 85.0
    circadian_adaptive_prune_percentile: float = 20.0
    circadian_sleep_warmup_steps: int = 0
    circadian_sleep_split_only_until_fraction: float = 0.50
    circadian_sleep_prune_only_after_fraction: float = 0.85
    circadian_sleep_max_change_fraction: float = 1.0
    circadian_sleep_min_change_count: int = 1
    circadian_prune_min_age_steps: int = 0
    circadian_split_threshold: float = 0.80
    circadian_prune_threshold: float = 0.08
    circadian_split_hysteresis_margin: float = 0.02
    circadian_prune_hysteresis_margin: float = 0.02
    circadian_split_cooldown_steps: int = 2
    circadian_prune_cooldown_steps: int = 2
    circadian_split_weight_norm_mix: float = 0.30
    circadian_prune_weight_norm_mix: float = 0.30
    circadian_split_importance_mix: float = 0.20
    circadian_prune_importance_mix: float = 0.35
    circadian_importance_ema_decay: float = 0.95
    circadian_max_split_per_sleep: int = 2
    circadian_max_prune_per_sleep: int = 2
    circadian_split_noise_scale: float = 0.01
    circadian_sleep_reset_factor: float = 0.45
    circadian_homeostatic_downscale_factor: float = 1.0
    circadian_homeostasis_target_input_norm: float = 0.0
    circadian_homeostasis_target_output_norm: float = 0.0
    circadian_homeostasis_strength: float = 0.50


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
    final_cross_entropy: float | None = None
    final_energy: float | None = None
    circadian_hidden_dim_start: int | None = None
    circadian_hidden_dim_end: int | None = None
    circadian_total_splits: int = 0
    circadian_total_prunes: int = 0
    circadian_total_rollbacks: int = 0


@dataclass(frozen=True)
class ResNet50BenchmarkResult:
    """Benchmark result across all model variants."""

    device: str
    config: ResNet50BenchmarkConfig
    reports: list[ModelSpeedReport]


def run_resnet50_benchmark(config: ResNet50BenchmarkConfig) -> ResNet50BenchmarkResult:
    """Benchmark all three model families on the same synthetic image task."""
    _validate_benchmark_config(config)
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
            noise_std=config.dataset_noise_std,
            difficulty=config.dataset_difficulty,
            seed=config.seed,
        )
    )

    reports = [
        _benchmark_backprop(torch=torch, device=device, loaders=loaders, config=config),
        _benchmark_predictive(torch=torch, device=device, loaders=loaders, config=config),
        _benchmark_circadian(torch=torch, device=device, loaders=loaders, config=config),
    ]
    _validate_report_models(reports)
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
            f"classes={result.config.num_classes}, image={result.config.image_size}x{result.config.image_size}, "
            f"difficulty={result.config.dataset_difficulty}, noise_std={result.config.dataset_noise_std:.3f}, "
            f"backbone_weights={result.config.backbone_weights}"
        ),
        "",
    ]
    backprop_report = _find_report_by_name(result.reports, "BackpropResNet50")
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
        if report.model_name != "BackpropResNet50":
            accuracy_delta = report.test_accuracy - backprop_report.test_accuracy
            speed_delta = report.train_samples_per_second - backprop_report.train_samples_per_second
            lines.append(
                (
                    "  vs backprop: "
                    f"acc_delta={accuracy_delta:+.3f}, "
                    f"train_samples_per_second_delta={speed_delta:+.1f}"
                )
            )
        if report.final_cross_entropy is not None and report.final_metric_name != "cross_entropy":
            lines.append(f"  cross_entropy={report.final_cross_entropy:.4f}")
        if report.final_energy is not None and report.final_metric_name != "energy":
            lines.append(f"  energy={report.final_energy:.4f}")
        if report.circadian_hidden_dim_start is not None and report.circadian_hidden_dim_end is not None:
            lines.append(
                (
                    "  circadian sleep: "
                    f"hidden={report.circadian_hidden_dim_start}->{report.circadian_hidden_dim_end}, "
                    f"splits={report.circadian_total_splits}, "
                    f"prunes={report.circadian_total_prunes}, "
                    f"rollbacks={report.circadian_total_rollbacks}"
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
        backbone_weights=config.backbone_weights,
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
    eval_batches = _resolve_eval_batches(config.evaluation_batches)

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
            seen_samples += int(labels.shape[0])

        epochs_ran = epoch
        eval_accuracy, _ = _compute_backprop_metrics(
            torch, model, loaders.test_loader, device, max_batches=eval_batches
        )
        if _should_stop_early(
            target_accuracy=config.target_accuracy,
            accuracy=eval_accuracy,
        ):
            break
    train_seconds = perf_counter() - train_timer_start

    model.backbone.eval()
    model.classifier.eval()
    test_accuracy, final_cross_entropy = _compute_backprop_metrics(
        torch, model, loaders.test_loader, device, max_batches=None
    )
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
        final_metric_name="cross_entropy",
        final_metric_value=final_cross_entropy,
        final_cross_entropy=final_cross_entropy,
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
        backbone_weights=config.backbone_weights,
    )

    step_times_ms: list[float] = []
    seen_samples = 0
    epochs_ran = 0
    final_energy = 0.0
    eval_batches = _resolve_eval_batches(config.evaluation_batches)

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
        eval_accuracy, _ = _compute_pc_metrics(
            torch, model, loaders.test_loader, device, max_batches=eval_batches
        )
        if _should_stop_early(
            target_accuracy=config.target_accuracy,
            accuracy=eval_accuracy,
        ):
            break
    train_seconds = perf_counter() - train_timer_start

    test_accuracy, final_cross_entropy = _compute_pc_metrics(
        torch, model, loaders.test_loader, device, max_batches=None
    )
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
        final_metric_name="cross_entropy",
        final_metric_value=final_cross_entropy,
        final_cross_entropy=final_cross_entropy,
        final_energy=final_energy,
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
    circadian_config = CircadianHeadConfig(
        chemical_decay=config.circadian_chemical_decay,
        chemical_buildup_rate=config.circadian_chemical_buildup_rate,
        use_saturating_chemical=config.circadian_use_saturating_chemical,
        chemical_max_value=config.circadian_chemical_max_value,
        chemical_saturation_gain=config.circadian_chemical_saturation_gain,
        use_dual_chemical=config.circadian_use_dual_chemical,
        dual_fast_mix=config.circadian_dual_fast_mix,
        slow_chemical_decay=config.circadian_slow_chemical_decay,
        slow_buildup_scale=config.circadian_slow_buildup_scale,
        plasticity_sensitivity=config.circadian_plasticity_sensitivity,
        use_adaptive_plasticity_sensitivity=config.circadian_use_adaptive_plasticity_sensitivity,
        plasticity_sensitivity_min=config.circadian_plasticity_sensitivity_min,
        plasticity_sensitivity_max=config.circadian_plasticity_sensitivity_max,
        plasticity_importance_mix=config.circadian_plasticity_importance_mix,
        min_plasticity=config.circadian_min_plasticity,
        use_adaptive_thresholds=config.circadian_use_adaptive_thresholds,
        adaptive_split_percentile=config.circadian_adaptive_split_percentile,
        adaptive_prune_percentile=config.circadian_adaptive_prune_percentile,
        sleep_warmup_steps=config.circadian_sleep_warmup_steps,
        sleep_split_only_until_fraction=config.circadian_sleep_split_only_until_fraction,
        sleep_prune_only_after_fraction=config.circadian_sleep_prune_only_after_fraction,
        sleep_max_change_fraction=config.circadian_sleep_max_change_fraction,
        sleep_min_change_count=config.circadian_sleep_min_change_count,
        prune_min_age_steps=config.circadian_prune_min_age_steps,
        split_threshold=config.circadian_split_threshold,
        prune_threshold=config.circadian_prune_threshold,
        split_hysteresis_margin=config.circadian_split_hysteresis_margin,
        prune_hysteresis_margin=config.circadian_prune_hysteresis_margin,
        split_cooldown_steps=config.circadian_split_cooldown_steps,
        prune_cooldown_steps=config.circadian_prune_cooldown_steps,
        split_weight_norm_mix=config.circadian_split_weight_norm_mix,
        prune_weight_norm_mix=config.circadian_prune_weight_norm_mix,
        split_importance_mix=config.circadian_split_importance_mix,
        prune_importance_mix=config.circadian_prune_importance_mix,
        importance_ema_decay=config.circadian_importance_ema_decay,
        max_split_per_sleep=config.circadian_max_split_per_sleep,
        max_prune_per_sleep=config.circadian_max_prune_per_sleep,
        split_noise_scale=config.circadian_split_noise_scale,
        sleep_reset_factor=config.circadian_sleep_reset_factor,
        homeostatic_downscale_factor=config.circadian_homeostatic_downscale_factor,
        homeostasis_target_input_norm=config.circadian_homeostasis_target_input_norm,
        homeostasis_target_output_norm=config.circadian_homeostasis_target_output_norm,
        homeostasis_strength=config.circadian_homeostasis_strength,
        use_adaptive_sleep_trigger=config.circadian_use_adaptive_sleep_trigger,
        min_sleep_steps=config.circadian_min_sleep_steps,
        sleep_energy_window=config.circadian_sleep_energy_window,
        sleep_plateau_delta=config.circadian_sleep_plateau_delta,
        sleep_chemical_variance_threshold=config.circadian_sleep_chemical_variance_threshold,
    )
    model = CircadianPredictiveCodingResNet50Classifier(
        num_classes=loaders.num_classes,
        device=device,
        head_hidden_dim=config.circadian_head_hidden_dim,
        seed=config.seed + 23,
        freeze_backbone=True,
        backbone_weights=config.backbone_weights,
        circadian_config=circadian_config,
        min_hidden_dim=config.circadian_min_hidden_dim,
        max_hidden_dim=config.circadian_max_hidden_dim,
    )
    hidden_dim_start = model.head.hidden_dim
    sleep_splits = 0
    sleep_prunes = 0
    sleep_rollbacks = 0

    step_times_ms: list[float] = []
    seen_samples = 0
    epochs_ran = 0
    final_energy = 0.0
    eval_batches = _resolve_eval_batches(config.evaluation_batches)
    rollback_eval_batches = _resolve_eval_batches(
        config.circadian_sleep_rollback_eval_batches
    )
    if rollback_eval_batches is None:
        rollback_eval_batches = eval_batches

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

        interval_triggered = (
            config.circadian_sleep_interval > 0 and epoch % config.circadian_sleep_interval == 0
        )
        adaptive_triggered = (
            config.circadian_use_adaptive_sleep_trigger and model.should_trigger_sleep()
        )
        if interval_triggered or adaptive_triggered:
            maybe_snapshot = None
            pre_sleep_accuracy = 0.0
            pre_sleep_cross_entropy = 0.0
            if config.circadian_enable_sleep_rollback:
                maybe_snapshot = model.snapshot_state()
                pre_sleep_accuracy, pre_sleep_cross_entropy = _compute_pc_metrics(
                    torch,
                    model,
                    loaders.test_loader,
                    device,
                    max_batches=rollback_eval_batches,
                )

            force_sleep = config.circadian_force_sleep and interval_triggered
            sleep_result = model.sleep_event(
                current_step=epoch,
                total_steps=config.epochs,
                force_sleep=force_sleep,
            )

            if config.circadian_enable_sleep_rollback and maybe_snapshot is not None:
                post_sleep_accuracy, post_sleep_cross_entropy = _compute_pc_metrics(
                    torch,
                    model,
                    loaders.test_loader,
                    device,
                    max_batches=rollback_eval_batches,
                )
                rollback_delta = _compute_rollback_delta(
                    metric_name=config.circadian_sleep_rollback_metric,
                    pre_accuracy=pre_sleep_accuracy,
                    post_accuracy=post_sleep_accuracy,
                    pre_cross_entropy=pre_sleep_cross_entropy,
                    post_cross_entropy=post_sleep_cross_entropy,
                )
                if rollback_delta > config.circadian_sleep_rollback_tolerance:
                    model.restore_state(maybe_snapshot)
                    sleep_rollbacks += 1
                    sleep_result = type(sleep_result)(
                        old_hidden_dim=model.head.hidden_dim,
                        new_hidden_dim=model.head.hidden_dim,
                        split_indices=(),
                        pruned_indices=(),
                    )

            sleep_splits += len(sleep_result.split_indices)
            sleep_prunes += len(sleep_result.pruned_indices)

        epochs_ran = epoch
        eval_accuracy, _ = _compute_pc_metrics(
            torch, model, loaders.test_loader, device, max_batches=eval_batches
        )
        if _should_stop_early(
            target_accuracy=config.target_accuracy,
            accuracy=eval_accuracy,
        ):
            break
    train_seconds = perf_counter() - train_timer_start

    test_accuracy, final_cross_entropy = _compute_pc_metrics(
        torch, model, loaders.test_loader, device, max_batches=None
    )
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
        final_metric_name="cross_entropy",
        final_metric_value=final_cross_entropy,
        final_cross_entropy=final_cross_entropy,
        final_energy=final_energy,
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
        circadian_total_rollbacks=sleep_rollbacks,
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
    accuracy, _ = _compute_backprop_metrics(
        torch, model, loader, device=device, max_batches=None
    )
    return accuracy


def _compute_backprop_metrics(
    torch: Any,
    model: Any,
    loader: Any,
    device: Any,
    max_batches: int | None,
) -> tuple[float, float]:
    correct = 0
    total = 0
    total_loss = 0.0
    model.backbone.eval()
    model.classifier.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            logits = model.forward_logits(images)
            batch_loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.shape[0])
            total_loss += float(batch_loss.item())
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    return _safe_div(correct, total), _safe_div(total_loss, total)


def _compute_pc_accuracy(torch: Any, model: Any, loader: Any, device: Any) -> float:
    accuracy, _ = _compute_pc_metrics(torch, model, loader, device=device, max_batches=None)
    return accuracy


def _compute_pc_metrics(
    torch: Any,
    model: Any,
    loader: Any,
    device: Any,
    max_batches: int | None,
) -> tuple[float, float]:
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            logits = model.predict_logits(images)
            batch_loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.shape[0])
            total_loss += float(batch_loss.item())
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    return _safe_div(correct, total), _safe_div(total_loss, total)


def _resolve_eval_batches(configured_batches: int) -> int | None:
    if configured_batches <= 0:
        return None
    return configured_batches


def _compute_rollback_delta(
    metric_name: str,
    pre_accuracy: float,
    post_accuracy: float,
    pre_cross_entropy: float,
    post_cross_entropy: float,
) -> float:
    if metric_name == "accuracy":
        return pre_accuracy - post_accuracy
    if metric_name == "cross_entropy":
        return post_cross_entropy - pre_cross_entropy
    raise ValueError(
        "circadian_sleep_rollback_metric must be one of: accuracy, cross_entropy."
    )


def _resolve_device(torch: Any, requested_device: str) -> Any:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def _find_report_by_name(reports: list[ModelSpeedReport], model_name: str) -> ModelSpeedReport:
    for report in reports:
        if report.model_name == model_name:
            return report
    raise ValueError(f"Missing report for model {model_name}.")


def _validate_report_models(reports: list[ModelSpeedReport]) -> None:
    expected = {
        "BackpropResNet50",
        "PredictiveCodingResNet50",
        "CircadianPredictiveCodingResNet50",
    }
    observed = {report.model_name for report in reports}
    if observed != expected:
        raise RuntimeError(
            "Benchmark must compare traditional backprop, traditional predictive coding, "
            f"and circadian predictive coding. Observed={sorted(observed)}"
        )


def _validate_benchmark_config(config: ResNet50BenchmarkConfig) -> None:
    if config.dataset_difficulty not in {"easy", "medium", "hard"}:
        raise ValueError("dataset_difficulty must be one of: easy, medium, hard.")
    if config.dataset_noise_std < 0.0:
        raise ValueError("dataset_noise_std must be non-negative.")
    if config.evaluation_batches < 0:
        raise ValueError("evaluation_batches must be non-negative.")
    if config.circadian_sleep_interval < 0:
        raise ValueError("circadian_sleep_interval must be non-negative.")
    if config.backbone_weights not in {"none", "imagenet"}:
        raise ValueError("backbone_weights must be one of: none, imagenet.")
    if config.circadian_chemical_max_value <= 0.0:
        raise ValueError("circadian_chemical_max_value must be positive.")
    if config.circadian_chemical_saturation_gain <= 0.0:
        raise ValueError("circadian_chemical_saturation_gain must be positive.")
    if config.circadian_plasticity_sensitivity_min <= 0.0:
        raise ValueError("circadian_plasticity_sensitivity_min must be positive.")
    if config.circadian_plasticity_sensitivity_max < config.circadian_plasticity_sensitivity_min:
        raise ValueError(
            "circadian_plasticity_sensitivity_max must be >= circadian_plasticity_sensitivity_min."
        )
    if not (0.0 <= config.circadian_plasticity_importance_mix <= 1.0):
        raise ValueError("circadian_plasticity_importance_mix must be between 0 and 1.")
    if config.circadian_sleep_warmup_steps < 0:
        raise ValueError("circadian_sleep_warmup_steps must be non-negative.")
    if not (0.0 <= config.circadian_sleep_split_only_until_fraction <= 1.0):
        raise ValueError("circadian_sleep_split_only_until_fraction must be between 0 and 1.")
    if not (0.0 <= config.circadian_sleep_prune_only_after_fraction <= 1.0):
        raise ValueError("circadian_sleep_prune_only_after_fraction must be between 0 and 1.")
    if (
        config.circadian_sleep_split_only_until_fraction
        > config.circadian_sleep_prune_only_after_fraction
    ):
        raise ValueError(
            "circadian_sleep_split_only_until_fraction must be <= "
            "circadian_sleep_prune_only_after_fraction."
        )
    if not (0.0 <= config.circadian_sleep_max_change_fraction <= 1.0):
        raise ValueError("circadian_sleep_max_change_fraction must be between 0 and 1.")
    if config.circadian_sleep_min_change_count < 0:
        raise ValueError("circadian_sleep_min_change_count must be non-negative.")
    if config.circadian_prune_min_age_steps < 0:
        raise ValueError("circadian_prune_min_age_steps must be non-negative.")
    if config.circadian_min_sleep_steps < 0:
        raise ValueError("circadian_min_sleep_steps must be non-negative.")
    if config.circadian_sleep_energy_window < 2:
        raise ValueError("circadian_sleep_energy_window must be at least 2.")
    if config.circadian_sleep_plateau_delta < 0.0:
        raise ValueError("circadian_sleep_plateau_delta must be non-negative.")
    if config.circadian_sleep_chemical_variance_threshold < 0.0:
        raise ValueError("circadian_sleep_chemical_variance_threshold must be non-negative.")
    if config.circadian_sleep_rollback_tolerance < 0.0:
        raise ValueError("circadian_sleep_rollback_tolerance must be non-negative.")
    if config.circadian_sleep_rollback_eval_batches < 0:
        raise ValueError("circadian_sleep_rollback_eval_batches must be non-negative.")
    if config.circadian_sleep_rollback_metric not in {"accuracy", "cross_entropy"}:
        raise ValueError(
            "circadian_sleep_rollback_metric must be one of: accuracy, cross_entropy."
        )


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
