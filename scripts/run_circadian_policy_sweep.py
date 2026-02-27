"""Focused hard-mode sweep for circadian policy knobs on ResNet-50 benchmark."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.app.resnet50_benchmark import (  # noqa: E402
    ResNet50BenchmarkConfig,
    _benchmark_circadian,
    _resolve_device,
    _set_seed,
)
from src.infra.vision_datasets import (  # noqa: E402
    SyntheticVisionDatasetConfig,
    build_synthetic_vision_dataloaders,
)
from src.shared.torch_runtime import require_torch  # noqa: E402


def main() -> None:
    torch = require_torch()
    base = ResNet50BenchmarkConfig(
        train_samples=2500,
        test_samples=700,
        num_classes=10,
        image_size=96,
        batch_size=64,
        dataset_difficulty="hard",
        dataset_noise_std=0.08,
        epochs=14,
        seed=7,
        device="cuda",
        target_accuracy=None,
        inference_batches=14,
        warmup_batches=4,
        backprop_freeze_backbone=True,
        backbone_weights="imagenet",
        circadian_head_hidden_dim=384,
        circadian_learning_rate=0.03,
        circadian_inference_steps=12,
        circadian_inference_learning_rate=0.15,
        circadian_sleep_interval=1,
        circadian_min_hidden_dim=128,
        circadian_max_hidden_dim=1024,
        circadian_use_dual_chemical=True,
        circadian_use_adaptive_thresholds=True,
        circadian_adaptive_split_percentile=88.0,
        circadian_adaptive_prune_percentile=15.0,
        circadian_split_hysteresis_margin=0.02,
        circadian_prune_hysteresis_margin=0.02,
        circadian_split_cooldown_steps=2,
        circadian_prune_cooldown_steps=2,
        circadian_split_weight_norm_mix=0.30,
        circadian_prune_weight_norm_mix=0.30,
        circadian_split_importance_mix=0.20,
        circadian_prune_importance_mix=0.35,
        circadian_importance_ema_decay=0.95,
        circadian_split_noise_scale=0.01,
        circadian_max_split_per_sleep=2,
        circadian_max_prune_per_sleep=2,
    )
    _set_seed(torch, base.seed)
    device = _resolve_device(torch, base.device)
    loaders = build_synthetic_vision_dataloaders(
        SyntheticVisionDatasetConfig(
            train_samples=base.train_samples,
            test_samples=base.test_samples,
            num_classes=base.num_classes,
            image_size=base.image_size,
            batch_size=base.batch_size,
            noise_std=base.dataset_noise_std,
            difficulty=base.dataset_difficulty,
            seed=base.seed,
        )
    )

    candidates = build_candidates(base)
    trials: list[dict[str, Any]] = []
    for index, override in enumerate(candidates, start=1):
        config = replace(base, **override)
        report = _benchmark_circadian(torch=torch, device=device, loaders=loaders, config=config)
        row = {
            "trial": index,
            "params": override,
            "report": report_to_dict(report),
        }
        trials.append(row)
        print(
            f"circadian-policy {index}/{len(candidates)} "
            f"acc={report.test_accuracy:.4f} train_sps={report.train_samples_per_second:.1f} "
            f"infer_sps={report.inference_samples_per_second:.1f} "
            f"hidden={report.circadian_hidden_dim_start}->{report.circadian_hidden_dim_end} "
            f"split={report.circadian_total_splits} prune={report.circadian_total_prunes} "
            f"rollback={report.circadian_total_rollbacks}"
        )

    scores = compute_balanced_scores(trials)
    for trial, score in zip(trials, scores):
        trial["report"]["balanced_score"] = score

    output: dict[str, Any] = {
        "dataset": {
            "difficulty": base.dataset_difficulty,
            "noise_std": base.dataset_noise_std,
            "train_samples": base.train_samples,
            "test_samples": base.test_samples,
            "classes": base.num_classes,
            "image_size": base.image_size,
            "device": str(device),
            "epochs": base.epochs,
            "backbone_weights": base.backbone_weights,
        },
        "trials": trials,
        "top10_by_accuracy": sorted(
            trials,
            key=lambda row: (
                row["report"]["test_accuracy"],
                row["report"]["train_samples_per_second"],
            ),
            reverse=True,
        )[:10],
        "top10_by_balanced_score": sorted(
            trials,
            key=lambda row: row["report"]["balanced_score"],
            reverse=True,
        )[:10],
        "best_accuracy": best_from_trials(trials, key="test_accuracy"),
        "best_train_speed": best_from_trials(trials, key="train_samples_per_second"),
        "best_inference_speed": best_from_trials(trials, key="inference_samples_per_second"),
        "best_balanced": best_from_trials(trials, key="balanced_score"),
    }
    output_path = Path("benchmark_circadian_policy_sweep_results.json")
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    print("Best accuracy:", output["best_accuracy"]["report"]["test_accuracy"])
    print("Best balanced:", output["best_balanced"]["report"]["balanced_score"])


def build_candidates(base: ResNet50BenchmarkConfig) -> list[dict[str, Any]]:
    del base
    return [
        {},
        {
            "circadian_adaptive_split_percentile": 90.0,
            "circadian_adaptive_prune_percentile": 10.0,
        },
        {
            "circadian_adaptive_split_percentile": 92.0,
            "circadian_adaptive_prune_percentile": 8.0,
            "circadian_split_cooldown_steps": 3,
            "circadian_prune_cooldown_steps": 3,
        },
        {
            "circadian_adaptive_split_percentile": 85.0,
            "circadian_adaptive_prune_percentile": 20.0,
            "circadian_split_hysteresis_margin": 0.03,
            "circadian_prune_hysteresis_margin": 0.03,
        },
        {
            "circadian_split_importance_mix": 0.30,
            "circadian_prune_importance_mix": 0.50,
            "circadian_split_weight_norm_mix": 0.20,
            "circadian_prune_weight_norm_mix": 0.20,
        },
        {
            "circadian_split_importance_mix": 0.15,
            "circadian_prune_importance_mix": 0.45,
            "circadian_importance_ema_decay": 0.98,
        },
        {
            "circadian_use_dual_chemical": True,
            "circadian_dual_fast_mix": 0.60,
            "circadian_slow_chemical_decay": 0.9995,
        },
        {
            "circadian_use_dual_chemical": True,
            "circadian_dual_fast_mix": 0.80,
            "circadian_slow_buildup_scale": 0.15,
        },
        {
            "circadian_use_dual_chemical": False,
            "circadian_use_adaptive_thresholds": True,
            "circadian_adaptive_split_percentile": 88.0,
            "circadian_adaptive_prune_percentile": 15.0,
        },
        {
            "circadian_split_noise_scale": 0.0,
            "circadian_max_split_per_sleep": 1,
            "circadian_max_prune_per_sleep": 1,
        },
        {
            "circadian_sleep_interval": 2,
            "circadian_max_split_per_sleep": 1,
            "circadian_max_prune_per_sleep": 1,
        },
        {
            "circadian_head_hidden_dim": 512,
            "circadian_learning_rate": 0.02,
            "circadian_inference_steps": 14,
            "circadian_inference_learning_rate": 0.12,
            "circadian_adaptive_split_percentile": 90.0,
            "circadian_adaptive_prune_percentile": 10.0,
        },
        {
            "circadian_head_hidden_dim": 512,
            "circadian_learning_rate": 0.03,
            "circadian_inference_steps": 14,
            "circadian_inference_learning_rate": 0.15,
            "circadian_adaptive_split_percentile": 92.0,
            "circadian_adaptive_prune_percentile": 8.0,
        },
        {
            "circadian_homeostatic_downscale_factor": 0.995,
            "circadian_homeostasis_target_input_norm": 1.2,
            "circadian_homeostasis_target_output_norm": 1.0,
            "circadian_homeostasis_strength": 0.35,
        },
        {
            "circadian_homeostatic_downscale_factor": 0.998,
            "circadian_homeostasis_target_input_norm": 1.0,
            "circadian_homeostasis_target_output_norm": 0.8,
            "circadian_homeostasis_strength": 0.5,
        },
        {
            "circadian_homeostatic_downscale_factor": 1.0,
            "circadian_homeostasis_target_input_norm": 0.0,
            "circadian_homeostasis_target_output_norm": 0.0,
        },
        {
            "circadian_learning_rate": 0.025,
            "circadian_inference_steps": 16,
            "circadian_inference_learning_rate": 0.12,
            "circadian_split_cooldown_steps": 3,
            "circadian_prune_cooldown_steps": 3,
            "circadian_split_importance_mix": 0.25,
            "circadian_prune_importance_mix": 0.45,
        },
        {
            "circadian_learning_rate": 0.035,
            "circadian_inference_steps": 10,
            "circadian_inference_learning_rate": 0.18,
            "circadian_split_cooldown_steps": 1,
            "circadian_prune_cooldown_steps": 1,
            "circadian_split_hysteresis_margin": 0.01,
            "circadian_prune_hysteresis_margin": 0.01,
        },
    ]


def report_to_dict(report: Any) -> dict[str, Any]:
    return {
        "model_name": report.model_name,
        "epochs_ran": report.epochs_ran,
        "final_metric_name": report.final_metric_name,
        "final_metric_value": report.final_metric_value,
        "final_cross_entropy": report.final_cross_entropy,
        "final_energy": report.final_energy,
        "test_accuracy": report.test_accuracy,
        "train_seconds": report.train_seconds,
        "train_samples_per_second": report.train_samples_per_second,
        "mean_train_step_ms": report.mean_train_step_ms,
        "inference_latency_mean_ms": report.inference_latency_mean_ms,
        "inference_latency_p95_ms": report.inference_latency_p95_ms,
        "inference_samples_per_second": report.inference_samples_per_second,
        "total_parameters": report.total_parameters,
        "trainable_parameters": report.trainable_parameters,
        "circadian_hidden_dim_start": report.circadian_hidden_dim_start,
        "circadian_hidden_dim_end": report.circadian_hidden_dim_end,
        "circadian_total_splits": report.circadian_total_splits,
        "circadian_total_prunes": report.circadian_total_prunes,
        "circadian_total_rollbacks": report.circadian_total_rollbacks,
    }


def compute_balanced_scores(trials: list[dict[str, Any]]) -> list[float]:
    accuracies = [trial["report"]["test_accuracy"] for trial in trials]
    train_speeds = [trial["report"]["train_samples_per_second"] for trial in trials]
    inference_speeds = [trial["report"]["inference_samples_per_second"] for trial in trials]

    acc_norm = normalize_values(accuracies)
    train_norm = normalize_values(train_speeds)
    infer_norm = normalize_values(inference_speeds)
    return [0.60 * a + 0.25 * t + 0.15 * i for a, t, i in zip(acc_norm, train_norm, infer_norm)]


def normalize_values(values: list[float]) -> list[float]:
    min_value = min(values)
    max_value = max(values)
    if max_value - min_value < 1e-12:
        return [1.0 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]


def best_from_trials(trials: list[dict[str, Any]], key: str) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for trial in trials:
        report = trial["report"]
        if best is None or report[key] > best["report"][key]:
            best = trial
    if best is None:
        raise ValueError("No trials available.")
    return best


if __name__ == "__main__":
    main()
