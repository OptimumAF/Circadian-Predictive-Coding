"""Run a Pareto sweep on the hardest benchmark setting and export top-10 rankings."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.app.resnet50_benchmark import (  # noqa: E402
    ResNet50BenchmarkConfig,
    _benchmark_backprop,
    _benchmark_circadian,
    _benchmark_predictive,
    _resolve_device,
    _set_seed,
)
from src.infra.vision_datasets import (  # noqa: E402
    SyntheticVisionDatasetConfig,
    build_synthetic_vision_dataloaders,
)
from src.shared.torch_runtime import require_torch  # noqa: E402

MULTI_SEEDS: tuple[int, ...] = (7, 13, 29)


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
        epochs=20,
        seed=7,
        device="cuda",
        target_accuracy=None,
        evaluation_batches=2,
        inference_batches=14,
        warmup_batches=4,
        backprop_freeze_backbone=True,
        backbone_weights="imagenet",
        predictive_head_hidden_dim=384,
        circadian_head_hidden_dim=384,
        circadian_learning_rate=0.025,
        circadian_inference_steps=14,
        circadian_inference_learning_rate=0.12,
        circadian_sleep_interval=2,
        circadian_force_sleep=False,
        circadian_use_adaptive_sleep_trigger=True,
        circadian_min_sleep_steps=60,
        circadian_sleep_energy_window=48,
        circadian_sleep_plateau_delta=5e-5,
        circadian_enable_sleep_rollback=True,
        circadian_sleep_rollback_tolerance=0.002,
        circadian_sleep_rollback_metric="cross_entropy",
        circadian_sleep_rollback_eval_batches=2,
        circadian_min_hidden_dim=384,
        circadian_max_hidden_dim=640,
        circadian_use_saturating_chemical=True,
        circadian_chemical_max_value=2.5,
        circadian_chemical_saturation_gain=1.0,
        circadian_use_dual_chemical=True,
        circadian_use_adaptive_thresholds=True,
        circadian_use_adaptive_plasticity_sensitivity=True,
        circadian_plasticity_sensitivity=0.45,
        circadian_plasticity_sensitivity_min=0.25,
        circadian_plasticity_sensitivity_max=0.55,
        circadian_plasticity_importance_mix=0.50,
        circadian_min_plasticity=0.5,
        circadian_sleep_warmup_steps=6,
        circadian_sleep_split_only_until_fraction=0.75,
        circadian_sleep_prune_only_after_fraction=0.95,
        circadian_sleep_max_change_fraction=0.01,
        circadian_sleep_min_change_count=1,
        circadian_prune_min_age_steps=120,
        circadian_adaptive_split_percentile=92.0,
        circadian_adaptive_prune_percentile=8.0,
        circadian_split_cooldown_steps=3,
        circadian_prune_cooldown_steps=3,
        circadian_max_split_per_sleep=1,
        circadian_max_prune_per_sleep=1,
    )
    _set_seed(torch, base.seed)
    device = _resolve_device(torch, base.device)
    seeds = MULTI_SEEDS

    backprop_trials = run_backprop_sweep(base, torch, device, seeds)
    predictive_trials = run_predictive_sweep(base, torch, device, seeds)
    circadian_trials = run_circadian_sweep(base, torch, device, seeds)

    output = {
        "dataset": {
            "difficulty": base.dataset_difficulty,
            "noise_std": base.dataset_noise_std,
            "train_samples": base.train_samples,
            "test_samples": base.test_samples,
            "classes": base.num_classes,
            "image_size": base.image_size,
            "device": str(device),
            "epochs": base.epochs,
            "seeds": list(seeds),
        },
        "backprop": summarize_model_trials(backprop_trials),
        "predictive": summarize_model_trials(predictive_trials),
        "circadian": summarize_model_trials(circadian_trials),
    }
    all_trial_reports = collect_all_trial_reports(output)
    output["global_best_accuracy"] = best_from_all_trials(
        all_trial_reports, key="test_accuracy"
    )
    output["global_best_train_speed"] = best_from_all_trials(
        all_trial_reports, key="train_samples_per_second"
    )
    output["global_best_inference_speed"] = best_from_all_trials(
        all_trial_reports, key="inference_samples_per_second"
    )
    output["global_best_efficiency"] = best_from_all_trials(
        all_trial_reports, key="balanced_score"
    )

    output_path = Path("benchmark_pareto_hard_results.json")
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    print("Best by accuracy:", output["global_best_accuracy"]["model_name"])
    print("Best by train speed:", output["global_best_train_speed"]["model_name"])
    print("Best by inference speed:", output["global_best_inference_speed"]["model_name"])
    print("Best by balanced score:", output["global_best_efficiency"]["model_name"])


def run_backprop_sweep(
    base: ResNet50BenchmarkConfig,
    torch_module: Any,
    device: Any,
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    candidate_params = [
        (0.003, 0.9),
        (0.005, 0.9),
        (0.01, 0.9),
        (0.02, 0.9),
        (0.03, 0.9),
        (0.05, 0.9),
        (0.01, 0.85),
        (0.01, 0.95),
        (0.02, 0.85),
        (0.02, 0.95),
    ]
    for index, (lr, momentum) in enumerate(candidate_params, start=1):
        override: dict[str, Any] = {"backprop_learning_rate": lr, "backprop_momentum": momentum}
        trial = _run_multiseed_trial(
            base=base,
            override=override,
            seeds=seeds,
            torch_module=torch_module,
            device=device,
            benchmark_fn=_benchmark_backprop,
        )
        candidates.append({"trial": index, "params": override, **trial})
        report = trial["report"]
        print(
            f"backprop {index}/{len(candidate_params)} "
            f"acc={report['test_accuracy']:.4f}+/-{report['test_accuracy_std']:.4f} "
            f"train_sps={report['train_samples_per_second']:.1f}+/-{report['train_samples_per_second_std']:.1f}"
        )
    return candidates


def run_predictive_sweep(
    base: ResNet50BenchmarkConfig,
    torch_module: Any,
    device: Any,
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    candidate_params = [
        (256, 0.008, 10, 0.09),
        (256, 0.01, 10, 0.10),
        (256, 0.015, 12, 0.12),
        (256, 0.02, 12, 0.12),
        (384, 0.008, 10, 0.09),
        (384, 0.01, 12, 0.10),
        (384, 0.015, 12, 0.12),
        (384, 0.02, 14, 0.12),
        (512, 0.008, 10, 0.09),
        (512, 0.01, 12, 0.10),
        (512, 0.015, 12, 0.12),
        (512, 0.02, 14, 0.12),
    ]
    for index, (hidden, lr, steps, inf_lr) in enumerate(candidate_params, start=1):
        override: dict[str, Any] = {
            "predictive_head_hidden_dim": hidden,
            "predictive_learning_rate": lr,
            "predictive_inference_steps": steps,
            "predictive_inference_learning_rate": inf_lr,
        }
        trial = _run_multiseed_trial(
            base=base,
            override=override,
            seeds=seeds,
            torch_module=torch_module,
            device=device,
            benchmark_fn=_benchmark_predictive,
        )
        candidates.append({"trial": index, "params": override, **trial})
        report = trial["report"]
        print(
            f"predictive {index}/{len(candidate_params)} "
            f"acc={report['test_accuracy']:.4f}+/-{report['test_accuracy_std']:.4f} "
            f"train_sps={report['train_samples_per_second']:.1f}+/-{report['train_samples_per_second_std']:.1f}"
        )
    return candidates


def run_circadian_sweep(
    base: ResNet50BenchmarkConfig,
    torch_module: Any,
    device: Any,
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    candidate_params: list[dict[str, Any]] = [
        {
            "circadian_head_hidden_dim": 384,
            "circadian_learning_rate": 0.03,
            "circadian_inference_steps": 12,
            "circadian_inference_learning_rate": 0.15,
            "circadian_sleep_interval": 1,
            "circadian_adaptive_split_percentile": 92.0,
            "circadian_adaptive_prune_percentile": 8.0,
            "circadian_split_cooldown_steps": 3,
            "circadian_prune_cooldown_steps": 3,
            "circadian_max_split_per_sleep": 2,
            "circadian_max_prune_per_sleep": 2,
            "circadian_min_hidden_dim": 128,
            "circadian_max_hidden_dim": 1024,
        },
        {
            "circadian_head_hidden_dim": 384,
            "circadian_learning_rate": 0.03,
            "circadian_inference_steps": 12,
            "circadian_inference_learning_rate": 0.15,
            "circadian_sleep_interval": 1,
            "circadian_adaptive_split_percentile": 90.0,
            "circadian_adaptive_prune_percentile": 10.0,
            "circadian_split_cooldown_steps": 3,
            "circadian_prune_cooldown_steps": 3,
            "circadian_max_split_per_sleep": 2,
            "circadian_max_prune_per_sleep": 2,
            "circadian_min_hidden_dim": 128,
            "circadian_max_hidden_dim": 1024,
        },
        {
            "circadian_head_hidden_dim": 384,
            "circadian_learning_rate": 0.03,
            "circadian_inference_steps": 12,
            "circadian_inference_learning_rate": 0.15,
            "circadian_sleep_interval": 1,
            "circadian_adaptive_split_percentile": 88.0,
            "circadian_adaptive_prune_percentile": 15.0,
            "circadian_split_cooldown_steps": 2,
            "circadian_prune_cooldown_steps": 2,
            "circadian_max_split_per_sleep": 2,
            "circadian_max_prune_per_sleep": 2,
            "circadian_min_hidden_dim": 128,
            "circadian_max_hidden_dim": 1024,
        },
        {
            "circadian_head_hidden_dim": 384,
            "circadian_learning_rate": 0.025,
            "circadian_inference_steps": 14,
            "circadian_inference_learning_rate": 0.12,
            "circadian_sleep_interval": 1,
            "circadian_adaptive_split_percentile": 92.0,
            "circadian_adaptive_prune_percentile": 8.0,
            "circadian_split_cooldown_steps": 3,
            "circadian_prune_cooldown_steps": 3,
            "circadian_max_split_per_sleep": 2,
            "circadian_max_prune_per_sleep": 2,
            "circadian_min_hidden_dim": 128,
            "circadian_max_hidden_dim": 1024,
        },
        {
            "circadian_head_hidden_dim": 384,
            "circadian_learning_rate": 0.035,
            "circadian_inference_steps": 10,
            "circadian_inference_learning_rate": 0.18,
            "circadian_sleep_interval": 1,
            "circadian_adaptive_split_percentile": 92.0,
            "circadian_adaptive_prune_percentile": 8.0,
            "circadian_split_cooldown_steps": 3,
            "circadian_prune_cooldown_steps": 3,
            "circadian_max_split_per_sleep": 2,
            "circadian_max_prune_per_sleep": 2,
            "circadian_min_hidden_dim": 128,
            "circadian_max_hidden_dim": 1024,
        },
        {
            "circadian_head_hidden_dim": 256,
            "circadian_learning_rate": 0.03,
            "circadian_inference_steps": 12,
            "circadian_inference_learning_rate": 0.15,
            "circadian_sleep_interval": 1,
            "circadian_adaptive_split_percentile": 92.0,
            "circadian_adaptive_prune_percentile": 8.0,
            "circadian_split_cooldown_steps": 3,
            "circadian_prune_cooldown_steps": 3,
            "circadian_max_split_per_sleep": 2,
            "circadian_max_prune_per_sleep": 2,
            "circadian_min_hidden_dim": 96,
            "circadian_max_hidden_dim": 768,
        },
        {
            "circadian_head_hidden_dim": 512,
            "circadian_learning_rate": 0.02,
            "circadian_inference_steps": 14,
            "circadian_inference_learning_rate": 0.12,
            "circadian_sleep_interval": 1,
            "circadian_adaptive_split_percentile": 92.0,
            "circadian_adaptive_prune_percentile": 8.0,
            "circadian_split_cooldown_steps": 3,
            "circadian_prune_cooldown_steps": 3,
            "circadian_max_split_per_sleep": 2,
            "circadian_max_prune_per_sleep": 2,
            "circadian_min_hidden_dim": 128,
            "circadian_max_hidden_dim": 1024,
        },
        {
            "circadian_head_hidden_dim": 512,
            "circadian_learning_rate": 0.03,
            "circadian_inference_steps": 14,
            "circadian_inference_learning_rate": 0.15,
            "circadian_sleep_interval": 1,
            "circadian_adaptive_split_percentile": 92.0,
            "circadian_adaptive_prune_percentile": 8.0,
            "circadian_split_cooldown_steps": 3,
            "circadian_prune_cooldown_steps": 3,
            "circadian_max_split_per_sleep": 2,
            "circadian_max_prune_per_sleep": 2,
            "circadian_min_hidden_dim": 128,
            "circadian_max_hidden_dim": 1024,
        },
        {
            "circadian_head_hidden_dim": 384,
            "circadian_learning_rate": 0.03,
            "circadian_inference_steps": 12,
            "circadian_inference_learning_rate": 0.15,
            "circadian_sleep_interval": 1,
            "circadian_adaptive_split_percentile": 92.0,
            "circadian_adaptive_prune_percentile": 8.0,
            "circadian_split_cooldown_steps": 3,
            "circadian_prune_cooldown_steps": 3,
            "circadian_use_dual_chemical": False,
            "circadian_max_split_per_sleep": 2,
            "circadian_max_prune_per_sleep": 2,
            "circadian_min_hidden_dim": 128,
            "circadian_max_hidden_dim": 1024,
        },
        {
            "circadian_head_hidden_dim": 384,
            "circadian_learning_rate": 0.03,
            "circadian_inference_steps": 12,
            "circadian_inference_learning_rate": 0.15,
            "circadian_sleep_interval": 1,
            "circadian_adaptive_split_percentile": 92.0,
            "circadian_adaptive_prune_percentile": 8.0,
            "circadian_split_cooldown_steps": 3,
            "circadian_prune_cooldown_steps": 3,
            "circadian_split_importance_mix": 0.30,
            "circadian_prune_importance_mix": 0.50,
            "circadian_split_weight_norm_mix": 0.20,
            "circadian_prune_weight_norm_mix": 0.20,
            "circadian_max_split_per_sleep": 2,
            "circadian_max_prune_per_sleep": 2,
            "circadian_min_hidden_dim": 128,
            "circadian_max_hidden_dim": 1024,
        },
        {
            "circadian_head_hidden_dim": 384,
            "circadian_learning_rate": 0.03,
            "circadian_inference_steps": 12,
            "circadian_inference_learning_rate": 0.15,
            "circadian_sleep_interval": 1,
            "circadian_adaptive_split_percentile": 92.0,
            "circadian_adaptive_prune_percentile": 8.0,
            "circadian_split_cooldown_steps": 3,
            "circadian_prune_cooldown_steps": 3,
            "circadian_homeostatic_downscale_factor": 0.995,
            "circadian_homeostasis_target_input_norm": 1.2,
            "circadian_homeostasis_target_output_norm": 1.0,
            "circadian_homeostasis_strength": 0.35,
            "circadian_max_split_per_sleep": 2,
            "circadian_max_prune_per_sleep": 2,
            "circadian_min_hidden_dim": 128,
            "circadian_max_hidden_dim": 1024,
        },
        {
            "circadian_head_hidden_dim": 384,
            "circadian_learning_rate": 0.03,
            "circadian_inference_steps": 12,
            "circadian_inference_learning_rate": 0.15,
            "circadian_sleep_interval": 1,
            "circadian_adaptive_split_percentile": 92.0,
            "circadian_adaptive_prune_percentile": 8.0,
            "circadian_split_cooldown_steps": 3,
            "circadian_prune_cooldown_steps": 3,
            "circadian_max_split_per_sleep": 1,
            "circadian_max_prune_per_sleep": 1,
            "circadian_split_noise_scale": 0.0,
            "circadian_min_hidden_dim": 128,
            "circadian_max_hidden_dim": 1024,
        },
    ]
    for index, override in enumerate(candidate_params, start=1):
        trial = _run_multiseed_trial(
            base=base,
            override=override,
            seeds=seeds,
            torch_module=torch_module,
            device=device,
            benchmark_fn=_benchmark_circadian,
        )
        candidates.append({"trial": index, "params": override, **trial})
        report = trial["report"]
        print(
            f"circadian {index}/{len(candidate_params)} "
            f"acc={report['test_accuracy']:.4f}+/-{report['test_accuracy_std']:.4f} "
            f"train_sps={report['train_samples_per_second']:.1f}+/-{report['train_samples_per_second_std']:.1f} "
            f"hidden={report['circadian_hidden_dim_start']:.1f}->{report['circadian_hidden_dim_end']:.1f} "
            f"split={report['circadian_total_splits']:.2f} prune={report['circadian_total_prunes']:.2f} "
            f"rollback={report['circadian_total_rollbacks']:.2f}"
        )
    return candidates


def _run_multiseed_trial(
    base: ResNet50BenchmarkConfig,
    override: dict[str, Any],
    seeds: tuple[int, ...],
    torch_module: Any,
    device: Any,
    benchmark_fn: Any,
) -> dict[str, Any]:
    if len(seeds) == 0:
        raise ValueError("seeds must be non-empty.")

    reports: list[dict[str, Any]] = []
    for seed in seeds:
        config = replace(base, seed=seed, **override)
        _set_seed(torch_module, seed)
        loaders = _build_loaders_for_config(config)
        report = benchmark_fn(torch=torch_module, device=device, loaders=loaders, config=config)
        reports.append(report_to_dict(report))

    aggregate = _aggregate_reports(reports)
    return {"report": aggregate, "seed_reports": reports}


def _build_loaders_for_config(config: ResNet50BenchmarkConfig) -> Any:
    return build_synthetic_vision_dataloaders(
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


def _aggregate_reports(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    if len(seed_reports) == 0:
        raise ValueError("seed_reports must be non-empty.")

    model_name = seed_reports[0]["model_name"]
    final_metric_name = seed_reports[0]["final_metric_name"]
    numeric_keys = [
        "epochs_ran",
        "final_metric_value",
        "test_accuracy",
        "train_seconds",
        "train_samples_per_second",
        "mean_train_step_ms",
        "inference_latency_mean_ms",
        "inference_latency_p95_ms",
        "inference_samples_per_second",
        "total_parameters",
        "trainable_parameters",
        "accuracy_per_train_second",
        "accuracy_per_million_trainable_params",
    ]
    optional_numeric_keys = [
        "final_cross_entropy",
        "final_energy",
        "circadian_hidden_dim_start",
        "circadian_hidden_dim_end",
        "circadian_total_splits",
        "circadian_total_prunes",
        "circadian_total_rollbacks",
    ]

    aggregate: dict[str, Any] = {
        "model_name": model_name,
        "final_metric_name": final_metric_name,
        "seed_count": len(seed_reports),
    }

    for key in numeric_keys:
        values = [float(report[key]) for report in seed_reports]
        aggregate[key] = float(np.mean(values))
        aggregate[f"{key}_std"] = float(np.std(values))

    for key in optional_numeric_keys:
        raw_values = [report[key] for report in seed_reports if report[key] is not None]
        if len(raw_values) == 0:
            aggregate[key] = None
            aggregate[f"{key}_std"] = None
        else:
            values = [float(value) for value in raw_values]
            aggregate[key] = float(np.mean(values))
            aggregate[f"{key}_std"] = float(np.std(values))

    return aggregate


def report_to_dict(report: Any) -> dict[str, Any]:
    row = {
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
    row["accuracy_per_train_second"] = (
        0.0 if row["train_seconds"] == 0.0 else row["test_accuracy"] / row["train_seconds"]
    )
    row["accuracy_per_million_trainable_params"] = (
        0.0
        if row["trainable_parameters"] == 0
        else row["test_accuracy"] / (row["trainable_parameters"] / 1_000_000.0)
    )
    return row


def summarize_model_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    scores = compute_balanced_scores(trials)
    for trial, score in zip(trials, scores):
        trial["report"]["balanced_score"] = score

    top10_by_accuracy = sorted(
        trials, key=lambda row: (row["report"]["test_accuracy"], row["report"]["train_samples_per_second"]), reverse=True
    )[:10]
    top10_by_train_speed = sorted(
        trials, key=lambda row: row["report"]["train_samples_per_second"], reverse=True
    )[:10]
    top10_by_inference_speed = sorted(
        trials, key=lambda row: row["report"]["inference_samples_per_second"], reverse=True
    )[:10]
    top10_by_balanced_score = sorted(
        trials, key=lambda row: row["report"]["balanced_score"], reverse=True
    )[:10]
    pareto_trials = pareto_front(trials)

    best_balanced = top10_by_balanced_score[0]
    return {
        "model_name": best_balanced["report"]["model_name"],
        "trials": trials,
        "pareto_front": pareto_trials,
        "top10_by_accuracy": top10_by_accuracy,
        "top10_by_train_speed": top10_by_train_speed,
        "top10_by_inference_speed": top10_by_inference_speed,
        "top10_by_balanced_score": top10_by_balanced_score,
        "best_balanced": best_balanced,
    }


def compute_balanced_scores(trials: list[dict[str, Any]]) -> list[float]:
    accuracies = [trial["report"]["test_accuracy"] for trial in trials]
    train_speeds = [trial["report"]["train_samples_per_second"] for trial in trials]
    inference_speeds = [trial["report"]["inference_samples_per_second"] for trial in trials]

    acc_norm = normalize_values(accuracies)
    train_norm = normalize_values(train_speeds)
    infer_norm = normalize_values(inference_speeds)
    scores = []
    for a, t, i in zip(acc_norm, train_norm, infer_norm):
        scores.append(0.55 * a + 0.25 * t + 0.20 * i)
    return scores


def normalize_values(values: list[float]) -> list[float]:
    min_value = min(values)
    max_value = max(values)
    if max_value - min_value < 1e-12:
        return [1.0 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]


def pareto_front(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for candidate in trials:
        if not is_dominated(candidate, trials):
            front.append(candidate)
    front_sorted = sorted(
        front,
        key=lambda row: (
            row["report"]["test_accuracy"],
            row["report"]["train_samples_per_second"],
            row["report"]["inference_samples_per_second"],
        ),
        reverse=True,
    )
    return front_sorted


def is_dominated(candidate: dict[str, Any], trials: list[dict[str, Any]]) -> bool:
    c = candidate["report"]
    for other in trials:
        if other is candidate:
            continue
        o = other["report"]
        no_worse = (
            o["test_accuracy"] >= c["test_accuracy"]
            and o["train_samples_per_second"] >= c["train_samples_per_second"]
            and o["inference_samples_per_second"] >= c["inference_samples_per_second"]
        )
        strictly_better = (
            o["test_accuracy"] > c["test_accuracy"]
            or o["train_samples_per_second"] > c["train_samples_per_second"]
            or o["inference_samples_per_second"] > c["inference_samples_per_second"]
        )
        if no_worse and strictly_better:
            return True
    return False


def collect_all_trial_reports(output: dict[str, Any]) -> list[dict[str, Any]]:
    all_reports: list[dict[str, Any]] = []
    for model_key in ["backprop", "predictive", "circadian"]:
        model_name = output[model_key]["model_name"]
        for trial in output[model_key]["trials"]:
            row = trial["report"].copy()
            row["model_name"] = model_name
            row["params"] = trial["params"]
            all_reports.append(row)
    return all_reports


def best_from_all_trials(all_trial_reports: list[dict[str, Any]], key: str) -> dict[str, Any]:
    best = None
    for row in all_trial_reports:
        if best is None or row[key] > best[key]:
            best = row
    if best is None:
        raise ValueError("No best result available.")
    return best


if __name__ == "__main__":
    main()

