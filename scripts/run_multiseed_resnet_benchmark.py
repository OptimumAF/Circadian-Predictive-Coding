"""Run multi-seed ResNet benchmark comparisons and export JSON/CSV summaries."""

from __future__ import annotations

import argparse
import csv
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
    ModelSpeedReport,
    ResNet50BenchmarkConfig,
    run_resnet50_benchmark,
)


CORE_METRICS: tuple[str, ...] = (
    "test_accuracy",
    "final_cross_entropy",
    "train_seconds",
    "train_samples_per_second",
    "mean_train_step_ms",
    "inference_latency_mean_ms",
    "inference_latency_p95_ms",
    "inference_samples_per_second",
)

OPTIONAL_CIRCADIAN_METRICS: tuple[str, ...] = (
    "final_energy",
    "circadian_hidden_dim_start",
    "circadian_hidden_dim_end",
    "circadian_total_splits",
    "circadian_total_prunes",
    "circadian_total_rollbacks",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run multi-seed ResNet benchmark for backprop, predictive coding, and circadian."
        )
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="7,13,29",
        help="Comma-separated seed list.",
    )
    parser.add_argument(
        "--dataset-name",
        choices=["synthetic", "cifar10", "cifar100"],
        default="cifar100",
    )
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--dataset-download", dest="dataset_download", action="store_true")
    parser.add_argument("--dataset-no-download", dest="dataset_download", action="store_false")
    parser.set_defaults(dataset_download=True)
    parser.add_argument("--dataset-train-subset-size", type=int, default=0)
    parser.add_argument("--dataset-test-subset-size", type=int, default=0)
    parser.add_argument("--dataset-num-workers", type=int, default=0)
    parser.add_argument(
        "--dataset-use-augmentation",
        dest="dataset_use_augmentation",
        action="store_true",
    )
    parser.add_argument(
        "--dataset-no-augmentation",
        dest="dataset_use_augmentation",
        action="store_false",
    )
    parser.set_defaults(dataset_use_augmentation=True)
    parser.add_argument("--dataset-difficulty", choices=["easy", "medium", "hard"], default="hard")
    parser.add_argument("--dataset-noise-std", type=float, default=0.08)

    parser.add_argument("--classes", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=2500)
    parser.add_argument("--test-samples", type=int, default=700)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target-accuracy", type=float, default=-1.0)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--inference-batches", type=int, default=20)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument(
        "--backbone-weights",
        choices=["none", "imagenet"],
        default="imagenet",
    )
    parser.add_argument(
        "--backprop-freeze-backbone",
        dest="backprop_freeze_backbone",
        action="store_true",
    )
    parser.add_argument(
        "--backprop-train-backbone",
        dest="backprop_freeze_backbone",
        action="store_false",
    )
    parser.set_defaults(backprop_freeze_backbone=True)
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="benchmark_multiseed_results",
        help="Output path prefix (without extension).",
    )
    return parser


def parse_seed_list(raw: str) -> tuple[int, ...]:
    items = [part.strip() for part in raw.split(",")]
    seeds: list[int] = []
    for item in items:
        if item == "":
            continue
        value = int(item)
        if value < 0:
            raise ValueError("seed values must be non-negative.")
        seeds.append(value)
    if not seeds:
        raise ValueError("At least one seed is required.")
    return tuple(seeds)


def resolve_num_classes(dataset_name: str, classes: int | None) -> int:
    if classes is not None:
        return classes
    if dataset_name == "cifar10":
        return 10
    if dataset_name == "cifar100":
        return 100
    return 10


def build_base_config(args: argparse.Namespace) -> ResNet50BenchmarkConfig:
    target_accuracy: float | None
    if args.target_accuracy < 0.0:
        target_accuracy = None
    else:
        target_accuracy = float(args.target_accuracy)

    return ResNet50BenchmarkConfig(
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        num_classes=resolve_num_classes(args.dataset_name, args.classes),
        image_size=args.image_size,
        batch_size=args.batch_size,
        dataset_name=args.dataset_name,
        dataset_data_root=args.dataset_root,
        dataset_download=args.dataset_download,
        dataset_train_subset_size=args.dataset_train_subset_size,
        dataset_test_subset_size=args.dataset_test_subset_size,
        dataset_num_workers=args.dataset_num_workers,
        dataset_use_augmentation=args.dataset_use_augmentation,
        dataset_difficulty=args.dataset_difficulty,
        dataset_noise_std=args.dataset_noise_std,
        epochs=args.epochs,
        device=args.device,
        target_accuracy=target_accuracy,
        evaluation_batches=args.eval_batches,
        inference_batches=args.inference_batches,
        warmup_batches=args.warmup_batches,
        backbone_weights=args.backbone_weights,
        backprop_freeze_backbone=args.backprop_freeze_backbone,
    )


def report_to_row(seed: int, report: ModelSpeedReport) -> dict[str, Any]:
    return {
        "seed": seed,
        "model_name": report.model_name,
        "final_metric_name": report.final_metric_name,
        "final_metric_value": float(report.final_metric_value),
        "test_accuracy": float(report.test_accuracy),
        "final_cross_entropy": _optional_float(report.final_cross_entropy),
        "final_energy": _optional_float(report.final_energy),
        "train_seconds": float(report.train_seconds),
        "train_samples_per_second": float(report.train_samples_per_second),
        "mean_train_step_ms": float(report.mean_train_step_ms),
        "inference_latency_mean_ms": float(report.inference_latency_mean_ms),
        "inference_latency_p95_ms": float(report.inference_latency_p95_ms),
        "inference_samples_per_second": float(report.inference_samples_per_second),
        "total_parameters": int(report.total_parameters),
        "trainable_parameters": int(report.trainable_parameters),
        "epochs_ran": int(report.epochs_ran),
        "circadian_hidden_dim_start": _optional_float(report.circadian_hidden_dim_start),
        "circadian_hidden_dim_end": _optional_float(report.circadian_hidden_dim_end),
        "circadian_total_splits": int(report.circadian_total_splits),
        "circadian_total_prunes": int(report.circadian_total_prunes),
        "circadian_total_rollbacks": int(report.circadian_total_rollbacks),
    }


def _optional_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model_name"]), []).append(row)

    aggregate: list[dict[str, Any]] = []
    for model_name, model_rows in grouped.items():
        summary: dict[str, Any] = {
            "model_name": model_name,
            "seed_count": len(model_rows),
            "final_metric_name": model_rows[0]["final_metric_name"],
        }
        for metric in CORE_METRICS:
            values = [float(r[metric]) for r in model_rows]
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values))
        for metric in OPTIONAL_CIRCADIAN_METRICS:
            values = [r[metric] for r in model_rows if r[metric] is not None]
            if not values:
                summary[f"{metric}_mean"] = None
                summary[f"{metric}_std"] = None
            else:
                numeric = [float(v) for v in values]
                summary[f"{metric}_mean"] = float(np.mean(numeric))
                summary[f"{metric}_std"] = float(np.std(numeric))
        aggregate.append(summary)
    return sorted(aggregate, key=lambda row: str(row["model_name"]))


def compute_efficiency_winner(summary_rows: list[dict[str, Any]]) -> str:
    acc_values = [float(row["test_accuracy_mean"]) for row in summary_rows]
    train_values = [float(row["train_samples_per_second_mean"]) for row in summary_rows]
    infer_values = [float(row["inference_samples_per_second_mean"]) for row in summary_rows]
    acc_norm = normalize(acc_values)
    train_norm = normalize(train_values)
    infer_norm = normalize(infer_values)
    best_score = -1.0
    best_model = ""
    for row, a, t, i in zip(summary_rows, acc_norm, train_norm, infer_norm):
        score = 0.60 * a + 0.25 * t + 0.15 * i
        row["balanced_score"] = score
        if score > best_score:
            best_score = score
            best_model = str(row["model_name"])
    return best_model


def normalize(values: list[float]) -> list[float]:
    min_value = min(values)
    max_value = max(values)
    if max_value - min_value < 1e-12:
        return [1.0 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("rows must be non-empty.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def best_by_metric(summary_rows: list[dict[str, Any]], metric: str) -> str:
    return str(max(summary_rows, key=lambda row: float(row[metric]))["model_name"])


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    seeds = parse_seed_list(args.seeds)
    base = build_base_config(args)

    per_seed_rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        config = replace(base, seed=seed)
        print(f"[{index}/{len(seeds)}] running seed={seed}")
        result = run_resnet50_benchmark(config)
        for report in result.reports:
            row = report_to_row(seed=seed, report=report)
            per_seed_rows.append(row)
            print(
                f"  {row['model_name']}: acc={row['test_accuracy']:.4f} "
                f"train_sps={row['train_samples_per_second']:.1f} "
                f"infer_sps={row['inference_samples_per_second']:.1f}"
            )

    summary_rows = aggregate_rows(per_seed_rows)
    best_efficiency_model = compute_efficiency_winner(summary_rows)
    winners = {
        "best_accuracy_model": best_by_metric(summary_rows, "test_accuracy_mean"),
        "best_train_speed_model": best_by_metric(summary_rows, "train_samples_per_second_mean"),
        "best_inference_speed_model": best_by_metric(
            summary_rows, "inference_samples_per_second_mean"
        ),
        "best_balanced_model": best_efficiency_model,
    }

    output_prefix = Path(args.output_prefix)
    json_path = output_prefix.with_suffix(".json")
    per_seed_csv_path = output_prefix.with_name(f"{output_prefix.name}_per_seed.csv")
    summary_csv_path = output_prefix.with_name(f"{output_prefix.name}_summary.csv")

    payload = {
        "dataset": {
            "name": base.dataset_name,
            "root": base.dataset_data_root,
            "download": base.dataset_download,
            "train_subset_size": base.dataset_train_subset_size,
            "test_subset_size": base.dataset_test_subset_size,
            "num_workers": base.dataset_num_workers,
            "use_augmentation": base.dataset_use_augmentation,
            "difficulty": base.dataset_difficulty,
            "noise_std": base.dataset_noise_std,
            "train_samples": base.train_samples,
            "test_samples": base.test_samples,
            "num_classes": base.num_classes,
            "image_size": base.image_size,
            "batch_size": base.batch_size,
        },
        "runtime": {
            "device": base.device,
            "epochs": base.epochs,
            "backbone_weights": base.backbone_weights,
            "backprop_freeze_backbone": base.backprop_freeze_backbone,
            "target_accuracy": base.target_accuracy,
            "evaluation_batches": base.evaluation_batches,
            "inference_batches": base.inference_batches,
            "warmup_batches": base.warmup_batches,
            "seeds": list(seeds),
        },
        "winners": winners,
        "summary": summary_rows,
        "per_seed": per_seed_rows,
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(per_seed_csv_path, per_seed_rows)
    write_csv(summary_csv_path, summary_rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {per_seed_csv_path}")
    print(f"Wrote {summary_csv_path}")
    print(
        "Winners: "
        f"accuracy={winners['best_accuracy_model']}, "
        f"train_speed={winners['best_train_speed_model']}, "
        f"inference_speed={winners['best_inference_speed_model']}, "
        f"balanced={winners['best_balanced_model']}"
    )


if __name__ == "__main__":
    main()
