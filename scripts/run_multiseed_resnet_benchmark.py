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
    VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL,
    VISION_SEEDED_UNMATCHED_PROTOCOL,
    VISION_VALIDATION_UNMATCHED_PROTOCOL,
    run_resnet50_benchmark,
)
from src.app.resnet_experiment_config import (  # noqa: E402
    MULTISEED_RESNET_PRESET_ID,
    build_resolved_multiseed_resnet_record,
    get_multiseed_resnet_preset,
    resolve_multiseed_resnet_overrides,
    validate_multiseed_resnet_config,
)


CORE_METRICS: tuple[str, ...] = (
    "validation_accuracy",
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
    preset = get_multiseed_resnet_preset()
    base = preset.config
    parser = argparse.ArgumentParser(
        description=(
            "Run multi-seed ResNet benchmark for backprop, predictive coding, and circadian."
        )
    )
    parser.add_argument("--preset", choices=[MULTISEED_RESNET_PRESET_ID], default=preset.preset_id)
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in preset.seeds),
        help="Comma-separated seed list.",
    )
    parser.add_argument(
        "--protocol-id",
        choices=[
            VISION_SEEDED_UNMATCHED_PROTOCOL,
            VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL,
            VISION_VALIDATION_UNMATCHED_PROTOCOL,
        ],
        default=base.protocol_id,
    )
    parser.add_argument(
        "--dataset-name",
        choices=["synthetic", "cifar10", "cifar100"],
        default=base.dataset_name,
    )
    parser.add_argument("--dataset-root", type=str, default=base.dataset_data_root)
    parser.add_argument("--dataset-download", dest="dataset_download", action="store_true")
    parser.add_argument("--dataset-no-download", dest="dataset_download", action="store_false")
    parser.set_defaults(dataset_download=base.dataset_download)
    parser.add_argument(
        "--dataset-train-subset-size", type=int, default=base.dataset_train_subset_size
    )
    parser.add_argument(
        "--dataset-validation-subset-size", type=int, default=base.dataset_validation_subset_size
    )
    parser.add_argument(
        "--dataset-guard-subset-size", type=int, default=base.dataset_guard_subset_size
    )
    parser.add_argument(
        "--dataset-test-subset-size", type=int, default=base.dataset_test_subset_size
    )
    parser.add_argument("--dataset-num-workers", type=int, default=base.dataset_num_workers)
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
    parser.set_defaults(dataset_use_augmentation=base.dataset_use_augmentation)
    parser.add_argument(
        "--dataset-difficulty", choices=["easy", "medium", "hard"], default=base.dataset_difficulty
    )
    parser.add_argument("--dataset-noise-std", type=float, default=base.dataset_noise_std)

    parser.add_argument("--classes", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=base.train_samples)
    parser.add_argument("--validation-samples", type=int, default=base.validation_samples)
    parser.add_argument("--guard-samples", type=int, default=base.guard_samples)
    parser.add_argument("--test-samples", type=int, default=base.test_samples)
    parser.add_argument("--image-size", type=int, default=base.image_size)
    parser.add_argument("--batch-size", type=int, default=base.batch_size)
    parser.add_argument("--epochs", type=int, default=base.epochs)
    parser.add_argument("--device", type=str, default=base.device)
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=-1.0 if base.target_accuracy is None else base.target_accuracy,
    )
    parser.add_argument("--eval-batches", type=int, default=base.evaluation_batches)
    parser.add_argument("--inference-batches", type=int, default=base.inference_batches)
    parser.add_argument("--warmup-batches", type=int, default=base.warmup_batches)
    parser.add_argument(
        "--backbone-weights",
        choices=["none", "imagenet"],
        default=base.backbone_weights,
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
    parser.set_defaults(backprop_freeze_backbone=base.backprop_freeze_backbone)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="FIELD=JSON",
        help="repeatable typed override of an existing CLI setting field",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=preset.output_prefix,
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
    preset = get_multiseed_resnet_preset(args.preset)
    target_accuracy: float | None
    if args.target_accuracy < 0.0:
        target_accuracy = None
    else:
        target_accuracy = float(args.target_accuracy)

    config = replace(
        preset.config,
        train_samples=args.train_samples,
        protocol_id=args.protocol_id,
        validation_samples=args.validation_samples,
        guard_samples=args.guard_samples,
        test_samples=args.test_samples,
        num_classes=resolve_num_classes(args.dataset_name, args.classes),
        image_size=args.image_size,
        batch_size=args.batch_size,
        dataset_name=args.dataset_name,
        dataset_data_root=args.dataset_root,
        dataset_download=args.dataset_download,
        dataset_train_subset_size=args.dataset_train_subset_size,
        dataset_validation_subset_size=args.dataset_validation_subset_size,
        dataset_guard_subset_size=args.dataset_guard_subset_size,
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
    validate_multiseed_resnet_config(config)
    return config


def report_to_row(seed: int, report: ModelSpeedReport) -> dict[str, Any]:
    return {
        "seed": seed,
        "model_name": report.model_name,
        "benchmark_track": report.benchmark_track,
        "backbone_trainable": report.backbone_trainable,
        "backbone_pretraining": report.backbone_pretraining,
        "head_type": report.head_type,
        "final_metric_name": report.final_metric_name,
        "final_metric_value": float(report.final_metric_value),
        "validation_accuracy": float(report.validation_accuracy),
        "test_accuracy": float(report.test_accuracy),
        "final_cross_entropy": _optional_float(report.final_cross_entropy),
        "final_energy": _optional_float(report.final_energy),
        "training_energy_id": report.training_energy_id,
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
        for metadata in (
            "benchmark_track",
            "backbone_trainable",
            "backbone_pretraining",
            "head_type",
            "training_energy_id",
        ):
            metadata_values = {row[metadata] for row in model_rows}
            if len(metadata_values) != 1:
                raise ValueError(f"Cannot aggregate mixed {metadata} for {model_name}.")
            summary[metadata] = model_rows[0][metadata]
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
    _require_one_track(summary_rows)
    acc_values = [float(row["validation_accuracy_mean"]) for row in summary_rows]
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
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def best_by_metric(summary_rows: list[dict[str, Any]], metric: str) -> str:
    _require_one_track(summary_rows)
    return str(max(summary_rows, key=lambda row: float(row[metric]))["model_name"])


def _require_one_track(rows: list[dict[str, Any]]) -> None:
    if not rows or any("benchmark_track" not in row for row in rows):
        raise ValueError("Ranking requires an explicit benchmark_track for every result.")
    tracks = {row["benchmark_track"] for row in rows}
    if len(tracks) != 1:
        raise ValueError("Cannot rank results from different benchmark tracks.")


def validation_selection_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose only declared selection metrics to winner calculations."""
    fields = (
        "model_name",
        "benchmark_track",
        "validation_accuracy_mean",
        "train_samples_per_second_mean",
        "inference_samples_per_second_mean",
    )
    return [{field: row[field] for field in fields} for row in summary_rows]


def _reject_nonfinite_json(token: str) -> object:
    raise ValueError(f"nonfinite JSON override is invalid: {token}")


def parse_overrides(raw_overrides: list[str]) -> dict[str, object]:
    """Parse repeatable field JSON without silently replacing a duplicate."""
    overrides: dict[str, object] = {}
    for raw in raw_overrides:
        if "=" not in raw:
            raise ValueError("ResNet override must use FIELD=JSON")
        name, encoded = raw.split("=", 1)
        if not name or name in overrides:
            raise ValueError(f"empty or duplicate ResNet override key: {name!r}")
        try:
            overrides[name] = json.loads(encoded, parse_constant=_reject_nonfinite_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON override for {name}") from exc
    return overrides


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_prefix = Path(args.output_prefix)
    json_path = output_prefix.with_suffix(".json")
    per_seed_csv_path = output_prefix.with_name(f"{output_prefix.name}_per_seed.csv")
    summary_csv_path = output_prefix.with_name(f"{output_prefix.name}_summary.csv")
    for output_path in (json_path, per_seed_csv_path, summary_csv_path):
        if output_path.exists():
            raise FileExistsError(f"Multi-seed output already exists: {output_path}")
    seeds = parse_seed_list(args.seeds)
    legacy_base = build_base_config(args)
    overrides = parse_overrides(args.override)
    base = resolve_multiseed_resnet_overrides(legacy_base, overrides)
    resolved_record = build_resolved_multiseed_resnet_record(
        base, seeds, args.preset, sys.argv[1:], overrides
    )

    per_seed_rows: list[dict[str, Any]] = []
    split_hashes_by_seed: dict[str, dict[str, str]] = {}
    trained_model_hashes_by_seed: dict[str, dict[str, str]] = {}
    for index, seed in enumerate(seeds, start=1):
        config = replace(base, seed=seed)
        print(f"[{index}/{len(seeds)}] running seed={seed}")
        result = run_resnet50_benchmark(config)
        if result.config != config:
            raise ValueError("ResNet runner result configuration differs from requested trial")
        split_hashes_by_seed[str(seed)] = result.split_hashes
        if result.trained_model_hashes is not None:
            trained_model_hashes_by_seed[str(seed)] = result.trained_model_hashes
        for report in result.reports:
            row = report_to_row(seed=seed, report=report)
            per_seed_rows.append(row)
            print(
                f"  {row['model_name']}: validation_acc={row['validation_accuracy']:.4f} "
                f"test_acc={row['test_accuracy']:.4f} "
                f"train_sps={row['train_samples_per_second']:.1f} "
                f"infer_sps={row['inference_samples_per_second']:.1f}"
            )

    summary_rows = aggregate_rows(per_seed_rows)
    selection_rows = validation_selection_rows(summary_rows)
    best_efficiency_model = compute_efficiency_winner(selection_rows)
    winners = {
        "best_accuracy_model": best_by_metric(selection_rows, "validation_accuracy_mean"),
        "best_train_speed_model": best_by_metric(selection_rows, "train_samples_per_second_mean"),
        "best_inference_speed_model": best_by_metric(
            selection_rows, "inference_samples_per_second_mean"
        ),
        "best_balanced_model": best_efficiency_model,
    }
    scores_by_model = {row["model_name"]: row["balanced_score"] for row in selection_rows}
    for row in summary_rows:
        row["balanced_score"] = scores_by_model[row["model_name"]]

    payload = {
        "protocol_id": base.protocol_id,
        "comparison_status": "unmatched reference; validation winners are descriptive only",
        "resolved_config": resolved_record,
        "dataset": {
            "name": base.dataset_name,
            "root": base.dataset_data_root,
            "download": base.dataset_download,
            "train_subset_size": base.dataset_train_subset_size,
            "validation_subset_size": base.dataset_validation_subset_size,
            "guard_subset_size": (
                base.dataset_guard_subset_size
                if base.protocol_id
                in {
                    VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL,
                    VISION_SEEDED_UNMATCHED_PROTOCOL,
                }
                else 0
            ),
            "test_subset_size": base.dataset_test_subset_size,
            "num_workers": base.dataset_num_workers,
            "use_augmentation": base.dataset_use_augmentation,
            "difficulty": base.dataset_difficulty,
            "noise_std": base.dataset_noise_std,
            "train_samples": base.train_samples,
            "validation_samples": base.validation_samples,
            "guard_samples": (
                base.guard_samples
                if base.protocol_id
                in {
                    VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL,
                    VISION_SEEDED_UNMATCHED_PROTOCOL,
                }
                else 0
            ),
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
        "split_hashes_by_seed": split_hashes_by_seed,
        "trained_model_hashes_by_seed": trained_model_hashes_by_seed,
        "summary": summary_rows,
        "per_seed": per_seed_rows,
    }

    with json_path.open("x", encoding="utf-8") as output_file:
        output_file.write(json.dumps(payload, indent=2))
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
