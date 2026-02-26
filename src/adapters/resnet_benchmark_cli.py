"""CLI adapter for ResNet-50 speed benchmarking."""

from __future__ import annotations

import argparse

from src.app.resnet50_benchmark import (
    ResNet50BenchmarkConfig,
    format_resnet50_benchmark_result,
    run_resnet50_benchmark,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build CLI parser for ResNet-50 benchmark runs."""
    parser = argparse.ArgumentParser(
        description="Benchmark Backprop, Predictive Coding, and Circadian Predictive Coding on ResNet-50."
    )
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--dataset-difficulty",
        choices=["easy", "medium", "hard"],
        default="medium",
        help="Controls class overlap/distractors/noise in synthetic data.",
    )
    parser.add_argument("--dataset-noise-std", type=float, default=0.06)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0")
    parser.add_argument("--target-accuracy", type=float, default=0.99)
    parser.add_argument("--inference-batches", type=int, default=50)
    parser.add_argument("--warmup-batches", type=int, default=10)

    parser.add_argument("--backprop-lr", type=float, default=0.02)
    parser.add_argument("--backprop-momentum", type=float, default=0.9)
    parser.add_argument(
        "--backbone-weights",
        choices=["none", "imagenet"],
        default="none",
        help="ResNet-50 initialization: random weights or ImageNet-pretrained weights.",
    )
    parser.add_argument(
        "--backprop-freeze-backbone",
        action="store_true",
        help="Freeze ResNet backbone for backprop baseline.",
    )

    parser.add_argument("--pc-hidden-dim", type=int, default=256)
    parser.add_argument("--pc-lr", type=float, default=0.03)
    parser.add_argument("--pc-steps", type=int, default=10)
    parser.add_argument("--pc-inference-lr", type=float, default=0.15)

    parser.add_argument("--circ-hidden-dim", type=int, default=256)
    parser.add_argument("--circ-lr", type=float, default=0.03)
    parser.add_argument("--circ-steps", type=int, default=10)
    parser.add_argument("--circ-inference-lr", type=float, default=0.15)
    parser.add_argument("--circ-sleep-interval", type=int, default=2)
    parser.add_argument("--circ-min-hidden-dim", type=int, default=96)
    parser.add_argument("--circ-max-hidden-dim", type=int, default=1024)
    parser.add_argument("--circ-chemical-decay", type=float, default=0.995)
    parser.add_argument("--circ-chemical-buildup-rate", type=float, default=0.02)
    parser.add_argument("--circ-plasticity-sensitivity", type=float, default=0.7)
    parser.add_argument("--circ-min-plasticity", type=float, default=0.2)
    parser.add_argument("--circ-split-threshold", type=float, default=0.8)
    parser.add_argument("--circ-prune-threshold", type=float, default=0.08)
    parser.add_argument("--circ-max-split-per-sleep", type=int, default=2)
    parser.add_argument("--circ-max-prune-per-sleep", type=int, default=2)
    parser.add_argument("--circ-split-noise-scale", type=float, default=0.01)
    parser.add_argument("--circ-sleep-reset-factor", type=float, default=0.45)
    return parser


def main() -> None:
    """Run benchmark and print formatted report."""
    parser = build_argument_parser()
    args = parser.parse_args()

    target_accuracy = args.target_accuracy
    if target_accuracy < 0.0:
        target_accuracy = None

    config = ResNet50BenchmarkConfig(
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        num_classes=args.classes,
        image_size=args.image_size,
        batch_size=args.batch_size,
        dataset_difficulty=args.dataset_difficulty,
        dataset_noise_std=args.dataset_noise_std,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        target_accuracy=target_accuracy,
        inference_batches=args.inference_batches,
        warmup_batches=args.warmup_batches,
        backprop_learning_rate=args.backprop_lr,
        backprop_momentum=args.backprop_momentum,
        backprop_freeze_backbone=args.backprop_freeze_backbone,
        backbone_weights=args.backbone_weights,
        predictive_head_hidden_dim=args.pc_hidden_dim,
        predictive_learning_rate=args.pc_lr,
        predictive_inference_steps=args.pc_steps,
        predictive_inference_learning_rate=args.pc_inference_lr,
        circadian_head_hidden_dim=args.circ_hidden_dim,
        circadian_learning_rate=args.circ_lr,
        circadian_inference_steps=args.circ_steps,
        circadian_inference_learning_rate=args.circ_inference_lr,
        circadian_sleep_interval=args.circ_sleep_interval,
        circadian_min_hidden_dim=args.circ_min_hidden_dim,
        circadian_max_hidden_dim=args.circ_max_hidden_dim,
        circadian_chemical_decay=args.circ_chemical_decay,
        circadian_chemical_buildup_rate=args.circ_chemical_buildup_rate,
        circadian_plasticity_sensitivity=args.circ_plasticity_sensitivity,
        circadian_min_plasticity=args.circ_min_plasticity,
        circadian_split_threshold=args.circ_split_threshold,
        circadian_prune_threshold=args.circ_prune_threshold,
        circadian_max_split_per_sleep=args.circ_max_split_per_sleep,
        circadian_max_prune_per_sleep=args.circ_max_prune_per_sleep,
        circadian_split_noise_scale=args.circ_split_noise_scale,
        circadian_sleep_reset_factor=args.circ_sleep_reset_factor,
    )
    try:
        result = run_resnet50_benchmark(config)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(format_resnet50_benchmark_result(result))


if __name__ == "__main__":
    main()
