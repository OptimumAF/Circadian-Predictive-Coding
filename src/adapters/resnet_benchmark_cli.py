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
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0")
    parser.add_argument("--target-accuracy", type=float, default=0.99)
    parser.add_argument("--inference-batches", type=int, default=50)
    parser.add_argument("--warmup-batches", type=int, default=10)

    parser.add_argument("--backprop-lr", type=float, default=0.02)
    parser.add_argument("--backprop-momentum", type=float, default=0.9)
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
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        target_accuracy=target_accuracy,
        inference_batches=args.inference_batches,
        warmup_batches=args.warmup_batches,
        backprop_learning_rate=args.backprop_lr,
        backprop_momentum=args.backprop_momentum,
        backprop_freeze_backbone=args.backprop_freeze_backbone,
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
    )
    try:
        result = run_resnet50_benchmark(config)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(format_resnet50_benchmark_result(result))


if __name__ == "__main__":
    main()
