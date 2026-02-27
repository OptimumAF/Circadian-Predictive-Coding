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
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=2,
        help="Per-epoch evaluation batch count for early-stop/rollback checks; 0 uses full test loader.",
    )
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

    parser.add_argument("--pc-hidden-dim", type=int, default=384)
    parser.add_argument("--pc-lr", type=float, default=0.03)
    parser.add_argument("--pc-steps", type=int, default=10)
    parser.add_argument("--pc-inference-lr", type=float, default=0.15)

    parser.add_argument("--circ-hidden-dim", type=int, default=384)
    parser.add_argument("--circ-lr", type=float, default=0.025)
    parser.add_argument("--circ-steps", type=int, default=14)
    parser.add_argument("--circ-inference-lr", type=float, default=0.12)
    parser.add_argument("--circ-sleep-interval", type=int, default=2)
    parser.add_argument(
        "--circ-use-adaptive-sleep-trigger",
        dest="circ_use_adaptive_sleep_trigger",
        action="store_true",
        help="Enable adaptive sleep triggering from energy/chemical dynamics.",
    )
    parser.add_argument(
        "--circ-disable-adaptive-sleep-trigger",
        dest="circ_use_adaptive_sleep_trigger",
        action="store_false",
        help="Disable adaptive sleep triggering.",
    )
    parser.set_defaults(circ_use_adaptive_sleep_trigger=True)
    parser.add_argument("--circ-min-sleep-steps", type=int, default=60)
    parser.add_argument("--circ-sleep-energy-window", type=int, default=48)
    parser.add_argument("--circ-sleep-plateau-delta", type=float, default=5e-5)
    parser.add_argument("--circ-sleep-chemical-variance-threshold", type=float, default=0.02)
    parser.add_argument(
        "--circ-force-sleep",
        dest="circ_force_sleep",
        action="store_true",
        help="Always execute sleep events at interval checkpoints.",
    )
    parser.add_argument(
        "--circ-respect-adaptive-sleep-trigger",
        dest="circ_force_sleep",
        action="store_false",
        help="At interval checkpoints, run sleep only when adaptive trigger conditions are met.",
    )
    parser.set_defaults(circ_force_sleep=False)
    parser.add_argument(
        "--circ-enable-sleep-rollback",
        dest="circ_enable_sleep_rollback",
        action="store_true",
        help="Enable post-sleep rollback guard.",
    )
    parser.add_argument(
        "--circ-disable-sleep-rollback",
        dest="circ_enable_sleep_rollback",
        action="store_false",
        help="Disable post-sleep rollback guard.",
    )
    parser.set_defaults(circ_enable_sleep_rollback=True)
    parser.add_argument("--circ-sleep-rollback-tolerance", type=float, default=0.002)
    parser.add_argument(
        "--circ-sleep-rollback-metric",
        choices=["accuracy", "cross_entropy"],
        default="cross_entropy",
    )
    parser.add_argument(
        "--circ-sleep-rollback-eval-batches",
        type=int,
        default=2,
        help="Evaluation batches used only for pre/post sleep rollback checks; 0 inherits --eval-batches.",
    )
    parser.add_argument("--circ-min-hidden-dim", type=int, default=384)
    parser.add_argument("--circ-max-hidden-dim", type=int, default=640)
    parser.add_argument("--circ-chemical-decay", type=float, default=0.995)
    parser.add_argument("--circ-chemical-buildup-rate", type=float, default=0.02)
    parser.add_argument("--circ-use-saturating-chemical", action="store_true", default=None)
    parser.add_argument("--circ-chemical-max-value", type=float, default=2.5)
    parser.add_argument("--circ-chemical-saturation-gain", type=float, default=1.0)
    parser.add_argument("--circ-use-dual-chemical", action="store_true", default=None)
    parser.add_argument("--circ-dual-fast-mix", type=float, default=0.70)
    parser.add_argument("--circ-slow-chemical-decay", type=float, default=0.999)
    parser.add_argument("--circ-slow-buildup-scale", type=float, default=0.25)
    parser.add_argument("--circ-plasticity-sensitivity", type=float, default=0.45)
    parser.add_argument("--circ-use-adaptive-plasticity-sensitivity", action="store_true", default=None)
    parser.add_argument("--circ-plasticity-sensitivity-min", type=float, default=0.25)
    parser.add_argument("--circ-plasticity-sensitivity-max", type=float, default=0.55)
    parser.add_argument("--circ-plasticity-importance-mix", type=float, default=0.50)
    parser.add_argument("--circ-min-plasticity", type=float, default=0.5)
    parser.add_argument("--circ-use-adaptive-thresholds", action="store_true", default=None)
    parser.add_argument("--circ-adaptive-split-percentile", type=float, default=92.0)
    parser.add_argument("--circ-adaptive-prune-percentile", type=float, default=8.0)
    parser.add_argument("--circ-sleep-warmup-steps", type=int, default=6)
    parser.add_argument("--circ-sleep-split-only-until-fraction", type=float, default=0.75)
    parser.add_argument("--circ-sleep-prune-only-after-fraction", type=float, default=0.95)
    parser.add_argument("--circ-sleep-max-change-fraction", type=float, default=0.01)
    parser.add_argument("--circ-sleep-min-change-count", type=int, default=1)
    parser.add_argument("--circ-prune-min-age-steps", type=int, default=120)
    parser.add_argument("--circ-split-threshold", type=float, default=0.8)
    parser.add_argument("--circ-prune-threshold", type=float, default=0.08)
    parser.add_argument("--circ-split-hysteresis-margin", type=float, default=0.02)
    parser.add_argument("--circ-prune-hysteresis-margin", type=float, default=0.02)
    parser.add_argument("--circ-split-cooldown-steps", type=int, default=3)
    parser.add_argument("--circ-prune-cooldown-steps", type=int, default=3)
    parser.add_argument("--circ-split-weight-norm-mix", type=float, default=0.30)
    parser.add_argument("--circ-prune-weight-norm-mix", type=float, default=0.30)
    parser.add_argument("--circ-split-importance-mix", type=float, default=0.20)
    parser.add_argument("--circ-prune-importance-mix", type=float, default=0.35)
    parser.add_argument("--circ-importance-ema-decay", type=float, default=0.95)
    parser.add_argument("--circ-max-split-per-sleep", type=int, default=1)
    parser.add_argument("--circ-max-prune-per-sleep", type=int, default=1)
    parser.add_argument("--circ-split-noise-scale", type=float, default=0.01)
    parser.add_argument("--circ-sleep-reset-factor", type=float, default=0.45)
    parser.add_argument("--circ-homeostatic-downscale-factor", type=float, default=1.0)
    parser.add_argument("--circ-homeostasis-target-input-norm", type=float, default=0.0)
    parser.add_argument("--circ-homeostasis-target-output-norm", type=float, default=0.0)
    parser.add_argument("--circ-homeostasis-strength", type=float, default=0.50)
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
        evaluation_batches=args.eval_batches,
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
        circadian_force_sleep=args.circ_force_sleep,
        circadian_use_adaptive_sleep_trigger=args.circ_use_adaptive_sleep_trigger,
        circadian_min_sleep_steps=args.circ_min_sleep_steps,
        circadian_sleep_energy_window=args.circ_sleep_energy_window,
        circadian_sleep_plateau_delta=args.circ_sleep_plateau_delta,
        circadian_sleep_chemical_variance_threshold=(
            args.circ_sleep_chemical_variance_threshold
        ),
        circadian_enable_sleep_rollback=args.circ_enable_sleep_rollback,
        circadian_sleep_rollback_tolerance=args.circ_sleep_rollback_tolerance,
        circadian_sleep_rollback_metric=args.circ_sleep_rollback_metric,
        circadian_sleep_rollback_eval_batches=args.circ_sleep_rollback_eval_batches,
        circadian_min_hidden_dim=args.circ_min_hidden_dim,
        circadian_max_hidden_dim=args.circ_max_hidden_dim,
        circadian_chemical_decay=args.circ_chemical_decay,
        circadian_chemical_buildup_rate=args.circ_chemical_buildup_rate,
        circadian_use_saturating_chemical=(
            True
            if args.circ_use_saturating_chemical is None
            else args.circ_use_saturating_chemical
        ),
        circadian_chemical_max_value=args.circ_chemical_max_value,
        circadian_chemical_saturation_gain=args.circ_chemical_saturation_gain,
        circadian_use_dual_chemical=(
            True if args.circ_use_dual_chemical is None else args.circ_use_dual_chemical
        ),
        circadian_dual_fast_mix=args.circ_dual_fast_mix,
        circadian_slow_chemical_decay=args.circ_slow_chemical_decay,
        circadian_slow_buildup_scale=args.circ_slow_buildup_scale,
        circadian_plasticity_sensitivity=args.circ_plasticity_sensitivity,
        circadian_use_adaptive_plasticity_sensitivity=(
            True
            if args.circ_use_adaptive_plasticity_sensitivity is None
            else args.circ_use_adaptive_plasticity_sensitivity
        ),
        circadian_plasticity_sensitivity_min=args.circ_plasticity_sensitivity_min,
        circadian_plasticity_sensitivity_max=args.circ_plasticity_sensitivity_max,
        circadian_plasticity_importance_mix=args.circ_plasticity_importance_mix,
        circadian_min_plasticity=args.circ_min_plasticity,
        circadian_use_adaptive_thresholds=(
            True
            if args.circ_use_adaptive_thresholds is None
            else args.circ_use_adaptive_thresholds
        ),
        circadian_adaptive_split_percentile=args.circ_adaptive_split_percentile,
        circadian_adaptive_prune_percentile=args.circ_adaptive_prune_percentile,
        circadian_sleep_warmup_steps=args.circ_sleep_warmup_steps,
        circadian_sleep_split_only_until_fraction=args.circ_sleep_split_only_until_fraction,
        circadian_sleep_prune_only_after_fraction=args.circ_sleep_prune_only_after_fraction,
        circadian_sleep_max_change_fraction=args.circ_sleep_max_change_fraction,
        circadian_sleep_min_change_count=args.circ_sleep_min_change_count,
        circadian_prune_min_age_steps=args.circ_prune_min_age_steps,
        circadian_split_threshold=args.circ_split_threshold,
        circadian_prune_threshold=args.circ_prune_threshold,
        circadian_split_hysteresis_margin=args.circ_split_hysteresis_margin,
        circadian_prune_hysteresis_margin=args.circ_prune_hysteresis_margin,
        circadian_split_cooldown_steps=args.circ_split_cooldown_steps,
        circadian_prune_cooldown_steps=args.circ_prune_cooldown_steps,
        circadian_split_weight_norm_mix=args.circ_split_weight_norm_mix,
        circadian_prune_weight_norm_mix=args.circ_prune_weight_norm_mix,
        circadian_split_importance_mix=args.circ_split_importance_mix,
        circadian_prune_importance_mix=args.circ_prune_importance_mix,
        circadian_importance_ema_decay=args.circ_importance_ema_decay,
        circadian_max_split_per_sleep=args.circ_max_split_per_sleep,
        circadian_max_prune_per_sleep=args.circ_max_prune_per_sleep,
        circadian_split_noise_scale=args.circ_split_noise_scale,
        circadian_sleep_reset_factor=args.circ_sleep_reset_factor,
        circadian_homeostatic_downscale_factor=args.circ_homeostatic_downscale_factor,
        circadian_homeostasis_target_input_norm=args.circ_homeostasis_target_input_norm,
        circadian_homeostasis_target_output_norm=args.circ_homeostasis_target_output_norm,
        circadian_homeostasis_strength=args.circ_homeostasis_strength,
    )
    try:
        result = run_resnet50_benchmark(config)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(format_resnet50_benchmark_result(result))


if __name__ == "__main__":
    main()
