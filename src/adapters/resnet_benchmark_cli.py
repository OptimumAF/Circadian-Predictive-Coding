"""CLI adapter for ResNet-50 speed benchmarking."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import sys

from src.app.resnet50_benchmark import (
    ResNet50BenchmarkConfig,
    ResNet50BenchmarkResult,
    VISION_DEFAULT_MODEL_ORDER,
    VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL,
    VISION_SEEDED_UNMATCHED_PROTOCOL,
    VISION_VALIDATION_UNMATCHED_PROTOCOL,
    format_resnet50_benchmark_result,
    run_resnet50_benchmark,
)
from src.app.single_resnet_experiment_config import (
    SINGLE_RESNET_PRESET_ID,
    SingleResnetPreset,
    build_resolved_single_resnet_record,
    get_single_resnet_preset,
    resolve_single_resnet_overrides,
)
from src.infra.local_result_json import write_local_json_payload


def build_argument_parser(preset: SingleResnetPreset | None = None) -> argparse.ArgumentParser:
    """Build CLI parser for ResNet-50 benchmark runs."""
    preset = preset or get_single_resnet_preset()
    parser = argparse.ArgumentParser(
        description="Benchmark Backprop, Predictive Coding, and Circadian Predictive Coding on ResNet-50."
    )
    parser.add_argument(
        "--preset", choices=[SINGLE_RESNET_PRESET_ID], default=SINGLE_RESNET_PRESET_ID
    )
    parser.add_argument("--json-result", type=str, default=None)
    parser.add_argument("--resolved-config", type=str, default=None)
    parser.add_argument("--override", action="append", default=[], metavar="FIELD=JSON")
    parser.add_argument("--train-samples", type=int)
    parser.add_argument(
        "--protocol-id",
        choices=[
            VISION_SEEDED_UNMATCHED_PROTOCOL,
            VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL,
            VISION_VALIDATION_UNMATCHED_PROTOCOL,
        ],
    )
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--guard-samples", type=int)
    parser.add_argument("--test-samples", type=int)
    parser.add_argument(
        "--classes",
        type=int,
        help="Class count for synthetic dataset mode. Ignored for CIFAR modes unless set explicitly.",
    )
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--dataset-name",
        choices=["synthetic", "cifar10", "cifar100"],
        help="Dataset source used for all three models.",
    )
    parser.add_argument("--dataset-root", type=str)
    parser.add_argument("--dataset-download", dest="dataset_download", action="store_true")
    parser.add_argument("--dataset-no-download", dest="dataset_download", action="store_false")
    parser.add_argument("--dataset-train-subset-size", type=int)
    parser.add_argument("--dataset-validation-subset-size", type=int)
    parser.add_argument("--dataset-guard-subset-size", type=int)
    parser.add_argument("--dataset-test-subset-size", type=int)
    parser.add_argument("--dataset-num-workers", type=int)
    parser.add_argument(
        "--dataset-use-augmentation",
        dest="dataset_use_augmentation",
        action="store_true",
        help="Enable train-time augmentation for torchvision datasets.",
    )
    parser.add_argument(
        "--dataset-no-augmentation",
        dest="dataset_use_augmentation",
        action="store_false",
        help="Disable train-time augmentation for torchvision datasets.",
    )
    parser.add_argument(
        "--dataset-difficulty",
        choices=["easy", "medium", "hard"],
        help="Controls class overlap/distractors/noise in synthetic data.",
    )
    parser.add_argument("--dataset-noise-std", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str, help="auto, cpu, cuda, cuda:0")
    parser.add_argument("--target-accuracy", type=float)
    parser.add_argument(
        "--eval-batches",
        type=int,
        help="Per-epoch validation batch count for early-stop checks; 0 uses full validation loader.",
    )
    parser.add_argument("--inference-batches", type=int)
    parser.add_argument("--warmup-batches", type=int)

    parser.add_argument("--backprop-lr", type=float)
    parser.add_argument("--backprop-momentum", type=float)
    parser.add_argument(
        "--backbone-weights",
        choices=["none", "imagenet"],
        help="ResNet-50 initialization: random weights or ImageNet-pretrained weights.",
    )
    parser.add_argument(
        "--backprop-freeze-backbone",
        action="store_true",
        help="Freeze ResNet backbone for backprop baseline.",
    )

    parser.add_argument("--pc-hidden-dim", type=int)
    parser.add_argument("--pc-lr", type=float)
    parser.add_argument("--pc-steps", type=int)
    parser.add_argument("--pc-inference-lr", type=float)

    parser.add_argument("--circ-hidden-dim", type=int)
    parser.add_argument("--circ-lr", type=float)
    parser.add_argument("--circ-steps", type=int)
    parser.add_argument("--circ-inference-lr", type=float)
    parser.add_argument("--circ-sleep-interval", type=int)
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
    parser.add_argument("--circ-min-sleep-steps", type=int)
    parser.add_argument("--circ-sleep-energy-window", type=int)
    parser.add_argument("--circ-sleep-plateau-delta", type=float)
    parser.add_argument("--circ-sleep-chemical-variance-threshold", type=float)
    parser.add_argument(
        "--circ-use-adaptive-sleep-budget",
        dest="circ_use_adaptive_sleep_budget",
        action="store_true",
        help="Enable adaptive split/prune budget scaling by plateau and chemical variance.",
    )
    parser.add_argument(
        "--circ-disable-adaptive-sleep-budget",
        dest="circ_use_adaptive_sleep_budget",
        action="store_false",
        help="Disable adaptive split/prune budget scaling.",
    )
    parser.add_argument("--circ-adaptive-sleep-budget-min-scale", type=float)
    parser.add_argument("--circ-adaptive-sleep-budget-max-scale", type=float)
    parser.add_argument("--circ-adaptive-sleep-budget-plateau-weight", type=float)
    parser.add_argument("--circ-adaptive-sleep-budget-variance-weight", type=float)
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
    parser.add_argument("--circ-sleep-rollback-tolerance", type=float)
    parser.add_argument(
        "--circ-sleep-rollback-metric",
        choices=["accuracy", "cross_entropy"],
    )
    parser.add_argument(
        "--circ-sleep-rollback-eval-batches",
        type=int,
        help="Evaluation batches used only for pre/post sleep rollback checks; 0 inherits --eval-batches.",
    )
    parser.add_argument(
        "--circ-sleep-rollback-cooldown-epochs",
        type=int,
        help="Completed epochs to pause after a rejected sleep; default 0 for legacy, 1 for components. A positive value also requires a new wake batch before retry.",
    )
    parser.add_argument("--circ-min-hidden-dim", type=int)
    parser.add_argument("--circ-max-hidden-dim", type=int)
    parser.add_argument("--circ-chemical-decay", type=float)
    parser.add_argument("--circ-chemical-buildup-rate", type=float)
    parser.add_argument("--circ-use-saturating-chemical", action="store_true")
    parser.add_argument("--circ-chemical-max-value", type=float)
    parser.add_argument("--circ-chemical-saturation-gain", type=float)
    parser.add_argument("--circ-use-dual-chemical", action="store_true")
    parser.add_argument("--circ-dual-fast-mix", type=float)
    parser.add_argument("--circ-slow-chemical-decay", type=float)
    parser.add_argument("--circ-slow-buildup-scale", type=float)
    parser.add_argument("--circ-plasticity-sensitivity", type=float)
    parser.add_argument("--circ-use-adaptive-plasticity-sensitivity", action="store_true")
    parser.add_argument("--circ-plasticity-sensitivity-min", type=float)
    parser.add_argument("--circ-plasticity-sensitivity-max", type=float)
    parser.add_argument("--circ-plasticity-importance-mix", type=float)
    parser.add_argument("--circ-min-plasticity", type=float)
    parser.add_argument(
        "--circ-use-reward-modulated-learning",
        action="store_true",
        help="Scale wake learning rate by batch difficulty relative to a moving baseline.",
    )
    parser.add_argument("--circ-reward-baseline-decay", type=float)
    parser.add_argument("--circ-reward-difficulty-exponent", type=float)
    parser.add_argument("--circ-reward-scale-min", type=float)
    parser.add_argument("--circ-reward-scale-max", type=float)
    parser.add_argument("--circ-use-adaptive-thresholds", action="store_true")
    parser.add_argument("--circ-adaptive-split-percentile", type=float)
    parser.add_argument("--circ-adaptive-prune-percentile", type=float)
    parser.add_argument("--circ-sleep-warmup-steps", type=int)
    parser.add_argument("--circ-sleep-split-only-until-fraction", type=float)
    parser.add_argument("--circ-sleep-prune-only-after-fraction", type=float)
    parser.add_argument("--circ-sleep-max-change-fraction", type=float)
    parser.add_argument("--circ-sleep-min-change-count", type=int)
    parser.add_argument("--circ-prune-min-age-steps", type=int)
    parser.add_argument("--circ-split-threshold", type=float)
    parser.add_argument("--circ-prune-threshold", type=float)
    parser.add_argument("--circ-split-hysteresis-margin", type=float)
    parser.add_argument("--circ-prune-hysteresis-margin", type=float)
    parser.add_argument("--circ-split-cooldown-steps", type=int)
    parser.add_argument("--circ-prune-cooldown-steps", type=int)
    parser.add_argument("--circ-split-weight-norm-mix", type=float)
    parser.add_argument("--circ-prune-weight-norm-mix", type=float)
    parser.add_argument("--circ-split-importance-mix", type=float)
    parser.add_argument("--circ-prune-importance-mix", type=float)
    parser.add_argument("--circ-importance-ema-decay", type=float)
    parser.add_argument("--circ-max-split-per-sleep", type=int)
    parser.add_argument("--circ-max-prune-per-sleep", type=int)
    parser.add_argument("--circ-split-noise-scale", type=float)
    parser.add_argument("--circ-sleep-reset-factor", type=float)
    parser.add_argument("--circ-sleep-mode", choices=("legacy", "components", "disabled"))
    parser.add_argument("--circ-disable-chemical-reset", action="store_true")
    parser.add_argument("--circ-disable-homeostasis", action="store_true")
    parser.add_argument("--circ-disable-split", action="store_true")
    parser.add_argument("--circ-disable-prune", action="store_true")
    parser.add_argument("--circ-homeostatic-downscale-factor", type=float)
    parser.add_argument("--circ-homeostasis-target-input-norm", type=float)
    parser.add_argument("--circ-homeostasis-target-output-norm", type=float)
    parser.add_argument("--circ-homeostasis-strength", type=float)
    # Why this: one typed dataclass owns all 110 historical defaults; the
    # adapter only translates field names and preserves --classes auto choice.
    parser.set_defaults(**_preset_argument_defaults(preset.config))
    return parser


def _preset_argument_defaults(config: ResNet50BenchmarkConfig) -> dict[str, object]:
    aliases = {
        "num_classes": "classes",
        "dataset_data_root": "dataset_root",
        "evaluation_batches": "eval_batches",
        "backprop_learning_rate": "backprop_lr",
        "predictive_head_hidden_dim": "pc_hidden_dim",
        "predictive_learning_rate": "pc_lr",
        "predictive_inference_steps": "pc_steps",
        "predictive_inference_learning_rate": "pc_inference_lr",
        "circadian_head_hidden_dim": "circ_hidden_dim",
        "circadian_learning_rate": "circ_lr",
        "circadian_inference_steps": "circ_steps",
        "circadian_inference_learning_rate": "circ_inference_lr",
    }
    defaults: dict[str, object] = {}
    for item in fields(config):
        name = item.name
        value = getattr(config, name)
        if name in aliases:
            destination = aliases[name]
        elif name.startswith("circadian_sleep_enable_"):
            destination = "circ_disable_" + name.removeprefix("circadian_sleep_enable_")
            value = not value
        elif name.startswith("circadian_"):
            destination = "circ_" + name.removeprefix("circadian_")
        else:
            destination = name
        # The old --classes omission follows the selected CIFAR dataset.
        defaults[destination] = None if name == "num_classes" else value
    return defaults


def main() -> None:
    """Run benchmark and print formatted report."""
    preset = get_single_resnet_preset()
    parser = build_argument_parser(preset)
    args = parser.parse_args()
    result_path = Path(args.json_result) if args.json_result is not None else None
    config_path = Path(args.resolved_config) if args.resolved_config is not None else None
    if args.override and result_path is None and config_path is None:
        parser.error("--override requires --json-result or --resolved-config")
    for path in (result_path, config_path):
        if path is not None and path.exists():
            raise FileExistsError(f"single-run ResNet artifact already exists: {path}")
    if (
        result_path is not None
        and config_path is not None
        and result_path.resolve() == config_path.resolve()
    ):
        parser.error("--json-result and --resolved-config must be different paths")
    try:
        overrides = _parse_overrides(args.override)
    except ValueError as exc:
        parser.error(str(exc))

    target_accuracy = args.target_accuracy
    if target_accuracy < 0.0:
        target_accuracy = None
    if args.classes is None:
        if args.dataset_name == "cifar10":
            num_classes = 10
        elif args.dataset_name == "cifar100":
            num_classes = 100
        else:
            num_classes = 10
    else:
        num_classes = args.classes

    config = replace(
        preset.config,
        train_samples=args.train_samples,
        protocol_id=args.protocol_id,
        validation_samples=args.validation_samples,
        guard_samples=args.guard_samples,
        test_samples=args.test_samples,
        num_classes=num_classes,
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
        circadian_sleep_chemical_variance_threshold=(args.circ_sleep_chemical_variance_threshold),
        circadian_use_adaptive_sleep_budget=args.circ_use_adaptive_sleep_budget,
        circadian_adaptive_sleep_budget_min_scale=args.circ_adaptive_sleep_budget_min_scale,
        circadian_adaptive_sleep_budget_max_scale=args.circ_adaptive_sleep_budget_max_scale,
        circadian_adaptive_sleep_budget_plateau_weight=(
            args.circ_adaptive_sleep_budget_plateau_weight
        ),
        circadian_adaptive_sleep_budget_variance_weight=(
            args.circ_adaptive_sleep_budget_variance_weight
        ),
        circadian_enable_sleep_rollback=args.circ_enable_sleep_rollback,
        circadian_sleep_rollback_tolerance=args.circ_sleep_rollback_tolerance,
        circadian_sleep_rollback_metric=args.circ_sleep_rollback_metric,
        circadian_sleep_rollback_eval_batches=args.circ_sleep_rollback_eval_batches,
        circadian_sleep_rollback_cooldown_epochs=args.circ_sleep_rollback_cooldown_epochs,
        circadian_min_hidden_dim=args.circ_min_hidden_dim,
        circadian_max_hidden_dim=args.circ_max_hidden_dim,
        circadian_chemical_decay=args.circ_chemical_decay,
        circadian_chemical_buildup_rate=args.circ_chemical_buildup_rate,
        circadian_use_saturating_chemical=(
            True if args.circ_use_saturating_chemical is None else args.circ_use_saturating_chemical
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
        circadian_use_reward_modulated_learning=args.circ_use_reward_modulated_learning,
        circadian_reward_baseline_decay=args.circ_reward_baseline_decay,
        circadian_reward_difficulty_exponent=args.circ_reward_difficulty_exponent,
        circadian_reward_scale_min=args.circ_reward_scale_min,
        circadian_reward_scale_max=args.circ_reward_scale_max,
        circadian_use_adaptive_thresholds=(
            True if args.circ_use_adaptive_thresholds is None else args.circ_use_adaptive_thresholds
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
        circadian_sleep_mode=args.circ_sleep_mode,
        circadian_sleep_enable_chemical_reset=not args.circ_disable_chemical_reset,
        circadian_sleep_enable_homeostasis=not args.circ_disable_homeostasis,
        circadian_sleep_enable_split=not args.circ_disable_split,
        circadian_sleep_enable_prune=not args.circ_disable_prune,
        circadian_homeostatic_downscale_factor=args.circ_homeostatic_downscale_factor,
        circadian_homeostasis_target_input_norm=args.circ_homeostasis_target_input_norm,
        circadian_homeostasis_target_output_norm=args.circ_homeostasis_target_output_norm,
        circadian_homeostasis_strength=args.circ_homeostasis_strength,
    )
    try:
        config = resolve_single_resnet_overrides(config, overrides)
        resolved_record = build_resolved_single_resnet_record(
            config, args.preset, sys.argv[1:], overrides
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        result = run_resnet50_benchmark(config)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if result_path is not None or config_path is not None:
        if not isinstance(result, ResNet50BenchmarkResult) or result.config != config:
            raise ValueError("single-run ResNet runner result config differs from request")
        if result.training_order != VISION_DEFAULT_MODEL_ORDER:
            raise ValueError("single-run ResNet runner order differs from request")
    if result_path is not None:
        payload = asdict(result)
        payload["resolved_config"] = resolved_record
        write_local_json_payload(payload, result_path)
    if config_path is not None:
        write_local_json_payload(resolved_record, config_path)
    print(format_resnet50_benchmark_result(result))


def _parse_overrides(raw_overrides: list[str]) -> dict[str, object]:
    """Parse JSON syntax once; app validation owns setting names and types."""
    overrides: dict[str, object] = {}
    for raw in raw_overrides:
        name, separator, value = raw.partition("=")
        if not separator or not name or name != name.strip() or not value:
            raise ValueError("override must use FIELD=JSON syntax")
        if name in overrides:
            raise ValueError(f"duplicate single-run ResNet override key: {name}")
        try:
            overrides[name] = json.loads(value, parse_constant=_reject_nonfinite_override)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"override {name} must contain valid finite JSON") from exc
    return overrides


def _reject_nonfinite_override(value: str) -> object:
    raise ValueError(f"nonfinite override token: {value}")


if __name__ == "__main__":
    main()
