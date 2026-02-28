"""Command-line adapter for running the baseline experiment."""

from __future__ import annotations

import argparse

from src.app.indepth_comparison import (
    format_indepth_comparison_result,
    run_indepth_comparison,
)
from src.app.experiment_runner import ExperimentConfig, format_experiment_result, run_experiment
from src.config.settings import load_settings_from_env
from src.core.circadian_predictive_coding import CircadianConfig


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    settings = load_settings_from_env()
    circadian_defaults = CircadianConfig()

    parser = argparse.ArgumentParser(
        description="Compare backprop, predictive coding, and circadian predictive coding."
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "indepth"],
        default="baseline",
        help="baseline: single run, indepth: aggregate over multiple seeds/noise levels.",
    )
    parser.add_argument("--samples", type=int, default=settings.dataset_size, help="Number of samples.")
    parser.add_argument("--epochs", type=int, default=settings.epoch_count, help="Training epochs.")
    parser.add_argument("--hidden-dim", type=int, default=12, help="Hidden layer width.")
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="",
        help="Optional comma-separated hidden-layer widths (for multi-hidden-layer models).",
    )
    parser.add_argument("--noise", type=float, default=0.8, help="Dataset noise scale.")
    parser.add_argument(
        "--noise-levels",
        type=str,
        default="0.6,0.8,1.0",
        help="Comma-separated noise levels for --mode indepth.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=settings.base_seed,
        help="Random seed for deterministic runs.",
    )
    parser.add_argument(
        "--seed-list",
        type=str,
        default="3,7,11,19,23",
        help="Comma-separated seeds for --mode indepth.",
    )
    parser.add_argument(
        "--sleep-interval",
        type=int,
        default=40,
        help="Epoch interval for circadian sleep events. Use 0 to disable sleep.",
    )
    parser.add_argument(
        "--respect-adaptive-sleep-trigger",
        action="store_true",
        help="If set, scheduled sleep checks adaptive trigger instead of forcing sleep.",
    )
    parser.add_argument(
        "--use-policy-for-sleep",
        action="store_true",
        help="If set, sleep uses NeuronAdaptationPolicy proposals for structural changes.",
    )

    parser.add_argument(
        "--adaptive-thresholds",
        action="store_true",
        help="Use percentile-based split/prune thresholds.",
    )
    parser.add_argument(
        "--adaptive-split-percentile",
        type=float,
        default=circadian_defaults.adaptive_split_percentile,
    )
    parser.add_argument(
        "--adaptive-prune-percentile",
        type=float,
        default=circadian_defaults.adaptive_prune_percentile,
    )
    parser.add_argument(
        "--split-threshold",
        type=float,
        default=circadian_defaults.split_threshold,
    )
    parser.add_argument(
        "--prune-threshold",
        type=float,
        default=circadian_defaults.prune_threshold,
    )

    parser.add_argument(
        "--adaptive-sleep-trigger",
        action="store_true",
        help="Enable adaptive sleep trigger based on energy plateau and chemical variance.",
    )
    parser.add_argument(
        "--min-epochs-between-sleep",
        type=int,
        default=circadian_defaults.min_epochs_between_sleep,
    )
    parser.add_argument(
        "--sleep-energy-window",
        type=int,
        default=circadian_defaults.sleep_energy_window,
    )
    parser.add_argument(
        "--sleep-plateau-delta",
        type=float,
        default=circadian_defaults.sleep_plateau_delta,
    )
    parser.add_argument(
        "--sleep-chemical-variance-threshold",
        type=float,
        default=circadian_defaults.sleep_chemical_variance_threshold,
    )
    parser.add_argument(
        "--adaptive-sleep-budget",
        action="store_true",
        help="Scale split/prune budgets based on plateau severity and chemical variance.",
    )
    parser.add_argument(
        "--adaptive-sleep-budget-min-scale",
        type=float,
        default=circadian_defaults.adaptive_sleep_budget_min_scale,
    )
    parser.add_argument(
        "--adaptive-sleep-budget-max-scale",
        type=float,
        default=circadian_defaults.adaptive_sleep_budget_max_scale,
    )

    parser.add_argument(
        "--reward-modulated-learning",
        action="store_true",
        help="Scale wake learning rate by batch difficulty relative to recent error baseline.",
    )
    parser.add_argument(
        "--reward-scale-min",
        type=float,
        default=circadian_defaults.reward_scale_min,
    )
    parser.add_argument(
        "--reward-scale-max",
        type=float,
        default=circadian_defaults.reward_scale_max,
    )

    parser.add_argument(
        "--split-weight-norm-mix",
        type=float,
        default=circadian_defaults.split_weight_norm_mix,
    )
    parser.add_argument(
        "--prune-weight-norm-mix",
        type=float,
        default=circadian_defaults.prune_weight_norm_mix,
    )
    parser.add_argument(
        "--prune-decay-steps",
        type=int,
        default=circadian_defaults.prune_decay_steps,
    )
    parser.add_argument(
        "--prune-decay-factor",
        type=float,
        default=circadian_defaults.prune_decay_factor,
    )
    parser.add_argument(
        "--homeostatic-downscale-factor",
        type=float,
        default=circadian_defaults.homeostatic_downscale_factor,
    )

    parser.add_argument(
        "--replay-steps",
        type=int,
        default=circadian_defaults.replay_steps,
    )
    parser.add_argument(
        "--replay-memory-size",
        type=int,
        default=circadian_defaults.replay_memory_size,
    )
    parser.add_argument(
        "--replay-learning-rate",
        type=float,
        default=circadian_defaults.replay_learning_rate,
    )
    parser.add_argument(
        "--replay-inference-steps",
        type=int,
        default=circadian_defaults.replay_inference_steps,
    )
    parser.add_argument(
        "--replay-inference-learning-rate",
        type=float,
        default=circadian_defaults.replay_inference_learning_rate,
    )
    return parser


def main() -> None:
    """Run experiment from command line."""
    parser = build_argument_parser()
    arguments = parser.parse_args()

    circadian_config = CircadianConfig(
        use_adaptive_thresholds=arguments.adaptive_thresholds,
        adaptive_split_percentile=arguments.adaptive_split_percentile,
        adaptive_prune_percentile=arguments.adaptive_prune_percentile,
        split_threshold=arguments.split_threshold,
        prune_threshold=arguments.prune_threshold,
        use_adaptive_sleep_trigger=arguments.adaptive_sleep_trigger,
        min_epochs_between_sleep=arguments.min_epochs_between_sleep,
        sleep_energy_window=arguments.sleep_energy_window,
        sleep_plateau_delta=arguments.sleep_plateau_delta,
        sleep_chemical_variance_threshold=arguments.sleep_chemical_variance_threshold,
        use_adaptive_sleep_budget=arguments.adaptive_sleep_budget,
        adaptive_sleep_budget_min_scale=arguments.adaptive_sleep_budget_min_scale,
        adaptive_sleep_budget_max_scale=arguments.adaptive_sleep_budget_max_scale,
        use_reward_modulated_learning=arguments.reward_modulated_learning,
        reward_scale_min=arguments.reward_scale_min,
        reward_scale_max=arguments.reward_scale_max,
        split_weight_norm_mix=arguments.split_weight_norm_mix,
        prune_weight_norm_mix=arguments.prune_weight_norm_mix,
        prune_decay_steps=arguments.prune_decay_steps,
        prune_decay_factor=arguments.prune_decay_factor,
        homeostatic_downscale_factor=arguments.homeostatic_downscale_factor,
        replay_steps=arguments.replay_steps,
        replay_memory_size=arguments.replay_memory_size,
        replay_learning_rate=arguments.replay_learning_rate,
        replay_inference_steps=arguments.replay_inference_steps,
        replay_inference_learning_rate=arguments.replay_inference_learning_rate,
    )
    hidden_dims = None
    if arguments.hidden_dims.strip():
        hidden_dims = tuple(_parse_int_list(arguments.hidden_dims))

    config = ExperimentConfig(
        sample_count=arguments.samples,
        noise_scale=arguments.noise,
        hidden_dim=arguments.hidden_dim,
        hidden_dims=hidden_dims,
        epoch_count=arguments.epochs,
        circadian_sleep_interval=arguments.sleep_interval,
        circadian_force_sleep=(not arguments.respect_adaptive_sleep_trigger),
        circadian_use_policy_for_sleep=arguments.use_policy_for_sleep,
        circadian_config=circadian_config,
        random_seed=arguments.seed,
    )
    if arguments.mode == "baseline":
        result = run_experiment(config=config)
        print(format_experiment_result(result))
        return

    noise_levels = _parse_float_list(arguments.noise_levels)
    seeds = _parse_int_list(arguments.seed_list)
    indepth_result = run_indepth_comparison(
        base_config=config,
        seeds=seeds,
        noise_levels=noise_levels,
    )
    print(format_indepth_comparison_result(indepth_result))


def _parse_int_list(raw_values: str) -> list[int]:
    values = [item.strip() for item in raw_values.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return [int(value) for value in values]


def _parse_float_list(raw_values: str) -> list[float]:
    values = [item.strip() for item in raw_values.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return [float(value) for value in values]


if __name__ == "__main__":
    main()
