"""Command-line adapter for running the baseline experiment."""

from __future__ import annotations

import argparse

from src.app.indepth_comparison import (
    format_indepth_comparison_result,
    run_indepth_comparison,
)
from src.app.experiment_runner import ExperimentConfig, format_experiment_result, run_experiment
from src.config.settings import load_settings_from_env


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    settings = load_settings_from_env()

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
    return parser


def main() -> None:
    """Run experiment from command line."""
    parser = build_argument_parser()
    arguments = parser.parse_args()

    config = ExperimentConfig(
        sample_count=arguments.samples,
        noise_scale=arguments.noise,
        hidden_dim=arguments.hidden_dim,
        epoch_count=arguments.epochs,
        circadian_sleep_interval=arguments.sleep_interval,
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
