"""Command-line adapter for running the baseline experiment."""

from __future__ import annotations

import argparse

from src.app.experiment_runner import ExperimentConfig, format_experiment_result, run_experiment
from src.config.settings import load_settings_from_env


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    settings = load_settings_from_env()

    parser = argparse.ArgumentParser(
        description="Compare predictive coding and backprop on a synthetic dataset."
    )
    parser.add_argument("--samples", type=int, default=settings.dataset_size, help="Number of samples.")
    parser.add_argument("--epochs", type=int, default=settings.epoch_count, help="Training epochs.")
    parser.add_argument("--hidden-dim", type=int, default=12, help="Hidden layer width.")
    parser.add_argument("--noise", type=float, default=0.8, help="Dataset noise scale.")
    parser.add_argument(
        "--seed",
        type=int,
        default=settings.base_seed,
        help="Random seed for deterministic runs.",
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
        random_seed=arguments.seed,
    )
    result = run_experiment(config=config)
    print(format_experiment_result(result))


if __name__ == "__main__":
    main()

