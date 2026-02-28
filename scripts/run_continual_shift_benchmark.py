"""Run continual-shift benchmark across backprop, predictive coding, and circadian PC."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.app.continual_shift_benchmark import (
    ContinualShiftConfig,
    format_continual_shift_benchmark,
    run_continual_shift_benchmark,
)
from src.core.circadian_predictive_coding import CircadianConfig


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for continual shift benchmark."""
    parser = argparse.ArgumentParser(
        description="Run phase-A/phase-B continual-shift benchmark for all three models."
    )
    parser.add_argument("--seeds", type=str, default="3,7,11,19,23,31,37")
    parser.add_argument(
        "--profile",
        type=str,
        choices=["baseline", "strength-case"],
        default="strength-case",
        help="Circadian profile: baseline uses defaults, strength-case emphasizes replay/splits.",
    )
    parser.add_argument("--sample-count-phase-a", type=int, default=500)
    parser.add_argument("--sample-count-phase-b", type=int, default=500)
    parser.add_argument("--phase-b-train-fraction", type=float, default=0.14)
    parser.add_argument("--phase-a-epochs", type=int, default=110)
    parser.add_argument("--phase-b-epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=12)
    parser.add_argument("--phase-a-noise-scale", type=float, default=0.8)
    parser.add_argument("--phase-b-noise-scale", type=float, default=1.0)
    parser.add_argument("--phase-b-rotation-degrees", type=float, default=40.0)
    parser.add_argument("--phase-b-translation-x", type=float, default=0.9)
    parser.add_argument("--phase-b-translation-y", type=float, default=-0.7)
    parser.add_argument("--sleep-interval-phase-a", type=int, default=40)
    parser.add_argument("--sleep-interval-phase-b", type=int, default=8)
    parser.add_argument("--output-file", type=str, default="")
    return parser


def main() -> None:
    """Run CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    seeds = _parse_int_list(args.seeds)
    circadian_config = (
        _build_strength_case_circadian_config()
        if args.profile == "strength-case"
        else CircadianConfig()
    )
    config = ContinualShiftConfig(
        sample_count_phase_a=args.sample_count_phase_a,
        sample_count_phase_b=args.sample_count_phase_b,
        phase_b_train_fraction=args.phase_b_train_fraction,
        phase_a_epochs=args.phase_a_epochs,
        phase_b_epochs=args.phase_b_epochs,
        hidden_dim=args.hidden_dim,
        phase_a_noise_scale=args.phase_a_noise_scale,
        phase_b_noise_scale=args.phase_b_noise_scale,
        phase_b_rotation_degrees=args.phase_b_rotation_degrees,
        phase_b_translation_x=args.phase_b_translation_x,
        phase_b_translation_y=args.phase_b_translation_y,
        circadian_sleep_interval_phase_a=args.sleep_interval_phase_a,
        circadian_sleep_interval_phase_b=args.sleep_interval_phase_b,
        circadian_config=circadian_config,
    )

    result = run_continual_shift_benchmark(config=config, seeds=seeds)
    formatted = format_continual_shift_benchmark(result)
    print(formatted)
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.write_text(formatted + "\n", encoding="utf-8")


def _build_strength_case_circadian_config() -> CircadianConfig:
    """Build a practical circadian profile for retention/adaptation stress tests."""
    return CircadianConfig(
        use_reward_modulated_learning=True,
        reward_scale_min=0.8,
        reward_scale_max=1.3,
        split_threshold=0.30,
        prune_threshold=0.04,
        max_split_per_sleep=1,
        max_prune_per_sleep=0,
        replay_steps=2,
        replay_memory_size=8,
        replay_learning_rate=0.03,
        replay_inference_steps=10,
        replay_inference_learning_rate=0.12,
    )


def _parse_int_list(raw_values: str) -> list[int]:
    items = [item.strip() for item in raw_values.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one integer seed.")
    return [int(item) for item in items]


if __name__ == "__main__":
    main()
