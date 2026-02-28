"""Run continual-shift benchmark across backprop, predictive coding, and circadian PC."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ProfileDefaults:
    """Typed defaults for benchmark profile presets."""

    sample_count_phase_a: int
    sample_count_phase_b: int
    phase_b_train_fraction: float
    phase_a_epochs: int
    phase_b_epochs: int
    hidden_dim: int
    phase_a_noise_scale: float
    phase_b_noise_scale: float
    phase_b_rotation_degrees: float
    phase_b_translation_x: float
    phase_b_translation_y: float
    sleep_interval_phase_a: int
    sleep_interval_phase_b: int


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for continual shift benchmark."""
    parser = argparse.ArgumentParser(
        description="Run phase-A/phase-B continual-shift benchmark for all three models."
    )
    parser.add_argument("--seeds", type=str, default="3,7,11,19,23,31,37")
    parser.add_argument(
        "--profile",
        type=str,
        choices=["baseline", "strength-case", "hardest-case"],
        default="strength-case",
        help=(
            "baseline: circadian defaults, strength-case: tuned moderate stress, "
            "hardest-case: aggressively difficult shift with tuned circadian policy."
        ),
    )
    parser.add_argument("--sample-count-phase-a", type=int, default=None)
    parser.add_argument("--sample-count-phase-b", type=int, default=None)
    parser.add_argument("--phase-b-train-fraction", type=float, default=None)
    parser.add_argument("--phase-a-epochs", type=int, default=None)
    parser.add_argument("--phase-b-epochs", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--phase-a-noise-scale", type=float, default=None)
    parser.add_argument("--phase-b-noise-scale", type=float, default=None)
    parser.add_argument("--phase-b-rotation-degrees", type=float, default=None)
    parser.add_argument("--phase-b-translation-x", type=float, default=None)
    parser.add_argument("--phase-b-translation-y", type=float, default=None)
    parser.add_argument("--sleep-interval-phase-a", type=int, default=None)
    parser.add_argument("--sleep-interval-phase-b", type=int, default=None)
    parser.add_argument("--output-file", type=str, default="")
    return parser


def main() -> None:
    """Run CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    seeds = _parse_int_list(args.seeds)
    profile_defaults = _build_profile_defaults(args.profile)
    circadian_config = (
        _build_baseline_circadian_config()
        if args.profile == "baseline"
        else (
            _build_strength_case_circadian_config()
            if args.profile == "strength-case"
            else _build_hardest_case_circadian_config()
        )
    )
    config = ContinualShiftConfig(
        sample_count_phase_a=_resolve_optional_int(
            args.sample_count_phase_a, profile_defaults.sample_count_phase_a
        ),
        sample_count_phase_b=_resolve_optional_int(
            args.sample_count_phase_b, profile_defaults.sample_count_phase_b
        ),
        phase_b_train_fraction=_resolve_optional_float(
            args.phase_b_train_fraction, profile_defaults.phase_b_train_fraction
        ),
        phase_a_epochs=_resolve_optional_int(args.phase_a_epochs, profile_defaults.phase_a_epochs),
        phase_b_epochs=_resolve_optional_int(args.phase_b_epochs, profile_defaults.phase_b_epochs),
        hidden_dim=_resolve_optional_int(args.hidden_dim, profile_defaults.hidden_dim),
        phase_a_noise_scale=_resolve_optional_float(
            args.phase_a_noise_scale, profile_defaults.phase_a_noise_scale
        ),
        phase_b_noise_scale=_resolve_optional_float(
            args.phase_b_noise_scale, profile_defaults.phase_b_noise_scale
        ),
        phase_b_rotation_degrees=_resolve_optional_float(
            args.phase_b_rotation_degrees, profile_defaults.phase_b_rotation_degrees
        ),
        phase_b_translation_x=_resolve_optional_float(
            args.phase_b_translation_x, profile_defaults.phase_b_translation_x
        ),
        phase_b_translation_y=_resolve_optional_float(
            args.phase_b_translation_y, profile_defaults.phase_b_translation_y
        ),
        circadian_sleep_interval_phase_a=_resolve_optional_int(
            args.sleep_interval_phase_a, profile_defaults.sleep_interval_phase_a
        ),
        circadian_sleep_interval_phase_b=_resolve_optional_int(
            args.sleep_interval_phase_b, profile_defaults.sleep_interval_phase_b
        ),
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


def _build_hardest_case_circadian_config() -> CircadianConfig:
    """Build circadian profile tuned for the hardest continual-shift setup."""
    return CircadianConfig(
        use_reward_modulated_learning=False,
        split_threshold=0.25,
        prune_threshold=0.04,
        max_split_per_sleep=1,
        max_prune_per_sleep=0,
        replay_steps=2,
        replay_memory_size=10,
        replay_learning_rate=0.04,
        replay_inference_steps=12,
        replay_inference_learning_rate=0.14,
    )


def _build_baseline_circadian_config() -> CircadianConfig:
    return CircadianConfig()


def _build_profile_defaults(profile: str) -> ProfileDefaults:
    if profile == "hardest-case":
        return ProfileDefaults(
            sample_count_phase_a=500,
            sample_count_phase_b=500,
            phase_b_train_fraction=0.08,
            phase_a_epochs=90,
            phase_b_epochs=120,
            hidden_dim=8,
            phase_a_noise_scale=0.8,
            phase_b_noise_scale=1.2,
            phase_b_rotation_degrees=44.0,
            phase_b_translation_x=0.9,
            phase_b_translation_y=-0.7,
            sleep_interval_phase_a=40,
            sleep_interval_phase_b=8,
        )
    return ProfileDefaults(
        sample_count_phase_a=500,
        sample_count_phase_b=500,
        phase_b_train_fraction=0.14,
        phase_a_epochs=110,
        phase_b_epochs=80,
        hidden_dim=12,
        phase_a_noise_scale=0.8,
        phase_b_noise_scale=1.0,
        phase_b_rotation_degrees=40.0,
        phase_b_translation_x=0.9,
        phase_b_translation_y=-0.7,
        sleep_interval_phase_a=40,
        sleep_interval_phase_b=8,
    )


def _resolve_optional_int(value: int | None, fallback: int) -> int:
    if value is None:
        return fallback
    return value


def _resolve_optional_float(value: float | None, fallback: float) -> float:
    if value is None:
        return fallback
    return value


def _parse_int_list(raw_values: str) -> list[int]:
    items = [item.strip() for item in raw_values.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one integer seed.")
    return [int(item) for item in items]


if __name__ == "__main__":
    main()
