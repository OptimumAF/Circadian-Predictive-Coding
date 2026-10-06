"""Run continual-shift benchmark across backprop, predictive coding, and circadian PC."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.app.continual_shift_benchmark import (
    CONTINUAL_BOUNDED_REPLAY_PROTOCOL,
    CONTINUAL_GLOBAL_SEAL_PROTOCOL,
    CONTINUAL_LEGACY_PROTOCOL,
    CONTINUAL_PHASE_ARRIVAL_PROTOCOL,
    CONTINUAL_PHASE_LOCAL_SCHEDULE_PROTOCOL,
    CONTINUAL_VALIDATION_PROTOCOL,
    ContinualBoundedReplayConfig,
    ContinualGlobalSealConfig,
    ContinualShiftConfig,
    format_continual_shift_benchmark,
    run_continual_shift_benchmark,
)
from src.app.continual_experiment_config import (
    build_resolved_continual_record,
    resolve_continual_overrides,
)
from src.core.circadian_predictive_coding import CircadianConfig
from src.infra.local_result_json import write_local_result_json


@dataclass(frozen=True)
class ProfileDefaults:
    """Typed defaults for benchmark profile presets."""

    sample_count_phase_a: int
    sample_count_phase_b: int
    phase_b_train_fraction: float
    phase_a_epochs: int
    phase_b_epochs: int
    hidden_dim: int
    hidden_dims: tuple[int, ...] | None
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
        "--protocol-id",
        choices=[
            CONTINUAL_VALIDATION_PROTOCOL,
            CONTINUAL_LEGACY_PROTOCOL,
            CONTINUAL_PHASE_ARRIVAL_PROTOCOL,
            CONTINUAL_PHASE_LOCAL_SCHEDULE_PROTOCOL,
            CONTINUAL_BOUNDED_REPLAY_PROTOCOL,
            CONTINUAL_GLOBAL_SEAL_PROTOCOL,
        ],
        default=CONTINUAL_VALIDATION_PROTOCOL,
    )
    parser.add_argument("--validation-fraction", type=float, default=0.20)
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
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="",
        help="Optional comma-separated hidden-layer widths (e.g. 24,24,24).",
    )
    parser.add_argument("--phase-a-noise-scale", type=float, default=None)
    parser.add_argument("--phase-b-noise-scale", type=float, default=None)
    parser.add_argument("--phase-b-rotation-degrees", type=float, default=None)
    parser.add_argument("--phase-b-translation-x", type=float, default=None)
    parser.add_argument("--phase-b-translation-y", type=float, default=None)
    parser.add_argument("--sleep-interval-phase-a", type=int, default=None)
    parser.add_argument("--sleep-interval-phase-b", type=int, default=None)
    parser.add_argument("--replay-max-examples", type=int, default=None)
    parser.add_argument("--replay-max-bytes", type=int, default=None)
    parser.add_argument("--sleep-mode", choices=["components"], default=None)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="FIELD=JSON",
        help="explicit typed config field override, applied after profile and legacy flags",
    )
    parser.add_argument("--output-file", type=str, default="")
    parser.add_argument("--json-result", type=str, default="")
    parser.add_argument("--resolved-config", type=str, default="")
    return parser


def main() -> None:
    """Run CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    output_path = Path(args.output_file) if args.output_file else None
    if output_path is not None and output_path.exists():
        raise FileExistsError(f"Continual benchmark output already exists: {output_path}")
    json_path = Path(args.json_result) if args.json_result else None
    if json_path is not None and json_path.exists():
        raise FileExistsError(f"Continual JSON result already exists: {json_path}")
    if json_path is not None and json_path == output_path:
        parser.error("--json-result and --output-file must be different paths")
    config_path = (
        Path(getattr(args, "resolved_config", "")) if getattr(args, "resolved_config", "") else None
    )
    if config_path is not None and config_path.exists():
        raise FileExistsError(f"Continual resolved config already exists: {config_path}")
    requested_paths = [path.resolve() for path in (output_path, json_path, config_path) if path]
    if len(requested_paths) != len(set(requested_paths)):
        parser.error("continual output, JSON result, and resolved config paths must differ")
    raw_overrides = getattr(args, "override", [])
    if raw_overrides and json_path is None and config_path is None:
        parser.error("--override requires --json-result or --resolved-config")
    try:
        overrides = _parse_overrides(raw_overrides)
    except ValueError as exc:
        parser.error(str(exc))

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
    cli_hidden_dims = _parse_optional_hidden_dims(args.hidden_dims)
    selected_hidden_dims = (
        cli_hidden_dims if cli_hidden_dims is not None else profile_defaults.hidden_dims
    )
    config = ContinualShiftConfig(
        protocol_id=args.protocol_id,
        validation_fraction=args.validation_fraction,
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
        hidden_dims=selected_hidden_dims,
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
    if args.protocol_id in {CONTINUAL_BOUNDED_REPLAY_PROTOCOL, CONTINUAL_GLOBAL_SEAL_PROTOCOL}:
        if args.replay_max_examples is None or args.replay_max_bytes is None:
            parser.error("bounded replay requires --replay-max-examples and --replay-max-bytes")
        if circadian_config.replay_steps <= 0:
            parser.error("bounded replay requires a profile with replay_steps > 0")
        if args.sleep_mode != "components":
            parser.error("bounded replay requires --sleep-mode components")
        # Why this: v1/v2/v3 keep their original dataclass shape and saved
        # config digest; only opt-in bounded routes add retention limits.
        base_fields = {
            item.name: getattr(config, item.name) for item in fields(ContinualShiftConfig)
        }
        base_fields["circadian_config"] = replace(circadian_config, sleep_mode="components")
        bounded_config_type = (
            ContinualGlobalSealConfig
            if args.protocol_id == CONTINUAL_GLOBAL_SEAL_PROTOCOL
            else ContinualBoundedReplayConfig
        )
        config = bounded_config_type(
            **base_fields,
            replay_max_examples=args.replay_max_examples,
            replay_max_bytes=args.replay_max_bytes,
        )
    elif (
        args.replay_max_examples is not None
        or args.replay_max_bytes is not None
        or args.sleep_mode is not None
    ):
        parser.error("replay budget and sleep-mode flags require a bounded replay protocol")

    try:
        config = resolve_continual_overrides(config, overrides)
        resolved_record = build_resolved_continual_record(config, seeds, args.profile, overrides)
    except ValueError as exc:
        parser.error(str(exc))

    result = run_continual_shift_benchmark(config=config, seeds=seeds)
    formatted = format_continual_shift_benchmark(result)
    print(formatted)
    if output_path is not None:
        with output_path.open("x", encoding="utf-8") as output_file:
            output_file.write(formatted + "\n")
    if json_path is not None:
        write_local_result_json(result, json_path)
    if config_path is not None:
        with config_path.open("x", encoding="utf-8", newline="\n") as output:
            output.write(
                json.dumps(resolved_record, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )


def _parse_overrides(raw_overrides: list[str]) -> dict[str, object]:
    """Parse CLI syntax once; app validation owns field names and types."""
    overrides: dict[str, object] = {}
    for raw in raw_overrides:
        name, separator, value = raw.partition("=")
        if not separator or not name or name != name.strip() or not value:
            raise ValueError("override must use FIELD=JSON syntax")
        if name in overrides:
            raise ValueError(f"duplicate continual override key: {name}")
        try:
            overrides[name] = json.loads(value, parse_constant=_reject_nonfinite_override)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"override {name} must contain valid finite JSON") from exc
    return overrides


def _reject_nonfinite_override(value: str) -> object:
    raise ValueError(f"nonfinite override token: {value}")


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
        split_threshold=0.22,
        prune_threshold=0.04,
        max_split_per_sleep=2,
        max_prune_per_sleep=0,
        replay_steps=3,
        replay_memory_size=14,
        replay_learning_rate=0.04,
        replay_inference_steps=14,
        replay_inference_learning_rate=0.14,
    )


def _build_baseline_circadian_config() -> CircadianConfig:
    return CircadianConfig()


def _build_profile_defaults(profile: str) -> ProfileDefaults:
    if profile == "hardest-case":
        return ProfileDefaults(
            sample_count_phase_a=700,
            sample_count_phase_b=700,
            phase_b_train_fraction=0.05,
            phase_a_epochs=120,
            phase_b_epochs=180,
            hidden_dim=24,
            hidden_dims=(24, 24, 24),
            phase_a_noise_scale=0.8,
            phase_b_noise_scale=1.45,
            phase_b_rotation_degrees=68.0,
            phase_b_translation_x=1.6,
            phase_b_translation_y=-1.3,
            sleep_interval_phase_a=40,
            sleep_interval_phase_b=6,
        )
    return ProfileDefaults(
        sample_count_phase_a=500,
        sample_count_phase_b=500,
        phase_b_train_fraction=0.14,
        phase_a_epochs=110,
        phase_b_epochs=80,
        hidden_dim=12,
        hidden_dims=None,
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


def _parse_optional_hidden_dims(raw_values: str) -> tuple[int, ...] | None:
    if not raw_values.strip():
        return None
    values = _parse_int_list(raw_values)
    if any(value <= 0 for value in values):
        raise ValueError("hidden_dims values must be positive")
    return tuple(values)


if __name__ == "__main__":
    main()
