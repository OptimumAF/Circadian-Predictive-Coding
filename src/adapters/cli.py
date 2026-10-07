"""Command-line adapter for running the baseline experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from src.app.indepth_comparison import (
    format_indepth_comparison_result,
    run_indepth_comparison,
)
from src.app.experiment_runner import (
    TOY_LEGACY_PROTOCOL,
    TOY_VALIDATION_PROTOCOL,
    format_experiment_result,
    run_experiment,
)
from src.app.toy_experiment_config import (
    TOY_PRESET_ID,
    ToyExperimentPreset,
    build_resolved_toy_record,
    get_toy_experiment_preset,
    resolve_toy_overrides,
)
from src.app.toy_execution_budget import ToyExecutionBudget
from src.adapters.toy_budget_cli import run_budgeted_toy_cli, validate_budget_paths
from src.config.settings import load_settings_from_env
from src.core.circadian_predictive_coding import CircadianConfig
from src.infra.toy_result_files import write_toy_result_json
from src.infra.local_result_json import write_local_json_payload


def build_argument_parser(preset: ToyExperimentPreset | None = None) -> argparse.ArgumentParser:
    """Create CLI parser."""
    preset = preset or get_toy_experiment_preset(load_settings_from_env())
    defaults = preset.config
    circadian_defaults = defaults.circadian_config
    assert circadian_defaults is not None

    parser = argparse.ArgumentParser(
        description="Compare backprop, predictive coding, and circadian predictive coding."
    )
    parser.add_argument("--preset", choices=[TOY_PRESET_ID], default=TOY_PRESET_ID)
    parser.add_argument(
        "--mode",
        choices=["baseline", "indepth"],
        default="baseline",
        help="baseline: single run, indepth: aggregate over multiple seeds/noise levels.",
    )
    parser.add_argument(
        "--samples", type=int, default=defaults.sample_count, help="Number of samples."
    )
    parser.add_argument(
        "--protocol-id",
        choices=[TOY_VALIDATION_PROTOCOL, TOY_LEGACY_PROTOCOL],
        default=defaults.protocol_id,
    )
    parser.add_argument(
        "--json-result",
        type=str,
        default=None,
        help="Write the completed baseline comparison and sleep events to a new local JSON file.",
    )
    parser.add_argument(
        "--resolved-config",
        type=str,
        default=None,
        help="Write the complete resolved settings to a new local JSON file.",
    )
    parser.add_argument(
        "--run-state",
        type=str,
        default=None,
        help="Exclusive local lifecycle JSON for an opt-in budgeted baseline run.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Trusted local toy checkpoint for a budgeted run and checked resume.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an incomplete/error budgeted run from its exact saved checkpoint.",
    )
    parser.add_argument("--max-training-updates", type=int, default=None)
    parser.add_argument("--max-wall-seconds", type=float, default=None)
    parser.add_argument("--max-replay-examples", type=int, default=None)
    parser.add_argument("--max-hidden-width", type=int, default=None)
    parser.add_argument("--max-process-rss-bytes", type=int, default=None)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="FIELD=JSON",
        help="Typed existing config-field override; requires a JSON result or config artifact.",
    )
    parser.add_argument("--validation-fraction", type=float, default=defaults.validation_fraction)
    parser.add_argument("--epochs", type=int, default=defaults.epoch_count, help="Training epochs.")
    parser.add_argument(
        "--hidden-dim", type=int, default=defaults.hidden_dim, help="Hidden layer width."
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="",
        help="Optional comma-separated hidden-layer widths (for multi-hidden-layer models).",
    )
    parser.add_argument(
        "--noise", type=float, default=defaults.noise_scale, help="Dataset noise scale."
    )
    parser.add_argument(
        "--noise-levels",
        type=str,
        default=",".join(str(value) for value in preset.indepth_noise_levels),
        help="Comma-separated noise levels for --mode indepth.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.random_seed,
        help="Random seed for deterministic runs.",
    )
    parser.add_argument(
        "--seed-list",
        type=str,
        default=",".join(str(value) for value in preset.indepth_seeds),
        help="Comma-separated seeds for --mode indepth.",
    )
    parser.add_argument(
        "--sleep-interval",
        type=int,
        default=defaults.circadian_sleep_interval,
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
    preset = get_toy_experiment_preset(load_settings_from_env())
    parser = build_argument_parser(preset)
    arguments = parser.parse_args()
    json_path = Path(arguments.json_result) if arguments.json_result is not None else None
    config_path = Path(arguments.resolved_config) if arguments.resolved_config is not None else None
    state_path = Path(arguments.run_state) if arguments.run_state is not None else None
    checkpoint_path = Path(arguments.checkpoint) if arguments.checkpoint is not None else None
    if arguments.override and json_path is None and config_path is None and state_path is None:
        parser.error(
            "--override requires --json-result or --resolved-config "
            "(or --run-state for budgeted runs)"
        )
    budget_requested = any(
        (
            arguments.max_training_updates is not None,
            arguments.max_wall_seconds is not None,
            arguments.max_replay_examples is not None,
            arguments.max_hidden_width is not None,
            arguments.max_process_rss_bytes is not None,
            state_path is not None,
            checkpoint_path is not None,
            arguments.resume,
        )
    )
    budget = None
    if budget_requested:
        if arguments.mode != "baseline":
            parser.error("toy execution budget is supported only for --mode baseline")
        if state_path is None:
            parser.error("toy execution budget requires --run-state")
        try:
            budget = ToyExecutionBudget(
                max_training_updates=arguments.max_training_updates,
                max_wall_seconds=arguments.max_wall_seconds,
                max_replay_examples=arguments.max_replay_examples,
                max_hidden_width=arguments.max_hidden_width,
                max_process_rss_bytes=arguments.max_process_rss_bytes,
            )
            validate_budget_paths(state_path, checkpoint_path, json_path, config_path)
        except ValueError as exc:
            parser.error(str(exc))
    if arguments.json_result is not None:
        if arguments.mode != "baseline":
            parser.error("--json-result is supported only for --mode baseline")
        if json_path is not None and json_path.exists():
            raise FileExistsError(f"toy JSON result already exists: {arguments.json_result}")
    if (
        config_path is not None
        and config_path.exists()
        and not (budget is not None and arguments.resume)
    ):
        raise FileExistsError(f"toy resolved config already exists: {config_path}")
    if (
        json_path is not None
        and config_path is not None
        and json_path.resolve() == config_path.resolve()
    ):
        parser.error("--json-result and --resolved-config must be different paths")

    circadian_config = replace(
        preset.config.circadian_config or CircadianConfig(),
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
    try:
        hidden_dims = None
        if arguments.hidden_dims.strip():
            hidden_dims = tuple(_parse_int_list(arguments.hidden_dims))
        overrides = _parse_overrides(arguments.override)
        config = replace(
            preset.config,
            sample_count=arguments.samples,
            protocol_id=arguments.protocol_id,
            validation_fraction=arguments.validation_fraction,
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
        config = resolve_toy_overrides(config, overrides)
        noise_levels = (
            [config.noise_scale]
            if arguments.mode == "baseline"
            else _parse_float_list(arguments.noise_levels)
        )
        seeds = (
            [config.random_seed]
            if arguments.mode == "baseline"
            else _parse_int_list(arguments.seed_list)
        )
        resolved_record = build_resolved_toy_record(
            config,
            arguments.mode,
            seeds,
            noise_levels,
            arguments.preset,
            sys.argv[1:],
            overrides,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if arguments.mode == "baseline":
        if budget is None:
            result = run_experiment(config=config)
            if json_path is not None:
                write_toy_result_json(result, json_path, resolved_record)
            if config_path is not None:
                write_local_json_payload(resolved_record, config_path)
        else:
            assert state_path is not None
            result = run_budgeted_toy_cli(
                config=config,
                resolved_record=resolved_record,
                budget=budget,
                state_path=state_path,
                checkpoint_path=checkpoint_path,
                result_path=json_path,
                config_path=config_path,
                resume=arguments.resume,
                input_tokens=sys.argv[1:],
                runner=run_experiment,
            )
        print(format_experiment_result(result))
        return

    indepth_result = run_indepth_comparison(
        base_config=config,
        seeds=seeds,
        noise_levels=noise_levels,
    )
    if config_path is not None:
        write_local_json_payload(resolved_record, config_path)
    print(format_indepth_comparison_result(indepth_result))


def _parse_overrides(raw_overrides: list[str]) -> dict[str, object]:
    """Parse JSON tokens once; the app resolver owns names and types."""
    overrides: dict[str, object] = {}
    for raw in raw_overrides:
        name, separator, value = raw.partition("=")
        if not separator or not name or name != name.strip() or not value:
            raise ValueError("override must use FIELD=JSON syntax")
        if name in overrides:
            raise ValueError(f"duplicate toy override key: {name}")
        try:
            overrides[name] = json.loads(value, parse_constant=_reject_nonfinite_override)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"override {name} must contain valid finite JSON") from exc
    return overrides


def _reject_nonfinite_override(value: str) -> object:
    raise ValueError(f"nonfinite override token: {value}")


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
