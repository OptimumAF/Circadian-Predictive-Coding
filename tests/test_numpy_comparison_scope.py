"""NumPy algorithm-version and attribution-scope reporting boundaries."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any

from PIL import Image

from src.app.comparison_scope import scope_for_hidden_dims
from src.app.continual_shift_benchmark import (
    ContinualShiftConfig,
    format_continual_shift_benchmark,
    run_continual_shift_benchmark,
)
from src.app.experiment_runner import ExperimentConfig, format_experiment_result, run_experiment
from src.app.indepth_comparison import format_indepth_comparison_result, run_indepth_comparison

dynamics = import_module("scripts.generate_hardest_mode_dynamics")


def _toy_config(hidden_dims: tuple[int, ...]) -> ExperimentConfig:
    return ExperimentConfig(
        sample_count=80,
        epoch_count=2,
        hidden_dim=hidden_dims[-1],
        hidden_dims=hidden_dims,
        circadian_sleep_interval=0,
        random_seed=13,
    )


def _continual_config(hidden_dims: tuple[int, ...]) -> ContinualShiftConfig:
    return ContinualShiftConfig(
        sample_count_phase_a=80,
        sample_count_phase_b=80,
        phase_a_epochs=2,
        phase_b_epochs=2,
        hidden_dim=hidden_dims[-1],
        hidden_dims=hidden_dims,
        phase_b_train_fraction=0.5,
        circadian_sleep_interval_phase_a=0,
        circadian_sleep_interval_phase_b=0,
    )


def _dynamics_config(hidden_dims: tuple[int, ...]) -> Any:
    return dynamics.HardestModeConfig(
        sample_count_phase_a=40,
        sample_count_phase_b=40,
        hidden_dim=hidden_dims[-1],
        hidden_dims=hidden_dims,
        phase_a_epochs=2,
        phase_b_epochs=2,
        phase_b_train_fraction=0.5,
        snapshot_interval=1,
        decision_grid_size=8,
        latency_repeats=1,
        sleep_interval_phase_a=0,
        sleep_interval_phase_b=0,
    )


def test_scope_ids_distinguish_shallow_descriptive_deeper_unmatched_and_unknown() -> None:
    shallow = scope_for_hidden_dims((4,))
    deeper = scope_for_hidden_dims((4, 4))
    unknown = scope_for_hidden_dims(None)

    assert shallow.scope_id == "numpy_shallow_descriptive_v1"
    assert deeper.scope_id == "numpy_deeper_unmatched_descriptive_v1"
    assert unknown.scope_id == "numpy_architecture_unknown_descriptive_v1"
    assert shallow.hidden_depth == 1
    assert deeper.hidden_depth == 2
    assert unknown.hidden_depth is None
    assert not shallow.causal_attribution_supported
    assert not deeper.causal_attribution_supported
    assert "all_latent" in deeper.predictive_algorithm_id
    assert "final_latent" in deeper.circadian_algorithm_id
    assert "unmatched" in deeper.description.lower()
    assert scope_for_hidden_dims(dynamics.HardestModeConfig().hidden_dims).hidden_depth == 3


def test_toy_and_indepth_reports_keep_protocols_and_label_deeper_algorithms() -> None:
    shallow = run_experiment(_toy_config((4,)))
    deeper = run_experiment(_toy_config((4, 4)))
    assert shallow.protocol_id == deeper.protocol_id == "toy_validation_v1"
    assert shallow.comparison_scope.scope_id == "numpy_shallow_descriptive_v1"
    assert deeper.comparison_scope.scope_id == "numpy_deeper_unmatched_descriptive_v1"
    assert "no circadian attribution" in format_experiment_result(deeper).lower()
    assert deeper.comparison_scope.predictive_algorithm_id in format_experiment_result(deeper)

    aggregate = run_indepth_comparison(_toy_config((4, 4)), seeds=[13], noise_levels=[0.8])
    assert aggregate.protocol_id == "toy_validation_v1"
    assert aggregate.comparison_scope == deeper.comparison_scope
    assert "no circadian attribution" in format_indepth_comparison_result(aggregate).lower()


def test_continual_report_keeps_protocol_and_marks_deeper_paths_unmatched() -> None:
    shallow = run_continual_shift_benchmark(_continual_config((4,)), [13])
    deeper = run_continual_shift_benchmark(_continual_config((4, 4)), [13])
    assert shallow.config.protocol_id == deeper.config.protocol_id == "continual_validation_v1"
    assert shallow.comparison_scope.scope_id == "numpy_shallow_descriptive_v1"
    assert deeper.comparison_scope.scope_id == "numpy_deeper_unmatched_descriptive_v1"
    report = format_continual_shift_benchmark(deeper)
    assert "no circadian attribution" in report.lower()
    assert deeper.comparison_scope.circadian_algorithm_id in report


def test_hardest_mode_payload_and_figures_label_deeper_scope(tmp_path: Path) -> None:
    run = dynamics.collect_hardest_mode_snapshots(_dynamics_config((4, 4)))
    assert run.protocol_id == dynamics.VALIDATION_PROTOCOL
    assert run.comparison_scope.scope_id == "numpy_deeper_unmatched_descriptive_v1"
    payload = dynamics.build_interactive_payload(
        snapshots=run.snapshots,
        phase_b_evaluation_input=run.phase_b_evaluation.input,
        phase_b_evaluation_target=run.phase_b_evaluation.target,
        x_bounds=run.x_bounds,
        y_bounds=run.y_bounds,
        evaluation_split=run.evaluation_split,
        protocol_id=run.protocol_id,
        split_hashes=run.split_hashes,
        final_test_accuracy=run.final_test_accuracy,
        comparison_scope=run.comparison_scope,
    )
    assert payload["comparison_scope"]["scope_id"] == run.comparison_scope.scope_id
    assert payload["comparison_scope"]["causal_attribution_supported"] is False
    assert payload["comparison_scope"]["predictive_algorithm_id"] == (
        run.comparison_scope.predictive_algorithm_id
    )

    html_path = tmp_path / "deep.html"
    dynamics.write_interactive_hardest_mode_html(
        snapshots=run.snapshots,
        phase_b_evaluation_input=run.phase_b_evaluation.input,
        phase_b_evaluation_target=run.phase_b_evaluation.target,
        x_bounds=run.x_bounds,
        y_bounds=run.y_bounds,
        output_path=html_path,
        evaluation_split=run.evaluation_split,
        protocol_id=run.protocol_id,
        split_hashes=run.split_hashes,
        final_test_accuracy=run.final_test_accuracy,
        comparison_scope=run.comparison_scope,
    )
    html = html_path.read_text(encoding="utf-8")
    assert "no circadian attribution" in html.lower()
    assert run.comparison_scope.scope_id in html

    gif_path = tmp_path / "deep.gif"
    dynamics.render_hardest_mode_gif(
        snapshots=run.snapshots,
        phase_b_evaluation_input=run.phase_b_evaluation.input,
        phase_b_evaluation_target=run.phase_b_evaluation.target,
        x_bounds=run.x_bounds,
        y_bounds=run.y_bounds,
        output_path=gif_path,
        frame_duration_ms=20,
        evaluation_split=run.evaluation_split,
        comparison_scope=run.comparison_scope,
    )
    with Image.open(gif_path) as gif:
        assert run.comparison_scope.scope_id.encode() in gif.info["comment"]
        assert run.comparison_scope.predictive_algorithm_id.encode() in gif.info["comment"]
        assert run.comparison_scope.circadian_algorithm_id.encode() in gif.info["comment"]


def test_hardest_mode_shallow_scope_is_still_descriptive() -> None:
    config = replace(_dynamics_config((4, 4)), hidden_dims=(4,))
    run = dynamics.collect_hardest_mode_snapshots(config)
    assert run.comparison_scope.scope_id == "numpy_shallow_descriptive_v1"
    assert not run.comparison_scope.causal_attribution_supported
