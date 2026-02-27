from __future__ import annotations

from src.app.experiment_runner import ExperimentConfig
from src.app.indepth_comparison import (
    format_indepth_comparison_result,
    run_indepth_comparison,
)


def test_should_run_indepth_comparison_and_return_aggregate_stats() -> None:
    config = ExperimentConfig(
        sample_count=220,
        epoch_count=50,
        hidden_dim=8,
        circadian_sleep_interval=25,
    )
    result = run_indepth_comparison(
        base_config=config,
        seeds=[3, 7],
        noise_levels=[0.7, 1.0],
    )

    assert result.seeds == [3, 7]
    assert len(result.scenario_reports) == 2

    first_scenario = result.scenario_reports[0]
    assert first_scenario.run_count == 2
    assert 0.0 <= first_scenario.backprop.mean_test_accuracy <= 1.0
    assert 0.0 <= first_scenario.predictive_coding.mean_test_accuracy <= 1.0
    assert 0.0 <= first_scenario.circadian_predictive_coding.mean_test_accuracy <= 1.0
    assert first_scenario.circadian_predictive_coding.mean_hidden_dim_end >= 4.0

    report_text = format_indepth_comparison_result(result)
    assert "In-Depth Model Comparison" in report_text
    assert "Circadian predictive coding" in report_text

