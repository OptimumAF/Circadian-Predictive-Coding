from __future__ import annotations

import pytest

from src.app.continual_shift_benchmark import (
    ContinualShiftConfig,
    format_continual_shift_benchmark,
    run_continual_shift_benchmark,
)
from src.core.circadian_predictive_coding import CircadianConfig


def test_should_run_continual_shift_benchmark_and_return_metrics() -> None:
    config = ContinualShiftConfig(
        sample_count_phase_a=180,
        sample_count_phase_b=180,
        phase_b_train_fraction=0.20,
        hidden_dim=8,
        phase_a_epochs=25,
        phase_b_epochs=20,
        circadian_sleep_interval_phase_a=10,
        circadian_sleep_interval_phase_b=4,
        circadian_config=CircadianConfig(
            split_threshold=0.35,
            max_split_per_sleep=1,
            max_prune_per_sleep=0,
            replay_steps=1,
            replay_memory_size=4,
        ),
    )
    result = run_continual_shift_benchmark(config=config, seeds=[3, 7])

    assert result.seeds == [3, 7]
    assert len(result.seed_results) == 2
    assert result.aggregate.run_count == 2

    for seed_result in result.seed_results:
        assert 0.0 <= seed_result.backprop.phase_a_pre_accuracy <= 1.0
        assert 0.0 <= seed_result.backprop.phase_a_post_accuracy <= 1.0
        assert 0.0 <= seed_result.backprop.phase_b_post_accuracy <= 1.0
        assert seed_result.backprop.retention_ratio >= 0.0
        assert 0.0 <= seed_result.backprop.balanced_score <= 1.0

        assert 0.0 <= seed_result.predictive_coding.phase_a_pre_accuracy <= 1.0
        assert 0.0 <= seed_result.predictive_coding.phase_a_post_accuracy <= 1.0
        assert 0.0 <= seed_result.predictive_coding.phase_b_post_accuracy <= 1.0
        assert seed_result.predictive_coding.retention_ratio >= 0.0
        assert 0.0 <= seed_result.predictive_coding.balanced_score <= 1.0

        assert 0.0 <= seed_result.circadian_predictive_coding.phase_a_pre_accuracy <= 1.0
        assert 0.0 <= seed_result.circadian_predictive_coding.phase_a_post_accuracy <= 1.0
        assert 0.0 <= seed_result.circadian_predictive_coding.phase_b_post_accuracy <= 1.0
        assert seed_result.circadian_predictive_coding.retention_ratio >= 0.0
        assert 0.0 <= seed_result.circadian_predictive_coding.balanced_score <= 1.0
        assert seed_result.circadian_predictive_coding.hidden_dim_start == config.hidden_dim
        assert seed_result.circadian_predictive_coding.hidden_dim_end >= 4

    report_text = format_continual_shift_benchmark(result)
    assert "Continual Shift Benchmark" in report_text
    assert "Circadian predictive coding" in report_text


def test_should_fail_when_seeds_are_empty() -> None:
    with pytest.raises(ValueError, match="seeds cannot be empty"):
        run_continual_shift_benchmark(config=ContinualShiftConfig(), seeds=[])


def test_should_fail_for_invalid_phase_b_train_fraction() -> None:
    with pytest.raises(ValueError, match="phase_b_train_fraction"):
        run_continual_shift_benchmark(
            config=ContinualShiftConfig(phase_b_train_fraction=1.2),
            seeds=[7],
        )
