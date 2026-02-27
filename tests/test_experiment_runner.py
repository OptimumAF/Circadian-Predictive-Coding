from __future__ import annotations

from src.app.experiment_runner import ExperimentConfig, run_experiment
from src.core.circadian_predictive_coding import CircadianConfig


def test_should_run_experiment_and_return_reports() -> None:
    config = ExperimentConfig(
        sample_count=220,
        noise_scale=0.8,
        hidden_dim=8,
        epoch_count=60,
        random_seed=5,
    )
    result = run_experiment(config)

    assert len(result.backprop.loss_history) == config.epoch_count
    assert len(result.predictive_coding.loss_history) == config.epoch_count
    assert len(result.circadian_predictive_coding.loss_history) == config.epoch_count
    assert 0.0 <= result.backprop.test_accuracy <= 1.0
    assert 0.0 <= result.predictive_coding.test_accuracy <= 1.0
    assert 0.0 <= result.circadian_predictive_coding.test_accuracy <= 1.0
    assert len(result.backprop.traffic_by_layer) == 1
    assert len(result.predictive_coding.traffic_by_layer) == 1
    assert len(result.circadian_predictive_coding.traffic_by_layer) == 2
    assert result.circadian_sleep.hidden_dim_start == config.hidden_dim
    assert result.circadian_sleep.hidden_dim_end >= 4


def test_should_support_adaptive_circadian_configuration_in_experiment() -> None:
    config = ExperimentConfig(
        sample_count=220,
        noise_scale=0.8,
        hidden_dim=8,
        epoch_count=40,
        circadian_sleep_interval=5,
        circadian_force_sleep=False,
        circadian_config=CircadianConfig(
            use_adaptive_thresholds=True,
            adaptive_split_percentile=80.0,
            adaptive_prune_percentile=20.0,
            use_adaptive_sleep_trigger=True,
            min_epochs_between_sleep=3,
            sleep_energy_window=3,
            sleep_plateau_delta=1.0,
            sleep_chemical_variance_threshold=0.0,
            replay_steps=1,
            replay_memory_size=4,
        ),
        random_seed=9,
    )
    result = run_experiment(config)

    assert len(result.circadian_predictive_coding.loss_history) == config.epoch_count
    assert 0.0 <= result.circadian_predictive_coding.test_accuracy <= 1.0
    assert result.circadian_sleep.hidden_dim_end >= 4
