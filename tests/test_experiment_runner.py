from __future__ import annotations

from src.app.experiment_runner import ExperimentConfig, run_experiment


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
    assert 0.0 <= result.backprop.test_accuracy <= 1.0
    assert 0.0 <= result.predictive_coding.test_accuracy <= 1.0
    assert len(result.backprop.traffic_by_layer) == 1
    assert len(result.predictive_coding.traffic_by_layer) == 1

