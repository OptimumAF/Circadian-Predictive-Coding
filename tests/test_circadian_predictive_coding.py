from __future__ import annotations

import numpy as np

from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.infra.datasets import generate_two_cluster_dataset


def test_should_build_chemical_and_learn_with_circadian_predictive_coding() -> None:
    dataset = generate_two_cluster_dataset(sample_count=320, noise_scale=0.75, seed=9)
    model = CircadianPredictiveCodingNetwork(input_dim=2, hidden_dim=10, seed=4)

    first_energy = model.train_epoch(
        input_batch=dataset.train_input,
        target_batch=dataset.train_target,
        learning_rate=0.05,
        inference_steps=20,
        inference_learning_rate=0.2,
    ).energy
    latest_energy = first_energy

    for _ in range(100):
        latest_energy = model.train_epoch(
            input_batch=dataset.train_input,
            target_batch=dataset.train_target,
            learning_rate=0.05,
            inference_steps=20,
            inference_learning_rate=0.2,
        ).energy

    chemical_state = model.get_chemical_state()
    plasticity_state = model.get_plasticity_state()
    accuracy = model.compute_accuracy(dataset.test_input, dataset.test_target)

    assert latest_energy < first_energy
    assert accuracy >= 0.72
    assert float(np.max(chemical_state)) > 0.0
    assert float(np.min(plasticity_state)) < 1.0


def test_should_split_busy_neurons_and_prune_idle_neurons_during_sleep() -> None:
    config = CircadianConfig(
        split_threshold=0.75,
        prune_threshold=0.12,
        max_split_per_sleep=2,
        max_prune_per_sleep=2,
        sleep_reset_factor=0.5,
    )
    model = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=6,
        seed=5,
        circadian_config=config,
        min_hidden_dim=4,
        max_hidden_dim=12,
    )
    model.set_chemical_state(
        np.array([0.95, 0.81, 0.04, 0.05, 0.60, 0.02], dtype=np.float64)
    )

    sleep_result = model.sleep_event()

    assert sleep_result.old_hidden_dim == 6
    assert sleep_result.new_hidden_dim == 6
    assert sleep_result.split_indices == (0, 1)
    assert sleep_result.pruned_indices == (2, 5)
    assert model.weight_input_hidden.shape == (2, 6)
    assert model.weight_hidden_output.shape == (6, 1)
    assert model.bias_hidden.shape == (1, 6)
    assert model.get_chemical_state().shape[0] == 6
