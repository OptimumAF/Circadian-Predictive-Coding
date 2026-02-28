from __future__ import annotations

from src.core.predictive_coding import PredictiveCodingNetwork
from src.infra.datasets import generate_two_cluster_dataset


def test_should_reduce_energy_and_learn_with_predictive_coding() -> None:
    dataset = generate_two_cluster_dataset(sample_count=300, noise_scale=0.7, seed=3)
    model = PredictiveCodingNetwork(input_dim=2, hidden_dim=10, seed=11)

    first_energy = model.train_epoch(
        input_batch=dataset.train_input,
        target_batch=dataset.train_target,
        learning_rate=0.05,
        inference_steps=25,
        inference_learning_rate=0.2,
    ).energy
    latest_energy = first_energy

    for _ in range(140):
        latest_energy = model.train_epoch(
            input_batch=dataset.train_input,
            target_batch=dataset.train_target,
            learning_rate=0.05,
            inference_steps=25,
            inference_learning_rate=0.2,
        ).energy

    accuracy = model.compute_accuracy(dataset.test_input, dataset.test_target)
    assert latest_energy < first_energy
    assert accuracy >= 0.75


def test_should_support_multi_hidden_layer_predictive_coding() -> None:
    dataset = generate_two_cluster_dataset(sample_count=320, noise_scale=0.75, seed=14)
    model = PredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=8,
        hidden_dims=[12, 8],
        seed=18,
    )

    for _ in range(150):
        model.train_epoch(
            input_batch=dataset.train_input,
            target_batch=dataset.train_target,
            learning_rate=0.05,
            inference_steps=20,
            inference_learning_rate=0.2,
        )

    accuracy = model.compute_accuracy(dataset.test_input, dataset.test_target)
    assert accuracy >= 0.72
