from __future__ import annotations

from src.core.backprop_mlp import BackpropMLP
from src.infra.datasets import generate_two_cluster_dataset


def test_should_improve_loss_and_accuracy_with_backprop() -> None:
    dataset = generate_two_cluster_dataset(sample_count=300, noise_scale=0.7, seed=3)
    model = BackpropMLP(input_dim=2, hidden_dim=10, seed=9)

    first_loss = model.train_epoch(
        input_batch=dataset.train_input,
        target_batch=dataset.train_target,
        learning_rate=0.12,
    ).loss
    latest_loss = first_loss

    for _ in range(140):
        latest_loss = model.train_epoch(
            input_batch=dataset.train_input,
            target_batch=dataset.train_target,
            learning_rate=0.12,
        ).loss

    accuracy = model.compute_accuracy(dataset.test_input, dataset.test_target)
    assert latest_loss < first_loss
    assert accuracy >= 0.85

