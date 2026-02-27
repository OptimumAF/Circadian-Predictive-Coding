from __future__ import annotations

import numpy as np

from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
    ReplaySnapshot,
)
from src.core.neuron_adaptation import LayerTraffic, NeuronChangeProposal
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
    assert set(sleep_result.split_indices) == {0, 1}
    assert sleep_result.pruned_indices == (2, 5)
    assert model.weight_input_hidden.shape == (2, 6)
    assert model.weight_hidden_output.shape == (6, 1)
    assert model.bias_hidden.shape == (1, 6)
    assert model.get_chemical_state().shape[0] == 6


def test_should_preserve_function_on_split_when_split_noise_is_zero() -> None:
    config = CircadianConfig(
        split_threshold=0.7,
        prune_threshold=-1.0,
        max_split_per_sleep=1,
        max_prune_per_sleep=0,
        split_noise_scale=0.0,
    )
    model = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=5,
        seed=21,
        circadian_config=config,
        min_hidden_dim=4,
        max_hidden_dim=10,
    )
    model.set_chemical_state(np.array([0.95, 0.10, 0.20, 0.30, 0.40], dtype=np.float64))

    probe_input = np.array([[0.2, -0.1], [-0.7, 0.4], [0.5, 0.3]], dtype=np.float64)
    pre_sleep_prediction = model.predict_proba(probe_input)
    sleep_result = model.sleep_event()
    post_sleep_prediction = model.predict_proba(probe_input)

    assert sleep_result.split_indices == (0,)
    assert sleep_result.pruned_indices == ()
    assert model.hidden_dim == 6
    assert np.allclose(pre_sleep_prediction, post_sleep_prediction, atol=1e-10)


def test_should_trigger_adaptive_sleep_when_plateau_and_chemical_variance_are_high() -> None:
    dataset = generate_two_cluster_dataset(sample_count=260, noise_scale=0.8, seed=10)
    config = CircadianConfig(
        use_adaptive_sleep_trigger=True,
        min_epochs_between_sleep=3,
        sleep_energy_window=3,
        sleep_plateau_delta=1.0,
        sleep_chemical_variance_threshold=0.0,
    )
    model = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=8,
        seed=13,
        circadian_config=config,
    )

    for _ in range(3):
        model.train_epoch(
            input_batch=dataset.train_input,
            target_batch=dataset.train_target,
            learning_rate=0.05,
            inference_steps=12,
            inference_learning_rate=0.2,
        )

    assert model.should_trigger_sleep() is True


class _FixedPolicy:
    def propose(self, traffic_by_layer: list[LayerTraffic]) -> list[NeuronChangeProposal]:
        return [
            NeuronChangeProposal(layer_name="hidden", add_count=1, remove_indices=(2,))
        ]


def test_should_apply_external_neuron_adaptation_policy_during_sleep() -> None:
    model = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=6,
        seed=31,
        min_hidden_dim=4,
        max_hidden_dim=10,
    )
    model.set_chemical_state(
        np.array([0.95, 0.72, 0.02, 0.70, 0.60, 0.10], dtype=np.float64)
    )

    sleep_result = model.sleep_event(adaptation_policy=_FixedPolicy())

    assert len(sleep_result.split_indices) == 1
    assert sleep_result.pruned_indices == (2,)
    assert model.hidden_dim == 6


def test_should_gradually_prune_marked_neurons_when_decay_is_enabled() -> None:
    config = CircadianConfig(
        split_threshold=2.0,
        prune_threshold=0.05,
        max_prune_per_sleep=1,
        prune_decay_steps=2,
        prune_decay_factor=0.5,
    )
    model = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=6,
        seed=8,
        circadian_config=config,
        min_hidden_dim=4,
    )
    model.set_chemical_state(
        np.array([0.90, 0.90, 0.01, 0.88, 0.91, 0.92], dtype=np.float64)
    )

    sleep_result = model.sleep_event()
    assert sleep_result.pruned_indices == (2,)
    assert model.hidden_dim == 6

    training_input = np.array([[0.2, -0.4], [0.5, 0.7]], dtype=np.float64)
    training_target = np.array([[0.0], [1.0]], dtype=np.float64)
    model.train_epoch(
        input_batch=training_input,
        target_batch=training_target,
        learning_rate=0.04,
        inference_steps=8,
        inference_learning_rate=0.2,
    )
    assert model.hidden_dim == 6

    model.train_epoch(
        input_batch=training_input,
        target_batch=training_target,
        learning_rate=0.04,
        inference_steps=8,
        inference_learning_rate=0.2,
    )
    assert model.hidden_dim == 5


def test_should_respect_split_cooldown_and_hysteresis_between_sleep_events() -> None:
    config = CircadianConfig(
        split_threshold=0.5,
        split_hysteresis_margin=0.1,
        split_cooldown_epochs=3,
        max_split_per_sleep=1,
        max_prune_per_sleep=0,
        split_noise_scale=0.0,
    )
    model = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=4,
        seed=42,
        circadian_config=config,
        min_hidden_dim=3,
        max_hidden_dim=8,
    )
    model.set_chemical_state(np.array([0.9, 0.2, 0.1, 0.1], dtype=np.float64))

    first_sleep = model.sleep_event()
    second_sleep = model.sleep_event()

    assert first_sleep.split_indices == (0,)
    assert second_sleep.split_indices == ()


def test_should_select_balanced_high_priority_replay_snapshots_when_enabled() -> None:
    config = CircadianConfig(
        replay_steps=2,
        replay_memory_size=6,
        replay_prioritized=True,
        replay_class_balanced=True,
    )
    model = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=4,
        seed=12,
        circadian_config=config,
    )
    model._replay_memory.clear()
    model._replay_memory.extend(
        [
            ReplaySnapshot(
                input_batch=np.zeros((2, 2), dtype=np.float64),
                target_batch=np.zeros((2, 1), dtype=np.float64),
                priority=0.20,
                positive_fraction=0.0,
            ),
            ReplaySnapshot(
                input_batch=np.zeros((2, 2), dtype=np.float64),
                target_batch=np.ones((2, 1), dtype=np.float64),
                priority=0.95,
                positive_fraction=1.0,
            ),
            ReplaySnapshot(
                input_batch=np.zeros((2, 2), dtype=np.float64),
                target_batch=np.zeros((2, 1), dtype=np.float64),
                priority=0.90,
                positive_fraction=0.0,
            ),
        ]
    )

    chosen = model._select_replay_snapshots(replay_count=2)

    assert len(chosen) == 2
    assert {snapshot.positive_fraction for snapshot in chosen} == {0.0, 1.0}


def test_should_keep_chemical_bounded_with_saturating_updates() -> None:
    config = CircadianConfig(
        use_saturating_chemical=True,
        chemical_max_value=0.5,
        chemical_saturation_gain=4.0,
        use_dual_chemical=False,
    )
    model = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=4,
        seed=33,
        circadian_config=config,
    )
    dataset = generate_two_cluster_dataset(sample_count=220, noise_scale=0.9, seed=10)

    for _ in range(120):
        model.train_epoch(
            input_batch=dataset.train_input,
            target_batch=dataset.train_target,
            learning_rate=0.05,
            inference_steps=12,
            inference_learning_rate=0.2,
        )

    chemical = model.get_chemical_state()
    assert float(np.max(chemical)) <= 0.5000001


def test_should_reduce_plasticity_for_high_importance_when_adaptive_sensitivity_is_enabled() -> None:
    config = CircadianConfig(
        use_adaptive_plasticity_sensitivity=True,
        plasticity_sensitivity_min=0.2,
        plasticity_sensitivity_max=1.0,
        plasticity_importance_mix=1.0,
    )
    model = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=4,
        seed=44,
        circadian_config=config,
    )
    model.set_chemical_state(np.array([0.4, 0.4, 0.4, 0.4], dtype=np.float64))
    model._importance_ema = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    plasticity = model.get_plasticity_state()

    assert float(plasticity[0]) < float(plasticity[1])
