"""Adaptive plateau history must not compare diagnostics across changed widths."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.neuron_adaptation import NeuronChangeProposal


def _numpy_split_config(mode: str) -> CircadianConfig:
    return CircadianConfig(
        sleep_mode=mode,
        max_split_per_sleep=1,
        max_prune_per_sleep=0,
        split_threshold=0.8,
        split_noise_scale=0.0,
        use_adaptive_sleep_budget=True,
        use_adaptive_sleep_trigger=True,
        min_epochs_between_sleep=0,
        sleep_energy_window=2,
        sleep_plateau_delta=1e9,
        sleep_chemical_variance_threshold=0.0,
        adaptive_sleep_budget_min_scale=0.25,
    )


def test_numpy_hidden_diagnostic_changes_with_width_for_equal_residual_sum() -> None:
    model = CircadianPredictiveCodingNetwork(2, 4, seed=151, min_hidden_dim=4)
    prediction = np.array([[0.25], [0.75]])
    targets = np.array([[1.0], [0.0]])
    four = np.zeros((2, 4))
    five = np.zeros((2, 5))
    four[0, 0] = five[0, 0] = 0.4
    energy_four = model._compute_energy(prediction, targets, four)
    energy_five = model._compute_energy(prediction, targets, five)
    assert energy_four - energy_five == pytest.approx(0.5 * 0.4**2 * (1 / 8 - 1 / 10))


@pytest.mark.parametrize("mode,expected_history", [("components", []), ("legacy", [1.0, 1.0])])
def test_numpy_split_resets_only_component_history(
    mode: str, expected_history: list[float]
) -> None:
    model = CircadianPredictiveCodingNetwork(
        2,
        4,
        seed=157,
        min_hidden_dim=4,
        max_hidden_dim=6,
        circadian_config=_numpy_split_config(mode),
    )
    model._energy_history = [1.0, 1.0]
    model.set_chemical_state(np.array([0.95, 0.6, 0.4, 0.0]))
    event = model.sleep_event(force_sleep=True)
    assert event.performed is True
    assert event.old_hidden_dim == 4 and event.new_hidden_dim == 5
    assert model._energy_history == expected_history
    assert model.should_trigger_sleep() is (mode == "legacy")
    if mode == "components":
        assert model._compute_adaptive_sleep_budget_scale() == pytest.approx(0.25)


def test_numpy_gradual_prune_restarts_component_history_at_new_width() -> None:
    config = replace(
        _numpy_split_config("components"),
        max_split_per_sleep=0,
        max_prune_per_sleep=1,
        prune_threshold=0.2,
        prune_decay_steps=2,
    )
    model = CircadianPredictiveCodingNetwork(
        2, 4, seed=163, min_hidden_dim=3, circadian_config=config
    )
    model._energy_history = [1.0, 1.0]
    model.set_chemical_state(np.array([0.95, 0.6, 0.4, 0.0]))
    event = model.sleep_event(force_sleep=True)
    assert event.pruned_indices
    assert model.hidden_dim == 4
    assert model._energy_history == [1.0, 1.0]
    features = np.array([[0.3, -0.2], [-0.1, 0.4]])
    targets = np.array([[1.0], [0.0]])
    model.train_epoch(features, targets, 0.03, 2, 0.2)
    assert model.hidden_dim == 4
    model.train_epoch(features, targets, 0.03, 2, 0.2)
    assert model.hidden_dim == 3
    assert len(model._energy_history) == 1
    assert model._compute_adaptive_sleep_budget_scale() == pytest.approx(0.25)


@pytest.mark.parametrize("mode,expected_history", [("components", []), ("legacy", [1.0, 1.0])])
def test_torch_split_resets_only_component_history(
    mode: str, expected_history: list[float]
) -> None:
    torch = pytest.importorskip("torch")
    from src.core.resnet50_variants import CircadianHeadConfig, CircadianPredictiveCodingHead

    config = CircadianHeadConfig(
        sleep_mode=mode,
        max_split_per_sleep=1,
        max_prune_per_sleep=0,
        split_threshold=0.8,
        split_noise_scale=0.0,
        use_adaptive_sleep_budget=True,
        use_adaptive_sleep_trigger=True,
        min_sleep_steps=0,
        sleep_energy_window=2,
        sleep_plateau_delta=1e9,
        sleep_chemical_variance_threshold=0.0,
        adaptive_sleep_budget_min_scale=0.25,
    )
    head = CircadianPredictiveCodingHead(
        2,
        4,
        3,
        torch.device("cpu"),
        seed=167,
        min_hidden_dim=4,
        max_hidden_dim=6,
        config=config,
    )
    head._energy_history = [1.0, 1.0]
    head._chemical = torch.tensor([0.95, 0.6, 0.4, 0.0])
    event = head.sleep_event(force_sleep=True)
    assert event.performed is True
    assert event.old_hidden_dim == 4 and event.new_hidden_dim == 5
    assert head._energy_history == expected_history
    assert head.should_trigger_sleep() is (mode == "legacy")
    if mode == "components":
        assert head._compute_adaptive_sleep_budget_scale() == pytest.approx(0.25)


def test_component_consolidation_without_width_change_preserves_history() -> None:
    config = replace(
        _numpy_split_config("components"),
        max_split_per_sleep=0,
        max_prune_per_sleep=0,
    )
    model = CircadianPredictiveCodingNetwork(
        2, 4, seed=173, min_hidden_dim=4, circadian_config=config
    )
    model._energy_history = [1.0, 1.0]
    event = model.sleep_event(force_sleep=True)
    assert event.performed is True
    assert event.old_hidden_dim == event.new_hidden_dim == 4
    assert model._energy_history == [1.0, 1.0]


def test_numpy_external_topology_proposal_resets_component_history() -> None:
    config = replace(
        _numpy_split_config("components"),
        max_split_per_sleep=0,
        max_prune_per_sleep=1,
        prune_decay_steps=1,
    )
    model = CircadianPredictiveCodingNetwork(
        2, 4, seed=179, min_hidden_dim=3, circadian_config=config
    )
    model._energy_history = [1.0, 1.0]
    model.apply_neuron_proposals([NeuronChangeProposal(layer_name="hidden", remove_indices=(3,))])
    assert model.hidden_dim == 3
    assert model._energy_history == []
