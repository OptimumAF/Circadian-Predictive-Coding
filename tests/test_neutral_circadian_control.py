"""A shallow no-circadian control paired with its ordinary PC baseline."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork


def _binary_batches() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    return (
        (
            np.array([[0.2, -0.3], [-0.4, 0.5]], dtype=np.float64),
            np.array([[1.0], [0.0]], dtype=np.float64),
        ),
        (
            np.array([[0.6, 0.1], [-0.2, -0.3]], dtype=np.float64),
            np.array([[0.0], [1.0]], dtype=np.float64),
        ),
    )


def _assert_numpy_shallow_parameters_equal(
    ordinary: PredictiveCodingNetwork, circadian: CircadianPredictiveCodingNetwork
) -> None:
    assert ordinary.hidden_dims == circadian.hidden_dims == (3,)
    for field in (
        "weight_input_hidden",
        "bias_hidden",
        "weight_hidden_output",
        "bias_output",
    ):
        np.testing.assert_array_equal(getattr(ordinary, field), getattr(circadian, field))


def test_numpy_neutral_circadian_matches_shallow_pc_across_repeated_wake_and_noop_sleep() -> None:
    ordinary = PredictiveCodingNetwork(input_dim=2, hidden_dim=3, seed=97)
    circadian = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=3,
        seed=97,
        min_hidden_dim=3,
        circadian_config=CircadianConfig.matched_pc_control(),
    )
    _assert_numpy_shallow_parameters_equal(ordinary, circadian)
    batches = _binary_batches()
    for epoch in range(4):
        features, targets = batches[epoch % len(batches)]
        ordinary_result = ordinary.train_epoch(features, targets, 0.03, 4, 0.2)
        circadian_result = circadian.train_epoch(features, targets, 0.03, 4, 0.2)
        _assert_numpy_shallow_parameters_equal(ordinary, circadian)
        np.testing.assert_array_equal(
            ordinary.predict_proba(features), circadian.predict_proba(features)
        )
        np.testing.assert_array_equal(
            ordinary.get_layer_traffic()[0].mean_abs_activation,
            circadian.get_layer_traffic()[0].mean_abs_activation,
        )
        np.testing.assert_array_equal(circadian.get_plasticity_state(), np.ones(3))
        assert circadian.get_last_reward_scale() == 1.0
        assert len(circadian._replay_memory) == 0
        assert ordinary_result.energy_definition != circadian_result.energy_definition
        assert ordinary_result.energy == circadian_result.energy

        before = circadian.weight_hidden_output.copy()
        sleep = circadian.sleep_event(force_sleep=True)
        assert sleep.old_hidden_dim == sleep.new_hidden_dim == 3
        assert sleep.split_indices == sleep.pruned_indices == ()
        np.testing.assert_array_equal(circadian.weight_hidden_output, before)
    assert np.any(circadian.get_chemical_state() > 0.0)


def test_torch_neutral_circadian_matches_shallow_pc_across_repeated_wake_and_noop_sleep() -> None:
    torch = pytest.importorskip("torch")
    from src.core.resnet50_variants import (
        CircadianHeadConfig,
        CircadianPredictiveCodingHead,
        PredictiveCodingHead,
    )

    ordinary = PredictiveCodingHead(2, 3, 3, torch.device("cpu"), seed=101)
    circadian = CircadianPredictiveCodingHead(
        feature_dim=2,
        hidden_dim=3,
        num_classes=3,
        device=torch.device("cpu"),
        seed=101,
        config=CircadianHeadConfig.matched_pc_control(),
        min_hidden_dim=3,
        max_hidden_dim=6,
    )
    fields = (
        "weight_feature_hidden",
        "bias_hidden",
        "weight_hidden_output",
        "bias_output",
    )

    def assert_matched() -> None:
        for field in fields:
            torch.testing.assert_close(
                getattr(ordinary, field), getattr(circadian, field), atol=0, rtol=0
            )

    assert_matched()
    batches = (
        (torch.tensor([[0.2, -0.3], [-0.4, 0.5]]), torch.tensor([0, 2])),
        (torch.tensor([[0.6, 0.1], [-0.2, -0.3]]), torch.tensor([1, 0])),
    )
    for epoch in range(4):
        features, targets = batches[epoch % len(batches)]
        ordinary_energy = ordinary.train_step(features, targets, 0.03, 4, 0.2)
        circadian_energy = circadian.train_step(features, targets, 0.03, 4, 0.2)
        assert_matched()
        torch.testing.assert_close(
            ordinary.predict_logits(features), circadian.predict_logits(features), atol=0, rtol=0
        )
        torch.testing.assert_close(
            ordinary.mean_hidden_traffic(), circadian.mean_hidden_traffic(), atol=0, rtol=0
        )
        torch.testing.assert_close(circadian._plasticity(), torch.ones(3), atol=0, rtol=0)
        assert circadian.last_reward_scale() == 1.0
        assert ordinary_energy == circadian_energy

        before = circadian.weight_hidden_output.clone()
        sleep = circadian.sleep_event(force_sleep=True)
        assert sleep.old_hidden_dim == sleep.new_hidden_dim == 3
        assert sleep.split_indices == sleep.pruned_indices == ()
        torch.testing.assert_close(circadian.weight_hidden_output, before, atol=0, rtol=0)
    assert bool(torch.any(circadian._chemical > 0).item())


def test_multilayer_numpy_paths_are_not_the_same_neutral_control() -> None:
    features, targets = _binary_batches()[0]
    ordinary = PredictiveCodingNetwork(input_dim=2, hidden_dim=2, hidden_dims=(2, 3), seed=103)
    circadian = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=3,
        hidden_dims=(2, 3),
        seed=103,
        min_hidden_dim=3,
        circadian_config=CircadianConfig.matched_pc_control(),
    )
    np.testing.assert_array_equal(ordinary._hidden_weights[0], circadian._pre_hidden_weights[0])
    np.testing.assert_array_equal(ordinary._hidden_weights[1], circadian.weight_input_hidden)
    np.testing.assert_array_equal(ordinary.weight_hidden_output, circadian.weight_hidden_output)
    ordinary.train_epoch(features, targets, 0.03, 3, 0.2)
    circadian.train_epoch(features, targets, 0.03, 3, 0.2)
    assert not np.allclose(
        ordinary._hidden_weights[0], circadian._pre_hidden_weights[0], atol=1e-12
    )
