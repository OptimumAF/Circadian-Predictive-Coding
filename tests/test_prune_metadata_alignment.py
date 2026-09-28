"""Pruning keeps adaptive metadata aligned with surviving neurons."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.resnet50_variants import CircadianHeadConfig, CircadianPredictiveCodingHead


NUMPY_VECTOR_FIELDS = (
    "_hidden_chemical",
    "_hidden_chemical_fast",
    "_hidden_chemical_slow",
    "_importance_ema",
    "_traffic_sum",
    "_neuron_age",
    "_split_cooldown",
    "_prune_cooldown",
    "_prune_ttl",
    "_prune_marked",
    "_neuron_ids",
    "_parent_ids",
)
TORCH_VECTOR_FIELDS = (
    "_chemical",
    "_chemical_fast",
    "_chemical_slow",
    "_importance_ema",
    "_traffic_sum",
    "_neuron_age",
    "_split_cooldown",
    "_prune_cooldown",
    "_neuron_ids",
    "_parent_ids",
)


def _numpy_model(
    *, width: int = 5, minimum: int = 3, **changes: Any
) -> CircadianPredictiveCodingNetwork:
    options: dict[str, Any] = dict(
        sleep_mode="components",
        sleep_enable_homeostasis=False,
        sleep_enable_chemical_reset=False,
        sleep_enable_replay=False,
        max_split_per_sleep=0,
        max_prune_per_sleep=2,
        prune_threshold=0.2,
        prune_weight_norm_mix=0.0,
        prune_importance_mix=0.0,
    )
    options.update(changes)
    return CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=width,
        seed=809,
        circadian_config=CircadianConfig(**options),
        min_hidden_dim=minimum,
        max_hidden_dim=7,
    )


def _torch_head(*, width: int = 5, minimum: int = 3) -> CircadianPredictiveCodingHead:
    torch = pytest.importorskip("torch")
    return CircadianPredictiveCodingHead(
        feature_dim=3,
        hidden_dim=width,
        num_classes=2,
        device=torch.device("cpu"),
        seed=811,
        config=CircadianHeadConfig(
            sleep_mode="components",
            sleep_enable_homeostasis=False,
            sleep_enable_chemical_reset=False,
            max_split_per_sleep=0,
            max_prune_per_sleep=2,
            prune_threshold=0.2,
            prune_weight_norm_mix=0.0,
            prune_importance_mix=0.0,
        ),
        min_hidden_dim=minimum,
        max_hidden_dim=7,
    )


def test_numpy_immediate_prune_masks_every_adaptive_array_and_downstream_width() -> None:
    model = _numpy_model()
    model.set_chemical_state(np.array([0.5, 0.0, 0.5, 0.0, 0.5]))
    model._hidden_chemical_fast[:] = np.arange(5) + 11.0
    model._hidden_chemical_slow[:] = np.arange(5) + 21.0
    model._importance_ema[:] = np.arange(5) + 31.0
    model._traffic_sum[:] = np.arange(5) + 41.0
    model._neuron_age[:] = np.arange(5) + 51.0
    model._split_cooldown[:] = np.arange(5) + 1
    model._prune_cooldown[:] = np.array([3, 0, 2, 0, 1])
    before = {name: getattr(model, name).copy() for name in NUMPY_VECTOR_FIELDS}
    weights_before = model.weight_input_hidden.copy()
    outputs_before = model.weight_hidden_output.copy()
    bias_before = model.bias_hidden.copy()

    event = model.sleep_event(force_sleep=True)

    assert event.pruned_indices == (1, 3)
    assert model.get_neuron_lineage().neuron_ids == (0, 2, 4)
    survivors = [0, 2, 4]
    for name in NUMPY_VECTOR_FIELDS:
        np.testing.assert_array_equal(getattr(model, name), before[name][survivors], err_msg=name)
    np.testing.assert_array_equal(model.weight_input_hidden, weights_before[:, survivors])
    np.testing.assert_array_equal(model.weight_hidden_output, outputs_before[survivors, :])
    np.testing.assert_array_equal(model.bias_hidden, bias_before[:, survivors])
    assert model.weight_hidden_output.shape == (3, 1)
    assert model.predict_proba(np.array([[0.2, -0.3]])).shape == (1, 1)
    assert model.get_sleep_clocks().sleep_events == 1
    assert model.get_sleep_clocks().wake_batches == 0


def test_numpy_gradual_prune_reserves_capacity_then_finalizes_during_wake() -> None:
    model = _numpy_model(
        minimum=4,
        max_prune_per_sleep=1,
        prune_decay_steps=2,
        prune_decay_factor=0.5,
        prune_cooldown_epochs=3,
    )
    model.set_chemical_state(np.array([0.5, 0.5, 0.0, 0.5, 0.5]))

    scheduled = model.sleep_event(force_sleep=True)

    assert scheduled.pruned_indices == (2,)
    assert model.hidden_dim == 5
    assert model._prune_marked.tolist() == [False, False, True, False, False]
    assert model._prune_ttl.tolist() == [0, 0, 2, 0, 0]
    assert model._prune_cooldown[2] == 3
    model.set_chemical_state(np.array([0.5, 0.5, 0.0, 0.0, 0.5]))
    assert model.sleep_event(force_sleep=True).pruned_indices == ()
    assert model._prune_marked.sum() == 1

    features = np.array([[0.2, -0.3], [-0.1, 0.4]])
    targets = np.array([[1.0], [0.0]])
    model.train_epoch(features, targets, 0.03, 2, 0.2)
    assert model.hidden_dim == 5
    assert model._prune_ttl[2] == 1
    model.train_epoch(features, targets, 0.03, 2, 0.2)

    assert model.hidden_dim == 4
    assert model.get_neuron_lineage().neuron_ids == (0, 1, 3, 4)
    assert not np.any(model._prune_marked)
    assert not np.any(model._prune_ttl)
    for name in NUMPY_VECTOR_FIELDS:
        assert getattr(model, name).shape == (4,), name
    assert model.weight_input_hidden.shape == (2, 4)
    assert model.weight_hidden_output.shape == (4, 1)
    assert model.bias_hidden.shape == (1, 4)
    assert model.predict_proba(features).shape == (2, 1)
    clocks = model.get_sleep_clocks()
    assert clocks.sleep_events == 2
    assert clocks.wake_batches == 2
    assert clocks.wake_examples == 4
    model.set_chemical_state(np.zeros(4))
    assert model.sleep_event(force_sleep=True).pruned_indices == ()
    assert model.hidden_dim == 4


def test_torch_immediate_prune_masks_every_adaptive_array_and_downstream_width() -> None:
    torch = pytest.importorskip("torch")
    head = _torch_head()
    head._chemical = torch.tensor([0.5, 0.0, 0.5, 0.0, 0.5])
    for name, offset in (
        ("_chemical_fast", 11.0),
        ("_chemical_slow", 21.0),
        ("_importance_ema", 31.0),
        ("_traffic_sum", 41.0),
        ("_neuron_age", 51.0),
    ):
        getattr(head, name)[:] = torch.arange(5, dtype=torch.float32) + offset
    head._split_cooldown[:] = torch.arange(5, dtype=torch.int32) + 1
    head._prune_cooldown[:] = torch.tensor([3, 0, 2, 0, 1], dtype=torch.int32)
    before = {name: getattr(head, name).clone() for name in TORCH_VECTOR_FIELDS}
    weights_before = head.weight_feature_hidden.clone()
    outputs_before = head.weight_hidden_output.clone()
    bias_before = head.bias_hidden.clone()

    event = head.sleep_event(force_sleep=True)

    assert event.pruned_indices == (1, 3)
    assert head.get_neuron_lineage().neuron_ids == (0, 2, 4)
    survivors = [0, 2, 4]
    for name in TORCH_VECTOR_FIELDS:
        assert torch.equal(getattr(head, name), before[name][survivors]), name
    assert torch.equal(head.weight_feature_hidden, weights_before[:, survivors])
    assert torch.equal(head.weight_hidden_output, outputs_before[survivors, :])
    assert torch.equal(head.bias_hidden, bias_before[:, survivors])
    assert head.weight_hidden_output.shape == (3, 2)
    assert head.predict_logits(torch.tensor([[0.2, -0.3, 0.4]])).shape == (1, 2)
    assert head.get_sleep_clocks().sleep_events == 1
    assert head.get_sleep_clocks().wake_batches == 0


def test_torch_minimum_width_prevents_prune_without_advancing_wake_clock() -> None:
    torch = pytest.importorskip("torch")
    head = _torch_head(width=4, minimum=4)
    head._chemical = torch.zeros(4)
    before = head.get_neuron_lineage()

    event = head.sleep_event(force_sleep=True)

    assert event.performed
    assert event.pruned_indices == ()
    assert head.get_neuron_lineage() == before
    assert head.weight_hidden_output.shape == (4, 2)
    assert head.get_sleep_clocks().sleep_events == 1
    assert head.get_sleep_clocks().wake_batches == 0
