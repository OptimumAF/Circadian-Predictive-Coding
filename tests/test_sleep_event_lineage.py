"""Executed sleep reports stable neuron IDs around structural changes."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.resnet50_variants import CircadianHeadConfig, CircadianPredictiveCodingHead


def _numpy_model(**changes: Any) -> CircadianPredictiveCodingNetwork:
    options: dict[str, Any] = dict(
        sleep_mode="components",
        sleep_enable_homeostasis=False,
        sleep_enable_chemical_reset=False,
        sleep_enable_replay=False,
        max_split_per_sleep=1,
        max_prune_per_sleep=1,
        split_threshold=0.8,
        prune_threshold=0.2,
        split_weight_norm_mix=0.0,
        split_importance_mix=0.0,
        prune_weight_norm_mix=0.0,
        prune_importance_mix=0.0,
        split_noise_scale=0.0,
    )
    options.update(changes)
    return CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=4,
        seed=611,
        circadian_config=CircadianConfig(**options),
        min_hidden_dim=3,
        max_hidden_dim=6,
    )


def _torch_head(**changes: Any) -> CircadianPredictiveCodingHead:
    torch = pytest.importorskip("torch")
    options: dict[str, Any] = dict(
        sleep_mode="components",
        sleep_enable_homeostasis=False,
        sleep_enable_chemical_reset=False,
        max_split_per_sleep=1,
        max_prune_per_sleep=1,
        split_threshold=0.8,
        prune_threshold=0.2,
        split_weight_norm_mix=0.0,
        split_importance_mix=0.0,
        prune_weight_norm_mix=0.0,
        prune_importance_mix=0.0,
        split_noise_scale=0.0,
    )
    options.update(changes)
    return CircadianPredictiveCodingHead(
        feature_dim=3,
        hidden_dim=4,
        num_classes=2,
        device=torch.device("cpu"),
        seed=613,
        config=CircadianHeadConfig(**options),
        min_hidden_dim=3,
        max_hidden_dim=6,
    )


def test_numpy_event_exposes_immutable_pre_post_ids_for_net_zero_width_change() -> None:
    model = _numpy_model()
    model.set_chemical_state(np.array([1.0, 0.5, 0.5, 0.0]))

    event = model.sleep_event(force_sleep=True)

    assert event.split_indices == (0,)
    assert event.pruned_indices == (3,)
    assert event.old_hidden_dim == event.new_hidden_dim == 4
    assert event.lineage_before is not None and event.lineage_after is not None
    assert event.lineage_before.neuron_ids == (0, 1, 2, 3)
    assert event.lineage_after.neuron_ids == (0, 1, 2, 4)
    assert event.lineage_after.parent_ids == (None, None, None, 0)
    assert event.lineage_after == model.get_neuron_lineage()
    with pytest.raises(FrozenInstanceError):
        setattr(event.lineage_before, "neuron_ids", ())
    model.set_chemical_state(np.array([0.5, 0.5, 0.5, 0.0]))
    model.sleep_event(force_sleep=True)
    assert event.lineage_after.neuron_ids == (0, 1, 2, 4)


def test_numpy_event_shows_unchanged_ids_when_prune_is_only_scheduled() -> None:
    model = _numpy_model(max_split_per_sleep=0, prune_decay_steps=2)
    model.set_chemical_state(np.array([0.5, 0.5, 0.5, 0.0]))

    event = model.sleep_event(force_sleep=True)

    assert event.pruned_indices == (3,)
    assert event.lineage_before == event.lineage_after == model.get_neuron_lineage()
    assert event.new_hidden_dim == 4


@pytest.mark.parametrize("removed_index", [0, 4])
def test_torch_event_exposes_parent_or_child_identity_after_post_split_prune(
    monkeypatch: pytest.MonkeyPatch, removed_index: int
) -> None:
    torch = pytest.importorskip("torch")
    head = _torch_head(prune_threshold=0.6)
    head._chemical = torch.tensor([1.0, 0.9, 0.9, 0.9])
    monkeypatch.setattr(head, "_select_prune_indices", lambda **_: (removed_index,))

    event = head.sleep_event(force_sleep=True)

    assert event.split_indices == (0,)
    assert event.pruned_indices == (removed_index,)
    assert event.lineage_before is not None and event.lineage_after is not None
    assert event.lineage_before.neuron_ids == (0, 1, 2, 3)
    expected = (1, 2, 3, 4) if removed_index == 0 else (0, 1, 2, 3)
    assert event.lineage_after.neuron_ids == expected
    assert event.lineage_after.next_neuron_id == 5
    assert event.lineage_after == head.get_neuron_lineage()


def test_width_changing_torch_event_reports_pre_post_ids() -> None:
    torch = pytest.importorskip("torch")
    head = _torch_head(max_prune_per_sleep=0)
    head._chemical = torch.tensor([1.0, 0.5, 0.5, 0.5])

    event = head.sleep_event(force_sleep=True)

    assert event.old_hidden_dim == 4
    assert event.new_hidden_dim == 5
    assert event.lineage_before is not None and event.lineage_after is not None
    assert event.lineage_before.neuron_ids == (0, 1, 2, 3)
    assert event.lineage_after.neuron_ids == (0, 1, 2, 3, 4)
    assert event.lineage_after.parent_ids[-1] == 0


def test_skipped_numpy_and_torch_events_keep_optional_lineage_empty() -> None:
    numpy_model = _numpy_model(sleep_mode="disabled")
    numpy_event = numpy_model.sleep_event(force_sleep=True)
    torch_head = _torch_head(sleep_mode="disabled")
    torch_event = torch_head.sleep_event(force_sleep=True)

    for event in (numpy_event, torch_event):
        assert not event.performed
        assert event.lineage_before is None
        assert event.lineage_after is None
