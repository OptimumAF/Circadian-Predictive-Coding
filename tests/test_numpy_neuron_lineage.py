"""NumPy adaptive-neuron identities survive index shifts and restoration."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.neuron_adaptation import NeuronChangeProposal


def _model(config: CircadianConfig | None = None) -> CircadianPredictiveCodingNetwork:
    return CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=4,
        seed=521,
        circadian_config=config
        or CircadianConfig(
            max_split_per_sleep=1,
            max_prune_per_sleep=1,
            split_weight_norm_mix=0.0,
            split_importance_mix=0.0,
            split_noise_scale=0.0,
        ),
        min_hidden_dim=3,
        max_hidden_dim=7,
    )


def test_numpy_lineage_tracks_repeated_splits_and_surviving_child_after_prune() -> None:
    model = _model()
    initial = model.get_neuron_lineage()
    assert initial.neuron_ids == (0, 1, 2, 3)
    assert initial.parent_ids == (None, None, None, None)
    assert initial.next_neuron_id == 4

    model.set_chemical_state(np.array([1.0, 0.0, 0.0, 0.0]))
    model.apply_neuron_proposals([NeuronChangeProposal("hidden", add_count=1)])
    first = model.get_neuron_lineage()
    assert first.neuron_ids == (0, 1, 2, 3, 4)
    assert first.parent_ids == (None, None, None, None, 0)

    model.set_chemical_state(np.array([0.0, 0.0, 0.0, 0.0, 1.0]))
    model.apply_neuron_proposals([NeuronChangeProposal("hidden", add_count=1)])
    second = model.get_neuron_lineage()
    assert second.neuron_ids == (0, 1, 2, 3, 4, 5)
    assert second.parent_ids[-1] == 4

    model.apply_neuron_proposals([NeuronChangeProposal("hidden", remove_indices=(1,))])
    assert model.get_neuron_lineage().neuron_ids == (0, 2, 3, 4, 5)
    saved = model.snapshot_state()
    model.apply_neuron_proposals([NeuronChangeProposal("hidden", remove_indices=(3,))])
    after = model.get_neuron_lineage()
    assert after.neuron_ids == (0, 2, 3, 5)
    assert after.parent_ids == (None, None, None, 4)
    assert after.next_neuron_id == 6
    model.set_chemical_state(np.array([1.0, 0.0, 0.0, 0.0]))
    model.apply_neuron_proposals([NeuronChangeProposal("hidden", add_count=1)])
    continued = model.get_neuron_lineage()
    assert continued.neuron_ids == (0, 2, 3, 5, 6)
    assert continued.parent_ids[-1] == 0

    model.restore_state(saved)
    assert model.get_neuron_lineage().neuron_ids == (0, 2, 3, 4, 5)
    model.apply_neuron_proposals([NeuronChangeProposal("hidden", remove_indices=(3,))])
    assert model.get_neuron_lineage() == after
    model.set_chemical_state(np.array([1.0, 0.0, 0.0, 0.0]))
    model.apply_neuron_proposals([NeuronChangeProposal("hidden", add_count=1)])
    assert model.get_neuron_lineage() == continued


def test_numpy_lineage_survives_gradual_prune_finalization() -> None:
    config = CircadianConfig(
        max_split_per_sleep=0,
        max_prune_per_sleep=1,
        prune_decay_steps=2,
        prune_decay_factor=0.5,
    )
    model = _model(config)
    model.apply_neuron_proposals([NeuronChangeProposal("hidden", remove_indices=(2,))])
    assert model.get_neuron_lineage().neuron_ids == (0, 1, 2, 3)
    features = np.array([[0.2, -0.3], [-0.1, 0.4]])
    targets = np.array([[1.0], [0.0]])

    model.train_epoch(features, targets, 0.03, 2, 0.2)
    assert model.get_neuron_lineage().neuron_ids == (0, 1, 2, 3)
    model.train_epoch(features, targets, 0.03, 2, 0.2)
    assert model.get_neuron_lineage().neuron_ids == (0, 1, 3)
    assert model.get_neuron_lineage().next_neuron_id == 4


def test_numpy_snapshot_rejects_duplicate_lineage_without_mutation() -> None:
    model = _model()
    saved = model.snapshot_state()
    broken = replace(saved, state={**saved.state, "_neuron_ids": np.array([0, 0, 2, 3])})

    with pytest.raises(ValueError, match="lineage"):
        model.restore_state(broken)
    assert model.get_neuron_lineage().neuron_ids == (0, 1, 2, 3)
