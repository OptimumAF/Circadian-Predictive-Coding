"""External structural requests are checked before NumPy state changes."""

from __future__ import annotations

from collections import deque
from dataclasses import fields, is_dataclass, replace
from typing import Any

import numpy as np
import pytest

from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.neuron_adaptation import LayerTraffic, NeuronChangeProposal


def _model(
    config: CircadianConfig | None = None,
    *,
    min_hidden_dim: int = 3,
    max_hidden_dim: int = 5,
) -> CircadianPredictiveCodingNetwork:
    model = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=4,
        seed=421,
        circadian_config=config
        or CircadianConfig(
            max_split_per_sleep=2,
            max_prune_per_sleep=2,
            split_weight_norm_mix=0.0,
            split_importance_mix=0.0,
            split_noise_scale=0.1,
        ),
        min_hidden_dim=min_hidden_dim,
        max_hidden_dim=max_hidden_dim,
    )
    model.set_chemical_state(np.array([0.95, 0.9, 0.2, 0.1]))
    return model


def _assert_equal(left: Any, right: Any) -> None:
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, np.random.Generator):
        assert left.bit_generator.state == right.bit_generator.state
    elif isinstance(left, (dict, list, tuple, deque)):
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                _assert_equal(left[key], right[key])
        else:
            assert len(left) == len(right)
            for old, new in zip(left, right):
                _assert_equal(old, new)
    elif is_dataclass(left):
        for field in fields(left):
            _assert_equal(getattr(left, field.name), getattr(right, field.name))
    else:
        assert left == right


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("invalid_index", "out of range"),
        ("duplicate_index", "unique"),
        ("negative_count", "nonnegative integer"),
        ("split_cap", "budget"),
        ("max_width", "maximum hidden width"),
        ("min_width", "minimum hidden width"),
        ("fraction", "budget"),
        ("split_cooldown", "eligible split"),
        ("prune_cooldown", "cooldown"),
        ("prune_age", "age"),
        ("unknown_layer", "layer_name"),
    ],
)
def test_external_proposal_rejects_invalid_request_without_mutation(
    case: str, message: str
) -> None:
    config = CircadianConfig(
        max_split_per_sleep=2,
        max_prune_per_sleep=2,
        split_weight_norm_mix=0.0,
        split_importance_mix=0.0,
        split_noise_scale=0.1,
    )
    if case == "fraction":
        config = replace(config, sleep_max_change_fraction=0.25)
    if case == "prune_age":
        config = replace(config, prune_min_age_steps=2)
    model = (
        _model(config, min_hidden_dim=2, max_hidden_dim=6) if case == "fraction" else _model(config)
    )
    proposals = {
        "invalid_index": NeuronChangeProposal("hidden", remove_indices=(99,)),
        "duplicate_index": NeuronChangeProposal("hidden", remove_indices=(3, 3)),
        "negative_count": NeuronChangeProposal("hidden", add_count=-1),
        "split_cap": NeuronChangeProposal("hidden", add_count=3),
        "max_width": NeuronChangeProposal("hidden", add_count=2),
        "min_width": NeuronChangeProposal("hidden", remove_indices=(2, 3)),
        "fraction": NeuronChangeProposal("hidden", remove_indices=(2, 3)),
        "split_cooldown": NeuronChangeProposal("hidden", add_count=1),
        "prune_cooldown": NeuronChangeProposal("hidden", remove_indices=(3,)),
        "prune_age": NeuronChangeProposal("hidden", remove_indices=(3,)),
        "unknown_layer": NeuronChangeProposal("output", add_count=1),
    }
    if case == "split_cooldown":
        model._split_cooldown[:] = 2
    if case == "prune_cooldown":
        model._prune_cooldown[3] = 2
    before = model.snapshot_state()

    with pytest.raises(ValueError, match=f"proposal.*{message}"):
        model.apply_neuron_proposals([proposals[case]])
    _assert_equal(model.snapshot_state(), before)


class _FixedPolicy:
    def __init__(self, proposal: NeuronChangeProposal) -> None:
        self.proposal = proposal

    def propose(self, traffic_by_layer: list[LayerTraffic]) -> list[NeuronChangeProposal]:
        return [self.proposal]


def test_policy_prune_request_takes_priority_over_same_split_source() -> None:
    model = _model()
    result = model.sleep_event(
        adaptation_policy=_FixedPolicy(
            NeuronChangeProposal("hidden", add_count=1, remove_indices=(0,))
        )
    )
    assert result.split_indices == (1,)
    assert result.pruned_indices == (0,)
    assert model.hidden_dim == 4


def test_policy_budget_rejection_leaves_state_unchanged() -> None:
    model = _model()
    before = model.snapshot_state()
    with pytest.raises(ValueError, match="proposal"):
        model.sleep_event(
            adaptation_policy=_FixedPolicy(NeuronChangeProposal("hidden", add_count=3))
        )
    _assert_equal(model.snapshot_state(), before)
