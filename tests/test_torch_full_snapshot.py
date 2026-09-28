"""Detached Torch head snapshots preserve adaptive state and future split draws."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from src.core.resnet50_variants import (  # noqa: E402
    CircadianHeadConfig,
    CircadianPredictiveCodingHead,
)


def _head(config: CircadianHeadConfig | None = None) -> CircadianPredictiveCodingHead:
    return CircadianPredictiveCodingHead(
        feature_dim=3,
        hidden_dim=4,
        num_classes=2,
        device=torch.device("cpu"),
        seed=241,
        config=config
        or CircadianHeadConfig(
            sleep_mode="components",
            split_threshold=0.8,
            prune_threshold=0.2,
            split_weight_norm_mix=0.0,
            split_importance_mix=0.0,
            prune_weight_norm_mix=0.0,
            prune_importance_mix=0.0,
            max_split_per_sleep=1,
            max_prune_per_sleep=0,
            split_noise_scale=0.1,
            use_dual_chemical=True,
            use_reward_modulated_learning=True,
        ),
        min_hidden_dim=3,
        max_hidden_dim=6,
    )


def _assert_same_state(left: dict[str, Any], right: dict[str, Any]) -> None:
    assert left.keys() == right.keys()
    for name, value in left.items():
        if torch.is_tensor(value):
            assert torch.equal(value, right[name]), name
        else:
            assert value == right[name], name


def test_torch_snapshot_restores_next_noisy_split_and_detaches_both_ways() -> None:
    head = _head()
    features = torch.tensor([[0.2, -0.3, 0.1], [-0.1, 0.4, 0.5]])
    targets = torch.tensor([1, 0], dtype=torch.long)
    head.train_step(features, targets, 0.03, 2, 0.2)
    head._chemical = torch.tensor([0.95, 0.6, 0.3, 0.1])
    head._chemical_fast = head._chemical.clone()
    head._chemical_slow = head._chemical.clone()
    head._energy_history = [0.4, 0.3]
    saved = head.snapshot_state()
    expected = deepcopy(saved)
    control = _head()
    control.restore_state(saved)
    control._split_generator.set_state(head._split_generator.get_state())

    changed = head.sleep_event(force_sleep=True)
    assert changed.split_indices == (0,)
    assert head.hidden_dim == 5
    head.restore_state(saved)
    _assert_same_state(head.snapshot_state(), expected)
    assert torch.equal(head._split_generator.get_state(), expected["split_generator_state"])

    saved["chemical"].fill_(-3)
    saved["energy_history"].append(-3.0)
    saved["split_generator_state"].zero_()
    saved["config"]["split_noise_scale"] = 999.0
    _assert_same_state(head.snapshot_state(), expected)

    repeat = head.sleep_event(force_sleep=True)
    expected_repeat = control.sleep_event(force_sleep=True)
    assert repeat == expected_repeat
    _assert_same_state(head.snapshot_state(), control.snapshot_state())


def test_torch_snapshot_restores_after_prune() -> None:
    config = CircadianHeadConfig(
        sleep_mode="components",
        max_split_per_sleep=0,
        max_prune_per_sleep=1,
        prune_threshold=0.2,
        prune_weight_norm_mix=0.0,
        prune_importance_mix=0.0,
    )
    head = _head(config)
    head._chemical = torch.tensor([0.95, 0.6, 0.3, 0.0])
    saved = head.snapshot_state()
    decision = head.sleep_event(force_sleep=True)
    assert decision.pruned_indices == (3,)
    assert head.hidden_dim == 3
    head.restore_state(saved)
    _assert_same_state(head.snapshot_state(), saved)


@pytest.mark.parametrize(
    "change",
    ["version", "config", "device", "topology", "generator", "missing_field"],
)
def test_torch_snapshot_rejects_incompatible_state_before_mutation(change: str) -> None:
    head = _head()
    before = head.snapshot_state()
    broken = deepcopy(before)
    if change == "version":
        broken["format_version"] = 99
    elif change == "config":
        broken["config"]["split_noise_scale"] = 0.0
    elif change == "device":
        broken["device"] = "cuda:0"
    elif change == "topology":
        broken["chemical"] = broken["chemical"][:-1]
    elif change == "generator":
        broken["split_generator_state"] = torch.zeros(1, dtype=torch.uint8)
    else:
        del broken["bias_output"]

    with pytest.raises((TypeError, ValueError), match="snapshot"):
        head.restore_state(broken)
    _assert_same_state(head.snapshot_state(), before)
