"""Torch head sleep components can run without structural changes."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from src.core.resnet50_variants import (  # noqa: E402
    CircadianHeadConfig,
    CircadianPredictiveCodingHead,
)


def _component_config(**changes: Any) -> CircadianHeadConfig:
    options: dict[str, Any] = dict(
        sleep_mode="components",
        sleep_enable_chemical_reset=False,
        sleep_enable_homeostasis=False,
        sleep_enable_split=False,
        sleep_enable_prune=False,
        max_split_per_sleep=0,
        max_prune_per_sleep=0,
        sleep_reset_factor=0.5,
        homeostatic_downscale_factor=0.8,
        homeostasis_target_input_norm=0.0,
        homeostasis_target_output_norm=0.0,
    )
    options.update(changes)
    return CircadianHeadConfig(**options)


def _head(config: CircadianHeadConfig, width: int = 3) -> CircadianPredictiveCodingHead:
    return CircadianPredictiveCodingHead(
        2,
        width,
        3,
        torch.device("cpu"),
        seed=89,
        min_hidden_dim=2,
        max_hidden_dim=width + 2,
        config=config,
    )


def test_zero_structural_budgets_still_reset_torch_chemical_only() -> None:
    head = _head(_component_config(sleep_enable_chemical_reset=True))
    head._chemical = torch.tensor([0.2, 0.6, 1.0])
    head._steps_since_sleep = 4
    before_weight = head.weight_feature_hidden.clone()
    event = head.sleep_event(force_sleep=True)
    assert event.performed is True
    assert event.old_hidden_dim == event.new_hidden_dim == 3
    assert event.split_indices == event.pruned_indices == ()
    torch.testing.assert_close(head._chemical, torch.tensor([0.1, 0.3, 0.5]), atol=0, rtol=0)
    assert torch.equal(head.weight_feature_hidden, before_weight)
    assert head._steps_since_sleep == 0


def test_zero_structural_budgets_still_run_torch_homeostasis_only() -> None:
    head = _head(_component_config(sleep_enable_homeostasis=True))
    head._chemical = torch.tensor([0.2, 0.6, 1.0])
    before_weight = head.weight_feature_hidden.clone()
    before_chemical = head._chemical.clone()
    event = head.sleep_event(force_sleep=True)
    assert event.performed is True
    assert event.split_indices == event.pruned_indices == ()
    torch.testing.assert_close(head.weight_feature_hidden, 0.8 * before_weight, atol=0, rtol=0)
    assert torch.equal(head._chemical, before_chemical)


@pytest.mark.parametrize("kind,expected_width", [("split", 5), ("prune", 3)])
def test_torch_split_and_prune_switches_act_independently(kind: str, expected_width: int) -> None:
    config = _component_config(
        sleep_enable_split=kind == "split",
        sleep_enable_prune=kind == "prune",
        max_split_per_sleep=1,
        max_prune_per_sleep=1,
        split_threshold=0.8,
        prune_threshold=0.2,
        split_noise_scale=0.0,
    )
    head = _head(config, width=4)
    head._chemical = torch.tensor([0.95, 0.6, 0.4, 0.0])
    event = head.sleep_event(force_sleep=True)
    assert event.performed is True
    assert event.old_hidden_dim == 4
    assert event.new_hidden_dim == expected_width
    assert len(event.split_indices) == (1 if kind == "split" else 0)
    assert len(event.pruned_indices) == (1 if kind == "prune" else 0)
    assert float(head._chemical[0]) > 0.4


@pytest.mark.parametrize("mode", ["disabled", "legacy"])
def test_torch_disabled_and_legacy_zero_budget_routes_stay_noop(mode: str) -> None:
    config = _component_config(
        sleep_mode=mode,
        sleep_enable_chemical_reset=True,
        sleep_enable_homeostasis=True,
        sleep_enable_split=True,
        sleep_enable_prune=True,
        max_split_per_sleep=0,
        max_prune_per_sleep=0,
    )
    head = _head(config)
    head._chemical = torch.tensor([0.2, 0.6, 1.0])
    head._steps_since_sleep = 4
    before = head.snapshot_state()
    event = head.sleep_event(force_sleep=True)
    assert event.performed is False
    assert event.split_indices == event.pruned_indices == ()
    assert torch.equal(head._chemical, before["chemical"])
    assert torch.equal(head.weight_feature_hidden, before["weight_feature_hidden"])
    assert head._steps_since_sleep == 4


def test_torch_disabled_mode_ignores_forced_structural_budget() -> None:
    config = _component_config(
        sleep_mode="disabled", max_split_per_sleep=1, sleep_enable_split=True
    )
    head = _head(config)
    head._chemical = torch.tensor([0.95, 0.2, 0.1])
    before = head.weight_feature_hidden.clone()
    event = head.sleep_event(force_sleep=True)
    assert event.performed is False
    assert event.new_hidden_dim == 3
    assert torch.equal(head.weight_feature_hidden, before)


def test_torch_invalid_mode_and_legacy_component_mix_fail_at_construction() -> None:
    with pytest.raises(ValueError, match="sleep_mode"):
        _head(replace(_component_config(), sleep_mode="unknown"))
    with pytest.raises(ValueError, match="sleep_mode.*components"):
        _head(replace(CircadianHeadConfig(), sleep_enable_prune=False))


@pytest.mark.parametrize("mode", ["components", "disabled"])
def test_torch_sleep_mode_does_not_change_valid_wake_step(mode: str) -> None:
    legacy = _head(CircadianHeadConfig(max_split_per_sleep=0, max_prune_per_sleep=0))
    alternate = _head(
        replace(
            legacy.config,
            sleep_mode=mode,
            sleep_enable_chemical_reset=False,
            sleep_enable_homeostasis=False,
            sleep_enable_split=False,
            sleep_enable_prune=False,
        )
    )
    features = torch.tensor([[0.3, -0.2], [-0.1, 0.4]])
    targets = torch.tensor([1, 0], dtype=torch.int64)
    first = legacy.train_step(features, targets, 0.03, 2, 0.2)
    second = alternate.train_step(features, targets, 0.03, 2, 0.2)
    assert first == second
    for field in ("weight_feature_hidden", "weight_hidden_output", "bias_hidden", "bias_output"):
        assert torch.equal(getattr(legacy, field), getattr(alternate, field))
    assert torch.equal(legacy._chemical, alternate._chemical)
    assert legacy._steps_since_sleep == alternate._steps_since_sleep
