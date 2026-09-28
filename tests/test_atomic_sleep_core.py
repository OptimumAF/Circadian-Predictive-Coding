"""Rejected core sleep events leave no model-owned state behind."""

from __future__ import annotations

from collections import deque
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import pytest

from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork

torch = pytest.importorskip("torch")

from src.core.resnet50_variants import (  # noqa: E402
    CircadianHeadConfig,
    CircadianPredictiveCodingHead,
)


def _same_numpy_value(actual: Any, expected: Any) -> None:
    assert type(actual) is type(expected)
    if isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(actual, np.random.Generator):
        _same_numpy_value(actual.bit_generator.state, expected.bit_generator.state)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key, value in actual.items():
            _same_numpy_value(value, expected[key])
    elif isinstance(actual, (list, tuple, deque)):
        assert len(actual) == len(expected)
        if isinstance(actual, deque):
            assert actual.maxlen == expected.maxlen
        for value, other in zip(actual, expected):
            _same_numpy_value(value, other)
    elif is_dataclass(actual):
        for field in fields(actual):
            _same_numpy_value(getattr(actual, field.name), getattr(expected, field.name))
    else:
        assert actual == expected


def _same_torch_state(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        if torch.is_tensor(value):
            assert torch.equal(value, expected[name]), name
        else:
            assert value == expected[name], name


def _numpy_model() -> CircadianPredictiveCodingNetwork:
    config = CircadianConfig(
        sleep_mode="components",
        split_threshold=0.8,
        split_noise_scale=0.1,
        max_split_per_sleep=1,
        max_prune_per_sleep=0,
        replay_steps=1,
        replay_memory_size=2,
        sleep_enable_prune=False,
        use_dual_chemical=True,
    )
    model = CircadianPredictiveCodingNetwork(
        2,
        4,
        seed=181,
        min_hidden_dim=3,
        max_hidden_dim=6,
        hidden_dims=(3, 4),
        circadian_config=config,
    )
    model.train_epoch(
        np.array([[0.3, -0.2], [-0.1, 0.4]]),
        np.array([[1.0], [0.0]]),
        0.03,
        2,
        0.2,
    )
    model.set_chemical_state(np.array([0.95, 0.6, 0.4, 0.0]))
    return model


def _torch_head() -> CircadianPredictiveCodingHead:
    config = CircadianHeadConfig(
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
    )
    head = CircadianPredictiveCodingHead(
        feature_dim=3,
        hidden_dim=4,
        num_classes=2,
        device=torch.device("cpu"),
        seed=241,
        config=config,
        min_hidden_dim=3,
        max_hidden_dim=6,
    )
    head.train_step(
        torch.tensor([[0.2, -0.3, 0.1], [-0.1, 0.4, 0.5]]),
        torch.tensor([1, 0], dtype=torch.long),
        0.03,
        2,
        0.2,
    )
    head._chemical = torch.tensor([0.95, 0.6, 0.3, 0.1])
    head._chemical_fast = head._chemical.clone()
    head._chemical_slow = head._chemical.clone()
    return head


def test_numpy_sleep_recovers_after_replay_mutates_then_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _numpy_model()
    before = model.snapshot_state()
    control = _numpy_model()
    control.restore_state(before)
    original = CircadianPredictiveCodingNetwork._run_replay_consolidation

    def fail_after_replay(self: CircadianPredictiveCodingNetwork) -> None:
        original(self)
        raise RuntimeError("injected replay failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            CircadianPredictiveCodingNetwork, "_run_replay_consolidation", fail_after_replay
        )
        with pytest.raises(RuntimeError, match="injected replay failure"):
            model.sleep_event()

    _same_numpy_value(model.snapshot_state().state, before.state)
    assert model.sleep_event() == control.sleep_event()
    _same_numpy_value(model.snapshot_state().state, control.snapshot_state().state)
    batch = np.array([[0.4, -0.1], [-0.2, 0.3]])
    labels = np.array([[1.0], [0.0]])
    assert model.train_epoch(batch, labels, 0.03, 2, 0.2) == control.train_epoch(
        batch, labels, 0.03, 2, 0.2
    )
    _same_numpy_value(model.snapshot_state().state, control.snapshot_state().state)


def test_numpy_sleep_rejects_nonfinite_post_state(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _numpy_model()
    before = model.snapshot_state()

    def corrupt_after_split(self: CircadianPredictiveCodingNetwork) -> None:
        self.weight_hidden_output[0, 0] = np.nan

    monkeypatch.setattr(
        CircadianPredictiveCodingNetwork, "_apply_homeostatic_downscaling", corrupt_after_split
    )
    with pytest.raises(FloatingPointError, match="nonfinite"):
        model.sleep_event()
    _same_numpy_value(model.snapshot_state().state, before.state)


def test_numpy_sleep_rejects_post_mutation_topology_error(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _numpy_model()
    before = model.snapshot_state()

    def corrupt_after_split(self: CircadianPredictiveCodingNetwork) -> None:
        self._traffic_sum = self._traffic_sum[:-1]

    monkeypatch.setattr(
        CircadianPredictiveCodingNetwork, "_apply_homeostatic_downscaling", corrupt_after_split
    )
    with pytest.raises(ValueError, match="topology"):
        model.sleep_event()
    _same_numpy_value(model.snapshot_state().state, before.state)


def test_torch_sleep_recovers_after_split_mutates_then_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = _torch_head()
    before = head.snapshot_state()
    control = _torch_head()
    control.restore_state(before)

    def fail_after_split(self: CircadianPredictiveCodingHead) -> None:
        self.weight_hidden_output[0, 0] += 1.0
        raise RuntimeError("injected structural failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            CircadianPredictiveCodingHead, "_apply_homeostatic_downscaling", fail_after_split
        )
        with pytest.raises(RuntimeError, match="injected structural failure"):
            head.sleep_event()

    _same_torch_state(head.snapshot_state(), before)
    assert head.sleep_event() == control.sleep_event()
    _same_torch_state(head.snapshot_state(), control.snapshot_state())
    features = torch.tensor([[0.1, 0.2, -0.3], [-0.2, 0.4, 0.3]])
    targets = torch.tensor([0, 1], dtype=torch.long)
    assert head.train_step(features, targets, 0.03, 2, 0.2) == control.train_step(
        features, targets, 0.03, 2, 0.2
    )
    _same_torch_state(head.snapshot_state(), control.snapshot_state())


def test_torch_sleep_rejects_nonfinite_post_state(monkeypatch: pytest.MonkeyPatch) -> None:
    head = _torch_head()
    before = head.snapshot_state()

    def corrupt_after_split(self: CircadianPredictiveCodingHead) -> None:
        self._chemical[0] = float("nan")

    monkeypatch.setattr(
        CircadianPredictiveCodingHead, "_apply_homeostatic_downscaling", corrupt_after_split
    )
    with pytest.raises(FloatingPointError, match="nonfinite"):
        head.sleep_event()
    _same_torch_state(head.snapshot_state(), before)


def test_torch_sleep_rejects_post_mutation_topology_error(monkeypatch: pytest.MonkeyPatch) -> None:
    head = _torch_head()
    before = head.snapshot_state()

    def corrupt_after_split(self: CircadianPredictiveCodingHead) -> None:
        self._traffic_sum = self._traffic_sum[:-1]

    monkeypatch.setattr(
        CircadianPredictiveCodingHead, "_apply_homeostatic_downscaling", corrupt_after_split
    )
    with pytest.raises(ValueError, match="topology"):
        head.sleep_event()
    _same_torch_state(head.snapshot_state(), before)
