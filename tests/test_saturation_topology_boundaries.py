"""Extreme finite logits and post-sleep widths retain their declared contracts."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
)
from src.core.predictive_coding import PredictiveCodingNetwork

torch = pytest.importorskip("torch")

from src.app.matched_head_benchmark import _evaluate_head  # noqa: E402
from src.core.resnet50_variants import (  # noqa: E402
    CircadianHeadConfig,
    CircadianPredictiveCodingHead,
    PredictiveCodingHead,
)


@pytest.mark.parametrize("kind", ["backprop", "predictive", "circadian"])
@pytest.mark.parametrize("logit,target", [(1e300, 0.0), (-1e300, 1.0)])
def test_numpy_binary_clipped_bce_remains_finite_at_extreme_logits(
    kind: str, logit: float, target: float
) -> None:
    if kind == "backprop":
        model: Any = BackpropMLP(2, 2, seed=61)
    elif kind == "predictive":
        model = PredictiveCodingNetwork(2, 2, seed=61)
    else:
        model = CircadianPredictiveCodingNetwork(2, 2, seed=61, min_hidden_dim=2)
    model.weight_input_hidden[:] = 0.0
    model.weight_hidden_output[:] = 0.0
    model.bias_hidden[:] = 0.0
    model.bias_output[:] = logit
    features = np.zeros((1, 2))
    targets = np.array([[target]])

    if kind == "backprop":
        metric = model.train_epoch(features, targets, 0.01).loss
    else:
        metric = model.train_epoch(features, targets, 0.01, 1, 0.2).energy

    assert np.isfinite(metric)
    clipped_tail = 1.0 - (1.0 - 1e-8) if logit > 0 else 1e-8
    assert metric == pytest.approx(-np.log(clipped_tail), rel=1e-12)
    assert np.isfinite(model.bias_output).all()


@pytest.mark.parametrize("kind", ["predictive", "circadian"])
def test_torch_multiclass_extreme_logit_diagnostic_and_ce_are_finite(kind: str) -> None:
    if kind == "predictive":
        head: Any = PredictiveCodingHead(2, 2, 3, torch.device("cpu"), seed=67)
    else:
        head = CircadianPredictiveCodingHead(
            2, 2, 3, torch.device("cpu"), seed=67, min_hidden_dim=2
        )
    for name in ("weight_feature_hidden", "bias_hidden", "weight_hidden_output", "bias_output"):
        setattr(head, name, getattr(head, name).double())
    head.weight_feature_hidden[:] = 0.0
    head.weight_hidden_output[:] = 0.0
    head.bias_hidden[:] = 0.0
    head.bias_output[:] = torch.tensor([[1e300, -1e300, 0.0]], dtype=torch.float64)
    features = torch.zeros((1, 2), dtype=torch.float64)
    targets = torch.tensor([1])

    accuracy, cross_entropy = _evaluate_head(
        torch, torch.device("cpu"), head.predict_logits, ((features, targets),), None
    )
    assert accuracy == 0.0
    assert cross_entropy == pytest.approx(2e300, rel=1e-15)
    energy = head.train_step(features, targets, 0.01, 1, 0.2)
    assert energy == pytest.approx(1.0 / 3.0, rel=1e-15)
    assert bool(torch.isfinite(head.bias_output).all().item())


def _numpy_structural_model(kind: str) -> CircadianPredictiveCodingNetwork:
    config = replace(
        CircadianConfig.matched_pc_control(),
        max_split_per_sleep=1 if kind == "split" else 0,
        max_prune_per_sleep=1 if kind == "prune" else 0,
        split_threshold=0.8,
        prune_threshold=0.2,
        split_noise_scale=0.0,
        prune_decay_steps=1,
    )
    model = CircadianPredictiveCodingNetwork(
        2, 4, seed=71, min_hidden_dim=3, max_hidden_dim=6, circadian_config=config
    )
    model.set_chemical_state(np.array([0.95, 0.6, 0.5, 0.0]))
    return model


def _torch_structural_head(kind: str) -> CircadianPredictiveCodingHead:
    config = replace(
        CircadianHeadConfig.matched_pc_control(),
        max_split_per_sleep=1 if kind == "split" else 0,
        max_prune_per_sleep=1 if kind == "prune" else 0,
        split_threshold=0.8,
        prune_threshold=0.2,
        split_noise_scale=0.0,
    )
    head = CircadianPredictiveCodingHead(
        2, 4, 3, torch.device("cpu"), seed=71, min_hidden_dim=3, max_hidden_dim=6, config=config
    )
    head._chemical = torch.tensor([0.95, 0.6, 0.5, 0.0])
    head._chemical_fast = head._chemical.clone()
    head._chemical_slow = head._chemical.clone()
    return head


@pytest.mark.parametrize("kind,expected_width", [("split", 5), ("prune", 3)])
def test_numpy_circadian_trains_with_post_sleep_width(kind: str, expected_width: int) -> None:
    model = _numpy_structural_model(kind)
    event = model.sleep_event(force_sleep=True)
    assert event.new_hidden_dim == expected_width
    result = model.train_epoch(np.array([[0.3, -0.2]]), np.array([[1.0]]), 0.03, 2, 0.2)
    assert np.isfinite(result.energy)
    assert model.weight_input_hidden.shape == (2, expected_width)
    assert model.weight_hidden_output.shape == (expected_width, 1)
    assert model.get_chemical_state().shape == (expected_width,)
    assert model._importance_ema.shape == (expected_width,)
    assert model._traffic_sum.shape == (expected_width,)


@pytest.mark.parametrize("kind,expected_width", [("split", 5), ("prune", 3)])
def test_torch_circadian_trains_with_post_sleep_width(kind: str, expected_width: int) -> None:
    head = _torch_structural_head(kind)
    event = head.sleep_event(force_sleep=True)
    assert event.new_hidden_dim == expected_width
    energy = head.train_step(torch.tensor([[0.3, -0.2]]), torch.tensor([1]), 0.03, 2, 0.2)
    assert np.isfinite(energy)
    assert tuple(head.weight_feature_hidden.shape) == (2, expected_width)
    assert tuple(head.weight_hidden_output.shape) == (expected_width, 3)
    for name in ("_chemical", "_importance_ema", "_traffic_sum", "_neuron_age"):
        assert tuple(getattr(head, name).shape) == (expected_width,)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_incompatible_internal_width_after_split_rejected_before_cooldown(backend: str) -> None:
    if backend == "numpy":
        model = _numpy_structural_model("split")
        model.sleep_event(force_sleep=True)
        model.weight_hidden_output = model.weight_hidden_output[:-1, :]
        before_cooldown = model._split_cooldown.copy()
        before_weights = model.weight_hidden_output.copy()
        with pytest.raises(ValueError, match="model topology"):
            model.train_epoch(np.array([[0.3, -0.2]]), np.array([[1.0]]), 0.03, 1, 0.2)
        np.testing.assert_array_equal(model._split_cooldown, before_cooldown)
        np.testing.assert_array_equal(model.weight_hidden_output, before_weights)
    else:
        head = _torch_structural_head("split")
        head.sleep_event(force_sleep=True)
        head.weight_hidden_output = head.weight_hidden_output[:-1, :]
        before_cooldown = head._split_cooldown.clone()
        before_weights = head.weight_hidden_output.clone()
        with pytest.raises(ValueError, match="model topology"):
            head.train_step(torch.tensor([[0.3, -0.2]]), torch.tensor([1]), 0.03, 1, 0.2)
        assert torch.equal(head._split_cooldown, before_cooldown)
        assert torch.equal(head.weight_hidden_output, before_weights)
