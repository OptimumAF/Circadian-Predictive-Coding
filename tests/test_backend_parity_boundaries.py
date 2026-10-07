"""Matched binary/two-logit fixtures for NumPy and Torch production paths."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork


def _fixture() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    features = np.array([[0.3, -0.2], [-0.5, 0.4]], dtype=np.float64)
    targets = np.array([[1.0], [0.0]], dtype=np.float64)
    values = {
        "input_weight": np.array([[0.13, -0.21, 0.08], [-0.17, 0.05, 0.19]]),
        "hidden_bias": np.array([[0.02, -0.04, 0.03]]),
        "output_weight": np.array([[0.2], [-0.3], [0.4]]),
        "output_bias": np.array([[0.07]]),
    }
    return features, targets, values


def _assign_binary_two_logit_gauge(
    numpy_model: Any, torch_head: Any, values: dict[str, np.ndarray], torch: Any
) -> None:
    """Represent sigmoid(z) as softmax([-z/2, z/2]) without changing either trainer."""
    np.copyto(numpy_model.weight_input_hidden, values["input_weight"])
    np.copyto(numpy_model.bias_hidden, values["hidden_bias"])
    np.copyto(numpy_model.weight_hidden_output, values["output_weight"])
    np.copyto(numpy_model.bias_output, values["output_bias"])
    torch_head.weight_feature_hidden = torch.tensor(values["input_weight"], dtype=torch.float64)
    torch_head.bias_hidden = torch.tensor(values["hidden_bias"], dtype=torch.float64)
    torch_head.weight_hidden_output = torch.tensor(
        np.concatenate((-values["output_weight"] / 2, values["output_weight"] / 2), axis=1),
        dtype=torch.float64,
    )
    torch_head.bias_output = torch.tensor(
        np.concatenate((-values["output_bias"] / 2, values["output_bias"] / 2), axis=1),
        dtype=torch.float64,
    )


@pytest.mark.parametrize("circadian", [False, True])
def test_common_output_gauge_matches_forward_hidden_step_and_chemistry_but_not_output_step(
    circadian: bool,
) -> None:
    torch = pytest.importorskip("torch")
    from src.core.resnet50_variants import (
        CircadianHeadConfig,
        CircadianPredictiveCodingHead,
        PredictiveCodingHead,
    )

    features, targets, values = _fixture()
    numpy_model: Any
    torch_head: Any
    if circadian:
        numpy_model = CircadianPredictiveCodingNetwork(
            2,
            3,
            seed=13,
            min_hidden_dim=3,
            circadian_config=CircadianConfig.matched_pc_control(),
        )
        torch_head = CircadianPredictiveCodingHead(
            2,
            3,
            2,
            torch.device("cpu"),
            seed=13,
            min_hidden_dim=3,
            config=CircadianHeadConfig.matched_pc_control(),
        )
    else:
        numpy_model = PredictiveCodingNetwork(2, 3, seed=13)
        torch_head = PredictiveCodingHead(2, 3, 2, torch.device("cpu"), seed=13)
    _assign_binary_two_logit_gauge(numpy_model, torch_head, values, torch)
    torch_features = torch.tensor(features, dtype=torch.float64)
    torch_targets = torch.tensor(targets[:, 0], dtype=torch.long)
    assert torch_head.weight_feature_hidden.dtype == torch.float64
    assert torch_head.weight_hidden_output.dtype == torch.float64

    numpy_probability = numpy_model.predict_proba(features)
    initial_logits = torch_head.predict_logits(torch_features)
    torch_probability = torch.softmax(initial_logits, dim=1)[:, 1:2]
    np.testing.assert_allclose(torch_probability.numpy(), numpy_probability, atol=1e-15, rtol=0)
    margin = initial_logits[:, 1:2] - initial_logits[:, 0:1]
    numpy_logits = (
        np.tanh(features @ values["input_weight"] + values["hidden_bias"]) @ values["output_weight"]
        + values["output_bias"]
    )
    np.testing.assert_allclose(margin.numpy(), numpy_logits, atol=1e-15, rtol=0)
    binary_loss = np.mean(np.logaddexp(0.0, numpy_logits) - targets * numpy_logits)
    torch_loss = torch.nn.functional.cross_entropy(initial_logits, torch_targets)
    np.testing.assert_allclose(float(torch_loss), binary_loss, atol=1e-15, rtol=0)

    initial_output_weight = values["output_weight"].copy()
    initial_output_bias = values["output_bias"].copy()
    numpy_model.train_epoch(features, targets, 0.03, 3, 0.2)
    torch_head.train_step(torch_features, torch_targets, 0.03, 3, 0.2)

    np.testing.assert_allclose(
        torch_head.weight_feature_hidden.numpy(),
        numpy_model.weight_input_hidden,
        atol=1e-15,
        rtol=0,
    )
    np.testing.assert_allclose(
        torch_head.bias_hidden.numpy(), numpy_model.bias_hidden, atol=1e-15, rtol=0
    )
    np.testing.assert_allclose(
        torch_head.mean_hidden_traffic().numpy(),
        numpy_model.get_layer_traffic()[0].mean_abs_activation,
        atol=1e-15,
        rtol=0,
    )
    if circadian:
        np.testing.assert_allclose(
            torch_head.mean_chemical().numpy(), numpy_model.get_chemical_state(), atol=1e-15, rtol=0
        )
        np.testing.assert_allclose(
            torch_head._plasticity().numpy(), numpy_model.get_plasticity_state(), atol=0, rtol=0
        )
        np.testing.assert_allclose(
            torch_head._importance_ema.numpy(), numpy_model._importance_ema, atol=1e-15, rtol=0
        )

    # The two free softmax columns each move by eta*g, so their margin moves
    # by 2*eta*g; the single NumPy output column moves by eta*g.
    updated_margin_weight = (
        torch_head.weight_hidden_output[:, 1:2] - torch_head.weight_hidden_output[:, 0:1]
    ).numpy()
    updated_margin_bias = (torch_head.bias_output[:, 1:2] - torch_head.bias_output[:, 0:1]).numpy()
    np.testing.assert_allclose(
        updated_margin_weight,
        2 * numpy_model.weight_hidden_output - initial_output_weight,
        atol=1e-15,
        rtol=0,
    )
    np.testing.assert_allclose(
        updated_margin_bias, 2 * numpy_model.bias_output - initial_output_bias, atol=1e-15, rtol=0
    )
    assert np.max(np.abs(updated_margin_weight - numpy_model.weight_hidden_output)) > 1e-5


def test_backprop_mlp_gauge_matches_hidden_sgd_step_but_doubles_output_margin_step() -> None:
    torch = pytest.importorskip("torch")
    from src.core.resnet50_variants import BackpropMLPHead

    features, targets, values = _fixture()
    numpy_model = BackpropMLP(2, 3, seed=19)
    torch_head = BackpropMLPHead(2, 3, 2, torch.device("cpu"), seed=19)
    _assign_binary_two_logit_gauge(numpy_model, torch_head, values, torch)
    for name in ("weight_feature_hidden", "bias_hidden", "weight_hidden_output", "bias_output"):
        setattr(torch_head, name, torch.nn.Parameter(getattr(torch_head, name)))
    torch_features = torch.tensor(features, dtype=torch.float64)
    torch_targets = torch.tensor(targets[:, 0], dtype=torch.long)
    logits = torch_head.forward_logits(torch_features)
    np.testing.assert_allclose(
        torch.softmax(logits, dim=1)[:, 1:2].detach().numpy(),
        numpy_model.predict_proba(features),
        atol=1e-15,
        rtol=0,
    )

    optimizer = torch.optim.SGD(torch_head.trainable_parameters(), lr=0.03, momentum=0.0)
    torch_loss = torch.nn.functional.cross_entropy(logits, torch_targets)
    optimizer.zero_grad()
    torch_loss.backward()
    optimizer.step()
    numpy_loss = numpy_model.train_epoch(features, targets, 0.03).loss

    np.testing.assert_allclose(float(torch_loss.detach().item()), numpy_loss, atol=1e-15, rtol=0)
    np.testing.assert_allclose(
        torch_head.weight_feature_hidden.detach().numpy(),
        numpy_model.weight_input_hidden,
        atol=1e-15,
        rtol=0,
    )
    np.testing.assert_allclose(
        torch_head.bias_hidden.detach().numpy(), numpy_model.bias_hidden, atol=1e-15, rtol=0
    )
    np.testing.assert_allclose(
        (torch_head.weight_hidden_output[:, 1:2] - torch_head.weight_hidden_output[:, 0:1])
        .detach()
        .numpy(),
        2 * numpy_model.weight_hidden_output - values["output_weight"],
        atol=1e-15,
        rtol=0,
    )
    np.testing.assert_allclose(
        (torch_head.bias_output[:, 1:2] - torch_head.bias_output[:, 0:1]).detach().numpy(),
        2 * numpy_model.bias_output - values["output_bias"],
        atol=1e-15,
        rtol=0,
    )


def _structural_fixture(kind: str) -> tuple[Any, Any, np.ndarray, Any]:
    torch = pytest.importorskip("torch")
    from src.core.resnet50_variants import CircadianHeadConfig, CircadianPredictiveCodingHead

    split_budget = 1 if kind == "split" else 0
    prune_budget = 1 if kind == "prune" else 0
    numpy_config = replace(
        CircadianConfig.matched_pc_control(),
        max_split_per_sleep=split_budget,
        max_prune_per_sleep=prune_budget,
        split_threshold=0.8,
        prune_threshold=0.2,
        split_noise_scale=0.0,
        prune_decay_steps=1,
    )
    torch_config = replace(
        CircadianHeadConfig.matched_pc_control(),
        max_split_per_sleep=split_budget,
        max_prune_per_sleep=prune_budget,
        split_threshold=0.8,
        prune_threshold=0.2,
        split_noise_scale=0.0,
    )
    numpy_model = CircadianPredictiveCodingNetwork(
        2,
        4,
        seed=17,
        min_hidden_dim=3,
        max_hidden_dim=6,
        circadian_config=numpy_config,
    )
    torch_head = CircadianPredictiveCodingHead(
        2,
        4,
        2,
        torch.device("cpu"),
        seed=17,
        min_hidden_dim=3,
        max_hidden_dim=6,
        config=torch_config,
    )
    values = {
        "input_weight": np.array([[0.2, -0.1, 0.3, -0.2], [0.1, 0.25, -0.15, 0.05]]),
        "hidden_bias": np.array([[0.02, -0.03, 0.01, 0.04]]),
        "output_weight": np.array([[0.35], [-0.10], [0.25], [-0.30]]),
        "output_bias": np.array([[0.07]]),
    }
    _assign_binary_two_logit_gauge(numpy_model, torch_head, values, torch)
    chemical = np.array([0.95, 0.85, 0.08, 0.12])
    importance = np.array([0.1, 0.4, 0.8, 0.0])
    numpy_model.set_chemical_state(chemical)
    numpy_model._importance_ema = importance.copy()
    torch_head._chemical = torch.tensor(chemical, dtype=torch.float64)
    torch_head._chemical_fast = torch_head._chemical.clone()
    torch_head._chemical_slow = torch_head._chemical.clone()
    torch_head._importance_ema = torch.tensor(importance, dtype=torch.float64)
    return numpy_model, torch_head, np.array([[0.3, -0.2], [-0.5, 0.4]]), torch


@pytest.mark.parametrize("kind,expected_index", [("split", 0), ("prune", 3)])
def test_matched_structural_scores_and_separate_sleep_events(
    kind: str, expected_index: int
) -> None:
    numpy_model, torch_head, features, torch = _structural_fixture(kind)
    split_scores = numpy_model._compute_split_scores()
    prune_scores = numpy_model._compute_prune_scores()
    np.testing.assert_allclose(
        split_scores,
        torch_head._compute_split_scores().numpy(),
        atol=1e-15,
        rtol=0,
    )
    np.testing.assert_allclose(
        prune_scores,
        torch_head._compute_prune_scores().numpy(),
        atol=1e-15,
        rtol=0,
    )
    if kind == "split":
        assert split_scores[0] > split_scores[1] + 0.1
        assert (
            numpy_model._select_split_indices()
            == torch_head._select_split_indices()
            == (expected_index,)
        )
    else:
        assert prune_scores[3] > prune_scores[2] + 0.1
        assert (
            numpy_model._select_prune_indices()
            == torch_head._select_prune_indices()
            == (expected_index,)
        )

    before_numpy = numpy_model.predict_proba(features)
    before_torch = torch.softmax(
        torch_head.predict_logits(torch.tensor(features, dtype=torch.float64)), dim=1
    )[:, 1:2]
    np.testing.assert_allclose(before_torch.numpy(), before_numpy, atol=1e-15, rtol=0)
    numpy_event = numpy_model.sleep_event(force_sleep=True)
    torch_event = torch_head.sleep_event(force_sleep=True)
    assert numpy_event.split_indices == torch_event.split_indices
    assert numpy_event.pruned_indices == torch_event.pruned_indices
    assert numpy_event.old_hidden_dim == torch_event.old_hidden_dim == 4
    assert numpy_event.new_hidden_dim == torch_event.new_hidden_dim == (5 if kind == "split" else 3)
    np.testing.assert_allclose(
        numpy_model.weight_input_hidden, torch_head.weight_feature_hidden.numpy(), atol=0, rtol=0
    )
    np.testing.assert_allclose(
        numpy_model.bias_hidden, torch_head.bias_hidden.numpy(), atol=0, rtol=0
    )
    np.testing.assert_allclose(
        numpy_model.get_chemical_state(), torch_head.mean_chemical().numpy(), atol=0, rtol=0
    )
    np.testing.assert_allclose(
        numpy_model.weight_hidden_output,
        (torch_head.weight_hidden_output[:, 1:2] - torch_head.weight_hidden_output[:, 0:1]).numpy(),
        atol=1e-15,
        rtol=0,
    )
    after_numpy = numpy_model.predict_proba(features)
    after_torch = torch.softmax(
        torch_head.predict_logits(torch.tensor(features, dtype=torch.float64)), dim=1
    )[:, 1:2]
    np.testing.assert_allclose(after_torch.numpy(), after_numpy, atol=1e-15, rtol=0)
    if kind == "split":
        np.testing.assert_allclose(after_numpy, before_numpy, atol=1e-15, rtol=0)
