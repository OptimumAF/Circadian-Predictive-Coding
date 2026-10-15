"""Independent BCE autodiff, matched initialization and model-state controls.

Fixed synthetic batches verify implementation correctness, not comparative
learning benefit. Torch owns the reference graph and simultaneous GD updates.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest
import torch
import torch.nn.functional as torch_functional

from src.core.coupled_energy import CoupledParameters, evaluate_coupled, feedforward_latents
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork, CoupledSnapshot
from src.core.matched_backprop import (
    MATCHED_BACKPROP_LOSS_ID,
    MatchedBackpropNetwork,
    MatchedBackpropSnapshot,
    evaluate_matched_backprop,
)

Array = NDArray[np.float64]
DEPTHS = ((3,), (3, 2), (3, 2, 3, 2))


def arrays(parameters: CoupledParameters) -> tuple[Array, ...]:
    return (
        *parameters.hidden_weights,
        *parameters.hidden_biases,
        parameters.output_weight,
        parameters.output_bias,
    )


def assert_parameters_equal(actual: CoupledParameters, expected: CoupledParameters) -> None:
    for a, b in zip(arrays(actual), arrays(expected), strict=True):
        np.testing.assert_array_equal(a, b)


def initial_parameters(depth: tuple[int, ...]) -> CoupledParameters:
    return CoupledPredictiveCodingNetwork(2, depth, seed=41).snapshot_parameters()


def batch() -> tuple[Array, Array]:
    return np.array([[0.3, -0.8], [1.2, 0.1], [-0.7, 0.4]]), np.array([[0.0], [0.35], [1.0]])


def torch_evaluate(parameters: CoupledParameters, x: Array, y: Array):
    tensors = tuple(
        torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in arrays(parameters)
    )
    depth = len(parameters.hidden_weights)
    previous = torch.tensor(x, dtype=torch.float64)
    for weight, bias in zip(tensors[:depth], tensors[depth : 2 * depth], strict=True):
        previous = torch.tanh(previous @ weight + bias)
    logits = previous @ tensors[-2] + tensors[-1]
    loss = torch_functional.binary_cross_entropy_with_logits(
        logits, torch.tensor(y, dtype=torch.float64)
    )
    gradients = torch.autograd.grad(loss, tensors)
    return loss, gradients, torch.sigmoid(logits)


def stepped(parameters: CoupledParameters, gradients: tuple[torch.Tensor, ...], rate: float):
    updated = tuple(
        a - rate * g.detach().numpy() for a, g in zip(arrays(parameters), gradients, strict=True)
    )
    depth = len(parameters.hidden_weights)
    return CoupledParameters(updated[:depth], updated[depth : 2 * depth], updated[-2], updated[-1])


@pytest.mark.parametrize("depth", DEPTHS)
@pytest.mark.parametrize("nonzero_bias", (False, True))
def test_should_match_torch_bce_and_every_gradient_coordinate(depth, nonzero_bias):
    parameters = initial_parameters(depth)
    if nonzero_bias:
        parameters = replace(
            parameters,
            hidden_biases=tuple(
                np.full_like(b, 0.13 * (i + 1)) for i, b in enumerate(parameters.hidden_biases)
            ),
            output_bias=np.array([[-0.27]]),
        )
    x, y = batch()
    expected_loss, expected_gradients, probabilities = torch_evaluate(parameters, x, y)
    observed = evaluate_matched_backprop(parameters, x, y)
    assert observed.loss == pytest.approx(expected_loss.item(), abs=2e-14)
    assert observed.loss_definition == MATCHED_BACKPROP_LOSS_ID
    assert MATCHED_BACKPROP_LOSS_ID == "numpy_matched_binary_bce_preupdate_v1"
    for actual, expected in zip(
        arrays(observed.parameter_gradients), expected_gradients, strict=True
    ):
        np.testing.assert_allclose(actual, expected.detach().numpy(), atol=2e-14, rtol=2e-12)
    model = MatchedBackpropNetwork(parameters)
    np.testing.assert_allclose(model.predict_proba(x), probabilities.detach().numpy(), atol=2e-14)
    assert (
        observed.loss_definition
        != evaluate_coupled(
            parameters, x, y, feedforward_latents(parameters, x), (1.0,) * len(depth)
        ).energy_definition
    )
    with pytest.raises(FrozenInstanceError):
        setattr(observed, "loss", 123.0)


@pytest.mark.parametrize("depth", DEPTHS)
def test_should_share_task_loss_with_pc_at_feedforward_latents(depth):
    parameters, (x, y) = initial_parameters(depth), batch()
    pc = evaluate_coupled(parameters, x, y, feedforward_latents(parameters, x), (1.0,) * len(depth))
    bp = evaluate_matched_backprop(parameters, x, y)
    assert pc.hidden_penalties == (0.0,) * len(depth)
    assert bp.loss == pc.task_loss == pc.energy


@pytest.mark.parametrize("depth", DEPTHS)
def test_should_follow_torch_simultaneous_updates_and_restore_continuation(depth):
    expected = initial_parameters(depth)
    model = MatchedBackpropNetwork(expected)
    assert model.input_dim == 2
    assert model.hidden_dims == depth
    assert_parameters_equal(model.snapshot_parameters(), expected)
    x, y = batch()
    checkpoint = None
    for step, indices in enumerate(([0, 1, 2], [2, 0], [1], [1, 2, 0, 2])):
        features, targets = x[indices], y[indices]
        loss, gradients, _ = torch_evaluate(expected, features, targets)
        if step == 2:
            checkpoint = model.snapshot_state()
        observed = model.train_epoch(features, targets, 0.03)
        assert observed.loss == pytest.approx(loss.item(), abs=2e-14)
        for actual, reference in zip(arrays(observed.parameter_gradients), gradients, strict=True):
            np.testing.assert_allclose(actual, reference.detach().numpy(), atol=2e-14, rtol=2e-12)
        expected = stepped(expected, gradients, 0.03)
        for actual, reference in zip(
            arrays(model.snapshot_parameters()), arrays(expected), strict=True
        ):
            np.testing.assert_allclose(actual, reference, atol=2e-14, rtol=2e-12)
    assert checkpoint is not None
    final_state = model.snapshot_state()
    model.restore_state(checkpoint)
    for indices in ([1], [1, 2, 0, 2]):
        model.train_epoch(x[indices], y[indices], 0.03)
    assert_parameters_equal(model.snapshot_state().parameters, final_state.parameters)


@pytest.mark.parametrize("logit", (-1000.0, 1000.0))
def test_should_keep_unclipped_bce_and_gradients_at_saturated_logits(logit):
    parameters = replace(
        initial_parameters((3,)), output_weight=np.zeros((3, 1)), output_bias=np.array([[logit]])
    )
    x, y = batch()
    loss, gradients, probabilities = torch_evaluate(parameters, x, y)
    result = evaluate_matched_backprop(parameters, x, y)
    assert result.loss == pytest.approx(loss.item(), abs=2e-12)
    assert result.loss > 400
    for actual, reference in zip(arrays(result.parameter_gradients), gradients, strict=True):
        np.testing.assert_allclose(actual, reference.detach().numpy(), atol=2e-14)
    np.testing.assert_array_equal(
        MatchedBackpropNetwork(parameters).predict_proba(x), probabilities.detach().numpy()
    )


def test_should_preserve_batch_mean_under_duplicate_rows():
    parameters, (x, y) = initial_parameters((3, 2)), batch()
    first = evaluate_matched_backprop(parameters, x, y)
    second = evaluate_matched_backprop(parameters, np.tile(x, (2, 1)), np.tile(y, (2, 1)))
    assert first.loss == pytest.approx(second.loss, abs=2e-14)
    for a, b in zip(
        arrays(first.parameter_gradients), arrays(second.parameter_gradients), strict=True
    ):
        np.testing.assert_allclose(a, b, atol=2e-14, rtol=2e-12)


def test_should_own_constructor_parameters_snapshots_predictions_and_gradients():
    parameters, (x, y) = initial_parameters((3, 2)), batch()
    original = initial_parameters((3, 2))
    model = MatchedBackpropNetwork(parameters)
    for a in arrays(parameters):
        a.fill(123)
    assert_parameters_equal(model.snapshot_parameters(), original)
    state = model.snapshot_state()
    assert type(state) is MatchedBackpropSnapshot
    assert type(state.format_version) is int and state.format_version == 1
    restored = MatchedBackpropNetwork(original)
    restored.restore_state(state)
    result = model.train_epoch(x, y, 0.03)
    detached = (
        *arrays(state.parameters),
        *arrays(result.parameter_gradients),
        model.predict_proba(x),
    )
    for index, a in enumerate(detached):
        assert a.dtype == np.float64
        for b in detached[index + 1 :]:
            assert not np.shares_memory(a, b)
        a.fill(999)
    assert_parameters_equal(restored.snapshot_parameters(), original)
    reference = MatchedBackpropNetwork(original)
    reference.train_epoch(x, y, 0.03)
    assert_parameters_equal(model.snapshot_parameters(), reference.snapshot_parameters())
    supplied = reference.snapshot_parameters()
    model.restore_parameters(supplied)
    for a in arrays(supplied):
        a.fill(-999)
    assert_parameters_equal(model.snapshot_parameters(), reference.snapshot_parameters())


def malformed_parameters() -> list[Any]:
    # Any is confined to deliberately malformed values at the public boundary.
    p: Any = initial_parameters((3, 2))
    return [
        None,
        arrays(p),
        replace(p, hidden_weights=[]),
        replace(p, hidden_weights=()),
        replace(p, hidden_biases=[]),
        replace(p, hidden_biases=p.hidden_biases[:-1]),
        replace(p, hidden_weights=(p.hidden_weights[0].astype(np.float32), p.hidden_weights[1])),
        replace(p, hidden_weights=(p.hidden_weights[0].tolist(), p.hidden_weights[1])),
        replace(p, hidden_weights=(np.zeros((2, 0)), p.hidden_weights[1])),
        replace(p, hidden_weights=(p.hidden_weights[0], np.ones((4, 2)))),
        replace(p, hidden_biases=(np.ones((3,)), p.hidden_biases[1])),
        replace(p, output_weight=np.ones((2, 2))),
        replace(p, output_bias=np.zeros((1,))),
        replace(p, output_bias=np.array([[np.inf]])),
        replace(p, hidden_biases=(np.full((1, 3), np.nan), p.hidden_biases[1])),
    ]


@pytest.mark.parametrize("parameters", malformed_parameters())
def test_should_refuse_malformed_parameter_families(parameters: Any):
    x, y = batch()
    with pytest.raises(ValueError):
        evaluate_matched_backprop(parameters, x, y)
    with pytest.raises(ValueError):
        MatchedBackpropNetwork(parameters)


@pytest.mark.parametrize(
    "bad_x,bad_y",
    [
        ([[1.0, 2.0]], np.ones((1, 1))),
        (np.ones((3, 2)), [[0.0], [0.5], [1.0]]),
        (np.ones((3, 2), dtype=np.float32), np.ones((3, 1))),
        (np.ones((3, 2)), np.ones((3, 1), dtype=np.int64)),
        (np.ones((3,)), np.ones((3, 1))),
        (np.ones((3, 2)), np.ones((3,))),
        (np.ones((0, 2)), np.ones((0, 1))),
        (np.ones((3, 3)), np.ones((3, 1))),
        (np.ones((3, 2)), np.ones((2, 1))),
        (np.full((3, 2), np.inf), np.ones((3, 1))),
        (np.ones((3, 2)), np.full((3, 1), np.nan)),
        (np.ones((3, 2)), np.full((3, 1), -0.1)),
        (np.ones((3, 2)), np.full((3, 1), 1.1)),
    ],
)
def test_should_refuse_malformed_batches_atomically(bad_x: Any, bad_y: Any):
    parameters = initial_parameters((3, 2))
    model = MatchedBackpropNetwork(parameters)
    with pytest.raises(ValueError):
        evaluate_matched_backprop(parameters, bad_x, bad_y)
    with pytest.raises(ValueError):
        model.train_epoch(bad_x, bad_y, 0.03)
    assert_parameters_equal(model.snapshot_parameters(), parameters)


@pytest.mark.parametrize(
    "rate", (True, False, 0, -0.1, float("nan"), float("inf"), np.float64(0.1), "0.1", 10**1000)
)
def test_should_refuse_invalid_learning_rates_atomically(rate: Any):
    parameters, (x, y) = initial_parameters((3, 2)), batch()
    model = MatchedBackpropNetwork(parameters)
    with pytest.raises(ValueError):
        model.train_epoch(x, y, rate)
    assert_parameters_equal(model.snapshot_parameters(), parameters)


@pytest.mark.parametrize("version", (True, False, 0, 2, 1.0, np.int64(1), "1"))
def test_should_refuse_unsupported_snapshot_versions_atomically(version: Any):
    parameters = initial_parameters((3, 2))
    model = MatchedBackpropNetwork(parameters)
    with pytest.raises(ValueError):
        model.restore_state(MatchedBackpropSnapshot(version, parameters))
    assert_parameters_equal(model.snapshot_parameters(), parameters)


def test_should_refuse_cross_family_and_cross_topology_restore_atomically():
    parameters = initial_parameters((3, 2))
    model = MatchedBackpropNetwork(parameters)
    wrong: Any
    for wrong in (None, parameters, CoupledSnapshot(1, parameters, (1.0, 1.0))):
        with pytest.raises(ValueError):
            model.restore_state(wrong)
    for wrong in (initial_parameters((2,)), initial_parameters((4, 2)), initial_parameters((3, 3))):
        with pytest.raises(ValueError):
            model.restore_parameters(wrong)
        with pytest.raises(ValueError):
            model.restore_state(MatchedBackpropSnapshot(1, wrong))
    bad: Any
    for bad in malformed_parameters():
        with pytest.raises(ValueError):
            model.restore_parameters(bad)
        with pytest.raises(ValueError):
            model.restore_state(MatchedBackpropSnapshot(1, bad))
    assert_parameters_equal(model.snapshot_parameters(), parameters)
    reference = MatchedBackpropNetwork(parameters)
    x, y = batch()
    model.train_epoch(x, y, 0.03)
    reference.train_epoch(x, y, 0.03)
    assert_parameters_equal(model.snapshot_parameters(), reference.snapshot_parameters())


def test_should_refuse_finite_parameter_candidate_with_nonfinite_forward_loss_atomically():
    # Old activations/logit are zero. A rate-2 update produces finite 8e307
    # hidden weights/biases, then tanh=1 makes output sum 3*8e307 overflow.
    parameters = CoupledParameters(
        (np.zeros((1, 3)),), (np.zeros((1, 3)),), np.full((3, 1), 8e307), np.zeros((1, 1))
    )
    model = MatchedBackpropNetwork(parameters)
    x, y = np.ones((1, 1)), np.ones((1, 1))
    old = evaluate_matched_backprop(parameters, x, y)
    candidate = tuple(
        a - 2 * g for a, g in zip(arrays(parameters), arrays(old.parameter_gradients), strict=True)
    )
    assert all(np.isfinite(a).all() for a in candidate)
    with pytest.raises(FloatingPointError):
        model.train_epoch(x, y, 2)
    assert_parameters_equal(model.snapshot_parameters(), parameters)


def test_should_refuse_nonfinite_parameter_candidate_atomically():
    parameters = CoupledParameters(
        (np.zeros((1, 1)),), (np.zeros((1, 1)),), np.ones((1, 1)), np.zeros((1, 1))
    )
    model = MatchedBackpropNetwork(parameters)
    x, y = np.array([[10.0]]), np.ones((1, 1))
    with pytest.raises(FloatingPointError):
        model.train_epoch(x, y, 1e308)
    assert_parameters_equal(model.snapshot_parameters(), parameters)
