"""ONE native public-history qualification: three SGD steps / 54 rows total.

The three paid steps are one raw trained-state refusal setup, one encoded TRAIN
pair, and a second encoded pair on an owned continuation fork. No environment,
teacher, independent oracle training, probe, or comparative experiment occurs.
"""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import torch_recurrent_episode_batch as paired
from src.adapters import torch_recurrent_public_history as history
from src.adapters import torch_train_episode_batch as arrays
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_episode import episode_arrays
from src.core.minigrid_public_history import encode_public_history_episode
from src.shared.torch_runtime import require_torch
from test_minigrid_learning_pilot import source_batch
from test_torch_feature_learner import assert_tree


@pytest.fixture
def cost(monkeypatch, record_property):
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    observed = SimpleNamespace(torch=torch, steps=0, rows=0, active_rows=None)
    original = torch.optim.SGD.step

    def step(owner, *args, **kwargs):
        assert observed.active_rows == 18, "unbudgeted physical SGD"
        assert observed.steps < 3, "physical qualification allowance is exhausted"
        observed.steps += 1
        observed.rows += observed.active_rows
        return original(owner, *args, **kwargs)

    monkeypatch.setattr(torch.optim.SGD, "step", step)
    yield observed
    record_property("physical_sgd_steps", observed.steps)
    record_property("physical_training_rows", observed.rows)
    assert (observed.steps, observed.rows) == (3, 54)


def paid(cost, operation):
    assert cost.active_rows is None
    cost.active_rows = 18
    try:
        return operation()
    finally:
        cost.active_rows = None


def pair_arrays(episodes):
    parts = [episode_arrays(episode) for episode in episodes]
    return np.concatenate([part[0] for part in parts]), np.concatenate([part[1] for part in parts])


def test_should_preserve_cold_raw_law_and_learn_with_owned_public_history(cost):
    torch = cost.torch
    raw = paired.TorchRecurrentEpisodeBatchLearner(TorchRecurrentPolicy(148, 8, 7, 17))
    cold = raw.snapshot_state()
    before = deepcopy(cold)
    torch_rng = torch.get_rng_state().clone()
    numpy_rng: Any = deepcopy(np.random.get_state())
    expanded = history.initialize_public_history_control(cold)
    assert expanded.policy == replace(cold.policy, feature_dim=296)
    assert (expanded.completed_updates, expanded.training_episodes, expanded.training_examples) == (
        0,
        0,
        0,
    )
    assert sum(value.numel() for value in cold.parameters.values()) == 1319
    assert sum(value.numel() for value in expanded.parameters.values()) == 2503
    assert torch.count_nonzero(expanded.parameters["weight_input"][:148]).item() == 0
    assert torch.equal(expanded.parameters["weight_input"][148:], cold.parameters["weight_input"])
    for name in cold.parameters:
        if name != "weight_input":
            assert torch.equal(expanded.parameters[name], cold.parameters[name])
        assert expanded.parameters[name].data_ptr() != cold.parameters[name].data_ptr()
    assert all(value is None for value in expanded.gradients.values())
    assert expanded.optimizer_state["state"] == {}
    assert_tree(torch, expanded.optimizer_state, cold.optimizer_state)
    assert_tree(torch, cold, before)
    assert_tree(torch, raw.snapshot_state(), before)
    assert torch.equal(torch_rng, torch.get_rng_state())
    current_numpy: Any = np.random.get_state()
    assert current_numpy[0] == numpy_rng[0] and current_numpy[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(current_numpy[1], numpy_rng[1])
    differentiable_input = deepcopy(cold)
    differentiable_input.parameters["weight_input"].requires_grad_(True)
    differentiable_before = deepcopy(differentiable_input)
    detached_control = history.initialize_public_history_control(differentiable_input)
    assert detached_control.parameters["weight_input"].requires_grad is False
    assert detached_control.parameters["weight_input"].is_leaf is True
    assert differentiable_input.parameters["weight_input"].requires_grad is True
    assert_tree(torch, differentiable_input, differentiable_before)

    blue, red = source_batch("A", "blue").episode, source_batch("A", "red").episode
    assert blue is not None and red is not None
    originals = tuple(
        replace((blue, red)[i % 2], episode_id=f"history-train-{i}") for i in range(4)
    )
    original_before = deepcopy(originals)
    encoded = tuple(encode_public_history_episode(episode) for episode in originals)
    raw_port = arrays.TorchTrainEpisodeBatchLearner(cold.policy, originals)
    raw_port.restore_state(cold)
    control = arrays.TorchTrainEpisodeBatchLearner(expanded.policy, encoded)
    control.restore_state(expanded)
    control_before = control.snapshot_state()
    for original, history_episode in zip(originals[:2], encoded[:2], strict=True):
        raw_features, _ = episode_arrays(original)
        history_features, _ = episode_arrays(history_episode)
        raw_probabilities, wider_probabilities = (
            raw.predict(raw_features),
            control.predict(history_features),
        )
        np.testing.assert_allclose(wider_probabilities, raw_probabilities, atol=1e-7, rtol=0)
        np.testing.assert_array_equal(
            np.argmax(wider_probabilities, axis=1), np.argmax(raw_probabilities, axis=1)
        )
        raw_session, history_session = raw_port.start_episode(), control.start_episode()
        assert raw_session.completed_steps == history_session.completed_steps == 0
        for row in range(9):
            np.testing.assert_allclose(
                history_session.predict(history_features[row : row + 1]),
                raw_session.predict(raw_features[row : row + 1]),
                atol=1e-7,
                rtol=0,
            )
        assert raw_session.completed_steps == history_session.completed_steps == 9
        assert_tree(torch, history_session.snapshot_state(), control_before)
    assert_tree(torch, control.snapshot_state(), control_before)

    for fault in ("extra", "missing_parameter", "nonfinite", "dtype", "optimizer", "cold_gradient"):
        invalid = deepcopy(cold)
        if fault == "extra":
            object.__setattr__(invalid, "unexpected", True)
        elif fault == "missing_parameter":
            del invalid.parameters["bias_output"]
        elif fault == "nonfinite":
            invalid.parameters["weight_input"][0, 0] = float("nan")
        elif fault == "dtype":
            invalid.parameters["weight_input"] = invalid.parameters["weight_input"].to(
                torch.float64
            )
        elif fault == "optimizer":
            invalid.optimizer_state["state"][0] = {}
        else:
            invalid.gradients["bias_output"] = torch.zeros_like(invalid.parameters["bias_output"])
        with pytest.raises(ValueError):
            history.initialize_public_history_control(invalid)
    with pytest.raises(ValueError, match="148"):
        history.initialize_public_history_control(expanded)

    trained = raw.fork()
    paid(cost, lambda: trained.train_episode_pair(originals[:2]))
    trained_before = trained.snapshot_state()
    with pytest.raises(ValueError, match="cold"):
        history.initialize_public_history_control(trained_before)
    assert_tree(torch, trained.snapshot_state(), trained_before)

    first_x, first_y = pair_arrays(encoded[:2])
    packet = AttributionBatch(
        "train",
        "A",
        tuple(f"{episode.episode_id}:step:{row}" for episode in encoded[:2] for row in range(9)),
        first_x,
        first_y,
        None,
        7,
    )
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 10.0)
    states: list[tuple[int, Any]] = []
    stream = AttributionStream(
        control,
        (packet,),
        (),
        checkpoint_updates=(0, 1),
        label_delay=1,
        budget=budget,
        observe_checkpoint=lambda point, state, batch: states.append((point, state)),
        training_only=True,
    )
    paid(cost, lambda: stream.run_until(1))
    first = stream.candidate_snapshot()
    result = stream.finish()
    assert result["evaluation_scope"] == "training_only"
    assert result["score_forward_calls"] == 0 and result["checkpoints"] == []
    assert result["completed_updates"] == budget.updates_completed == 1
    assert result["training_examples"] == 18
    assert len(result["applied_updates"]) == 1
    assert (
        result["applied_updates"][0]["diagnostic"]["definition"]
        == paired.TORCH_RECURRENT_EPISODE_PAIR_CE_ID
    )
    assert [point for point, _ in states] == [0, 1]
    assert_tree(torch, control.snapshot_state(), control_before)
    assert_tree(torch, stream._runtime.actor.snapshot_state().state, control_before)
    assert (first.completed_updates, first.training_episodes, first.training_examples) == (1, 2, 18)
    assert all(value is not None for value in first.gradients.values())
    assert len(first.optimizer_state["state"]) == 5
    assert torch.count_nonzero(first.parameters["weight_input"][:148]).item() > 0
    control.restore_state(first)
    continuation = control.fork()
    assert continuation is not control
    assert_tree(torch, continuation.snapshot_state(), first)
    second_x, second_y = pair_arrays(encoded[2:])
    paid(cost, lambda: continuation.train_batch(second_x, second_y))
    final = continuation.snapshot_state()
    assert (final.completed_updates, final.training_episodes, final.training_examples) == (2, 4, 36)
    assert_tree(torch, control.snapshot_state(), first)
    restored = arrays.TorchTrainEpisodeBatchLearner(expanded.policy, encoded)
    restored.restore_state(final)
    assert_tree(torch, restored.snapshot_state(), final)
    np.testing.assert_array_equal(
        restored.predict(episode_arrays(encoded[0])[0]),
        continuation.predict(episode_arrays(encoded[0])[0]),
    )
    detached = history.initialize_public_history_control(cold)
    detached.parameters["weight_input"][148, 0] += 2.0
    assert_tree(torch, cold, before)
    assert_tree(torch, raw.snapshot_state(), before)
    assert originals == original_before
