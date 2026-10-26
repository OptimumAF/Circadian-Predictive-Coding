"""Tiny synthetic native SGD controls; no environment or comparative learning.

The prospectively budgeted explicit invocation has three cases, eight physical
optimizer steps (one staged failure), seven commits and sixteen processed rows.
"""

import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import torch_recurrent_learner as native
from src.adapters.torch_demonstration_reward_learner import MomentumOnlyUpdate
from src.adapters.torch_recurrent_zero_loss import TORCH_RECURRENT_ZERO_LOSS_ID
from src.core.demonstration_episode import DemonstrationTransition
from src.core.policy_gradient_episode import PolicyGradientEpisode
from src.shared.torch_runtime import require_torch
from test_torch_feature_learner import assert_tree

NAMES = ("weight_input", "weight_recurrent", "bias_hidden", "weight_output", "bias_output")


@pytest.fixture
def native_cost(monkeypatch, record_property):
    torch = require_torch()
    cost = dict(
        optimizer_steps_attempted=0,
        committed_updates=0,
        committed_rows=0,
        source_graphs=0,
        control_forks=0,
        raw_forwards=0,
        raw_rows=0,
    )
    original = torch.optim.SGD.step

    def step(optimizer, *args, **kwargs):
        cost["optimizer_steps_attempted"] += 1
        return original(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.SGD, "step", step)
    yield cost
    for key, value in cost.items():
        record_property(key, value)


def warm_source(cost, *, momentum):
    policy = native.TorchRecurrentPolicy(3, 2, 2, 17, momentum=momentum)
    source = native.TorchRecurrentLearner(policy)
    cost["source_graphs"] += 1
    values = np.array([[0.2, -0.1, 0.3], [-0.2, 0.4, 0.1]], dtype=np.float64)
    targets = np.array([[0.0], [1.0]], dtype=np.float64)
    source.train_batch(values, targets)
    cost["committed_updates"] += 1
    cost["committed_rows"] += 2
    probabilities = source.predict(values)
    cost["raw_forwards"] += 1
    cost["raw_rows"] += 2
    observations = tuple(tuple(float(v) for v in row) for row in values)
    episode = PolicyGradientEpisode(
        "synthetic-momentum-unit",
        "synthetic-unit",
        "train",
        tuple(
            DemonstrationTransition(
                i,
                observations[i],
                i,
                (0.25, -0.5)[i],
                observations[1] if i == 0 else (0.0, 0.0, 0.0),
                i == 1,
                2,
            )
            for i in range(2)
        ),
        tuple(tuple(float(p) for p in row) for row in probabilities),
        "terminal",
        2,
    )
    control = source.fork()
    cost["control_forks"] += 1
    return source, control, episode


@pytest.mark.parametrize("momentum", [0.0, 0.9])
def test_should_apply_dense_zero_sgd_recurrence_without_changing_original_episode(
    native_cost, momentum
):
    torch = require_torch()
    source, control, episode = warm_source(native_cost, momentum=momentum)
    source_before = source.snapshot_state()
    original_episode = pickle.dumps(episode)
    packet = MomentumOnlyUpdate(episode)
    assert packet.episode is not episode and pickle.dumps(packet.episode) == original_episode
    if momentum:
        assert any(
            bool(torch.count_nonzero(state["momentum_buffer"]).item())
            for state in source_before.optimizer_state["state"].values()
        )
    moved = False
    for iteration in range(2):
        before = control.snapshot_state()
        expected: dict[str, Any] = {}
        expected_buffers: dict[str, Any] = {}
        with torch.no_grad():
            for index, name in enumerate(NAMES):
                parameter = before.parameters[name].detach().clone()
                if momentum:
                    buffer = before.optimizer_state["state"][index]["momentum_buffer"].clone()
                    buffer.mul_(momentum)
                    parameter.add_(buffer, alpha=-before.policy.learning_rate)
                    expected_buffers[name] = buffer
                expected[name] = parameter
        diagnostic = control.train_zero_loss_episode(packet.episode)
        native_cost["committed_updates"] += 1
        native_cost["committed_rows"] += 2
        after = control.snapshot_state()
        assert diagnostic.definition == TORCH_RECURRENT_ZERO_LOSS_ID and diagnostic.value == 0.0
        assert (
            after.completed_updates == 2 + iteration
            and after.training_examples == 4 + 2 * iteration
        )
        assert after.policy == before.policy
        assert after.optimizer_state["param_groups"] == before.optimizer_state["param_groups"]
        for index, name in enumerate(NAMES):
            gradient = after.gradients[name]
            assert gradient is not None and gradient.layout == torch.strided
            assert bool(torch.isfinite(gradient).all().item())
            assert torch.equal(gradient, torch.zeros_like(gradient))
            assert torch.equal(after.parameters[name], expected[name])
            moved |= not torch.equal(after.parameters[name], source_before.parameters[name])
            if momentum:
                assert torch.equal(
                    after.optimizer_state["state"][index]["momentum_buffer"], expected_buffers[name]
                )
        if not momentum:
            assert after.optimizer_state["state"] == {}
        assert_tree(torch, source.snapshot_state(), source_before)
        assert pickle.dumps(episode) == pickle.dumps(packet.episode) == original_episode
    assert moved is bool(momentum)
    assert native_cost["optimizer_steps_attempted"] == 3
    assert native_cost["committed_updates"] == 3 and native_cost["committed_rows"] == 6


def test_should_preserve_live_native_state_when_full_staged_validation_refuses(
    monkeypatch, native_cost
):
    torch = require_torch()
    source, control, episode = warm_source(native_cost, momentum=0.9)
    before, source_before = control.snapshot_state(), source.snapshot_state()
    payload, original_episode = control._payload, pickle.dumps(episode)
    capture = native._capture
    staged: list[Any] = []

    def refuse(parameters, optimizer, policy, updates, examples):
        result = capture(parameters, optimizer, policy, updates, examples)
        if updates == 2 and examples == 4:
            staged.append(result)
            raise RuntimeError("synthetic staged control refusal")
        return result

    monkeypatch.setattr(native, "_capture", refuse)
    with pytest.raises(RuntimeError, match="synthetic staged control refusal"):
        control.train_zero_loss_episode(episode)
    assert len(staged) == 1 and native_cost["optimizer_steps_attempted"] == 2
    assert all(
        gradient is not None and torch.equal(gradient, torch.zeros_like(gradient))
        for gradient in staged[0].gradients.values()
    )
    assert control._payload is payload
    assert_tree(torch, control.snapshot_state(), before)
    assert_tree(torch, source.snapshot_state(), source_before)
    assert pickle.dumps(episode) == original_episode
    assert native_cost["committed_updates"] == 1 and native_cost["committed_rows"] == 2
