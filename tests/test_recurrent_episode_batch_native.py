"""Bounded native qualification of an equal-weight two-episode CE update.

Prospective ledger: at most 13 physical SGD steps and 75 observed training rows,
including independently owned oracle steps. No environment or learner sweep.
The oracle uses its own explicit recurrent forward, not the adapter's forward.
"""

from copy import deepcopy
import json
import pickle
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from src.adapters import torch_recurrent_episode_batch as batch
from src.adapters import torch_recurrent_learner as legacy
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    episode_arrays,
)
from src.shared.torch_runtime import require_torch
from test_minigrid_learning_pilot import source_batch
from test_torch_feature_learner import assert_tree, tensors

NAMES = ("weight_input", "weight_recurrent", "bias_hidden", "weight_output", "bias_output")


@pytest.fixture
def cost(monkeypatch, record_property):
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    work = SimpleNamespace(torch=torch, steps=0, rows=0, active_rows=None, expected=(0, 0))
    original = torch.optim.SGD.step

    def step(owner, *args, **kwargs):
        assert type(work.active_rows) is int, "unbudgeted physical SGD step"
        work.steps += 1
        work.rows += work.active_rows
        return original(owner, *args, **kwargs)

    monkeypatch.setattr(torch.optim.SGD, "step", step)
    yield work
    record_property("physical_sgd_steps", work.steps)
    record_property("physical_training_rows", work.rows)
    assert (work.steps, work.rows) == work.expected


def paid(cost, rows, action):
    assert cost.active_rows is None
    cost.active_rows = rows
    try:
        return action()
    finally:
        cost.active_rows = None


def episode(name, values, actions):
    frames = tuple(tuple(float(value) for value in row) for row in values)
    return DemonstrationEpisode(
        name,
        "A",
        "train",
        tuple(
            DemonstrationTransition(
                index,
                frames[index],
                action,
                1.0 if index == len(actions) - 1 else 0.0,
                frames[index + 1],
                index == len(actions) - 1,
                3,
            )
            for index, action in enumerate(actions)
        ),
        3,
    )


def pair():
    return (
        episode("A-blue", ((0.2, -0.4), (0.7, 0.1), (-0.1, 0.2)), (0, 1)),
        episode("A-red", ((-0.3, 0.8), (0.4, -0.2), (-0.5, 0.3), (0.1, -0.1)), (2, 2, 1)),
    )


def learner(cost):
    owned = batch.TorchRecurrentEpisodeBatchLearner(legacy.TorchRecurrentPolicy(2, 3, 3, 17))
    state = owned.snapshot_state()
    explicit = (
        [[0.2, -0.1, 0.3], [0.4, 0.2, -0.2]],
        [[0.1, 0.2, 0.0], [0.0, -0.2, 0.1], [0.3, 0.0, 0.2]],
        [[0.05, -0.1, 0.02]],
        [[0.3, -0.1, 0.2], [-0.2, 0.4, 0.1], [0.1, 0.2, -0.3]],
        [[0.1, -0.2, 0.05]],
    )
    for name, values in zip(NAMES, explicit, strict=True):
        state.parameters[name].copy_(cost.torch.tensor(values, dtype=cost.torch.float32))
    owned.restore_state(state)
    return owned


def explicit_logits(torch, parameters, features, hidden=None):
    hidden = torch.zeros((1, 3), dtype=torch.float32) if hidden is None else hidden
    outputs = []
    for row in torch.from_numpy(features.astype(np.float32)):
        hidden = torch.tanh(
            row.reshape(1, -1) @ parameters["weight_input"]
            + hidden @ parameters["weight_recurrent"]
            + parameters["bias_hidden"]
        )
        outputs.append(hidden @ parameters["weight_output"] + parameters["bias_output"])
    return torch.cat(outputs, dim=0), hidden


def oracle_step(cost, state, episodes):
    torch = cost.torch
    parameters = {
        name: state.parameters[name].detach().clone().requires_grad_(True) for name in NAMES
    }
    optimizer = torch.optim.SGD(
        list(parameters.values()), lr=state.policy.learning_rate, momentum=state.policy.momentum
    )
    optimizer.load_state_dict(deepcopy(state.optimizer_state))
    losses, all_logits, all_targets = [], [], []
    for demonstration in episodes:
        features, targets = episode_arrays(demonstration)
        logits, _ = explicit_logits(torch, parameters, features)
        labels = torch.from_numpy(targets[:, 0].astype(np.int64))
        losses.append(torch.nn.functional.cross_entropy(logits, labels))
        all_logits.append(logits)
        all_targets.append(labels)
    loss = (losses[0] + losses[1]) / 2.0
    row_weighted = torch.nn.functional.cross_entropy(torch.cat(all_logits), torch.cat(all_targets))
    concatenated = np.concatenate([episode_arrays(item)[0] for item in episodes], axis=0)
    carried_logits, _ = explicit_logits(torch, parameters, concatenated)
    carried_loss = torch.nn.functional.cross_entropy(carried_logits, torch.cat(all_targets))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    paid(cost, sum(len(item.transitions) for item in episodes), optimizer.step)
    return dict(
        loss=float(loss.item()),
        episode_losses=tuple(float(value.item()) for value in losses),
        row_weighted=float(row_weighted.item()),
        concatenated=float(carried_loss.item()),
        parameters={name: value.detach().clone() for name, value in parameters.items()},
        gradients={name: value.grad.detach().clone() for name, value in parameters.items()},
        optimizer=deepcopy(optimizer.state_dict()),
    )


@pytest.mark.parametrize("warm", [False, True])
def test_should_match_independent_equal_episode_bptt_and_native_sgd_with_carried_momentum(
    cost, warm
):
    torch = cost.torch
    cost.expected = (3, 12) if warm else (2, 10)
    episodes = pair()
    if warm:
        single = legacy.TorchRecurrentLearner(legacy.TorchRecurrentPolicy(2, 3, 3, 17))
        paid(cost, 2, lambda: single.train_episode(episodes[0]))
        original_single = single.snapshot_state()
        owned = batch.TorchRecurrentEpisodeBatchLearner.from_single_episode_snapshot(
            original_single
        )
    else:
        owned = learner(cost)
        original_single = None
    before = owned.snapshot_state()
    if original_single is not None:
        assert (
            before.completed_updates
            == before.training_episodes
            == original_single.completed_updates
            == 1
        )
        assert before.training_examples == original_single.training_examples == 2
        for field in ("parameters", "optimizer_state", "gradients"):
            assert_tree(torch, getattr(before, field), getattr(original_single, field))
        assert all(value is not None for value in before.gradients.values())
        assert any(
            bool(torch.count_nonzero(value["momentum_buffer"]).item())
            for value in before.optimizer_state["state"].values()
        )
    immutable = pickle.dumps(episodes)
    rng, numpy_rng = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state())
    expected = oracle_step(cost, before, episodes)
    diagnostic = paid(cost, 5, lambda: owned.train_episode_pair(episodes))
    after = owned.snapshot_state()
    assert diagnostic.definition == batch.TORCH_RECURRENT_EPISODE_PAIR_CE_ID
    assert diagnostic.value == pytest.approx(expected["loss"], rel=2e-6, abs=1e-7)
    assert diagnostic.episode_ids == tuple(item.episode_id for item in episodes)
    np.testing.assert_allclose(
        diagnostic.episode_losses, expected["episode_losses"], rtol=2e-6, atol=1e-7
    )
    assert diagnostic.work["forward_calls_completed"] == 2
    assert diagnostic.work["forward_rows_completed"] == 5
    assert diagnostic.work["backward_calls_completed"] == 1
    assert diagnostic.work["backward_rows_completed"] == 5
    assert diagnostic.work["optimizer_steps"] == diagnostic.work["published_updates"] == 1
    if not warm:
        assert abs(expected["loss"] - expected["row_weighted"]) > 1e-5
        assert abs(expected["loss"] - expected["concatenated"]) > 1e-5
    for name in NAMES:
        torch.testing.assert_close(
            after.parameters[name], expected["parameters"][name], rtol=2e-6, atol=2e-7
        )
        torch.testing.assert_close(
            after.gradients[name], expected["gradients"][name], rtol=2e-6, atol=2e-7
        )
    for index in range(5):
        torch.testing.assert_close(
            after.optimizer_state["state"][index]["momentum_buffer"],
            expected["optimizer"]["state"][index]["momentum_buffer"],
            rtol=2e-6,
            atol=2e-7,
        )
    assert after.completed_updates == before.completed_updates + 1
    assert after.training_episodes == before.training_episodes + 2
    assert after.training_examples == before.training_examples + 5
    assert pickle.dumps(episodes) == immutable
    assert (
        torch.equal(torch.get_rng_state(), rng) and pickle.dumps(np.random.get_state()) == numpy_rng
    )
    if original_single is not None:
        assert_tree(torch, single.snapshot_state(), original_single)


def test_should_preserve_pair_objective_when_episode_order_is_reversed(cost):
    cost.expected = (2, 10)
    left, right = learner(cost), learner(cost)
    first = paid(cost, 5, lambda: left.train_episode_pair(pair()))
    second = paid(cost, 5, lambda: right.train_episode_pair(pair()[::-1]))
    assert first.value == second.value
    assert first.episode_losses == second.episode_losses[::-1]
    # A two-operand sum is symmetric; both independent zero-hidden graphs have
    # identical contributions. Small gradient accumulation order changes are allowed.
    for name in NAMES:
        cost.torch.testing.assert_close(
            left.snapshot_state().parameters[name],
            right.snapshot_state().parameters[name],
            rtol=2e-6,
            atol=2e-7,
        )


def test_should_roundtrip_fork_and_continue_without_aliasing_optimizer_or_gradients(cost):
    cost.expected = (3, 15)
    owned = learner(cost)
    paid(cost, 5, lambda: owned.train_episode_pair(pair()))
    before = owned.snapshot_state()
    fork = owned.fork()
    restored = batch.TorchRecurrentEpisodeBatchLearner(owned.policy)
    restored.restore_state(before)
    first = paid(cost, 5, lambda: fork.train_episode_pair(pair()))
    second = paid(cost, 5, lambda: restored.train_episode_pair(pair()))
    assert first.value == second.value
    assert_tree(cost.torch, fork.snapshot_state(), restored.snapshot_state())
    assert_tree(cost.torch, owned.snapshot_state(), before)
    for left, right in zip(
        tensors(cost.torch, owned.snapshot_state()),
        tensors(cost.torch, fork.snapshot_state()),
        strict=True,
    ):
        assert left.data_ptr() != right.data_ptr()
    before.parameters["weight_input"][0, 0] += 10
    before.gradients["bias_hidden"][0, 0] += 10
    before.optimizer_state["state"][0]["momentum_buffer"][0, 0] += 10
    assert (
        owned.snapshot_state().parameters["weight_input"][0, 0]
        != before.parameters["weight_input"][0, 0]
    )
    prediction = owned.predict(episode_arrays(pair()[0])[0])
    prediction[:] = 0.0
    np.testing.assert_allclose(owned.predict(episode_arrays(pair()[0])[0]).sum(axis=1), 1.0)


def test_should_count_two_nine_row_148_episodes_as_one_new_step_without_legacy_snapshot_coercion(
    cost,
):
    cost.expected = (1, 18)
    policy = legacy.TorchRecurrentPolicy(148, 8, 7, 17)
    owned = batch.TorchRecurrentEpisodeBatchLearner(policy)
    blue, red = source_batch("A", "blue").episode, source_batch("A", "red").episode
    assert blue is not None and red is not None
    paid(cost, 18, lambda: owned.train_episode_pair((blue, red)))
    state = owned.snapshot_state()
    assert (state.completed_updates, state.training_episodes, state.training_examples) == (1, 2, 18)
    assert type(state) is batch.TorchRecurrentEpisodeBatchSnapshot
    with pytest.raises(ValueError):
        legacy.TorchRecurrentLearner(policy).restore_state(cast(Any, state))
    with pytest.raises(ValueError):
        owned.restore_state(cast(Any, legacy.TorchRecurrentLearner(policy).snapshot_state()))


def test_should_enable_grad_only_within_training_and_restore_outer_no_grad_mode(cost):
    cost.expected = (1, 5)
    owned = learner(cost)
    with cost.torch.no_grad():
        assert not cost.torch.is_grad_enabled()
        diagnostic = paid(cost, 5, lambda: owned.train_episode_pair(pair()))
        assert not cost.torch.is_grad_enabled()
    assert np.isfinite(diagnostic.value) and owned.completed_updates == 1
    assert all(value is not None for value in owned.snapshot_state().gradients.values())


def test_should_refuse_inference_mode_before_sgd_and_leave_original_state_unchanged(cost):
    owned = learner(cost)
    before = owned.snapshot_state()
    with cost.torch.inference_mode():
        with pytest.raises((ValueError, RuntimeError)):
            owned.train_episode_pair(pair())
        assert cost.torch.is_inference_mode_enabled()
    assert_tree(cost.torch, owned.snapshot_state(), before)


def test_should_preserve_owner_when_finite_capture_fails_after_one_physical_step(monkeypatch, cost):
    cost.expected = (1, 5)
    owned = learner(cost)
    before = owned.snapshot_state()
    capture = batch._capture

    def fail_after_step(*args, **kwargs):
        if cost.steps:
            raise FloatingPointError("post-SGD finite capture refused")
        return capture(*args, **kwargs)

    monkeypatch.setattr(batch, "_capture", fail_after_step)
    with pytest.raises(batch.EpisodePairUpdateFailed) as error:
        paid(cost, 5, lambda: owned.train_episode_pair(pair()))
    monkeypatch.setattr(batch, "_capture", capture)
    assert "post-SGD finite capture refused" in str(error.value)
    assert error.value.work["optimizer_steps"] == 1 and error.value.work["published_updates"] == 0
    assert error.value.work["failure_work_may_be_unobserved"] is True
    assert_tree(cost.torch, owned.snapshot_state(), before)


def test_should_refuse_nonfinite_logits_before_sgd_without_publishing_partial_state(cost):
    owned = learner(cost)
    state = owned.snapshot_state()
    state.parameters["weight_input"].fill_(1.0)
    state.parameters["weight_recurrent"].fill_(0.0)
    state.parameters["bias_hidden"].fill_(1.0)
    state.parameters["weight_output"].fill_(cost.torch.finfo(cost.torch.float32).max)
    state.parameters["bias_output"].fill_(cost.torch.finfo(cost.torch.float32).max)
    owned.restore_state(state)
    before = owned.snapshot_state()
    with pytest.raises(batch.EpisodePairUpdateFailed):
        owned.train_episode_pair(pair())
    assert_tree(cost.torch, owned.snapshot_state(), before)


@pytest.mark.parametrize("fault", ["parameter", "gradient", "optimizer"])
def test_should_refuse_in_place_owner_changes_without_repair_or_publishing(
    monkeypatch, record_property, cost, fault
):
    owned = learner(cost)
    before = owned.snapshot_state()
    forward = legacy._forward
    armed = [True]

    def corrupt(*args, **kwargs):
        result = forward(*args, **kwargs)
        if armed[0]:
            armed[0] = False
            if fault == "parameter":
                with cost.torch.no_grad():
                    owned._payload[0]["weight_input"].add_(0.125)
            elif fault == "gradient":
                parameter = owned._payload[0]["weight_input"]
                parameter.grad = cost.torch.ones_like(parameter)
            else:
                owned._payload[1].param_groups[0]["momentum"] = 0.5
        return result

    monkeypatch.setattr(legacy, "_forward", corrupt)
    with pytest.raises(batch.EpisodePairUpdateFailed) as error:
        owned.train_episode_pair(pair())
    work = error.value.work
    record_property("staged_failure_work", json.dumps(work, sort_keys=True))
    assert work["forward_rows_completed"] == work["backward_rows_completed"] == 5
    assert work["optimizer_steps"] == work["published_updates"] == 0
    assert work["failure_work_may_be_unobserved"] is True
    assert owned.completed_updates == owned.training_episodes == owned.training_examples == 0
    if fault == "parameter":
        assert not cost.torch.equal(
            before.parameters["weight_input"], owned.snapshot_state().parameters["weight_input"]
        )
    elif fault == "gradient":
        assert owned._payload[0]["weight_input"].grad is not None
    else:
        assert owned._payload[1].param_groups[0]["momentum"] == 0.5
