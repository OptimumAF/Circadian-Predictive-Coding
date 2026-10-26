"""Independent NumPy reward BPTT and owned on-policy native continuation."""

from dataclasses import replace
import pickle

import numpy as np
import pytest

from src.adapters.torch_recurrent_learner import TorchRecurrentLearner
from src.core.demonstration_episode import DemonstrationTransition
from src.core.policy_gradient_episode import PolicyGradientEpisode
from src.shared.torch_runtime import require_torch
from test_torch_feature_learner import assert_tree, tensors
from test_torch_recurrent_learner import NAMES, policy


@pytest.fixture
def torch():
    return require_torch()


def features():
    return np.array([[0.4, -0.2], [0.0, 0.3], [-0.1, 0.0]], dtype=np.float64)


def episode(probabilities, *, rewards=(0.25, -0.5, 1.0), role="train", classes=3, width=2):
    observations = features() if width == 2 else np.zeros((3, width), dtype=np.float64)
    transitions = tuple(
        DemonstrationTransition(
            index,
            tuple(float(value) for value in observations[index]),
            index,
            rewards[index],
            tuple(float(value) for value in observations[index + 1])
            if index < 2
            else (0.0,) * width,
            index == 2,
            classes,
        )
        for index in range(3)
    )
    return PolicyGradientEpisode(
        "native-reward-episode",
        "reward-C",
        role,
        transitions,
        tuple(tuple(float(value) for value in row) for row in probabilities),
        "terminal",
        classes,
    )


def observed(owned, **options):
    before = owned.snapshot_state()
    return episode(owned.predict(features()), **options), before


def numpy_reward_bptt(parameters, inputs, rewards, discount):
    values = {
        name: parameter.detach().numpy().astype(np.float64)
        for name, parameter in parameters.items()
    }
    hidden = np.zeros((1, values["weight_recurrent"].shape[0]))
    states, probability_rows = [hidden], []
    for row in inputs:
        hidden = np.tanh(
            row[None, :] @ values["weight_input"]
            + hidden @ values["weight_recurrent"]
            + values["bias_hidden"]
        )
        logits = hidden @ values["weight_output"] + values["bias_output"]
        masses = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probability_rows.append((masses / masses.sum(axis=1, keepdims=True))[0])
        states.append(hidden)
    probabilities = np.array(probability_rows)
    weights = np.array(
        [
            sum(discount**index * rewards[index] for index in range(start, len(rewards)))
            for start in range(len(rewards))
        ]
    )
    objective = -sum(
        weights[index] * np.log(probabilities[index, index]) for index in range(len(rewards))
    )
    gradients = {name: np.zeros_like(value) for name, value in values.items()}
    carried = np.zeros_like(hidden)
    for index in reversed(range(len(inputs))):
        d_logits = probabilities[index : index + 1].copy()
        d_logits[0, index] -= 1
        d_logits *= weights[index]
        gradients["weight_output"] += states[index + 1].T @ d_logits
        gradients["bias_output"] += d_logits
        d_hidden = d_logits @ values["weight_output"].T + carried
        d_activation = d_hidden * (1 - states[index + 1] ** 2)
        gradients["weight_input"] += inputs[index : index + 1].T @ d_activation
        gradients["weight_recurrent"] += states[index].T @ d_activation
        gradients["bias_hidden"] += d_activation
        carried = d_activation @ values["weight_recurrent"].T
    return float(objective), gradients


@pytest.mark.parametrize(
    "discount,rewards",
    [
        (0.0, (0.25, -0.5, 1.0)),
        (0.5, (0.25, -0.5, 1.0)),
        (1.0, (0.25, -0.5, 1.0)),
        (1.0, (-0.25, -0.5, -1.0)),
    ],
)
def test_should_match_independent_discounted_sum_bptt_gradients_and_native_sgd(
    torch, discount, rewards
):
    owned = TorchRecurrentLearner(policy())
    packet, before = observed(owned, rewards=rewards)
    torch_before, numpy_before = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state())
    record_before = pickle.dumps(packet)
    expected, gradients = numpy_reward_bptt(before.parameters, features(), rewards, discount)
    diagnostic = owned.train_reward_episode(packet, before, discount=discount)
    after = owned.snapshot_state()
    assert diagnostic.definition == "torch_recurrent_reward_logprob_sum_v1"
    assert diagnostic.value == pytest.approx(expected, rel=2e-6, abs=2e-7)
    assert after.completed_updates == 1 and after.training_examples == 3
    for index, name in enumerate(NAMES):
        np.testing.assert_allclose(
            after.gradients[name].numpy(), gradients[name], rtol=2e-5, atol=2e-7
        )
        expected_parameter = before.parameters[name].numpy() - 0.03 * gradients[name]
        np.testing.assert_allclose(
            after.parameters[name].numpy(), expected_parameter, rtol=2e-6, atol=4e-8
        )
        np.testing.assert_allclose(
            after.optimizer_state["state"][index]["momentum_buffer"].numpy(),
            gradients[name],
            rtol=2e-5,
            atol=2e-7,
        )
    assert pickle.dumps(packet) == record_before
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert pickle.dumps(np.random.get_state()) == numpy_before


def test_zero_reward_should_preserve_gradient_zero_but_spend_existing_native_momentum(torch):
    owned = TorchRecurrentLearner(policy())
    packet, before = observed(owned)
    owned.train_reward_episode(packet, before)
    zero_packet, moving = observed(owned, rewards=(0.0, 0.0, 0.0))
    diagnostic = owned.train_reward_episode(zero_packet, moving)
    after = owned.snapshot_state()
    assert diagnostic.value == 0 and after.completed_updates == 2 and after.training_examples == 6
    moved = False
    for index, name in enumerate(NAMES):
        assert torch.equal(after.gradients[name], torch.zeros_like(after.gradients[name]))
        expected_momentum = moving.optimizer_state["state"][index]["momentum_buffer"] * 0.9
        assert_tree(
            torch, after.optimizer_state["state"][index]["momentum_buffer"], expected_momentum
        )
        np.testing.assert_allclose(
            after.parameters[name].numpy(),
            (moving.parameters[name] - 0.03 * expected_momentum).numpy(),
            rtol=2e-6,
            atol=4e-8,
        )
        moved |= not torch.equal(after.parameters[name], moving.parameters[name])
    assert moved


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test"])
def test_should_refuse_reserved_roles_before_any_native_reward_update(monkeypatch, torch, role):
    owned = TorchRecurrentLearner(policy())
    before = owned.snapshot_state()
    packet = episode(np.full((3, 3), 1 / 3), role=role)
    monkeypatch.setattr(
        torch.optim.SGD, "step", lambda *args, **kwargs: pytest.fail("reserved role reached SGD")
    )
    with pytest.raises(ValueError):
        owned.train_reward_episode(packet, before)
    assert_tree(torch, owned.snapshot_state(), before)


@pytest.mark.parametrize(
    "fault",
    [
        "classes",
        "width",
        "probabilities",
        "stale",
        "discount",
        "behavior_shape",
        "behavior_policy",
        "behavior_optimizer",
        "behavior_gradient",
        "behavior_momentum",
        "behavior_clock",
    ],
)
def test_should_refuse_wrong_geometry_probabilities_or_complete_behavior_state_atomically(
    torch, fault
):
    owned = TorchRecurrentLearner(policy())
    packet, behavior = observed(owned)
    if fault in ("behavior_gradient", "behavior_momentum", "behavior_clock"):
        owned.train_reward_episode(packet, behavior)
        packet, behavior = observed(owned)
    if fault == "classes":
        packet = episode(np.full((3, 4), 0.25), classes=4)
    elif fault == "width":
        packet = episode(np.full((3, 3), 1 / 3), width=3)
    elif fault == "probabilities":
        packet = episode(np.tile(np.array([[0.8, 0.1, 0.1]]), (3, 1)))
    elif fault == "stale":
        owned.train_batch(features(), np.array([[0.0], [1.0], [2.0]]))
    elif fault == "behavior_shape":
        behavior.parameters["weight_recurrent"] = torch.zeros((2, 2))
    elif fault == "behavior_policy":
        object.__setattr__(behavior, "policy", replace(behavior.policy, seed=18))
    elif fault == "behavior_optimizer":
        behavior.optimizer_state["param_groups"][0]["lr"] = 0.04
    elif fault == "behavior_gradient":
        behavior.gradients["weight_input"][0, 0] += 0.1
    elif fault == "behavior_momentum":
        behavior.optimizer_state["state"][0]["momentum_buffer"][0, 0] += 0.1
    elif fault == "behavior_clock":
        object.__setattr__(behavior, "completed_updates", 2)
        object.__setattr__(behavior, "training_examples", 6)
    before = owned.snapshot_state()
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        owned.train_reward_episode(packet, behavior, discount=True if fault == "discount" else 1)
    assert_tree(torch, owned.snapshot_state(), before)


def test_should_own_reward_fork_and_restore_exact_next_update_with_original_momentum(torch):
    source = TorchRecurrentLearner(policy())
    packet, initial = observed(source)
    source.train_reward_episode(packet, initial, discount=0.5)
    saved = source.snapshot_state()
    child = source.fork()
    restored = TorchRecurrentLearner(policy())
    restored.restore_state(saved)
    next_packet, behavior = observed(source)
    results = [
        item.train_reward_episode(next_packet, behavior, discount=0.5)
        for item in (source, child, restored)
    ]
    assert results[0] == results[1] == results[2]
    expected = source.snapshot_state()
    for item in (child, restored):
        assert_tree(torch, item.snapshot_state(), expected)
    for value in tensors(torch, saved):
        value.zero_()
    assert_tree(torch, child.snapshot_state(), expected)
    assert_tree(torch, restored.snapshot_state(), expected)


def test_should_discard_staged_reward_mutation_when_native_sgd_raises_late(monkeypatch, torch):
    owned = TorchRecurrentLearner(policy())
    packet, before = observed(owned)
    original = torch.optim.SGD.step
    primary = RuntimeError("late recurrent reward SGD fault")

    def broken(*args, **kwargs):
        original(*args, **kwargs)
        raise primary

    monkeypatch.setattr(torch.optim.SGD, "step", broken)
    with pytest.raises(RuntimeError) as failure:
        owned.train_reward_episode(packet, before)
    assert failure.value is primary
    assert_tree(torch, owned.snapshot_state(), before)
