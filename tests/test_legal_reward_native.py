"""Independent NumPy masked-policy BPTT, native atomicity and explicit reward ports."""

import math
import pickle

import numpy as np
import pytest

from src.adapters.torch_demonstration_reward_learner import TorchDemonstrationRewardLearner
from src.adapters.torch_recurrent_learner import TorchRecurrentLearner
from src.adapters.torch_recurrent_reward import TORCH_RECURRENT_LEGAL_REWARD_ID
from src.adapters.torch_reward_update_learner import LegalRewardUpdate, TorchRewardUpdateLearner
from src.core.legal_policy_gradient import LegalPolicyGradientEpisode
from src.shared.torch_runtime import require_torch
from test_torch_feature_learner import assert_tree, tensors
from test_torch_recurrent_learner import NAMES, policy
from test_torch_recurrent_reward import episode, features


VARYING = ((0, 2), (1,), (0, 1, 2))
FULL = ((0, 1, 2),) * 3
SINGLETONS = ((0,), (1,), (2,))


@pytest.fixture
def torch():
    return require_torch()


def conditioned(raw, supports):
    return tuple(
        tuple(
            float(row[c]) / math.fsum(row[i] for i in legal) if c in legal else 0.0
            for c in range(3)
        )
        for row, legal in zip(raw, supports, strict=True)
    )


def observed(owned, *, supports=VARYING, rewards=(0.25, -0.5, 1.0), role="train"):
    before = owned.snapshot_state()
    raw = tuple(tuple(float(value) for value in row) for row in owned.predict(features()))
    packet = LegalPolicyGradientEpisode(
        episode(conditioned(raw, supports), rewards=rewards, role=role), supports, raw
    )
    return packet, before


def numpy_legal_bptt(parameters, inputs, rewards, discount, supports):
    values = {name: value.detach().numpy().astype(np.float64) for name, value in parameters.items()}
    hidden = np.zeros((1, values["weight_recurrent"].shape[0]))
    states, probability_rows = [hidden], []
    for row, legal in zip(inputs, supports, strict=True):
        hidden = np.tanh(
            row[None, :] @ values["weight_input"]
            + hidden @ values["weight_recurrent"]
            + values["bias_hidden"]
        )
        logits = hidden @ values["weight_output"] + values["bias_output"]
        selected = logits[0, list(legal)]
        masses = np.exp(selected - max(selected))
        row_probabilities = np.zeros(3)
        row_probabilities[list(legal)] = masses / masses.sum()
        probability_rows.append(row_probabilities)
        states.append(hidden)
    probabilities = np.array(probability_rows)
    weights = np.array(
        [sum(discount**k * rewards[k] for k in range(t, len(rewards))) for t in range(len(rewards))]
    )
    loss = -sum(weights[t] * np.log(probabilities[t, t]) for t in range(len(rewards)))
    gradients = {name: np.zeros_like(value) for name, value in values.items()}
    carried = np.zeros_like(hidden)
    for index in reversed(range(len(inputs))):
        derivative = probabilities[index : index + 1].copy()
        derivative[0, index] -= 1
        derivative *= weights[index]
        gradients["weight_output"] += states[index + 1].T @ derivative
        gradients["bias_output"] += derivative
        activation = (derivative @ values["weight_output"].T + carried) * (
            1 - states[index + 1] ** 2
        )
        gradients["weight_input"] += inputs[index : index + 1].T @ activation
        gradients["weight_recurrent"] += states[index].T @ activation
        gradients["bias_hidden"] += activation
        carried = activation @ values["weight_recurrent"].T
    return float(loss), gradients


@pytest.mark.parametrize(
    "discount,supports,rewards,warm",
    [
        (0.0, VARYING, (0.25, -0.5, 1.0), False),
        (0.5, VARYING, (0.25, -0.5, 1.0), True),
        (1.0, FULL, (0.25, -0.5, 1.0), False),
        (1.0, SINGLETONS, (0.25, -0.5, 1.0), True),
        (1.0, VARYING, (-0.25, -0.5, -1.0), False),
    ],
)
def test_should_match_independent_masked_sum_bptt_and_existing_sgd_momentum(
    torch, discount, supports, rewards, warm
):
    owned = TorchRecurrentLearner(policy())
    if warm:
        owned.train_batch(features(), np.array([[0.0], [1.0], [2.0]]))
    packet, before = observed(owned, supports=supports, rewards=rewards)
    immutable = pickle.dumps(packet)
    torch_before, numpy_before = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state())
    loss, gradients = numpy_legal_bptt(before.parameters, features(), rewards, discount, supports)
    diagnostic = owned.train_reward_episode(packet, before, discount=discount)
    after = owned.snapshot_state()
    assert diagnostic.definition == TORCH_RECURRENT_LEGAL_REWARD_ID
    assert diagnostic.value == pytest.approx(loss, rel=2e-6, abs=2e-7)
    assert after.completed_updates == before.completed_updates + 1
    assert after.training_examples == before.training_examples + 3
    for index, name in enumerate(NAMES):
        previous = (
            before.optimizer_state["state"][index]["momentum_buffer"].numpy()
            if warm
            else np.zeros_like(gradients[name])
        )
        momentum = 0.9 * previous + gradients[name]
        np.testing.assert_allclose(
            after.gradients[name].numpy(), gradients[name], rtol=2e-5, atol=2e-7
        )
        np.testing.assert_allclose(
            after.optimizer_state["state"][index]["momentum_buffer"].numpy(),
            momentum,
            rtol=2e-5,
            atol=2e-7,
        )
        np.testing.assert_allclose(
            after.parameters[name].numpy(),
            before.parameters[name].numpy() - 0.03 * momentum,
            rtol=2e-6,
            atol=4e-8,
        )
    assert pickle.dumps(packet) == immutable
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert pickle.dumps(np.random.get_state()) == numpy_before


def test_full_support_should_preserve_original_unmasked_complete_native_update(torch):
    legal, ordinary = TorchRecurrentLearner(policy()), TorchRecurrentLearner(policy())
    packet, before = observed(legal, supports=FULL)
    masked = legal.train_reward_episode(packet, before, discount=0.5)
    original = ordinary.train_reward_episode(packet.episode, before, discount=0.5)
    assert masked.value == original.value and masked.definition != original.definition
    assert_tree(torch, legal.snapshot_state(), ordinary.snapshot_state())


@pytest.mark.parametrize("fault", ["raw", "conditioned", "gradient", "momentum", "discount"])
def test_should_refuse_wrong_behavior_law_or_full_checkpoint_without_publishing_native_changes(
    torch, fault
):
    owned = TorchRecurrentLearner(policy())
    if fault in ("gradient", "momentum"):
        owned.train_batch(features(), np.array([[0.0], [1.0], [2.0]]))
    packet, behavior = observed(owned)
    if fault == "raw":
        raw = ((0.7, 0.2, 0.1),) * 3
        packet = LegalPolicyGradientEpisode(episode(conditioned(raw, VARYING)), VARYING, raw)
    elif fault == "conditioned":
        object.__setattr__(packet.episode, "behavior_probabilities", ((0.2, 0.3, 0.5),) * 3)
    elif fault == "gradient":
        behavior.gradients["weight_input"][0, 0] += 0.125
    elif fault == "momentum":
        behavior.optimizer_state["state"][0]["momentum_buffer"][0, 0] += 0.125
    before = owned.snapshot_state()
    with pytest.raises(ValueError):
        owned.train_reward_episode(packet, behavior, discount=True if fault == "discount" else 1.0)
    assert_tree(torch, owned.snapshot_state(), before)


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test"])
def test_should_refuse_reserved_roles_before_native_step(monkeypatch, torch, role):
    owned = TorchRecurrentLearner(policy())
    packet, before = observed(owned, role=role)
    monkeypatch.setattr(
        torch.optim.SGD, "step", lambda *args, **kwargs: pytest.fail("reserved role reached SGD")
    )
    with pytest.raises(ValueError):
        owned.train_reward_episode(packet, before)
    assert_tree(torch, owned.snapshot_state(), before)


def test_should_continue_legal_reward_from_owned_fork_and_restore_with_full_optimizer_state(torch):
    source = TorchRecurrentLearner(policy())
    packet, initial = observed(source)
    source.train_reward_episode(packet, initial, discount=0.5)
    saved = source.snapshot_state()
    child, restored = source.fork(), TorchRecurrentLearner(policy())
    restored.restore_state(saved)
    packet, behavior = observed(source, rewards=(-0.25, 0.5, 1.0))
    diagnostics = [
        owner.train_reward_episode(packet, behavior, discount=0.5)
        for owner in (source, child, restored)
    ]
    assert diagnostics[0] == diagnostics[1] == diagnostics[2]
    expected = source.snapshot_state()
    assert_tree(torch, child.snapshot_state(), expected)
    assert_tree(torch, restored.snapshot_state(), expected)
    for tensor in tensors(torch, saved):
        tensor.zero_()
    assert_tree(torch, child.snapshot_state(), expected)


def test_should_discard_staged_masked_sgd_changes_when_optimizer_raises_late(monkeypatch, torch):
    owned = TorchRecurrentLearner(policy())
    packet, before = observed(owned)
    original, primary = torch.optim.SGD.step, RuntimeError("late legal reward SGD failure")

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise primary

    monkeypatch.setattr(torch.optim.SGD, "step", fail)
    with pytest.raises(RuntimeError) as caught:
        owned.train_reward_episode(packet, before)
    assert caught.value is primary
    assert_tree(torch, owned.snapshot_state(), before)


@pytest.mark.parametrize("fault", ["zero_legal_mass", "zero_selected_mass"])
def test_should_refuse_forged_tiny_raw_mass_even_within_absolute_behavior_tolerance(torch, fault):
    owned = TorchRecurrentLearner(policy())
    altered = owned.snapshot_state()
    for parameter in altered.parameters.values():
        parameter.zero_()
    altered.parameters["bias_output"][0] = torch.tensor(
        [0.0, -1000.0, -1000.0 if fault == "zero_legal_mass" else -0.2]
    )
    owned.restore_state(altered)
    before = owned.snapshot_state()
    actual = owned.predict(features())
    assert np.all(actual[:, 1] == 0)
    raw: tuple[tuple[float, ...], ...]
    if fault == "zero_legal_mass":
        assert np.all(actual[:, 2] == 0)
        raw = ((1.0 - 2e-13, 1e-13, 1e-13),) * 3
    else:
        assert np.all(actual[:, 2] > 0)
        raw = tuple((float(row[0]) - 1e-13, 1e-13, float(row[2])) for row in actual)
    # Tiny raw errors fit the absolute comparison tolerance. The legal law
    # must still be derived from the actual checkpoint's public probabilities.
    supports = ((0,), (1, 2), (1, 2))
    packet = LegalPolicyGradientEpisode(episode(conditioned(raw, supports)), supports, raw)
    with pytest.raises(ValueError):
        owned.train_reward_episode(packet, before)
    assert_tree(torch, owned.snapshot_state(), before)


@pytest.mark.parametrize("factory", [TorchRewardUpdateLearner, TorchDemonstrationRewardLearner])
def test_should_release_owned_legal_reward_dto_through_both_native_ports(torch, factory):
    owner = factory(policy())
    packet, before = observed(owner)
    update = LegalRewardUpdate(packet, before)
    identifier = update.episode.episode.episode_id
    object.__setattr__(packet.episode, "episode_id", "caller-mutated")
    for tensor in tensors(torch, before):
        tensor.zero_()
    diagnostic = owner.train_batch(update, identifier)
    assert diagnostic.definition == TORCH_RECURRENT_LEGAL_REWARD_ID
    assert owner.completed_updates == 1 and owner.training_examples == 3


@pytest.mark.parametrize("factory", [TorchRewardUpdateLearner, TorchDemonstrationRewardLearner])
def test_should_refuse_wrong_release_identifier_in_both_typed_ports_before_sgd(
    monkeypatch, torch, factory
):
    owner = factory(policy())
    packet, before = observed(owner)
    update = LegalRewardUpdate(packet, before)
    monkeypatch.setattr(
        torch.optim.SGD,
        "step",
        lambda *args, **kwargs: pytest.fail("unreleased legal trajectory reached SGD"),
    )
    with pytest.raises(ValueError):
        owner.train_batch(update, "wrong-release")
    assert_tree(torch, owner.snapshot_state(), before)
