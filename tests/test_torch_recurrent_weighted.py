"""Independent NumPy weighted BPTT on the unchanged owned recurrent state."""

from dataclasses import fields, is_dataclass
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters.torch_recurrent_learner import TorchRecurrentLearner, TorchRecurrentSnapshot
from src.adapters.torch_recurrent_weighted import TORCH_RECURRENT_WEIGHTED_CE_ID
from src.adapters.torch_demonstration_reward_learner import (
    ClassWeightedDemonstrationUpdate,
    TorchDemonstrationRewardLearner,
)
from src.app.actor_shadow import ActorShadowRuntime, StableActor
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_energy import Array
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.policy_gradient_episode import PolicyGradientEpisode
from src.shared.torch_runtime import require_torch
from test_torch_feature_learner import assert_tree, tensors
from test_torch_recurrent_learner import NAMES, policy


@pytest.fixture
def torch():
    return require_torch()


def episode(*, actions=(0, 0, 1, 2), role="train", classes=3, width=2):
    observations = [
        tuple(float((index + column) % 5 - 2) / 5 for column in range(width))
        for index in range(len(actions) + 1)
    ]
    return DemonstrationEpisode(
        "weighted-native-train",
        "weighted-C",
        role,
        tuple(
            DemonstrationTransition(
                index,
                observations[index],
                action,
                float(index == len(actions) - 1),
                observations[index + 1],
                index == len(actions) - 1,
                classes,
            )
            for index, action in enumerate(actions)
        ),
        classes,
    )


def arrays(packet):
    return np.array([step.observation for step in packet.transitions], dtype=np.float64), np.array(
        [[step.action] for step in packet.transitions], dtype=np.float64
    )


def numpy_weighted_bptt(parameters, packet, weights):
    values = {
        name: parameter.detach().numpy().astype(np.float64)
        for name, parameter in parameters.items()
    }
    inputs, targets = arrays(packet)
    labels = targets[:, 0].astype(np.int64)
    selected_weights = np.array(weights)[labels]
    coefficients = selected_weights / np.sum(selected_weights)
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
    objective = -np.sum(coefficients * np.log(probabilities[np.arange(len(labels)), labels]))
    gradients = {name: np.zeros_like(value) for name, value in values.items()}
    carried = np.zeros_like(hidden)
    for index in reversed(range(len(labels))):
        d_logits = probabilities[index : index + 1].copy()
        d_logits[0, labels[index]] -= 1
        d_logits *= coefficients[index]
        gradients["weight_output"] += states[index + 1].T @ d_logits
        gradients["bias_output"] += d_logits
        d_hidden = d_logits @ values["weight_output"].T + carried
        d_activation = d_hidden * (1 - states[index + 1] ** 2)
        gradients["weight_input"] += inputs[index : index + 1].T @ d_activation
        gradients["weight_recurrent"] += states[index].T @ d_activation
        gradients["bias_hidden"] += d_activation
        carried = d_activation @ values["weight_recurrent"].T
    return float(objective), gradients


def assert_close_state(torch, actual, expected):
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual)
        assert actual.dtype == expected.dtype and actual.device == expected.device
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=2e-6, atol=2e-7)
    elif is_dataclass(expected) and not isinstance(expected, type):
        assert type(actual) is type(expected)
        for field in fields(expected):
            assert_close_state(torch, getattr(actual, field.name), getattr(expected, field.name))
    elif type(expected) is dict:
        assert type(actual) is dict and actual.keys() == expected.keys()
        for key in expected:
            assert_close_state(torch, actual[key], expected[key])
    elif type(expected) in (tuple, list):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_close_state(torch, left, right)
    else:
        assert type(actual) is type(expected) and actual == expected


@pytest.mark.parametrize("warm_momentum", [False, True])
def test_should_match_independent_normalized_weighted_bptt_and_sgd_momentum(torch, warm_momentum):
    owned, packet = TorchRecurrentLearner(policy()), episode()
    if warm_momentum:
        owned.train_episode(packet)
    before = owned.snapshot_state()
    weights = (0.25, 1.0, 4.0)
    expected_loss, gradients = numpy_weighted_bptt(before.parameters, packet, weights)
    diagnostic = owned.train_weighted_episode(packet, weights)
    after = owned.snapshot_state()
    assert diagnostic.definition == TORCH_RECURRENT_WEIGHTED_CE_ID
    assert diagnostic.value == pytest.approx(expected_loss, rel=2e-6, abs=2e-7)
    assert after.policy == before.policy and after.format_version == before.format_version == 1
    assert after.completed_updates == before.completed_updates + 1
    assert after.training_examples == before.training_examples + 4
    assert np.any(gradients["weight_input"] != 0) and np.any(gradients["weight_recurrent"] != 0)
    for index, name in enumerate(NAMES):
        old_momentum = (
            before.optimizer_state["state"][index]["momentum_buffer"].numpy()
            if warm_momentum
            else np.zeros_like(gradients[name])
        )
        momentum = 0.9 * old_momentum + gradients[name]
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


def test_neutral_weights_should_match_old_mean_ce_and_complete_state_with_float32_tolerance(torch):
    weighted, ordinary, packet = (
        TorchRecurrentLearner(policy()),
        TorchRecurrentLearner(policy()),
        episode(),
    )
    actual = weighted.train_weighted_episode(packet, (1.0, 1.0, 1.0))
    expected = ordinary.train_episode(packet)
    assert (
        actual.definition == TORCH_RECURRENT_WEIGHTED_CE_ID
        and actual.definition != expected.definition
    )
    assert actual.value == pytest.approx(expected.value, rel=2e-6, abs=2e-7)
    assert_close_state(torch, weighted.snapshot_state(), ordinary.snapshot_state())


def test_common_weight_scaling_should_not_change_normalized_objective_or_native_clocks(torch):
    left, right, packet = (
        TorchRecurrentLearner(policy()),
        TorchRecurrentLearner(policy()),
        episode(),
    )
    first = left.train_weighted_episode(packet, (0.25, 1.0, 4.0))
    second = right.train_weighted_episode(packet, (1.75, 7.0, 28.0))
    assert first.value == pytest.approx(second.value, rel=2e-6, abs=2e-7)
    assert_close_state(torch, left.snapshot_state(), right.snapshot_state())
    assert left.completed_updates == right.completed_updates == 1
    assert left.training_examples == right.training_examples == 4


def test_absent_class_weight_should_not_contribute_to_loss_denominator_or_gradient(torch):
    left, right, packet = (
        TorchRecurrentLearner(policy()),
        TorchRecurrentLearner(policy()),
        episode(actions=(0, 0, 1, 1)),
    )
    assert left.train_weighted_episode(packet, (1.0, 2.0, 1.0)) == right.train_weighted_episode(
        packet, (1.0, 2.0, 999.0)
    )
    assert_tree(torch, left.snapshot_state(), right.snapshot_state())


@pytest.mark.parametrize(
    "weights",
    [
        [1.0, 1.0, 1.0],
        (1.0, 1.0),
        (True, 1.0, 1.0),
        (np.float64(1.0), 1.0, 1.0),
        (0.0, 1.0, 1.0),
        (float("nan"), 1.0, 1.0),
        (1e-7, 1.0, 1.0),
        (1e7, 1.0, 1.0),
    ],
)
def test_should_refuse_invalid_weights_before_snapshot_or_native_work(monkeypatch, weights):
    owned = TorchRecurrentLearner(policy())

    def forbidden():
        pytest.fail("invalid weights reached an owned model snapshot")

    monkeypatch.setattr(owned, "snapshot_state", forbidden)
    with pytest.raises(ValueError):
        owned.train_weighted_episode(episode(), weights)


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test"])
def test_should_refuse_nontrain_demonstrations_without_native_update(monkeypatch, torch, role):
    owned = TorchRecurrentLearner(policy())
    before = owned.snapshot_state()
    monkeypatch.setattr(
        torch.optim.SGD, "step", lambda *args, **kwargs: pytest.fail("reserved role reached SGD")
    )
    with pytest.raises(ValueError):
        owned.train_weighted_episode(episode(role=role), (1.0, 1.0, 1.0))
    assert_tree(torch, owned.snapshot_state(), before)


@pytest.mark.parametrize("fault", ["classes", "width", "length"])
def test_should_refuse_episode_geometry_before_publishing_any_native_field(torch, fault):
    owned = TorchRecurrentLearner(policy())
    packet = (
        episode(classes=4)
        if fault == "classes"
        else episode(width=3)
        if fault == "width"
        else episode(actions=(0, 0, 1, 2, 0))
    )
    before = owned.snapshot_state()
    with pytest.raises(ValueError):
        owned.train_weighted_episode(packet, (1.0,) * packet.num_classes)
    assert_tree(torch, owned.snapshot_state(), before)


def test_should_preserve_source_serving_actor_inputs_policy_and_global_rng(torch):
    torch_before, numpy_before = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state())
    source, packet, weights = TorchRecurrentLearner(policy()), episode(), (0.25, 1.0, 4.0)
    source.train_episode(packet)
    actor = StableActor(source, version="a0")
    before, actor_before = source.snapshot_state(), actor.snapshot_state()
    originals = pickle.dumps((packet, weights))
    child = source.fork()
    child.train_weighted_episode(packet, weights)
    assert_tree(torch, source.snapshot_state(), before)
    assert_tree(torch, actor.snapshot_state(), actor_before)
    prediction = actor.predict(arrays(packet)[0]).prediction
    prediction.fill(999)
    assert_tree(torch, actor.snapshot_state(), actor_before)
    assert pickle.dumps((packet, weights)) == originals
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert pickle.dumps(np.random.get_state()) == numpy_before


def test_should_continue_weighted_native_state_through_owned_reward_fork_and_restore(torch):
    source, packet = TorchRecurrentLearner(policy()), episode()
    source.train_episode(packet)
    source.train_weighted_episode(packet, (0.25, 1.0, 4.0))
    saved = source.snapshot_state()
    child, restored = source.fork(), TorchRecurrentLearner(policy())
    restored.restore_state(saved)
    for item in (child, restored):
        assert_tree(torch, item.snapshot_state(), saved)
        for left, right in zip(
            tensors(torch, saved), tensors(torch, item.snapshot_state()), strict=True
        ):
            assert left.data_ptr() != right.data_ptr()
    probabilities = source.predict(arrays(packet)[0])
    reward = PolicyGradientEpisode(
        "weighted-then-reward",
        "weighted-C",
        "train",
        packet.transitions,
        tuple(tuple(float(value) for value in row) for row in probabilities),
        "terminal",
        3,
    )
    diagnostics = [item.train_reward_episode(reward, saved) for item in (source, child, restored)]
    assert diagnostics[0] == diagnostics[1] == diagnostics[2]
    expected = source.snapshot_state()
    assert expected.completed_updates == 3 and expected.training_examples == 12
    for tensor in tensors(torch, saved):
        tensor.zero_()
    assert_tree(torch, child.snapshot_state(), expected)
    assert_tree(torch, restored.snapshot_state(), expected)


def test_should_discard_late_weighted_sgd_failure_after_staged_parameter_mutation(
    monkeypatch, torch
):
    owned, packet = TorchRecurrentLearner(policy()), episode()
    owned.train_episode(packet)
    before = owned.snapshot_state()
    original = torch.optim.SGD.step
    primary = RuntimeError("late weighted native SGD fault")

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise primary

    monkeypatch.setattr(torch.optim.SGD, "step", fail)
    with pytest.raises(RuntimeError) as failure:
        owned.train_weighted_episode(packet, (0.25, 1.0, 4.0))
    assert failure.value is primary
    assert_tree(torch, owned.snapshot_state(), before)


def test_should_refuse_staged_nonfinite_parameters_and_gradients_without_publishing(
    monkeypatch, torch
):
    owned, packet = TorchRecurrentLearner(policy()), episode()
    before = owned.snapshot_state()
    original = torch.optim.SGD.step

    def corrupt(optimizer, *args, **kwargs):
        result = original(optimizer, *args, **kwargs)
        parameter = optimizer.param_groups[0]["params"][0]
        with torch.no_grad():
            parameter[0, 0] = float("nan")
            parameter.grad[0, 0] = float("nan")
        return result

    monkeypatch.setattr(torch.optim.SGD, "step", corrupt)
    with pytest.raises(ValueError):
        owned.train_weighted_episode(packet, (0.25, 1.0, 4.0))
    assert_tree(torch, owned.snapshot_state(), before)


def test_should_release_one_typed_weighted_episode_through_original_runtime_and_budget(torch):
    source, packet = TorchDemonstrationRewardLearner(policy()), episode()
    update = ClassWeightedDemonstrationUpdate(packet)
    assert update.class_weights == (2 / 3, 4 / 3, 4 / 3)
    source_before = source.snapshot_state()
    expected = source.fork()
    diagnostic = expected.train_batch(update, packet.episode_id)
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    # The current runtime shares one feature generic across typed update packets
    # and prediction arrays; its opaque input is intentionally explicit here.
    runtime: ActorShadowRuntime[Any, str, Array, TorchRecurrentSnapshot] = ActorShadowRuntime(
        source,
        actor_version="a0",
        candidate_version="c0",
        clock=clock,
        budget=budget,
        max_experiences=1,
        max_consolidations=0,
    )
    actor_before = runtime.actor.snapshot_state()
    runtime.record_experience(
        Experience(
            "weighted-sample",
            packet.episode_id,
            0,
            "a0",
            update,
            "train",
            ExperiencePermissions(training=True),
        )
    )
    runtime.record_label(
        LabelArrival(
            "weighted-release",
            "weighted-sample",
            packet.episode_id,
            1,
            "a0",
            packet.episode_id,
            "train",
        )
    )
    clock.advance_to(1)
    receipts = runtime.train_ready(max_updates=1)
    assert len(receipts) == 1 and receipts[0].diagnostic == diagnostic
    assert receipts[0].diagnostic.definition == TORCH_RECURRENT_WEIGHTED_CE_ID
    assert receipts[0].event_id == "weighted-release" and receipts[0].update_number == 1
    assert budget.updates_completed == 1
    candidate = runtime.candidate_snapshot()
    assert candidate.wake_updates == 1 and candidate.state.completed_updates == 1
    assert candidate.state.training_examples == 4
    assert_tree(torch, candidate.state, expected.snapshot_state())
    assert_tree(torch, source.snapshot_state(), source_before)
    assert_tree(torch, runtime.actor.snapshot_state(), actor_before)
    assert runtime.train_ready(max_updates=1) == ()
    assert budget.updates_completed == 1
