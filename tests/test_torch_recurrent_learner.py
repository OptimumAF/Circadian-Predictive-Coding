"""Episode BPTT and frozen recurrent sessions preserve native owned state."""

from copy import deepcopy
from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

import src.adapters.torch_recurrent_learner as port
from src.adapters.torch_recurrent_learner import TorchRecurrentLearner, TorchRecurrentPolicy
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.shared.torch_runtime import require_torch
from test_torch_feature_learner import assert_tree, forbidden, tensors


NAMES = ("weight_input", "weight_recurrent", "bias_hidden", "weight_output", "bias_output")


@pytest.fixture
def torch():
    return require_torch()


def policy(**changes):
    values: dict[str, Any] = dict(
        feature_dim=2,
        hidden_dim=3,
        num_classes=3,
        seed=17,
        learning_rate=0.03,
        momentum=0.9,
        max_episode_steps=4,
    )
    values.update(changes)
    return TorchRecurrentPolicy(**values)


def batch():
    return np.array([[0.7, -0.2], [0.0, 0.4], [-0.3, 0.0]]), np.array([[0.0], [1.0], [2.0]])


def episode(*, role="train", classes=3, dimension=2):
    def observation(index):
        return tuple((index + column) / 8.0 for column in range(dimension))

    return DemonstrationEpisode(
        "recurrent-episode",
        "A",
        role,
        tuple(
            DemonstrationTransition(
                index,
                observation(index),
                index % classes,
                float(index == 2),
                observation(index + 1),
                index == 2,
                classes,
            )
            for index in range(3)
        ),
        classes,
    )


def logits(torch, parameters, inputs):
    hidden = torch.zeros((1, parameters["weight_recurrent"].shape[0]), dtype=torch.float32)
    outputs = []
    for index in range(len(inputs)):
        hidden = torch.tanh(
            inputs[index : index + 1] @ parameters["weight_input"]
            + hidden @ parameters["weight_recurrent"]
            + parameters["bias_hidden"]
        )
        outputs.append(hidden @ parameters["weight_output"] + parameters["bias_output"])
    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("feature_dim", 0),
        ("feature_dim", True),
        ("hidden_dim", 33),
        ("num_classes", 1),
        ("num_classes", 17),
        ("seed", 2**31),
        ("learning_rate", 0.0),
        ("learning_rate", np.nan),
        ("momentum", True),
        ("momentum", 1.1),
        ("max_episode_steps", 0),
        ("max_episode_steps", 17),
    ],
)
def test_should_refuse_invalid_policy_before_loading_torch(monkeypatch, field, value):
    monkeypatch.setattr(port, "require_torch", forbidden)
    with pytest.raises(ValueError):
        policy(**{field: value})


def test_should_revalidate_tampered_policy_before_native_construction(monkeypatch):
    config = policy()
    object.__setattr__(config, "num_classes", True)
    monkeypatch.setattr(port, "require_torch", forbidden)
    with pytest.raises(ValueError):
        TorchRecurrentLearner(config)


def test_should_own_policy_before_caller_or_snapshot_mutation(torch):
    config = policy()
    owned = TorchRecurrentLearner(config)
    returned = owned.snapshot_state()
    object.__setattr__(config, "seed", 18)
    object.__setattr__(returned.policy, "seed", 19)
    assert owned.policy.seed == 17 and owned.snapshot_state().policy.seed == 17


@pytest.mark.parametrize(
    "fault",
    [
        "dtype",
        "width",
        "empty",
        "length",
        "feature_nan",
        "conversion",
        "soft",
        "class",
        "target_dtype",
    ],
)
def test_should_refuse_invalid_complete_episode_arrays_before_sgd(monkeypatch, torch, fault):
    owned = TorchRecurrentLearner(policy())
    features, targets = batch()
    if fault == "dtype":
        features = features.astype(np.float32)
    elif fault == "width":
        features = np.zeros((3, 3))
    elif fault == "empty":
        features, targets = np.zeros((0, 2)), np.zeros((0, 1))
    elif fault == "length":
        features, targets = np.zeros((5, 2)), np.zeros((5, 1))
    elif fault == "feature_nan":
        features[0, 0] = np.nan
    elif fault == "conversion":
        features[0, 0] = np.finfo(np.float64).max
    elif fault == "soft":
        targets[0, 0] = 0.5
    elif fault == "class":
        targets[0, 0] = 3
    else:
        targets = targets.astype(np.float32)
    before = owned.snapshot_state()
    monkeypatch.setattr(torch.optim.SGD, "step", forbidden)
    with pytest.raises(ValueError):
        owned.train_batch(features, targets)
    assert_tree(torch, owned.snapshot_state(), before)


@pytest.mark.parametrize("fault", ["final", "classes", "dimension", "length"])
def test_should_refuse_wrong_role_or_bound_episode_metadata_before_sgd(monkeypatch, torch, fault):
    owned = TorchRecurrentLearner(policy(max_episode_steps=2 if fault == "length" else 4))
    packet = episode(
        role="final_test" if fault == "final" else "train",
        classes=4 if fault == "classes" else 3,
        dimension=3 if fault == "dimension" else 2,
    )
    before = owned.snapshot_state()
    monkeypatch.setattr(torch.optim.SGD, "step", forbidden)
    with pytest.raises(ValueError):
        owned.train_episode(packet)
    assert_tree(torch, owned.snapshot_state(), before)


def test_should_match_independent_full_bptt_and_sgd_momentum_for_two_episodes(torch):
    torch_before = torch.get_rng_state().clone()
    numpy_before = pickle.dumps(np.random.get_state())
    owned = TorchRecurrentLearner(policy())
    initial = owned.snapshot_state()
    assert initial.completed_updates == initial.training_examples == 0
    assert initial.optimizer_state["state"] == {}
    assert all(value is None for value in initial.gradients.values())
    assert all(
        value.dtype == torch.float32 and value.device == torch.device("cpu")
        for value in initial.parameters.values()
    )
    parameters = {name: initial.parameters[name].clone().requires_grad_(True) for name in NAMES}
    optimizer = torch.optim.SGD(list(parameters.values()), lr=0.03, momentum=0.9)
    features, targets = batch()
    caller_before = pickle.dumps((features, targets))
    for index in range(1, 3):
        inputs = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(targets[:, 0], dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(logits(torch, parameters, inputs), labels)
        expected = float(loss.detach().item())
        loss.backward()
        assert bool((parameters["weight_input"].grad != 0).any().item())
        assert bool((parameters["weight_recurrent"].grad != 0).any().item())
        optimizer.step()
        actual = owned.train_batch(features, targets)
        assert actual.definition == "torch_recurrent_episode_cross_entropy_mean_v1"
        assert actual.value == expected
        state = owned.snapshot_state()
        assert state.format_version == 1 and state.policy == policy()
        assert state.completed_updates == index and state.training_examples == 3 * index
        assert_tree(
            torch, state.parameters, {name: value.detach() for name, value in parameters.items()}
        )
        assert_tree(
            torch, state.gradients, {name: value.grad for name, value in parameters.items()}
        )
        assert_tree(torch, state.optimizer_state, optimizer.state_dict())
        probability = (
            torch.softmax(logits(torch, parameters, inputs), dim=1)
            .detach()
            .numpy()
            .astype(np.float64)
        )
        returned = owned.predict(features)
        assert returned.shape == (3, 3) and returned.dtype == np.float64
        np.testing.assert_array_equal(returned, probability)
        returned.fill(999)
        assert_tree(torch, owned.snapshot_state(), state)
    assert pickle.dumps((features, targets)) == caller_before
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert pickle.dumps(np.random.get_state()) == numpy_before


def test_should_continue_complete_owned_fork_and_restore_with_next_episode_exact(torch):
    source = TorchRecurrentLearner(policy())
    source.train_batch(*batch())
    source.train_episode(episode())
    saved = source.snapshot_state()
    child = source.fork()
    restored = TorchRecurrentLearner(policy())
    restored.restore_state(saved)
    for item in (child, restored):
        assert_tree(torch, item.snapshot_state(), saved)
        for left, right in zip(
            tensors(torch, saved), tensors(torch, item.snapshot_state()), strict=True
        ):
            assert left.data_ptr() != right.data_ptr()
    diagnostics = [item.train_batch(*batch()) for item in (source, child, restored)]
    assert diagnostics[0] == diagnostics[1] == diagnostics[2]
    expected = source.snapshot_state()
    assert expected.completed_updates == 3 and expected.training_examples == 9
    for tensor in tensors(torch, saved):
        tensor.zero_()
    object.__setattr__(saved.policy, "num_classes", 4)
    for item in (source, child, restored):
        assert_tree(torch, item.snapshot_state(), expected)


def test_should_return_full_two_class_distribution_and_reset_batch_hidden(torch):
    owned = TorchRecurrentLearner(policy(num_classes=2))
    features, _ = batch()
    first, second = owned.predict(features), owned.predict(features)
    assert first.shape == (3, 2) and first.dtype == np.float64
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.sum(axis=1), 1, rtol=0, atol=2e-7)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "extra",
        "shape",
        "dtype",
        "nan",
        "policy",
        "updates",
        "examples",
        "gradient",
        "momentum",
        "optimizer_ids",
        "group_ids",
        "version",
        "extra_snapshot",
    ],
)
def test_should_atomically_refuse_corrupted_complete_recurrent_state(torch, fault):
    owned = TorchRecurrentLearner(policy())
    owned.train_batch(*batch())
    before = owned.snapshot_state()
    incoming = deepcopy(before)
    if fault == "missing":
        del incoming.parameters["bias_output"]
    elif fault == "extra":
        incoming.parameters["extra"] = torch.zeros(1)
    elif fault == "shape":
        incoming.parameters["weight_recurrent"] = torch.zeros((3, 2))
    elif fault == "dtype":
        incoming.parameters["weight_output"] = incoming.parameters["weight_output"].double()
    elif fault == "nan":
        incoming.parameters["bias_hidden"][0, 0] = np.nan
    elif fault == "policy":
        object.__setattr__(incoming, "policy", replace(incoming.policy, num_classes=4))
    elif fault == "updates":
        object.__setattr__(incoming, "completed_updates", True)
    elif fault == "examples":
        object.__setattr__(incoming, "training_examples", 5)
    elif fault == "gradient":
        incoming.gradients["weight_recurrent"] = torch.zeros((2, 2))
    elif fault == "momentum":
        incoming.optimizer_state["state"][1]["momentum_buffer"] = torch.zeros((2, 2))
    elif fault == "optimizer_ids":
        incoming.optimizer_state["state"][True] = incoming.optimizer_state["state"].pop(1)
    elif fault == "group_ids":
        incoming.optimizer_state["param_groups"][0]["params"][0] = False
    elif fault == "version":
        object.__setattr__(incoming, "format_version", 2)
    else:
        object.__setattr__(incoming, "unexpected", 1)
    torch_before = torch.get_rng_state().clone()
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        owned.restore_state(incoming)
    assert_tree(torch, owned.snapshot_state(), before)
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_should_refuse_valid_owned_snapshot_from_other_bound_class_policy(torch):
    source = TorchRecurrentLearner(policy(num_classes=2))
    destination = TorchRecurrentLearner(policy())
    before = destination.snapshot_state()
    with pytest.raises(ValueError):
        destination.restore_state(source.snapshot_state())
    assert_tree(torch, destination.snapshot_state(), before)


def test_should_rollback_late_sgd_failure_after_actual_staged_native_mutation(monkeypatch, torch):
    owned = TorchRecurrentLearner(policy())
    owned.train_batch(*batch())
    before = owned.snapshot_state()
    original = torch.optim.SGD.step
    primary = RuntimeError("recurrent staged SGD fault")

    def broken(*args, **kwargs):
        original(*args, **kwargs)
        raise primary

    monkeypatch.setattr(torch.optim.SGD, "step", broken)
    with pytest.raises(RuntimeError) as failure:
        owned.train_batch(*batch())
    assert failure.value is primary
    assert_tree(torch, owned.snapshot_state(), before)


def test_should_match_incremental_session_predictions_to_reset_batch_prefix_without_mutating_model(
    torch,
):
    owned = TorchRecurrentLearner(policy())
    owned.train_batch(*batch())
    before = owned.snapshot_state()
    session = owned.start_episode()
    features, _ = batch()
    predicted = []
    for index in range(len(features)):
        value = session.predict(features[index : index + 1])
        assert value.shape == (1, 3) and value.dtype == np.float64
        predicted.append(value.copy())
        np.testing.assert_array_equal(value, owned.predict(features[: index + 1])[-1:])
        assert session.completed_steps == index + 1
        value.fill(999)
        assert_tree(torch, session.snapshot_state(), before)
    np.testing.assert_array_equal(np.concatenate(predicted), owned.predict(features))
    assert_tree(torch, owned.snapshot_state(), before)


def test_should_use_only_prior_observations_to_distinguish_identical_current_inputs(torch):
    owned = TorchRecurrentLearner(policy())
    state = owned.snapshot_state()
    for value in state.parameters.values():
        value.zero_()
    state.parameters["weight_input"][0, 0] = 1
    state.parameters["weight_recurrent"][0, 0] = 1
    state.parameters["weight_output"][0, 0] = 1
    state.parameters["weight_output"][0, 1] = -1
    owned.restore_state(state)
    positive, negative, fresh = owned.start_episode(), owned.start_episode(), owned.start_episode()
    positive.predict(np.array([[1.0, 0.0]]))
    negative.predict(np.array([[-1.0, 0.0]]))
    current = np.array([[0.0, 0.0]])
    positive_probability = positive.predict(current)
    negative_probability = negative.predict(current)
    fresh_probability = fresh.predict(current)
    assert positive_probability[0, 0] > positive_probability[0, 1]
    assert negative_probability[0, 0] < negative_probability[0, 1]
    np.testing.assert_array_equal(fresh_probability, owned.predict(current))
    np.testing.assert_array_equal(positive_probability[:, [0, 1]], negative_probability[:, [1, 0]])
    assert_tree(torch, positive.snapshot_state(), owned.snapshot_state())
    assert_tree(torch, negative.snapshot_state(), owned.snapshot_state())


def test_should_stop_session_before_forward_after_original_episode_step_limit(monkeypatch, torch):
    owned = TorchRecurrentLearner(policy(max_episode_steps=1))
    session = owned.start_episode()
    before = session.snapshot_state()
    session.predict(np.array([[0.2, 0.3]]))
    monkeypatch.setattr(port, "_forward", forbidden)
    with pytest.raises(ValueError):
        session.predict(np.array([[0.2, 0.3]]))
    assert session.completed_steps == 1
    assert_tree(torch, session.snapshot_state(), before)


@pytest.mark.parametrize("fault", ["dtype", "shape", "nan", "conversion"])
def test_should_refuse_invalid_session_input_without_advancing_hidden_or_step_count(torch, fault):
    owned = TorchRecurrentLearner(policy())
    session = owned.start_episode()
    features = np.array([[0.2, 0.3]])
    if fault == "dtype":
        features = features.astype(np.float32)
    elif fault == "shape":
        features = np.zeros((2, 2))
    elif fault == "nan":
        features[0, 0] = np.nan
    else:
        features[0, 0] = np.finfo(np.float64).max
    with pytest.raises(ValueError):
        session.predict(features)
    assert session.completed_steps == 0
    valid = np.array([[0.2, 0.3]])
    np.testing.assert_array_equal(session.predict(valid), owned.start_episode().predict(valid))


def test_should_freeze_session_checkpoint_while_source_trains_and_other_sessions_advance(torch):
    owned = TorchRecurrentLearner(policy())
    frozen = owned.start_episode()
    checkpoint = frozen.snapshot_state()
    other = port.RecurrentEpisodePredictor(owned)
    other.predict(np.array([[0.8, -0.2]]))
    owned.train_batch(*batch())
    assert frozen.completed_steps == 0 and other.completed_steps == 1
    assert_tree(torch, frozen.snapshot_state(), checkpoint)
    np.testing.assert_array_equal(
        frozen.predict(np.array([[0.0, 0.0]])),
        TorchRecurrentLearner(policy()).predict(np.array([[0.0, 0.0]])),
    )
    for tensor in tensors(torch, checkpoint):
        tensor.zero_()
    assert_tree(torch, frozen.snapshot_state(), other.snapshot_state())


def test_should_discard_next_hidden_when_probability_conversion_raises_after_forward(
    monkeypatch, torch
):
    owned = TorchRecurrentLearner(policy())
    session, reference = owned.start_episode(), owned.start_episode()
    first, second = np.array([[0.8, -0.2]]), np.array([[0.0, 0.0]])
    session.predict(first)
    reference.predict(first)
    before = session.snapshot_state()
    original = port._probabilities
    primary = RuntimeError("post-forward probability fault")

    def broken(logit_values):
        original(logit_values)
        raise primary

    with monkeypatch.context() as context:
        context.setattr(port, "_probabilities", broken)
        with pytest.raises(RuntimeError) as failure:
            session.predict(second)
    assert failure.value is primary and session.completed_steps == 1
    assert_tree(torch, session.snapshot_state(), before)
    np.testing.assert_array_equal(session.predict(second), reference.predict(second))


def test_should_not_publish_inplace_hidden_mutation_when_forward_raises(monkeypatch, torch):
    owned = TorchRecurrentLearner(policy())
    session = owned.start_episode()
    before = session.snapshot_state()
    primary = RuntimeError("inplace hidden fault")

    def broken(parameters, features, hidden):
        hidden.fill_(999)
        raise primary

    with monkeypatch.context() as context:
        context.setattr(port, "_forward", broken)
        with pytest.raises(RuntimeError) as failure:
            session.predict(np.array([[0.2, 0.3]]))
    assert failure.value is primary and session.completed_steps == 0
    assert_tree(torch, session.snapshot_state(), before)
    np.testing.assert_array_equal(
        session.predict(np.array([[0.2, 0.3]])),
        owned.start_episode().predict(np.array([[0.2, 0.3]])),
    )


def test_should_refuse_observed_learned_mutation_in_frozen_session_without_hiding_it(
    monkeypatch, torch
):
    owned = TorchRecurrentLearner(policy())
    before = owned.snapshot_state()
    session = owned.start_episode()
    original = port._forward

    def mutate(parameters, features, hidden):
        parameters["weight_input"][0, 0] += 1
        return original(parameters, features, hidden)

    with monkeypatch.context() as context:
        context.setattr(port, "_forward", mutate)
        with pytest.raises(ValueError, match="changed"):
            session.predict(np.array([[0.2, 0.3]]))
    assert session.completed_steps == 0
    with pytest.raises(ValueError, match="changed"):
        session.snapshot_state()
    assert_tree(torch, owned.snapshot_state(), before)
