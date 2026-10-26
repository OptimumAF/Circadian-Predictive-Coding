"""Owned CPU feature-head state, native parity, and component continuation.

The replay continuation case covers a model and a copied buffer only. It does
not claim restoration of an application runtime, receipts, budgets or authority.
"""

from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
import pickle
from typing import Any

import numpy as np
import pytest

import src.adapters.torch_feature_learner as port
from src.adapters.torch_feature_learner import (
    TorchFeatureLearner,
    TorchFeaturePolicy,
    TorchFeatureSnapshot,
)
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.resnet50_variants import (
    BackpropMLPHead,
    CircadianHeadConfig,
    CircadianPredictiveCodingHead,
    PredictiveCodingHead,
    TORCH_PC_ENERGY_ID,
)
from src.core.shared_replay_schedule import ReplayAllocation, SharedReplayBuffer
from src.shared.torch_runtime import require_torch


METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")
NAMES = ("weight_feature_hidden", "bias_hidden", "weight_hidden_output", "bias_output")


@pytest.fixture
def torch():
    return require_torch()


def policy(method="pc", **changes):
    return TorchFeaturePolicy(
        **{
            "method": method,
            "feature_dim": 2,
            "hidden_dim": 3,
            "seed": 17,
            "learning_rate": 0.03,
            "inference_steps": 2,
            "inference_learning_rate": 0.04,
            "momentum": 0.9,
            **changes,
        }
    )


def batch():
    return np.array([[0.3, -0.2], [-0.5, 0.4]]), np.array([[1.0], [0.0]])


def assert_tree(torch, actual, expected):
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual)
        assert actual.dtype == expected.dtype and actual.device == expected.device
        assert actual.shape == expected.shape and torch.equal(actual, expected)
    elif is_dataclass(expected) and not isinstance(expected, type):
        assert type(actual) is type(expected)
        for field in fields(expected):
            assert_tree(torch, getattr(actual, field.name), getattr(expected, field.name))
    elif type(expected) is dict:
        assert type(actual) is dict and actual.keys() == expected.keys()
        for name in expected:
            assert_tree(torch, actual[name], expected[name])
    elif type(expected) in (tuple, list):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_tree(torch, left, right)
    else:
        assert type(actual) is type(expected) and actual == expected


def tensors(torch, value):
    if torch.is_tensor(value):
        return [value]
    if is_dataclass(value) and not isinstance(value, type):
        return [
            tensor
            for field in fields(value)
            for tensor in tensors(torch, getattr(value, field.name))
        ]
    if type(value) is dict:
        return [tensor for child in value.values() for tensor in tensors(torch, child)]
    if type(value) in (tuple, list):
        return [tensor for child in value for tensor in tensors(torch, child)]
    return []


def reference_head(torch, selected):
    arguments = (selected.feature_dim, selected.hidden_dim, 2, torch.device("cpu"), selected.seed)
    if selected.method == "bp":
        head = BackpropMLPHead(*arguments)
        return head, torch.optim.SGD(
            head.trainable_parameters(), lr=selected.learning_rate, momentum=selected.momentum
        )
    if selected.method == "pc":
        return PredictiveCodingHead(*arguments), None
    config = (
        CircadianHeadConfig.matched_pc_control()
        if selected.method == "neutral_cpc"
        else CircadianHeadConfig()
    )
    return CircadianPredictiveCodingHead(
        *arguments,
        config=config,
        min_hidden_dim=selected.hidden_dim,
        max_hidden_dim=selected.hidden_dim,
    ), None


def native_state(head, method):
    if method in ("neutral_cpc", "gated_cpc"):
        return head.snapshot_state()
    result = {
        "format_version": 1,
        "feature_dim": 2,
        "hidden_dim": 3,
        "num_classes": 2,
        "device": "cpu",
        **{name: getattr(head, name).detach().clone() for name in NAMES},
    }
    if method == "pc":
        result.update(traffic_sum=head._traffic_sum.clone(), traffic_steps=head._traffic_steps)
    return result


def forbidden(*args, **kwargs):
    pytest.fail("invalid feature input reached native training")


@pytest.mark.parametrize(
    "field, value",
    [
        ("method", "unknown"),
        ("feature_dim", 0),
        ("feature_dim", True),
        ("hidden_dim", 33),
        ("seed", 2**31),
        ("learning_rate", 0.0),
        ("inference_steps", 65),
        ("inference_learning_rate", float("inf")),
        ("momentum", 1.1),
        ("momentum", True),
    ],
)
def test_should_refuse_invalid_policy_before_loading_torch(monkeypatch, field, value):
    monkeypatch.setattr(port, "require_torch", forbidden)
    with pytest.raises(ValueError):
        policy(**{field: value})


def test_should_own_policy_and_revalidate_exact_constructor_input_before_torch(monkeypatch):
    invalid: Any = object()
    tampered = policy()
    object.__setattr__(tampered, "feature_dim", True)
    with monkeypatch.context() as context:
        context.setattr(port, "require_torch", forbidden)
        for value in (invalid, tampered):
            with pytest.raises(ValueError):
                TorchFeatureLearner(value)
    caller = policy()
    owned = TorchFeatureLearner(caller)
    returned = owned.snapshot_state().policy
    assert returned == caller and returned is not caller
    object.__setattr__(caller, "seed", 18)
    object.__setattr__(returned, "seed", 19)
    assert owned.snapshot_state().policy.seed == 17


@pytest.mark.parametrize(
    "fault", ["dtype", "empty", "width", "soft", "class", "target_nan", "conversion", "feature_nan"]
)
def test_should_refuse_invalid_or_nonrepresentable_training_data_before_native_work(
    monkeypatch, torch, fault
):
    owned = TorchFeatureLearner(policy())
    features, targets = batch()
    if fault == "dtype":
        features = features.astype(np.float32)
    elif fault == "empty":
        features, targets = np.zeros((0, 2)), np.zeros((0, 1))
    elif fault == "width":
        features = np.zeros((2, 3))
    elif fault == "soft":
        targets[0, 0] = 0.5
    elif fault == "class":
        targets[0, 0] = 2
    elif fault == "target_nan":
        targets[0, 0] = float("nan")
    elif fault == "conversion":
        features[0, 0] = np.finfo(np.float64).max
    else:
        features[0, 0] = float("nan")
    before = owned.snapshot_state()
    monkeypatch.setattr(PredictiveCodingHead, "train_step", forbidden)
    monkeypatch.setattr(CircadianPredictiveCodingHead, "train_step", forbidden)
    monkeypatch.setattr(torch.optim.SGD, "step", forbidden)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError):
            owned.train_batch(features, targets)
    assert_tree(torch, owned.snapshot_state(), before)


@pytest.mark.parametrize("method", METHODS)
def test_should_match_actual_native_head_updates_and_leave_inputs_and_global_rng_unchanged(
    torch, method
):
    selected = policy(method)
    rng_before = torch.get_rng_state().clone()
    numpy_before = pickle.dumps(np.random.get_state())
    owned = TorchFeatureLearner(selected)
    head, optimizer = reference_head(torch, selected)
    features, targets = batch()
    before = pickle.dumps((features, targets))
    inputs = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(targets[:, 0], dtype=torch.long)
    for index in range(1, 3):
        if optimizer is None:
            expected = head.train_step(inputs.clone(), labels.clone(), 0.03, 2, 0.04)
        else:
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(head.forward_logits(inputs), labels)
            expected = float(loss.detach().item())
            loss.backward()
            optimizer.step()
        actual = owned.train_batch(features, targets)
        assert actual.value == expected
        assert actual.definition == (port.TORCH_CE_ID if method == "bp" else TORCH_PC_ENERGY_ID)
        state = owned.snapshot_state()
        assert type(state) is TorchFeatureSnapshot and state.format_version == 1
        assert state.completed_updates == owned.completed_updates == index
        assert state.training_examples == owned.training_examples == 2 * index
        assert_tree(torch, state.head_state, native_state(head, method))
        if optimizer is not None:
            assert_tree(torch, state.optimizer_state, optimizer.state_dict())
            assert_tree(torch, state.gradients, {name: getattr(head, name).grad for name in NAMES})
        else:
            assert state.optimizer_state is state.gradients is None
        logits = (
            torch.tanh(inputs @ head.weight_feature_hidden + head.bias_hidden)
            @ head.weight_hidden_output
            + head.bias_output
        )
        expected_prediction = (
            torch.softmax(logits, dim=1)[:, [1]].detach().numpy().astype(np.float64)
        )
        prediction = owned.predict(features)
        assert prediction.dtype == np.float64 and prediction.shape == (2, 1)
        np.testing.assert_array_equal(prediction, expected_prediction)
        prediction.fill(999)
        assert_tree(torch, owned.snapshot_state(), state)
    assert pickle.dumps((features, targets)) == before
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert pickle.dumps(np.random.get_state()) == numpy_before


@pytest.mark.parametrize("method", METHODS)
def test_should_own_complete_fork_and_restore_continuation_including_sgd_gradients(torch, method):
    source = TorchFeatureLearner(policy(method))
    features, targets = batch()
    source.train_batch(features, targets)
    source.train_batch(features, targets)
    saved = source.snapshot_state()
    child = source.fork()
    restored = TorchFeatureLearner(policy(method))
    restored.restore_state(saved)
    for copy in (child, restored):
        assert_tree(torch, copy.snapshot_state(), saved)
        for left, right in zip(
            tensors(torch, saved), tensors(torch, copy.snapshot_state()), strict=True
        ):
            assert left.data_ptr() != right.data_ptr()
    diagnostics = [item.train_batch(features, targets) for item in (source, child, restored)]
    assert diagnostics[0] == diagnostics[1] == diagnostics[2]
    expected = source.snapshot_state()
    assert expected.completed_updates == 3 and expected.training_examples == 6
    for item in (child, restored):
        assert_tree(torch, item.snapshot_state(), expected)
    for tensor in tensors(torch, saved):
        tensor.zero_()
    object.__setattr__(saved.policy, "seed", 19)
    for item in (source, child, restored):
        assert_tree(torch, item.snapshot_state(), expected)
        prediction_before = item.snapshot_state()
        assert item.predict(features).shape == (2, 1)
        assert_tree(torch, item.snapshot_state(), prediction_before)


@pytest.mark.parametrize(
    "method, fault",
    [
        ("pc", "version"),
        ("pc", "policy"),
        ("pc", "traffic"),
        ("pc", "missing"),
        ("gated_cpc", "generator"),
        ("gated_cpc", "clock"),
        ("gated_cpc", "age"),
        ("bp", "optimizer_rate"),
        ("bp", "momentum"),
        ("bp", "gradient"),
    ],
)
def test_should_refuse_corrupt_complete_state_before_publishing_any_field(torch, method, fault):
    owned = TorchFeatureLearner(policy(method))
    owned.train_batch(*batch())
    before = owned.snapshot_state()
    incoming = deepcopy(before)
    if fault == "version":
        object.__setattr__(incoming, "format_version", True)
    elif fault == "policy":
        object.__setattr__(incoming, "policy", replace(incoming.policy, seed=18))
    elif fault == "traffic":
        incoming.head_state["traffic_steps"] = True
    elif fault == "missing":
        del incoming.head_state["bias_output"]
    elif fault == "generator":
        incoming.head_state["split_generator_state"] = torch.zeros(1, dtype=torch.uint8)
    elif fault == "clock":
        incoming.head_state["wake_examples"] += 1
    elif fault == "age":
        incoming.head_state["neuron_age"][0] += 0.5
    elif fault == "optimizer_rate":
        assert incoming.optimizer_state is not None
        incoming.optimizer_state["param_groups"][0]["lr"] = 0.1
    elif fault == "momentum":
        assert incoming.optimizer_state is not None
        first = next(iter(incoming.optimizer_state["state"].values()))
        first["momentum_buffer"] = first["momentum_buffer"].double()
    else:
        assert incoming.gradients is not None
        incoming.gradients[NAMES[0]] = torch.zeros(1)
    rng_before = torch.get_rng_state().clone()
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        owned.restore_state(incoming)
    assert_tree(torch, owned.snapshot_state(), before)
    assert torch.equal(torch.get_rng_state(), rng_before)


@pytest.mark.parametrize("method", ["pc", "gated_cpc", "bp"])
def test_should_discard_staged_native_work_when_the_native_call_raises_after_mutation(
    monkeypatch, torch, method
):
    owned = TorchFeatureLearner(policy(method))
    owned.train_batch(*batch())
    before = owned.snapshot_state()
    problem = RuntimeError("injected fault after native work")
    native_class = (
        torch.optim.SGD
        if method == "bp"
        else PredictiveCodingHead
        if method == "pc"
        else CircadianPredictiveCodingHead
    )
    name = "step" if method == "bp" else "train_step"
    original = getattr(native_class, name)

    def broken(*args, **kwargs):
        original(*args, **kwargs)
        raise problem

    monkeypatch.setattr(native_class, name, broken)
    with pytest.raises(RuntimeError) as caught:
        owned.train_batch(*batch())
    assert caught.value is problem
    assert_tree(torch, owned.snapshot_state(), before)


def test_should_continue_a_copied_buffer_and_complete_cpc_state_without_claiming_runtime_resume(
    torch,
):
    owned = TorchFeatureLearner(policy("gated_cpc"))
    buffer = SharedReplayBuffer(
        2,
        ReplayRetentionBudget(4, 96),
        ReplayRetentionPolicy("seeded_reservoir", 53),
        allocation=ReplayAllocation(ReplayRetentionBudget(2, 48), ReplayRetentionBudget(2, 48)),
    )
    features, targets = batch()
    buffer.observe_train_batch(features, targets)
    buffer.observe_train_batch(features + 0.1, 1 - targets)
    first = buffer.select_partitioned_rotating(1)
    owned.train_batch(*first.training_batches()[0])
    # Move the existing local split generator without structural changes. Its
    # exact continuation must survive even though this port never calls sleep.
    torch.randn((3,), generator=owned._head._split_generator)
    saved_model = owned.snapshot_state()
    saved_buffer = deepcopy(buffer)
    saved_data = pickle.dumps(saved_buffer)
    expected_ids, expected_diagnostics = [], []
    for _ in range(2):
        selected = buffer.select_partitioned_rotating(1)
        expected_ids.append(selected.sample_ids)
        expected_diagnostics.append(owned.train_batch(*selected.training_batches()[0]))
    expected_state = owned.snapshot_state()
    resumed = TorchFeatureLearner(policy("gated_cpc"))
    resumed.restore_state(saved_model)
    resumed_buffer = deepcopy(saved_buffer)
    for ids, diagnostic in zip(expected_ids, expected_diagnostics, strict=True):
        selected = resumed_buffer.select_partitioned_rotating(1)
        assert selected.sample_ids == ids
        assert resumed.train_batch(*selected.training_batches()[0]) == diagnostic
    assert_tree(torch, resumed.snapshot_state(), expected_state)
    assert resumed_buffer.rotation_cursor == buffer.rotation_cursor
    assert resumed_buffer.retention == buffer.retention
    assert torch.equal(
        torch.randn((4,), generator=owned._head._split_generator),
        torch.randn((4,), generator=resumed._head._split_generator),
    )
    assert_tree(torch, resumed.snapshot_state(), owned.snapshot_state())
    np.testing.assert_array_equal(resumed.predict(features), owned.predict(features))
    assert pickle.dumps(saved_buffer) == saved_data
    assert owned.completed_updates == resumed.completed_updates == 3
