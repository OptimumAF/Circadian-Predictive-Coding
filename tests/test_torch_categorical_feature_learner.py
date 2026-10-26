"""Bounded native categorical heads preserve their own objectives and full state."""

from copy import deepcopy
from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

import src.adapters.torch_feature_learner as port
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.core.resnet50_variants import (
    BackpropMLPHead,
    CircadianHeadConfig,
    CircadianPredictiveCodingHead,
    PredictiveCodingHead,
    TORCH_PC_ENERGY_ID,
)
from src.shared.torch_runtime import require_torch
from test_torch_feature_learner import METHODS, NAMES, assert_tree, forbidden, tensors


@pytest.fixture
def torch():
    return require_torch()


def policy(method="pc", **changes):
    values: dict[str, Any] = dict(
        method=method,
        feature_dim=2,
        hidden_dim=3,
        seed=17,
        learning_rate=0.03,
        inference_steps=2,
        inference_learning_rate=0.04,
        momentum=0.9,
        num_classes=3,
    )
    values.update(changes)
    return TorchFeaturePolicy(**values)


def batch():
    return np.array([[0.3, -0.2], [-0.5, 0.4], [0.1, 0.6]]), np.array([[0.0], [1.0], [2.0]])


def reference_head(torch, selected):
    args = (2, 3, selected.num_classes, torch.device("cpu"), 17)
    if selected.method == "bp":
        head = BackpropMLPHead(*args)
        return head, torch.optim.SGD(head.trainable_parameters(), lr=0.03, momentum=0.9)
    if selected.method == "pc":
        return PredictiveCodingHead(*args), None
    config = (
        CircadianHeadConfig.matched_pc_control()
        if selected.method == "neutral_cpc"
        else CircadianHeadConfig()
    )
    return CircadianPredictiveCodingHead(
        *args, config=config, min_hidden_dim=3, max_hidden_dim=3
    ), None


def native_state(head, method, classes):
    if method in ("neutral_cpc", "gated_cpc"):
        return head.snapshot_state()
    result = dict(format_version=1, feature_dim=2, hidden_dim=3, num_classes=classes, device="cpu")
    result.update({name: getattr(head, name).detach().clone() for name in NAMES})
    if method == "pc":
        result.update(traffic_sum=head._traffic_sum.clone(), traffic_steps=head._traffic_steps)
    return result


@pytest.mark.parametrize("classes", [True, 1, 17, 3.0, np.int64(3), "3"])
def test_should_refuse_non_native_or_out_of_bound_class_count_before_torch(monkeypatch, classes):
    monkeypatch.setattr(port, "require_torch", forbidden)
    with pytest.raises(ValueError):
        policy(num_classes=classes)


def test_should_revalidate_tampered_class_count_before_native_construction(monkeypatch):
    config = policy()
    object.__setattr__(config, "num_classes", True)
    monkeypatch.setattr(port, "require_torch", forbidden)
    with pytest.raises(ValueError):
        TorchFeatureLearner(config)


@pytest.mark.parametrize(
    "fault", ["fraction", "negative", "upper", "nan", "infinity", "dtype", "shape", "soft"]
)
def test_should_refuse_non_hard_or_wrong_range_labels_before_native_training(
    monkeypatch, torch, fault
):
    owned = TorchFeatureLearner(policy())
    features, targets = batch()
    if fault == "fraction":
        targets[0, 0] = 1.5
    elif fault == "negative":
        targets[0, 0] = -1
    elif fault == "upper":
        targets[0, 0] = 3
    elif fault == "nan":
        targets[0, 0] = np.nan
    elif fault == "infinity":
        targets[0, 0] = np.inf
    elif fault == "dtype":
        targets = targets.astype(np.float32)
    elif fault == "shape":
        targets = targets[:, 0]
    else:
        targets = np.eye(3, dtype=np.float64)
    before = owned.snapshot_state()
    monkeypatch.setattr(PredictiveCodingHead, "train_step", forbidden)
    monkeypatch.setattr(CircadianPredictiveCodingHead, "train_step", forbidden)
    monkeypatch.setattr(torch.optim.SGD, "step", forbidden)
    with pytest.raises(ValueError):
        owned.train_batch(features, targets)
    assert_tree(torch, owned.snapshot_state(), before)


@pytest.mark.parametrize("method", METHODS)
def test_should_match_own_native_three_class_objective_tensors_probabilities_and_rng(torch, method):
    config = policy(method)
    torch_before = torch.get_rng_state().clone()
    numpy_before = pickle.dumps(np.random.get_state())
    owned = TorchFeatureLearner(config)
    head, optimizer = reference_head(torch, config)
    features, targets = batch()
    inputs_before = pickle.dumps((features, targets))
    inputs = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(targets[:, 0], dtype=torch.long)
    for update in range(1, 3):
        if optimizer is None:
            expected = head.train_step(inputs.clone(), labels.clone(), 0.03, 2, 0.04)
        else:
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(head.forward_logits(inputs), labels)
            expected = float(loss.detach().item())
            loss.backward()
            optimizer.step()
        diagnostic = owned.train_batch(features, targets)
        assert diagnostic.value == expected
        assert diagnostic.definition == (port.TORCH_CE_ID if method == "bp" else TORCH_PC_ENERGY_ID)
        state = owned.snapshot_state()
        assert state.format_version == 1 and state.policy.num_classes == 3
        assert state.completed_updates == update and state.training_examples == 3 * update
        assert_tree(torch, state.head_state, native_state(head, method, 3))
        if optimizer is None:
            assert state.optimizer_state is state.gradients is None
        else:
            assert_tree(torch, state.optimizer_state, optimizer.state_dict())
            assert_tree(torch, state.gradients, {name: getattr(head, name).grad for name in NAMES})
        logits = (
            torch.tanh(inputs @ head.weight_feature_hidden + head.bias_hidden)
            @ head.weight_hidden_output
            + head.bias_output
        )
        expected_probability = torch.softmax(logits, dim=1).detach().numpy().astype(np.float64)
        actual = owned.predict(features)
        assert actual.shape == (3, 3) and actual.dtype == np.float64
        np.testing.assert_array_equal(actual, expected_probability)
        np.testing.assert_allclose(actual.sum(axis=1), 1, rtol=0, atol=1e-7)
        actual.fill(999)
        assert_tree(torch, owned.snapshot_state(), state)
    assert pickle.dumps((features, targets)) == inputs_before
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert pickle.dumps(np.random.get_state()) == numpy_before


def test_should_keep_three_class_pc_and_neutral_control_math_exact(torch):
    pc = TorchFeatureLearner(policy("pc"))
    neutral = TorchFeatureLearner(policy("neutral_cpc"))
    for _ in range(2):
        assert pc.train_batch(*batch()) == neutral.train_batch(*batch())
        np.testing.assert_array_equal(pc.predict(batch()[0]), neutral.predict(batch()[0]))
        for name in NAMES:
            assert_tree(
                torch,
                pc.snapshot_state().head_state[name],
                neutral.snapshot_state().head_state[name],
            )


@pytest.mark.parametrize("method", METHODS)
def test_should_continue_three_class_fork_and_restored_next_update_with_complete_owned_state(
    torch, method
):
    source = TorchFeatureLearner(policy(method))
    source.train_batch(*batch())
    source.train_batch(*batch())
    saved = source.snapshot_state()
    child = source.fork()
    restored = TorchFeatureLearner(policy(method))
    restored.restore_state(saved)
    for copy in (child, restored):
        assert_tree(torch, copy.snapshot_state(), saved)
        for original, copied in zip(
            tensors(torch, saved), tensors(torch, copy.snapshot_state()), strict=True
        ):
            assert original.data_ptr() != copied.data_ptr()
    diagnostics = [item.train_batch(*batch()) for item in (source, child, restored)]
    assert diagnostics[0] == diagnostics[1] == diagnostics[2]
    expected = source.snapshot_state()
    assert expected.completed_updates == 3 and expected.training_examples == 9
    for item in (child, restored):
        assert_tree(torch, item.snapshot_state(), expected)
    for tensor in tensors(torch, saved):
        tensor.zero_()
    object.__setattr__(saved.policy, "num_classes", 4)
    for item in (source, child, restored):
        assert item.predict(batch()[0]).shape == (3, 3)
        assert_tree(torch, item.snapshot_state(), expected)


@pytest.mark.parametrize(
    "method,fault",
    [
        ("pc", "metadata"),
        ("pc", "output_shape"),
        ("neutral_cpc", "output_shape"),
        ("gated_cpc", "policy"),
        ("bp", "gradient"),
        ("bp", "momentum"),
        ("bp", "output_shape"),
        ("bp", "metadata"),
    ],
)
def test_should_atomically_refuse_class_bound_state_geometry_and_optimizer_corruption(
    torch, method, fault
):
    owned = TorchFeatureLearner(policy(method))
    owned.train_batch(*batch())
    before = owned.snapshot_state()
    incoming = deepcopy(before)
    if fault == "metadata":
        incoming.head_state["num_classes"] = 2
    elif fault == "output_shape":
        incoming.head_state["weight_hidden_output"] = torch.zeros((3, 2))
    elif fault == "policy":
        object.__setattr__(incoming, "policy", replace(incoming.policy, num_classes=4))
    elif fault == "gradient":
        assert incoming.gradients is not None
        incoming.gradients["bias_output"] = torch.zeros((1, 2))
    else:
        assert incoming.optimizer_state is not None
        incoming.optimizer_state["state"][2]["momentum_buffer"] = torch.zeros((3, 2))
    torch_before = torch.get_rng_state().clone()
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        owned.restore_state(incoming)
    assert_tree(torch, owned.snapshot_state(), before)
    assert torch.equal(torch.get_rng_state(), torch_before)


@pytest.mark.parametrize("method", METHODS)
def test_should_rollback_three_class_native_fault_after_actual_staged_mutation(
    monkeypatch, torch, method
):
    owned = TorchFeatureLearner(policy(method))
    owned.train_batch(*batch())
    before = owned.snapshot_state()
    native = (
        torch.optim.SGD
        if method == "bp"
        else PredictiveCodingHead
        if method == "pc"
        else CircadianPredictiveCodingHead
    )
    name = "step" if method == "bp" else "train_step"
    original = getattr(native, name)
    primary = RuntimeError("categorical staged native failure")

    def broken(*args, **kwargs):
        original(*args, **kwargs)
        raise primary

    monkeypatch.setattr(native, name, broken)
    with pytest.raises(RuntimeError) as failure:
        owned.train_batch(*batch())
    assert failure.value is primary
    assert_tree(torch, owned.snapshot_state(), before)


@pytest.mark.parametrize("method", METHODS)
def test_should_keep_default_binary_positional_policy_and_explicit_two_exact(torch, method):
    default = TorchFeaturePolicy(method, 2, 3, 17, 0.03, 2, 0.04, 0.9)
    explicit = policy(method, num_classes=2)
    assert default == explicit
    left, right = TorchFeatureLearner(default), TorchFeatureLearner(explicit)
    features, _ = batch()
    labels = np.array([[0.0], [1.0], [0.0]])
    for _ in range(2):
        assert left.train_batch(features, labels) == right.train_batch(features, labels)
        assert_tree(torch, left.snapshot_state(), right.snapshot_state())
        left_prediction, right_prediction = left.predict(features), right.predict(features)
        assert left_prediction.shape == (3, 1)
        np.testing.assert_array_equal(left_prediction, right_prediction)


def test_should_admit_upper_bound_class_and_return_owned_complete_distribution(torch):
    owned = TorchFeatureLearner(policy(num_classes=16))
    features, _ = batch()
    owned.train_batch(features, np.array([[0.0], [8.0], [15.0]]))
    before = owned.snapshot_state()
    probabilities = owned.predict(features)
    assert probabilities.shape == (3, 16) and probabilities.dtype == np.float64
    assert (
        np.isfinite(probabilities).all()
        and (probabilities >= 0).all()
        and (probabilities <= 1).all()
    )
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=0, atol=2e-7)
    probabilities.fill(0)
    assert_tree(torch, owned.snapshot_state(), before)


@pytest.mark.parametrize("method", METHODS)
def test_should_refuse_otherwise_valid_snapshot_with_different_bound_class_count(torch, method):
    source = TorchFeatureLearner(policy(method, num_classes=2))
    destination = TorchFeatureLearner(policy(method))
    before = destination.snapshot_state()
    with pytest.raises(ValueError):
        destination.restore_state(source.snapshot_state())
    assert_tree(torch, destination.snapshot_state(), before)


def test_should_refuse_malformed_native_logits_without_publishing_state(monkeypatch, torch):
    owned = TorchFeatureLearner(policy())
    before = owned.snapshot_state()
    monkeypatch.setattr(
        PredictiveCodingHead,
        "predict_logits",
        lambda head, inputs: torch.zeros((len(inputs), 2), dtype=torch.float32),
    )
    with pytest.raises((ValueError, FloatingPointError)):
        owned.predict(batch()[0])
    assert_tree(torch, owned.snapshot_state(), before)
