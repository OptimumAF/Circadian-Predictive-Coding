"""Exact-value/owner refusal regressions; no fitting, forward or gradient work."""

from copy import deepcopy
from dataclasses import dataclass
import random
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import torch_decision as decision
from src.adapters import torch_categorical_readout_fit as readout
from src.adapters import torch_feature_learner as native
from src.adapters.torch_array_weighted import _rng


def forbidden(*args, **kwargs):
    raise AssertionError("unexpected native work")


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (1, 1, True),
        (1, 2, False),
        (True, 1, False),
        (1, 1.0, False),
        (True, True, True),
        (False, True, False),
        (None, None, True),
        ("state", "state", True),
        ("state", "stale", False),
        (b"abc", b"abd", False),
        (b"abc", b"abc", True),
        (-0.0, 0.0, True),
        (float("nan"), float("nan"), False),
        (float("inf"), float("inf"), True),
        (None, 0, False),
    ],
)
def test_exact_builtin_scalars_do_not_resolve_torch(monkeypatch, left, right, expected):
    monkeypatch.setattr(decision, "require_torch", forbidden)
    assert decision._same_values(left, right) is expected


def test_nested_state_compares_every_scalar_and_preserves_container_types(monkeypatch):
    @dataclass
    class State:
        values: dict[str, Any]

    original = State({"rng": (b"state", list(range(625)), None), "policy": {"frozen": True}})
    copied = deepcopy(original)
    monkeypatch.setattr(decision, "require_torch", forbidden)
    assert decision._same_values(original, copied)
    copied.values["rng"][1][-1] = 0
    assert not decision._same_values(original, copied)
    assert not decision._same_values((1,), [1])
    assert not decision._same_values({"a": 1}, {"b": 1})
    assert not decision._same_values([1], [1, 2])


def test_scalar_subclasses_keep_fallback_equality(monkeypatch):
    calls = []

    class Scalar(int):
        def __eq__(self, other):
            calls.append((int(self), int(other)))
            return int(self) + 1 == int(other)

    class Tensor:
        pass

    runtime = SimpleNamespace(Tensor=Tensor)
    resolutions = []

    def resolve():
        resolutions.append(1)
        return runtime

    monkeypatch.setattr(decision, "require_torch", resolve)
    assert decision._same_values(Scalar(1), Scalar(2))
    assert calls == [(1, 2)] and resolutions == [1]
    assert not decision._same_values(Scalar(1), 1)


def test_tensor_checks_keep_dtype_shape_device_value_and_nan_semantics():
    torch = pytest.importorskip("torch")
    value = torch.tensor([1.0, 2.0])
    assert decision._same_values(value, value.clone())
    assert not decision._same_values(value, value.double())
    assert not decision._same_values(value, value.reshape(1, 2))
    assert not decision._same_values(value, torch.tensor([1.0, 3.0]))
    assert not decision._same_values(value, torch.empty(2, device="meta"))
    nan = torch.tensor([float("nan")])
    assert not decision._same_values(nan, nan)


@pytest.fixture(scope="module")
def owners():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    result = [
        native.TorchFeatureLearner(native.TorchFeaturePolicy("bp", 9, 3, 17, num_classes=6))
        for _ in range(2)
    ]
    for learner in result:
        for name in native._PARAMETERS:
            value = getattr(learner._head, name)
            value.grad = torch.zeros_like(value)
            learner._optimizer.state[value]["momentum_buffer"] = torch.zeros_like(value)
        learner._payload = (learner._head, learner._optimizer, 12, 72)
    return torch, result


@pytest.mark.parametrize("field", ["parameter", "gradient", "momentum", "payload", "binding"])
def test_owner_guard_refuses_native_mutations_without_erasing_them(owners, monkeypatch, field):
    torch, learners = owners
    learner = learners[0]
    monkeypatch.setattr(native.TorchFeatureLearner, "train_batch", forbidden)
    monkeypatch.setattr(torch.optim.SGD, "step", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    original_payload = learner._payload
    parameter = learner._head.weight_feature_hidden
    original_parameter = parameter.detach().clone()
    original_gradient = parameter.grad.clone()
    momentum = learner._optimizer.state[parameter]["momentum_buffer"]
    original_momentum = momentum.clone()
    group = learner._optimizer.param_groups[0]
    parameters = group["params"]

    def mutate():
        if field == "payload":
            learner._payload = (*original_payload[:2], 13, 78)
        elif field == "binding":
            group["params"] = parameters.copy()
        else:
            value = (
                parameter
                if field == "parameter"
                else parameter.grad
                if field == "gradient"
                else momentum
            )
            with torch.no_grad():
                value[0, 0] += 1

    try:
        with pytest.raises(ValueError, match="owner|RNG"):
            with readout._unchanged_owner(learner) as (_, require):
                require()
                mutate()
                require()
        # Why: refusal must leave the foreign state visible, not restore it.
        if field == "payload":
            assert learner.completed_updates == 13
        elif field == "binding":
            assert group["params"] is not parameters
        elif field == "parameter":
            assert not torch.equal(parameter, original_parameter)
        elif field == "gradient":
            assert not torch.equal(parameter.grad, original_gradient)
        else:
            assert not torch.equal(momentum, original_momentum)
    finally:
        # Test-fixture cleanup only, after proving production preserved evidence.
        learner._payload = original_payload
        group["params"] = parameters
        with torch.no_grad():
            parameter.copy_(original_parameter)
            parameter.grad.copy_(original_gradient)
            momentum.copy_(original_momentum)


@pytest.mark.parametrize("domain", ["python", "numpy", "torch"])
def test_owner_guard_refuses_changed_rng_without_restoring_it(owners, domain):
    torch, learners = owners
    learner = learners[1]
    before = _rng(torch)
    python, numpy, native_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    assert isinstance(numpy, tuple)
    try:
        with pytest.raises(ValueError, match="owner|RNG"):
            with readout._unchanged_owner(learner) as (_, require):
                require()
                if domain == "python":
                    vector = list(python[1])
                    vector[1] ^= 1
                    random.setstate((python[0], tuple(vector), python[2]))
                elif domain == "numpy":
                    numpy_copy = deepcopy(numpy)
                    numpy_copy[1][1] ^= 1
                    np.random.set_state(numpy_copy)
                else:
                    changed = native_rng.clone()
                    changed[0] ^= 1
                    torch.set_rng_state(changed)
                require()
        assert not decision._same_values(before, _rng(torch))
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(native_rng)
