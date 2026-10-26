"""Exact recursive native-state comparison with a deterministic fake Torch port.

Synthetic snapshot payloads exercise comparison, not native snapshot admission or
learning. No actual tensor allocation, forward pass or SGD is needed here.
"""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import torch_feature_rehearsal_pilot as native
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy, TorchRecurrentSnapshot


class Tensor:
    def __init__(self, values, *, dtype="float32", device="cpu"):
        self.values = np.array(values, dtype=np.float64)
        self.dtype = dtype
        self.device = device


@pytest.fixture
def torch_port(monkeypatch):
    resolutions: list[bool] = []
    comparisons: list[tuple[Any, Any]] = []

    def equal(left, right):
        comparisons.append((left, right))
        return bool(np.array_equal(left.values, right.values))

    port = SimpleNamespace(Tensor=Tensor, equal=equal)

    def require():
        resolutions.append(True)
        return port

    monkeypatch.setattr(native, "require_torch", require)
    return SimpleNamespace(resolutions=resolutions, comparisons=comparisons)


def snapshot():
    return TorchRecurrentSnapshot(
        format_version=1,
        policy=TorchRecurrentPolicy(148, 8, 7, 17),
        parameters={"weight_input": Tensor([[0.1, -0.2]]), "bias_hidden": Tensor([[0.0]])},
        optimizer_state={
            "state": {0: {"momentum_buffer": Tensor([[0.03, -0.04]])}},
            "param_groups": [{"params": [0, 1], "lr": 0.03, "momentum": 0.9}],
        },
        gradients={"weight_input": Tensor([[0.01, 0.02]]), "bias_hidden": None},
        completed_updates=4,
        training_examples=36,
    )


def test_should_resolve_torch_once_for_full_nested_snapshot_and_every_outer_comparison(torch_port):
    original = snapshot()
    owned = deepcopy(original)
    assert native._same_state(original, owned) is True
    assert len(torch_port.resolutions) == 1
    assert len(torch_port.comparisons) == 4
    assert native._same_state(owned, original) is True
    assert len(torch_port.resolutions) == 2
    assert len(torch_port.comparisons) == 8


@pytest.mark.parametrize(
    "fault",
    [
        "parameter",
        "optimizer_tensor",
        "optimizer_lr",
        "optimizer_extra",
        "gradient",
        "missing_gradient",
        "counter",
        "counter_type",
        "examples",
        "policy",
        "extra_key",
        "missing_key",
        "list_to_tuple",
    ],
)
def test_should_refuse_any_changed_native_parameter_optimizer_gradient_or_metadata(
    torch_port, fault
):
    original = snapshot()
    altered = deepcopy(original)
    if fault == "parameter":
        altered.parameters["weight_input"].values[0, 0] += 1e-7
    elif fault == "optimizer_tensor":
        altered.optimizer_state["state"][0]["momentum_buffer"].values[0, 0] += 1e-7
    elif fault == "optimizer_lr":
        altered.optimizer_state["param_groups"][0]["lr"] = 0.04
    elif fault == "optimizer_extra":
        altered.optimizer_state["state"][0]["extra"] = 0
    elif fault == "gradient":
        altered.gradients["weight_input"].values[0, 0] += 1e-7
    elif fault == "missing_gradient":
        altered.gradients["weight_input"] = None
    elif fault == "counter":
        altered = replace(altered, completed_updates=5)
    elif fault == "counter_type":
        altered = replace(altered, completed_updates=4.0)
    elif fault == "examples":
        altered = replace(altered, training_examples=37)
    elif fault == "policy":
        altered = replace(altered, policy=replace(altered.policy, seed=18))
    elif fault == "extra_key":
        altered.parameters["extra"] = Tensor([[0.0]])
    elif fault == "missing_key":
        del altered.parameters["bias_hidden"]
    else:
        group = altered.optimizer_state["param_groups"][0]
        group["params"] = tuple(group["params"])
    assert native._same_state(original, altered) is False
    assert len(torch_port.resolutions) == 1


@pytest.mark.parametrize("fault", ["dtype", "device", "shape", "subclass"])
def test_should_refuse_tensor_representation_changes_without_numeric_coercion(torch_port, fault):
    original = Tensor([[0.25, 0.5]])
    altered: Any = deepcopy(original)
    if fault == "dtype":
        altered.dtype = "float64"
    elif fault == "device":
        altered.device = "cuda:0"
    elif fault == "shape":
        altered.values = altered.values.reshape(2, 1)
    else:

        class DerivedTensor(Tensor):
            pass

        altered = DerivedTensor([[0.25, 0.5]])
    assert native._same_state(original, altered) is False


def test_should_preserve_tensor_equality_rules_for_nan_and_signed_zero(torch_port):
    not_a_number = Tensor([[float("nan")]])
    assert native._same_state(not_a_number, not_a_number) is False
    assert native._same_state(Tensor([[-0.0]]), Tensor([[0.0]])) is True


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (True, 1, False),
        (1, 1.0, False),
        ([1, 2], (1, 2), False),
        ({"a": None}, {"a": None}, True),
        ({"a": 1}, {"a": 1, "b": 2}, False),
        (([1, None], {"x": (2.0,)}), ([1, None], {"x": (2.0,)}), True),
        (float("nan"), float("nan"), False),
    ],
)
def test_should_preserve_exact_recursive_python_types_and_container_metadata(
    torch_port, left, right, expected
):
    assert native._same_state(left, right) is expected
