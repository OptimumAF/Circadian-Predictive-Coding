"""Exact comparator laws; tiny tensors only, no learners or model execution.

The local reference preserves the prechange recursive dispatch, so a fast path
must retain fallback calls, short circuits and exceptions as well as booleans.
"""

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import numpy as np
import pytest

from src.adapters import torch_feature_rehearsal_pilot as pilot


def previous_same_state(left, right, torch):
    """Frozen semantics from r43-exact-comparison/previous_same_state.py."""
    if type(left) is not type(right):
        return False

    def equal_values(first, second):
        if type(first) is not type(second):
            return False
        if type(first) is torch.Tensor:
            return (
                first.dtype == second.dtype
                and first.device == second.device
                and torch.equal(first, second)
            )
        if is_dataclass(first):
            return all(
                equal_values(getattr(first, field.name), getattr(second, field.name))
                for field in fields(first)
            )
        if type(first) is dict:
            return first.keys() == second.keys() and all(
                equal_values(first[key], second[key]) for key in first
            )
        if type(first) in (list, tuple):
            return len(first) == len(second) and all(
                equal_values(a, b) for a, b in zip(first, second)
            )
        return bool(first == second)

    return equal_values(left, right)


class TorchStub:
    class Tensor:
        pass


@pytest.mark.parametrize("left,right", [(1, 1), ({"a": [1, True]}, {"a": [1, True]})])
def test_should_resolve_torch_once_even_for_pure_builtin_comparisons(monkeypatch, left, right):
    calls = []

    def require():
        calls.append("torch")
        return TorchStub

    monkeypatch.setattr(pilot, "require_torch", require)
    assert pilot._same_state(left, right) is True
    assert calls == ["torch"]


def test_should_propagate_torch_dependency_failure_before_comparing_equal_builtins(monkeypatch):
    error = RuntimeError("optional Torch unavailable")
    calls = []

    def require():
        calls.append("torch")
        raise error

    monkeypatch.setattr(pilot, "require_torch", require)
    with pytest.raises(RuntimeError, match="optional Torch unavailable") as caught:
        pilot._same_state(1, 1)
    assert caught.value is error and calls == ["torch"]
    calls.clear()
    assert pilot._same_state(False, 0) is False
    assert calls == []


@pytest.fixture(params=("optimized", "previous"))
def compare(request, monkeypatch):
    monkeypatch.setattr(pilot, "require_torch", lambda: TorchStub)
    if request.param == "previous":
        return lambda left, right: previous_same_state(left, right, TorchStub)
    return pilot._same_state


@dataclass
class NativeRecord:
    counters: Any
    parameters: Any


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (None, None, True),
        (False, False, True),
        (False, 0, False),
        (True, 1, False),
        (1, 1.0, False),
        (0.0, -0.0, True),
        (float("inf"), float("inf"), True),
        (float("inf"), -float("inf"), False),
        ("1", "1", True),
        ("1", b"1", False),
        (b"a", b"b", False),
        (complex(1, 2), complex(1, 2), True),
        (complex(1, 2), complex(1, 3), False),
        (np.int64(2), np.int64(2), True),
        (np.int64(2), 2, False),
        (np.int32(2), np.int64(2), False),
        (np.float64(2), 2.0, False),
        (np.float32(2), np.float64(2), False),
        (np.bool_(True), True, False),
        (np.bool_(False), np.bool_(False), True),
        (np.float64("nan"), np.float64("nan"), False),
        ([1, 2], (1, 2), False),
        ({"x": [1, (2,)]}, {"x": [1, (2,)]}, True),
        ({"x": [1, (2,)]}, {"x": [1, [2]]}, False),
        ({True: "value"}, {1: "value"}, True),
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}, True),
    ],
)
def test_should_preserve_exact_leaf_types_and_legacy_container_semantics(
    compare, left, right, expected
):
    assert compare(left, right) is expected


@pytest.mark.parametrize(
    "wrap",
    [
        lambda x: x,
        lambda x: [x],
        lambda x: (x,),
        lambda x: {"value": x},
        lambda x: NativeRecord(0, x),
    ],
)
def test_should_refuse_nan_even_when_comparing_the_same_object(compare, wrap):
    value = wrap(float("nan"))
    assert compare(value, value) is False


def test_should_recheck_mutation_and_replacement_after_a_previous_true_result(compare):
    left = NativeRecord({"updates": 192}, {"weights": [1.0, {"gradient": (2.0,)}]})
    right = deepcopy(left)
    assert compare(left, right) is True
    right.parameters["weights"][1]["gradient"] = (3.0,)
    assert compare(left, right) is False
    right.parameters = deepcopy(left.parameters)
    assert compare(left, right) is True
    right.counters["updates"] = 193
    assert compare(left, right) is False
    right.counters = {"updates": 192.0}
    assert compare(left, right) is False


def test_should_compare_dataclass_declared_fields_and_preserve_dynamic_attribute_behavior(compare):
    left = NativeRecord({"updates": 2}, ([1.0], {"slot": 0}))
    right = deepcopy(left)
    setattr(left, "extra", "legacy comparator ignores undeclared attributes")
    setattr(right, "extra", "different")
    assert compare(left, right) is True
    right.parameters[0][0] = float("nan")
    assert compare(left, right) is False


class TraceEquality:
    def __init__(self, label, trace, result=True, error=None):
        self.label, self.trace, self.result, self.error = label, trace, result, error

    def __eq__(self, other):
        self.trace.append(self.label)
        if self.error is not None:
            raise self.error
        return self.result


class TraceTruth:
    def __init__(self, trace, error=None):
        self.trace, self.error = trace, error

    def __bool__(self):
        self.trace.append("bool")
        if self.error is not None:
            raise self.error
        return True


def test_should_call_unknown_equality_and_result_truth_conversion_even_for_identity(compare):
    trace: list[Any] = []
    value = TraceEquality("eq", trace, TraceTruth(trace))
    assert compare(value, value) is True
    assert trace == ["eq", "bool"]


def test_should_preserve_python_notimplemented_fallback_calls_even_for_identity(compare):
    trace: list[Any] = []

    class UnsupportedEquality:
        def __init__(self, label):
            self.label = label

        def __eq__(self, other):
            trace.append(self.label)
            return NotImplemented

    left, right = UnsupportedEquality("left"), UnsupportedEquality("right")
    assert compare(left, right) is False
    assert trace == ["left", "right"]
    trace.clear()
    assert compare(left, left) is True
    assert trace == ["left", "left"]


def test_should_preserve_dictionary_key_equality_without_treating_keys_as_value_leaves(compare):
    key = float("nan")
    assert compare({key: 1}, {key: 1}) is True
    assert compare({key: 1}, {float("nan"): 1}) is False


@pytest.mark.parametrize("error_at", ["eq", "bool"])
def test_should_propagate_original_unknown_object_exceptions(compare, error_at):
    trace: list[Any] = []
    error = RuntimeError("fallback failure")
    value = TraceEquality(
        "eq",
        trace,
        TraceTruth(trace, error if error_at == "bool" else None),
        error if error_at == "eq" else None,
    )
    with pytest.raises(RuntimeError, match="fallback failure") as caught:
        compare(value, value)
    assert caught.value is error
    assert trace == (["eq"] if error_at == "eq" else ["eq", "bool"])


@pytest.mark.parametrize("shape", ["dict", "list", "tuple", "dataclass"])
def test_should_visit_fields_in_left_order_and_short_circuit_after_first_difference(compare, shape):
    trace: list[Any] = []
    first = TraceEquality("first", trace, False)
    forbidden = TraceEquality("unreached", trace, error=AssertionError("visited after false"))
    left: Any
    right: Any
    if shape == "dict":
        left, right = {"b": first, "a": forbidden}, {"a": forbidden, "b": first}
    elif shape == "list":
        left = right = [first, forbidden]
    elif shape == "tuple":
        left = right = (first, forbidden)
    else:
        left = right = NativeRecord(first, forbidden)
    assert compare(left, right) is False
    assert trace == ["first"]


@pytest.mark.parametrize("shape", ["keys", "length", "type"])
def test_should_short_circuit_schema_difference_before_unknown_leaf_equality(compare, shape):
    trace: list[Any] = []
    value = TraceEquality("forbidden", trace, error=AssertionError("unreachable"))
    left, right = {
        "keys": ({"a": value}, {"b": value}),
        "length": ([value], [value, value]),
        "type": ([value], (value,)),
    }[shape]
    assert compare(left, right) is False
    assert trace == []


class CustomList(list):
    def __eq__(self, other):
        return False


def test_should_keep_container_subclasses_on_their_own_equality_fallback(compare):
    value = CustomList([1, 2])
    assert compare(value, CustomList([1, 2])) is False
    assert compare(value, value) is False
    assert compare(value, [1, 2]) is False


def test_should_preserve_numpy_array_ambiguous_truth_exception(compare):
    array = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="truth value"):
        compare(array, array.copy())


@pytest.mark.parametrize("mode", ["list", "tuple", "neither", "exception"])
def test_should_preserve_custom_metaclass_membership_calls_and_exception_order(monkeypatch, mode):
    monkeypatch.setattr(pilot, "require_torch", lambda: TorchStub)

    def observe(comparator):
        trace: list[Any] = []

        class MembershipMeta(type):
            def __eq__(cls, other):
                trace.append(("type_eq", other.__name__))
                if mode == "exception":
                    raise RuntimeError("metaclass membership failure")
                return other is (list if mode == "list" else tuple) if mode != "neither" else False

        class Unknown(metaclass=MembershipMeta):
            def __len__(self):
                trace.append(("len",))
                return 1

            def __iter__(self):
                trace.append(("iter",))
                return iter((7,))

            def __eq__(self, other):
                trace.append(("object_eq",))
                return False

        outcome: tuple[Any, ...]
        try:
            outcome = ("return", comparator(Unknown(), Unknown()))
        except Exception as error:
            outcome = ("raise", type(error), str(error))
        return outcome, trace

    expected = observe(lambda left, right: previous_same_state(left, right, TorchStub))
    actual = observe(pilot._same_state)
    assert actual == expected
    assert expected[1][0] == ("type_eq", "list")
    if mode == "exception":
        assert expected[0] == ("raise", RuntimeError, "metaclass membership failure")
        assert expected[1] == [("type_eq", "list")]
    elif mode == "neither":
        assert expected[0] == ("return", False)
        assert expected[1][-1] == ("object_eq",)
    else:
        assert expected[0] == ("return", True)
        assert ("object_eq",) not in expected[1]


@pytest.fixture(params=("optimized", "previous"))
def tensor_compare(request, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(pilot, "require_torch", lambda: torch)
    comparator = (
        pilot._same_state
        if request.param == "optimized"
        else (lambda left, right: previous_same_state(left, right, torch))
    )
    return torch, comparator


def test_should_compare_tiny_tensor_clones_dtype_shape_and_contents_exactly(tensor_compare):
    torch, compare = tensor_compare
    value = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    clone = value.clone()
    assert compare(value, clone) is True
    assert compare(value, clone.to(dtype=torch.float32)) is False
    assert compare(value, clone.reshape(2, 1)) is False
    clone[0, 1] = 3.0
    assert compare(value, clone) is False
    clone[0, 1] = 2.0
    assert compare(value, clone) is True
    assert compare(torch.empty((0, 1)), torch.empty((0, 2))) is False


def test_should_refuse_tensor_nan_in_identity_clone_and_nested_dataclass(tensor_compare):
    torch, compare = tensor_compare
    value = torch.tensor([float("nan")])
    assert compare(value, value) is False
    assert compare(value, value.clone()) is False
    record = NativeRecord({"updates": 192}, {"gradient": value})
    assert compare(record, record) is False


def test_should_keep_tensor_device_mismatch_short_circuit_and_meta_exception(tensor_compare):
    torch, compare = tensor_compare
    cpu = torch.ones((1,), dtype=torch.float32)
    meta = torch.empty((1,), dtype=torch.float32, device="meta")
    assert compare(cpu, meta) is False
    assert compare(meta, cpu) is False
    assert compare(meta, torch.empty((1,), dtype=torch.float64, device="meta")) is False
    # Meta has no data to compare: retain the original torch.equal exception.
    with pytest.raises(NotImplementedError):
        compare(meta, meta)
