"""Pure bounded integrity-stamp contracts, with no checkpoint/native authority.

Why this: explicit allocation counters and digest witnesses make payload limits
observable without constructing a model, learner, runtime, or checkpoint owner.
"""

from collections import deque
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, asdict, dataclass, replace
import hashlib
import json

import numpy as np
import pytest

from src.core import checkpoint_content as content
from src.core.checkpoint_content import CheckpointContentLimits, checkpoint_content_stamp
from src.core.learner_ports import TrainingDiagnostic


LIMITS = CheckpointContentLimits(128, 8192, 4096, 16)


@dataclass
class ResourceBudget:
    arrays: int = 0
    array_bytes: int = 0
    views: int = 0
    graphs: int = 0
    generators: int = 0
    draws: int = 0

    def array(self, values, dtype=float):
        expected_bytes = len(values) * np.dtype(dtype).itemsize
        assert self.arrays + self.views + 1 <= 24
        assert self.array_bytes + expected_bytes <= 4096
        value = np.array(values, dtype=dtype)
        assert value.nbytes == expected_bytes
        self.arrays += 1
        self.array_bytes += value.nbytes
        self.require_bounds()
        return value

    def view(self, create):
        self.views += 1
        self.require_bounds()
        return create()

    def graph(self, value):
        self.graphs += 1
        self.require_bounds()
        return value

    def generator(self):
        self.generators += 1
        self.require_bounds()
        return np.random.default_rng(7)

    def require_bounds(self):
        assert self.arrays + self.views <= 24
        assert self.array_bytes <= 4096
        assert self.graphs <= 128
        assert self.generators <= 2
        assert self.draws == 0


@pytest.fixture(scope="module")
def budget(tmp_path_factory):
    resources = ResourceBudget()
    yield resources
    resources.require_bounds()
    artifact = tmp_path_factory.getbasetemp() / "checkpoint-content-resources.json"
    artifact.write_text(
        json.dumps(asdict(resources), indent=2, sort_keys=True) + "\n", encoding="utf8"
    )


class DigestWitness:
    def __init__(self, initial):
        self.digest = hashlib.sha256(initial)
        self.payload_updates = 0
        self.payload_views: list[object] = []

    def update(self, value):
        if type(value) is memoryview:
            self.payload_updates += 1
            self.payload_views.append(value.obj)
        self.digest.update(value)

    def hexdigest(self):
        return self.digest.hexdigest()


@pytest.mark.parametrize("before,after", [(None, False), (False, 0), (0, 0.0), (0.0, "0")])
def test_should_distinguish_exact_scalar_types_and_values(budget, before, after):
    graph = budget.graph({"value": before})
    original = checkpoint_content_stamp(graph, LIMITS)
    graph["value"] = after
    assert checkpoint_content_stamp(graph, LIMITS) != original


@pytest.mark.parametrize("kind", ["dict", "list", "tuple", "deque", "set", "frozenset"])
def test_should_stamp_supported_container_mutation(budget, kind):
    constructors = {
        "dict": lambda: {"value": 1},
        "list": lambda: [1],
        "tuple": lambda: (1,),
        "deque": lambda: deque([1], maxlen=2),
        "set": lambda: {"one"},
        "frozenset": lambda: frozenset({"one"}),
    }
    graph = budget.graph({"nested": constructors[kind]()})
    original = checkpoint_content_stamp(graph, LIMITS)
    if kind == "dict":
        graph["nested"]["value"] = 2
    elif kind in ("list", "deque"):
        graph["nested"][0] = 2
    elif kind == "set":
        graph["nested"].add("two")
    else:
        graph["nested"] = (2,) if kind == "tuple" else frozenset({"two"})
    assert checkpoint_content_stamp(graph, LIMITS) != original


def test_should_allow_another_root_dictionary_with_the_same_nested_values(budget):
    nested = [1, {"value": "same"}]
    original = budget.graph({"nested": nested})
    replacement = budget.graph(dict(original))
    assert replacement is not original
    assert replacement["nested"] is original["nested"]
    assert checkpoint_content_stamp(replacement, LIMITS) == checkpoint_content_stamp(
        original, LIMITS
    )


def test_should_detect_equal_valued_nested_container_replacement(budget):
    graph = budget.graph({"nested": [1, "same"]})
    original = checkpoint_content_stamp(graph, LIMITS)
    old = graph["nested"]
    graph["nested"] = list(old)
    assert graph["nested"] == old and graph["nested"] is not old
    assert checkpoint_content_stamp(graph, LIMITS) != original


def test_should_detect_exact_record_scalar_mutation(budget):
    record = TrainingDiagnostic("loss", 1.0)
    graph = budget.graph({"diagnostic": record})
    original = checkpoint_content_stamp(graph, LIMITS)
    object.__setattr__(record, "value", 2.0)
    assert checkpoint_content_stamp(graph, LIMITS) != original


@pytest.mark.parametrize("schema_change", ["extra", "missing"])
def test_should_reject_changed_exact_record_schema(budget, schema_change):
    record = budget.graph(TrainingDiagnostic("loss", 1.0))
    fields = object.__getattribute__(record, "__dict__")
    if schema_change == "extra":
        fields["unexpected"] = 1
    else:
        del fields["value"]
    with pytest.raises(ValueError, match="record fields changed"):
        checkpoint_content_stamp(record, LIMITS)


@pytest.mark.parametrize("dtype", [np.bool_, np.int16, np.uint16, np.float64])
def test_should_accept_exact_real_numeric_arrays_without_changing_them(budget, dtype):
    value = budget.array([0, 1], dtype=dtype)
    graph = budget.graph({"array": value})
    first = checkpoint_content_stamp(graph, LIMITS)
    assert checkpoint_content_stamp(graph, LIMITS) == first
    assert value.tolist() == [0, 1]


def test_should_detect_array_buffer_mutation(budget):
    value = budget.array([1, 2, 3])
    graph = budget.graph({"array": value})
    original = checkpoint_content_stamp(graph, LIMITS)
    value[1] = 9
    assert checkpoint_content_stamp(graph, LIMITS) != original


def test_should_detect_equal_valued_nested_array_replacement(budget):
    first = budget.array([1, 2])
    second = budget.array([1, 2])
    graph = budget.graph({"array": first})
    original = checkpoint_content_stamp(graph, LIMITS)
    graph["array"] = second
    assert first.tolist() == second.tolist()
    assert checkpoint_content_stamp(graph, LIMITS) != original


@pytest.mark.parametrize("schema_change", ["shape", "writeable"])
def test_should_detect_array_schema_mutation(budget, schema_change):
    value = budget.array([1, 2, 3, 4])
    graph = budget.graph({"array": value})
    original = checkpoint_content_stamp(graph, LIMITS)
    if schema_change == "shape":
        value.shape = (2, 2)
    else:
        value.flags.writeable = False
    assert checkpoint_content_stamp(graph, LIMITS) != original
    if schema_change == "shape":
        shaped_stamp = checkpoint_content_stamp(graph, LIMITS)
        old_strides = value.strides
        value.strides = tuple(reversed(old_strides))
        assert value.strides != old_strides
        assert value.flags.f_contiguous
        assert checkpoint_content_stamp(graph, LIMITS) != shaped_stamp


@pytest.mark.parametrize("layout", ["C", "F"])
def test_should_hash_contiguous_views_without_array_copy(budget, monkeypatch, layout):
    backing = budget.array([1, 2, 3, 4])
    value = budget.view(lambda: backing.reshape((2, 2), order="C"))
    if layout == "F":
        value = budget.view(lambda: value.T)
    assert np.shares_memory(value, backing)
    witness = DigestWitness(b"original_checkpoint_content_v1")
    monkeypatch.setattr(content, "sha256", lambda initial: witness)

    def reject_copy(*args, **kwargs):
        pytest.fail("content stamp invoked an array-copy constructor")

    monkeypatch.setattr(np, "array", reject_copy)
    monkeypatch.setattr(np, "copy", reject_copy)
    monkeypatch.setattr(np, "asarray", reject_copy)
    stamp = checkpoint_content_stamp(budget.graph({"array": value}), LIMITS)
    assert len(stamp) == 64
    assert witness.payload_updates == 1
    assert np.shares_memory(witness.payload_views[0], backing)


@pytest.mark.parametrize("bound", ["nodes", "depth", "metadata", "array"])
def test_should_enforce_caps_before_hashing_the_capped_payload(budget, monkeypatch, bound):
    value = budget.array([1, 2, 3, 4])
    graph = budget.graph({"array": [value]})
    limits = {
        "nodes": replace(LIMITS, max_nodes=2),
        "depth": replace(LIMITS, max_depth=1),
        "metadata": replace(LIMITS, max_metadata_bytes=1),
        "array": replace(LIMITS, max_array_bytes=value.nbytes - 1),
    }[bound]
    witness = DigestWitness(b"original_checkpoint_content_v1")
    monkeypatch.setattr(content, "sha256", lambda initial: witness)
    with pytest.raises(ValueError, match="bound exceeded"):
        checkpoint_content_stamp(graph, limits)
    assert witness.payload_updates == 0


def test_should_read_rng_without_draws_and_detect_direct_state_mutation(budget):
    generator = budget.generator()
    graph = budget.graph({"rng": generator})
    state = generator.bit_generator.state
    original = checkpoint_content_stamp(graph, LIMITS)
    assert checkpoint_content_stamp(graph, LIMITS) == original
    assert generator.bit_generator.state == state
    state["state"]["state"] = (state["state"]["state"] + 1) % (2**128)
    generator.bit_generator.state = state
    assert checkpoint_content_stamp(graph, LIMITS) != original
    assert budget.draws == 0
    # Reuse scalar state in the second and final generator, with zero draws.
    second = budget.generator()
    second.bit_generator.state = generator.bit_generator.state
    assert generator.bit_generator.state == second.bit_generator.state
    mutated_stamp = checkpoint_content_stamp(graph, LIMITS)
    graph["rng"] = second
    assert checkpoint_content_stamp(graph, LIMITS) != mutated_stamp


def _make_callback_type(calls):
    class CallbackMetaclass(type):
        def __eq__(cls, other):
            calls.append("metaclass equality")
            raise AssertionError("metaclass equality callback ran")

        def __hash__(cls):
            calls.append("metaclass hash")
            raise AssertionError("metaclass hash callback ran")

    class CallbackValue(metaclass=CallbackMetaclass):
        def __hash__(self):
            calls.append("key hash")
            return hash("value")

        def __eq__(self, other):
            calls.append("key equality")
            return False

    return CallbackValue


def test_should_reject_metaclass_callbacks_before_unsupported_value_traversal(budget):
    calls: list[str] = []
    value = _make_callback_type(calls)()
    graph = budget.graph(value)
    calls.clear()
    with pytest.raises(ValueError, match="unsupported exact checkpoint content type"):
        checkpoint_content_stamp(graph, LIMITS)
    assert calls == []


@pytest.mark.parametrize("container_kind", ["dictionary", "record"])
def test_should_reject_malicious_key_before_metaclass_or_key_callbacks(budget, container_kind):
    calls: list[str] = []
    key = _make_callback_type(calls)()
    if container_kind == "dictionary":
        graph = budget.graph({key: 1})
        error = "unsupported exact checkpoint dictionary key"
    else:
        graph = budget.graph(TrainingDiagnostic("loss", 1.0))
        data = object.__getattribute__(graph, "__dict__")
        del data["value"]
        data[key] = 1.0
        error = "record fields changed"
    calls.clear()
    with pytest.raises(ValueError, match=error):
        checkpoint_content_stamp(graph, LIMITS)
    assert calls == []


def test_should_reject_custom_sequence_and_dictionary_without_callbacks(budget):
    calls: list[str] = []

    class CallbackSequence(Sequence):
        def __len__(self):
            calls.append("len")
            raise AssertionError("custom sequence callback ran")

        def __getitem__(self, key):
            calls.append("getitem")
            raise AssertionError("custom sequence callback ran")

    class CallbackDictionary(dict):
        def items(self):
            calls.append("items")
            raise AssertionError("custom dictionary callback ran")

    for value in (CallbackSequence(), CallbackDictionary(value=1)):
        with pytest.raises(ValueError, match="unsupported exact checkpoint content type"):
            checkpoint_content_stamp(budget.graph(value), LIMITS)
    assert calls == []


def test_should_reject_custom_record_and_supported_record_subclass(budget):
    @dataclass
    class CustomRecord:
        value: int

    class DiagnosticSubclass(TrainingDiagnostic):
        pass

    for value in (CustomRecord(1), DiagnosticSubclass("loss", 1.0)):
        with pytest.raises(ValueError, match="unsupported exact checkpoint content type"):
            checkpoint_content_stamp(budget.graph(value), LIMITS)


@pytest.mark.parametrize("dtype", [object, complex])
def test_should_reject_object_and_complex_array_dtypes(budget, dtype):
    value = budget.array([1, 2], dtype=dtype)
    with pytest.raises(ValueError, match="exact contiguous numeric arrays"):
        checkpoint_content_stamp(budget.graph(value), LIMITS)


def test_should_reject_array_subclass_before_buffer_traversal(budget):
    class ArraySubclass(np.ndarray):
        pass

    backing = budget.array([1, 2])
    value = budget.view(lambda: backing.view(ArraySubclass))
    with pytest.raises(ValueError, match="unsupported exact checkpoint content type"):
        checkpoint_content_stamp(budget.graph(value), LIMITS)


def test_should_reject_noncontiguous_numeric_view(budget):
    backing = budget.array([1, 2, 3, 4])
    value = budget.view(lambda: backing[::2])
    assert not value.flags.c_contiguous and not value.flags.f_contiguous
    with pytest.raises(ValueError, match="exact contiguous numeric arrays"):
        checkpoint_content_stamp(budget.graph(value), LIMITS)


@pytest.mark.parametrize("invalid", [0, -1, True, 1.0, 2**63])
def test_should_require_positive_bounded_exact_integer_limits(invalid):
    with pytest.raises(ValueError, match="positive bounded integers"):
        CheckpointContentLimits(invalid, 32, 32, 4)


def test_should_require_exact_frozen_limits(budget):
    class LimitsSubclass(CheckpointContentLimits):
        pass

    with pytest.raises(ValueError, match="exact original limits"):
        checkpoint_content_stamp(budget.graph({"value": 1}), LimitsSubclass(16, 1024, 32, 4))
    with pytest.raises(FrozenInstanceError):
        setattr(LIMITS, "max_nodes", 1)


def test_should_bound_escaped_astral_string_before_hashing(budget, monkeypatch):
    # Nonprintable astral characters expand to ten-byte escapes under repr.
    graph = budget.graph("\U000fffff" * 5)
    witness = DigestWitness(b"original_checkpoint_content_v1")
    original_digest = witness.hexdigest()
    monkeypatch.setattr(content, "sha256", lambda initial: witness)
    with pytest.raises(ValueError, match="string bound exceeded"):
        checkpoint_content_stamp(graph, replace(LIMITS, max_metadata_bytes=56))
    assert witness.hexdigest() == original_digest
    assert witness.payload_updates == 0


@pytest.mark.parametrize("kind", [set, frozenset])
def test_should_bound_set_string_metadata_before_sorting(budget, monkeypatch, kind):
    graph = budget.graph(kind(("alpha", "bravo")))
    witness = DigestWitness(b"original_checkpoint_content_v1")
    monkeypatch.setattr(content, "sha256", lambda initial: witness)

    def reject_sort(*args, **kwargs):
        pytest.fail("capped set was sorted before aggregate metadata rejection")

    monkeypatch.setattr(content, "sorted", reject_sort, raising=False)
    with pytest.raises(ValueError, match="set string bound exceeded"):
        checkpoint_content_stamp(graph, replace(LIMITS, max_metadata_bytes=100))
    assert witness.payload_updates == 0


@pytest.mark.parametrize(
    "field_name", ["max_nodes", "max_metadata_bytes", "max_array_bytes", "max_depth"]
)
def test_should_revalidate_mutated_exact_frozen_limits_before_traversal(
    budget, monkeypatch, field_name
):
    local_limits = replace(LIMITS)
    object.__setattr__(local_limits, field_name, 0)
    calls: list[str] = []
    graph = budget.graph(_make_callback_type(calls)())
    calls.clear()

    def reject_digest(*args, **kwargs):
        pytest.fail("invalid mutated limits reached digest creation or traversal")

    monkeypatch.setattr(content, "sha256", reject_digest)
    with pytest.raises(ValueError, match="positive bounded integers"):
        checkpoint_content_stamp(graph, local_limits)
    assert calls == []
    assert LIMITS == CheckpointContentLimits(128, 8192, 4096, 16)
