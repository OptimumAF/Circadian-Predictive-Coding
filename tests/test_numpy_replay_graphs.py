"""Small borrowed graph fixtures; no model construction, training, or copies.

One module fixture allocates 14 tiny payload buffers (under 1 KiB combined).
Every parametrized case reuses those buffers; ndarray subclass/noncontiguous
fixtures are views and add no payload allocation.
"""

from collections import deque
import json
from math import prod

import numpy as np
import pytest

from src.adapters import numpy_replay_graphs as graphs
from src.core.circadian_predictive_coding import CircadianNetworkSnapshot, ReplaySnapshot
from src.core.replay_graph_origin import ReplayGraphPorts


class IntSubclass(int):
    pass


class DictSubclass(dict):
    pass


class DequeSubclass(deque):
    pass


class NetworkSnapshotSubclass(CircadianNetworkSnapshot):
    pass


class ReplaySnapshotSubclass(ReplaySnapshot):
    pass


class ArraySubclass(np.ndarray):
    pass


@pytest.fixture(scope="module")
def arrays(tmp_path_factory):
    allocation_count = 0
    allocation_bytes = 0

    def allocate(shape, dtype):
        nonlocal allocation_count, allocation_bytes
        expected_bytes = prod(shape) * np.dtype(dtype).itemsize
        assert allocation_count + 1 <= 160
        assert allocation_bytes + expected_bytes <= 64 * 1024
        array = np.empty(shape, dtype=dtype)
        allocation_count += 1
        allocation_bytes += array.nbytes
        assert array.nbytes == expected_bytes
        return array

    result = {
        "features": allocate((2, 2), np.float64),
        "targets": allocate((2, 1), np.float64),
        "integer": allocate((2, 2), np.int16),
        "empty": allocate((0, 2), np.float64),
        "flat": allocate((2,), np.float64),
        "three_dimensions": allocate((1, 2, 1), np.float64),
        "mismatched": allocate((3, 1), np.float64),
        "wide_targets": allocate((2, 2), np.float64),
        "complex": allocate((2, 2), np.complex128),
        "object": allocate((2, 2), object),
        "bool": allocate((2, 2), bool),
        "string": allocate((2, 2), "U1"),
        "noncontiguous": allocate((2, 4), np.float64)[:, ::2],
        "subclass": allocate((2, 2), np.float64).view(ArraySubclass),
    }
    assert allocation_count == 14
    assert allocation_bytes == 324 + 4 * np.dtype(object).itemsize
    resource_path = tmp_path_factory.mktemp("numpy-replay-graphs") / "resources.json"
    resource_path.write_text(
        json.dumps(
            {
                "dispatch": "MW-20261008-02",
                "observed_payload_array_allocations": allocation_count,
                "observed_payload_array_bytes": allocation_bytes,
                "max_payload_array_allocations": 160,
                "max_payload_array_bytes": 64 * 1024,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def make_row(features, targets, row_type: type[ReplaySnapshot] = ReplaySnapshot) -> ReplaySnapshot:
    # Why this: inventory tests need exact native rows without native constructors.
    row = object.__new__(row_type)
    object.__setattr__(row, "input_batch", features)
    object.__setattr__(row, "target_batch", targets)
    object.__setattr__(row, "priority", 0.5)
    object.__setattr__(row, "positive_fraction", 0.5)
    return row


def make_snapshot(state, snapshot_type=CircadianNetworkSnapshot):
    snapshot = object.__new__(snapshot_type)
    object.__setattr__(snapshot, "state", state)
    return snapshot


def make_root(rows, root_kind):
    state = {"_replay_memory": deque(rows)}
    return state if root_kind == "state" else make_snapshot(state)


def assert_inventory_refuses(root, match):
    with pytest.raises(ValueError, match=match):
        graphs.replay_graph_rows(root, 3)
    with pytest.raises(ValueError, match=match):
        graphs.replay_graph_payload_bytes(root, 3, 1000)


@pytest.mark.parametrize("root_kind", ["state", "snapshot"])
def test_should_return_identical_rows_and_arrays_for_supported_roots(arrays, root_kind):
    first = make_row(arrays["features"], arrays["targets"])
    second = make_row(arrays["integer"], arrays["targets"])
    root = make_root([first, second], root_kind)
    ports = ReplayGraphPorts(graphs.replay_graph_rows, graphs.replay_graph_payload_bytes)

    rows = ports.rows(root, 2)

    assert type(rows) is tuple
    assert rows[0] is first and rows[1] is second
    assert first.input_batch is arrays["features"]
    assert first.target_batch is arrays["targets"]
    assert ports.payload_bytes(root, 2, 72) == 72


@pytest.mark.parametrize("root_kind", ["state", "snapshot"])
def test_should_report_zero_for_empty_inventory(root_kind):
    root = make_root([], root_kind)

    assert graphs.replay_graph_rows(root, 1) == ()
    assert graphs.replay_graph_payload_bytes(root, 1, 1) == 0


@pytest.mark.parametrize(
    "invalid", [False, True, 0, -1, 1.0, "1", None, 2**63, IntSubclass(1), np.int64(1)]
)
def test_should_refuse_nonexact_or_out_of_range_bounds(invalid):
    root = make_root([], "state")

    with pytest.raises(ValueError, match="max_rows"):
        graphs.replay_graph_rows(root, invalid)
    with pytest.raises(ValueError, match="max_rows"):
        graphs.replay_graph_payload_bytes(root, invalid, 1)
    with pytest.raises(ValueError, match="max_bytes"):
        graphs.replay_graph_payload_bytes(root, 1, invalid)


def test_should_accept_largest_permitted_bounds():
    root = make_root([], "state")

    assert graphs.replay_graph_rows(root, 2**63 - 1) == ()
    assert graphs.replay_graph_payload_bytes(root, 2**63 - 1, 2**63 - 1) == 0


@pytest.mark.parametrize(
    "root", [None, [], object(), DictSubclass(), make_snapshot({}, NetworkSnapshotSubclass)]
)
def test_should_refuse_unsupported_root_types(root):
    assert_inventory_refuses(root, "root must be")


@pytest.mark.parametrize("state", [None, [], DictSubclass()])
def test_should_refuse_snapshot_state_that_is_not_exact_dict(state):
    assert_inventory_refuses(make_snapshot(state), "state must be an exact dict")


def test_should_refuse_snapshot_without_state():
    assert_inventory_refuses(object.__new__(CircadianNetworkSnapshot), "must contain state")


@pytest.mark.parametrize("memory", [None, [], (), DequeSubclass()])
def test_should_refuse_missing_or_nonexact_replay_deque(memory):
    assert_inventory_refuses({"_replay_memory": memory}, "exact native replay deque")
    assert_inventory_refuses({}, "exact native replay deque")


@pytest.mark.parametrize("row", [object(), None, object.__new__(ReplaySnapshotSubclass)])
def test_should_refuse_nonexact_replay_rows(row):
    assert_inventory_refuses(make_root([row], "state"), "actual native ReplaySnapshot")


def test_should_refuse_row_without_payload_fields():
    assert_inventory_refuses(
        make_root([object.__new__(ReplaySnapshot)], "state"), "must contain replay payload"
    )


@pytest.mark.parametrize(
    "key", ["flat", "three_dimensions", "complex", "object", "bool", "string", "subclass"]
)
@pytest.mark.parametrize("field", ["features", "targets"])
def test_should_refuse_unsupported_arrays_in_either_field(arrays, key, field):
    features = arrays[key] if field == "features" else arrays["features"]
    targets = arrays[key] if field == "targets" else arrays["targets"]

    assert_inventory_refuses(
        make_root([make_row(features, targets)], "state"), "exact real numeric 2D arrays"
    )


@pytest.mark.parametrize("field", ["features", "targets"])
def test_should_refuse_nonarray_payloads(arrays, field):
    features = [] if field == "features" else arrays["features"]
    targets = [] if field == "targets" else arrays["targets"]

    assert_inventory_refuses(
        make_root([make_row(features, targets)], "state"), "exact real numeric 2D arrays"
    )


@pytest.mark.parametrize(
    "feature_key,target_key",
    [("empty", "targets"), ("features", "mismatched"), ("features", "wide_targets")],
)
def test_should_refuse_empty_or_misaligned_binary_batches(arrays, feature_key, target_key):
    row = make_row(arrays[feature_key], arrays[target_key])

    assert_inventory_refuses(make_root([row], "state"), "nonempty aligned binary batch")


def test_should_refuse_oversized_deque_before_reading_rows(monkeypatch):
    calls = []

    def fail_if_read(row):
        calls.append(row)
        raise AssertionError("oversized deque rows were inspected")

    monkeypatch.setattr(graphs, "replay_payload_references", fail_if_read)
    root = make_root([object(), object()], "state")

    with pytest.raises(ValueError, match="row bound"):
        graphs.replay_graph_rows(root, 1)
    with pytest.raises(ValueError, match="row bound"):
        graphs.replay_graph_payload_bytes(root, 1, 1)
    assert calls == []


def test_should_charge_aliased_payloads_per_row_and_accept_exact_byte_cap(arrays):
    row = make_row(arrays["features"], arrays["targets"])
    root = make_root([row, row], "state")

    rows = graphs.replay_graph_rows(root, 2)

    assert rows[0] is row and rows[1] is row
    assert graphs.replay_graph_payload_bytes(root, 2, 96) == 96
    with pytest.raises(ValueError, match="payload byte bound"):
        graphs.replay_graph_payload_bytes(root, 2, 95)


@pytest.mark.parametrize("max_bytes", [31, 47])
def test_should_refuse_byte_excess_before_reading_later_rows(arrays, monkeypatch, max_bytes):
    row = make_row(arrays["features"], arrays["targets"])
    calls = []
    original = graphs.replay_payload_references

    def record_read(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(graphs, "replay_payload_references", record_read)
    root = make_root([row, object()], "state")

    with pytest.raises(ValueError, match="payload byte bound"):
        graphs.replay_graph_payload_bytes(root, 2, max_bytes)
    assert len(calls) == 1 and calls[0] is row


def test_should_count_borrowed_noncontiguous_arrays_by_payload_bytes(arrays):
    row = make_row(arrays["noncontiguous"], arrays["targets"])
    root = make_root([row], "snapshot")

    assert graphs.replay_graph_rows(root, 1)[0] is row
    assert row.input_batch is arrays["noncontiguous"]
    assert graphs.replay_graph_payload_bytes(root, 1, 48) == 48


def test_should_ignore_unrelated_native_state_fields(arrays):
    row = make_row(arrays["features"], arrays["targets"])
    root = {"_replay_memory": deque([row]), "weights": object(), "model": object()}

    assert graphs.replay_graph_payload_bytes(root, 1, 48) == 48


def test_should_charge_array_aliases_within_one_row(arrays):
    row = make_row(arrays["targets"], arrays["targets"])
    root = make_root([row], "state")

    assert graphs.replay_graph_payload_bytes(root, 1, 32) == 32
    with pytest.raises(ValueError, match="payload byte bound"):
        graphs.replay_graph_payload_bytes(root, 1, 31)


def test_should_leave_scalar_metadata_validation_to_caller(arrays):
    row = make_row(arrays["features"], arrays["targets"])
    object.__setattr__(row, "priority", object())
    object.__setattr__(row, "positive_fraction", object())
    root = make_root([row], "state")

    assert graphs.replay_graph_rows(root, 1)[0] is row
    assert graphs.replay_graph_payload_bytes(root, 1, 48) == 48
