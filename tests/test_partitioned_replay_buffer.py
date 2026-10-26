"""Distinct recent and seeded bottom-k retention under one physical array cap."""

from dataclasses import FrozenInstanceError, asdict
from hashlib import sha256
import pickle
from typing import Any

import numpy as np
import pytest

from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.shared_replay_schedule import (
    PartitionRetention,
    ReplayAllocation,
    SharedReplayBuffer,
)


def arrays(numbers, label=0.0):
    values = np.asarray(numbers, dtype=np.float64)
    return np.column_stack((values, values / 10)), np.full((len(values), 1), label)


def content_id(inputs, targets):
    # Independent wire reference: two little-endian float64 shaped matrices.
    digest = sha256(b"numpy_labeled_replay_row_v1")
    for value in (inputs, targets):
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(np.asarray(value, dtype="<f8", order="C").tobytes(order="C"))
    return digest.hexdigest()


def ranked(sample_id, seed):
    return sha256(
        b"numpy_replay_bottom_k_v1" + seed.to_bytes(8, "little") + bytes.fromhex(sample_id)
    ).digest(), sample_id


def ids(inputs, targets):
    return tuple(
        content_id(inputs[index : index + 1], targets[index : index + 1])
        for index in range(len(inputs))
    )


def make_buffer(*, seed=53, recent=(2, 48), reservoir=(2, 48), global_budget=None):
    allocation = ReplayAllocation(ReplayRetentionBudget(*recent), ReplayRetentionBudget(*reservoir))
    budget = global_budget or ReplayRetentionBudget(
        recent[0] + reservoir[0], recent[1] + reservoir[1]
    )
    return SharedReplayBuffer(
        2, budget, ReplayRetentionPolicy("seeded_reservoir", seed), allocation=allocation
    )


def partition(buffer) -> PartitionRetention:
    result = buffer.partition_retention
    assert result is not None
    return result


def expected_membership(observed, recent_cap, reservoir_cap, seed):
    # A full prefix sort is deliberately independent of streaming eviction.
    first: dict[str, int] = {}
    last: dict[str, int] = {}
    for position, sample_id in enumerate(observed):
        first.setdefault(sample_id, position)
        last[sample_id] = position
    recent = tuple(sorted(last, key=last.__getitem__)[-recent_cap:])
    reservoir = tuple(sorted(first, key=lambda sample_id: ranked(sample_id, seed))[:reservoir_cap])
    older = tuple(sorted(set(reservoir) - set(recent), key=first.__getitem__))
    return recent, reservoir, older + recent


def expected_selection(recent, reservoir, maximum):
    pools = (list(reversed(recent)), list(reservoir))
    offsets = [0, 0]
    selected: list[str] = []
    turn = 0
    while len(selected) < maximum:
        chosen = None
        for pool_number in (turn, 1 - turn):
            pool = pools[pool_number]
            while offsets[pool_number] < len(pool):
                value = pool[offsets[pool_number]]
                offsets[pool_number] += 1
                if value not in selected:
                    chosen = value
                    break
            if chosen is not None:
                break
        if chosen is None:
            break
        selected.append(chosen)
        turn = 1 - turn
    return tuple(selected)


def stored_arrays(buffer):
    seen: set[int] = set()
    found: dict[int, np.ndarray[Any, Any]] = {}

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if type(value) is np.ndarray:
            found[id(value)] = value
        elif type(value) is dict:
            for child in value.values():
                visit(child)
        elif type(value) in (tuple, list):
            for child in value:
                visit(child)

    visit(vars(buffer))
    return tuple(found.values())


def stored_content_ids(buffer):
    seen: set[int] = set()
    found: set[str] = set()

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if (
            type(value) is str
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value)
        ):
            found.add(value)
        elif type(value) is dict:
            for key, child in value.items():
                visit(key)
                visit(child)
        elif type(value) in (tuple, list, set, frozenset):
            for child in value:
                visit(child)

    visit(vars(buffer))
    return found


def assert_snapshot(snapshot, expected_ids, row_bytes=24):
    assert snapshot.sample_ids == tuple(sorted(expected_ids))
    assert snapshot.example_count == len(expected_ids)
    assert snapshot.retained_bytes == row_bytes * len(expected_ids)


@pytest.mark.parametrize(
    "fault",
    [
        "recent_type",
        "reservoir_type",
        "recent_count",
        "recent_boolean",
        "recent_bytes",
        "reservoir_count",
        "reservoir_boolean",
        "reservoir_bytes",
    ],
)
def test_should_refuse_nonexact_or_tampered_nested_partition_budgets(fault):
    recent: Any = ReplayRetentionBudget(2, 48)
    reservoir: Any = ReplayRetentionBudget(2, 48)
    if fault == "recent_type":
        recent = object()
    elif fault == "reservoir_type":
        reservoir = object()
    else:
        target = recent if fault.startswith("recent") else reservoir
        field = "max_bytes" if fault.endswith("bytes") else "max_examples"
        object.__setattr__(target, field, True if fault.endswith("boolean") else 0)
    with pytest.raises((TypeError, ValueError)):
        ReplayAllocation(recent, reservoir)


@pytest.mark.parametrize(
    "fault",
    [
        "global_rows",
        "global_bytes",
        "recent_row_bytes",
        "reservoir_row_bytes",
        "fifo",
        "hash",
        "allocation_type",
        "tampered_global",
        "tampered_seed",
    ],
)
def test_should_refuse_invalid_global_partition_geometry_and_nonreservoir_policies(fault):
    budget = ReplayRetentionBudget(4, 96)
    recent, reservoir = ReplayRetentionBudget(2, 48), ReplayRetentionBudget(2, 48)
    policy = ReplayRetentionPolicy("seeded_reservoir", 53)
    if fault == "global_rows":
        budget = ReplayRetentionBudget(3, 96)
    elif fault == "global_bytes":
        budget = ReplayRetentionBudget(4, 95)
    elif fault == "recent_row_bytes":
        recent = ReplayRetentionBudget(2, 23)
    elif fault == "reservoir_row_bytes":
        reservoir = ReplayRetentionBudget(2, 23)
    elif fault == "fifo":
        policy = ReplayRetentionPolicy("recent_fifo")
    elif fault == "hash":
        policy = ReplayRetentionPolicy("content_hash")
    elif fault == "tampered_global":
        object.__setattr__(budget, "max_examples", True)
    elif fault == "tampered_seed":
        object.__setattr__(policy, "seed", True)
    allocation: Any = (
        object() if fault == "allocation_type" else ReplayAllocation(recent, reservoir)
    )
    with pytest.raises((TypeError, ValueError)):
        SharedReplayBuffer(2, budget, policy, allocation=allocation)


def test_should_own_global_policy_and_nested_allocation_without_mutable_caller_sharing():
    budget = ReplayRetentionBudget(4, 96)
    policy = ReplayRetentionPolicy("seeded_reservoir", 53)
    allocation = ReplayAllocation(ReplayRetentionBudget(2, 48), ReplayRetentionBudget(2, 48))
    buffer = SharedReplayBuffer(2, budget, policy, allocation=allocation)
    expected = asdict(allocation)
    object.__setattr__(budget, "max_examples", 1)
    object.__setattr__(policy, "seed", 17)
    object.__setattr__(allocation.recent, "max_examples", 1)
    detached = buffer.allocation
    assert detached is not None and asdict(detached) == expected
    assert detached is not allocation and detached.recent is not allocation.recent
    object.__setattr__(detached.reservoir, "max_bytes", 1)
    fresh = buffer.allocation
    assert fresh is not None and asdict(fresh) == expected
    inputs, targets = arrays(range(8))
    buffer.observe_train_batch(inputs, targets)
    recent, reservoir, union = expected_membership(ids(inputs, targets), 2, 2, 53)
    assert_snapshot(partition(buffer).recent, recent)
    assert_snapshot(partition(buffer).reservoir, reservoir)
    assert buffer.retained_order_ids == union


def test_should_return_frozen_detached_empty_partition_snapshots_and_empty_selection():
    buffer = make_buffer()
    report = partition(buffer)
    assert type(report) is PartitionRetention
    assert_snapshot(report.recent, ())
    assert_snapshot(report.reservoir, ())
    assert_snapshot(report.unique, ())
    assert report.overlap_examples == 0
    assert buffer.select_partitioned(4).sample_ids == ()
    assert buffer.select_partitioned(4).training_batches() == ()
    with pytest.raises(FrozenInstanceError):
        setattr(report, "overlap_examples", 1)
    object.__setattr__(report.unique, "example_count", 999)
    assert partition(buffer).unique.example_count == 0


def test_should_charge_overlap_to_both_quotas_but_store_one_detached_array_pair_per_union_id():
    buffer = make_buffer()
    inputs, targets = arrays(range(3))
    buffer.observe_train_batch(inputs, targets)
    recent, reservoir, union = expected_membership(ids(inputs, targets), 2, 2, 53)
    report = partition(buffer)
    assert report is not None
    assert_snapshot(report.recent, recent)
    assert_snapshot(report.reservoir, reservoir)
    assert_snapshot(report.unique, set(recent) | set(reservoir))
    assert buffer.retention == report.unique
    assert report.overlap_examples == len(set(recent) & set(reservoir)) >= 1
    assert (
        report.recent.retained_bytes
        + report.reservoir.retained_bytes
        - report.unique.retained_bytes
        == 24 * report.overlap_examples
    )
    assert buffer.retained_order_ids == union
    storage = stored_arrays(buffer)
    assert len(storage) == 2 * report.unique.example_count
    assert sum(value.nbytes for value in storage) == report.unique.retained_bytes
    assert all(value.dtype == np.float64 and value.flags.c_contiguous for value in storage)
    assert all(
        not np.shares_memory(value, original) for value in storage for original in (inputs, targets)
    )


@pytest.mark.parametrize("seed", [17, 53])
def test_should_refresh_duplicate_recent_membership_without_biasing_distinct_bottom_k(seed):
    buffer = make_buffer(seed=seed)
    inputs, targets = arrays(range(6))
    observed = ids(inputs, targets)
    buffer.observe_train_batch(inputs, targets)
    before = partition(buffer)
    buffer.observe_train_batch(inputs[:1], targets[:1])
    recent, reservoir, union = expected_membership((*observed, observed[0]), 2, 2, seed)
    assert_snapshot(partition(buffer).recent, recent)
    assert_snapshot(partition(buffer).reservoir, reservoir)
    assert partition(buffer).reservoir == before.reservoir
    assert buffer.retained_order_ids == union
    stable = (buffer.retention, buffer.partition_retention, buffer.retained_order_ids)
    buffer.observe_train_batch(inputs[:1], targets[:1])
    assert (buffer.retention, buffer.partition_retention, buffer.retained_order_ids) == stable


@pytest.mark.parametrize("seed", [17, 19])
def test_should_keep_bottom_k_membership_across_reordered_repeated_epochs(seed):
    inputs, targets = arrays(range(16))
    expected = set(sorted(ids(inputs, targets), key=lambda value: ranked(value, seed))[:2])
    for order in (
        tuple(range(16)),
        tuple(reversed(range(16))),
        tuple(range(0, 16, 2)) + tuple(range(1, 16, 2)),
    ):
        buffer = make_buffer(seed=seed)
        for _ in range(3):
            buffer.observe_train_batch(inputs[list(order)], targets[list(order)])
        assert set(partition(buffer).reservoir.sample_ids) == expected
        assert partition(buffer).reservoir.example_count == 2


def test_should_treat_arrived_changed_labels_as_new_content_without_accessing_future_labels():
    buffer = make_buffer()
    inputs, targets = arrays([1])
    buffer.observe_train_batch(inputs, targets)
    original_id = content_id(inputs, targets)
    assert buffer.retention.sample_ids == (original_id,)
    changed = np.ones((1, 1))
    changed_id = content_id(inputs, changed)
    buffer.observe_train_batch(inputs, changed)
    assert original_id != changed_id
    assert set(buffer.retention.sample_ids) == {original_id, changed_id}
    assert partition(buffer).overlap_examples == 2


@pytest.mark.parametrize(
    "recent,reservoir,expected_counts",
    [
        ((1, 72), (3, 72), (1, 3)),
        ((3, 72), (1, 72), (3, 1)),
        ((3, 47), (2, 48), (1, 2)),
        ((2, 48), (3, 47), (2, 1)),
    ],
)
def test_should_enforce_each_partition_row_and_byte_cap_even_when_global_space_remains(
    recent, reservoir, expected_counts
):
    buffer = make_buffer(recent=recent, reservoir=reservoir)
    inputs, targets = arrays(range(12))
    buffer.observe_train_batch(inputs, targets)
    report = partition(buffer)
    assert (report.recent.example_count, report.reservoir.example_count) == expected_counts
    assert report.recent.retained_bytes == 24 * expected_counts[0] <= recent[1]
    assert report.reservoir.retained_bytes == 24 * expected_counts[1] <= reservoir[1]
    expected_recent, expected_reservoir, _ = expected_membership(
        ids(inputs, targets), expected_counts[0], expected_counts[1], 53
    )
    assert_snapshot(report.recent, expected_recent)
    assert_snapshot(report.reservoir, expected_reservoir)


def test_should_preserve_union_newest_semantics_for_the_existing_recent_selector():
    buffer = make_buffer()
    inputs, targets = arrays(range(8))
    buffer.observe_train_batch(inputs, targets)
    _, _, union = expected_membership(ids(inputs, targets), 2, 2, 53)
    assert buffer.select_recent(2).sample_ids == union[-2:]
    assert buffer.select_recent(99).sample_ids == union


@pytest.mark.parametrize("maximum", [1, 2, 3, 99])
def test_should_alternate_unique_recent_and_ranked_reservoir_rows_within_each_turn(maximum):
    buffer = make_buffer(recent=(3, 72), reservoir=(3, 72))
    inputs, targets = arrays(range(10))
    buffer.observe_train_batch(inputs, targets)
    recent, reservoir, union = expected_membership(ids(inputs, targets), 3, 3, 53)
    # Force overlap at the first turn using a genuinely re-arrived ranked row.
    chosen = reservoir[0]
    index = ids(inputs, targets).index(chosen)
    buffer.observe_train_batch(inputs[index : index + 1], targets[index : index + 1])
    recent, reservoir, union = expected_membership((*ids(inputs, targets), chosen), 3, 3, 53)
    selection = buffer.select_partitioned(maximum)
    assert selection.sample_ids == expected_selection(recent, reservoir, maximum)
    assert len(selection.sample_ids) == min(maximum, len(union))
    assert len(set(selection.sample_ids)) == len(selection.sample_ids)
    assert selection.sample_ids[0] == chosen


@pytest.mark.parametrize("recent,reservoir", [((1, 24), (4, 96)), ((4, 96), (1, 24))])
def test_should_fill_from_the_other_partition_after_one_pool_is_exhausted(recent, reservoir):
    buffer = make_buffer(recent=recent, reservoir=reservoir)
    inputs, targets = arrays(range(20))
    buffer.observe_train_batch(inputs, targets)
    recent_ids, reservoir_ids, union = expected_membership(
        ids(inputs, targets), recent[0], reservoir[0], 53
    )
    assert buffer.select_partitioned(99).sample_ids == expected_selection(
        recent_ids, reservoir_ids, 99
    )
    assert set(buffer.select_partitioned(99).sample_ids) == set(union)


@pytest.mark.parametrize("maximum", [0, -1, True, 1.0, None])
def test_should_refuse_invalid_partitioned_selection_limits_without_mutating_rows(maximum):
    buffer = make_buffer()
    buffer.observe_train_batch(*arrays(range(4)))
    before = pickle.dumps(buffer)
    with pytest.raises(ValueError):
        buffer.select_partitioned(maximum)
    assert pickle.dumps(buffer) == before


def test_should_keep_default_mode_unpartitioned_and_refuse_implicit_partition_selection():
    buffer = SharedReplayBuffer(
        2, ReplayRetentionBudget(4, 96), ReplayRetentionPolicy("seeded_reservoir", 53)
    )
    assert buffer.allocation is buffer.partition_retention is None
    buffer.observe_train_batch(*arrays(range(8)))
    before = pickle.dumps(buffer)
    with pytest.raises(ValueError):
        buffer.select_partitioned(2)
    assert pickle.dumps(buffer) == before


@pytest.mark.parametrize("operation", ["observe", "select", "retention"])
def test_should_revalidate_malformed_owned_configuration_before_array_or_selection_work(operation):
    buffer = make_buffer()
    buffer.observe_train_batch(*arrays(range(4)))
    object.__setattr__(buffer.budget, "max_bytes", 0)
    before = pickle.dumps(buffer)
    invalid: Any = object()
    with pytest.raises(ValueError):
        if operation == "observe":
            buffer.observe_train_batch(invalid, invalid)
        elif operation == "select":
            buffer.select_partitioned(2)
        else:
            _ = buffer.partition_retention
    assert pickle.dumps(buffer) == before


def test_should_detach_observed_arrays_and_every_selection_consumer_without_changing_retention():
    buffer = make_buffer()
    inputs, targets = arrays(range(6))
    inputs, targets = inputs.astype(np.int16), targets.astype(np.int16)
    buffer.observe_train_batch(inputs, targets)
    before = pickle.dumps(buffer)
    selected = buffer.select_partitioned(4)
    first, second = selected.training_batches(), selected.training_batches()
    assert all(value.dtype == np.float64 for pair in first for value in pair)
    originals = tuple((x.copy(), y.copy()) for x, y in first)
    inputs.fill(999)
    targets.fill(0.5)
    for x, y in first:
        x.fill(-999)
        y.fill(1)
    for actual, expected in zip(second, originals):
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
    for x, y in selected._rows:
        x.fill(333)
        y.fill(0.5)
    assert pickle.dumps(buffer) == before
    current = buffer.select_partitioned(4)
    assert current.sample_ids == selected.sample_ids
    for actual, expected in zip(current.training_batches(), originals):
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize(
    "fault",
    [
        "rank",
        "width",
        "empty",
        "target_rows",
        "target_columns",
        "late_target_range",
        "late_nan",
        "object_dtype",
    ],
)
def test_should_validate_the_whole_arrived_batch_before_any_partition_mutation(fault):
    buffer = make_buffer()
    buffer.observe_train_batch(*arrays(range(4)))
    inputs, targets = arrays([100, 101])
    if fault == "rank":
        inputs = np.zeros(2)
    elif fault == "width":
        inputs = np.zeros((2, 3))
    elif fault == "empty":
        inputs, targets = np.zeros((0, 2)), np.zeros((0, 1))
    elif fault == "target_rows":
        targets = np.zeros((1, 1))
    elif fault == "target_columns":
        targets = np.zeros((2, 2))
    elif fault == "late_target_range":
        targets[-1, 0] = 2
    elif fault == "late_nan":
        inputs[-1, 0] = np.nan
    else:
        inputs = np.array([[object(), object()], [object(), object()]], dtype=object)
    before = pickle.dumps(buffer)
    with pytest.raises(ValueError):
        buffer.observe_train_batch(inputs, targets)
    assert pickle.dumps(buffer) == before


@pytest.mark.parametrize("seed", [17, 19, 53])
def test_should_keep_a_bounded_distinct_old_context_reservoir_across_long_arrived_a_b_a_stream(
    seed,
):
    buffer = make_buffer(seed=seed, recent=(4, 96), reservoir=(8, 192))
    observed: list[str] = []
    context_a: set[str] = set()
    for phase, (numbers, label) in enumerate(
        ((range(64), 0.0), (range(100, 164), 1.0), (range(64), 0.0))
    ):
        inputs, targets = arrays(numbers, label)
        phase_ids = ids(inputs, targets)
        if phase == 0:
            context_a.update(phase_ids)
        # Observe only the currently arrived row; reference never sees a future label.
        for index, sample_id in enumerate(phase_ids):
            buffer.observe_train_batch(inputs[index : index + 1], targets[index : index + 1])
            observed.append(sample_id)
            recent, reservoir, union = expected_membership(observed, 4, 8, seed)
            report = partition(buffer)
            assert_snapshot(report.recent, recent)
            assert_snapshot(report.reservoir, reservoir)
            assert_snapshot(report.unique, set(union))
            assert buffer.retained_order_ids == union
            assert report.unique.example_count <= 12 and report.unique.retained_bytes <= 288
            assert len(stored_arrays(buffer)) == 2 * report.unique.example_count
            assert stored_content_ids(buffer) == set(union)
        if phase == 1:
            assert not set(partition(buffer).recent.sample_ids) & context_a
            # A finite reservoir may retain zero old rows; do not select favorable seeds.
            expected_old = set(reservoir) & context_a
            actual_old = set(partition(buffer).reservoir.sample_ids) & context_a
            assert actual_old == expected_old and len(actual_old) <= 8
    assert set(buffer.select_partitioned(12).sample_ids) == set(buffer.retention.sample_ids)
