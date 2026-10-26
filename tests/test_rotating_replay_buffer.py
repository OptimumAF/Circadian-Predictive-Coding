"""Persistent recent/reservoir turns with bounded, failure-atomic cursor state."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, fields, is_dataclass
import pickle

import numpy as np
import pytest

from src.core import shared_replay_schedule as module
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.shared_replay_schedule import ReplayRotationCursor, SharedReplayBuffer
from test_partitioned_replay_buffer import (
    arrays,
    expected_membership,
    ids,
    make_buffer,
    ranked,
    stored_arrays,
)


def rotation_reference(recent, reservoir, previous, maximum, seed=53):
    """Build a finite circular rank order, then consume distinct proposal turns."""
    ordered = tuple(sorted(reservoir, key=lambda value: ranked(value, seed)))
    after = previous.reservoir_after
    if after is not None:
        later = tuple(value for value in ordered if ranked(value, seed) > ranked(after, seed))
        earlier = tuple(value for value in ordered if ranked(value, seed) <= ranked(after, seed))
        ordered = later + earlier
    pools = [list(reversed(recent)), list(ordered)]
    selected: list[str] = []
    turn = previous.next_partition
    while len(selected) < min(maximum, len(set(recent) | set(reservoir))):
        emitted = None
        for requested in (turn, 1 - turn):
            while pools[requested]:
                candidate = pools[requested].pop(0)
                if requested == 1:
                    after = candidate
                if candidate not in selected:
                    emitted = candidate
                    break
            if emitted is not None:
                break
        if emitted is None:
            break
        selected.append(emitted)
        turn = 1 - turn
    if not selected:
        return (), previous
    return tuple(selected), ReplayRotationCursor(turn, after)


def populated(*, recent=(2, 48), reservoir=(3, 72)):
    buffer = make_buffer(recent=recent, reservoir=reservoir)
    inputs, targets = arrays(range(12))
    buffer.observe_train_batch(inputs, targets)
    membership = expected_membership(ids(inputs, targets), recent[0], reservoir[0], 53)
    return buffer, membership


def test_should_begin_at_recent_with_no_reservoir_position_including_the_default_mode():
    buffers = (
        make_buffer(),
        SharedReplayBuffer(2, ReplayRetentionBudget(4, 96), ReplayRetentionPolicy("recent_fifo")),
    )
    for buffer in buffers:
        cursor = buffer.rotation_cursor
        assert type(cursor) is ReplayRotationCursor
        assert asdict(cursor) == {"next_partition": 0, "reservoir_after": None}
        with pytest.raises(FrozenInstanceError):
            setattr(cursor, "next_partition", 1)
        object.__setattr__(cursor, "next_partition", 1)
        assert buffer.rotation_cursor == ReplayRotationCursor(0, None)


@pytest.mark.parametrize(
    "turn,after",
    [
        (True, None),
        (-1, None),
        (2, None),
        (1.0, None),
        (0, ""),
        (0, "A" * 64),
        (0, "z" * 64),
        (0, b"0" * 64),
    ],
)
def test_should_validate_exact_cursor_fields_without_accepting_noncanonical_content_ids(
    turn, after
):
    with pytest.raises(ValueError):
        ReplayRotationCursor(turn, after)


def test_should_persist_requested_turns_when_each_call_can_emit_only_one_row():
    buffer, (recent, reservoir, _) = populated()
    expected = (recent[-1], reservoir[0], recent[-1], reservoir[1])
    actual = tuple(buffer.select_partitioned_rotating(1).sample_ids[0] for _ in range(4))
    assert actual == expected
    assert buffer.rotation_cursor == ReplayRotationCursor(0, reservoir[1])


@pytest.mark.parametrize("maximum", [1, 2, 3, 99])
def test_should_match_an_independent_persistent_circular_rank_oracle_across_wraps(maximum):
    buffer, (recent, reservoir, union) = populated()
    cursor = ReplayRotationCursor(0, None)
    before = (buffer.retention, buffer.partition_retention, buffer.retained_order_ids)
    for _ in range(8):
        expected, cursor = rotation_reference(recent, reservoir, cursor, maximum)
        selected = buffer.select_partitioned_rotating(maximum)
        assert selected.sample_ids == expected
        assert len(expected) == min(maximum, len(union)) and len(set(expected)) == len(expected)
        assert buffer.rotation_cursor == cursor
        assert (buffer.retention, buffer.partition_retention, buffer.retained_order_ids) == before


@pytest.mark.parametrize("reservoir_size", [1, 2, 4])
@pytest.mark.parametrize("maximum", [1, 2])
def test_should_propose_every_stable_reservoir_member_within_twice_its_size_successful_calls(
    reservoir_size, maximum
):
    buffer, (_, reservoir, _) = populated(reservoir=(reservoir_size, 24 * reservoir_size))
    exposed: set[str] = set()
    for _ in range(2 * reservoir_size):
        selection = buffer.select_partitioned_rotating(maximum)
        assert selection.sample_ids
        exposed.update(selection.sample_ids)
    assert set(reservoir) <= exposed


def test_should_record_a_scanned_duplicate_even_when_its_reservoir_turn_falls_back():
    buffer = make_buffer(recent=(2, 48), reservoir=(1, 24))
    inputs, targets = arrays(range(8))
    all_ids = ids(inputs, targets)
    chosen = min(all_ids, key=lambda value: ranked(value, 53))
    index = all_ids.index(chosen)
    buffer.observe_train_batch(inputs, targets)
    buffer.observe_train_batch(inputs[index : index + 1], targets[index : index + 1])
    recent, _, _ = expected_membership((*all_ids, chosen), 2, 1, 53)
    selected = buffer.select_partitioned_rotating(2)
    assert selected.sample_ids == tuple(reversed(recent))
    assert buffer.rotation_cursor == ReplayRotationCursor(0, chosen)


@pytest.mark.parametrize("recent,reservoir", [((1, 24), (4, 96)), ((4, 96), (1, 24))])
def test_should_fall_back_after_one_pool_is_exhausted_without_repeating_a_candidate_in_the_call(
    recent, reservoir
):
    buffer, (recent_ids, reservoir_ids, union) = populated(recent=recent, reservoir=reservoir)
    previous = ReplayRotationCursor(0, None)
    for _ in range(3):
        expected, previous = rotation_reference(recent_ids, reservoir_ids, previous, 99)
        selection = buffer.select_partitioned_rotating(99)
        assert selection.sample_ids == expected and set(expected) == set(union)
        assert buffer.rotation_cursor == previous


@pytest.mark.parametrize("maximum", [0, -1, True, 1.0, None])
def test_should_refuse_invalid_limits_without_advancing_or_mutating_the_cursor_and_pools(maximum):
    buffer, _ = populated()
    buffer.select_partitioned_rotating(1)
    before = pickle.dumps(buffer)
    with pytest.raises(ValueError):
        buffer.select_partitioned_rotating(maximum)
    assert pickle.dumps(buffer) == before


def test_should_validate_malformed_configuration_before_advancing_rotation():
    buffer, _ = populated()
    buffer.select_partitioned_rotating(1)
    object.__setattr__(buffer.budget, "max_bytes", 0)
    before = pickle.dumps(buffer)
    with pytest.raises(ValueError):
        buffer.select_partitioned_rotating(2)
    assert pickle.dumps(buffer) == before


class CopyFailure(np.ndarray):
    def copy(self, *args, **kwargs):
        raise MemoryError("injected selection array copy failure")


@pytest.mark.parametrize("fault", ["row_copy", "selection_construction"])
def test_should_publish_no_cursor_until_all_return_copies_and_selection_construction_succeed(
    fault, monkeypatch
):
    buffer, _ = populated()
    buffer.select_partitioned_rotating(1)
    error = MemoryError("injected selection construction failure")
    if fault == "row_copy":
        # Why this: a faulting ndarray view injects allocation failure at copy(),
        # preserving the already-admitted row's values, geometry and membership.
        for sample_id, (features, targets) in tuple(buffer._rows.items()):
            buffer._rows[sample_id] = (features.view(CopyFailure), targets)
    else:

        def fail(*args, **kwargs):
            raise error

        monkeypatch.setattr(module, "SharedReplaySelection", fail)
    cursor = buffer.rotation_cursor
    order = buffer.retained_order_ids
    storage = tuple(
        (id(features), features.tobytes(), id(targets), targets.tobytes())
        for features, targets in buffer._rows.values()
    )
    with pytest.raises(MemoryError) as caught:
        buffer.select_partitioned_rotating(3)
    if fault == "selection_construction":
        assert caught.value is error
    else:
        assert str(caught.value) == "injected selection array copy failure"
    assert buffer.rotation_cursor == cursor and buffer.retained_order_ids == order
    assert (
        tuple(
            (id(features), features.tobytes(), id(targets), targets.tobytes())
            for features, targets in buffer._rows.values()
        )
        == storage
    )


def test_should_leave_empty_rotation_and_existing_selectors_and_observations_without_cursor_effects():
    empty = make_buffer()
    initial = empty.rotation_cursor
    assert empty.select_partitioned_rotating(2).sample_ids == ()
    assert empty.rotation_cursor == initial
    buffer, _ = populated()
    buffer.select_partitioned_rotating(2)
    cursor = buffer.rotation_cursor
    buffer.select_recent(3)
    buffer.select_partitioned(3)
    buffer.observe_train_batch(*arrays([100, 101]))
    with pytest.raises(ValueError):
        buffer.observe_train_batch(np.zeros((1, 2)), np.array([[2.0]]))
    assert buffer.rotation_cursor == cursor


def test_should_resume_strictly_by_rank_after_the_last_scanned_id_has_been_evicted():
    inputs, targets = arrays(range(20))
    all_ids = ids(inputs, targets)
    ordering = sorted(range(20), key=lambda index: ranked(all_ids[index], 53))
    first, later = ordering[-4:], ordering[:4]
    buffer = make_buffer()
    buffer.observe_train_batch(inputs[first], targets[first])
    buffer.select_partitioned_rotating(2)
    cursor = buffer.rotation_cursor
    assert cursor.reservoir_after is not None
    buffer.observe_train_batch(inputs[later], targets[later])
    assert cursor.reservoir_after not in buffer.retention.sample_ids
    assert buffer.rotation_cursor == cursor
    recent, reservoir, _ = expected_membership(
        tuple(all_ids[index] for index in (*first, *later)), 2, 2, 53
    )
    expected, cursor = rotation_reference(recent, reservoir, cursor, 2)
    assert buffer.select_partitioned_rotating(2).sample_ids == expected
    assert buffer.rotation_cursor == cursor


@pytest.mark.parametrize("copy_kind", ["deepcopy", "pickle"])
def test_should_preserve_the_trusted_cursor_and_following_sequence_across_in_memory_copy(copy_kind):
    buffer, _ = populated()
    buffer.select_partitioned_rotating(3)
    copied = deepcopy(buffer) if copy_kind == "deepcopy" else pickle.loads(pickle.dumps(buffer))
    assert copied.rotation_cursor == buffer.rotation_cursor
    assert all(
        not np.shares_memory(left, right)
        for left in stored_arrays(buffer)
        for right in stored_arrays(copied)
    )
    for maximum in (1, 2, 1, 99, 3):
        assert (
            copied.select_partitioned_rotating(maximum).sample_ids
            == buffer.select_partitioned_rotating(maximum).sample_ids
        )
        assert copied.rotation_cursor == buffer.rotation_cursor


def test_should_detach_cursor_and_selected_arrays_and_retain_only_one_bounded_cursor_record():
    buffer, _ = populated()
    length = len(pickle.dumps(buffer))
    selected = buffer.select_partitioned_rotating(2)
    before = tuple((value.tobytes(), id(value)) for value in stored_arrays(buffer))
    for features, targets in selected._rows:
        features.fill(999)
        targets.fill(0.5)
    returned = buffer.rotation_cursor
    object.__setattr__(returned, "reservoir_after", "0" * 64)
    assert buffer.rotation_cursor != returned
    for _ in range(64):
        buffer.select_partitioned_rotating(1)
    assert tuple((value.tobytes(), id(value)) for value in stored_arrays(buffer)) == before
    assert len(pickle.dumps(buffer)) <= length + 512
    assert cursor_records(buffer) == 1


def cursor_records(buffer):
    found: set[int] = set()
    seen: set[int] = set()

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if type(value) is ReplayRotationCursor:
            found.add(id(value))
        if is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name))
        elif type(value) is dict:
            for child in value.values():
                visit(child)
        elif type(value) in (tuple, list):
            for child in value:
                visit(child)

    visit(vars(buffer))
    return len(found)


def test_should_refuse_rotation_without_allocation_and_keep_default_buffer_behavior_unchanged():
    buffer = SharedReplayBuffer(
        2, ReplayRetentionBudget(4, 96), ReplayRetentionPolicy("recent_fifo")
    )
    buffer.observe_train_batch(*arrays(range(6)))
    before = pickle.dumps(buffer)
    with pytest.raises(ValueError):
        buffer.select_partitioned_rotating(1)
    assert pickle.dumps(buffer) == before
    assert buffer.select_recent(2).sample_ids == buffer.retained_order_ids[-2:]
