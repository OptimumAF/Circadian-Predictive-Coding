"""FIFO retention/count tradeoffs and atomic native state boundaries, without models."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from src.adapters.action_memory_learner import ActionMemoryLearner
from src.adapters.fifo_action_memory_learner import FifoActionMemoryLearner
from src.core.action_memory import (
    ACTION_MEMORY_COUNT_DIAGNOSTIC,
    ActionMemoryEntry,
    ActionMemoryPolicy,
)
from src.core.fifo_action_memory import (
    FIFO_ACTION_MEMORY_COUNT_DIAGNOSTIC,
    FIFO_RETIREMENT_METADATA_BYTES,
    FifoActionMemoryPolicy,
    FifoActionMemorySnapshot,
    convert_action_memory_to_fifo,
    copy_fifo_action_memory_policy,
    copy_fifo_action_memory_snapshot,
    count_fifo_action_observations,
    predict_fifo_action_memory,
)


GEOMETRIES = ((16, 5), (296, 7))


def policy(geometry=(16, 5), **options: Any) -> FifoActionMemoryPolicy:
    return FifoActionMemoryPolicy(feature_dim=geometry[0], num_classes=geometry[1], **options)


def features(keys: list[float], width=16) -> Any:
    values = np.zeros((len(keys), width), dtype=np.float64)
    values[:, -1] = keys
    return values


def labels(values: list[int]) -> Any:
    return np.array(values, dtype=np.float64).reshape(-1, 1)


def retained_keys(state) -> list[float]:
    return [entry.features[-1] for entry in state.entries]


def exposures(state) -> int:
    return sum(sum(entry.counts) for entry in state.entries)


def trained() -> FifoActionMemoryLearner:
    learner = FifoActionMemoryLearner(policy(max_entries=2))
    learner.train_batch(features([0.0, 0.25, 0.5]), labels([0, 1, 2]))
    return learner


def test_should_keep_explicit_distinct_identity_geometry_caps_and_metadata_cost():
    old, large = policy(), policy((296, 7))
    assert old.method == large.method == "observation_memory_fifo"
    assert (old.max_entries, old.max_payload_bytes, old.entry_payload_bytes) == (64, 10752, 168)
    assert (large.max_entries, large.max_payload_bytes, large.entry_payload_bytes) == (
        64,
        10752,
        2424,
    )
    assert policy((296, 7), max_payload_bytes=155136).max_payload_bytes == 64 * 2424
    assert FIFO_RETIREMENT_METADATA_BYTES == 16
    assert FIFO_ACTION_MEMORY_COUNT_DIAGNOSTIC != ACTION_MEMORY_COUNT_DIAGNOSTIC
    for declared in (old, large):
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(declared, "entry_payload_bytes", 1)
        copied = copy_fifo_action_memory_policy(declared)
        assert copied == declared and copied is not declared


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_should_evict_oldest_inserted_key_without_refreshing_hits_and_reinsert_fresh(geometry):
    width, classes = geometry
    learner = FifoActionMemoryLearner(policy(geometry, max_entries=2))
    learner.train_batch(features([0.0, 0.0, 0.25], width), labels([0, 1, 2]))
    learner.train_batch(features([0.0], width), labels([classes - 1]))
    assert retained_keys(learner.snapshot_state()) == [0.0, 0.25]
    learner.train_batch(features([0.5], width), labels([3]))
    first = learner.snapshot_state()
    assert retained_keys(first) == [0.25, 0.5]
    assert (first.evictions, first.retired_examples, first.training_examples) == (1, 3, 5)
    np.testing.assert_array_equal(
        learner.predict(features([0.0], width)), np.full((1, classes), 1 / classes)
    )
    learner.train_batch(features([0.25], width), labels([0]))
    assert retained_keys(learner.snapshot_state()) == [0.25, 0.5]
    diagnostic = learner.train_batch(features([0.0], width), labels([1]))
    final = learner.snapshot_state()
    assert retained_keys(final) == [0.5, 0.0]
    assert final.entries[-1].counts == (0, 1) + (0,) * (classes - 2)
    assert (
        final.completed_updates,
        final.training_examples,
        final.evictions,
        final.retired_examples,
    ) == (5, 7, 2, 5)
    assert exposures(final) + final.retired_examples == final.training_examples
    assert diagnostic.definition == FIFO_ACTION_MEMORY_COUNT_DIAGNOSTIC and diagnostic.value == 1.0


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_should_process_over_capacity_batch_in_row_order_retiring_intermediate_counts(geometry):
    width, classes = geometry
    learner = FifoActionMemoryLearner(policy(geometry, max_entries=2))
    learner.train_batch(
        features([0.0, 0.25, 0.0, 0.5, 0.75, 0.0], width), labels([0, 1, 2, 3, 4, 1])
    )
    state = learner.snapshot_state()
    assert retained_keys(state) == [0.75, 0.0]
    assert state.entries[0].counts == (0, 0, 0, 0, 1) + (0,) * (classes - 5)
    assert state.entries[1].counts == (0, 1) + (0,) * (classes - 2)
    assert (
        state.completed_updates,
        state.training_examples,
        state.evictions,
        state.retired_examples,
    ) == (1, 6, 3, 4)
    assert exposures(state) == 2
    assert (
        len(state.entries) * state.policy.entry_payload_bytes + FIFO_RETIREMENT_METADATA_BYTES
        == 2 * (width + classes) * 8 + 16
    )


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_should_count_duplicate_conflicts_before_retiring_them_at_capacity_one(geometry):
    width, classes = geometry
    learner = FifoActionMemoryLearner(policy(geometry, max_entries=1))
    learner.train_batch(
        features([0.0, 0.0, 0.25, 0.25, 0.0], width), labels([0, 1, 2, 2, classes - 1])
    )
    state = learner.snapshot_state()
    assert retained_keys(state) == [0.0]
    assert state.entries[0].counts == (0,) * (classes - 1) + (1,)
    assert (state.evictions, state.retired_examples, state.training_examples) == (2, 4, 5)
    np.testing.assert_array_equal(
        learner.predict(features([0.0], width)), [[0] * (classes - 1) + [1]]
    )


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("limiter", ["entries", "bytes"])
def test_should_use_the_smaller_entry_or_whole_byte_capacity_and_keep_metadata_separate(
    geometry, limiter
):
    width, classes = geometry
    entry_bytes = (width + classes) * 8
    limits = (
        dict(max_entries=2, max_payload_bytes=3 * entry_bytes)
        if limiter == "entries"
        else dict(max_entries=4, max_payload_bytes=3 * entry_bytes - 1)
    )
    learner = FifoActionMemoryLearner(policy(geometry, **limits))
    learner.train_batch(features([0.0, 0.25, 0.5], width), labels([0, 1, 2]))
    state = learner.snapshot_state()
    assert retained_keys(state) == [0.25, 0.5]
    assert state.evictions == state.retired_examples == 1
    assert len(state.entries) * entry_bytes <= state.policy.max_payload_bytes
    assert 3 * entry_bytes > state.policy.max_payload_bytes or state.policy.max_entries == 2


def test_should_keep_the_unraised_minigrid_default_byte_budget_and_evict_the_fifth_key():
    learner = FifoActionMemoryLearner(policy((296, 7)))
    learner.train_batch(features([0.0, 0.125, 0.25, 0.375, 0.5], 296), labels([0, 1, 2, 3, 6]))
    state = learner.snapshot_state()
    assert retained_keys(state) == [0.125, 0.25, 0.375, 0.5]
    assert state.policy.max_payload_bytes == 10752 and 4 * 2424 == 9696
    assert state.evictions == state.retired_examples == 1


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_should_refuse_zero_effective_capacity_without_publishing_any_exposure(geometry):
    width, classes = geometry
    learner = FifoActionMemoryLearner(policy(geometry, max_payload_bytes=(width + classes) * 8 - 1))
    before = learner.snapshot_state()
    np.testing.assert_array_equal(
        learner.predict(features([0.0], width)), np.full((1, classes), 1 / classes)
    )
    with pytest.raises(ValueError):
        learner.train_batch(features([0.0], width), labels([classes - 1]))
    assert learner.snapshot_state() == before
    assert (before.evictions, before.retired_examples, before.training_examples) == (0, 0, 0)


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_should_preserve_exact_keys_uniform_misses_conflict_probabilities_and_first_argmax(
    geometry,
):
    width, classes = geometry
    learner = FifoActionMemoryLearner(policy(geometry, max_entries=2))
    learner.train_batch(features([0.0, 0.0], width), labels([classes - 1, 1]))
    p = learner.predict(features([0.0], width))
    expected = np.zeros((1, classes), dtype=np.float64)
    expected[0, 1] = expected[0, -1] = 0.5
    np.testing.assert_array_equal(p, expected)
    assert np.argmax(p[0]) == 1
    changed = features([np.nextafter(0.0, 1.0)], width)
    np.testing.assert_array_equal(learner.predict(changed), np.full((1, classes), 1 / classes))
    p.fill(0)
    assert np.argmax(learner.predict(features([0.0], width))[0]) == 1


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_should_own_input_arrays_policies_snapshots_forks_restores_and_pure_transitions(geometry):
    width, classes = geometry
    declared = policy(geometry, max_entries=2)
    learner = FifoActionMemoryLearner(declared)
    x, y = features([0.0, 0.25, 0.5], width), labels([0, 1, classes - 1])
    empty = learner.snapshot_state()
    expected = count_fifo_action_observations(empty, x, y)
    learner.train_batch(x, y)
    x.fill(0.75)
    y.fill(0)
    object.__setattr__(declared, "max_entries", 1)
    assert learner.snapshot_state() == expected and empty.entries == ()
    child = learner.fork()
    restored = FifoActionMemoryLearner(policy(geometry, max_entries=2))
    restored.restore_state(expected)
    snapshot = child.snapshot_state()
    assert snapshot == restored.snapshot_state() == expected
    assert snapshot is not expected and snapshot.policy is not expected.policy
    assert snapshot.entries is not expected.entries
    assert all(
        left is not right for left, right in zip(snapshot.entries, expected.entries, strict=True)
    )
    object.__setattr__(expected.entries[0], "features", (0.875,) * width)
    object.__setattr__(expected.policy, "max_entries", 1)
    assert child.snapshot_state() == restored.snapshot_state() == learner.snapshot_state()
    child.train_batch(features([0.75], width), labels([2]))
    assert (
        child.training_examples == 4
        and learner.training_examples == restored.training_examples == 3
    )
    assert child.snapshot_state().retired_examples == 2


class ArraySubclass(np.ndarray):
    pass


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "subclass",
        "float32",
        "bool",
        "width",
        "empty",
        "129rows",
        "late_nan",
        "late_inf",
        "late_range",
        "late_label_nan",
        "late_label_class",
        "late_label_fraction",
        "label_float32",
        "label_bool",
        "label_rows",
    ],
)
def test_should_preflight_the_entire_batch_before_earlier_rows_can_evict_or_increment(fault):
    learner = trained()
    before = learner.snapshot_state()
    x: Any = features([0.75, 0.875])
    y: Any = labels([3, 4])
    if fault == "list":
        x = x.tolist()
    elif fault == "subclass":
        x = x.view(ArraySubclass)
    elif fault in ("float32", "bool"):
        x = x.astype(np.float32 if fault == "float32" else bool)
    elif fault == "width":
        x = np.zeros((2, 296), dtype=np.float64)
    elif fault in ("empty", "129rows"):
        rows = 0 if fault == "empty" else 129
        x, y = np.zeros((rows, 16)), np.zeros((rows, 1))
    elif fault in ("late_nan", "late_inf", "late_range"):
        x[1, -1] = {"late_nan": float("nan"), "late_inf": float("inf"), "late_range": -1.01}[fault]
    elif fault in ("late_label_nan", "late_label_class", "late_label_fraction"):
        y[1, 0] = {
            "late_label_nan": float("nan"),
            "late_label_class": 5,
            "late_label_fraction": 0.5,
        }[fault]
    elif fault in ("label_float32", "label_bool"):
        y = y.astype(np.float32 if fault == "label_float32" else bool)
    else:
        y = y[:1]
    with pytest.raises(ValueError):
        learner.train_batch(x, y)
    assert learner.snapshot_state() == before


@pytest.mark.parametrize(
    "fault",
    [
        "extra_snapshot",
        "extra_policy",
        "extra_entry",
        "entries_list",
        "duplicate_keys",
        "foreign_geometry",
        "features_numpy",
        "counts_bool",
        "count_overflow",
        "evictions_bool",
        "evictions_negative",
        "retired_bool",
        "retired_negative",
        "retired_overflow",
        "evictions_over_retired",
        "zero_evictions_retired",
        "zero_retired_evictions",
        "wrong_lifetime_sum",
        "updates_bool",
        "counter_overflow",
        "evicted_underfull",
        "empty_after_history",
    ],
)
def test_should_refuse_tampered_fifo_snapshots_and_retirement_invariants_atomically(fault):
    learner = trained()
    before = learner.snapshot_state()
    invalid: Any = deepcopy(before)
    entry = invalid.entries[0]
    if fault in ("extra_snapshot", "extra_policy", "extra_entry"):
        owner = {"extra_snapshot": invalid, "extra_policy": invalid.policy, "extra_entry": entry}[
            fault
        ]
        object.__setattr__(owner, "extra", 1)
    elif fault == "entries_list":
        object.__setattr__(invalid, "entries", list(invalid.entries))
    elif fault == "duplicate_keys":
        object.__setattr__(invalid, "entries", (entry, entry))
    elif fault == "foreign_geometry":
        object.__setattr__(
            invalid,
            "entries",
            (ActionMemoryEntry((0.0,) * 296, (1, 0, 0, 0, 0, 0, 0)), invalid.entries[1]),
        )
    elif fault == "features_numpy":
        object.__setattr__(entry, "features", (np.float64(0.0),) + entry.features[1:])
    elif fault == "counts_bool":
        object.__setattr__(entry, "counts", (True, 0, 0, 0, 0))
    elif fault == "count_overflow":
        object.__setattr__(entry, "counts", (2**63, 0, 0, 0, 0))
    elif fault == "evictions_bool":
        object.__setattr__(invalid, "evictions", True)
    elif fault == "evictions_negative":
        object.__setattr__(invalid, "evictions", -1)
    elif fault == "retired_bool":
        object.__setattr__(invalid, "retired_examples", True)
    elif fault == "retired_negative":
        object.__setattr__(invalid, "retired_examples", -1)
    elif fault == "retired_overflow":
        object.__setattr__(invalid, "retired_examples", 2**53)
    elif fault == "evictions_over_retired":
        object.__setattr__(invalid, "evictions", 2)
    elif fault == "zero_evictions_retired":
        object.__setattr__(invalid, "evictions", 0)
    elif fault == "zero_retired_evictions":
        object.__setattr__(invalid, "retired_examples", 0)
    elif fault == "wrong_lifetime_sum":
        object.__setattr__(invalid, "training_examples", 4)
    elif fault == "updates_bool":
        object.__setattr__(invalid, "completed_updates", True)
    elif fault == "counter_overflow":
        object.__setattr__(invalid, "completed_updates", 2**53)
    elif fault == "evicted_underfull":
        object.__setattr__(invalid, "policy", policy(max_entries=3))
    else:
        object.__setattr__(invalid, "entries", ())
    with pytest.raises(ValueError):
        learner.restore_state(invalid)
    assert learner.snapshot_state() == before


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_should_refuse_lifetime_overflow_without_changing_retained_or_retired_counts(geometry):
    width, classes = geometry
    maximum = 2**53 - 1
    state = FifoActionMemorySnapshot(
        policy(geometry, max_entries=1),
        (ActionMemoryEntry((0.0,) * width, (1,) + (0,) * (classes - 1)),),
        maximum,
        maximum,
        maximum - 1,
        maximum - 1,
    )
    learner = FifoActionMemoryLearner(state.policy)
    learner.restore_state(state)
    before = learner.snapshot_state()
    with pytest.raises(ValueError):
        learner.train_batch(features([0.25], width), labels([classes - 1]))
    assert learner.snapshot_state() == before


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_should_explicitly_convert_old_state_own_its_queue_and_preserve_all_native_clocks(geometry):
    width, classes = geometry
    old_policy = ActionMemoryPolicy(feature_dim=width, num_classes=classes, max_entries=2)
    old = ActionMemoryLearner(old_policy)
    old.train_batch(features([0.0, 0.25, 0.0], width), labels([0, 1, classes - 1]))
    original = old.snapshot_state()
    converted = convert_action_memory_to_fifo(original)
    assert (
        converted.policy.method == "observation_memory_fifo"
        and original.policy.method == "observation_memory"
    )
    assert (
        converted.completed_updates,
        converted.training_examples,
        converted.evictions,
        converted.retired_examples,
    ) == (1, 3, 0, 0)
    assert converted.entries == original.entries and converted.entries is not original.entries
    assert all(
        left is not right for left, right in zip(converted.entries, original.entries, strict=True)
    )
    fifo = FifoActionMemoryLearner(converted.policy)
    fifo.restore_state(converted)
    np.testing.assert_array_equal(
        fifo.predict(features([0.0, 0.25], width)), old.predict(features([0.0, 0.25], width))
    )
    invalid_old: Any = original
    invalid_fifo: Any = converted
    with pytest.raises(ValueError):
        fifo.restore_state(invalid_old)
    with pytest.raises(ValueError):
        old.restore_state(invalid_fifo)
    object.__setattr__(original.entries[0], "features", (0.75,) * width)
    assert retained_keys(fifo.snapshot_state()) == [0.0, 0.25]
    fifo.train_batch(features([0.5], width), labels([2]))
    state = fifo.snapshot_state()
    assert (
        retained_keys(state) == [0.25, 0.5] and state.retired_examples == 2 and state.evictions == 1
    )
    assert retained_keys(old.snapshot_state()) == [0.0, 0.25]
    with pytest.raises(ValueError):
        old.train_batch(features([0.5], width), labels([2]))


@pytest.mark.parametrize("geometry", [(16, 7), (296, 5), (148, 7), (295, 7)])
def test_should_refuse_unfrozen_geometry_pairs(geometry):
    with pytest.raises(ValueError):
        policy(geometry)


@pytest.mark.parametrize(
    "field,value",
    [
        ("method", "observation_memory"),
        ("feature_dim", True),
        ("num_classes", np.int64(5)),
        ("max_entries", 0),
        ("max_entries", 65),
        ("max_entries", False),
        ("max_payload_bytes", 0),
        ("max_payload_bytes", 10753),
    ],
)
def test_should_refuse_invalid_native_policy_fields(field, value):
    with pytest.raises(ValueError):
        FifoActionMemoryPolicy(**{field: value})


@pytest.mark.parametrize("old_target", [False, True])
def test_should_refuse_cross_geometry_fifo_restore_even_with_empty_valid_states(old_target):
    target = FifoActionMemoryLearner(policy((16, 5) if old_target else (296, 7)))
    foreign = FifoActionMemoryLearner(policy((296, 7) if old_target else (16, 5)))
    before = target.snapshot_state()
    with pytest.raises(ValueError):
        target.restore_state(foreign.snapshot_state())
    assert target.snapshot_state() == before


def test_should_refuse_subclass_states_or_policies_and_copy_conversion_only_from_exact_old_state():
    class PolicySubclass(FifoActionMemoryPolicy):
        pass

    class SnapshotSubclass(FifoActionMemorySnapshot):
        pass

    with pytest.raises(ValueError):
        FifoActionMemoryLearner(PolicySubclass())
    before = trained().snapshot_state()
    invalid: Any = SnapshotSubclass(
        before.policy,
        before.entries,
        before.completed_updates,
        before.training_examples,
        before.evictions,
        before.retired_examples,
    )
    with pytest.raises(ValueError):
        copy_fifo_action_memory_snapshot(invalid)
    not_old: Any = before
    with pytest.raises(ValueError):
        convert_action_memory_to_fifo(not_old)
    np.testing.assert_array_equal(
        predict_fifo_action_memory(before, features([0.25])), [[0, 1, 0, 0, 0]]
    )
