"""Exact count-memory behavior and atomic capacity/state admission without Torch."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.adapters.action_memory_learner import ActionMemoryLearner
from src.core.action_memory import (
    ACTION_MEMORY_COUNT_DIAGNOSTIC,
    ACTION_MEMORY_ENTRY_BYTES,
    ActionMemoryEntry,
    ActionMemoryPolicy,
    ActionMemorySnapshot,
    copy_action_memory_snapshot,
    count_action_observations,
    predict_action_memory,
)


def features(keys):
    values = np.zeros((len(keys), 16), dtype=np.float64)
    values[:, 0] = keys
    return values


def labels(values):
    return np.array(values, dtype=np.float64).reshape(-1, 1)


def trained():
    learner = ActionMemoryLearner()
    learner.train_batch(features([0.0, 0.0, 0.25]), labels([0, 1, 4]))
    return learner


def test_should_freeze_default_policy_and_logical_payload_law():
    policy = ActionMemoryPolicy()
    assert (policy.method, policy.feature_dim, policy.num_classes) == ("observation_memory", 16, 5)
    assert policy.max_entries == 64 and policy.max_payload_bytes == 10752
    assert ACTION_MEMORY_ENTRY_BYTES == 168 and 64 * ACTION_MEMORY_ENTRY_BYTES == 10752
    state = ActionMemoryLearner().snapshot_state()
    assert state.entries == () and state.completed_updates == state.training_examples == 0


def test_should_return_uniform_for_unseen_keys_without_mutating_memory():
    learner = ActionMemoryLearner()
    before = learner.snapshot_state()
    result = learner.predict(features([0.0, 0.25]))
    np.testing.assert_array_equal(result, np.full((2, 5), 0.2))
    assert result.dtype == np.float64
    result.fill(0)
    np.testing.assert_array_equal(learner.predict(features([0.0])), np.full((1, 5), 0.2))
    assert learner.snapshot_state() == before


def test_should_count_every_conflicting_exposure_without_overwriting_or_hidden_disambiguation():
    learner = trained()
    np.testing.assert_array_equal(
        learner.predict(features([0.0, 0.25, 0.5])),
        [[0.5, 0.5, 0, 0, 0], [0, 0, 0, 0, 1], [0.2] * 5],
    )
    result = learner.train_batch(features([0.0, 0.0, 0.0]), labels([1, 2, 1]))
    state = learner.snapshot_state()
    assert state.entries[0].counts == (1, 3, 1, 0, 0)
    np.testing.assert_array_equal(learner.predict(features([0.0])), [[0.2, 0.6, 0.2, 0, 0]])
    assert (state.completed_updates, state.training_examples) == (2, 6)
    assert result.definition == ACTION_MEMORY_COUNT_DIAGNOSTIC and result.value == 3.0


def test_should_use_exact_sixteen_coordinate_keys_without_rounding_or_context_input():
    learner = ActionMemoryLearner()
    first = features([0.0])
    learner.train_batch(first, labels([3]))
    changed = first.copy()
    changed[0, 15] = np.nextafter(0.0, 1.0)
    assert tuple(learner.predict(first)[0]) == (0, 0, 0, 1, 0)
    assert tuple(learner.predict(changed)[0]) == (0.2,) * 5


def test_should_keep_core_transition_pure_and_own_input_arrays_snapshots_and_forks():
    policy = ActionMemoryPolicy()
    learner = ActionMemoryLearner(policy)
    x, y = features([0.0, 0.25]), labels([1, 2])
    initial = learner.snapshot_state()
    expected = count_action_observations(initial, x, y)
    assert initial.entries == ()
    learner.train_batch(x, y)
    x.fill(0.75)
    y.fill(4)
    object.__setattr__(policy, "max_entries", 1)
    assert learner.policy.max_entries == 64 and learner.snapshot_state() == expected
    child = learner.fork()
    child_state = child.snapshot_state()
    assert child is not learner and child_state == expected
    assert child_state is not expected and child_state.policy is not expected.policy
    assert child_state.entries is not expected.entries
    assert all(a is not b for a, b in zip(child_state.entries, expected.entries, strict=True))
    child.train_batch(features([0.5]), labels([4]))
    assert learner.snapshot_state() == expected
    object.__setattr__(child_state.entries[0], "counts", (8, 0, 0, 0, 0))
    object.__setattr__(expected.policy, "max_payload_bytes", 168)
    assert learner.policy.max_payload_bytes == 10752
    assert learner.snapshot_state().entries[0].counts == (0, 1, 0, 0, 0)


def test_should_restore_full_owned_state_and_keep_old_snapshot_unchanged():
    source = trained()
    snapshot = source.snapshot_state()
    target = ActionMemoryLearner()
    target.restore_state(snapshot)
    assert target.snapshot_state() == snapshot
    object.__setattr__(snapshot.entries[0], "features", (0.5,) * 16)
    object.__setattr__(snapshot.policy, "max_entries", 1)
    assert target.snapshot_state() == source.snapshot_state()
    target.train_batch(features([0.25]), labels([4]))
    assert target.training_examples == 4 and source.training_examples == 3


@pytest.mark.parametrize(
    "policy",
    [
        ActionMemoryPolicy(max_entries=1),
        ActionMemoryPolicy(max_payload_bytes=168),
        ActionMemoryPolicy(max_payload_bytes=167),
    ],
)
def test_should_refuse_whole_first_batch_if_either_capacity_is_exceeded(policy):
    learner = ActionMemoryLearner(policy)
    before = learner.snapshot_state()
    with pytest.raises(ValueError, match="capacity"):
        learner.train_batch(features([0.0, 0.25]), labels([0, 1]))
    assert learner.snapshot_state() == before


def test_should_not_commit_earlier_known_key_counts_or_evict_when_a_later_key_exceeds_capacity():
    learner = ActionMemoryLearner(ActionMemoryPolicy(max_entries=1))
    learner.train_batch(features([0.0]), labels([0]))
    before = learner.snapshot_state()
    with pytest.raises(ValueError, match="capacity"):
        learner.train_batch(features([0.0, 0.25]), labels([1, 4]))
    assert learner.snapshot_state() == before
    np.testing.assert_array_equal(
        learner.predict(features([0.0, 0.25])), [[1, 0, 0, 0, 0], [0.2] * 5]
    )


def test_should_refuse_sixty_fifth_default_key_atomically_and_admit_128_duplicate_rows():
    learner = ActionMemoryLearner()
    with pytest.raises(ValueError, match="capacity"):
        learner.train_batch(features([index / 128 for index in range(65)]), labels([0] * 65))
    assert learner.snapshot_state().entries == ()
    learner.train_batch(features([0.0] * 128), labels([4] * 128))
    assert learner.snapshot_state().entries[0].counts == (0, 0, 0, 0, 128)
    assert learner.completed_updates == 1 and learner.training_examples == 128


class ArraySubclass(np.ndarray):
    pass


@pytest.mark.parametrize(
    "fault",
    [
        "nonarray",
        "subclass",
        "float32",
        "bool",
        "width",
        "empty",
        "too_many",
        "nan",
        "range",
        "bad_later_label",
        "label_bool",
        "label_float32",
        "label_fraction",
        "label_range",
    ],
)
def test_should_preflight_all_rows_and_labels_before_any_commit(fault):
    learner = trained()
    before = learner.snapshot_state()
    x, y = features([0.0, 0.5]), labels([0, 1])
    if fault == "nonarray":
        x = x.tolist()
    elif fault == "subclass":
        x = x.view(ArraySubclass)
    elif fault == "float32":
        x = x.astype(np.float32)
    elif fault == "bool":
        x = x.astype(bool)
    elif fault == "width":
        x = np.zeros((2, 8))
    elif fault == "empty":
        x, y = np.zeros((0, 16)), np.zeros((0, 1))
    elif fault == "too_many":
        x, y = np.zeros((129, 16)), np.zeros((129, 1))
    elif fault == "nan":
        x[1, 15] = float("nan")
    elif fault == "range":
        x[1, 15] = -1.01
    elif fault == "bad_later_label":
        y[1, 0] = float("nan")
    elif fault == "label_bool":
        y = y.astype(bool)
    elif fault == "label_float32":
        y = y.astype(np.float32)
    elif fault == "label_fraction":
        y[1, 0] = 0.5
    else:
        y[1, 0] = 5
    with pytest.raises(ValueError):
        learner.train_batch(x, y)
    assert learner.snapshot_state() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("method", "bp"),
        ("feature_dim", True),
        ("feature_dim", 16.0),
        ("num_classes", np.int64(5)),
        ("max_entries", False),
        ("max_entries", 0),
        ("max_entries", 65),
        ("max_payload_bytes", 0),
        ("max_payload_bytes", 10753),
    ],
)
def test_should_refuse_invalid_policy_fields(field, value):
    with pytest.raises(ValueError):
        ActionMemoryPolicy(**{field: value})


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_type",
        "snapshot_extra",
        "entries_list",
        "entry_extra",
        "features_list",
        "counts_list",
        "duplicate",
        "nan",
        "bool_key",
        "numpy_key",
        "negative_count",
        "bool_count",
        "overflow_count",
        "zero_entry",
        "updates_bool",
        "examples_negative",
        "wrong_sum",
        "wrong_updates",
        "changed_policy",
        "policy_extra",
    ],
)
def test_should_reject_corrupt_restore_atomically_even_after_frozen_record_tampering(fault):
    learner = trained()
    before = learner.snapshot_state()
    corrupted = deepcopy(before)
    entry = corrupted.entries[0]
    if fault == "wrong_type":
        corrupted = object()
    elif fault == "snapshot_extra":
        object.__setattr__(corrupted, "extra", 1)
    elif fault == "entries_list":
        object.__setattr__(corrupted, "entries", list(corrupted.entries))
    elif fault == "entry_extra":
        object.__setattr__(entry, "extra", 1)
    elif fault == "features_list":
        object.__setattr__(entry, "features", list(entry.features))
    elif fault == "counts_list":
        object.__setattr__(entry, "counts", list(entry.counts))
    elif fault == "duplicate":
        object.__setattr__(corrupted, "entries", (entry, entry))
    elif fault in ("nan", "bool_key", "numpy_key"):
        value = {"nan": float("nan"), "bool_key": False, "numpy_key": np.float64(0.0)}[fault]
        object.__setattr__(entry, "features", (value,) + entry.features[1:])
    elif fault in ("negative_count", "bool_count", "overflow_count", "zero_entry"):
        counts = {
            "negative_count": (-1, 3, 0, 0, 0),
            "bool_count": (True, 1, 0, 0, 0),
            "overflow_count": (2**63, 0, 0, 0, 0),
            "zero_entry": (0, 0, 0, 0, 0),
        }[fault]
        object.__setattr__(entry, "counts", counts)
    elif fault == "updates_bool":
        object.__setattr__(corrupted, "completed_updates", True)
    elif fault == "examples_negative":
        object.__setattr__(corrupted, "training_examples", -1)
    elif fault == "wrong_sum":
        object.__setattr__(corrupted, "training_examples", 2)
    elif fault == "wrong_updates":
        object.__setattr__(corrupted, "completed_updates", 4)
    elif fault == "changed_policy":
        object.__setattr__(corrupted, "policy", replace(corrupted.policy, max_entries=2))
    else:
        object.__setattr__(corrupted.policy, "extra", 1)
    with pytest.raises(ValueError):
        learner.restore_state(corrupted)
    assert learner.snapshot_state() == before


def test_should_refuse_counter_overflow_without_wrapping_counts_or_changing_state():
    maximum = 2**53 - 1
    policy = ActionMemoryPolicy()
    snapshot = ActionMemorySnapshot(
        policy, (ActionMemoryEntry((0.0,) * 16, (maximum, 0, 0, 0, 0)),), maximum, maximum
    )
    learner = ActionMemoryLearner()
    learner.restore_state(snapshot)
    before = learner.snapshot_state()
    with pytest.raises(ValueError, match="overflow"):
        learner.train_batch(features([0.0]), labels([0]))
    assert learner.snapshot_state() == before
    np.testing.assert_array_equal(predict_action_memory(before, features([0.0])), [[1, 0, 0, 0, 0]])


def test_should_reject_int64_overflow_and_inconsistent_empty_history():
    with pytest.raises(ValueError, match="int64"):
        ActionMemoryEntry((0.0,) * 16, (2**63, 0, 0, 0, 0))
    for updates, examples in ((1, 0), (0, 1), (0, 2**53)):
        with pytest.raises(ValueError):
            copy_action_memory_snapshot(
                ActionMemorySnapshot(ActionMemoryPolicy(), (), updates, examples)
            )


def test_should_reject_policy_snapshot_and_entry_subclasses_without_coercion():
    class PolicySubclass(ActionMemoryPolicy):
        pass

    class SnapshotSubclass(ActionMemorySnapshot):
        pass

    class EntrySubclass(ActionMemoryEntry):
        pass

    with pytest.raises(ValueError, match="exact complete"):
        ActionMemoryLearner(PolicySubclass())
    learner = trained()
    before = learner.snapshot_state()
    invalid: Any = SnapshotSubclass(
        before.policy, before.entries, before.completed_updates, before.training_examples
    )
    with pytest.raises(ValueError, match="exact complete"):
        learner.restore_state(invalid)
    entry = EntrySubclass(before.entries[0].features, before.entries[0].counts)
    invalid = ActionMemorySnapshot(
        before.policy,
        (entry, *before.entries[1:]),
        before.completed_updates,
        before.training_examples,
    )
    with pytest.raises(ValueError, match="exact complete"):
        learner.restore_state(invalid)
    assert learner.snapshot_state() == before
