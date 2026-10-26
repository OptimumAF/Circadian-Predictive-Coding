"""Full validation and ownership when FIFO entries are copied once.

Pure synthetic snapshots exercise the serialization boundary without learning,
prediction, random initialization, environment or research artifact access.
"""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.core.action_memory import ActionMemoryEntry
from src.core.fifo_action_memory import (
    FifoActionMemoryPolicy,
    FifoActionMemorySnapshot,
    copy_fifo_action_memory_snapshot,
)


class TupleSubclass(tuple):
    pass


class EntrySubclass(ActionMemoryEntry):
    pass


def source_snapshot(width, classes):
    policy = FifoActionMemoryPolicy(feature_dim=width, num_classes=classes, max_entries=2)
    entries = (
        ActionMemoryEntry((0.0,) * width, (1,) + (0,) * (classes - 1)),
        ActionMemoryEntry((0.5,) * width, (0, 1) + (0,) * (classes - 2)),
    )
    return FifoActionMemorySnapshot(policy, entries, 1, 2, 0, 0)


@pytest.mark.parametrize("geometry", [(16, 5), (296, 7)])
def test_should_detach_every_record_and_tuple_and_revalidate_each_call(geometry):
    source = source_snapshot(*geometry)
    copied = copy_fifo_action_memory_snapshot(source)
    assert copied == source and copied is not source and copied.policy is not source.policy
    assert copied.entries is not source.entries
    for actual, original in zip(copied.entries, source.entries, strict=True):
        assert actual is not original
        assert actual.features is not original.features and actual.counts is not original.counts
    object.__setattr__(source.entries[-1], "counts", (0,) * geometry[1])
    with pytest.raises(ValueError):
        copy_fifo_action_memory_snapshot(source)
    assert copied == source_snapshot(*geometry)


@pytest.mark.parametrize("geometry", [(16, 5), (296, 7)])
@pytest.mark.parametrize(
    "fault",
    [
        "feature_list",
        "count_list",
        "feature_tuple_subclass",
        "count_tuple_subclass",
        "feature_nan",
        "feature_bool",
        "feature_numpy",
        "count_numpy",
        "zero_counts",
        "feature_none",
        "count_none",
        "short_features",
        "short_counts",
        "entry_subclass",
        "extra_field",
    ],
)
def test_should_refuse_late_corrupt_entry_without_coercing_or_mutating_source(geometry, fault):
    source = source_snapshot(*geometry)
    original_first = source.entries[0]
    bad: Any = source.entries[-1]
    value: Any = None
    field = "features" if fault.startswith(("feature", "short_features")) else "counts"
    if fault == "feature_list":
        value = list(bad.features)
    elif fault == "count_list":
        value = list(bad.counts)
    elif fault == "feature_tuple_subclass":
        value = TupleSubclass(bad.features)
    elif fault == "count_tuple_subclass":
        value = TupleSubclass(bad.counts)
    elif fault == "feature_nan":
        value = bad.features[:-1] + (float("nan"),)
    elif fault == "feature_bool":
        value = bad.features[:-1] + (True,)
    elif fault == "feature_numpy":
        value = bad.features[:-1] + (np.float64(0.5),)
    elif fault == "count_numpy":
        value = (np.int64(0),) + bad.counts[1:]
    elif fault == "zero_counts":
        value = (0,) * geometry[1]
    elif fault in ("feature_none", "count_none"):
        value = None
    elif fault == "short_features":
        value = bad.features[:-1]
    elif fault == "short_counts":
        value = bad.counts[:-1]
    elif fault == "entry_subclass":
        source = replace(source, entries=(original_first, EntrySubclass(bad.features, bad.counts)))
    else:
        object.__setattr__(bad, "extra", 1)
    if fault not in ("entry_subclass", "extra_field"):
        object.__setattr__(bad, field, value)
    with pytest.raises(ValueError):
        copy_fifo_action_memory_snapshot(source)
    assert source.entries[0] is original_first
    assert original_first.features == (0.0,) * geometry[0]
    assert original_first.counts == (1,) + (0,) * (geometry[1] - 1)


@pytest.mark.parametrize("geometry", [(16, 5), (296, 7)])
def test_should_reject_tuple_subclasses_before_iterating_them(geometry):
    class HostileTuple(tuple):
        def __iter__(self):
            pytest.fail("a rejected container must not be coerced or iterated")

    for field in ("features", "counts"):
        source = source_snapshot(*geometry)
        entry = source.entries[-1]
        object.__setattr__(entry, field, HostileTuple(getattr(entry, field)))
        with pytest.raises(ValueError):
            copy_fifo_action_memory_snapshot(source)
