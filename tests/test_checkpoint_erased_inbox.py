"""M1: exactly 32 scalar cases; no arrays/native work/update sequences/cleanup."""

from dataclasses import replace
from typing import Any

import pytest

from src.app.checkpoint_erased_inbox import (
    bind_erased_inbox_records,
    require_erased_inbox_records,
    require_inbox_partition,
    require_same_erased_inbox_history,
)
from src.app.erased_inbox_origins import erased_inbox_origin_stamp, prepare_erased_inbox_origin
from src.core.checkpoint_content import CheckpointContentLimits
from src.core.data_erasure import ErasedExperience
from src.core.experience import AppliedExperience, Experience, ExperiencePermissions, LabelArrival
from src.core.inbox_cursor import InboxCursor
from src.core.inbox_origin import InboxOriginData
from src.core.learner_ports import TrainingDiagnostic


def _case():
    limits = CheckpointContentLimits(512, 16384, 64, 16)
    data = InboxOriginData(
        ("episode", "first"),
        "event1",
        "actor",
        "learner",
        "subject",
        "source",
        1,
        2,
        1,
        64,
        "a" * 64,
        "b" * 64,
    )
    first = AppliedExperience(
        "first",
        "episode",
        "event1",
        "actor",
        "learner",
        1,
        2,
        3,
        1,
        TrainingDiagnostic("native.loss", 0.25),
    )
    tombstone = ErasedExperience(("episode", "first"), "actor", 1, "event1", 2, 4, "deleted")
    erased = prepare_erased_inbox_origin(data, first, tombstone, limits)
    source = Experience(
        "second", "episode", 4, "actor", 1, "train", ExperiencePermissions(True, True, False)
    )
    label = LabelArrival("event2", "second", "episode", 5, "actor", 2)
    second = AppliedExperience(
        "second",
        "episode",
        "event2",
        "actor",
        "learner",
        4,
        5,
        6,
        2,
        TrainingDiagnostic("native.loss", 0.125),
    )
    cursor = InboxCursor(
        2, "learner", 4, (source,), (label,), (first, second), 8, False, 2, (tombstone,)
    )
    return cursor, (second,), (erased,), limits


def _memo(records):
    result: dict[int, Any] = {}
    for record in records:
        receipt, tombstone = record.receipt(), record.tombstone()
        assert receipt is not None and tombstone is not None
        result[id(receipt)] = replace(receipt, diagnostic=replace(receipt.diagnostic))
        result[id(tombstone)] = replace(tombstone)
    return result


def _two_erased(cursor, erased, limits):
    second = cursor.applied[1]
    data = replace(
        erased[0].data,
        key=("episode", "second"),
        event_id="event2",
        observed_at=4,
        arrived_at=5,
        update_number=2,
    )
    tombstone = ErasedExperience(("episode", "second"), "actor", 4, "event2", 5, 7, "deleted")
    record = prepare_erased_inbox_origin(data, second, tombstone, limits)
    return erased + (record,), tombstone


def test_should_accept_complete_disjoint_actual_mixed_partition():
    cursor, live, erased, limits = _case()
    require_inbox_partition(cursor, live, erased, limits, 4)


def test_should_accept_empty_erased_record_tuple():
    _, _, _, limits = _case()
    require_erased_inbox_records((), limits, 4)


def test_should_bind_actual_scalar_memo_objects_with_same_admitted_data():
    cursor, _, erased, limits = _case()
    memo = _memo(erased)
    copied = bind_erased_inbox_records(erased, lambda value: memo[id(value)], limits, 4)

    assert copied[0].data is erased[0].data
    assert copied[0].receipt() is memo[id(cursor.applied[0])]
    assert copied[0].tombstone() is memo[id(cursor.erased[0])]
    require_same_erased_inbox_history(erased[0], copied[0], limits)


@pytest.mark.parametrize("mode", ["list", "aggregate_metadata", "duplicate", "custom_maximum"])
def test_should_refuse_unbounded_or_duplicate_erased_record_collections(mode):
    cursor, _, erased, limits = _case()
    records: Any = erased
    maximum: Any = 4
    if mode == "list":
        records = list(erased)
    elif mode == "aggregate_metadata":
        records, tombstone = _two_erased(cursor, erased, limits)
        object.__setattr__(limits, "max_nodes", 64)
        # Each original scalar witness is within the bound; their aggregate is not.
        for record in records:
            erased_inbox_origin_stamp(record, limits)
        assert records[1].tombstone() is tombstone
    elif mode == "duplicate":
        records = erased * 2
    else:
        maximum = True
    assert cursor.applied  # Keep the original weak receipt/tombstone owners alive.
    with pytest.raises(ValueError):
        require_erased_inbox_records(records, limits, maximum)


@pytest.mark.parametrize(
    "mode",
    [
        "missing_applied",
        "extra_applied",
        "live_clone",
        "erased_receipt_clone",
        "tombstone_clone",
        "unknown_tombstone",
        "duplicate_live",
        "overlap",
        "cursor_collection",
        "cursor_scalar",
        "work",
    ],
)
def test_should_refuse_incomplete_or_foreign_partition_coverage(mode):
    cursor, live, erased, limits = _case()
    original_receipt, original_tombstone = cursor.applied[0], cursor.erased[0]
    if mode == "missing_applied":
        object.__setattr__(cursor, "applied", cursor.applied[:1])
        object.__setattr__(cursor, "completed_updates", 1)
    elif mode == "extra_applied":
        object.__setattr__(cursor, "applied", cursor.applied + (cursor.applied[-1],))
        object.__setattr__(cursor, "completed_updates", 3)
    elif mode == "live_clone":
        live = (replace(live[0]),)
    elif mode == "erased_receipt_clone":
        object.__setattr__(cursor, "applied", (replace(cursor.applied[0]), cursor.applied[1]))
    elif mode == "tombstone_clone":
        object.__setattr__(cursor, "erased", (replace(cursor.erased[0]),))
    elif mode == "unknown_tombstone":
        object.__setattr__(
            cursor, "erased", cursor.erased + (replace(cursor.erased[0], key=("episode", "other")),)
        )
    elif mode == "duplicate_live":
        live = live * 2
    elif mode == "overlap":
        live = live + (cursor.applied[0],)
    elif mode == "cursor_collection":
        object.__setattr__(cursor, "applied", list(cursor.applied))
    elif mode == "cursor_scalar":
        object.__setattr__(cursor, "last_tick", object())
    else:
        object.__setattr__(cursor, "completed_updates", 3)

    assert erased[0].receipt() is original_receipt
    assert erased[0].tombstone() is original_tombstone
    with pytest.raises(ValueError):
        require_inbox_partition(cursor, live, erased, limits, 4)


@pytest.mark.parametrize(
    "mode",
    [
        "missing_receipt",
        "missing_tombstone",
        "receipt_alias",
        "tombstone_alias",
        "diagnostic_alias",
        "changed_diagnostic",
        "changed_applied_at",
        "changed_tombstone_reason",
    ],
)
def test_should_refuse_missing_alias_or_changed_scalar_memo_objects(mode):
    cursor, _, erased, limits = _case()
    memo = _memo(erased)
    receipt, tombstone = cursor.applied[0], cursor.erased[0]
    if mode == "missing_receipt":
        del memo[id(receipt)]
    elif mode == "missing_tombstone":
        del memo[id(tombstone)]
    elif mode == "receipt_alias":
        memo[id(receipt)] = receipt
    elif mode == "tombstone_alias":
        memo[id(tombstone)] = tombstone
    elif mode == "diagnostic_alias":
        memo[id(receipt)] = replace(receipt)
    elif mode == "changed_diagnostic":
        memo[id(receipt)] = replace(receipt, diagnostic=TrainingDiagnostic("native.loss", 0.5))
    elif mode == "changed_applied_at":
        memo[id(receipt)] = replace(receipt, applied_at=2, diagnostic=replace(receipt.diagnostic))
    else:
        memo[id(tombstone)] = replace(tombstone, reason="expired")

    with pytest.raises(ValueError):
        bind_erased_inbox_records(erased, lambda value: memo[id(value)], limits, 4)


def test_should_reprove_original_content_after_all_opaque_memo_calls():
    cursor, _, erased, limits = _case()
    records, second_tombstone = _two_erased(cursor, erased, limits)
    memo = _memo(records)

    def lookup(value):
        if value is second_tombstone:
            object.__setattr__(cursor.applied[0].diagnostic, "value", 0.5)
        return memo[id(value)]

    with pytest.raises(ValueError):
        bind_erased_inbox_records(records, lookup, limits, 4)


def test_should_refuse_copied_metadata_changed_by_later_memo_call():
    cursor, _, erased, limits = _case()
    records, second_tombstone = _two_erased(cursor, erased, limits)
    memo = _memo(records)

    def lookup(value):
        if value is second_tombstone:
            changed: Any = memo[id(cursor.applied[0])]
            object.__setattr__(changed.diagnostic, "value", 0.5)
        return memo[id(value)]

    with pytest.raises(ValueError):
        bind_erased_inbox_records(records, lookup, limits, 4)


def test_should_require_same_original_admitted_data_identity():
    cursor, _, erased, limits = _case()
    memo = _memo(erased)
    copied = prepare_erased_inbox_origin(
        replace(erased[0].data), memo[id(cursor.applied[0])], memo[id(cursor.erased[0])], limits
    )

    with pytest.raises(ValueError):
        require_same_erased_inbox_history(erased[0], copied, limits)


def test_should_distinguish_equal_scalar_history_from_current_cursor_identity_authority():
    cursor, live, erased, limits = _case()
    memo = _memo(erased)
    copied = prepare_erased_inbox_origin(
        erased[0].data, memo[id(cursor.applied[0])], memo[id(cursor.erased[0])], limits
    )
    # Same-history validation grants no memo/original authority. An equal clone
    # still cannot cover the cursor's different actual receipt/tombstone objects.
    require_same_erased_inbox_history(erased[0], copied, limits)
    with pytest.raises(ValueError):
        require_inbox_partition(cursor, live, (copied,), limits, 4)


def test_should_refuse_callable_weak_reference_without_invoking_it():
    cursor, _, erased, limits = _case()
    calls: list[str] = []

    def opaque():
        calls.append("called")
        return cursor.applied[0]

    object.__setattr__(erased[0], "receipt", opaque)
    with pytest.raises(ValueError):
        require_erased_inbox_records(erased, limits, 4)
    assert calls == []


def test_should_refuse_expired_original_erased_reference():
    cursor, _, erased, limits = _case()
    object.__setattr__(cursor, "applied", cursor.applied[1:])

    with pytest.raises(ValueError):
        require_erased_inbox_records(erased, limits, 4)
