"""U2: forty pure scalar memo cases; no payloads, receipts or authority."""

from dataclasses import replace
from typing import Any

import pytest

from src.app.checkpoint_untrained_inbox import (
    bind_untrained_inbox_records,
    require_same_untrained_inbox_history,
    require_untrained_inbox_records,
)
from src.app.untrained_inbox_origins import (
    prepare_untrained_inbox_origin,
    untrained_inbox_origin_stamp,
)
from src.core.checkpoint_content import CheckpointContentLimits
from src.core.data_erasure import ErasedExperience
from src.core.untrained_inbox_origin import UntrainedInboxOriginData


def _limits():
    return CheckpointContentLimits(512, 16384, 64, 16)


def _record(limits, form="pair", sample="first", event="event1"):
    observed = None if form == "label" else 1
    arrived = None if form == "source" else 2
    event_id = None if form == "source" else event
    data = UntrainedInboxOriginData(
        ("episode", sample),
        "actor",
        "learner",
        "person",
        "local",
        observed,
        event_id,
        arrived,
        4,
        "deleted",
        0,
    )
    tombstone = ErasedExperience(data.key, "actor", observed, event_id, arrived, 4, "deleted")
    return prepare_untrained_inbox_origin(data, tombstone, limits), tombstone


def _case(form="pair"):
    limits = _limits()
    record, tombstone = _record(limits, form)
    return (record,), (tombstone,), limits


def _two():
    limits = _limits()
    first, first_tombstone = _record(limits)
    second, second_tombstone = _record(limits, sample="second", event="event2")
    return (first, second), (first_tombstone, second_tombstone), limits


def _memo(owners):
    return {id(tombstone): replace(tombstone) for tombstone in owners}


@pytest.mark.parametrize("form", ["source", "label", "pair"])
def test_should_bind_actual_memo_tombstone_with_same_original_scalar_data(form):
    records, owners, limits = _case(form)
    memo = _memo(owners)
    copied = bind_untrained_inbox_records(records, lambda value: memo[id(value)], limits, 4)

    assert copied[0].data is records[0].data
    assert copied[0].data.completed_updates == 0
    assert copied[0].tombstone() is memo[id(owners[0])]
    assert copied[0].tombstone() is not owners[0]
    require_same_untrained_inbox_history(records[0], copied[0], limits)
    require_untrained_inbox_records(copied, limits, 4)


def test_should_accept_empty_collection_without_invoking_lookup():
    limits = _limits()
    calls: list[object] = []

    def lookup(value):
        calls.append(value)
        raise AssertionError("empty records need no memo access")

    require_untrained_inbox_records((), limits, 4)
    assert bind_untrained_inbox_records((), lookup, limits, 4) == ()
    assert calls == []


def test_should_preserve_mixed_form_order_and_exact_supplied_memo_objects():
    limits = _limits()
    pairs = tuple(
        _record(limits, form, f"sample{index}", f"event{index}")
        for index, form in enumerate(("label", "source", "pair"))
    )
    records = tuple(pair[0] for pair in pairs)
    owners = tuple(pair[1] for pair in pairs)
    memo = _memo(owners)
    seen: list[object] = []

    def lookup(value):
        seen.append(value)
        return memo[id(value)]

    copied = bind_untrained_inbox_records(records, lookup, limits, 4)
    assert all(value is owner for value, owner in zip(seen, owners))
    assert tuple(record.data.key for record in copied) == tuple(
        record.data.key for record in records
    )
    assert all(record.data is original.data for record, original in zip(copied, records))
    assert all(record.tombstone() is memo[id(owner)] for record, owner in zip(copied, owners))


def test_should_leave_copied_tombstone_ownership_with_caller():
    records, owners, limits = _case()
    memo = _memo(owners)
    copied = bind_untrained_inbox_records(records, lambda value: memo[id(value)], limits, 4)
    assert copied[0].tombstone() is memo[id(owners[0])]
    memo.clear()

    assert copied[0].tombstone() is None
    assert records[0].tombstone() is owners[0]
    with pytest.raises(ValueError, match="expired"):
        require_untrained_inbox_records(copied, limits, 4)


@pytest.mark.parametrize(
    "mode",
    [
        "list",
        "tuple_subclass",
        "maximum_bool",
        "record_type",
        "maximum_large",
        "count",
        "limits_type",
        "limits_scalar",
        "aggregate",
    ],
)
def test_should_refuse_unsupported_or_unbounded_collection_before_callbacks(mode):
    records, owners, limits = _case()
    maximum: Any = 4
    values: Any = records
    proof_limits: Any = limits
    if mode == "list":
        values = list(records)
    elif mode == "tuple_subclass":

        class TupleSubclass(tuple):
            pass

        values = TupleSubclass(records)
    elif mode == "maximum_bool":
        maximum = True
    elif mode == "record_type":
        values = (object(),)
    elif mode == "maximum_large":
        maximum = 4097
    elif mode == "count":
        maximum = 1
        values = records * 2
    elif mode == "limits_type":
        proof_limits = object()
    elif mode == "limits_scalar":
        object.__setattr__(limits, "max_nodes", True)
    else:
        pairs = tuple(
            _record(limits, sample=f"sample{index}", event=f"event{index}") for index in range(3)
        )
        values = tuple(pair[0] for pair in pairs)
        owners = tuple(pair[1] for pair in pairs)
        object.__setattr__(limits, "max_nodes", 64)
        # Each witness fits; the complete three-witness scalar graph does not.
        for record in values:
            untrained_inbox_origin_stamp(record, limits)
    calls: list[object] = []

    def lookup(value):
        calls.append(value)
        return value

    assert owners
    with pytest.raises(ValueError):
        bind_untrained_inbox_records(values, lookup, proof_limits, maximum)
    assert calls == []


@pytest.mark.parametrize(
    "mode",
    [
        "data_schema",
        "tombstone_schema",
        "callable_ref",
        "seal_type",
        "seal_content",
        "data_content",
        "tombstone_content",
        "expired",
    ],
)
def test_should_refuse_corrupt_or_expired_original_witness_without_opaque_reference_call(mode):
    records, owners, limits = _case()
    record = records[0]
    calls: list[str] = []
    if mode == "data_schema":
        vars(record.data)["unexpected"] = object()
    elif mode == "tombstone_schema":
        vars(owners[0])["unexpected"] = object()
    elif mode == "callable_ref":

        def opaque_reference():
            calls.append("called")
            return owners[0]

        object.__setattr__(record, "tombstone", opaque_reference)
    elif mode == "seal_type":
        object.__setattr__(record, "tombstone_stamp", object())
    elif mode == "seal_content":
        object.__setattr__(record, "tombstone_stamp", "0" * 64)
    elif mode == "data_content":
        object.__setattr__(record.data, "subject_id", "other")
    elif mode == "tombstone_content":
        object.__setattr__(owners[0], "erased_at", 5)
    else:
        del owners

    with pytest.raises(ValueError):
        require_untrained_inbox_records(records, limits, 4)
    assert calls == []


@pytest.mark.parametrize("duplicate", ["key", "event", "tombstone"])
def test_should_refuse_duplicate_original_inventory(duplicate):
    records, owners, limits = _case()
    if duplicate == "key":
        second_tombstone = replace(owners[0])
        second = prepare_untrained_inbox_origin(replace(records[0].data), second_tombstone, limits)
    elif duplicate == "event":
        second, second_tombstone = _record(limits, sample="second", event="event1")
    else:
        second_tombstone = owners[0]
        second = prepare_untrained_inbox_origin(records[0].data, second_tombstone, limits)
    assert second.tombstone() is second_tombstone

    with pytest.raises(ValueError, match="duplicate"):
        require_untrained_inbox_records(records + (second,), limits, 4)


@pytest.mark.parametrize(
    "mode",
    [
        "missing",
        "wrong_type",
        "alias",
        "wrong_key",
        "changed_reason",
        "changed_erased_at",
    ],
)
def test_should_refuse_missing_alias_or_changed_memo_tombstone(mode):
    records, owners, limits = _case()
    memo: dict[int, Any] = _memo(owners)
    key = id(owners[0])
    if mode == "missing":
        del memo[key]
    elif mode == "wrong_type":
        memo[key] = object()
    elif mode == "alias":
        memo[key] = owners[0]
    elif mode == "wrong_key":
        memo[key] = replace(owners[0], key=("episode", "other"))
    elif mode == "changed_reason":
        memo[key] = replace(owners[0], reason="expired")
    else:
        memo[key] = replace(owners[0], erased_at=5)

    with pytest.raises(ValueError):
        bind_untrained_inbox_records(records, lambda value: memo[id(value)], limits, 4)


def test_should_refuse_equal_scalar_data_clone_as_copied_history():
    records, owners, limits = _case()
    copied_tombstone = replace(owners[0])
    copied = prepare_untrained_inbox_origin(replace(records[0].data), copied_tombstone, limits)
    assert copied.data is not records[0].data

    with pytest.raises(ValueError):
        require_same_untrained_inbox_history(records[0], copied, limits)


@pytest.mark.parametrize(
    "mutation",
    [
        "original",
        "previous_result",
        "later_reference",
        "max_nodes",
        "max_metadata_bytes",
        "max_array_bytes",
        "max_depth",
    ],
)
def test_should_reprove_originals_results_and_all_limits_after_opaque_lookup(mutation):
    records, owners, limits = _two()
    memo = _memo(owners)
    seen: list[object] = []
    reference_calls: list[str] = []

    def opaque_reference():
        reference_calls.append("called")
        return owners[1]

    def lookup(value):
        seen.append(value)
        if mutation == "original" and value is owners[1]:
            object.__setattr__(owners[0], "erased_at", 5)
        elif mutation == "previous_result" and value is owners[1]:
            object.__setattr__(memo[id(owners[0])], "erased_at", 5)
        elif mutation == "later_reference" and value is owners[0]:
            object.__setattr__(records[1], "tombstone", opaque_reference)
        elif mutation.startswith("max_") and value is owners[0]:
            object.__setattr__(limits, mutation, getattr(limits, mutation) + 1)
        return memo[id(value)]

    with pytest.raises(ValueError):
        bind_untrained_inbox_records(records, lookup, limits, 4)
    assert reference_calls == []
    if mutation.startswith("max_") or mutation == "later_reference":
        assert len(seen) == 1
    else:
        assert len(seen) == 2
