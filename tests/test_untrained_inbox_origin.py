"""U1: exactly 32 scalar cases, with no arrays, native work or cleanup."""

from dataclasses import fields, replace
from typing import Any
from weakref import ref

import pytest

from src.app.untrained_inbox_origins import (
    UntrainedInboxOrigin,
    prepare_untrained_inbox_origin,
    untrained_inbox_origin_stamp,
)
from src.core.checkpoint_content import CheckpointContentLimits
from src.core.data_erasure import ErasedExperience
from src.core.replay_origin import ReplayOriginAdmission, ReplayOriginLimits
from src.core.untrained_inbox_origin import (
    UntrainedInboxOriginData,
    untrained_inbox_origin_metadata,
)


def _case(form="pair"):
    observed = None if form == "label" else 1
    event = None if form == "source" else "event"
    arrived = None if form == "source" else 2
    data = UntrainedInboxOriginData(
        ("episode", "sample"),
        "actor",
        "learner",
        "subject",
        "source",
        observed,
        event,
        arrived,
        4,
        "deleted",
        0,
    )
    tombstone = ErasedExperience(data.key, "actor", observed, event, arrived, 4, "deleted")
    return data, tombstone, CheckpointContentLimits(256, 8192, 64, 16)


@pytest.mark.parametrize("form", ["source", "label", "pair"])
def test_should_preserve_each_original_arrival_form_without_a_receipt(form):
    data, tombstone, limits = _case(form)
    record = prepare_untrained_inbox_origin(data, tombstone, limits)

    assert record.data is data
    assert record.tombstone() is tombstone
    assert data.completed_updates == 0
    assert untrained_inbox_origin_stamp(record, limits) == untrained_inbox_origin_stamp(
        record, limits
    )
    assert "receipt" not in tuple(field.name for field in fields(UntrainedInboxOrigin))


@pytest.mark.parametrize(
    "changes",
    [
        {"observed_at": None, "event_id": None, "arrived_at": None},
        {"event_id": None},
        {"arrived_at": None},
        {"observed_at": 3},
        {"observed_at": 5},
        {"arrived_at": 5},
        {"reason": "unknown"},
        {"event_id": ""},
    ],
)
def test_should_refuse_optional_arrival_or_chronology_corruption(changes):
    data, tombstone, limits = _case()
    for name, value in changes.items():
        object.__setattr__(data, name, value)

    with pytest.raises(ValueError):
        prepare_untrained_inbox_origin(data, tombstone, limits)


@pytest.mark.parametrize(
    "name,value",
    [
        ("erased_at", -1),
        ("erased_at", 2**63),
        ("erased_at", True),
        ("completed_updates", float("nan")),
        ("completed_updates", float("inf")),
        ("completed_updates", 2**63),
    ],
)
def test_should_refuse_nonexact_nonfinite_or_unbounded_counters(name, value):
    data, tombstone, limits = _case()
    object.__setattr__(data, name, value)

    with pytest.raises(ValueError):
        prepare_untrained_inbox_origin(data, tombstone, limits)


@pytest.mark.parametrize("target", ["data", "tombstone", "aggregate_text"])
def test_should_bound_schema_and_escaped_text_before_encoding(target):
    data, tombstone, limits = _case()
    if target == "aggregate_text":
        # Each string is below the cap; their escaped aggregate exceeds it.
        object.__setattr__(data, "subject_id", "s" * 200)
        object.__setattr__(data, "source_id", "s" * 200)
        with pytest.raises(ValueError, match="metadata capacity"):
            untrained_inbox_origin_metadata(data, 2000)
    else:
        vars(data if target == "data" else tombstone)["unexpected"] = object()
        with pytest.raises(ValueError, match="field count"):
            prepare_untrained_inbox_origin(data, tombstone, limits)


def test_should_refuse_expired_original_tombstone():
    data, tombstone, limits = _case()
    record = prepare_untrained_inbox_origin(data, tombstone, limits)
    del tombstone

    with pytest.raises(ValueError, match="expired"):
        untrained_inbox_origin_stamp(record, limits)


def test_should_refuse_equal_tombstone_clone_substitution():
    data, tombstone, limits = _case()
    record = prepare_untrained_inbox_origin(data, tombstone, limits)
    clone = replace(tombstone)
    object.__setattr__(record, "tombstone", ref(clone))

    with pytest.raises(ValueError, match="seal changed"):
        untrained_inbox_origin_stamp(record, limits)


def test_should_refuse_equal_original_data_clone_substitution():
    data, tombstone, limits = _case()
    record = prepare_untrained_inbox_origin(data, tombstone, limits)
    object.__setattr__(record, "data", replace(data))

    with pytest.raises(ValueError, match="seal changed"):
        untrained_inbox_origin_stamp(record, limits)


def test_should_refuse_callable_reference_without_invoking_it():
    data, tombstone, limits = _case()
    record = prepare_untrained_inbox_origin(data, tombstone, limits)
    calls: list[str] = []

    def opaque_reference():
        calls.append("called")
        return tombstone

    object.__setattr__(record, "tombstone", opaque_reference)
    with pytest.raises(ValueError, match="exact weak reference"):
        untrained_inbox_origin_stamp(record, limits)
    assert calls == []


def test_should_refuse_original_tombstone_content_mutation():
    data, tombstone, limits = _case()
    record = prepare_untrained_inbox_origin(data, tombstone, limits)
    object.__setattr__(tombstone, "erased_at", 5)

    with pytest.raises(ValueError, match="scalar metadata"):
        untrained_inbox_origin_stamp(record, limits)


def test_should_refuse_seal_schema_corruption_before_hashing():
    data, tombstone, limits = _case()
    record = prepare_untrained_inbox_origin(data, tombstone, limits)
    object.__setattr__(record, "tombstone_stamp", object())

    with pytest.raises(ValueError, match="SHA256 seal"):
        untrained_inbox_origin_stamp(record, limits)


def test_should_refuse_custom_counter_without_equality_or_hash_callbacks():
    data, tombstone, limits = _case()
    calls: list[str] = []

    class CallbackMetaclass(type):
        def __eq__(self, other: Any) -> bool:
            calls.append("equality")
            return True

        def __hash__(self) -> int:
            calls.append("hash")
            return 0

    class CallbackCounter(metaclass=CallbackMetaclass):
        pass

    object.__setattr__(data, "completed_updates", CallbackCounter())
    with pytest.raises(ValueError, match="exact integers"):
        prepare_untrained_inbox_origin(data, tombstone, limits)
    assert calls == []


def test_should_charge_same_original_admission_without_work_or_payload_authority():
    data, tombstone, content_limits = _case()
    limits = ReplayOriginLimits(2, 2, 2, 8192, 120, 64)
    admission = ReplayOriginAdmission(limits)
    encoded = untrained_inbox_origin_metadata(data, 8192)
    admission.reserve_untrained(data, 0)
    record = prepare_untrained_inbox_origin(data, tombstone, content_limits)
    progress = admission.accounting(1)

    assert admission.limits is limits
    assert progress.records_created == 1
    assert progress.invocations_started == 0
    assert progress.metadata_bytes_charged == len(encoded) + 1024
    assert progress.live_records == 1
    assert record.data.completed_updates == 0


def test_should_refuse_second_record_without_replacing_or_refunding_original_authority():
    data, _, _ = _case()
    limits = ReplayOriginLimits(2, 1, 2, 8192, 120, 64)
    admission = ReplayOriginAdmission(limits)
    admission.reserve_untrained(data, 0)
    spent = admission.accounting(1)

    with pytest.raises(ValueError):
        admission.reserve_untrained(data, 1)
    assert admission.limits is limits
    assert admission.accounting(1) == spent


def test_should_refuse_original_metadata_exhaustion_before_witness_allocation():
    data, _, _ = _case()
    limits = ReplayOriginLimits(2, 2, 2, 1024, 120, 64)
    admission = ReplayOriginAdmission(limits)

    with pytest.raises(ValueError):
        admission.reserve_untrained(data, 0)
    assert admission.limits is limits
    assert admission.accounting().records_created == 0
    assert admission.accounting().metadata_bytes_charged == 0


def test_should_retain_only_three_payload_free_slotted_fields():
    data, tombstone, limits = _case()
    record = prepare_untrained_inbox_origin(data, tombstone, limits)

    assert tuple(field.name for field in fields(UntrainedInboxOrigin)) == (
        "data",
        "tombstone",
        "tombstone_stamp",
    )
    assert not hasattr(record, "__dict__")
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(record, "receipt", object())


def test_should_bound_aggregate_scalar_proof_before_sealing():
    data, tombstone, limits = _case()
    bounded = replace(limits, max_nodes=16)

    with pytest.raises(ValueError, match="node"):
        prepare_untrained_inbox_origin(data, tombstone, bounded)
