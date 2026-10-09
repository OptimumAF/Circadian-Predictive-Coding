"""E1: exactly 32 scalar cases; no arrays, native work, cleanup or payload copies."""

from dataclasses import fields, replace
from typing import Any
from weakref import ref

import pytest

from src.app.erased_inbox_origins import (
    ErasedInboxOrigin,
    erased_inbox_origin_stamp,
    prepare_erased_inbox_origin,
)
from src.core.checkpoint_content import CheckpointContentLimits
from src.core.data_erasure import ErasedExperience
from src.core.experience import AppliedExperience
from src.core.inbox_origin import InboxOriginData
from src.core.learner_ports import TrainingDiagnostic


def _case():
    data = InboxOriginData(
        ("episode", "sample"),
        "event",
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
    receipt = AppliedExperience(
        "sample",
        "episode",
        "event",
        "actor",
        "learner",
        1,
        2,
        3,
        1,
        TrainingDiagnostic("native.loss", 0.25),
    )
    tombstone = ErasedExperience(("episode", "sample"), "actor", 1, "event", 2, 4, "deleted")
    return data, receipt, tombstone, CheckpointContentLimits(256, 8192, 64, 16)


def test_should_preserve_exact_weak_identities_and_stable_proof():
    data, receipt, tombstone, limits = _case()
    record = prepare_erased_inbox_origin(data, receipt, tombstone, limits)

    assert record.data is data
    assert record.receipt() is receipt
    assert record.tombstone() is tombstone
    assert erased_inbox_origin_stamp(record, limits) == erased_inbox_origin_stamp(record, limits)


@pytest.mark.parametrize(
    "name,value",
    [
        ("sample_id", "other"),
        ("episode_id", "other"),
        ("event_id", "other"),
        ("actor_version", "other"),
        ("learner_version", "other"),
        ("observed_at", 0),
        ("arrived_at", 3),
        ("update_number", 2),
        ("applied_at", 5),
    ],
)
def test_should_refuse_receipt_metadata_or_chronology_mismatch(name, value):
    data, receipt, tombstone, limits = _case()
    changed = replace(receipt, **{name: value})

    with pytest.raises(ValueError):
        prepare_erased_inbox_origin(data, changed, tombstone, limits)


@pytest.mark.parametrize(
    "name,value",
    [
        ("key", ("episode", "other")),
        ("actor_version", "other"),
        ("observed_at", 0),
        ("event_id", "other"),
        ("arrived_at", 3),
        ("erased_at", 2),
    ],
)
def test_should_refuse_tombstone_metadata_or_chronology_mismatch(name, value):
    data, receipt, tombstone, limits = _case()
    # Corrupt an originally valid record so its constructor is not the gate.
    object.__setattr__(tombstone, name, value)

    with pytest.raises(ValueError):
        prepare_erased_inbox_origin(data, receipt, tombstone, limits)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), (1 << 1024) - 1])
def test_should_refuse_nonfinite_or_unrepresentable_diagnostic(value):
    data, receipt, tombstone, limits = _case()
    object.__setattr__(receipt.diagnostic, "value", value)

    with pytest.raises(ValueError):
        prepare_erased_inbox_origin(data, receipt, tombstone, limits)


@pytest.mark.parametrize("target", ["data", "diagnostic"])
def test_should_refuse_extra_metadata_fields_before_traversal(target):
    data, receipt, tombstone, limits = _case()
    values = vars(data if target == "data" else receipt.diagnostic)
    values["unexpected"] = object()

    with pytest.raises(ValueError):
        prepare_erased_inbox_origin(data, receipt, tombstone, limits)


def test_should_refuse_expired_original_receipt():
    data, receipt, tombstone, limits = _case()
    record = prepare_erased_inbox_origin(data, receipt, tombstone, limits)
    del receipt

    with pytest.raises(ValueError, match="expired"):
        erased_inbox_origin_stamp(record, limits)


def test_should_refuse_expired_original_tombstone():
    data, receipt, tombstone, limits = _case()
    record = prepare_erased_inbox_origin(data, receipt, tombstone, limits)
    del tombstone

    with pytest.raises(ValueError, match="expired"):
        erased_inbox_origin_stamp(record, limits)


def test_should_refuse_callable_weak_reference_replacement_without_calling_it():
    data, receipt, tombstone, limits = _case()
    record = prepare_erased_inbox_origin(data, receipt, tombstone, limits)
    calls: list[str] = []

    def opaque_reference():
        calls.append("called")
        return receipt

    object.__setattr__(record, "receipt", opaque_reference)
    with pytest.raises(ValueError, match="weak references"):
        erased_inbox_origin_stamp(record, limits)
    assert calls == []


def test_should_refuse_equal_receipt_clone_substitution():
    data, receipt, tombstone, limits = _case()
    record = prepare_erased_inbox_origin(data, receipt, tombstone, limits)
    clone = replace(receipt)
    object.__setattr__(record, "receipt", ref(clone))

    with pytest.raises(ValueError, match="seal changed"):
        erased_inbox_origin_stamp(record, limits)


def test_should_refuse_equal_tombstone_clone_substitution():
    data, receipt, tombstone, limits = _case()
    record = prepare_erased_inbox_origin(data, receipt, tombstone, limits)
    clone = replace(tombstone)
    object.__setattr__(record, "tombstone", ref(clone))

    with pytest.raises(ValueError, match="seal changed"):
        erased_inbox_origin_stamp(record, limits)


def test_should_refuse_equal_admitted_metadata_clone_substitution():
    data, receipt, tombstone, limits = _case()
    record = prepare_erased_inbox_origin(data, receipt, tombstone, limits)
    object.__setattr__(record, "data", replace(data))

    with pytest.raises(ValueError, match="seal changed"):
        erased_inbox_origin_stamp(record, limits)


def test_should_refuse_original_numeric_digest_metadata_mutation():
    data, receipt, tombstone, limits = _case()
    record = prepare_erased_inbox_origin(data, receipt, tombstone, limits)
    object.__setattr__(data, "features_digest", "c" * 64)

    with pytest.raises(ValueError, match="seal changed"):
        erased_inbox_origin_stamp(record, limits)


def test_should_refuse_corrupt_seal_schema_before_hashing():
    data, receipt, tombstone, limits = _case()
    record = prepare_erased_inbox_origin(data, receipt, tombstone, limits)
    object.__setattr__(record, "tombstone_stamp", object())

    with pytest.raises(ValueError, match="SHA256 seals"):
        erased_inbox_origin_stamp(record, limits)


def test_should_have_only_payload_free_slotted_witness_fields():
    data, receipt, tombstone, limits = _case()
    record = prepare_erased_inbox_origin(data, receipt, tombstone, limits)

    assert tuple(field.name for field in fields(ErasedInboxOrigin)) == (
        "data",
        "receipt",
        "tombstone",
        "receipt_stamp",
        "tombstone_stamp",
    )
    assert not hasattr(record, "__dict__")
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(record, "source", receipt)


def test_should_refuse_custom_metadata_types_without_equality_callback():
    data, receipt, tombstone, limits = _case()
    calls: list[str] = []

    class CallbackMetaclass(type):
        def __eq__(self, other: Any) -> bool:
            calls.append("equality")
            return True

        def __hash__(self) -> int:
            calls.append("hash")
            return 0

    class CallbackValue(metaclass=CallbackMetaclass):
        pass

    object.__setattr__(receipt.diagnostic, "value", CallbackValue())
    with pytest.raises(ValueError, match="exact finite scalar"):
        prepare_erased_inbox_origin(data, receipt, tombstone, limits)
    assert calls == []
