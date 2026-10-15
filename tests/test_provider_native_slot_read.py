"""Native metadata getter equivalence; no source compile or provider dispatch."""

import json
from types import GetSetDescriptorType, MemberDescriptorType
from typing import Any, cast

import pytest

import provider_deferred_construction as deferred
import provider_source_classes as classes
import provider_source_loader as loader


TYPE_DICT = cast(GetSetDescriptorType, type.__dict__["__dict__"])
MEMBER_GET = TYPE_DICT.__get__(MemberDescriptorType, type)["__get__"]
SCHEMAS: tuple[tuple[Any, type, tuple[str, ...], str], ...] = (
    (loader, loader.ApprovedSourceImport, ("name", "module"), "invalid source witness"),
    (
        loader,
        loader.ApprovedSourceDecorator,
        (
            "root",
            "module_name",
            "attribute",
            "function",
            "wrapper_type",
            "descriptor",
            "dependencies",
            "classes",
        ),
        "invalid source witness",
    ),
    (
        loader,
        loader.PreparedFixtureSource,
        (
            "module_name",
            "filename",
            "source_sha256",
            "module_code",
            "root_codes",
        ),
        "invalid source witness",
    ),
    (
        loader,
        loader.LoadedFixtureSource,
        ("prepared", "module", "namespace", "roots"),
        "invalid source witness",
    ),
    (
        classes,
        classes.SourceClassCode,
        ("slot", "class_code", "constructor_code", "call_code"),
        "invalid source class witness",
    ),
    (classes, classes.PreparedSourceClasses, ("source", "classes"), "invalid source class witness"),
    (
        classes,
        classes.SourceClassBinding,
        (
            "namespace",
            "code",
            "owner_type",
            "descriptor",
            "constructor",
            "raw_call",
        ),
        "invalid source class witness",
    ),
    (
        classes,
        classes.LoadedSourceClasses,
        ("prepared", "source", "classes"),
        "invalid source class witness",
    ),
    (
        classes,
        loader.PreparedFixtureSource,
        (
            "module_name",
            "filename",
            "source_sha256",
            "module_code",
            "root_codes",
        ),
        "invalid source class witness",
    ),
    (
        classes,
        loader.LoadedFixtureSource,
        ("prepared", "module", "namespace", "roots"),
        "invalid source class witness",
    ),
    (
        deferred,
        deferred.DeferredConstructionRole,
        (
            "source_classes",
            "prepared",
            "class_index",
            "caller_root",
            "caller_indexes",
            "max_instances",
            "callback_slot",
            "callback_argument",
            "callback_root",
            "callback_indexes",
        ),
        "invalid deferred construction witness",
    ),
    (
        deferred,
        classes.PreparedSourceClasses,
        ("source", "classes"),
        "invalid deferred construction witness",
    ),
    (
        deferred,
        classes.SourceClassCode,
        (
            "slot",
            "class_code",
            "constructor_code",
            "call_code",
        ),
        "invalid deferred construction witness",
    ),
    (
        deferred,
        loader.PreparedFixtureSource,
        (
            "module_name",
            "filename",
            "source_sha256",
            "module_code",
            "root_codes",
        ),
        "invalid deferred construction witness",
    ),
    (
        deferred,
        classes.LoadedSourceClasses,
        ("prepared", "source", "classes"),
        "invalid deferred construction witness",
    ),
    (
        deferred,
        loader.LoadedFixtureSource,
        ("prepared", "module", "namespace", "roots"),
        "invalid deferred construction witness",
    ),
    (
        deferred,
        classes.SourceClassBinding,
        (
            "namespace",
            "code",
            "owner_type",
            "descriptor",
            "constructor",
            "raw_call",
        ),
        "invalid deferred construction witness",
    ),
)


def native_descriptors(kind, names):
    namespace = TYPE_DICT.__get__(kind, type)
    return tuple(namespace[name] for name in names)


def reference_values(row, kind, names, message, descriptors=None):
    """Original generator semantics live only in this inactive owned fixture."""
    if type(row) is not kind:
        raise TypeError(message)
    readers = native_descriptors(kind, names) if descriptors is None else descriptors
    try:
        return tuple(MEMBER_GET(reader, row, kind) for reader in readers)
    except AttributeError:
        raise ValueError(message) from None


def reference_same(row, saved, kind, names, message):
    values = reference_values(row, kind, names, message)
    if any(value is not old for value, old in zip(values, saved)):
        raise ValueError(message)
    return values


def make_row(kind, names, events):
    class Sentinel:
        def __eq__(self, other):
            events.append("value_equality")
            raise AssertionError

    row = object.__new__(kind)
    values = tuple(object.__new__(Sentinel) for _ in names)
    for name, value in zip(names, values):
        object.__setattr__(row, name, value)
    return row, values


def compare_values(module, row, kind, names, message, expected, same=False, saved=None):
    if same:
        actual = module._same(row, saved, kind)
        original = reference_same(row, saved, kind, names, message)
    else:
        actual = module._values(row, kind)
        original = reference_values(row, kind, names, message)
    assert type(actual) is tuple and type(original) is tuple
    assert len(actual) == len(original) == len(expected)
    assert all(
        actual[index] is value and original[index] is value for index, value in enumerate(expected)
    )


def compare_errors(
    module, row, kind, names, message, error, same=False, saved=None, descriptors=None
):
    with pytest.raises(error) as actual:
        if same:
            module._same(row, saved, kind)
        else:
            module._values(row, kind)
    with pytest.raises(error) as original:
        if same:
            reference_same(row, saved, kind, names, message)
        else:
            reference_values(row, kind, names, message, descriptors)
    assert type(actual.value) is type(original.value) is error
    assert str(actual.value) == str(original.value)
    assert actual.value.__suppress_context__ is original.value.__suppress_context__


def save_record(tmp_path, name, fixtures, successes, type_errors, value_errors, events):
    assert len(SCHEMAS) == 17 and sum(len(names) for _, _, names, _ in SCHEMAS) == 77
    assert 0 <= fixtures <= 256 and events == []
    (tmp_path / "slot-work.json").write_text(
        json.dumps(
            dict(
                test_name=name,
                row_cases=17,
                fixtures=fixtures,
                comparisons=successes + type_errors + value_errors,
                successes=successes,
                errors=dict(TypeError=type_errors, ValueError=value_errors),
                provider_calls=0,
                factory_calls=0,
                constructor_calls=0,
                delegate_calls=0,
                source_compilations=0,
                events=events,
            )
        ),
        encoding="utf8",
    )


def test_slot_reads_preserve_original_native_identity_and_order(tmp_path):
    events: list[str] = []
    for module, kind, names, message in SCHEMAS:
        row, expected = make_row(kind, names, events)
        compare_values(module, row, kind, names, message, expected)
    save_record(
        tmp_path,
        "test_slot_reads_preserve_original_native_identity_and_order",
        17,
        17,
        0,
        0,
        events,
    )


def test_slot_reads_refuse_wrong_types_and_hostile_subclasses(tmp_path):
    events: list[str] = []

    def hostile(self, name):
        events.append("subclass_getter")
        raise AssertionError

    for module, kind, names, message in SCHEMAS:
        compare_errors(module, object(), kind, names, message, TypeError)
        foreign_kind = type("ForeignMetadata", (kind,), {"__getattribute__": hostile})
        foreign: Any = object.__new__(foreign_kind)
        compare_errors(module, foreign, kind, names, message, TypeError)
    save_record(
        tmp_path, "test_slot_reads_refuse_wrong_types_and_hostile_subclasses", 34, 0, 34, 0, events
    )


def test_slot_reads_map_unset_native_fields_to_original_errors(tmp_path):
    events: list[str] = []
    for module, kind, names, message in SCHEMAS:
        incomplete: Any = object.__new__(kind)
        compare_errors(module, incomplete, kind, names, message, ValueError)
    save_record(
        tmp_path, "test_slot_reads_map_unset_native_fields_to_original_errors", 17, 0, 0, 17, events
    )


def test_slot_reads_map_deleted_fields_and_accept_restoration(tmp_path):
    events: list[str] = []
    for module, kind, names, message in SCHEMAS:
        row, expected = make_row(kind, names, events)
        for name, value in zip(names, expected):
            object.__delattr__(row, name)
            try:
                compare_errors(module, row, kind, names, message, ValueError)
            finally:
                object.__setattr__(row, name, value)
            compare_values(module, row, kind, names, message, expected)
    save_record(
        tmp_path, "test_slot_reads_map_deleted_fields_and_accept_restoration", 17, 77, 0, 77, events
    )


def test_slot_reads_bypass_owned_native_attribute_hooks(tmp_path):
    events: list[str] = []
    missing = object()

    def hostile(self, name):
        events.append("native_attribute_hook")
        raise AssertionError

    for module, kind, names, message in SCHEMAS:
        row, expected = make_row(kind, names, events)
        original = TYPE_DICT.__get__(kind, type).get("__getattribute__", missing)
        setattr(kind, "__getattribute__", hostile)
        try:
            compare_values(module, row, kind, names, message, expected)
        finally:
            if original is missing:
                delattr(kind, "__getattribute__")
            else:
                setattr(kind, "__getattribute__", original)
    save_record(
        tmp_path, "test_slot_reads_bypass_owned_native_attribute_hooks", 17, 17, 0, 0, events
    )


def test_slot_reads_preserve_foreign_native_descriptor_refusal(tmp_path):
    events: list[str] = []
    for module, kind, names, message in SCHEMAS:
        row, expected = make_row(kind, names, events)
        foreign_kind: type[Any]
        if kind is loader.ApprovedSourceImport:
            foreign_kind, foreign_name = loader.PreparedFixtureSource, "module_name"
        else:
            foreign_kind, foreign_name = loader.ApprovedSourceImport, "name"
        foreign = TYPE_DICT.__get__(foreign_kind, type)[foreign_name]
        independent = native_descriptors(kind, names)
        malformed = (foreign, *independent[1:])
        original = module._FIELDS[kind]
        module._FIELDS[kind] = malformed
        try:
            compare_errors(module, row, kind, names, message, TypeError, descriptors=malformed)
        finally:
            module._FIELDS[kind] = original
        compare_values(module, row, kind, names, message, expected)
    save_record(
        tmp_path,
        "test_slot_reads_preserve_foreign_native_descriptor_refusal",
        17,
        17,
        17,
        0,
        events,
    )


def test_slot_snapshots_refuse_replaced_fields_and_accept_restoration(tmp_path):
    events: list[str] = []
    for module, kind, names, message in SCHEMAS:
        row, saved = make_row(kind, names, events)
        for name, original in zip(names, saved):
            replacement = object()
            assert replacement is not original
            object.__setattr__(row, name, replacement)
            try:
                compare_errors(
                    module, row, kind, names, message, ValueError, same=True, saved=saved
                )
            finally:
                object.__setattr__(row, name, original)
            compare_values(module, row, kind, names, message, saved, same=True, saved=saved)
    save_record(
        tmp_path,
        "test_slot_snapshots_refuse_replaced_fields_and_accept_restoration",
        17,
        77,
        0,
        77,
        events,
    )


def test_slot_snapshots_accept_identical_fields_in_native_row_clones(tmp_path):
    events: list[str] = []
    for module, kind, names, message in SCHEMAS:
        row, saved = make_row(kind, names, events)
        clone: Any = object.__new__(kind)
        assert clone is not row
        for name, value in zip(names, saved):
            object.__setattr__(clone, name, value)
        compare_values(module, clone, kind, names, message, saved)
        compare_values(module, clone, kind, names, message, saved, same=True, saved=saved)
        compare_values(module, row, kind, names, message, saved, same=True, saved=saved)
    save_record(
        tmp_path,
        "test_slot_snapshots_accept_identical_fields_in_native_row_clones",
        34,
        51,
        0,
        0,
        events,
    )
