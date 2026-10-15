"""Membership-preserving batch controls; owned corruption fixtures, no providers."""

from dataclasses import replace
from hashlib import sha256
import json
from types import FunctionType, GetSetDescriptorType, ModuleType
from typing import cast

import pytest

from provider_source_classes import FixtureSourceClasses, LoadedSourceClasses, SourceClassBinding
from provider_source_loader import ApprovedSourceImport, FixtureSourceLoader


SOURCE = b"""import batcheffects
batcheffects.counts['module'] += 1
class Clock:
    def __init__(self):
        batcheffects.counts['constructor'] += 1
    def __call__(self):
        batcheffects.counts['provider'] += 1
        return 7
batcheffects.counts['class'] += 1
class Port:
    def __init__(self):
        batcheffects.counts['constructor'] += 1
    def __call__(self):
        batcheffects.counts['provider'] += 1
        return 7
batcheffects.counts['class'] += 1
def factory():
    batcheffects.counts['factory'] += 1
    return Clock()
"""
LABELS = ("module", "class", "factory", "constructor", "provider", "delegate")
MODULE_DICT = cast(
    GetSetDescriptorType, type.__dict__["__dict__"].__get__(ModuleType, type)["__dict__"]
)


def make_case(loads=1):
    dependency = ModuleType("batcheffects")
    effects = dict.fromkeys(LABELS, 0)
    MODULE_DICT.__get__(dependency, ModuleType)["counts"] = effects
    helper = FixtureSourceClasses(
        FixtureSourceLoader((ApprovedSourceImport("batcheffects", dependency),))
    )
    rows = []
    for index in range(loads):
        prepared = helper.prepare(
            SOURCE,
            sha256(SOURCE).hexdigest(),
            f"batch{index}",
            "batch.py",
            ("factory",),
            class_roots=("Clock", "Port"),
        )
        rows.append(helper.execute(prepared))
    return dict(helper=helper, rows=rows, effects=effects, loads=loads, events=[])


def observe_current(case):
    helper = case["helper"]
    original = helper._current
    calls = []

    def observed(issue):
        calls.append(next(index for index, owned in enumerate(helper._issues) if owned is issue))
        original(issue)

    # Owned diagnostic observer delegates the retained original proof; no profile
    # is installed and no implementation cache or guard state is patched.
    object.__setattr__(helper, "_current", observed)
    return calls, original


def save_unit(tmp_path, cases, calls, expected, test_name):
    assert calls == expected
    effects = {label: sum(case["effects"][label] for case in cases) for label in LABELS}
    loads = sum(case["loads"] for case in cases)
    assert effects == dict(
        module=loads, **{"class": 2 * loads}, factory=0, constructor=0, provider=0, delegate=0
    )
    assert all(case["events"] == [] for case in cases)
    (tmp_path / "batch-work.json").write_text(
        json.dumps(
            dict(
                test_name=test_name,
                loads=loads,
                module_bodies=loads,
                class_bodies=2 * loads,
                factory_calls=0,
                constructor_calls=0,
                provider_calls=0,
                delegate_calls=0,
                current_calls=calls,
                effects=effects,
                events=[],
            )
        ),
        encoding="utf8",
    )


def refuse_batch(helper, loaded):
    with pytest.raises((TypeError, ValueError)):
        helper.require_current_bindings(loaded)


def test_batch_reproves_same_issue_once_per_invocation(tmp_path):
    case = make_case()
    helper, loaded = case["helper"], case["rows"][0]
    assert len(loaded.classes) == 2 and loaded.classes[0] is not loaded.classes[1]
    calls, original = observe_current(case)
    try:
        helper.require_current_bindings(loaded)
        assert calls == [0]
        helper.require_current_bindings(loaded)
    finally:
        object.__setattr__(helper, "_current", original)
    save_unit(tmp_path, [case], calls, [0, 0], "test_batch_reproves_same_issue_once_per_invocation")


def test_batch_rejects_capture_clone_membership_and_accepts_restoration(tmp_path):
    case = make_case()
    helper, loaded = case["helper"], case["rows"][0]
    issue = helper._issues[0]
    captures = issue.captures
    calls, original = observe_current(case)
    try:
        clone = replace(captures[0].binding)
        assert clone is not loaded.classes[0]
        issue.captures = (replace(captures[0], binding=clone), captures[1])
        # Whole-source continuity accepts identical fields. Original binding
        # membership remains a separate observable refusal.
        helper.require_current(loaded)
        with pytest.raises(TypeError, match="invalid source class witness"):
            helper.require_binding(loaded.classes[0])
        with pytest.raises(TypeError, match="invalid source class witness"):
            helper.require_current_bindings(loaded)
        issue.captures = captures
        helper.require_current_bindings(loaded)
    finally:
        issue.captures = captures
        object.__setattr__(helper, "_current", original)
    save_unit(
        tmp_path,
        [case],
        calls,
        [0, 0, 0],
        "test_batch_rejects_capture_clone_membership_and_accepts_restoration",
    )


def test_batch_refuses_foreign_and_malformed_loaded_rows(tmp_path):
    case, foreign = make_case(), make_case()
    helper, loaded = case["helper"], case["rows"][0]

    class ForeignLoaded(LoadedSourceClasses):
        def __getattribute__(self, name):
            case["events"].append("loaded_getter")
            raise AssertionError

    candidates = (
        object(),
        object.__new__(LoadedSourceClasses),
        object.__new__(ForeignLoaded),
        replace(loaded),
        foreign["rows"][0],
    )
    calls, original = observe_current(case)
    try:
        for candidate in candidates:
            refuse_batch(helper, candidate)
        assert calls == []
        helper.require_current_bindings(loaded)
    finally:
        object.__setattr__(helper, "_current", original)
    save_unit(
        tmp_path,
        [case, foreign],
        calls,
        [0],
        "test_batch_refuses_foreign_and_malformed_loaded_rows",
    )


def test_batch_refuses_mutated_original_loaded_fields_and_rows(tmp_path):
    case = make_case()
    helper, loaded = case["helper"], case["rows"][0]
    row = loaded.classes[0]

    class ForeignBinding(SourceClassBinding):
        def __getattribute__(self, name):
            case["events"].append("binding_getter")
            raise AssertionError

    class ForeignTuple(tuple):
        def __iter__(self):
            case["events"].append("tuple_iteration")
            raise AssertionError

        def __len__(self):
            case["events"].append("tuple_length")
            raise AssertionError

    mutations = (
        (loaded, "prepared", replace(loaded.prepared)),
        (loaded, "source", replace(loaded.source)),
        (loaded, "classes", tuple(list(loaded.classes))),
        (loaded, "classes", list(loaded.classes)),
        (loaded, "classes", ()),
        (loaded, "classes", (row,) * 17),
        (loaded, "classes", (replace(row), loaded.classes[1])),
        (loaded, "classes", (object(), loaded.classes[1])),
        (loaded, "classes", (object.__new__(ForeignBinding), loaded.classes[1])),
        (loaded, "classes", ForeignTuple(loaded.classes)),
        (loaded.source, "namespace", loaded.source.namespace.copy()),
        (loaded.source, "module", object()),
        (loaded.source, "roots", tuple(list(loaded.source.roots))),
        (row, "namespace", row.namespace.copy()),
        (row, "code", replace(row.code)),
        (row, "owner_type", object()),
        (row, "descriptor", object()),
        (row, "constructor", FunctionType(row.constructor.__code__, row.namespace)),
        (row, "raw_call", FunctionType(row.raw_call.__code__, row.namespace)),
        (row.code, "call_code", row.code.call_code.replace()),
    )
    assert len(mutations) == 20
    calls, original = observe_current(case)
    try:
        for owner, field, replacement in mutations:
            previous = getattr(owner, field)
            assert replacement is not previous
            object.__setattr__(owner, field, replacement)
            try:
                refuse_batch(helper, loaded)
            finally:
                object.__setattr__(owner, field, previous)
            helper.require_current_bindings(loaded)
    finally:
        object.__setattr__(helper, "_current", original)
    save_unit(
        tmp_path,
        [case],
        calls,
        [0] * 40,
        "test_batch_refuses_mutated_original_loaded_fields_and_rows",
    )


def test_batch_refuses_native_class_function_and_namespace_mutations(tmp_path):
    case = make_case()
    helper, loaded = case["helper"], case["rows"][0]
    row = loaded.classes[0]
    namespace, owner, raw_call = row.namespace, row.owner_type, row.raw_call

    def hostile(*args):
        case["events"].append("foreign_dispatch")
        raise AssertionError

    class ForeignKey:
        __hash__ = object.__hash__

        def __eq__(self, other):
            case["events"].append("key_equality")
            raise AssertionError

    calls, original = observe_current(case)
    try:
        # Each refused native mutation is restored before a fresh accepted proof.
        for replacement in (
            object(),
            FunctionType(raw_call.__code__, namespace),
            property(hostile),
        ):
            setattr(owner, "__call__", replacement)
            try:
                refuse_batch(helper, loaded)
            finally:
                setattr(owner, "__call__", raw_call)
            helper.require_current_bindings(loaded)
        for field, replacement in (
            ("__code__", raw_call.__code__.replace()),
            ("__defaults__", (1,)),
            ("__kwdefaults__", {"x": 1}),
        ):
            previous = getattr(raw_call, field)
            setattr(raw_call, field, replacement)
            try:
                refuse_batch(helper, loaded)
            finally:
                setattr(raw_call, field, previous)
            helper.require_current_bindings(loaded)
        previous_class = namespace["Clock"]
        namespace["Clock"] = object()
        try:
            refuse_batch(helper, loaded)
        finally:
            namespace["Clock"] = previous_class
        helper.require_current_bindings(loaded)
        root = namespace["factory"]
        namespace["factory"] = FunctionType(root.__code__, namespace.copy())
        try:
            refuse_batch(helper, loaded)
        finally:
            namespace["factory"] = root
        helper.require_current_bindings(loaded)
        key = ForeignKey()
        namespace[key] = object()
        try:
            refuse_batch(helper, loaded)
        finally:
            del namespace[key]
        helper.require_current_bindings(loaded)
        with pytest.raises(AttributeError, match="not writable"):
            setattr(owner, "__dict__", property(hostile))
        assert type.__dict__["__dict__"].__get__(owner, type)["__dict__"] is row.descriptor
        helper.require_current_bindings(loaded)
    finally:
        object.__setattr__(helper, "_current", original)
    save_unit(
        tmp_path,
        [case],
        calls,
        [0] * 19,
        "test_batch_refuses_native_class_function_and_namespace_mutations",
    )


def test_batch_refuses_missing_original_capture_membership(tmp_path):
    case = make_case()
    helper, loaded = case["helper"], case["rows"][0]
    issue = helper._issues[0]
    captures = issue.captures
    calls, original = observe_current(case)
    try:
        issue.captures = ()
        with pytest.raises(TypeError, match="invalid source class witness"):
            helper.require_current_bindings(loaded)
        issue.captures = captures
        helper.require_current_bindings(loaded)
    finally:
        issue.captures = captures
        object.__setattr__(helper, "_current", original)
    save_unit(
        tmp_path, [case], calls, [0, 0], "test_batch_refuses_missing_original_capture_membership"
    )


def test_batch_reproves_first_cross_issue_owner_once(tmp_path):
    case = make_case(2)
    helper, loaded = case["helper"], case["rows"][0]
    first, second = helper._issues
    first_captures, second_captures = first.captures, second.captures
    calls, original = observe_current(case)
    try:
        first.captures = tuple(
            replace(capture, binding=replace(capture.binding)) for capture in first_captures
        )
        second.captures = second_captures + first_captures
        helper.require_current_bindings(loaded)
        assert calls == [0, 1]
        helper.require_current_bindings(loaded)
    finally:
        first.captures, second.captures = first_captures, second_captures
        object.__setattr__(helper, "_current", original)
    save_unit(
        tmp_path, [case], calls, [0, 1, 0, 1], "test_batch_reproves_first_cross_issue_owner_once"
    )


def test_batch_refuses_invalid_first_owner_despite_later_valid_owner(tmp_path):
    case = make_case(3)
    helper, loaded = case["helper"], case["rows"][0]
    main, first, later = helper._issues
    main_captures, first_captures, later_captures = main.captures, first.captures, later.captures
    first_loaded = case["rows"][1]
    original_source = first_loaded.source
    calls, original = observe_current(case)
    try:
        main.captures = tuple(
            replace(capture, binding=replace(capture.binding)) for capture in main_captures
        )
        first.captures = first_captures + main_captures
        later.captures = later_captures + main_captures
        # The later candidate is a valid issue. Earliest membership still governs.
        helper._current(later)
        object.__setattr__(first_loaded, "source", object())
        try:
            refuse_batch(helper, loaded)
            assert calls == [2, 0, 1]
        finally:
            object.__setattr__(first_loaded, "source", original_source)
        helper.require_current_bindings(loaded)
    finally:
        object.__setattr__(first_loaded, "source", original_source)
        main.captures, first.captures, later.captures = (
            main_captures,
            first_captures,
            later_captures,
        )
        object.__setattr__(helper, "_current", original)
    save_unit(
        tmp_path,
        [case],
        calls,
        [2, 0, 1, 0, 1],
        "test_batch_refuses_invalid_first_owner_despite_later_valid_owner",
    )
