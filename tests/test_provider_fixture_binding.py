"""Independent fixture alias controls using native stand-ins only.

These witnesses check retained object/code/storage continuity. They do not prove
original fixture loading, source authority, callback birth or globals contents.
Only the final integration dispatches: one factory and one returned provider.
"""

from dataclasses import FrozenInstanceError, replace
import json
import sys
import threading
from types import CodeType, FunctionType, GetSetDescriptorType
from typing import cast

import pytest

from provider_code_path import pin_provider_code_path, require_provider_callback_code
from provider_entry_profile import ProviderEntryProfile
from provider_fixture_binding import FixtureRootBinding, require_fixture_root_binding
from provider_owner_namespace_profile import OwnerNamespaceBinding


def make_case():
    bodies: list[str] = []
    events: list[str] = []

    def template():
        bodies.append("factory")

        def provider():
            bodies.append("provider")
            return 7

        return provider

    namespace: dict[str, object] = {"__name__": "bounded_fixture_standin"}
    template_function = cast(FunctionType, template)
    root = FunctionType(
        template_function.__code__, namespace, "factory", closure=template_function.__closure__
    )

    def hostile(*args):
        events.append("attribute_or_call")
        raise AssertionError("fixture predicate dispatched user code")

    class Wrapper:
        __getattribute__ = hostile
        __getattr__ = hostile
        __setattr__ = hostile
        __call__ = hostile
        _fixture_function = property(hostile)
        __wrapped__ = property(hostile)

    owner = object.__new__(Wrapper)
    descriptor = cast(GetSetDescriptorType, Wrapper.__dict__["__dict__"])
    owner_namespace = descriptor.__get__(owner, Wrapper)
    owner_namespace.update({"_fixture_function": root, "__wrapped__": root})
    namespace["fixture_root"] = owner
    wrapper = OwnerNamespaceBinding(owner, Wrapper, descriptor, owner_namespace)
    binding = FixtureRootBinding(namespace, "fixture_root", wrapper, root, root.__code__, namespace)
    return dict(
        binding=binding,
        wrapper=wrapper,
        root=root,
        namespace=namespace,
        owner_namespace=owner_namespace,
        bodies=bodies,
        events=events,
        Wrapper=Wrapper,
    )


def assert_refused(parts, binding=None):
    with pytest.raises((TypeError, ValueError)):
        require_fixture_root_binding(parts["binding"] if binding is None else binding)
    assert parts["bodies"] == []
    assert parts["events"] == []


def test_should_accept_original_native_aliases_without_attributes_or_factory_dispatch():
    parts = make_case()
    require_fixture_root_binding(parts["binding"])
    require_fixture_root_binding(parts["binding"])
    assert parts["owner_namespace"]["_fixture_function"] is parts["root"]
    assert parts["owner_namespace"]["__wrapped__"] is parts["root"]
    assert parts["bodies"] == [] and parts["events"] == []
    with pytest.raises(FrozenInstanceError):
        setattr(parts["binding"], "slot", "other")


@pytest.mark.parametrize("alias", ("_fixture_function", "__wrapped__", "both"))
def test_should_refuse_same_code_fresh_function_alias_identity(alias):
    parts = make_case()
    root = parts["root"]
    foreign = FunctionType(root.__code__, root.__globals__, closure=root.__closure__)
    assert foreign is not root and foreign.__code__ is root.__code__
    for key in ("_fixture_function", "__wrapped__") if alias == "both" else (alias,):
        parts["owner_namespace"][key] = foreign
    assert_refused(parts)


def test_should_refuse_equal_foreign_code_and_changed_original_code():
    parts = make_case()
    original = parts["root"].__code__
    clone = original.replace()
    assert clone is not original and clone == original
    assert_refused(parts, replace(parts["binding"], raw_code=clone))
    parts["root"].__code__ = clone
    assert_refused(parts)
    parts["root"].__code__ = original
    require_fixture_root_binding(parts["binding"])
    assert parts["bodies"] == [] and parts["events"] == []


@pytest.mark.parametrize(
    "change", ("missing_slot", "foreign_wrapper", "missing_alias", "bad_alias")
)
def test_should_refuse_replaced_module_wrapper_or_missing_raw_alias(change):
    parts = make_case()
    if change == "missing_slot":
        del parts["namespace"]["fixture_root"]
    elif change == "foreign_wrapper":
        parts["namespace"]["fixture_root"] = object.__new__(parts["Wrapper"])
    elif change == "missing_alias":
        del parts["owner_namespace"]["__wrapped__"]
    else:
        parts["owner_namespace"]["_fixture_function"] = None
    assert_refused(parts)


def test_should_refuse_replaced_owner_dictionary_and_class_with_original_witness():
    parts = make_case()
    owner = parts["wrapper"].owner
    original_namespace = parts["owner_namespace"]
    object.__setattr__(owner, "__dict__", original_namespace.copy())
    assert_refused(parts)
    object.__setattr__(owner, "__dict__", original_namespace)
    require_fixture_root_binding(parts["binding"])
    other_class = type("ForeignWrapper", (), {})
    object.__setattr__(owner, "__class__", other_class)
    assert_refused(parts)
    object.__setattr__(owner, "__class__", parts["Wrapper"])
    require_fixture_root_binding(parts["binding"])
    assert parts["bodies"] == [] and parts["events"] == []


def test_should_refuse_globals_mismatch_and_copied_module_namespace():
    parts = make_case()
    copied_namespace = parts["namespace"].copy()
    assert_refused(parts, replace(parts["binding"], function_globals=copied_namespace))
    assert_refused(
        parts,
        replace(
            parts["binding"],
            namespace=copied_namespace,
            function_globals=copied_namespace,
        ),
    )
    root = parts["root"]
    foreign = FunctionType(root.__code__, copied_namespace, closure=root.__closure__)
    parts["owner_namespace"]["_fixture_function"] = foreign
    parts["owner_namespace"]["__wrapped__"] = foreign
    assert_refused(parts, replace(parts["binding"], raw_function=foreign))


def test_should_refuse_foreign_incomplete_rows_and_wrong_native_fields_without_getters():
    parts = make_case()
    events = parts["events"]

    class ForeignBinding(FixtureRootBinding):
        def __getattribute__(self, name):
            events.append("binding_getter")
            raise AssertionError("foreign retained row dispatched")

    class ForeignOwnerBinding(OwnerNamespaceBinding):
        def __getattribute__(self, name):
            events.append("owner_row_getter")
            raise AssertionError("foreign owner row dispatched")

    class ForeignString(str):
        def __len__(self):
            events.append("slot_length")
            raise AssertionError("foreign slot length dispatched")

        def __hash__(self):
            events.append("slot_hash")
            raise AssertionError("foreign slot hash dispatched")

        def __eq__(self, other):
            events.append("slot_equality")
            raise AssertionError("foreign slot equality dispatched")

    malformed: list[object] = [
        object(),
        object.__new__(FixtureRootBinding),
        object.__new__(ForeignBinding),
    ]
    for binding in malformed:
        assert_refused(parts, binding)
    variants = (
        replace(parts["binding"], wrapper=object.__new__(OwnerNamespaceBinding)),
        replace(parts["binding"], wrapper=object.__new__(ForeignOwnerBinding)),
        replace(parts["binding"], wrapper=object()),
        replace(parts["binding"], raw_function=len),
        replace(parts["binding"], raw_code=object()),
        replace(parts["binding"], function_globals=object()),
        replace(parts["binding"], namespace=object()),
        replace(parts["binding"], slot=""),
        replace(parts["binding"], slot="s" * 129),
        replace(parts["binding"], slot=3),
        replace(parts["binding"], slot=ForeignString("fixture_root")),
        replace(parts["binding"], wrapper=replace(parts["wrapper"], descriptor=object())),
    )
    for binding in variants:
        assert_refused(parts, binding)


@pytest.mark.parametrize("location", ("module", "owner"))
def test_should_preflight_hostile_dictionary_keys_before_alias_lookup(location):
    parts = make_case()
    target = parts["namespace"] if location == "module" else parts["owner_namespace"]
    events = parts["events"]
    armed = False
    collision = hash("fixture_root" if location == "module" else "_fixture_function")

    class HostileKey:
        def __hash__(self):
            if armed:
                events.append("key_hash")
                raise AssertionError("hostile key hash dispatched")
            return collision

        def __eq__(self, other):
            if armed:
                events.append("key_equality")
                raise AssertionError("hostile key equality dispatched")
            return False

        def __repr__(self):
            events.append("key_repr")
            raise AssertionError("hostile key repr dispatched")

    target[HostileKey()] = None
    armed = True
    assert_refused(parts)


def test_should_refuse_dictionary_subclasses_without_length_iteration_or_lookup():
    parts = make_case()
    events = parts["events"]

    class ForeignDict(dict):
        def __len__(self):
            events.append("dict_length")
            raise AssertionError("foreign length dispatched")

        def __iter__(self):
            events.append("dict_iteration")
            raise AssertionError("foreign iteration dispatched")

        def get(self, *args):
            events.append("dict_lookup")
            raise AssertionError("foreign lookup dispatched")

    foreign_module = ForeignDict(parts["namespace"])
    assert_refused(parts, replace(parts["binding"], namespace=foreign_module))
    assert_refused(parts, replace(parts["binding"], function_globals=foreign_module))
    foreign_owner = ForeignDict(parts["owner_namespace"])
    object.__setattr__(parts["wrapper"].owner, "__dict__", foreign_owner)
    owner_binding = replace(parts["wrapper"], namespace=foreign_owner)
    assert_refused(parts, replace(parts["binding"], wrapper=owner_binding))


def test_should_accept_dictionary_key_bounds_and_refuse_overflow_or_long_keys():
    parts = make_case()
    for namespace in (parts["namespace"], parts["owner_namespace"]):
        namespace["k" * 128] = None
        for index in range(512 - len(namespace)):
            namespace[f"field{index}"] = None
    require_fixture_root_binding(parts["binding"])
    for namespace in (parts["namespace"], parts["owner_namespace"]):
        namespace["overflow"] = None
        assert_refused(parts)
        del namespace["overflow"]
        require_fixture_root_binding(parts["binding"])
        del namespace["k" * 128]
        namespace["k" * 129] = None
        assert_refused(parts)
        del namespace["k" * 129]
        namespace["k" * 128] = None
        require_fixture_root_binding(parts["binding"])
    assert parts["bodies"] == [] and parts["events"] == []


def test_should_preserve_same_guard_for_retained_factory_and_provider_once(tmp_path):
    parts = make_case()
    root = parts["root"]
    original_code = root.__code__
    index, provider_code = next(
        (index, value)
        for index, value in enumerate(original_code.co_consts)
        if type(value) is CodeType
    )
    path = pin_provider_code_path(root, original_code, (index,))
    require_fixture_root_binding(parts["binding"])
    guard = ProviderEntryProfile(
        ((original_code, "factory", 1), (provider_code, "provider", 1)),
        tmp_path / "breach",
    )
    table, counters, lock = guard._table, guard._counts, guard._lock
    before_main, before_thread = sys.getprofile(), threading.getprofile()
    with guard.measure():
        require_fixture_root_binding(parts["binding"])
        provider = root()
        require_provider_callback_code(path, provider)
        assert provider.__code__ is provider_code
        assert provider() == 7
        require_fixture_root_binding(parts["binding"])
    assert guard._table is table and guard._counts is counters and guard._lock is lock
    assert guard.counts == {"factory": 1, "provider": 1}
    assert parts["bodies"] == ["factory", "provider"] and parts["events"] == []
    assert sys.getprofile() is before_main and threading.getprofile() is before_thread
    assert threading.active_count() == 1 and not (tmp_path / "breach").exists()
    require_fixture_root_binding(parts["binding"])
    (tmp_path / "work.json").write_text(
        json.dumps(
            dict(
                bodies=2,
                factory_calls=1,
                provider_calls=1,
                counts=guard.counts,
                workers=0,
                joined=True,
                restored=True,
            )
        ),
        encoding="utf8",
    )
