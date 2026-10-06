"""Native namespace behavior that an exact-type optimization must preserve.

Why this: readonly __dict__ properties are deliberate runtime descriptor controls;
their three narrow ignores reflect object stubs' writable attribute assumption.
Dynamic member/type changes use setattr so static types do not claim extra fields.
"""

import types
import weakref

import pytest

from src.infra.runtime_python_objects import _namespace, _namespace_edges


def operation():
    raise AssertionError("namespace inspection must not execute a callable")


@pytest.mark.parametrize("value", [{}, [], (), set(), frozenset()])
def test_should_have_no_instance_namespace_for_exact_builtin_containers(value):
    assert _namespace(value) is None


@pytest.mark.parametrize("parent", [dict, list, tuple, set, frozenset])
def test_should_observe_callable_members_of_builtin_subclasses(parent):
    def native_access(self, name):
        if name == "__dict__":
            raise AssertionError("native slot lookup must bypass user attribute code")
        return object.__getattribute__(self, name)

    container = type("Container", (parent,), {"__getattribute__": native_access})
    value = container()
    setattr(value, "operation", operation)
    namespace = object.__getattribute__(value, "__dict__")
    assert _namespace(value) is namespace
    assert any(row[1][1] == id(operation) for row in _namespace_edges(namespace))


def test_should_not_invoke_custom_dictionary_properties():
    calls = []

    class PropertyDictionary:
        @property
        def __dict__(self):  # type: ignore[override]
            calls.append("property")
            raise AssertionError("inspection must not invoke a dictionary property")

    assert _namespace(PropertyDictionary()) is None
    assert calls == []


def test_should_read_module_native_dictionary():
    module = types.ModuleType("namespace_behavior_fixture")
    setattr(module, "operation", operation)
    assert _namespace(module) is module.__dict__


def test_should_read_class_native_dictionary_without_custom_attribute_code():
    class Meta(type):
        def __getattribute__(cls, name):
            if name == "__dict__":
                raise AssertionError("native class namespace must bypass the metaclass")
            return super().__getattribute__(name)

    class Subject(metaclass=Meta):
        handler = staticmethod(operation)

    namespace = _namespace(Subject)
    assert type(namespace) is types.MappingProxyType
    assert namespace["handler"].__func__ is operation


def test_should_not_compare_or_hash_foreign_metaclasses():
    class Meta(type):
        def __eq__(cls, other):
            raise AssertionError("type dispatch must not execute metaclass equality")

        def __hash__(cls):
            raise AssertionError("type dispatch must not execute metaclass hashing")

    class Subject(metaclass=Meta):
        pass

    value = Subject()
    setattr(value, "operation", operation)
    assert _namespace(value) is value.__dict__


@pytest.mark.parametrize("callable_proxy", [False, True])
def test_should_not_forward_namespace_access_through_weak_proxies(callable_proxy):
    class Subject:
        def __getattribute__(self, name):
            if name == "__dict__":
                raise AssertionError("a weak proxy must not forward dictionary access")
            return super().__getattribute__(name)

    class CallableSubject(Subject):
        __call__ = staticmethod(operation)

    value = CallableSubject() if callable_proxy else Subject()
    proxy = weakref.proxy(value)
    assert _namespace(proxy) is None
    del value
    assert _namespace(proxy) is None


@pytest.mark.parametrize("wrapper", [staticmethod, classmethod])
def test_should_preserve_native_callable_wrapper_dictionary(wrapper):
    value = wrapper(operation)
    setattr(value, "handler", operation)
    namespace = _namespace(value)
    assert namespace is value.__dict__
    assert namespace["handler"] is operation


def test_should_observe_inherited_native_instance_dictionary():
    class Parent:
        pass

    class Child(Parent):
        __slots__ = ()

    value = Child()
    setattr(value, "handler", operation)
    assert _namespace(value) is value.__dict__


def test_should_observe_class_changes_without_stale_namespace_cache():
    class NativeDictionary:
        pass

    class PropertyDictionary:
        @property
        def __dict__(self):  # type: ignore[override]
            raise AssertionError("new type property must never be invoked")

    value = NativeDictionary()
    setattr(value, "handler", operation)
    original = _namespace(value)
    setattr(value, "__class__", PropertyDictionary)
    assert _namespace(value) is None
    setattr(value, "__class__", NativeDictionary)
    assert _namespace(value) is original
    assert original is not None
    assert original["handler"] is operation


def test_should_observe_late_dictionary_descriptor_changes():
    class NativeParent:
        pass

    class PropertyParent(NativeParent):
        __slots__ = ()

        @property
        def __dict__(self):  # type: ignore[override]
            raise AssertionError("late inherited dictionary property must not run")

    class Subject(NativeParent):
        __slots__ = ()

    value = Subject()
    setattr(value, "handler", operation)
    original = _namespace(value)
    Subject.__bases__ = (PropertyParent,)
    try:
        assert _namespace(value) is None
    finally:
        Subject.__bases__ = (NativeParent,)
    assert _namespace(value) is original


def test_should_observe_late_callable_members_without_reusing_edges():
    class Subject:
        pass

    value = Subject()
    assert _namespace_edges(_namespace(value)) == []
    setattr(value, "handler", operation)
    assert _namespace_edges(_namespace(value))[0][1][1] == id(operation)
    delattr(value, "handler")
    assert _namespace_edges(_namespace(value)) == []
