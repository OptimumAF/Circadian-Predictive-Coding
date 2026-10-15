"""Bounded decorated-root controls; one real pytest creation path, no originals."""

from dataclasses import replace
import functools
from hashlib import sha256
import inspect
import json
import sys
import threading
from types import CodeType, FunctionType, GetSetDescriptorType, ModuleType
from typing import cast

import _pytest.deprecated as deprecated
import _pytest.fixtures as fixtures
import pytest

from provider_code_path import pin_provider_code_path, require_provider_callback_code
from provider_entry_profile import ProviderEntryProfile
from provider_fixture_binding import FixtureRootBinding, require_fixture_root_binding
from provider_owner_namespace_profile import OwnerNamespaceBinding
from provider_source_loader import (
    ApprovedSourceDecorator,
    ApprovedSourceImport,
    FixtureSourceLoader,
    LoadedFixtureSource,
    PreparedFixtureSource,
)

_TYPE_DICT = cast(GetSetDescriptorType, type.__dict__["__dict__"])
_MODULE_DICT = cast(GetSetDescriptorType, _TYPE_DICT.__get__(ModuleType, type)["__dict__"])
SOURCE = b"""import pytest
import decoratedeffects
decoratedeffects.counts['module'] += 1
@pytest.fixture(scope="module")
def factory():
    decoratedeffects.counts['factory'] += 1
    def provider():
        decoratedeffects.counts['provider'] += 1
        return 7
    return provider
"""
ZERO = dict(module=0, factory=0, provider=0, failed=0)
POSITIVE = dict(module=1, factory=1, provider=1, failed=0)
EXPECTED = dict(
    fixture=1,
    marker_init=1,
    marker_post_init=1,
    check_ispytest=2,
    marker_call=1,
    isclass=1,
    definition_init=1,
    update_wrapper=1,
    module=1,
    factory=1,
    provider=1,
)


def native_module(module):
    return _MODULE_DICT.__get__(module, ModuleType)


def native_class(owner_type):
    return _TYPE_DICT.__get__(owner_type, type)


def make_case():
    # Why explicit objects: the loader must not discover this live dependency graph.
    fixture_ns = native_module(fixtures)
    marker = fixture_ns["FixtureFunctionMarker"]
    wrapper = fixture_ns["FixtureFunctionDefinition"]
    marker_ns, wrapper_ns = native_class(marker), native_class(wrapper)
    function = native_module(pytest)["fixture"]
    support = (
        marker_ns["__init__"],
        marker_ns["__post_init__"],
        native_module(deprecated)["check_ispytest"],
        marker_ns["__call__"],
        native_module(inspect)["isclass"],
        wrapper_ns["__init__"],
        native_module(functools)["update_wrapper"],
    )
    assert type(function) is FunctionType and all(type(item) is FunctionType for item in support)
    descriptor = cast(GetSetDescriptorType, wrapper_ns["__dict__"])
    row = ApprovedSourceDecorator(
        "factory",
        "pytest",
        "fixture",
        function,
        wrapper,
        descriptor,
        cast(tuple[FunctionType, ...], support),
        (marker,),
    )
    dependency = ModuleType("decoratedeffects")
    effects = ZERO.copy()
    namespace = native_module(dependency)
    namespace["counts"] = effects
    imports = (
        ApprovedSourceImport("pytest", pytest),
        ApprovedSourceImport("decoratedeffects", dependency),
    )
    return dict(
        row=row,
        imports=imports,
        effects=effects,
        events=[],
        namespace=namespace,
        loader=FixtureSourceLoader(imports, decorators=(row,)),
        function=function,
        support=support,
        marker=marker,
        wrapper=wrapper,
        descriptor=descriptor,
    )


def prepare(parts, source=SOURCE, name="bounded_decorated_source"):
    return parts["loader"].prepare(
        source, sha256(source).hexdigest(), name, "decorated.py", ("factory",)
    )


def require_zero(parts):
    assert parts["effects"] == ZERO and parts["events"] == []


def refuse_loader(parts, decorators):
    with pytest.raises((TypeError, ValueError)):
        FixtureSourceLoader(parts["imports"], decorators=decorators)
    require_zero(parts)


def refuse_prepare(parts, source=SOURCE, name="bounded_decorated_source"):
    with pytest.raises((TypeError, ValueError)):
        prepare(parts, source, name)
    require_zero(parts)


def refuse_execute(parts, candidate):
    with pytest.raises((TypeError, ValueError)):
        parts["loader"].execute(candidate)
    require_zero(parts)


def refuse_current(parts, candidate):
    with pytest.raises((TypeError, ValueError)):
        parts["loader"].require_current(candidate)
    assert parts["events"] == [] and parts["effects"] == POSITIVE


def test_decorated_policy_rejects_malformed_native_inputs():
    parts = make_case()
    row, events = parts["row"], parts["events"]

    class ForeignTuple(tuple):
        def __len__(self):
            events.append("tuple_length")
            raise AssertionError

        def __iter__(self):
            events.append("tuple_iteration")
            raise AssertionError

    class ForeignRow(ApprovedSourceDecorator):
        def __getattribute__(self, name):
            events.append("row_getter")
            raise AssertionError

    class ForeignName(str):
        def __hash__(self):
            events.append("name_hash")
            raise AssertionError

        def __eq__(self, other):
            events.append("name_equality")
            raise AssertionError

    candidate: object
    for candidate in (
        [],
        ForeignTuple(()),
        (object(),),
        (object.__new__(ApprovedSourceDecorator),),
        (object.__new__(ForeignRow),),
        (row, row),
        (row,) * 17,
    ):
        refuse_loader(parts, candidate)
    changes: tuple[dict[str, object], ...] = (
        dict(root=""),
        dict(root="bad.name"),
        dict(root=ForeignName("factory")),
        dict(root="x" * 129),
        dict(module_name="__private"),
        dict(attribute=3),
        dict(function=object()),
        dict(wrapper_type=object()),
        dict(descriptor=object()),
        dict(dependencies=[]),
        dict(dependencies=ForeignTuple(())),
        dict(dependencies=(parts["function"],)),
        dict(dependencies=(parts["support"][0],) * 2),
        dict(dependencies=(object(),)),
        dict(dependencies=parts["support"] * 3),
        dict(classes=[]),
        dict(classes=ForeignTuple(())),
        dict(classes=(parts["wrapper"],)),
        dict(classes=(parts["marker"],) * 2),
        dict(classes=(object(),)),
        dict(classes=(parts["marker"],) * 17),
    )
    for change in changes:
        refuse_loader(parts, (replace(row, **change),))
    for declaration in (
        b"def factory(value=0):",
        b"def factory(*, value=0):",
        b"def factory[T]():",
    ):
        refuse_prepare(parts, SOURCE.replace(b"def factory():", declaration))
    refuse_prepare(parts, SOURCE.replace(b'@pytest.fixture(scope="module")', b"@pytest.fixture()"))
    # Bare decoration is also used by the originals; preparation executes no bodies.
    bare = SOURCE.replace(b'@pytest.fixture(scope="module")', b"@pytest.fixture")
    prepare(parts, bare, "bounded_decorated_bare")
    request_source = SOURCE.replace(b"factory", b"request")
    with pytest.raises((TypeError, ValueError)):
        request_loader = FixtureSourceLoader(
            parts["imports"], decorators=(replace(row, root="request"),)
        )
        request_loader.prepare(
            request_source,
            sha256(request_source).hexdigest(),
            "decorated_request",
            "request.py",
            ("request",),
        )
    # A policy root must actually be among the selected compiled roots.
    with pytest.raises((TypeError, ValueError)):
        parts["loader"].prepare(
            b"def other():\n    return None\n",
            sha256(b"def other():\n    return None\n").hexdigest(),
            "decorated_wrong_root",
            "other.py",
            ("other",),
        )
    require_zero(parts)


def test_decorated_policy_rejects_hostile_surfaces():
    parts = make_case()
    row, events = parts["row"], parts["events"]

    class CallableObject:
        def __call__(self, *args):
            events.append("foreign_call")
            raise AssertionError

        def __getattribute__(self, name):
            events.append("foreign_getter")
            raise AssertionError

    class HostileMeta(type):
        def __getattribute__(self, name):
            events.append("metaclass_getter")
            raise AssertionError

    class HostileOwner(metaclass=HostileMeta):
        pass

    class Parent:
        pass

    class Inherited(Parent):
        pass

    class PropertyOwner:
        @property  # Intentional hostile surface.
        def __dict__(self):  # type: ignore[override]
            events.append("property_getter")
            raise AssertionError

    for change in (
        dict(function=CallableObject()),
        dict(wrapper_type=HostileOwner),
        dict(classes=(HostileOwner,)),
        dict(wrapper_type=Inherited, descriptor=native_class(Parent)["__dict__"]),
        dict(wrapper_type=PropertyOwner, descriptor=native_class(PropertyOwner)["__dict__"]),
        dict(descriptor=native_class(Parent)["__dict__"]),
        dict(descriptor=native_class(parts["wrapper"])["__weakref__"]),
    ):
        refuse_loader(parts, (replace(row, **change),))
    wide = type("WideOwner", (), {f"k{i}": None for i in range(513)})
    refuse_loader(
        parts, (replace(row, wrapper_type=wide, descriptor=native_class(wide)["__dict__"]),)
    )
    long_key = type("LongKeyOwner", (), {"k" * 129: None})
    refuse_loader(
        parts, (replace(row, wrapper_type=long_key, descriptor=native_class(long_key)["__dict__"]),)
    )
    armed = False

    class HostileKey:
        def __hash__(self):
            if armed:
                events.append("key_hash")
                raise AssertionError
            return hash("counts")

        def __eq__(self, other):
            if armed:
                events.append("key_equality")
                raise AssertionError
            return False

    key = HostileKey()
    parts["namespace"][key] = None
    armed = True
    try:
        refuse_prepare(parts)
    finally:
        armed = False
        del parts["namespace"][key]
    require_zero(parts)


def test_decorated_policy_requires_original_binding():
    parts = make_case()
    function = parts["function"]
    function_code = function.__code__
    fn_globals = function.__globals__
    pytest_ns = native_module(pytest)
    prepared = prepare(parts)
    for foreign in (
        FunctionType(function_code, fn_globals),
        FunctionType(function_code.replace(), fn_globals),
        object(),
    ):
        pytest_ns["fixture"] = foreign
        try:
            refuse_prepare(parts)
            refuse_execute(parts, prepared)
        finally:
            pytest_ns["fixture"] = function
    for original_function in (function, *parts["support"]):
        code = original_function.__code__
        clone = code.replace()
        assert clone == code and clone is not code
        original_function.__code__ = clone
        try:
            refuse_prepare(parts)
            refuse_execute(parts, prepared)
        finally:
            original_function.__code__ = code
    for owner_type in (parts["wrapper"], parts["marker"]):
        type.__setattr__(owner_type, "bounded_changed", object())
        try:
            refuse_prepare(parts)
            refuse_execute(parts, prepared)
        finally:
            type.__delattr__(owner_type, "bounded_changed")
    fixture_namespace = native_module(fixtures)
    original = fixture_namespace["FixtureFunctionMarker"]
    fixture_namespace["FixtureFunctionMarker"] = object()
    try:
        refuse_execute(parts, prepared)
    finally:
        fixture_namespace["FixtureFunctionMarker"] = original
    descriptor = parts["descriptor"]
    with pytest.raises(AttributeError, match="not writable"):
        type.__setattr__(
            parts["wrapper"],
            "__dict__",
            property(lambda self: parts["events"].append("descriptor")),
        )
    assert native_class(parts["wrapper"])["__dict__"] is descriptor
    # Mutating a retained row cannot replace the originally captured policy.
    original_root = parts["row"].root
    object.__setattr__(parts["row"], "root", "other")
    try:
        refuse_execute(parts, prepared)
    finally:
        object.__setattr__(parts["row"], "root", original_root)
    require_zero(parts)


def test_decorated_prepared_rows_are_owned_immutable():
    parts = make_case()
    prepared = prepare(parts)
    events = parts["events"]

    class ForeignPrepared(PreparedFixtureSource):
        def __getattribute__(self, name):
            events.append("prepared_getter")
            raise AssertionError

    for foreign in (
        object(),
        object.__new__(PreparedFixtureSource),
        object.__new__(ForeignPrepared),
        replace(prepared),
    ):
        refuse_execute(parts, foreign)
    other = make_case()
    refuse_execute(other, prepared)
    for name, replacement in (
        ("module_name", "decorated_other"),
        ("filename", "other.py"),
        ("source_sha256", "0" * 64),
        ("module_code", prepared.module_code.replace()),
        ("root_codes", tuple(list(prepared.root_codes))),
        ("root_codes", (("factory", prepared.root_codes[0][1].replace()),)),
    ):
        original = getattr(prepared, name)
        assert replacement is not original
        object.__setattr__(prepared, name, replacement)
        try:
            refuse_execute(parts, prepared)
        finally:
            object.__setattr__(prepared, name, original)
    require_zero(other)


def test_decorated_cache_refuses_without_overwrite():
    parts = make_case()
    cache = cast(dict[str, object], sys.modules)
    name = "bounded_decorated_cache"
    assert name not in sys.modules
    for cached in (ModuleType(name), object()):
        cache[name] = cached
        try:
            refuse_prepare(parts, name=name)
            assert sys.modules[name] is cached
        finally:
            del sys.modules[name]
    prepared = prepare(parts, name=name)
    for cached in (ModuleType(name), object()):
        cache[name] = cached
        try:
            refuse_execute(parts, prepared)
            assert sys.modules[name] is cached
        finally:
            del sys.modules[name]
    require_zero(parts)


@pytest.mark.parametrize(
    "syntax",
    (
        "@unknown",
        "@pytest.other",
        "@pytest.fixture\n@pytest.fixture",
        "@(pytest.fixture if True else pytest.fixture)",
        "@pytest.fixture.other",
        "@pytest.fixture(decoratedeffects.counts.update(module=99))",
        "@pytest.fixture(autouse=True)",
        "pytest = None\n@pytest.fixture",
    ),
    ids=(
        "unapproved-name",
        "unapproved-attribute",
        "stacked",
        "computed",
        "deeper-attribute",
        "effectful-argument",
        "unapproved-keyword",
        "rebound-module",
    ),
)
def test_decorated_syntax_preflight(syntax):
    parts = make_case()
    source = (
        "import pytest\nimport decoratedeffects\ndecoratedeffects.counts['module'] += 1\n"
        + syntax
        + "\ndef factory():\n    return None\n"
    ).encode()
    refuse_prepare(parts, source)
    # Additional import/rebinding forms stay preparation-only within each case.
    for prefix in (
        "import pytest as renamed\n",
        "from pytest import fixture\n",
        "import pytest, decoratedeffects\n",
        "import pytest\nimport pytest\n",
        "import pytest\ndef unused(pytest):\n    return None\n",
        "import pytest\npytest.fixture = None\n",
    ):
        text = (
            prefix
            + "import decoratedeffects\ndecoratedeffects.counts['module'] += 1\n@pytest.fixture\ndef factory():\n    return None\n"
        )
        refuse_prepare(parts, text.encode())
    require_zero(parts)


def test_decorated_source_preserves_original_wrapper_under_one_guard(tmp_path):
    parts = make_case()
    prepared = prepare(parts)
    root_name, root_code = prepared.root_codes[0]
    assert root_name == "factory"
    index, provider_code = next(
        (i, code) for i, code in enumerate(root_code.co_consts) if type(code) is CodeType
    )
    support_labels = (
        "marker_init",
        "marker_post_init",
        "check_ispytest",
        "marker_call",
        "isclass",
        "definition_init",
        "update_wrapper",
    )
    entries = (
        (parts["function"].__code__, "fixture", 1),
        *(
            (function.__code__, label, EXPECTED[label])
            for function, label in zip(parts["support"], support_labels)
        ),
        (prepared.module_code, "module", 1),
        (root_code, "factory", 1),
        (provider_code, "provider", 1),
    )
    assert len(entries) == 11 and sum(EXPECTED.values()) == 12
    guard = ProviderEntryProfile(entries, tmp_path / "breach")
    table, counters, lock = guard._table, guard._counts, guard._lock
    before_main, before_thread = sys.getprofile(), threading.getprofile()
    original_dependency = sys.modules.get("decoratedeffects")
    assert original_dependency is None
    fake = ModuleType("decoratedeffects")
    sys.modules["decoratedeffects"] = fake
    try:
        with guard.measure():
            loaded = parts["loader"].execute(prepared)
            root = loaded.roots[0]
            namespace = loaded.namespace
            owner = namespace["factory"]
            assert type(owner) is parts["wrapper"]
            assert type(root) is FunctionType and root.__code__ is root_code
            assert root.__globals__ is namespace
            assert namespace["pytest"] is pytest
            assert namespace["decoratedeffects"] is parts["imports"][1].module
            storage = parts["descriptor"].__get__(owner, parts["wrapper"])
            assert storage["_fixture_function"] is root and storage["__wrapped__"] is root
            wrapper = OwnerNamespaceBinding(owner, parts["wrapper"], parts["descriptor"], storage)
            binding = FixtureRootBinding(namespace, "factory", wrapper, root, root_code, namespace)
            require_fixture_root_binding(binding)
            path = pin_provider_code_path(root, root_code, (index,))
            provider = root()
            require_provider_callback_code(path, provider)
            assert provider() == 7
            require_fixture_root_binding(binding)
            parts["loader"].require_current(loaded)
        assert sys.modules["decoratedeffects"] is fake
    finally:
        del sys.modules["decoratedeffects"]
    assert guard._table is table and guard._counts is counters and guard._lock is lock
    assert guard.counts == EXPECTED and parts["effects"] == POSITIVE
    assert sys.getprofile() is before_main and threading.getprofile() is before_thread
    assert threading.active_count() == 1 and not (tmp_path / "breach").exists()
    parts["loader"].require_current(loaded)
    with pytest.raises(ValueError):
        parts["loader"].execute(prepared)

    class ForeignLoaded(LoadedFixtureSource):
        def __getattribute__(self, name):
            parts["events"].append("loaded_getter")
            raise AssertionError

    candidate: object
    for candidate in (
        object(),
        object.__new__(LoadedFixtureSource),
        object.__new__(ForeignLoaded),
        replace(loaded),
    ):
        refuse_current(parts, candidate)
    for name, replacement in (
        ("prepared", replace(prepared)),
        ("module", ModuleType("foreign")),
        ("namespace", namespace.copy()),
        ("roots", tuple(list(loaded.roots))),
    ):
        original = getattr(loaded, name)
        object.__setattr__(loaded, name, replacement)
        try:
            refuse_current(parts, loaded)
        finally:
            object.__setattr__(loaded, name, original)
        parts["loader"].require_current(loaded)
    clone = root_code.replace()
    assert clone == root_code and clone is not root_code
    foreign: object
    for alias in ("_fixture_function", "__wrapped__"):
        for foreign in (
            FunctionType(root_code, namespace),
            FunctionType(clone, namespace),
            FunctionType(root_code, namespace.copy()),
            None,
        ):
            storage[alias] = foreign
            try:
                refuse_current(parts, loaded)
                with pytest.raises((TypeError, ValueError)):
                    require_fixture_root_binding(binding)
            finally:
                storage[alias] = root
            parts["loader"].require_current(loaded)
    root.__code__ = clone
    try:
        refuse_current(parts, loaded)
    finally:
        root.__code__ = root_code
    for foreign in (root, object(), FunctionType(root_code, namespace)):
        namespace["factory"] = foreign
        try:
            refuse_current(parts, loaded)
        finally:
            namespace["factory"] = owner
    original_storage = storage
    object.__setattr__(owner, "__dict__", storage.copy())
    try:
        refuse_current(parts, loaded)
    finally:
        object.__setattr__(owner, "__dict__", original_storage)

    class OtherOwner:
        pass

    object.__setattr__(owner, "__class__", OtherOwner)
    try:
        refuse_current(parts, loaded)
    finally:
        object.__setattr__(owner, "__class__", parts["wrapper"])
    with pytest.raises(AttributeError, match="not writable"):
        type.__setattr__(
            parts["wrapper"],
            "__dict__",
            property(lambda self: parts["events"].append("descriptor")),
        )
    assert native_class(parts["wrapper"])["__dict__"] is parts["descriptor"]
    # Both original aliases are necessary even when changed together coherently.
    foreign = FunctionType(root_code, namespace)
    storage.update(_fixture_function=foreign, __wrapped__=foreign)
    try:
        refuse_current(parts, loaded)
    finally:
        storage.update(_fixture_function=root, __wrapped__=root)
    for function in (parts["function"], *parts["support"]):
        code = function.__code__
        function.__code__ = code.replace()
        try:
            refuse_current(parts, loaded)
        finally:
            function.__code__ = code
    pytest_ns = native_module(pytest)
    pytest_ns["fixture"] = FunctionType(parts["function"].__code__, parts["function"].__globals__)
    try:
        refuse_current(parts, loaded)
    finally:
        pytest_ns["fixture"] = parts["function"]
    original_dependencies = parts["row"].dependencies
    object.__setattr__(parts["row"], "dependencies", tuple(list(original_dependencies)))
    try:
        refuse_current(parts, loaded)
    finally:
        object.__setattr__(parts["row"], "dependencies", original_dependencies)
    for owner_type in (parts["wrapper"], parts["marker"]):
        type.__setattr__(owner_type, "bounded_changed", object())
        try:
            refuse_current(parts, loaded)
        finally:
            type.__delattr__(owner_type, "bounded_changed")
    type.__setattr__(
        parts["wrapper"],
        "bounded_property",
        property(lambda self: parts["events"].append("property_getter")),
    )
    try:
        refuse_current(parts, loaded)
    finally:
        type.__delattr__(parts["wrapper"], "bounded_property")
    key_namespace = cast(dict[object, object], storage)
    key_namespace[3] = None
    try:
        refuse_current(parts, loaded)
    finally:
        del key_namespace[3]
    parts["loader"].require_current(loaded)
    require_fixture_root_binding(binding)
    assert storage is parts["descriptor"].__get__(owner, parts["wrapper"])
    assert namespace["factory"] is owner and root.__code__ is root_code
    assert guard.counts == EXPECTED and parts["effects"] == POSITIVE and parts["events"] == []
    with pytest.raises(RuntimeError, match="cannot renew"):
        with guard.measure():
            raise AssertionError("guard unexpectedly renewed")
    (tmp_path / "work.json").write_text(
        json.dumps(
            dict(
                bodies=12,
                module_calls=1,
                factory_calls=1,
                provider_calls=1,
                counts=guard.counts,
                side_effects=parts["effects"],
                workers=0,
                joined=True,
                restored=True,
            )
        ),
        encoding="utf8",
    )
