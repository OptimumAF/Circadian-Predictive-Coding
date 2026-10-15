"""Eight source-class controls; one real pytest window and one spent exception."""

from dataclasses import replace
from hashlib import sha256
import json
import sys
import threading
from types import CodeType, FunctionType, GetSetDescriptorType, ModuleType
from typing import cast

import pytest

from provider_entry_profile import ProviderEntryProfile
from provider_fixture_binding import FixtureRootBinding, require_fixture_root_binding
from provider_owner_namespace_profile import OwnerNamespaceBinding
from provider_source_classes import (
    FixtureSourceClasses,
    LoadedSourceClasses,
    PreparedSourceClasses,
    SourceClassBinding,
    SourceClassCode,
)
from provider_source_loader import FixtureSourceLoader
from test_provider_decorated_source_loader import make_case


CLASS = """class Clock:
    def __init__(self):
        self.value = 7
    def __call__(self):
        decoratedeffects.counts['provider'] += 1
        return self.value
"""
PREFIX = "import pytest\nimport decoratedeffects\ndecoratedeffects.counts['module'] += 1\n"
FACTORY = """@pytest.fixture(scope="module")
def factory():
    decoratedeffects.counts['factory'] += 1
    return Clock()
"""
SOURCE = (PREFIX + CLASS + FACTORY).encode()
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
    class_body=1,
    constructor=1,
)
TYPE_DICT = cast(GetSetDescriptorType, type.__dict__["__dict__"])


def parts_for(max_classes=16):
    parts = make_case()
    parts["classes"] = FixtureSourceClasses(parts["loader"], max_classes)
    return parts


def prepare(parts, source=SOURCE, name="bounded_source_classes", classes=("Clock",)):
    return parts["classes"].prepare(
        source, sha256(source).hexdigest(), name, "classes.py", ("factory",), class_roots=classes
    )


def zero(parts):
    assert parts["effects"] == ZERO and parts["events"] == []


def refuse_prepare(parts, source=SOURCE, classes=("Clock",), name="bounded_source_classes"):
    with pytest.raises((TypeError, ValueError)):
        prepare(parts, source, name, classes)
    zero(parts)


def refuse_execute(parts, candidate):
    with pytest.raises((TypeError, ValueError)):
        parts["classes"].execute(candidate)
    zero(parts)


def refuse_arguments(parts, arguments):
    with pytest.raises((TypeError, ValueError)):
        parts["classes"].prepare(*arguments, class_roots=("Clock",))
    zero(parts)


def refuse_current(parts, loaded, binding):
    with pytest.raises((TypeError, ValueError)):
        parts["classes"].require_current(loaded)
    with pytest.raises((TypeError, ValueError)):
        parts["classes"].require_binding(binding)
    assert parts["effects"] == POSITIVE and parts["events"] == []


def save_work(tmp_path, guard, effects, bodies, factory_calls, provider_calls):
    (tmp_path / "work.json").write_text(
        json.dumps(
            dict(
                bodies=bodies,
                module_calls=1,
                factory_calls=factory_calls,
                provider_calls=provider_calls,
                counts=guard.counts,
                side_effects=effects,
                workers=0,
                joined=True,
                restored=True,
            )
        ),
        encoding="utf8",
    )


def test_source_classes_refuses_malformed_native_inputs():
    parts = parts_for()
    events = parts["events"]

    class ForeignLoader(FixtureSourceLoader):
        def __getattribute__(self, name):
            events.append("loader_getter")
            raise AssertionError

    class ForeignTuple(tuple):
        def __iter__(self):
            events.append("tuple_iteration")
            raise AssertionError

        def __len__(self):
            events.append("tuple_length")
            raise AssertionError

    class ForeignName(str):
        def __hash__(self):
            events.append("name_hash")
            raise AssertionError

        def __eq__(self, other):
            events.append("name_equality")
            raise AssertionError

    candidate: object
    for candidate in (object(), object.__new__(ForeignLoader)):
        with pytest.raises((TypeError, ValueError)):
            FixtureSourceClasses(cast(FixtureSourceLoader, candidate))
    for candidate in (False, 0, 17, 1.0, "1", object()):
        with pytest.raises((TypeError, ValueError)):
            FixtureSourceClasses(parts["loader"], cast(int, candidate))
    for candidate in (
        [],
        (),
        ForeignTuple(("Clock",)),
        (3,),
        (ForeignName("Clock"),),
        ("",),
        ("__private",),
        ("bad.name",),
        ("é",),
        ("x" * 129,),
        ("Clock", "Clock"),
        ("Clock",) * 17,
        ("Missing",),
        ("factory",),
    ):
        refuse_prepare(parts, classes=candidate)
    # These native input refusals must happen before parsing or imported effects.
    bad_arguments: tuple[tuple[object, ...], ...] = (
        (bytearray(SOURCE), sha256(SOURCE).hexdigest(), "bounded_bad", "classes.py", ("factory",)),
        (b"", sha256(b"").hexdigest(), "bounded_bad", "classes.py", ("factory",)),
        (
            SOURCE + b" " * 16384,
            sha256(SOURCE + b" " * 16384).hexdigest(),
            "bounded_bad",
            "classes.py",
            ("factory",),
        ),
        (SOURCE, "0" * 64, "bounded_bad", "classes.py", ("factory",)),
        (SOURCE, 3, "bounded_bad", "classes.py", ("factory",)),
        (SOURCE, sha256(SOURCE).hexdigest(), "bad.name", "classes.py", ("factory",)),
        (SOURCE, sha256(SOURCE).hexdigest(), "bounded_bad", 3, ("factory",)),
        (SOURCE, sha256(SOURCE).hexdigest(), "bounded_bad", "classes.py", ("factory", "Clock")),
    )
    for arguments in bad_arguments:
        refuse_arguments(parts, arguments)
    one = parts_for(1)
    first = prepare(one, name="bounded_class_capacity_one")
    assert len(first.classes) == 1
    refuse_prepare(one, name="bounded_class_capacity_spent")
    many = parts_for(16)
    names = tuple(f"Clock{i}" for i in range(16))
    source = (PREFIX + "".join(CLASS.replace("Clock", name) for name in names) + FACTORY).encode()
    largest = prepare(many, source, "bounded_class_capacity_sixteen", names)
    assert tuple(row.slot for row in largest.classes) == names
    refuse_prepare(many, name="bounded_class_capacity_overflow")
    zero(one)
    zero(many)
    zero(parts)


def test_source_classes_refuses_class_grammar_before_effects():
    parts = parts_for()
    variants = (
        "@unknown\n" + CLASS,
        CLASS.replace("class Clock:", "class Clock(object, object):"),
        CLASS.replace("class Clock:", "class Clock(unknown):"),
        CLASS.replace("class Clock:", "class Clock(metaclass=type):"),
        CLASS.replace("class Clock:", "class Clock[T]:"),
        CLASS.replace("class Clock:\n", "class Clock:\n    extra = 1\n"),
        CLASS.replace("class Clock:\n", "class Clock:\n    __slots__ = ()\n"),
        CLASS.replace(
            "class Clock:\n", "class Clock:\n    def __new__(cls):\n        return None\n"
        ),
        CLASS.replace(
            "class Clock:\n",
            "class Clock:\n    def __getattribute__(self, name):\n        return None\n",
        ),
        CLASS.replace("    def __call__(self):", "    @property\n    def __call__(self):"),
        CLASS.replace("    def __init__(self):", "    @unknown\n    def __init__(self):"),
        CLASS.replace("    def __init__(self):", "    async def __init__(self):"),
        CLASS.replace("    def __call__(self):", "    async def __call__(self):"),
        CLASS.replace("def __init__(self):", "def __init__():"),
        CLASS.replace("def __call__(self):", "def __call__(other):"),
        CLASS.replace("def __init__(self):", "def __init__(self, value=0):"),
        CLASS.replace("def __call__(self):", "def __call__(self, *, value=0):"),
        CLASS.replace("def __init__(self):", "def __init__(self: object):"),
        CLASS.replace("def __call__(self):", "def __call__(self) -> int:"),
        CLASS.replace("return self.value", "yield self.value"),
        CLASS.replace("self.value = 7", "yield 7"),
        CLASS.replace("return self.value", "return __class__"),
        CLASS.replace(
            "    def __call__(self):\n        decoratedeffects.counts['provider'] += 1\n        return self.value\n",
            "",
        ),
        CLASS + CLASS,
        "if True:\n" + "".join("    " + line + "\n" for line in CLASS.splitlines()),
    )
    for variant in variants:
        refuse_prepare(parts, (PREFIX + variant + FACTORY).encode())
    # Constant-heavy local writes avoid an unrelated wide attribute AST.
    wide = (
        "class Clock:\n    def __init__(self):\n        self.value = 7\n    def __call__(self):\n"
        + "".join(f"        value = 'value{i}'\n" for i in range(513))
        + "        return self.value\n"
    )
    wide_source = (PREFIX + wide + FACTORY).encode()
    assert len(wide_source) <= 16384
    refuse_prepare(parts, wide_source)
    for variant in (
        CLASS.replace("class Clock:", "class Clock(object):"),
        CLASS.replace("class Clock:\n", 'class Clock:\n    "retained class documentation"\n'),
    ):
        prepared = prepare(
            parts, (PREFIX + variant + FACTORY).encode(), "bounded_class_grammar_native"
        )
        assert len(prepared.classes) == 1 and prepared.classes[0].slot == "Clock"
        zero(parts)


def test_source_classes_refuses_root_rebindings_before_effects():
    parts = parts_for()
    explicit_base = SOURCE.replace(b"class Clock:", b"class Clock(object):")
    for prefix in (
        b"object = None\n",
        b"def object():\n    return None\n",
        b"import decoratedeffects as object\n",
        b"from decoratedeffects import counts as object\n",
        b"def later():\n    global object\n    object = None\n",
        b"def later(object):\n    return None\n",
        b"del object\n",
    ):
        refuse_prepare(parts, prefix + explicit_base)
    statements = (
        "Clock = None\n",
        "Clock += 1\n",
        "del Clock\n",
        "Clock.changed = None\n",
        "del Clock.changed\n",
        "alias = Clock\n",
        "Clock()\n",
        "(lambda: Clock)()\n",
        "values = [Clock for item in range(1)]\n",
        "for Clock in range(1):\n    pass\n",
        "import decoratedeffects as Clock\n",
        "from decoratedeffects import counts as Clock\n",
        "def later(Clock):\n    return None\n",
        "def later(value=Clock):\n    return None\n",
        "def later(value: Clock):\n    return None\n",
        "@Clock\ndef later():\n    return None\n",
        "def later():\n    global Clock\n    return None\n",
        "def later():\n    class Clock:\n        pass\n",
        "def later():\n    Clock.changed = None\n",
        "try:\n    pass\nexcept RuntimeError as Clock:\n    pass\n",
        "match None:\n    case Clock:\n        pass\n",
    )
    for statement in statements:
        refuse_prepare(parts, SOURCE + statement.encode())
    # A lambda default is evaluated eagerly even though its body is a function.
    refuse_prepare(parts, SOURCE + b"alias = (lambda value=Clock: value)\n")


def test_source_classes_refuses_foreign_prepared_rows():
    parts = parts_for()
    prepared = prepare(parts)
    events = parts["events"]

    class ForeignPrepared(PreparedSourceClasses):
        def __getattribute__(self, name):
            events.append("prepared_getter")
            raise AssertionError

    class ForeignCode(SourceClassCode):
        def __getattribute__(self, name):
            events.append("code_getter")
            raise AssertionError

    for candidate in (
        object(),
        object.__new__(PreparedSourceClasses),
        object.__new__(ForeignPrepared),
        replace(prepared),
        replace(prepared, source=replace(prepared.source)),
        replace(prepared, classes=(replace(prepared.classes[0]),)),
        replace(prepared, classes=(object.__new__(SourceClassCode),)),
        replace(prepared, classes=(object.__new__(ForeignCode),)),
    ):
        refuse_execute(parts, candidate)
    other = parts_for()
    refuse_execute(other, prepared)
    zero(other)
    zero(parts)


def test_source_classes_refuses_mutated_prepared_rows():
    parts = parts_for()
    prepared = prepare(parts)
    row, source = prepared.classes[0], prepared.source

    class ForeignCode(SourceClassCode):
        def __getattribute__(self, name):
            parts["events"].append("code_getter")
            raise AssertionError

    mutations = (
        (prepared, "source", replace(source)),
        (prepared, "classes", tuple(list(prepared.classes))),
        (prepared, "classes", (replace(row),)),
        (prepared, "classes", (object.__new__(ForeignCode),)),
        (source, "module_name", "bounded_changed_class_name"),
        (source, "filename", "changed.py"),
        (source, "source_sha256", "0" * 64),
        (source, "module_code", source.module_code.replace()),
        (source, "root_codes", tuple(list(source.root_codes))),
        (row, "slot", "Other"),
        (row, "class_code", row.class_code.replace()),
        (row, "constructor_code", row.constructor_code.replace()),
        (row, "call_code", row.call_code.replace()),
        (row, "call_code", row.call_code.replace(co_consts=(None,) * 513)),
    )
    for owner, field, replacement in mutations:
        original = getattr(owner, field)
        assert replacement is not original
        object.__setattr__(owner, field, replacement)
        try:
            refuse_execute(parts, prepared)
        finally:
            object.__setattr__(owner, field, original)
    armed = False

    class HostileKey:
        def __hash__(self):
            if armed:
                parts["events"].append("key_hash")
                raise AssertionError
            return hash("counts")

        def __eq__(self, other):
            if armed:
                parts["events"].append("key_equality")
                raise AssertionError
            return False

    key = HostileKey()
    parts["namespace"][key] = None
    armed = True
    try:
        refuse_execute(parts, prepared)
    finally:
        armed = False
        del parts["namespace"][key]
    zero(parts)


def test_source_classes_spends_execution_after_exception(tmp_path):
    parts = parts_for(1)
    prefix = "import pytest\nimport decoratedeffects\ndecoratedeffects.counts['failed'] += 1\nif decoratedeffects.counts['failed'] == 1:\n    raise RuntimeError('planned class module failure')\n"
    prepared = prepare(parts, (prefix + CLASS + FACTORY).encode(), "bounded_class_failure")
    guard = ProviderEntryProfile(((prepared.source.module_code, "module", 1),), tmp_path / "breach")
    table, counters, lock = guard._table, guard._counts, guard._lock
    before_main, before_thread = sys.getprofile(), threading.getprofile()
    with guard.measure():
        with pytest.raises(RuntimeError, match="planned class module failure"):
            parts["classes"].execute(prepared)
        with pytest.raises(ValueError):
            parts["classes"].execute(prepared)
    assert guard._table is table and guard._counts is counters and guard._lock is lock
    assert guard.counts == dict(module=1)
    assert parts["effects"] == dict(module=0, factory=0, provider=0, failed=1)
    assert parts["events"] == []
    assert sys.getprofile() is before_main and threading.getprofile() is before_thread
    assert threading.active_count() == 1 and not (tmp_path / "breach").exists()
    with pytest.raises(ValueError):
        parts["classes"].execute(prepared)
    # An execution exception cannot refund an issued class reservation.
    with pytest.raises(ValueError):
        prepare(parts, name="bounded_class_failure_replacement")
    with pytest.raises(RuntimeError, match="cannot renew"):
        with guard.measure():
            raise AssertionError("guard unexpectedly renewed")
    save_work(tmp_path, guard, parts["effects"], 1, 0, 0)


def test_source_classes_refuses_cached_module_without_overwrite():
    parts = parts_for()
    name = "bounded_class_cached"
    cache = cast(dict[str, object], sys.modules)
    assert name not in cache
    for cached in (ModuleType(name), object()):
        cache[name] = cached
        try:
            refuse_prepare(parts, name=name)
            assert cache[name] is cached
        finally:
            del cache[name]
    for cached in (ModuleType(name), object()):
        prepared = prepare(parts, name=name)
        cache[name] = cached
        try:
            refuse_execute(parts, prepared)
            assert cache[name] is cached
        finally:
            del cache[name]
    zero(parts)


def test_source_classes_captures_original_class_under_same_guard(tmp_path):
    parts = parts_for()
    prepared = prepare(parts, name="bounded_class_positive")
    row = prepared.classes[0]
    root_name, root_code = prepared.source.root_codes[0]
    assert root_name == "factory" and row.slot == "Clock"
    assert all(
        type(code) is CodeType and len(code.co_consts) <= 512
        for code in (
            prepared.source.module_code,
            row.class_code,
            row.constructor_code,
            row.call_code,
            root_code,
        )
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
        (prepared.source.module_code, "module", 1),
        (row.class_code, "class_body", 1),
        (root_code, "factory", 1),
        (row.constructor_code, "constructor", 1),
        (row.call_code, "provider", 1),
    )
    assert len(entries) == 13 and sum(EXPECTED.values()) == 14
    guard = ProviderEntryProfile(entries, tmp_path / "breach")
    table, counters, lock = guard._table, guard._counts, guard._lock
    before_main, before_thread = sys.getprofile(), threading.getprofile()
    with guard.measure():
        loaded = parts["classes"].execute(prepared)
        binding = loaded.classes[0]
        namespace = loaded.source.namespace
        owner_type = namespace["Clock"]
        assert type(owner_type) is type and binding.owner_type is owner_type
        assert binding.code is row and binding.namespace is namespace
        assert binding.constructor.__code__ is row.constructor_code
        assert binding.raw_call.__code__ is row.call_code
        assert (
            binding.constructor.__globals__ is namespace
            and binding.raw_call.__globals__ is namespace
        )
        class_namespace = TYPE_DICT.__get__(owner_type, type)
        assert class_namespace["__dict__"] is binding.descriptor
        assert class_namespace["__init__"] is binding.constructor
        assert class_namespace["__call__"] is binding.raw_call
        parts["classes"].require_current(loaded)
        parts["classes"].require_binding(binding)
        root = loaded.source.roots[0]
        assert type(root) is FunctionType and root.__code__ is root_code
        assert root.__globals__ is namespace
        wrapper_owner = namespace["factory"]
        storage = parts["descriptor"].__get__(wrapper_owner, parts["wrapper"])
        wrapper = OwnerNamespaceBinding(
            wrapper_owner, parts["wrapper"], parts["descriptor"], storage
        )
        fixture_binding = FixtureRootBinding(
            namespace, "factory", wrapper, root, root_code, namespace
        )
        require_fixture_root_binding(fixture_binding)
        provider = root()
        assert type(provider) is owner_type
        assert provider() == 7
        parts["classes"].require_current(loaded)
        parts["classes"].require_binding(binding)
        require_fixture_root_binding(fixture_binding)
    assert guard._table is table and guard._counts is counters and guard._lock is lock
    assert guard.counts == EXPECTED and parts["effects"] == POSITIVE
    assert sys.getprofile() is before_main and threading.getprofile() is before_thread
    assert threading.active_count() == 1 and not (tmp_path / "breach").exists()
    with pytest.raises(ValueError):
        parts["classes"].execute(prepared)

    class ForeignLoaded(LoadedSourceClasses):
        def __getattribute__(self, name):
            parts["events"].append("loaded_getter")
            raise AssertionError

    class ForeignBinding(SourceClassBinding):
        def __getattribute__(self, name):
            parts["events"].append("binding_getter")
            raise AssertionError

    class HostileMeta(type):
        def __getattribute__(self, name):
            parts["events"].append("metaclass_getter")
            raise AssertionError

    class HostileOwner(metaclass=HostileMeta):
        pass

    for candidate in (
        object(),
        object.__new__(LoadedSourceClasses),
        object.__new__(ForeignLoaded),
        replace(loaded),
    ):
        with pytest.raises((TypeError, ValueError)):
            parts["classes"].require_current(candidate)
    for candidate in (
        object(),
        object.__new__(SourceClassBinding),
        object.__new__(ForeignBinding),
        replace(binding),
    ):
        with pytest.raises((TypeError, ValueError)):
            parts["classes"].require_binding(candidate)
    other = parts_for()
    with pytest.raises((TypeError, ValueError)):
        other["classes"].require_current(loaded)
    with pytest.raises((TypeError, ValueError)):
        other["classes"].require_binding(binding)
    zero(other)
    metadata = (
        (loaded, "prepared", replace(prepared)),
        (loaded, "source", replace(loaded.source)),
        (loaded, "classes", tuple(list(loaded.classes))),
        (loaded, "classes", (replace(binding),)),
        (loaded.source, "prepared", replace(prepared.source)),
        (loaded.source, "module", ModuleType("foreign_class_module")),
        (loaded.source, "namespace", namespace.copy()),
        (binding, "namespace", namespace.copy()),
        (binding, "code", replace(row)),
        (binding, "owner_type", HostileOwner),
        (binding, "descriptor", object()),
        (binding, "constructor", FunctionType(row.constructor_code, namespace)),
        (binding, "raw_call", FunctionType(row.call_code, namespace.copy())),
        (row, "call_code", row.call_code.replace()),
    )
    for owner, field, replacement in metadata:
        original = getattr(owner, field)
        object.__setattr__(owner, field, replacement)
        try:
            refuse_current(parts, loaded, binding)
        finally:
            object.__setattr__(owner, field, original)
        parts["classes"].require_current(loaded)
        parts["classes"].require_binding(binding)
    constructor, raw_call = binding.constructor, binding.raw_call
    for function in (constructor, raw_call):
        code = function.__code__
        clone = code.replace()
        assert clone == code and clone is not code
        function.__code__ = clone
        try:
            refuse_current(parts, loaded, binding)
        finally:
            function.__code__ = code
        for field, replacement in (("__defaults__", (None,)), ("__kwdefaults__", {"value": None})):
            original = getattr(function, field)
            object.__setattr__(function, field, replacement)
            try:
                refuse_current(parts, loaded, binding)
            finally:
                object.__setattr__(function, field, original)
    for field, original in (("__init__", constructor), ("__call__", raw_call)):
        for replacement in (
            FunctionType(original.__code__, namespace),
            FunctionType(original.__code__.replace(), namespace),
            FunctionType(original.__code__, namespace.copy()),
            property(lambda self: parts["events"].append("property_getter")),
        ):
            type.__setattr__(owner_type, field, replacement)
            try:
                refuse_current(parts, loaded, binding)
            finally:
                type.__setattr__(owner_type, field, original)
    # Native closure slots are read-only; a foreign closure-bearing method stays uncalled.
    cell = (lambda value: lambda: value)(None).__closure__
    assert cell is not None
    closure_code = row.call_code.replace(co_freevars=("retained",))
    foreign_closure = FunctionType(closure_code, namespace, closure=cell)
    type.__setattr__(owner_type, "__call__", foreign_closure)
    try:
        refuse_current(parts, loaded, binding)
    finally:
        type.__setattr__(owner_type, "__call__", raw_call)
    coherent = type(
        "Clock",
        (object,),
        {
            "__init__": FunctionType(row.constructor_code, namespace),
            "__call__": FunctionType(row.call_code, namespace),
        },
    )
    for replacement in (HostileOwner, coherent, object(), None):
        namespace["Clock"] = replacement
        try:
            refuse_current(parts, loaded, binding)
        finally:
            namespace["Clock"] = owner_type
    del namespace["Clock"]
    try:
        refuse_current(parts, loaded, binding)
    finally:
        namespace["Clock"] = owner_type
    original_bases = owner_type.__bases__
    replacement_bases = tuple([object])
    assert replacement_bases is not original_bases
    type.__setattr__(owner_type, "__bases__", replacement_bases)
    try:
        refuse_current(parts, loaded, binding)
    finally:
        type.__setattr__(owner_type, "__bases__", original_bases)
    for field, value in (
        ("added", object()),
        ("k" * 129, None),
        ("added_property", property(lambda self: parts["events"].append("property_getter"))),
    ):
        type.__setattr__(owner_type, field, value)
        try:
            refuse_current(parts, loaded, binding)
        finally:
            type.__delattr__(owner_type, field)
    with pytest.raises(AttributeError, match="not writable"):
        type.__setattr__(
            owner_type,
            "__dict__",
            property(lambda self: parts["events"].append("descriptor_getter")),
        )
    assert TYPE_DICT.__get__(owner_type, type)["__dict__"] is binding.descriptor
    foreign_keys = cast(dict[object, object], namespace)
    foreign_keys[3] = None
    try:
        refuse_current(parts, loaded, binding)
    finally:
        del foreign_keys[3]
    fake = ModuleType(prepared.source.module_name)
    sys.modules[prepared.source.module_name] = fake
    try:
        refuse_current(parts, loaded, binding)
        assert sys.modules[prepared.source.module_name] is fake
    finally:
        del sys.modules[prepared.source.module_name]
    parts["classes"].require_current(loaded)
    parts["classes"].require_binding(binding)
    require_fixture_root_binding(fixture_binding)
    assert namespace["Clock"] is owner_type and namespace["factory"] is wrapper_owner
    assert constructor.__code__ is row.constructor_code and raw_call.__code__ is row.call_code
    assert guard.counts == EXPECTED and parts["effects"] == POSITIVE and parts["events"] == []
    with pytest.raises(RuntimeError, match="cannot renew"):
        with guard.measure():
            raise AssertionError("guard unexpectedly renewed")
    save_work(tmp_path, guard, parts["effects"], 14, 1, 1)
