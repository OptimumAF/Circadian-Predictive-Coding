from dataclasses import replace
from hashlib import sha256
import json
import sys
import threading
from types import CodeType, FunctionType, GetSetDescriptorType, ModuleType
from typing import cast

import pytest

from provider_code_path import pin_provider_code_path, require_provider_callback_code
from provider_entry_profile import ProviderEntryProfile
from provider_fixture_binding import FixtureRootBinding, require_fixture_root_binding
from provider_owner_namespace_profile import OwnerNamespaceBinding
from provider_source_loader import (
    ApprovedSourceImport,
    FixtureSourceLoader,
    LoadedFixtureSource,
    PreparedFixtureSource,
)


SOURCE = b"""import boundedapproved
from boundedapproved import marker
boundedapproved.counts['module'] += 1
def factory():
    boundedapproved.counts['factory'] += 1
    def provider():
        boundedapproved.counts['provider'] += 1
        return marker
    return provider
"""
ZERO = dict(module=0, factory=0, provider=0, failed=0)
POSITIVE = dict(module=1, factory=1, provider=1, failed=0)
_MODULE_DICT = cast(
    GetSetDescriptorType, type.__dict__["__dict__"].__get__(ModuleType, type)["__dict__"]
)


def make_case(max_sources=16):
    dependency = ModuleType("boundedapproved")
    namespace = _MODULE_DICT.__get__(dependency, ModuleType)
    counts = ZERO.copy()
    namespace.update(counts=counts, marker=7, failure=RuntimeError("planned module failure"))
    return dict(
        dependency=dependency,
        namespace=namespace,
        counts=counts,
        events=[],
        loader=FixtureSourceLoader(
            (ApprovedSourceImport("boundedapproved", dependency),), max_sources
        ),
    )


def prepare(parts, source=SOURCE, name="bounded_source_control", roots=("factory",)):
    assert name not in sys.modules
    return parts["loader"].prepare(source, sha256(source).hexdigest(), name, "standin.py", roots)


def require_zero(parts):
    assert parts["counts"] == ZERO and parts["events"] == []


def refuse_prepare(parts, source=SOURCE, **changes):
    args = dict(
        source=source,
        expected_sha256=sha256(source).hexdigest(),
        module_name="bounded_source_control",
        filename="standin.py",
        roots=("factory",),
    )
    args.update(changes)
    with pytest.raises((TypeError, ValueError)):
        parts["loader"].prepare(**args)
    require_zero(parts)


def refuse_execute(parts, prepared):
    with pytest.raises((TypeError, ValueError)):
        parts["loader"].execute(prepared)
    require_zero(parts)


def refuse_loader(parts, imports, capacity=16):
    with pytest.raises((TypeError, ValueError)):
        FixtureSourceLoader(imports, capacity)
    require_zero(parts)


def test_should_refuse_wrong_digest_before_module_effects():
    parts = make_case()
    refuse_prepare(parts, expected_sha256="0" * 64)
    refuse_prepare(parts, b"def(", expected_sha256="0" * 64)
    assert sha256(SOURCE).hexdigest() != "0" * 64


@pytest.mark.parametrize(
    "denied",
    (
        "import deniedmodule",
        "from deniedmodule import value",
        "def unused():\n    import deniedmodule",
        "from . import value",
        "from boundedapproved import *",
        "import boundedapproved.child",
    ),
)
def test_should_preflight_denied_imports_before_earlier_effects(denied):
    parts = make_case()
    source = (
        "import boundedapproved\nboundedapproved.counts['module'] += 1\n"
        + denied
        + "\ndef factory():\n    return None\n"
    ).encode("utf8")
    refuse_prepare(parts, source)


def test_should_refuse_malformed_native_inputs_without_hooks():
    parts = make_case()
    events = parts["events"]

    class ForeignBytes(bytes):
        def __len__(self):
            events.append("bytes_length")
            raise AssertionError

    class ForeignTuple(tuple):
        def __len__(self):
            events.append("tuple_length")
            raise AssertionError

    class ForeignModule(ModuleType):
        def __getattribute__(self, name):
            events.append("module_getter")
            raise AssertionError

    class ForeignImport(ApprovedSourceImport):
        def __getattribute__(self, name):
            events.append("import_getter")
            raise AssertionError

    malformed: tuple[dict[str, object], ...] = (
        dict(source=ForeignBytes(SOURCE)),
        dict(source=bytearray(SOURCE)),
        dict(source=b""),
        dict(expected_sha256="F" * 64),
        dict(expected_sha256="f" * 63),
        dict(module_name="__private"),
        dict(module_name="bad.name"),
        dict(module_name=3),
        dict(filename=""),
        dict(filename="x" * 257),
        dict(filename=object()),
        dict(roots=[]),
        dict(roots=ForeignTuple(("factory",))),
        dict(roots=()),
        dict(roots=("factory", "factory")),
        dict(roots=("bad.name",)),
    )
    for changes in malformed:
        refuse_prepare(parts, **changes)
    invalid_imports: tuple[object, ...] = (
        [],
        ForeignTuple(()),
        (object(),),
        (object.__new__(ApprovedSourceImport),),
        (object.__new__(ForeignImport),),
        (ApprovedSourceImport("bad..name", parts["dependency"]),),
        (ApprovedSourceImport("boundedapproved", cast(ModuleType, object())),),
        (ApprovedSourceImport("boundedapproved", ForeignModule("foreign")),),
        (ApprovedSourceImport("boundedapproved", parts["dependency"]),) * 2,
    )
    for imports in invalid_imports:
        refuse_loader(parts, imports)
    for capacity in (False, 0, -1, 17, 1.0):
        refuse_loader(parts, (), capacity)
    require_zero(parts)


def test_should_enforce_source_ast_root_import_and_constant_bounds():
    parts = make_case()
    plain = b"def factory():\n    return 0\n"
    padded = plain + b"#" + b"x" * (16384 - len(plain) - 1)
    assert len(padded) == 16384
    prepare(parts, padded, "bounded_source_bytes")
    refuse_prepare(parts, padded + b"x")
    node_boundary = b"0\n" * 2045 + b"pass\n" + plain
    prepare(parts, node_boundary, "bounded_source_nodes")
    refuse_prepare(parts, b"0\n" * 2046 + plain)
    root_names = tuple(f"root{index}" for index in range(16))
    roots_source = "".join(f"def {name}():\n    return None\n" for name in root_names).encode()
    prepare(parts, roots_source, "bounded_source_roots", root_names)
    refuse_prepare(parts, roots_source, roots=root_names + ("extra",))
    constant_boundary = "".join(f"v{i}='c{i}'\n" for i in range(510)).encode() + plain
    boundary = prepare(parts, constant_boundary, "bounded_source_constants")
    assert len(boundary.module_code.co_consts) == 512
    refuse_prepare(parts, b"v510='c510'\n" + constant_boundary)
    imports = tuple(
        ApprovedSourceImport(f"dep{index}", ModuleType(f"dep{index}")) for index in range(16)
    )
    FixtureSourceLoader(imports)
    with pytest.raises(ValueError):
        FixtureSourceLoader(imports + (ApprovedSourceImport("extra", ModuleType("extra")),))
    from_parts = make_case()
    from_parts["namespace"].update({f"attr{index}": index for index in range(17)})
    from_parts["loader"] = FixtureSourceLoader(
        (ApprovedSourceImport("boundedapproved", from_parts["dependency"]),)
    )
    allowed_from = (
        b"from boundedapproved import "
        + ",".join(f"attr{index}" for index in range(16)).encode()
        + b"\n"
        + plain
    )
    prepare(from_parts, allowed_from, "bounded_source_from")
    refuse_prepare(from_parts, allowed_from.replace(b"attr15\n", b"attr15,attr16\n"))
    single = make_case(1)
    prepare(single, plain, "bounded_source_single")
    refuse_prepare(single, plain, module_name="bounded_source_overflow")
    refuse_prepare(parts, b"# coding: utf8\n\xff\n" + plain)
    prepare(parts, b"# coding: latin-1\n# \xc3\xa9\n" + plain, "bounded_source_cookie")
    require_zero(parts)
    require_zero(single)
    require_zero(from_parts)


def test_should_refuse_dynamic_access_and_all_selected_root_rebindings():
    parts = make_case()
    names = (
        "__builtins__ __import__ eval exec compile globals locals vars getattr setattr delattr open"
    ).split()
    declaration = "def factory():\n    return None\n"
    binding_parts = make_case()
    binding_parts["namespace"].update(dict.fromkeys(names))
    binding_parts["loader"] = FixtureSourceLoader(
        (ApprovedSourceImport("boundedapproved", binding_parts["dependency"]),)
    )
    for name in names:
        refuse_prepare(parts, f"def factory():\n    return {name}\n".encode())
        for statement in (
            f"import boundedapproved as {name}",
            f"from boundedapproved import marker as {name}",
            f"from boundedapproved import {name}",
        ):
            refuse_prepare(binding_parts, (statement + "\n" + declaration).encode())
    for name in "__dict__ __class__ __globals__ __code__ __closure__ __getattribute__".split():
        refuse_prepare(parts, f"def factory():\n    return value.{name}\n".encode())
    for binding in (
        "factory = 0",
        "del factory",
        "import boundedapproved as factory",
        "from boundedapproved import marker as factory",
        "def other(factory):\n    return None",
        "try:\n    pass\nexcept ValueError as factory:\n    pass",
        "def other():\n    global factory\n    return None",
        "def other():\n    nonlocal factory\n    return None",
        "match 0:\n    case factory:\n        pass",
        "match []:\n    case [*factory]:\n        pass",
        "match {}:\n    case {**factory}:\n        pass",
        "class factory:\n    pass",
        declaration,
        "def other():\n    def factory():\n        return None\n",
    ):
        refuse_prepare(parts, (declaration + binding + "\n").encode())
    for declaration in (
        "@object\ndef factory():\n    return None\n",
        "def factory(value=0):\n    return None\n",
        "def factory(*, value=0):\n    return None\n",
        "async def factory():\n    return None\n",
    ):
        refuse_prepare(parts, declaration.encode())


def test_should_refuse_foreign_incomplete_and_mutated_prepared_rows_before_execution():
    parts = make_case()
    prepared = prepare(parts)
    events = parts["events"]

    class ForeignPrepared(PreparedFixtureSource):
        def __getattribute__(self, name):
            events.append("prepared_getter")
            raise AssertionError

    for candidate in (
        object(),
        object.__new__(PreparedFixtureSource),
        object.__new__(ForeignPrepared),
        replace(prepared),
    ):
        refuse_execute(parts, candidate)
    other_owner = make_case()
    refuse_execute(other_owner, prepared)
    changes = (
        ("module_name", "bounded_changed_name"),
        ("filename", "other.py"),
        ("source_sha256", "0" * 64),
        ("module_code", prepared.module_code.replace()),
        ("root_codes", tuple(list(prepared.root_codes))),
        ("root_codes", (("factory", prepared.root_codes[0][1].replace()),)),
    )
    for name, replacement in changes:
        original = getattr(prepared, name)
        assert replacement is not original
        object.__setattr__(prepared, name, replacement)
        refuse_execute(parts, prepared)
        object.__setattr__(prepared, name, original)
    require_zero(other_owner)


def test_should_refuse_cached_module_at_prepare_without_overwriting():
    parts = make_case()
    name = "bounded_source_cached_prepare"
    assert name not in sys.modules
    fake = ModuleType(name)
    sys.modules[name] = fake
    try:
        refuse_prepare(parts, module_name=name)
        assert sys.modules[name] is fake
    finally:
        del sys.modules[name]


def test_should_refuse_cache_or_dependency_changes_before_execute_without_effects():
    parts = make_case()
    prepared = prepare(parts)
    fake = ModuleType(prepared.module_name)
    sys.modules[prepared.module_name] = fake
    try:
        refuse_execute(parts, prepared)
        assert sys.modules[prepared.module_name] is fake
    finally:
        del sys.modules[prepared.module_name]
    original = parts["namespace"]["counts"]
    parts["namespace"]["counts"] = original.copy()
    refuse_execute(parts, prepared)
    parts["namespace"]["counts"] = original
    events = parts["events"]
    armed = False
    collision = hash("counts")

    class HostileKey:
        def __hash__(self):
            if armed:
                events.append("key_hash")
                raise AssertionError
            return collision

        def __eq__(self, other):
            if armed:
                events.append("key_equality")
                raise AssertionError
            return False

    parts["namespace"][HostileKey()] = None
    armed = True
    refuse_execute(parts, prepared)


def require_loaded_refused(parts, loaded):
    with pytest.raises((TypeError, ValueError)):
        parts["loader"].require_current(loaded)
    assert parts["events"] == []


def test_should_execute_exact_prepared_module_factory_and_provider_under_one_guard(tmp_path):
    parts = make_case()
    prepared = prepare(parts)
    root_name, root_code = prepared.root_codes[0]
    assert root_name == "factory"
    index, provider_code = next(
        (index, token) for index, token in enumerate(root_code.co_consts) if type(token) is CodeType
    )
    guard = ProviderEntryProfile(
        (
            (prepared.module_code, "module", 1),
            (root_code, "factory", 1),
            (provider_code, "provider", 1),
        ),
        tmp_path / "breach",
    )
    table, counters, lock = guard._table, guard._counts, guard._lock
    before_main, before_thread = sys.getprofile(), threading.getprofile()
    assert "boundedapproved" not in sys.modules
    fake_dependency = ModuleType("boundedapproved")
    sys.modules["boundedapproved"] = fake_dependency
    try:
        with guard.measure():
            loaded = parts["loader"].execute(prepared)
            root = loaded.roots[0]
            assert root.__code__ is root_code and root.__globals__ is loaded.namespace
            assert loaded.namespace["boundedapproved"] is parts["dependency"]
            path = pin_provider_code_path(root, root_code, (index,))

            class Wrapper:
                pass

            owner = Wrapper()
            descriptor = cast(GetSetDescriptorType, Wrapper.__dict__["__dict__"])
            wrapper_namespace = descriptor.__get__(owner, Wrapper)
            wrapper_namespace.update(_fixture_function=root, __wrapped__=root)
            loaded.namespace["fixture_root"] = owner
            wrapper = OwnerNamespaceBinding(owner, Wrapper, descriptor, wrapper_namespace)
            binding = FixtureRootBinding(
                loaded.namespace, "fixture_root", wrapper, root, root_code, loaded.namespace
            )
            require_fixture_root_binding(binding)
            provider = root()
            require_provider_callback_code(path, provider)
            assert provider() == 7
            require_fixture_root_binding(binding)
            parts["loader"].require_current(loaded)
        assert sys.modules["boundedapproved"] is fake_dependency
    finally:
        del sys.modules["boundedapproved"]
    assert guard._table is table and guard._counts is counters and guard._lock is lock
    assert guard.counts == dict(module=1, factory=1, provider=1)
    assert parts["counts"] == POSITIVE and parts["events"] == []
    assert sys.getprofile() is before_main and threading.getprofile() is before_thread
    assert threading.active_count() == 1 and not (tmp_path / "breach").exists()
    parts["loader"].require_current(loaded)
    with pytest.raises(ValueError):
        parts["loader"].execute(prepared)

    class ForeignLoaded(LoadedFixtureSource):
        def __getattribute__(self, name):
            parts["events"].append("loaded_getter")
            raise AssertionError

    for candidate in (
        object(),
        object.__new__(LoadedFixtureSource),
        object.__new__(ForeignLoaded),
        replace(loaded),
    ):
        require_loaded_refused(parts, candidate)
    for name, replacement in (
        ("prepared", replace(prepared)),
        ("module", ModuleType("foreign")),
        ("namespace", loaded.namespace.copy()),
        ("roots", tuple(list(loaded.roots))),
    ):
        original = getattr(loaded, name)
        object.__setattr__(loaded, name, replacement)
        require_loaded_refused(parts, loaded)
        object.__setattr__(loaded, name, original)
        parts["loader"].require_current(loaded)
    fake_module = ModuleType(prepared.module_name)
    sys.modules[prepared.module_name] = fake_module
    try:
        require_loaded_refused(parts, loaded)
        assert sys.modules[prepared.module_name] is fake_module
    finally:
        del sys.modules[prepared.module_name]
    namespace = loaded.namespace
    root = loaded.roots[0]
    for foreign in (
        FunctionType(root_code, namespace),
        FunctionType(root_code, namespace.copy()),
        None,
    ):
        namespace["factory"] = foreign
        require_loaded_refused(parts, loaded)
        namespace["factory"] = root
        parts["loader"].require_current(loaded)
    del namespace["factory"]
    require_loaded_refused(parts, loaded)
    namespace["factory"] = root
    clone = root_code.replace()
    assert clone == root_code and clone is not root_code
    root.__code__ = clone
    require_loaded_refused(parts, loaded)
    root.__code__ = root_code
    module_descriptor = _MODULE_DICT
    with pytest.raises(AttributeError):
        object.__setattr__(loaded.module, "__dict__", namespace.copy())
    assert module_descriptor.__get__(loaded.module, ModuleType) is namespace
    parts["loader"].require_current(loaded)
    foreign_keys = cast(dict[object, object], namespace)
    foreign_keys[3] = None
    require_loaded_refused(parts, loaded)
    del foreign_keys[3]
    namespace["k" * 129] = None
    require_loaded_refused(parts, loaded)
    del namespace["k" * 129]
    parts["loader"].require_current(loaded)
    assert guard.counts == dict(module=1, factory=1, provider=1) and parts["counts"] == POSITIVE
    assert parts["events"] == [] and namespace["factory"] is root
    require_fixture_root_binding(binding)
    (tmp_path / "work.json").write_text(
        json.dumps(
            dict(
                bodies=3,
                module_calls=1,
                factory_calls=1,
                provider_calls=1,
                counts=guard.counts,
                side_effects=parts["counts"],
                workers=0,
                joined=True,
                restored=True,
            )
        ),
        encoding="utf8",
    )


def test_should_spend_prepared_execution_after_module_exception(tmp_path):
    parts = make_case()
    source = (
        b"import boundedapproved\ndef factory():\n    return None\n"
        b"boundedapproved.counts['failed'] += 1\nraise boundedapproved.failure\n"
    )
    prepared = prepare(parts, source, "bounded_source_failure")
    guard = ProviderEntryProfile(((prepared.module_code, "module", 1),), tmp_path / "breach")
    before_main, before_thread = sys.getprofile(), threading.getprofile()
    table, counters, lock = guard._table, guard._counts, guard._lock
    with guard.measure():
        with pytest.raises(RuntimeError, match="planned module failure"):
            parts["loader"].execute(prepared)
        with pytest.raises(ValueError):
            parts["loader"].execute(prepared)
    assert guard._table is table and guard._counts is counters and guard._lock is lock
    assert guard.counts == {"module": 1}
    assert parts["counts"] == dict(module=0, factory=0, provider=0, failed=1)
    assert parts["events"] == [] and prepared.module_name not in sys.modules
    assert sys.getprofile() is before_main and threading.getprofile() is before_thread
    assert threading.active_count() == 1 and not (tmp_path / "breach").exists()
    (tmp_path / "work-failure.json").write_text(
        json.dumps(
            dict(
                bodies=1,
                module_calls=1,
                factory_calls=0,
                provider_calls=0,
                counts=guard.counts,
                side_effects=parts["counts"],
                workers=0,
                joined=True,
                restored=True,
            )
        ),
        encoding="utf8",
    )
