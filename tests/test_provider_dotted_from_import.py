"""Finite public dotted-FROM controls; no package discovery or domain imports."""

import builtins
from hashlib import sha256
import json
from pathlib import Path
import sys
import threading
from types import BuiltinFunctionType, CodeType, FunctionType, GetSetDescriptorType, ModuleType
from typing import Any, Callable, cast

import pytest

from provider_code_path import pin_provider_code_path, require_provider_callback_code
from provider_entry_profile import ProviderEntryProfile
from provider_source_loader import ApprovedSourceImport, FixtureSourceLoader


DOT = "bounded.pkg.values"
ORIGINAL_CALLABLE = cast(BuiltinFunctionType, builtins.callable)
MODULE_DICT = cast(
    GetSetDescriptorType, type.__dict__["__dict__"].__get__(ModuleType, type)["__dict__"]
)
PREFIX = b"""from bounded.pkg.values import counts, marker, value
counts['module'] += 1
"""
TAIL = b"""def factory():
    counts['factory'] += 1
    return value
"""
SOURCE = PREFIX + TAIL
BASE = b"def factory():\n    return 7\n"
POSITIVE = b"""from bounded.pkg.values import counts, marker as selected, value
counts['module'] += 1
def factory():
    counts['factory'] += 1
    def provider():
        counts['provider'] += 1
        return callable(selected)
    return provider
"""


class Case:
    def __init__(self) -> None:
        self.effects = dict(module=0, factory=0, provider=0)
        self.events: list[str] = []
        self.operations: list[str] = []
        self.module = ModuleType("independent_declared_origin")
        self.namespace = MODULE_DICT.__get__(self.module, ModuleType)
        self.marker = object()
        self.namespace.update(counts=self.effects, marker=self.marker, value=7)
        self.setup = dict(
            cases=1, modules=1, approval_rows=0, hostile_classes=0, hostile_instances=0
        )
        self.module_bodies = 0
        self.prior = (sys.getprofile(), threading.getprofile())

    def row(self, name: Any = DOT, module: Any = None) -> ApprovedSourceImport:
        self.setup["approval_rows"] += 1
        return ApprovedSourceImport(name, self.module if module is None else module)

    def op(
        self,
        kind: str,
        action: Callable[[], Any],
        *,
        refused: bool = False,
        errors: tuple[type[BaseException], ...] = (TypeError, ValueError),
    ) -> Any:
        if refused:
            try:
                action()
            except errors:
                self.operations.append(kind + ":refuse")
                return None
            raise AssertionError("required public import refusal did not occur")
        result = action()
        self.operations.append(kind + ":accept")
        return result

    def loader(self, imports: Any = None, capacity: Any = 16, **options: Any) -> Any:
        rows = (self.row(),) if imports is None else imports
        return self.op("loader", lambda: FixtureSourceLoader(rows, capacity), **options)

    def prepare(self, loader: Any, source: bytes = SOURCE, **options: Any) -> Any:
        name = "dotted_control_" + str(len(self.operations))
        return self.op(
            "prepare",
            lambda: loader.prepare(
                source, sha256(source).hexdigest(), name, "dotted.py", ("factory",)
            ),
            **options,
        )

    def execute(self, loader: Any, prepared: Any, **options: Any) -> Any:
        loaded = self.op("execute", lambda: loader.execute(prepared), **options)
        if not options.get("refused", False):
            self.module_bodies += 1
        return loaded

    def current(self, loader: Any, loaded: Any, **options: Any) -> None:
        self.op("current", lambda: loader.require_current(loaded), **options)

    def runtime(self, loaded: Any, name: Any, names: Any, level: Any = 0, **options: Any) -> Any:
        importer = loaded.namespace["__builtins__"]["__import__"]
        return self.op(
            "import",
            lambda: importer(name, None, None, names, level),
            errors=(ImportError,),
            **options,
        )

    def write(
        self,
        directory: Path,
        name: str,
        *,
        registered: dict[str, int] | None = None,
        results: list[Any] | None = None,
        renewal: int = 0,
    ) -> None:
        assert self.events == []
        assert sys.getprofile() is self.prior[0] and threading.getprofile() is self.prior[1]
        assert threading.active_count() == 1 and builtins.callable is ORIGINAL_CALLABLE
        assert not (directory / "dotted-breach.json").exists()
        entries = {} if registered is None else registered
        record = dict(
            test_name=name,
            operations=self.operations,
            source_effects=self.effects,
            successful_preparations=self.operations.count("prepare:accept"),
            source_module_bodies=self.module_bodies,
            fixture_setup=self.setup,
            hostile_events=self.events,
            registered_counts=entries,
            registered_attempts=sum(entries.values()),
            registered_python_bodies=sum(
                count for label, count in entries.items() if label != "native_callable"
            ),
            results=[] if results is None else results,
            renewal_refusals=renewal,
            restored=True,
            workers=0,
            joined=True,
            receipt_exists=False,
        )
        (directory / "dotted-import-work.json").write_text(json.dumps(record), encoding="utf8")


def test_dotted_executes_aliases_under_original_guard(tmp_path):
    case = Case()
    row = case.row()
    loader = case.op(
        "loader", lambda: FixtureSourceLoader((row,), callable_builtin=ORIGINAL_CALLABLE)
    )
    prepared = case.prepare(loader, POSITIVE)
    root_code = prepared.root_codes[0][1]
    assert len(root_code.co_consts) <= 512
    matches = [
        (index, token)
        for index, token in enumerate(root_code.co_consts)
        if type(token) is CodeType and token.co_name == "provider"
    ]
    assert len(matches) == 1
    index, provider_code = matches[0]
    guard = ProviderEntryProfile(
        (
            (prepared.module_code, "module", 1),
            (root_code, "factory", 1),
            (provider_code, "provider", 1),
            (ORIGINAL_CALLABLE, "native_callable", 1),
        ),
        tmp_path / "dotted-breach.json",
    )
    assert all(name not in sys.modules for name in ("bounded", "bounded.pkg", DOT))
    with guard.measure():
        loaded = case.execute(loader, prepared)
        assert loaded.namespace["selected"] is case.marker
        assert loaded.namespace["counts"] is case.effects
        assert loaded.namespace["value"] is case.namespace["value"]
        root = cast(FunctionType, loaded.roots[0])
        assert root.__code__ is root_code and root.__globals__ is loaded.namespace
        path = pin_provider_code_path(root, root_code, (index,))
        provider = root()
        require_provider_callback_code(path, provider)
        result = provider()
        assert type(result) is bool and result is False
        case.current(loader, loaded)
    case.current(loader, loaded)
    case.execute(loader, prepared, refused=True)
    with pytest.raises(RuntimeError, match="^entry profile cannot renew its allowance$"):
        with guard.measure():
            pytest.fail("guard allowance renewed")
    case.operations.append("renewal:refuse")
    assert guard.counts == dict(module=1, factory=1, provider=1, native_callable=1)
    assert all(name not in sys.modules for name in ("bounded", "bounded.pkg", DOT))
    case.write(
        tmp_path,
        "test_dotted_executes_aliases_under_original_guard",
        registered=guard.counts,
        results=[result],
        renewal=1,
    )


def test_dotted_preserves_empty_and_simple_approvals(tmp_path):
    case = Case()
    empty = case.loader(())
    case.prepare(empty, refused=True)
    prepared = case.prepare(empty, BASE)
    loaded = case.execute(empty, prepared)
    assert "callable" not in loaded.namespace["__builtins__"]
    case.runtime(loaded, DOT, ("marker",), refused=True)
    case.runtime(loaded, "bounded", (), refused=True)
    simple = case.loader((case.row("bounded"),))
    case.prepare(simple, refused=True)
    valid = case.prepare(simple, SOURCE.replace(DOT.encode(), b"bounded"))
    simple_loaded = case.execute(simple, valid)
    case.current(simple, simple_loaded)
    for names in (None, (), ("marker",)):
        assert case.runtime(simple_loaded, "bounded", names) is case.module
    assert case.effects == dict(module=1, factory=0, provider=0)
    case.write(tmp_path, "test_dotted_preserves_empty_and_simple_approvals")


def test_dotted_rejects_foreign_and_malformed_paths(tmp_path):
    case = Case()

    class ForeignPath(str):
        def __getattribute__(self, name):
            case.events.append("path_get")
            raise AssertionError

        def __len__(self):
            case.events.append("path_len")
            raise AssertionError

    class Hostile:
        def __str__(self):
            case.events.append("path_str")
            raise AssertionError

        def __getattribute__(self, name):
            case.events.append("object_get")
            raise AssertionError

    class ForeignRow(ApprovedSourceImport):
        def __getattribute__(self, name):
            case.events.append("row_get")
            raise AssertionError

    bad = (
        "",
        ".a",
        "a.",
        "a..b",
        "a-b",
        "é.a",
        "a.__hidden",
        "a." + "x" * 127,
        ".".join("a" for _ in range(17)),
        None,
        1,
        b"a.b",
        object(),
        ForeignPath("a.b"),
        object.__new__(Hostile),
    )
    case.setup.update(hostile_classes=3, hostile_instances=3)
    for candidate in bad:
        case.loader((case.row(candidate),), refused=True)
    foreign: Any = object.__new__(ForeignRow)
    case.setup["approval_rows"] += 1
    case.loader((foreign,), refused=True)
    for name in (".".join("a" for _ in range(16)), "a." + "x" * 126):
        assert len(name) <= 128
        loader = case.loader((case.row(name),))
        case.prepare(loader, SOURCE.replace(DOT.encode(), name.encode()))
    assert case.effects == dict(module=0, factory=0, provider=0)
    case.write(tmp_path, "test_dotted_rejects_foreign_and_malformed_paths")


def test_dotted_rechecks_approval_rows_and_native_module(tmp_path):
    case = Case()
    row = case.row()
    loader = case.loader((row,))
    first, pending = case.prepare(loader), case.prepare(loader)
    loaded = case.execute(loader, first)
    case.current(loader, loaded)
    name = row.name
    equal_name = name.encode().decode()
    assert equal_name == name and equal_name is not name
    other = ModuleType("other")
    case.setup["modules"] += 1
    for field, value, original in (("name", equal_name, name), ("module", other, case.module)):
        object.__setattr__(row, field, value)
        try:
            case.prepare(loader, refused=True)
            case.execute(loader, pending, refused=True)
            case.current(loader, loaded, refused=True)
        finally:
            object.__setattr__(row, field, original)
        case.current(loader, loaded)
    case.op(
        "module_dict",
        lambda: object.__setattr__(case.module, "__dict__", case.namespace.copy()),
        refused=True,
        errors=(AttributeError,),
    )
    assert MODULE_DICT.__get__(case.module, ModuleType) is case.namespace
    case.current(loader, loaded)

    class ForeignModule(ModuleType):
        def __getattribute__(self, name):
            case.events.append("module_get")
            raise AssertionError

    foreign = cast(Any, ModuleType.__new__)(ForeignModule, "foreign")
    case.setup.update(hostile_classes=1, hostile_instances=1)
    case.setup["modules"] += 1
    case.loader((case.row(module=foreign),), refused=True)
    case.loader((case.row(module=object()),), refused=True)
    second_loaded = case.execute(loader, pending)
    case.current(loader, second_loaded)
    case.write(tmp_path, "test_dotted_rechecks_approval_rows_and_native_module")


def test_dotted_rechecks_namespace_keys_and_imported_slots(tmp_path):
    case = Case()
    loader = case.loader()
    first, pending = case.prepare(loader), case.prepare(loader)
    loaded = case.execute(loader, first)
    case.current(loader, loaded)
    case.namespace["marker"] = object()
    try:
        assert loaded.namespace["marker"] is case.marker
        case.prepare(loader, refused=True)
        case.execute(loader, pending, refused=True)
        case.current(loader, loaded, refused=True)
    finally:
        case.namespace["marker"] = case.marker
    case.current(loader, loaded)

    class HostileKey(str):
        __hash__ = str.__hash__

        def __len__(self):
            case.events.append("key_len")
            raise AssertionError

        def __eq__(self, other):
            case.events.append("key_eq")
            raise AssertionError

    case.setup.update(hostile_classes=1, hostile_instances=1)
    for key in (object(), "k" * 129, HostileKey("dotted_key_control")):
        case.namespace[key] = None
        try:
            assert any(item is key for item in case.namespace)
            case.loader(refused=True)
            case.prepare(loader, refused=True)
            case.execute(loader, pending, refused=True)
            case.current(loader, loaded, refused=True)
        finally:
            del case.namespace[key]
        case.current(loader, loaded)
    second_loaded = case.execute(loader, pending)
    case.current(loader, second_loaded)
    case.write(tmp_path, "test_dotted_rechecks_namespace_keys_and_imported_slots")


def test_dotted_refuses_relative_star_missing_and_plain_syntax(tmp_path):
    case = Case()
    loader = case.loader()
    statements = (
        "from .bounded.pkg.values import marker",
        "from .. import marker",
        "from bounded.pkg.values import *",
        "from bounded.pkg.values import missing",
        "import bounded.pkg.values",
        "import bounded.pkg.values as alias",
        "from bounded.pkg.values import marker as globals",
        "from bounded.pkg.values import marker as factory",
        "from bounded.pkg.values import marker as __hidden",
    )
    for statement in statements:
        case.prepare(loader, PREFIX + statement.encode() + b"\n" + TAIL, refused=True)
        assert case.effects["module"] == 0
    prepared = case.prepare(loader)
    loaded = case.execute(loader, prepared)
    case.current(loader, loaded)
    for names, level in ((("*",), 0), (("missing",), 0), (("marker",), 1), ((), 0), (None, 0)):
        case.runtime(loaded, DOT, names, level, refused=True)
    case.write(tmp_path, "test_dotted_refuses_relative_star_missing_and_plain_syntax")


def test_dotted_preserves_table_source_and_path_limits(tmp_path):
    case = Case()
    rows = (case.row(), *(case.row("p" + str(index) + ".values") for index in range(15)))
    extra = case.row("extra.values")
    case.loader((*rows, extra), refused=True)
    case.loader((rows[0], rows[0]), refused=True)
    loader = case.loader(rows, capacity=1)
    maximum = SOURCE + b"\n#" + b"x" * (16384 - len(SOURCE) - 2)
    assert len(maximum) == 16384
    case.prepare(loader, maximum + b"x", refused=True)
    case.prepare(loader, b"", refused=True)
    wide_ast = b"x=0\n" * 1100 + SOURCE
    assert len(wide_ast) < 16384
    case.prepare(loader, wide_ast, refused=True)
    prepared = case.prepare(loader, maximum)
    case.prepare(loader, refused=True)
    loaded = case.execute(loader, prepared)
    case.execute(loader, prepared, refused=True)
    case.current(loader, loaded)
    case.prepare(loader, refused=True)
    case.write(tmp_path, "test_dotted_preserves_table_source_and_path_limits")


def test_dotted_runtime_importer_requires_native_from_inputs(tmp_path):
    case = Case()
    case.namespace.update(("v" + str(index), index) for index in range(16))
    loader = case.loader()
    prepared = case.prepare(loader)
    loaded = case.execute(loader, prepared)
    case.current(loader, loaded)
    names16 = tuple("v" + str(index) for index in range(16))
    for names in (("marker",), ("marker", "value"), names16):
        assert case.runtime(loaded, DOT, names) is case.module

    class ForeignPath(str):
        def __getattribute__(self, name):
            case.events.append("runtime_path_get")
            raise AssertionError

    class ForeignNames(tuple):
        def __len__(self):
            case.events.append("runtime_names_len")
            raise AssertionError

        def __iter__(self):
            case.events.append("runtime_names_iter")
            raise AssertionError

    class ForeignLevel:
        def __int__(self):
            case.events.append("runtime_level_int")
            raise AssertionError

        def __eq__(self, other):
            case.events.append("runtime_level_eq")
            raise AssertionError

    case.setup.update(hostile_classes=3, hostile_instances=3)
    bad: tuple[tuple[Any, Any, Any], ...] = (
        (ForeignPath(DOT), ("marker",), 0),
        (7, ("marker",), 0),
        (DOT, ["marker"], 0),
        (DOT, ForeignNames(("marker",)), 0),
        (DOT, ("missing",), 0),
        (DOT, ("*",), 0),
        (DOT, ("v0", "v0"), 0),
        (DOT, (*names16, "v16"), 0),
        (DOT, ("bad.name",), 0),
        (DOT, ("marker",), True),
        (DOT, ("marker",), 1),
        (DOT, ("marker",), ForeignLevel()),
    )
    for name, names, level in bad:
        case.runtime(loaded, name, names, level, refused=True)
    aliases = ", ".join(name + " as a" + str(index) for index, name in enumerate(names16))
    source16 = ("from " + DOT + " import " + aliases + "\n").encode() + BASE
    alias_prepared = case.prepare(loader, source16)
    source17 = (
        PREFIX + ("from " + DOT + " import " + ", ".join((*names16, "v16")) + "\n").encode() + TAIL
    )
    case.prepare(loader, source17, refused=True)
    alias_loaded = case.execute(loader, alias_prepared)
    for index, name in enumerate(names16):
        assert alias_loaded.namespace["a" + str(index)] is case.namespace[name]
    case.current(loader, alias_loaded)
    case.write(tmp_path, "test_dotted_runtime_importer_requires_native_from_inputs")
