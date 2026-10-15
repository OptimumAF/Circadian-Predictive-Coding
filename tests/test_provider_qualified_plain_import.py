"""Explicit native package-chain controls; no discovery or original fixtures."""

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
from provider_source_loader import (
    ApprovedSourceDecorator,
    ApprovedSourceImport,
    FixtureSourceLoader,
)


PATH = "declared.pkg.leaf"
ORIGINAL = cast(BuiltinFunctionType, builtins.callable)
MODULE_DICT = cast(
    GetSetDescriptorType, type.__dict__["__dict__"].__get__(ModuleType, type)["__dict__"]
)
BASE = b"def factory():\n    return 7\n"


def source_for(path: str) -> bytes:
    return (
        "import " + path + " as chosen\nfrom " + path + " import counts\n"
        "counts['module'] += 1\ndef factory():\n    return 7\n"
    ).encode()


POSITIVE = b"""import declared.pkg.leaf
import declared.pkg.leaf as chosen
import declared.pkg.leaf as first, declared.pkg.other as second
from declared.pkg import leaf as from_choice
from declared.pkg.leaf import counts, marker
counts['module'] += 1
def factory():
    counts['factory'] += 1
    def provider():
        counts['provider'] += 1
        return callable(marker)
    return provider
"""


class Case:
    def __init__(self) -> None:
        self.effects = dict(module=0, factory=0, provider=0)
        self.events: list[str] = []
        self.operations: list[str] = []
        self.setup = dict(
            cases=1,
            modules=0,
            approval_rows=0,
            classes=0,
            hostile_instances=0,
            decorator_rows=0,
            padding_keys=0,
        )
        self.marker = object()
        self.getter = self.module_getter
        self.rows, self.modules = self.chain(("declared", "pkg", "leaf"))
        self.root, self.parent, self.leaf = self.modules
        self.root_ns, self.parent_ns, self.leaf_ns = map(self.namespace, self.modules)
        self.module_bodies = 0
        self.prior = (sys.getprofile(), threading.getprofile())

    def module_getter(self, name):
        self.events.append("module_getter")
        raise AssertionError("module getter dispatched")

    def namespace(self, module: ModuleType) -> Any:
        return MODULE_DICT.__get__(module, ModuleType)

    def module(self, name: str) -> ModuleType:
        module = ModuleType(name)
        self.namespace(module)["__getattr__"] = self.getter
        self.setup["modules"] += 1
        return module

    def row(self, name: Any, module: Any) -> ApprovedSourceImport:
        self.setup["approval_rows"] += 1
        return ApprovedSourceImport(name, module)

    def chain(
        self, parts: tuple[str, ...]
    ) -> tuple[tuple[ApprovedSourceImport, ...], tuple[ModuleType, ...]]:
        modules = tuple(self.module(".".join(parts[: index + 1])) for index in range(len(parts)))
        for index, module in enumerate(modules[:-1]):
            self.namespace(module)[parts[index + 1]] = modules[index + 1]
        self.namespace(modules[-1]).update(counts=self.effects, marker=self.marker)
        rows = tuple(
            self.row(".".join(parts[: index + 1]), module) for index, module in enumerate(modules)
        )
        return rows, modules

    def named_rows(
        self, name: str, root: ModuleType | None = None
    ) -> tuple[ApprovedSourceImport, ...]:
        return (
            self.row(name, self.root if root is None else root),
            self.row(name + ".pkg", self.parent),
            self.row(name + ".pkg.leaf", self.leaf),
        )

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
            raise AssertionError("required public refusal did not occur")
        result = action()
        self.operations.append(kind + ":accept")
        return result

    def loader(self, rows: Any = None, capacity: Any = 16, *, enabled=False, **options: Any) -> Any:
        imports = self.rows if rows is None else rows
        return self.op(
            "loader",
            lambda: FixtureSourceLoader(
                imports, capacity, callable_builtin=ORIGINAL if enabled else None
            ),
            **options,
        )

    def prepare(self, loader: Any, source: bytes | None = None, **options: Any) -> Any:
        raw = source_for(PATH) if source is None else source
        name = "qualified_control_" + str(len(self.operations))
        return self.op(
            "prepare",
            lambda: loader.prepare(
                raw, sha256(raw).hexdigest(), name, "qualified.py", ("factory",)
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

    def runtime(
        self, loaded: Any, name: Any, names: Any = (), level: Any = 0, **options: Any
    ) -> Any:
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
        counts: dict[str, int] | None = None,
        results: list[Any] | None = None,
        windows=0,
        renewal=0,
        mutations=0,
        failed=0,
    ) -> None:
        assert self.events == [] and builtins.callable is ORIGINAL
        assert sys.getprofile() is self.prior[0] and threading.getprofile() is self.prior[1]
        assert threading.active_count() == 1 and not (directory / "qualified-breach.json").exists()
        entries = {} if counts is None else counts
        record = dict(
            test_name=name,
            operations=self.operations,
            source_effects=self.effects,
            source_module_bodies=self.module_bodies,
            successful_preparations=self.operations.count("prepare:accept"),
            fixture_setup=self.setup,
            hostile_events=self.events,
            registered_counts=entries,
            registered_attempts=sum(entries.values()),
            registered_python_bodies=sum(
                n for label, n in entries.items() if label != "native_callable"
            ),
            native_calls=entries.get("native_callable", 0),
            guard_windows=windows,
            renewal_refusals=renewal,
            prior_mutations=mutations,
            failed_module_entries=failed,
            results=[] if results is None else results,
            restored=True,
            workers=0,
            joined=True,
            receipt_exists=False,
        )
        (directory / "qualified-plain-work.json").write_text(json.dumps(record), encoding="utf8")


def test_qualified_plain_preserves_root_alias_from_and_multi_identity(tmp_path):
    case = Case()
    other = case.module("other_origin")
    case.parent_ns["other"] = other
    rows = (*case.rows, case.row("declared.pkg.other", other))
    loader = case.loader(rows, enabled=True)
    prepared = case.prepare(loader, POSITIVE)
    root_code = prepared.root_codes[0][1]
    matches = [
        (index, code)
        for index, code in enumerate(root_code.co_consts)
        if type(code) is CodeType and code.co_name == "provider"
    ]
    assert len(root_code.co_consts) <= 512 and len(matches) == 1
    index, provider_code = matches[0]
    guard = ProviderEntryProfile(
        (
            (prepared.module_code, "module", 1),
            (root_code, "factory", 1),
            (provider_code, "provider", 1),
            (ORIGINAL, "native_callable", 1),
        ),
        tmp_path / "qualified-breach.json",
    )
    assert all(name not in sys.modules for name in ("declared", "declared.pkg", PATH))
    with guard.measure():
        loaded = case.execute(loader, prepared)
        assert loaded.namespace["declared"] is case.root
        assert all(
            loaded.namespace[name] is case.leaf for name in ("chosen", "first", "from_choice")
        )
        assert loaded.namespace["second"] is other
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
            pytest.fail("guard renewed")
    case.operations.append("renewal:refuse")
    assert guard.counts == dict(module=1, factory=1, provider=1, native_callable=1)
    assert all(name not in sys.modules for name in ("declared", "declared.pkg", PATH))
    case.write(
        tmp_path,
        "test_qualified_plain_preserves_root_alias_from_and_multi_identity",
        counts=guard.counts,
        results=[result],
        windows=1,
        renewal=1,
    )


def test_qualified_plain_refuses_missing_authority_without_getters(tmp_path):
    case = Case()
    unrelated = case.module("unrelated")
    configurations = (
        "root-row",
        "parent-row",
        "leaf-row",
        "root-link",
        "leaf-link",
        "foreign-link",
        "unrelated-link",
    )
    for mode in configurations:
        rows = case.rows
        if mode.endswith("-row"):
            omitted = {"root-row": 0, "parent-row": 1, "leaf-row": 2}[mode]
            rows = tuple(row for index, row in enumerate(case.rows) if index != omitted)
        if mode == "root-link":
            del case.root_ns["pkg"]
        if mode == "leaf-link":
            del case.parent_ns["leaf"]
        if mode == "foreign-link":
            case.parent_ns["leaf"] = object()
        if mode == "unrelated-link":
            case.parent_ns["leaf"] = unrelated
        try:
            loader = case.loader(rows)
            case.prepare(loader, refused=True)
            prepared = case.prepare(loader, BASE)
            loaded = case.execute(loader, prepared)
            case.runtime(loaded, PATH, refused=True)
        finally:
            case.root_ns["pkg"] = case.parent
            case.parent_ns["leaf"] = case.leaf
        assert case.effects == dict(module=0, factory=0, provider=0)
    case.write(tmp_path, "test_qualified_plain_refuses_missing_authority_without_getters")


def test_qualified_plain_preserves_default_simple_and_from_only(tmp_path):
    case = Case()
    empty = case.loader(())
    prepared = case.prepare(empty, BASE)
    loaded = case.execute(empty, prepared)
    assert "callable" not in loaded.namespace["__builtins__"]
    case.runtime(loaded, PATH, None, refused=True)
    simple = case.loader((case.rows[0],))
    simple_prepared = case.prepare(simple, b"import declared\n" + BASE)
    simple_loaded = case.execute(simple, simple_prepared)
    assert simple_loaded.namespace["declared"] is case.root
    assert case.runtime(simple_loaded, "declared", None) is case.root
    from_only = case.loader((case.rows[2],))
    from_prepared = case.prepare(from_only, b"from declared.pkg.leaf import marker\n" + BASE)
    from_loaded = case.execute(from_only, from_prepared)
    assert from_loaded.namespace["marker"] is case.marker
    assert case.runtime(from_loaded, PATH, ("marker",)) is case.leaf
    case.runtime(from_loaded, PATH, (), refused=True)
    case.current(from_only, from_loaded)
    case.write(tmp_path, "test_qualified_plain_preserves_default_simple_and_from_only")


def test_qualified_plain_rechecks_original_rows_modules_keys_and_nested_slots(tmp_path):
    case = Case()
    loader = case.loader()
    first, pending = case.prepare(loader), case.prepare(loader)
    loaded = case.execute(loader, first)
    case.current(loader, loaded)
    other = case.module("foreign_leaf")
    old_name = case.rows[1].name
    equal_name = old_name.encode().decode()
    assert equal_name == old_name and equal_name is not old_name
    changes = (
        (
            lambda: object.__setattr__(case.rows[1], "name", equal_name),
            lambda: object.__setattr__(case.rows[1], "name", old_name),
        ),
        (
            lambda: object.__setattr__(case.rows[2], "module", other),
            lambda: object.__setattr__(case.rows[2], "module", case.leaf),
        ),
        (lambda: case.root_ns.update(pkg=object()), lambda: case.root_ns.update(pkg=case.parent)),
        (lambda: case.parent_ns.update(leaf=other), lambda: case.parent_ns.update(leaf=case.leaf)),
        (
            lambda: case.leaf_ns.update(marker=object()),
            lambda: case.leaf_ns.update(marker=case.marker),
        ),
        (lambda: case.leaf_ns.update(extra=None), lambda: case.leaf_ns.pop("extra")),
    )
    for change, restore in changes:
        change()
        try:
            assert loaded.namespace["chosen"] is case.leaf
            case.prepare(loader, refused=True)
            case.execute(loader, pending, refused=True)
            case.runtime(loaded, PATH, refused=True)
            case.current(loader, loaded, refused=True)
        finally:
            restore()
        case.current(loader, loaded)
    case.op(
        "module_dict",
        lambda: object.__setattr__(case.leaf, "__dict__", case.leaf_ns.copy()),
        refused=True,
        errors=(AttributeError,),
    )
    assert case.namespace(case.leaf) is case.leaf_ns
    case.current(loader, loaded)

    class HostileKey(str):
        __hash__ = str.__hash__

        def __len__(self):
            case.events.append("key_len")
            raise AssertionError

        def __eq__(self, other):
            case.events.append("key_eq")
            raise AssertionError

    class ForeignModule(ModuleType):
        def __getattribute__(self, name):
            case.events.append("module_getattribute")
            raise AssertionError

    case.setup.update(classes=2, hostile_instances=2)
    for key in (object(), "k" * 129, HostileKey("chain_key_control")):
        case.leaf_ns[key] = None
        try:
            assert any(item is key for item in case.leaf_ns)
            case.loader(refused=True)
            case.prepare(loader, refused=True)
            case.execute(loader, pending, refused=True)
            case.runtime(loaded, PATH, refused=True)
            case.current(loader, loaded, refused=True)
        finally:
            del case.leaf_ns[key]
        case.current(loader, loaded)
    foreign = cast(Any, ModuleType.__new__)(ForeignModule, "foreign")
    case.setup["modules"] += 1
    bad_row = case.row(PATH, foreign)
    case.loader((*case.rows[:2], bad_row), refused=True)
    final = case.execute(loader, pending)
    case.current(loader, final)
    case.write(
        tmp_path, "test_qualified_plain_rechecks_original_rows_modules_keys_and_nested_slots"
    )


def test_qualified_plain_preserves_implicit_alias_and_decorator_bindings(tmp_path):
    case = Case()
    factory_rows, globals_rows = case.named_rows("factory"), case.named_rows("globals")
    for name, rows in (("factory", factory_rows), ("globals", globals_rows)):
        loader = case.loader(rows)
        for statement in (
            "import " + name + ".pkg.leaf\n",
            "def unused():\n    import " + name + ".pkg.leaf\n",
        ):
            case.prepare(loader, statement.encode() + BASE, refused=True)
    loader = case.loader()
    for alias in ("factory", "globals"):
        for statement in (
            "import declared.pkg.leaf as " + alias + "\n",
            "def unused():\n    import declared.pkg.leaf as " + alias + "\n",
        ):
            case.prepare(loader, statement.encode() + BASE, refused=True)

    class Wrapper:
        pass

    def decorator(function):
        case.events.append("decorator_body")
        raise AssertionError

    decor = case.module("decor")
    decor_ns = case.namespace(decor)
    decor_ns.update(pkg=case.parent, fixture=decorator)
    decor_rows = case.named_rows("decor", decor)
    policy = ApprovedSourceDecorator(
        "factory",
        "decor",
        "fixture",
        cast(FunctionType, decorator),
        Wrapper,
        cast(GetSetDescriptorType, Wrapper.__dict__["__dict__"]),
    )
    case.setup.update(classes=1, decorator_rows=1)
    protected = case.op(
        "loader", lambda: FixtureSourceLoader((*case.rows, *decor_rows), decorators=(policy,))
    )
    suffix = b"@decor.fixture(scope='module')\ndef factory():\n    return 7\n"
    statements = (
        "import decor.pkg.leaf\n",
        "def unused():\n    import decor.pkg.leaf\n",
        "import declared.pkg.leaf as decor\n",
        "def unused():\n    import declared.pkg.leaf as decor\n",
    )
    for statement in statements:
        case.prepare(protected, b"import decor\n" + statement.encode() + suffix, refused=True)
    case.write(tmp_path, "test_qualified_plain_preserves_implicit_alias_and_decorator_bindings")


def test_qualified_plain_refuses_malformed_paths_and_runtime_protocols(tmp_path):
    case = Case()
    case.leaf_ns.update(("v" + str(i), i) for i in range(16))

    class ForeignPath(str):
        def __getattribute__(self, name):
            case.events.append("path_getattribute")
            raise AssertionError

    class ForeignNames(tuple):
        def __len__(self):
            case.events.append("names_len")
            raise AssertionError

        def __iter__(self):
            case.events.append("names_iter")
            raise AssertionError

    class ForeignLevel:
        def __eq__(self, other):
            case.events.append("level_eq")
            raise AssertionError

    foreign_path, foreign_names, foreign_level = (
        ForeignPath(PATH),
        ForeignNames(("marker",)),
        ForeignLevel(),
    )
    case.setup.update(classes=3, hostile_instances=3)
    for candidate in (
        "",
        ".a",
        "a.",
        "a..b",
        ".".join("a" for _ in range(17)),
        "a." + "x" * 127,
        foreign_path,
        object(),
    ):
        case.loader((case.row(candidate, case.leaf),), refused=True)
    loader = case.loader()
    prepared = case.prepare(loader, BASE)
    loaded = case.execute(loader, prepared)
    case.current(loader, loaded)
    names16 = tuple("v" + str(i) for i in range(16))
    for names, original in (
        ((), case.root),
        (None, case.root),
        (("marker",), case.leaf),
        (names16, case.leaf),
    ):
        assert case.runtime(loaded, PATH, names) is original
    bad: tuple[tuple[Any, Any, Any], ...] = (
        (foreign_path, (), 0),
        (object(), (), 0),
        ("a..b", (), 0),
        (PATH, ["marker"], 0),
        (PATH, foreign_names, 0),
        (PATH, ("marker",), True),
        (PATH, ("marker",), 1),
        (PATH, ("marker",), foreign_level),
        (PATH, ("missing",), 0),
        (PATH, ("*",), 0),
        (PATH, ("marker", "marker"), 0),
        (PATH, (*names16, "v16"), 0),
    )
    for name, names, level in bad:
        case.runtime(loaded, name, names, level, refused=True)
    for statement in (
        "from .declared import pkg\n",
        "from declared.pkg.leaf import *\n",
        "from declared.pkg.leaf import missing\n",
        "import unknown.pkg\n",
    ):
        case.prepare(loader, statement.encode() + BASE, refused=True)
    case.write(tmp_path, "test_qualified_plain_refuses_malformed_paths_and_runtime_protocols")


def test_qualified_plain_preserves_finite_existing_bounds(tmp_path):
    case = Case()
    rows16, _ = case.chain(tuple("p" + str(i) for i in range(16)))
    extra = case.row("extra", case.leaf)
    case.loader((*rows16, extra), refused=True)
    case.loader((rows16[0], rows16[0]), refused=True)
    for capacity in (0, 17, True):
        case.loader(rows16, capacity, refused=True)
    loader = case.loader(rows16, capacity=1)
    path16 = ".".join("p" + str(i) for i in range(16))
    raw = source_for(path16)
    maximum = raw + b"\n#" + b"x" * (16384 - len(raw) - 2)
    assert len(maximum) == 16384
    case.prepare(loader, maximum + b"x", refused=True)
    case.prepare(loader, b"", refused=True)
    wide = b"x=0\n" * 1100 + raw
    assert len(wide) < 16384
    case.prepare(loader, wide, refused=True)
    prepared = case.prepare(loader, maximum)
    case.prepare(loader, raw, refused=True)
    loaded = case.execute(loader, prepared)
    case.execute(loader, prepared, refused=True)
    case.current(loader, loaded)
    case.prepare(loader, raw, refused=True)
    long_parts = ("r", "x" * 126)
    long_rows, long_modules = case.chain(long_parts)
    leaf = case.namespace(long_modules[-1])
    leaf.clear()
    leaf.update(__name__="owned", counts=case.effects, marker=case.marker)
    key128 = "k" * 128
    leaf[key128] = case.marker
    leaf.update(("pad" + str(i), case.marker) for i in range(508))
    case.setup["padding_keys"] = 509
    assert len(leaf) == 512
    long_path = ".".join(long_parts)
    assert len(long_path) == 128
    long_loader = case.loader(long_rows)
    long_source = source_for(long_path)
    case.prepare(long_loader, long_source)
    leaf["over"] = None
    try:
        case.loader(long_rows, refused=True)
        case.prepare(long_loader, long_source, refused=True)
    finally:
        del leaf["over"]
    original = leaf.pop(key128)
    leaf["k" * 129] = original
    try:
        case.loader(long_rows, refused=True)
        case.prepare(long_loader, long_source, refused=True)
    finally:
        del leaf["k" * 129]
        leaf[key128] = original
    case.prepare(long_loader, long_source)
    case.write(tmp_path, "test_qualified_plain_preserves_finite_existing_bounds")


def test_qualified_plain_refuses_cooperative_prior_hook_mutation(tmp_path):
    case = Case()
    loader = case.loader()
    prepared = case.prepare(loader)
    seen: list[str] = []

    def prior(frame, event, arg):
        if event == "call" and frame.f_code is prepared.module_code:
            seen.append("original-module-entry")
            del case.parent_ns["leaf"]

    sys.setprofile(prior)
    try:
        guard = ProviderEntryProfile(
            ((prepared.module_code, "module", 1),), tmp_path / "qualified-breach.json"
        )
        with guard.measure():
            try:
                case.execute(loader, prepared, refused=True, errors=(ImportError,))
            finally:
                case.parent_ns["leaf"] = case.leaf
            case.execute(loader, prepared, refused=True)
        assert sys.getprofile() is prior
    finally:
        sys.setprofile(case.prior[0])
        case.parent_ns["leaf"] = case.leaf
    assert guard.counts == dict(module=1) and seen == ["original-module-entry"]
    assert case.effects == dict(module=0, factory=0, provider=0)
    case.module_bodies += guard.counts["module"]
    with pytest.raises(RuntimeError, match="^entry profile cannot renew its allowance$"):
        with guard.measure():
            pytest.fail("guard renewed")
    case.operations.append("renewal:refuse")
    case.write(
        tmp_path,
        "test_qualified_plain_refuses_cooperative_prior_hook_mutation",
        counts=guard.counts,
        windows=1,
        renewal=1,
        mutations=len(seen),
        failed=1,
    )
