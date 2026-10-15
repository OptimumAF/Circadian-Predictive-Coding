"""Public opt-in callable continuity; cooperative stand-ins, no original fixtures."""

import builtins
from hashlib import sha256
import json
from pathlib import Path
import sys
import threading
from types import BuiltinFunctionType, CodeType, FunctionType, GetSetDescriptorType, ModuleType
from typing import Any, Callable, cast

import pytest

from provider_entry_profile import ProviderEntryProfile
from provider_source_loader import ApprovedSourceImport, FixtureSourceLoader


ORIGINAL = cast(BuiltinFunctionType, builtins.callable)
MODULE_DICT = cast(
    GetSetDescriptorType, type.__dict__["__dict__"].__get__(ModuleType, type)["__dict__"]
)
BUILTIN_NAMESPACE = MODULE_DICT.__get__(builtins, ModuleType)
LABELS = ("module", "class_body", "factory", "constructor", "provider")
SIMPLE = b"""import callableeffects
callableeffects.counts['module'] += 1
def factory():
    callableeffects.counts['factory'] += 1
    return 7
"""
POSITIVE = b"""import callableeffects
callableeffects.counts['module'] += 1
class Gate:
    callableeffects.counts['class_body'] += 1
    def __init__(self, value):
        callableeffects.counts['constructor'] += 1
        self.value = value
    def __call__(self):
        callableeffects.counts['provider'] += 1
        return callable(self.value)
def factory(value):
    callableeffects.counts['factory'] += 1
    return Gate(value)()
"""


class Case:
    def __init__(self) -> None:
        self.counts = dict.fromkeys(LABELS, 0)
        self.events: list[str] = []
        self.operations: list[str] = []
        self.module = ModuleType("callableeffects")
        MODULE_DICT.__get__(self.module, ModuleType)["counts"] = self.counts
        self.imports = (ApprovedSourceImport("callableeffects", self.module),)
        self.prior = (sys.getprofile(), threading.getprofile())
        self.hostile_classes = self.hostile_instances = 0

    def attempt(
        self,
        kind: str,
        action: Callable[[], Any],
        *,
        refused: bool = False,
        errors: tuple[type[BaseException], ...] = (TypeError, ValueError),
    ) -> Any:
        # Why no pytest inside: some callers temporarily replace builtins.callable.
        if refused:
            try:
                action()
            except errors:
                self.operations.append(kind + ":refuse")
                return None
            raise AssertionError("required capability refusal did not occur")
        result = action()
        self.operations.append(kind + ":accept")
        return result

    def loader(self, candidate: Any = ORIGINAL, capacity: Any = 16, **options: Any) -> Any:
        return self.attempt(
            "loader",
            lambda: FixtureSourceLoader(self.imports, capacity, callable_builtin=candidate),
            **options,
        )

    def prepare(self, loader: Any, source: bytes = SIMPLE, **options: Any) -> Any:
        name = "callable_control_" + str(len(self.operations))
        return self.attempt(
            "prepare",
            lambda: loader.prepare(
                source, sha256(source).hexdigest(), name, "callable.py", ("factory",)
            ),
            **options,
        )

    def execute(self, loader: Any, prepared: Any, **options: Any) -> Any:
        return self.attempt("execute", lambda: loader.execute(prepared), **options)

    def current(self, loader: Any, loaded: Any, **options: Any) -> None:
        self.attempt("current", lambda: loader.require_current(loaded), **options)

    def write(
        self,
        directory: Path,
        name: str,
        *,
        registered: dict[str, int] | None = None,
        results: list[Any] | None = None,
        renewal: int = 0,
    ) -> None:
        assert builtins.callable is ORIGINAL
        assert self.events == []
        assert sys.getprofile() is self.prior[0]
        assert threading.getprofile() is self.prior[1]
        assert threading.active_count() == 1
        assert not (directory / "callable-breach.json").exists()
        entries = {} if registered is None else registered
        record = dict(
            test_name=name,
            operations=self.operations,
            source_effects=self.counts,
            successful_preparations=self.operations.count("prepare:accept"),
            fixture_setup=dict(
                case_constructors=1,
                approved_modules=1,
                hostile_classes=self.hostile_classes,
                hostile_instances=self.hostile_instances,
            ),
            hostile_events=self.events,
            registered_counts=entries,
            registered_attempts=sum(entries.values()),
            registered_python_bodies=sum(
                value for label, value in entries.items() if label != "native_callable"
            ),
            results=[] if results is None else results,
            renewal_refusals=renewal,
            restored=True,
            workers=0,
            joined=True,
            receipt_exists=False,
        )
        (directory / "callable-work.json").write_text(json.dumps(record), encoding="utf8")


def test_callable_default_remains_absent(tmp_path):
    case = Case()
    loader = case.attempt("loader", lambda: FixtureSourceLoader(case.imports))
    assert loader.callable_builtin is None
    source = SIMPLE.replace(b"return 7", b"return callable(7)")
    prepared = case.prepare(loader, source)
    loaded = case.execute(loader, prepared)
    assert type(loaded.namespace["__builtins__"]) is dict
    assert "callable" not in loaded.namespace["__builtins__"]
    case.current(loader, loaded)
    case.attempt("factory", loaded.roots[0], refused=True, errors=(NameError,))
    case.current(loader, loaded)
    case.write(tmp_path, "test_callable_default_remains_absent")


def test_callable_rejects_foreign_candidates_without_dispatch(tmp_path):
    case = Case()

    def replacement(*args: Any, **kwargs: Any) -> bool:
        case.events.append("python_call")
        raise AssertionError("foreign callable dispatched")

    class Hostile:
        def __getattribute__(self, name):
            case.events.append("getattribute")
            raise AssertionError

        def __call__(self, *args, **kwargs):
            case.events.append("call")
            raise AssertionError

    hostile = object.__new__(Hostile)
    case.hostile_classes = case.hostile_instances = 1
    for candidate in (True, False, 1, object(), len, isinstance, replacement, hostile):
        case.loader(candidate, refused=True)
    loader = case.loader(ORIGINAL)
    assert loader.callable_builtin is ORIGINAL
    assert all(value == 0 for value in case.counts.values())
    case.write(tmp_path, "test_callable_rejects_foreign_candidates_without_dispatch")


def test_callable_rechecks_original_builtin_at_all_boundaries(tmp_path):
    case = Case()

    def replacement(*args: Any) -> bool:
        case.events.append("replacement_call")
        raise AssertionError

    def changed(action: Callable[[], None]) -> None:
        BUILTIN_NAMESPACE["callable"] = replacement
        try:
            action()
        finally:
            BUILTIN_NAMESPACE["callable"] = ORIGINAL

    changed(lambda: case.loader(refused=True))
    loader = case.loader()
    changed(lambda: case.prepare(loader, refused=True))
    assert case.counts["module"] == 0
    prepared = case.prepare(loader)
    changed(lambda: case.execute(loader, prepared, refused=True))
    assert case.counts["module"] == 0
    loaded = case.execute(loader, prepared)
    changed(lambda: case.current(loader, loaded, refused=True))
    case.current(loader, loaded)
    case.write(tmp_path, "test_callable_rechecks_original_builtin_at_all_boundaries")


def test_callable_retains_public_selection_identity(tmp_path):
    case = Case()
    default = case.loader(None)
    default.callable_builtin = ORIGINAL
    try:
        case.prepare(default, refused=True)
    finally:
        default.callable_builtin = None
    plain = case.prepare(default)
    plain_loaded = case.execute(default, plain)
    case.current(default, plain_loaded)
    enabled = case.loader()
    first, second = case.prepare(enabled), case.prepare(enabled)
    loaded = case.execute(enabled, first)
    for foreign in (None, len):
        enabled.callable_builtin = foreign
        try:
            case.prepare(enabled, refused=True)
            case.execute(enabled, second, refused=True)
            case.current(enabled, loaded, refused=True)
        finally:
            enabled.callable_builtin = ORIGINAL
        case.current(enabled, loaded)
    second_loaded = case.execute(enabled, second)
    case.current(enabled, second_loaded)
    case.write(tmp_path, "test_callable_retains_public_selection_identity")


def test_callable_retains_loaded_builtin_dictionary_and_slot(tmp_path):
    case = Case()
    loader = case.loader()
    prepared = case.prepare(loader)
    loaded = case.execute(loader, prepared)
    case.current(loader, loaded)
    namespace = loaded.namespace
    original = namespace["__builtins__"]
    assert type(original) is dict and original["callable"] is ORIGINAL

    class ForeignDict(dict):
        def __len__(self):
            case.events.append("dict_len")
            raise AssertionError

        def __iter__(self):
            case.events.append("dict_iter")
            raise AssertionError

        def __getitem__(self, key):
            case.events.append("dict_get")
            raise AssertionError

    foreign = ForeignDict(original)
    case.hostile_classes = case.hostile_instances = 1
    for candidate in (original.copy(), foreign):
        namespace["__builtins__"] = candidate
        try:
            case.current(loader, loaded, refused=True)
        finally:
            namespace["__builtins__"] = original
        case.current(loader, loaded)
    original["callable"] = len
    try:
        case.current(loader, loaded, refused=True)
    finally:
        original["callable"] = ORIGINAL
    case.current(loader, loaded)
    del original["callable"]
    try:
        case.current(loader, loaded, refused=True)
    finally:
        original["callable"] = ORIGINAL
    case.current(loader, loaded)
    del namespace["__builtins__"]
    try:
        case.current(loader, loaded, refused=True)
    finally:
        namespace["__builtins__"] = original
    case.current(loader, loaded)
    case.write(tmp_path, "test_callable_retains_loaded_builtin_dictionary_and_slot")


def test_callable_preserves_capacity_and_source_bounds(tmp_path):
    case = Case()
    for capacity in (0, 17, True):
        case.loader(capacity=capacity, refused=True)
    loader = case.loader(capacity=1)
    maximum = SIMPLE + b"\n#" + b"x" * (16384 - len(SIMPLE) - 2)
    assert len(maximum) == 16384
    case.prepare(loader, maximum + b"x", refused=True)
    case.prepare(loader, b"", refused=True)
    prepared = case.prepare(loader, maximum)
    case.prepare(loader, refused=True)
    loaded = case.execute(loader, prepared)
    case.execute(loader, prepared, refused=True)
    case.current(loader, loaded)
    case.prepare(loader, refused=True)
    case.write(tmp_path, "test_callable_preserves_capacity_and_source_bounds")


def test_callable_rejects_builtin_keys_before_authority_lookup(tmp_path):
    case = Case()
    loader = case.loader()
    prepared = case.prepare(loader)
    loaded = case.execute(loader, prepared)
    case.current(loader, loaded)

    class HostileKey(str):
        __hash__ = str.__hash__

        def __len__(self):
            case.events.append("key_len")
            raise AssertionError

        def __eq__(self, other):
            case.events.append("key_eq")
            raise AssertionError

    key = HostileKey("callable_capability_control")
    case.hostile_classes = case.hostile_instances = 1
    for foreign in (object(), "k" * 129, key):
        BUILTIN_NAMESPACE[foreign] = None
        try:
            assert any(item is foreign for item in BUILTIN_NAMESPACE)
            case.loader(refused=True)
            case.prepare(loader, refused=True)
            case.current(loader, loaded, refused=True)
        finally:
            del BUILTIN_NAMESPACE[foreign]
        case.current(loader, loaded)
    assert BUILTIN_NAMESPACE["callable"] is ORIGINAL
    case.write(tmp_path, "test_callable_rejects_builtin_keys_before_authority_lookup")


def test_callable_executes_original_native_under_one_guard(tmp_path):
    case = Case()
    loader = case.loader()
    prepared = case.prepare(loader, POSITIVE)
    factory_code = prepared.root_codes[0][1]

    def selected(parent: CodeType, name: str) -> CodeType:
        assert len(parent.co_consts) <= 512
        matches = [
            value for value in parent.co_consts if type(value) is CodeType and value.co_name == name
        ]
        assert len(matches) == 1
        return matches[0]

    class_code = selected(prepared.module_code, "Gate")
    constructor_code = selected(class_code, "__init__")
    provider_code = selected(class_code, "__call__")
    entries = (
        (prepared.module_code, "module", 1),
        (class_code, "class_body", 1),
        (factory_code, "factory", 1),
        (constructor_code, "constructor", 1),
        (provider_code, "provider", 1),
        (ORIGINAL, "native_callable", 1),
    )
    guard = ProviderEntryProfile(entries, tmp_path / "callable-breach.json")
    with guard.measure():
        loaded = case.execute(loader, prepared)
        injected = loaded.namespace["__builtins__"]
        assert type(injected) is dict and injected["callable"] is ORIGINAL
        root = cast(FunctionType, loaded.roots[0])
        assert root.__code__ is factory_code and root.__globals__ is loaded.namespace
        owner_type = loaded.namespace["Gate"]
        assert type(owner_type) is type
        assert owner_type.__dict__["__init__"].__code__ is constructor_code
        assert owner_type.__dict__["__call__"].__code__ is provider_code
        result = root(7)
        assert type(result) is bool and result is False
        case.current(loader, loaded)
    case.current(loader, loaded)
    case.execute(loader, prepared, refused=True)
    with pytest.raises(RuntimeError, match="^entry profile cannot renew its allowance$"):
        with guard.measure():
            pytest.fail("one-use guard renewed")
    case.operations.append("renewal:refuse")
    assert guard.counts == dict.fromkeys((*LABELS, "native_callable"), 1)
    assert case.counts == dict.fromkeys(LABELS, 1)
    case.write(
        tmp_path,
        "test_callable_executes_original_native_under_one_guard",
        registered=guard.counts,
        results=[result],
        renewal=1,
    )
