"""One-use source loader with optional callable continuity; no sandbox.

Example: FixtureSourceLoader((), callable_builtin=builtins.callable).
Declare every package prefix and native parent-child slot for plain dotted imports.
FROM imports need only their named module. Traversal is cooperative, not atomic;
these declarations do not prove cold import or source origin.
The caller retains original authority before fixture work. Checks prove continuity,
not process birth, import origin or general globals contents.
"""

import ast
import builtins
from dataclasses import dataclass
import hashlib
from itertools import repeat
import sys
from types import (
    BuiltinFunctionType,
    CodeType,
    FunctionType,
    GetSetDescriptorType,
    MappingProxyType,
    MemberDescriptorType,
    ModuleType,
)
from typing import Any

from provider_fixture_binding import FixtureRootBinding, require_fixture_root_binding
from provider_owner_namespace_profile import OwnerNamespaceBinding

_TD = type.__dict__["__dict__"]
_GET = _TD.__get__(MemberDescriptorType, type)["__get__"]
_MOD = _TD.__get__(ModuleType, type)["__dict__"]
_FN_CODE = _TD.__get__(FunctionType, type)["__code__"]
_FN_GLOBALS = _TD.__get__(FunctionType, type)["__globals__"]
_CONSTS = _TD.__get__(CodeType, type)["co_consts"]
_NAME = _TD.__get__(CodeType, type)["co_name"]
_BASES = _TD.__get__(type, type)["__bases__"]
_DESC_GET = _TD.__get__(GetSetDescriptorType, type)["__get__"]
_COMPILE, _EXEC = compile, exec
_B = "invalid source witness"
_M = object()


@dataclass(frozen=True, slots=True)
class ApprovedSourceImport:
    name: str
    module: ModuleType


@dataclass(frozen=True, slots=True)
class ApprovedSourceDecorator:
    root: str
    module_name: str
    attribute: str
    function: FunctionType
    wrapper_type: type
    descriptor: GetSetDescriptorType
    dependencies: tuple[FunctionType, ...] = ()
    classes: tuple[type, ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedFixtureSource:
    module_name: str
    filename: str
    source_sha256: str
    module_code: CodeType
    root_codes: tuple[tuple[str, CodeType], ...]


@dataclass(frozen=True, slots=True)
class LoadedFixtureSource:
    prepared: PreparedFixtureSource
    module: ModuleType
    namespace: dict[str, object]
    roots: tuple[FunctionType, ...]


_A, _P, _L = ApprovedSourceImport, PreparedFixtureSource, LoadedFixtureSource
_D = ApprovedSourceDecorator
_FIELDS = {
    kind: tuple(_TD.__get__(kind, type)[name] for name in names)
    for kind, names in (
        (_A, ("name", "module")),
        (
            _D,
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
        ),
        (_P, ("module_name", "filename", "source_sha256", "module_code", "root_codes")),
        (_L, ("prepared", "module", "namespace", "roots")),
    )
}
_BINDS = (
    (ast.arg, "arg"),
    (ast.ExceptHandler, "name"),
    (ast.MatchAs, "name"),
    (ast.MatchStar, "name"),
    (ast.MatchMapping, "rest"),
)
_NAMES_DENIED = tuple(
    (
        "__builtins__ __import__ eval exec compile globals locals vars getattr setattr delattr open"
    ).split()
)
_ATTRS_DENIED = tuple(
    "__dict__ __class__ __globals__ __code__ __closure__ __getattribute__".split()
)


def _reject():
    raise ValueError(_B)


def _foreign():
    raise TypeError(_B)


def _exact(value, kind):
    if type(value) is not kind:
        _foreign()


def _values(row, kind):
    _exact(row, kind)
    try:
        return tuple(map(_GET, _FIELDS[kind], repeat(row), repeat(kind)))
    except AttributeError:
        raise ValueError(_B) from None


def _same(row, saved, kind):
    values = _values(row, kind)
    if saved is None or any(value is not old for value, old in zip(values, saved)):
        _reject()
    return values


def _snapshot(ns, bound=512):
    if len(ns) > bound:
        _reject()
    for key in ns:
        if type(key) is not str or len(key) > 128:
            _reject()
    return tuple(ns.items())


def _unchanged(ns, saved):
    _snapshot(ns)
    if len(ns) != len(saved) or any(key not in ns or ns[key] is not value for key, value in saved):
        _reject()


def _dict(ns, bound=512):
    _exact(ns, dict)
    _snapshot(ns, bound)
    return ns


def _module(module):
    _exact(module, ModuleType)
    return _dict(_MOD.__get__(module, ModuleType))


_CACHE = _module(sys)["modules"]
_BUILTINS = {
    name: _module(builtins)[name]
    for name in (
        "object int str bool len tuple dict range ValueError RuntimeError __build_class__"
    ).split()
}


_ORIGINAL_BUILTINS_MODULE = builtins
_ORIGINAL_BUILTINS_NAMESPACE = _module(_ORIGINAL_BUILTINS_MODULE)
_ORIGINAL_CALLABLE = _ORIGINAL_BUILTINS_NAMESPACE.get("callable", _M)


def _name(name):
    _exact(name, str)
    if (
        not 0 < len(name) <= 128
        or not name.isascii()
        or not name.isidentifier()
        or name.startswith("__")
    ):
        _reject()


def _module_name(name):
    _exact(name, str)
    if not 0 < len(name) <= 128:
        _reject()
    parts = name.split(".")
    if not 1 <= len(parts) <= 16:
        _reject()
    for part in parts:
        _name(part)


def _names(names, minimum=0):
    _exact(names, tuple)
    if not minimum <= len(names) <= 16:
        _reject()
    for name in names:
        _name(name)
    if len(set(names)) != len(names):
        _reject()


def _codes(code, names):
    constants = _CONSTS.__get__(code, CodeType)
    if type(constants) is not tuple or len(constants) > 512:
        _reject()
    result = []
    for name in names:
        matches = [
            item
            for item in constants
            if type(item) is CodeType and _NAME.__get__(item, CodeType) == name
        ]
        if len(matches) != 1:
            _reject()
        result.append((name, matches[0]))
    return tuple(result)


def _class(cls):
    _exact(cls, type)
    ns = _TD.__get__(cls, type)
    _exact(ns, MappingProxyType)
    _snapshot(ns)
    return ns


def _check_function(fn, code, ns):
    _exact(fn, FunctionType)
    if (
        _FN_CODE.__get__(fn, FunctionType) is not code
        or _FN_GLOBALS.__get__(fn, FunctionType) is not ns
    ):
        _reject()


def _function(fn):
    _exact(fn, FunctionType)
    code = _FN_CODE.__get__(fn, FunctionType)
    ns = _dict(_FN_GLOBALS.__get__(fn, FunctionType))
    return fn, code, ns, _snapshot(ns)


def _distinct(items, kind, excluded):
    _exact(items, tuple)
    if len(items) > 16:
        _reject()
    for index, item in enumerate(items):
        if type(item) is not kind:
            _foreign()
        if item is excluded or any(item is old for old in items[:index]):
            _reject()


def _pin_class(cls):
    ns = _class(cls)
    return cls, _BASES.__get__(cls, type), _snapshot(ns)


def _check_decorator(node: Any, data):
    if type(node) is ast.Call:
        if (
            node.args
            or len(node.keywords) != 1
            or node.keywords[0].arg != "scope"
            or type(node.keywords[0].value) is not ast.Constant
            or type(node.keywords[0].value.value) is not str
            or node.keywords[0].value.value != "module"
        ):
            _reject()
        node = node.func
    if (
        type(node) is not ast.Attribute
        or node.attr != data[2]
        or type(node.value) is not ast.Name
        or node.value.id != data[1]
    ):
        _reject()


@dataclass(slots=True)
class _Issue:
    row: PreparedFixtureSource
    data: tuple
    spent: bool = False
    loaded: LoadedFixtureSource | None = None
    saved: tuple | None = None
    bindings: tuple = ()
    builtin_namespace: dict[str, object] | None = None


def _pin_policy(loader, row):
    data = _values(row, _D)
    for name in data[:3]:
        _name(name)
    if data[0] == "request":
        _reject()
    _distinct(data[6], FunctionType, data[3])
    _distinct(data[7], type, data[4])
    funcs = tuple(_function(fn) for fn in (data[3], *data[6]))
    classes = tuple(_pin_class(cls) for cls in (data[4], *data[7]))
    desc = data[5]
    _exact(desc, GetSetDescriptorType)
    # Why native reads: avoid hooks.
    name = desc.__name__
    if (
        desc.__objclass__ is not data[4]
        or type(name) is not str
        or name != "__dict__"
        or _class(data[4]).get("__dict__", _M) is not desc
    ):
        _reject()
    _, ns = loader._lookup(data[1])
    if ns.get(data[2], _M) is not data[3]:
        _reject()
    return row, data, ns, funcs, classes


def _preflight(loader, tree, roots):
    node: Any
    defs = []
    admitted = []
    for _, data, _, _, _ in loader._policies.values():
        if data[0] not in roots:
            _reject()
    for name in roots:
        found = [node for node in tree.body if type(node) is ast.FunctionDef and node.name == name]
        if len(found) != 1:
            _reject()
        node = found[0]
        if (
            node.args.defaults
            or any(value is not None for value in node.args.kw_defaults)
            or getattr(node, "type_params", ())
        ):
            _reject()
        policy = loader._policies.get(name)
        if policy is None:
            if node.decorator_list:
                _reject()
        else:
            data = policy[1]
            if len(node.decorator_list) != 1:
                _reject()
            _check_decorator(node.decorator_list[0], data)
            earlier = tree.body[: tree.body.index(node)]
            imports = [
                entry
                for entry in earlier
                if type(entry) is ast.Import
                and len(entry.names) == 1
                and entry.names[0].name == data[1]
                and entry.names[0].asname is None
            ]
            if len(imports) != 1:
                _reject()
            admitted.append(imports[0].names[0])
        defs.append(node)
    modules = tuple(data[1] for _, data, _, _, _ in loader._policies.values())
    for index, node in enumerate(ast.walk(tree)):
        kind = type(node)
        if index >= 4096:
            _reject()
        bound = []
        if kind is ast.Name:
            if node.id in _NAMES_DENIED:
                _reject()
            if type(node.ctx) in (ast.Store, ast.Del):
                bound.append(node.id)
        if kind is ast.Attribute:
            base = node.value
            while type(base) in (ast.Attribute, ast.Subscript):
                base = base.value
            if node.attr in _ATTRS_DENIED or (
                type(node.ctx) in (ast.Store, ast.Del)
                and type(base) is ast.Name
                and base.id in modules
            ):
                _reject()
        if kind in (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef):
            if node.name in modules:
                _reject()
            if not any(node is root for root in defs):
                bound.append(node.name)
        for cls, field in _BINDS:
            if isinstance(node, cls):
                bound.append(getattr(node, field))
        if kind in (ast.Global, ast.Nonlocal):
            bound.extend(node.names)
        if kind.__name__ in ("TypeVar", "ParamSpec", "TypeVarTuple"):
            bound.append(node.name)
        if kind is ast.alias:
            bound.append(node.asname or node.name.split(".")[0])
        if kind in (ast.Import, ast.ImportFrom):
            if kind is ast.ImportFrom:
                if node.level != 0 or node.module is None:
                    _reject()
                _names(tuple(alias.name for alias in node.names))
                _, ns = loader._lookup(node.module)
            for alias in node.names:
                if kind is ast.Import:
                    _module_name(alias.name)
                else:
                    _name(alias.name)
                if alias.asname is not None:
                    _name(alias.asname)
                if kind is ast.Import:
                    loader._plain_import(alias.name)
                elif alias.name not in ns:
                    _reject()
        if any(
            name is not None
            and name in (*roots, *_NAMES_DENIED, *modules)
            and not (kind is ast.alias and any(node is alias for alias in admitted))
            for name in bound
        ):
            _reject()


def _prepared(item):
    _same(item.row, item.data, _P)
    current = _codes(item.data[3], tuple(name for name, _ in item.data[4]))
    if any(code is not old for (_, code), (_, old) in zip(current, item.data[4])):
        _reject()


def _roots(loader, item, module, ns, pinned=None):
    if _module(module) is not ns or ns.get("__name__", _M) is not item.data[0]:
        _reject()
    found = []
    for index, (name, code) in enumerate(item.data[4]):
        fn = ns.get(name, _M)
        policy = loader._policies.get(name)
        if policy is not None:
            if pinned is None:
                data = policy[1]
                if type(fn) is not data[4]:
                    _foreign()
                storage = _dict(_DESC_GET(data[5], fn, data[4]))
                raw = storage.get("_fixture_function", _M)
                owner = OwnerNamespaceBinding(fn, data[4], data[5], storage)
                binding = FixtureRootBinding(ns, name, owner, raw, code, ns)
                require_fixture_root_binding(binding)
                item.bindings += ((name, binding),)
                fn = raw
            else:
                binding = next(binding for slot, binding in item.bindings if slot == name)
                require_fixture_root_binding(binding)
                fn = pinned[index]
        _check_function(fn, code, ns)
        if pinned is not None and fn is not pinned[index]:
            _reject()
        found.append(fn)
    return tuple(found)


class FixtureSourceLoader:
    callable_builtin: BuiltinFunctionType | None
    _original_callable_builtin: BuiltinFunctionType | None
    _imports: tuple
    _items: list[_Issue]
    _cap: int
    _policies: dict[str, tuple]

    def __init__(
        self,
        imports: tuple[ApprovedSourceImport, ...],
        max_sources: int = 16,
        *,
        decorators: tuple[ApprovedSourceDecorator, ...] = (),
        callable_builtin: BuiltinFunctionType | None = None,
    ) -> None:
        self.callable_builtin = self._original_callable_builtin = callable_builtin
        self._callable_policy()
        if type(imports) is not tuple or type(max_sources) is not int:
            _foreign()
        if len(imports) > 16 or not 1 <= max_sources <= 16:
            _reject()
        rows: list[tuple] = []
        for row in imports:
            data = _values(row, _A)
            _module_name(data[0])
            if any(data[0] == prior[1][0] for prior in rows):
                _reject()
            ns = _module(data[1])
            rows.append((row, data, ns, tuple(ns.items())))
        self._imports, self._items, self._cap = tuple(rows), [], max_sources
        self._policies = {}
        _exact(decorators, tuple)
        if len(decorators) > 16:
            _reject()
        policies: dict[str, tuple] = {}
        for decorator in decorators:
            policy = _pin_policy(self, decorator)
            name = policy[1][0]
            if name in policies:
                _reject()
            policies[name] = policy
        self._policies = policies
        self._callable_policy()

    def _policy(self):
        self._callable_policy()
        for row, data, ns, snapshot in self._imports:
            _same(row, data, _A)
            actual = _module(data[1])
            if actual is not ns:
                _reject()
            _unchanged(actual, snapshot)

        for row, data, ns, funcs, classes in self._policies.values():
            _same(row, data, _D)
            if ns.get(data[2], _M) is not data[3]:
                _reject()
            for fn, code, namespace, saved in funcs:
                _check_function(fn, code, namespace)
                _unchanged(_dict(namespace), saved)
            for cls, bases, saved in classes:
                if _BASES.__get__(cls, type) is not bases:
                    _reject()
                _unchanged(_class(cls), saved)

    def _callable_policy(self):
        if self.callable_builtin is not self._original_callable_builtin:
            _reject()
        selected = self._original_callable_builtin
        if selected is None:
            return
        _exact(selected, BuiltinFunctionType)
        if selected is not _ORIGINAL_CALLABLE:
            _reject()
        namespace = _module(_ORIGINAL_BUILTINS_MODULE)
        if (
            namespace is not _ORIGINAL_BUILTINS_NAMESPACE
            or namespace.get("callable", _M) is not selected
        ):
            _reject()

    def _injected_builtins(self, item, namespace):
        if self._original_callable_builtin is None:
            return
        actual = _dict(namespace).get("__builtins__", _M)
        if actual is not item.builtin_namespace:
            _reject()
        if _dict(actual).get("callable", _M) is not self._original_callable_builtin:
            _reject()

    def _cache(self, name):
        if _module(sys).get("modules", _M) is not _CACHE:
            _reject()
        if name in _dict(_CACHE, 4096):
            _reject()

    def _lookup(self, name):
        _module_name(name)
        self._policy()
        for _, data, ns, _ in self._imports:
            if name == data[0]:
                return data[1], ns
        _reject()

    def _plain_import(self, name):
        module, _ = self._lookup(name)
        parts = name.split(".")
        if len(parts) == 1:
            return module
        root, parent_namespace = self._lookup(parts[0])
        for index in range(1, len(parts)):
            child, child_namespace = self._lookup(".".join(parts[: index + 1]))
            if parent_namespace.get(parts[index], _M) is not child:
                _reject()
            parent_namespace = child_namespace
        return root

    def prepare(
        self,
        source: bytes,
        expected_sha256: str,
        module_name: str,
        filename: str,
        roots: tuple[str, ...],
    ) -> PreparedFixtureSource:
        if (
            type(source) is not bytes
            or type(expected_sha256) is not str
            or type(filename) is not str
        ):
            _foreign()
        if (
            not 1 <= len(source) <= 16384
            or not 1 <= len(filename) <= 256
            or len(expected_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha256)
        ):
            _reject()
        _name(module_name)
        _names(roots, 1)
        if len(self._items) >= self._cap or hashlib.sha256(source).hexdigest() != expected_sha256:
            _reject()
        self._policy()
        self._cache(module_name)
        text = source.decode("utf-8", "strict")
        _preflight(self, ast.parse(text, filename=filename, mode="exec"), roots)
        code = _COMPILE(text, filename, "exec", dont_inherit=True)
        prepared = _P(module_name, filename, expected_sha256, code, _codes(code, roots))
        self._policy()
        self._items.append(_Issue(prepared, _values(prepared, _P)))
        return prepared

    def _itemd(self, prepared):
        _exact(prepared, _P)
        for item in self._items:
            if prepared is item.row:
                _prepared(item)
                return item
        _foreign()

    def _import(self, name, fromlist, level):
        try:
            if type(level) is not int or level != 0:
                _reject()
            _names(fromlist)
            if not fromlist:
                return self._plain_import(name)
            module, ns = self._lookup(name)
            if any(item not in ns for item in fromlist):
                _reject()
            return module
        except (TypeError, ValueError):
            raise ImportError("source import denied") from None

    def execute(self, prepared: PreparedFixtureSource) -> LoadedFixtureSource:
        item = self._itemd(prepared)
        if item.spent:
            _reject()
        self._cache(item.data[0])
        self._policy()
        item.spent = True
        module = ModuleType(item.data[0])
        ns = _module(module)
        ns["__name__"] = item.data[0]

        def importer(name, globals=None, locals=None, fromlist=(), level=0):
            return self._import(name, () if fromlist is None else fromlist, level)

        builtin_namespace = dict(_BUILTINS, __import__=importer)
        if self._original_callable_builtin is not None:
            builtin_namespace["callable"] = self._original_callable_builtin
            item.builtin_namespace = builtin_namespace
        ns["__builtins__"] = builtin_namespace
        self._injected_builtins(item, ns)
        _EXEC(item.data[3], ns, ns)
        self._injected_builtins(item, ns)
        self._cache(item.data[0])
        self._policy()
        loaded = _L(prepared, module, ns, _roots(self, item, module, ns))
        self._policy()
        self._injected_builtins(item, ns)
        item.loaded, item.saved = loaded, _values(loaded, _L)
        return loaded

    def require_current(self, loaded: LoadedFixtureSource) -> None:
        _exact(loaded, _L)
        for item in self._items:
            if loaded is item.loaded:
                data = _same(loaded, item.saved, _L)
                _prepared(item)
                self._cache(item.data[0])
                self._policy()
                self._injected_builtins(item, data[2])
                _roots(self, item, data[1], data[2], data[3])
                self._policy()
                self._injected_builtins(item, data[2])
                return
        _foreign()
