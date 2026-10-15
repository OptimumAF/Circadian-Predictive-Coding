"""Retain compiled class/method code before execution and native originals after it.

Inputs are a public source loader, verified bytes and explicit class/function roots.
Outputs are privately issued one-use rows and read-only continuity checks.
No profiling, constructor/provider dispatch, sandbox or physical/birth proof.
"""

import ast
from dataclasses import dataclass
import hashlib
from itertools import repeat
from types import (
    CodeType,
    FunctionType,
    GetSetDescriptorType,
    MappingProxyType,
    MemberDescriptorType,
)
from typing import Any

from provider_source_loader import (
    FixtureSourceLoader,
    LoadedFixtureSource,
    PreparedFixtureSource,
)

_TYPE_DICT = type.__dict__["__dict__"]
_MEMBER_GET = _TYPE_DICT.__get__(MemberDescriptorType, type)["__get__"]
_PREPARE = FixtureSourceLoader.prepare
_EXECUTE = FixtureSourceLoader.execute
_CURRENT = FixtureSourceLoader.require_current
_ERROR = "invalid source class witness"
_MISSING = object()


@dataclass(frozen=True, slots=True)
class SourceClassCode:
    slot: str
    class_code: CodeType
    constructor_code: CodeType
    call_code: CodeType


@dataclass(frozen=True, slots=True)
class PreparedSourceClasses:
    source: PreparedFixtureSource
    classes: tuple[SourceClassCode, ...]


@dataclass(frozen=True, slots=True)
class SourceClassBinding:
    namespace: dict[str, object]
    code: SourceClassCode
    owner_type: type
    descriptor: GetSetDescriptorType
    constructor: FunctionType
    raw_call: FunctionType


@dataclass(frozen=True, slots=True)
class LoadedSourceClasses:
    prepared: PreparedSourceClasses
    source: LoadedFixtureSource
    classes: tuple[SourceClassBinding, ...]


def _readers(kind: type, names: tuple[str, ...]) -> tuple[Any, ...]:
    namespace = _TYPE_DICT.__get__(kind, type)
    return tuple(namespace[name] for name in names)


_FIELDS = {
    SourceClassCode: _readers(
        SourceClassCode, ("slot", "class_code", "constructor_code", "call_code")
    ),
    PreparedSourceClasses: _readers(PreparedSourceClasses, ("source", "classes")),
    SourceClassBinding: _readers(
        SourceClassBinding,
        ("namespace", "code", "owner_type", "descriptor", "constructor", "raw_call"),
    ),
    LoadedSourceClasses: _readers(LoadedSourceClasses, ("prepared", "source", "classes")),
    PreparedFixtureSource: _readers(
        PreparedFixtureSource,
        ("module_name", "filename", "source_sha256", "module_code", "root_codes"),
    ),
    LoadedFixtureSource: _readers(
        LoadedFixtureSource, ("prepared", "module", "namespace", "roots")
    ),
}
_CONSTANTS, _CODE_NAME, _VARS, _ARGS, _FLAGS, _FREE = _readers(
    CodeType, ("co_consts", "co_name", "co_varnames", "co_argcount", "co_flags", "co_freevars")
)
_FN_CODE, _GLOBALS, _DEFAULTS, _KEYWORDS, _CLOSURE = _readers(
    FunctionType, ("__code__", "__globals__", "__defaults__", "__kwdefaults__", "__closure__")
)
_BASES = _readers(type, ("__bases__",))[0]
_DESC_OWNER, _DESC_NAME = _readers(GetSetDescriptorType, ("__objclass__", "__name__"))


def _reject() -> None:
    raise ValueError(_ERROR)


def _exact(value: object, kind: type) -> None:
    if type(value) is not kind:
        raise TypeError(_ERROR)


def _values(row: object, kind: type) -> tuple[Any, ...]:
    _exact(row, kind)
    try:
        return tuple(map(_MEMBER_GET, _FIELDS[kind], repeat(row), repeat(kind)))
    except AttributeError:
        raise ValueError(_ERROR) from None


def _same(row: object, saved: tuple, kind: type) -> tuple[Any, ...]:
    current = _values(row, kind)
    if any(value is not old for value, old in zip(current, saved)):
        _reject()
    return current


def _name(name: Any) -> None:
    _exact(name, str)
    if (
        not 0 < len(name) <= 128
        or not name.isascii()
        or not name.isidentifier()
        or name.startswith("__")
    ):
        _reject()


def _names(names: Any) -> None:
    _exact(names, tuple)
    if not 1 <= len(names) <= 16:
        _reject()
    for name in names:
        _name(name)
    if len(set(names)) != len(names):
        _reject()


def _snapshot(namespace: Any) -> tuple:
    if len(namespace) > 512 or any(type(key) is not str or len(key) > 128 for key in namespace):
        _reject()
    return tuple(namespace.items())


def _dictionary(namespace: Any) -> None:
    _exact(namespace, dict)
    _snapshot(namespace)


def _class_namespace(owner: Any) -> Any:
    _exact(owner, type)
    namespace = _TYPE_DICT.__get__(owner, type)
    _exact(namespace, MappingProxyType)
    _snapshot(namespace)
    return namespace


def _method_syntax(node: Any) -> None:
    args = node.args
    positional = args.posonlyargs + args.args
    all_args = positional + args.kwonlyargs
    all_args += [arg for arg in (args.vararg, args.kwarg) if arg is not None]
    if (
        not positional
        or positional[0].arg != "self"
        or node.decorator_list
        or args.defaults
        or any(value is not None for value in args.kw_defaults)
        or node.returns is not None
        or any(arg.annotation is not None for arg in all_args)
        or getattr(node, "type_params", ())
    ):
        _reject()
    pending = list(node.body)
    while pending:
        part = pending.pop()
        if type(part) in (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef):
            continue
        if type(part) in (ast.Yield, ast.YieldFrom, ast.Await):
            _reject()
        pending.extend(ast.iter_child_nodes(part))


def _class_syntax(node: Any) -> None:
    if (
        node.decorator_list
        or node.keywords
        or getattr(node, "type_params", ())
        or len(node.bases) > 1
        or any(type(base) is not ast.Name or base.id != "object" for base in node.bases)
    ):
        _reject()
    body = node.body
    if (
        body
        and type(body[0]) is ast.Expr
        and type(body[0].value) is ast.Constant
        and type(body[0].value.value) is str
    ):
        body = body[1:]
    if (
        len(body) != 2
        or any(type(method) is not ast.FunctionDef for method in body)
        or {method.name for method in body} != {"__init__", "__call__"}
    ):
        _reject()
    for method in body:
        _method_syntax(method)


def _eager_loads(node: Any, roots: tuple[str, ...], deferred: bool = False) -> None:
    if type(node) is ast.Name and type(node.ctx) is ast.Load:
        if node.id in roots and not deferred:
            _reject()
    # Why field traversal: function headers run in the enclosing context.
    if type(node) in (ast.FunctionDef, ast.AsyncFunctionDef):
        for field, value in ast.iter_fields(node):
            children = value if type(value) is list else [value]
            for child in children:
                if isinstance(child, ast.AST):
                    _eager_loads(child, roots, True if field == "body" else deferred)
        return
    for child in ast.iter_child_nodes(node):
        _eager_loads(child, roots, deferred)


def _preflight(text: str, filename: str, roots: tuple[str, ...]) -> None:
    tree: Any = ast.parse(text, filename=filename, mode="exec")
    node: Any
    admitted = []
    # Why protect object: literal bases must resolve to the trusted builtin.
    protected = (*roots, "object")
    for name in roots:
        found = [node for node in tree.body if type(node) is ast.ClassDef and node.name == name]
        if len(found) != 1:
            _reject()
        _class_syntax(found[0])
        admitted.append(found[0])
    for index, node in enumerate(ast.walk(tree)):
        if index >= 4096:
            _reject()
        kind = type(node)
        bound = []
        if kind is ast.Name and type(node.ctx) in (ast.Store, ast.Del):
            bound.append(node.id)
        if kind in (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef):
            if node.name == "object" or not any(node is original for original in admitted):
                bound.append(node.name)
        if kind is ast.alias:
            bound.append(node.asname or node.name)
        for cls, field in (
            (ast.arg, "arg"),
            (ast.ExceptHandler, "name"),
            (ast.MatchAs, "name"),
            (ast.MatchStar, "name"),
            (ast.MatchMapping, "rest"),
        ):
            if isinstance(node, cls):
                bound.append(getattr(node, field))
        if kind in (ast.Global, ast.Nonlocal):
            bound.extend(node.names)
        if kind.__name__ in ("TypeVar", "ParamSpec", "TypeVarTuple"):
            bound.append(node.name)
        if any(name in protected for name in bound if name is not None):
            _reject()
        if kind is ast.Attribute and type(node.ctx) in (ast.Store, ast.Del):
            base = node.value
            while type(base) in (ast.Attribute, ast.Subscript):
                base = base.value
            if type(base) is ast.Name and base.id in protected:
                _reject()
    _eager_loads(tree, roots)


def _constants(code: Any) -> tuple:
    _exact(code, CodeType)
    constants = _CONSTANTS.__get__(code, CodeType)
    _exact(constants, tuple)
    if len(constants) > 512:
        _reject()
    return constants


def _child(code: CodeType, name: str) -> CodeType:
    found = [
        child
        for child in _constants(code)
        if type(child) is CodeType and _CODE_NAME.__get__(child, CodeType) == name
    ]
    if len(found) != 1:
        _reject()
    _constants(found[0])
    return found[0]


def _method_code(code: CodeType) -> None:
    _constants(code)
    names = _VARS.__get__(code, CodeType)
    free = _FREE.__get__(code, CodeType)
    if (
        type(names) is not tuple
        or not names
        or names[0] != "self"
        or _ARGS.__get__(code, CodeType) < 1
        or _FLAGS.__get__(code, CodeType) & (0x20 | 0x80 | 0x200)
        or type(free) is not tuple
        or free
    ):
        _reject()


def _check_function(function: Any, code: CodeType, namespace: dict) -> None:
    _exact(function, FunctionType)
    if (
        _FN_CODE.__get__(function, FunctionType) is not code
        or _GLOBALS.__get__(function, FunctionType) is not namespace
        or _DEFAULTS.__get__(function, FunctionType) is not None
        or _KEYWORDS.__get__(function, FunctionType) is not None
        or _CLOSURE.__get__(function, FunctionType) is not None
    ):
        _reject()


@dataclass
class _Capture:
    binding: SourceClassBinding
    fields: tuple
    snapshot: tuple
    bases: tuple


@dataclass
class _Issue:
    row: PreparedSourceClasses
    fields: tuple
    source_fields: tuple
    codes: tuple
    spent: bool = False
    loaded: LoadedSourceClasses | None = None
    loaded_fields: tuple = ()
    base_fields: tuple = ()
    captures: tuple[_Capture, ...] = ()


def _prepared(issue: _Issue) -> None:
    source, classes = _same(issue.row, issue.fields, PreparedSourceClasses)
    module = _same(source, issue.source_fields, PreparedFixtureSource)[3]
    for row, saved in zip(classes, issue.codes):
        slot, class_code, init_code, call_code = _same(row, saved, SourceClassCode)
        if (
            _child(module, slot) is not class_code
            or _child(class_code, "__init__") is not init_code
            or _child(class_code, "__call__") is not call_code
        ):
            _reject()
        _method_code(init_code)
        _method_code(call_code)


def _binding_fields(binding: SourceClassBinding) -> tuple:
    namespace, code, owner, descriptor, init, call = _values(binding, SourceClassBinding)
    _dictionary(namespace)
    slot, _, init_code, call_code = _values(code, SourceClassCode)
    native = _class_namespace(owner)
    bases = _BASES.__get__(owner, type)
    if (
        type(bases) is not tuple
        or len(bases) != 1
        or bases[0] is not object
        or namespace.get(slot, _MISSING) is not owner
        or native.get("__dict__", _MISSING) is not descriptor
        or native.get("__init__", _MISSING) is not init
        or native.get("__call__", _MISSING) is not call
    ):
        _reject()
    _exact(descriptor, GetSetDescriptorType)
    name = _DESC_NAME.__get__(descriptor, GetSetDescriptorType)
    if (
        _DESC_OWNER.__get__(descriptor, GetSetDescriptorType) is not owner
        or type(name) is not str
        or name != "__dict__"
    ):
        _reject()
    _check_function(init, init_code, namespace)
    _check_function(call, call_code, namespace)
    return native, bases


class FixtureSourceClasses:
    """Compose public loader ownership with permanent, finite class capture."""

    def __init__(self, loader: FixtureSourceLoader, max_classes: int = 16) -> None:
        _exact(loader, FixtureSourceLoader)
        _exact(max_classes, int)
        if not 1 <= max_classes <= 16:
            _reject()
        self._loader = loader
        self._capacity = max_classes
        self._reserved = 0
        self._issues: list[_Issue] = []

    def prepare(
        self,
        source: bytes,
        expected_sha256: str,
        module_name: str,
        filename: str,
        roots: tuple[str, ...],
        *,
        class_roots: tuple[str, ...],
    ) -> PreparedSourceClasses:
        _exact(source, bytes)
        _exact(expected_sha256, str)
        _exact(filename, str)
        _names(roots)
        _names(class_roots)
        _name(module_name)
        if (
            not 1 <= len(source) <= 16384
            or not 1 <= len(filename) <= 256
            or len(expected_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha256)
            or any(name in roots for name in class_roots)
            or len(self._issues) >= 16
            or self._reserved + len(class_roots) > self._capacity
            or hashlib.sha256(source).hexdigest() != expected_sha256
        ):
            _reject()
        _preflight(source.decode("utf-8", "strict"), filename, class_roots)
        prepared = _PREPARE(self._loader, source, expected_sha256, module_name, filename, roots)
        source_fields = _values(prepared, PreparedFixtureSource)
        module = source_fields[3]
        codes = []
        for slot in class_roots:
            class_code = _child(module, slot)
            init_code = _child(class_code, "__init__")
            call_code = _child(class_code, "__call__")
            _method_code(init_code)
            _method_code(call_code)
            codes.append(SourceClassCode(slot, class_code, init_code, call_code))
        row = PreparedSourceClasses(prepared, tuple(codes))
        issue = _Issue(
            row,
            _values(row, PreparedSourceClasses),
            source_fields,
            tuple(_values(code, SourceClassCode) for code in codes),
        )
        _prepared(issue)
        self._issues.append(issue)
        self._reserved += len(codes)
        return row

    def _issued(self, prepared: PreparedSourceClasses) -> _Issue:
        _exact(prepared, PreparedSourceClasses)
        for issue in self._issues:
            if prepared is issue.row:
                _prepared(issue)
                return issue
        raise TypeError(_ERROR)

    def require_unexecuted(self, prepared: PreparedSourceClasses) -> None:
        """Reprove original issuance without renewing or running the source."""
        issue = self._issued(prepared)
        if issue.spent or issue.loaded is not None:
            _reject()

    def execute(self, prepared: PreparedSourceClasses) -> LoadedSourceClasses:
        issue = self._issued(prepared)
        if issue.spent:
            _reject()
        # Why before dispatch: failed module or capture work cannot renew this handle.
        issue.spent = True
        loaded = _EXECUTE(self._loader, issue.fields[0])
        _CURRENT(self._loader, loaded)
        base_fields = _values(loaded, LoadedFixtureSource)
        namespace = base_fields[2]
        _dictionary(namespace)
        captures = []
        for code in issue.fields[1]:
            slot = _values(code, SourceClassCode)[0]
            owner = namespace.get(slot, _MISSING)
            native = _class_namespace(owner)
            binding = SourceClassBinding(
                namespace,
                code,
                owner,
                native.get("__dict__", _MISSING),
                native.get("__init__", _MISSING),
                native.get("__call__", _MISSING),
            )
            native, bases = _binding_fields(binding)
            captures.append(
                _Capture(binding, _values(binding, SourceClassBinding), _snapshot(native), bases)
            )
        _prepared(issue)
        row = LoadedSourceClasses(prepared, loaded, tuple(item.binding for item in captures))
        issue.loaded, issue.loaded_fields = row, _values(row, LoadedSourceClasses)
        issue.base_fields, issue.captures = base_fields, tuple(captures)
        self.require_current(row)
        return row

    def _current(self, issue: _Issue) -> None:
        if issue.loaded is None:
            _reject()
        _same(issue.loaded, issue.loaded_fields, LoadedSourceClasses)
        _prepared(issue)
        _same(issue.loaded_fields[1], issue.base_fields, LoadedFixtureSource)
        _CURRENT(self._loader, issue.loaded_fields[1])
        for capture in issue.captures:
            _same(capture.binding, capture.fields, SourceClassBinding)
            native, bases = _binding_fields(capture.binding)
            if (
                bases is not capture.bases
                or len(native) != len(capture.snapshot)
                or any(native.get(key, _MISSING) is not value for key, value in capture.snapshot)
            ):
                _reject()

    def require_current(self, loaded: LoadedSourceClasses) -> None:
        _exact(loaded, LoadedSourceClasses)
        for issue in self._issues:
            if loaded is issue.loaded:
                self._current(issue)
                return
        raise TypeError(_ERROR)

    def require_binding(self, binding: SourceClassBinding) -> None:
        _exact(binding, SourceClassBinding)
        for issue in self._issues:
            if any(binding is capture.binding for capture in issue.captures):
                self._current(issue)
                return
        raise TypeError(_ERROR)

    def require_current_bindings(self, loaded: LoadedSourceClasses) -> None:
        """Reprove whole-source continuity and exact original row membership."""
        _exact(loaded, LoadedSourceClasses)
        owner = next((issue for issue in self._issues if loaded is issue.loaded), None)
        if owner is None:
            raise TypeError(_ERROR)
        self._current(owner)
        bindings = _values(loaded, LoadedSourceClasses)[2]
        _exact(bindings, tuple)
        if not 1 <= len(bindings) <= 16:
            _reject()
        # Why local identity tracking: sharing a proof never renews its freshness.
        validated = [owner]
        for binding in bindings:
            _exact(binding, SourceClassBinding)
            selected = next(
                (
                    issue
                    for issue in self._issues
                    if any(binding is capture.binding for capture in issue.captures)
                ),
                None,
            )
            if selected is None:
                raise TypeError(_ERROR)
            if not any(selected is previous for previous in validated):
                self._current(selected)
                validated.append(selected)
