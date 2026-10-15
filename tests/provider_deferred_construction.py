"""Deferred native role admission under the original one-use construction guard.

Inputs are issued public source/class rows and predeclared constant paths.
Outputs are native role arguments and a finite module/admission state machine.
No profiler, private loader/helper access, provider dispatch or birth proof.
"""

from dataclasses import dataclass, field
from itertools import repeat
from types import CodeType, FrameType, FunctionType, MemberDescriptorType, MethodType
from typing import Any, NoReturn

from provider_code_path import pin_provider_code_path
from provider_source_classes import (
    FixtureSourceClasses,
    LoadedSourceClasses,
    PreparedSourceClasses,
    SourceClassBinding,
    SourceClassCode,
)
from provider_source_loader import LoadedFixtureSource, PreparedFixtureSource

_TYPE_DICT = type.__dict__["__dict__"]
_MEMBER_GET = _TYPE_DICT.__get__(MemberDescriptorType, type)["__get__"]
_UNEXECUTED = FixtureSourceClasses.require_unexecuted
_CURRENT = FixtureSourceClasses.require_current
_BINDING = FixtureSourceClasses.require_binding
_BATCH = FixtureSourceClasses.require_current_bindings
_ERROR = "invalid deferred construction witness"
_MISSING = object()


@dataclass(frozen=True, slots=True)
class DeferredConstructionRole:
    source_classes: FixtureSourceClasses
    prepared: PreparedSourceClasses
    class_index: int
    caller_root: str
    caller_indexes: tuple[int, ...]
    max_instances: int
    callback_slot: str | None = None
    callback_argument: str | None = None
    callback_root: str | None = None
    callback_indexes: tuple[int, ...] | None = None


def _readers(kind: type, names: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(_TYPE_DICT.__get__(kind, type)[name] for name in names)


_FIELDS = {
    DeferredConstructionRole: _readers(
        DeferredConstructionRole,
        (
            "source_classes",
            "prepared",
            "class_index",
            "caller_root",
            "caller_indexes",
            "max_instances",
            "callback_slot",
            "callback_argument",
            "callback_root",
            "callback_indexes",
        ),
    ),
    PreparedSourceClasses: _readers(PreparedSourceClasses, ("source", "classes")),
    SourceClassCode: _readers(
        SourceClassCode,
        (
            "slot",
            "class_code",
            "constructor_code",
            "call_code",
        ),
    ),
    PreparedFixtureSource: _readers(
        PreparedFixtureSource,
        (
            "module_name",
            "filename",
            "source_sha256",
            "module_code",
            "root_codes",
        ),
    ),
    LoadedSourceClasses: _readers(LoadedSourceClasses, ("prepared", "source", "classes")),
    LoadedFixtureSource: _readers(
        LoadedFixtureSource,
        (
            "prepared",
            "module",
            "namespace",
            "roots",
        ),
    ),
    SourceClassBinding: _readers(
        SourceClassBinding,
        (
            "namespace",
            "code",
            "owner_type",
            "descriptor",
            "constructor",
            "raw_call",
        ),
    ),
}
_CONSTANTS, _VARS, _ARGS, _KEY_ARGS, _BYTES = _readers(
    CodeType,
    (
        "co_consts",
        "co_varnames",
        "co_argcount",
        "co_kwonlyargcount",
        "co_code",
    ),
)
_FN_CODE, _FN_GLOBALS = _readers(FunctionType, ("__code__", "__globals__"))
_FRAME_CODE, _FRAME_GLOBALS, _FRAME_PARENT, _FRAME_LASTI = _readers(
    FrameType,
    (
        "f_code",
        "f_globals",
        "f_back",
        "f_lasti",
    ),
)
_METHOD_FUNC, _METHOD_SELF = _readers(MethodType, ("__func__", "__self__"))


def _reject() -> NoReturn:
    raise ValueError(_ERROR)


def _exact(value: object, kind: type) -> None:
    if type(value) is not kind:
        raise TypeError(_ERROR)


def _values(row: object, kind: type) -> tuple:
    _exact(row, kind)
    try:
        return tuple(map(_MEMBER_GET, _FIELDS[kind], repeat(row), repeat(kind)))
    except AttributeError:
        raise ValueError(_ERROR) from None


def _same(row: object, saved: tuple, kind: type) -> tuple:
    current = _values(row, kind)
    if any(value is not old for value, old in zip(current, saved)):
        _reject()
    return current


def _name(name: Any) -> None:
    _exact(name, str)
    if not 0 < len(name) <= 128 or not name.isascii() or not name.isidentifier():
        _reject()


def _keys(namespace: Any) -> None:
    if len(namespace) > 512 or any(type(key) is not str or len(key) > 128 for key in namespace):
        _reject()


def _dictionary(namespace: Any) -> None:
    _exact(namespace, dict)
    _keys(namespace)


def _constants(code: Any) -> tuple:
    _exact(code, CodeType)
    constants = _CONSTANTS.__get__(code, CodeType)
    _exact(constants, tuple)
    if len(constants) > 512:
        _reject()
    return constants


def _path(code: CodeType, indexes: Any) -> CodeType:
    _exact(indexes, tuple)
    if len(indexes) > 16:
        _reject()
    token = code
    for index in indexes:
        if type(index) is not int or not 0 <= index <= 511:
            _reject()
        constants = _constants(token)
        if index >= len(constants):
            _reject()
        token = constants[index]
        _exact(token, CodeType)
    _constants(token)
    return token


def _root(roots: tuple, name: Any) -> tuple[int, CodeType]:
    _name(name)
    matches = [(index, code) for index, (slot, code) in enumerate(roots) if slot == name]
    if len(matches) != 1:
        _reject()
    return matches[0]


@dataclass
class _Checker:
    method: MethodType
    owner: FixtureSourceClasses
    function: FunctionType
    code: CodeType
    namespace: dict

    def require(self) -> None:
        if (
            _METHOD_FUNC.__get__(self.method, MethodType) is not self.function
            or _METHOD_SELF.__get__(self.method, MethodType) is not self.owner
            or _FN_CODE.__get__(self.function, FunctionType) is not self.code
            or _FN_GLOBALS.__get__(self.function, FunctionType) is not self.namespace
        ):
            _reject()


def _checker(function: Any, owner: FixtureSourceClasses) -> _Checker:
    _exact(function, FunctionType)
    return _Checker(
        MethodType(function, owner),
        owner,
        function,
        _FN_CODE.__get__(function, FunctionType),
        _FN_GLOBALS.__get__(function, FunctionType),
    )


@dataclass
class _Module:
    helper: FixtureSourceClasses
    prepared: PreparedSourceClasses
    prepared_values: tuple
    base_values: tuple
    class_values: tuple
    roots: tuple
    current: _Checker
    binding: _Checker
    unexecuted: _Checker
    batch: _Checker
    phase: str = "unseen"
    frame: FrameType | None = None
    namespace: dict | None = None
    seen: list[CodeType] = field(default_factory=list)
    loaded: LoadedSourceClasses | None = None
    loaded_values: tuple = ()
    loaded_base_values: tuple = ()
    binding_values: tuple = ()

    @property
    def code(self) -> CodeType:
        return self.base_values[3]

    def require(self) -> None:
        base, classes = _same(self.prepared, self.prepared_values, PreparedSourceClasses)
        _same(base, self.base_values, PreparedFixtureSource)
        for row, saved in zip(classes, self.class_values):
            _same(row, saved, SourceClassCode)
        self.current.require()
        self.binding.require()
        self.unexecuted.require()
        self.batch.require()
        if self.phase == "running":
            _require_frame(self.frame, self.code, self.namespace)
        elif self.frame is not None:
            _reject()
        if self.phase == "admitted":
            _same(self.loaded, self.loaded_values, LoadedSourceClasses)
            _same(self.loaded_values[1], self.loaded_base_values, LoadedFixtureSource)
            self.batch.method(self.loaded)
            for row, saved in zip(self.loaded_values[2], self.binding_values):
                _same(row, saved, SourceClassBinding)


@dataclass
class _Plan:
    row: DeferredConstructionRole
    values: tuple
    source: _Module
    class_values: tuple
    caller_index: int
    caller_code: CodeType
    caller_token: CodeType
    callback_index: int | None
    callback_code: CodeType | None
    callback_token: CodeType | None


def _require_frame(frame: Any, code: CodeType, namespace: Any) -> None:
    _exact(frame, FrameType)
    _dictionary(namespace)
    if (
        _FRAME_CODE.__get__(frame, FrameType) is not code
        or _FRAME_GLOBALS.__get__(frame, FrameType) is not namespace
    ):
        _reject()


def _normal_return(frame: FrameType, code: CodeType, arg: object, returns: tuple) -> bool:
    raw = _BYTES.__get__(code, CodeType)
    lasti = _FRAME_LASTI.__get__(frame, FrameType)
    if type(raw) is not bytes or len(raw) > 65536 or type(lasti) is not int:
        _reject()
    return arg is None and 0 <= lasti < len(raw) and lasti % 2 == 0 and raw[lasti] in returns


def _module(helper: Any, prepared: Any) -> _Module:
    _exact(helper, FixtureSourceClasses)
    prepared_values = _values(prepared, PreparedSourceClasses)
    base_values = _values(prepared_values[0], PreparedFixtureSource)
    classes = prepared_values[1]
    _exact(classes, tuple)
    if not 1 <= len(classes) <= 16:
        _reject()
    class_values = tuple(_values(row, SourceClassCode) for row in classes)
    roots = base_values[4]
    _exact(roots, tuple)
    if not 1 <= len(roots) <= 16:
        _reject()
    for pair in roots:
        _exact(pair, tuple)
        if len(pair) != 2:
            _reject()
        _name(pair[0])
        _constants(pair[1])
    current, binding, unexecuted, batch = (
        _checker(function, helper) for function in (_CURRENT, _BINDING, _UNEXECUTED, _BATCH)
    )
    unexecuted.require()
    unexecuted.method(prepared)
    return _Module(
        helper,
        prepared,
        prepared_values,
        base_values,
        class_values,
        roots,
        current,
        binding,
        unexecuted,
        batch,
    )


def _method_shape(code: CodeType) -> None:
    _constants(code)
    names = _VARS.__get__(code, CodeType)
    raw = _BYTES.__get__(code, CodeType)
    if (
        type(names) is not tuple
        or not names
        or len(names) > 512
        or any(type(name) is not str or len(name) > 128 for name in names)
        or names[0] != "self"
        or _ARGS.__get__(code, CodeType) < 1
        or type(raw) is not bytes
        or len(raw) > 65536
    ):
        _reject()


def _plan(row: DeferredConstructionRole, source: _Module) -> _Plan:
    values = _values(row, DeferredConstructionRole)
    index, caller, indexes, cap = values[2:6]
    if type(index) is not int or not 0 <= index < len(source.class_values):
        _reject()
    if type(cap) is not int or not 1 <= cap <= 64:
        _reject()
    class_values = source.class_values[index]
    for code in class_values[2:4]:
        _method_shape(code)
    caller_index, caller_code = _root(source.roots, caller)
    caller_token = _path(caller_code, indexes)
    slot, argument, callback, callback_indexes = values[6:]
    callback_index: int | None = None
    callback_code: CodeType | None = None
    callback_token: CodeType | None = None
    if not all(value is None for value in values[6:]):
        _name(slot)
        _name(argument)
        if slot in ("__class__", "__dict__", "__weakref__", "__init__", "__call__"):
            _reject()
        names = _VARS.__get__(class_values[2], CodeType)
        _exact(names, tuple)
        if len(names) > 512 or any(type(name) is not str or len(name) > 128 for name in names):
            _reject()
        count = _ARGS.__get__(class_values[2], CodeType) + _KEY_ARGS.__get__(
            class_values[2], CodeType
        )
        if type(count) is not int or not 1 <= count <= len(names):
            _reject()
        if argument == "self" or argument not in names[1:count]:
            _reject()
        callback_index, callback_code = _root(source.roots, callback)
        callback_token = _path(callback_code, callback_indexes)
    return _Plan(
        row,
        values,
        source,
        class_values,
        caller_index,
        caller_code,
        caller_token,
        callback_index,
        callback_code,
        callback_token,
    )


@dataclass
class AdmissionBatch:
    source: _Module
    loaded: LoadedSourceClasses
    loaded_values: tuple
    base_values: tuple
    binding_values: tuple
    indexes: tuple[int, ...]
    arguments: tuple[tuple, ...]


class DeferredAdmission:
    """Private declarations; role construction stays in the existing profile."""

    modules: tuple[_Module, ...]
    plans: tuple[_Plan, ...]

    def __init__(self, rows: tuple[DeferredConstructionRole, ...]) -> None:
        _exact(rows, tuple)
        if len(rows) > 64:
            _reject()
        modules: list[_Module] = []
        plans = []
        for row in rows:
            values = _values(row, DeferredConstructionRole)
            source = next((item for item in modules if item.prepared is values[1]), None)
            if source is None:
                source = _module(values[0], values[1])
                if any(source.code is old.code for old in modules):
                    _reject()
                modules.append(source)
            elif source.helper is not values[0]:
                _reject()
            plans.append(_plan(row, source))
        self.modules = tuple(modules)
        self.plans = tuple(plans)

    @property
    def capacities(self) -> tuple[int, ...]:
        return tuple(plan.values[5] for plan in self.plans)

    @property
    def role_tokens(self) -> tuple[CodeType, ...]:
        return tuple(token for plan in self.plans for token in plan.class_values[2:4])

    @property
    def caller_tokens(self) -> tuple[CodeType, ...]:
        return tuple(plan.caller_token for plan in self.plans)

    @property
    def callback_tokens(self) -> tuple[CodeType, ...]:
        return tuple(plan.callback_token for plan in self.plans if plan.callback_token is not None)

    @property
    def required_tokens(self) -> tuple[CodeType, ...]:
        return (
            tuple(source.code for source in self.modules)
            + tuple(plan.class_values[1] for plan in self.plans)
            + self.role_tokens
            + self.caller_tokens
            + self.callback_tokens
        )

    def pending_module(self) -> _Module | None:
        return next((source for source in self.modules if source.phase == "running"), None)

    def require_current(self) -> None:
        for plan in self.plans:
            _same(plan.row, plan.values, DeferredConstructionRole)
        for source in self.modules:
            source.require()

    def observe(
        self,
        frame: FrameType,
        event: str,
        arg: object,
        after_hook: bool,
        main_thread: bool,
        pending_constructor: bool,
        returns: tuple,
    ) -> None:
        code = _FRAME_CODE.__get__(frame, FrameType)
        namespace = _FRAME_GLOBALS.__get__(frame, FrameType)
        for source in self.modules:
            if code is source.code:
                if not main_thread or pending_constructor:
                    _reject()
                if event == "call":
                    if not after_hook:
                        if source.phase != "unseen" or self.pending_module() is not None:
                            _reject()
                        _dictionary(namespace)
                        source.namespace, source.frame, source.phase = namespace, frame, "running"
                    _require_frame(source.frame, source.code, source.namespace)
                    if frame is not source.frame:
                        _reject()
                elif event == "return":
                    _require_frame(source.frame, source.code, source.namespace)
                    if frame is not source.frame:
                        _reject()
                    normal = _normal_return(frame, source.code, arg, returns)
                    if normal and any(
                        not any(token is old for old in source.seen)
                        for token in (
                            plan.class_values[1] for plan in self.plans if plan.source is source
                        )
                    ):
                        _reject()
                    if after_hook:
                        source.frame = None
                        source.phase = "complete" if normal else "failed"
                return
        for plan in self.plans:
            source = plan.source
            if code is plan.class_values[1] and event in ("call", "return"):
                if (
                    not main_thread
                    or source.phase != "running"
                    or _FRAME_PARENT.__get__(frame, FrameType) is not source.frame
                ):
                    _reject()
                _require_frame(frame, code, source.namespace)
                if event == "call" and not after_hook:
                    if any(code is old for old in source.seen):
                        _reject()
                    source.seen.append(code)
                return
            tokens: tuple[CodeType, ...] = (
                plan.class_values[2],
                plan.class_values[3],
                plan.caller_token,
            )
            if plan.callback_token is not None:
                tokens += (plan.callback_token,)
            if event == "call" and any(code is token for token in tokens):
                if (
                    not main_thread
                    or source.phase != "admitted"
                    or self.pending_module() is not None
                ):
                    _reject()
                _require_frame(frame, code, source.namespace)

    def prepare_admission(self, loaded: LoadedSourceClasses) -> AdmissionBatch:
        values = _values(loaded, LoadedSourceClasses)
        source = next((item for item in self.modules if item.prepared is values[0]), None)
        if source is None:
            raise TypeError(_ERROR)
        if source.phase != "complete" or source.frame is not None:
            _reject()
        self.require_current()
        source.current.method(loaded)
        base_values = _values(values[1], LoadedFixtureSource)
        if (
            base_values[0] is not source.prepared_values[0]
            or base_values[2] is not source.namespace
        ):
            _reject()
        bindings = values[2]
        _exact(bindings, tuple)
        if len(bindings) != len(source.class_values):
            _reject()
        binding_values = tuple(_values(row, SourceClassBinding) for row in bindings)
        roots = base_values[3]
        _exact(roots, tuple)
        if len(roots) != len(source.roots):
            _reject()
        indexes, arguments = [], []
        for index, plan in enumerate(self.plans):
            if plan.source is not source:
                continue
            binding = bindings[plan.values[2]]
            source.binding.method(binding)
            fields = binding_values[plan.values[2]]
            if (
                fields[0] is not source.namespace
                or fields[1] is not source.prepared_values[1][plan.values[2]]
            ):
                _reject()
            root = roots[plan.caller_index]
            _exact(root, FunctionType)
            caller = pin_provider_code_path(root, plan.caller_code, plan.values[4])
            if _FN_GLOBALS.__get__(root, FunctionType) is not source.namespace:
                _reject()
            callback_path = None
            if plan.callback_index is not None:
                callback_root = roots[plan.callback_index]
                _exact(callback_root, FunctionType)
                callback_code = plan.callback_code
                if callback_code is None:
                    _reject()
                callback_path = pin_provider_code_path(callback_root, callback_code, plan.values[9])
                if _FN_GLOBALS.__get__(callback_root, FunctionType) is not source.namespace:
                    _reject()
            arguments.append(
                (
                    fields[2],
                    fields[3],
                    fields[4],
                    plan.class_values[2],
                    caller,
                    fields[5],
                    plan.class_values[3],
                    plan.values[5],
                    plan.values[6],
                    plan.values[7],
                    callback_path,
                )
            )
            indexes.append(index)
        self.require_current()
        source.current.method(loaded)
        return AdmissionBatch(
            source,
            loaded,
            values,
            base_values,
            binding_values,
            tuple(indexes),
            tuple(arguments),
        )

    def commit(self, batch: AdmissionBatch) -> None:
        # Permission changes only after all native roles exist in the profile.
        batch.source.loaded = batch.loaded
        batch.source.loaded_values = batch.loaded_values
        batch.source.loaded_base_values = batch.base_values
        batch.source.binding_values = batch.binding_values
        batch.source.phase = "admitted"
