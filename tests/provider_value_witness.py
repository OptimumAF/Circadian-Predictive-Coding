"""Retain primitive, declared record and original callback value witnesses.

Inputs are caller-attested native record metadata and a retained public code path.
Outputs are finitely issued value rows and fresh read-only continuity checks.
Native readers avoid candidate hooks; no value or callback body is dispatched.
No profile, IO, original-domain import, sandbox, birth/provenance or record method
internals/defaults/closures/global-content authority is established here.
"""

from dataclasses import dataclass
from itertools import repeat
from math import isfinite as _ISFINITE
from types import (
    CodeType,
    FunctionType,
    GetSetDescriptorType,
    MappingProxyType,
    MemberDescriptorType,
    ModuleType,
)
from typing import Any, NoReturn, cast

from provider_code_path import (
    ProviderCodePath,
    require_provider_callback_code,
    require_provider_code_path,
)


_TYPE_DICT = type.__dict__["__dict__"]
_MEMBER_GET = _TYPE_DICT.__get__(MemberDescriptorType, type)["__get__"]
_GETSET_GET = _TYPE_DICT.__get__(GetSetDescriptorType, type)["__get__"]
_MODULE_DICT = _TYPE_DICT.__get__(ModuleType, type)["__dict__"]
_BASES = _TYPE_DICT.__get__(type, type)["__bases__"]
_DESCRIPTOR_OWNER = _TYPE_DICT.__get__(GetSetDescriptorType, type)["__objclass__"]
_DESCRIPTOR_NAME = _TYPE_DICT.__get__(GetSetDescriptorType, type)["__name__"]
_FUNCTION_CODE = _TYPE_DICT.__get__(FunctionType, type)["__code__"]
_FUNCTION_GLOBALS = _TYPE_DICT.__get__(FunctionType, type)["__globals__"]
_REQUIRE_PATH = require_provider_code_path
_REQUIRE_CALLBACK = require_provider_callback_code
_ERROR = "invalid retained provider value witness"
_MISSING = object()

_FORBIDDEN_RECORD_ATTRIBUTES = (
    "__call__",
    "__new__",
    "__getattribute__",
    "__getattr__",
    "__slots__",
    "snapshots",
    "examples",
    "payload_bytes",
)


@dataclass(frozen=True, slots=True)
class ProviderValueBinding:
    mode: str
    value: object


_BINDING_FIELDS = tuple(
    _TYPE_DICT.__get__(ProviderValueBinding, type)[name] for name in ("mode", "value")
)
_PATH_FIELDS = tuple(
    _TYPE_DICT.__get__(ProviderCodePath, type)[name]
    for name in ("root", "root_code", "indexes", "token")
)


def _exact(value: object, kind: type) -> None:
    if type(value) is not kind:
        raise TypeError(_ERROR)


def _reject() -> NoReturn:
    raise ValueError(_ERROR)


def _read(reader: Any, value: object, kind: type) -> Any:
    if type(reader) is MemberDescriptorType:
        return _MEMBER_GET(reader, value, kind)
    if type(reader) is GetSetDescriptorType:
        return _GETSET_GET(reader, value, kind)
    raise TypeError(_ERROR)


def _row_values(row: object, kind: type, readers: tuple[Any, ...]) -> tuple[Any, ...]:
    _exact(row, kind)
    try:
        return tuple(map(_MEMBER_GET, readers, repeat(row), repeat(kind)))
    except AttributeError:
        raise ValueError(_ERROR) from None


def _same_references(current: tuple, saved: tuple) -> None:
    if len(current) != len(saved):
        _reject()
    for value, original in zip(current, saved):
        if value is not original:
            _reject()


def _require_keys(namespace: Any) -> None:
    if len(namespace) > 512:
        _reject()
    # Preflight all keys before lookup: foreign keys must never supply authority.
    for key in namespace:
        if type(key) is not str or len(key) > 128:
            _reject()


def _module_namespace(module: ModuleType) -> dict[str, object]:
    _exact(module, ModuleType)
    namespace = _read(_MODULE_DICT, module, ModuleType)
    _exact(namespace, dict)
    _require_keys(namespace)
    return namespace


def _class_namespace(kind: type) -> MappingProxyType:
    _exact(kind, type)
    namespace = _GETSET_GET(_TYPE_DICT, kind, type)
    _exact(namespace, MappingProxyType)
    _require_keys(namespace)
    return namespace


def _require_snapshot(namespace: Any, saved: tuple[tuple[str, object], ...]) -> None:
    _require_keys(namespace)
    if len(namespace) != len(saved):
        _reject()
    for key, original in saved:
        if namespace.get(key, _MISSING) is not original:
            _reject()


def _record_declaration(
    kind: type, descriptor: GetSetDescriptorType
) -> tuple[MappingProxyType, tuple]:
    namespace = _class_namespace(kind)
    _exact(descriptor, GetSetDescriptorType)
    bases = _read(_BASES, kind, type)
    _exact(bases, tuple)
    if len(bases) != 1 or bases[0] is not object:
        _reject()
    if any(name in namespace for name in _FORBIDDEN_RECORD_ATTRIBUTES):
        _reject()
    name = _read(_DESCRIPTOR_NAME, descriptor, GetSetDescriptorType)
    _exact(name, str)
    if (
        _read(_DESCRIPTOR_OWNER, descriptor, GetSetDescriptorType) is not kind
        or name != "__dict__"
        or namespace.get("__dict__", _MISSING) is not descriptor
    ):
        _reject()
    return namespace, bases


def _record_values(
    value: object, kind: type, descriptor: GetSetDescriptorType
) -> tuple[dict[str, object], tuple[int, int, int]]:
    _exact(value, kind)
    namespace = _GETSET_GET(descriptor, value, kind)
    _exact(namespace, dict)
    _require_keys(namespace)
    if len(namespace) != 3:
        _reject()
    snapshots = namespace.get("snapshots", _MISSING)
    examples = namespace.get("examples", _MISSING)
    payload_bytes = namespace.get("payload_bytes", _MISSING)
    for field_value in (snapshots, examples, payload_bytes):
        _exact(field_value, int)
        if field_value < 0:
            _reject()
    if snapshots == 0:
        if examples != 0 or payload_bytes != 0:
            _reject()
    elif examples < snapshots or payload_bytes == 0:
        _reject()
    return namespace, (snapshots, examples, payload_bytes)


@dataclass(slots=True)
class _Value:
    mode: str
    value: object
    storage: dict[str, object] | None = None
    fields: tuple[int, int, int] | None = None
    code: CodeType | None = None
    namespace: dict[str, object] | None = None


@dataclass(slots=True)
class _Issued:
    binding: ProviderValueBinding
    fields: tuple
    retained: _Value


class ProviderValueWitnesses:
    """One shared, permanently charged capacity; each require rechecks originals."""

    def __init__(
        self,
        record_module: ModuleType,
        record_attribute: str,
        record_type: type,
        descriptor: GetSetDescriptorType,
        callback_path: ProviderCodePath,
        *,
        max_values: int = 64,
    ) -> None:
        _exact(record_module, ModuleType)
        _exact(record_attribute, str)
        _exact(record_type, type)
        _exact(descriptor, GetSetDescriptorType)
        _exact(callback_path, ProviderCodePath)
        _exact(max_values, int)
        if not 1 <= len(record_attribute) <= 128 or not 1 <= max_values <= 64:
            _reject()
        module_namespace = _module_namespace(record_module)
        class_namespace, bases = _record_declaration(record_type, descriptor)
        if module_namespace.get(record_attribute, _MISSING) is not record_type:
            _reject()
        _REQUIRE_PATH(callback_path)
        path_fields = _row_values(callback_path, ProviderCodePath, _PATH_FIELDS)
        root_globals = _read(_FUNCTION_GLOBALS, path_fields[0], FunctionType)
        _exact(root_globals, dict)

        self._module = record_module
        self._attribute = record_attribute
        self._record_type = record_type
        self._descriptor = descriptor
        self._module_namespace = module_namespace
        self._module_snapshot = tuple(module_namespace.items())
        self._class_snapshot = tuple(class_namespace.items())
        self._bases = bases
        self._path = callback_path
        self._path_fields = path_fields
        self._root_globals = root_globals
        self._max_values = max_values
        self._attempts = 0
        self._issued: list[_Issued] = []
        self._require_declaration()

    @property
    def attempts(self) -> int:
        return self._attempts

    @property
    def binding_count(self) -> int:
        return len(self._issued)

    def require_current(self) -> None:
        """Reprove retained declarations without issuing or charging values."""
        self._require_declaration()

    def _require_declaration(self) -> None:
        namespace = _module_namespace(self._module)
        if namespace is not self._module_namespace:
            _reject()
        _require_snapshot(namespace, self._module_snapshot)
        class_namespace, bases = _record_declaration(self._record_type, self._descriptor)
        if (
            bases is not self._bases
            or namespace.get(self._attribute, _MISSING) is not self._record_type
        ):
            _reject()
        _require_snapshot(class_namespace, self._class_snapshot)
        path_fields = _row_values(self._path, ProviderCodePath, _PATH_FIELDS)
        _same_references(path_fields, self._path_fields)
        _REQUIRE_PATH(self._path)
        if _read(_FUNCTION_GLOBALS, path_fields[0], FunctionType) is not self._root_globals:
            _reject()
        _same_references(_row_values(self._path, ProviderCodePath, _PATH_FIELDS), self._path_fields)

    def _capture(self, value: Any) -> _Value:
        kind = type(value)
        if kind is int:
            return _Value("primitive", value)
        if kind is float:
            if not _ISFINITE(value):
                _reject()
            return _Value("primitive", value)
        if kind is self._record_type:
            storage, fields = _record_values(value, self._record_type, self._descriptor)
            return _Value("record", value, storage=storage, fields=fields)
        if kind is FunctionType:
            _REQUIRE_CALLBACK(self._path, value)
            code = _read(_FUNCTION_CODE, value, FunctionType)
            namespace = _read(_FUNCTION_GLOBALS, value, FunctionType)
            if namespace is not self._root_globals:
                _reject()
            return _Value("callback", value, code=code, namespace=namespace)
        raise TypeError(_ERROR)

    def _require_value(self, retained: _Value, candidate: Any) -> None:
        if candidate is not retained.value:
            _reject()
        if retained.mode == "primitive":
            kind = type(candidate)
            if kind is float:
                if not _ISFINITE(cast(float, candidate)):
                    _reject()
            elif kind is not int:
                raise TypeError(_ERROR)
        elif retained.mode == "record":
            storage, fields = _record_values(candidate, self._record_type, self._descriptor)
            if storage is not retained.storage or retained.fields is None:
                _reject()
            _same_references(fields, retained.fields)
        elif retained.mode == "callback":
            _exact(candidate, FunctionType)
            _REQUIRE_CALLBACK(self._path, cast(FunctionType, candidate))
            if (
                _read(_FUNCTION_CODE, candidate, FunctionType) is not retained.code
                or _read(_FUNCTION_GLOBALS, candidate, FunctionType) is not retained.namespace
                or retained.namespace is not self._root_globals
            ):
                _reject()
        else:
            _reject()

    def pin(self, value: object) -> ProviderValueBinding:
        # Failed validation and overflow attempts consume allowance permanently.
        self._attempts += 1
        if self._attempts > self._max_values:
            _reject()
        self._require_declaration()
        retained = self._capture(value)
        self._require_value(retained, value)
        self._require_declaration()
        self._require_value(retained, value)
        binding = ProviderValueBinding(retained.mode, value)
        fields = _row_values(binding, ProviderValueBinding, _BINDING_FIELDS)
        self._issued.append(_Issued(binding, fields, retained))
        return binding

    def require(self, binding: ProviderValueBinding, candidate: object) -> None:
        self._require_declaration()
        fields = _row_values(binding, ProviderValueBinding, _BINDING_FIELDS)
        issued = next((item for item in self._issued if binding is item.binding), None)
        if issued is None:
            raise TypeError(_ERROR)
        _same_references(fields, issued.fields)
        self._require_value(issued.retained, candidate)
        self._require_declaration()
        _same_references(_row_values(binding, ProviderValueBinding, _BINDING_FIELDS), issued.fields)
        self._require_value(issued.retained, candidate)
