"""Observe whole imported membership and actual Python executable object graphs.

Read native namespaces without running user descriptors. Include every GC-tracked
function, even detached ones, its nested code/default/closure/globals binding, and
code-bearing module/class/instance namespaces. Opaque and generated provenance is
recorded, never treated as original source-version attestation.
"""

from __future__ import annotations

import gc
from hashlib import sha256
import marshal
from pathlib import Path
import sys
import types
from typing import Any

from src.infra.runtime_native_images import whole_runtime_file
from src.infra.runtime_python_retention import retain_python_runtime_values

_TYPE_DICT = type.__dict__["__dict__"]
_MODULE_DICT = _TYPE_DICT.__get__(types.ModuleType, type)["__dict__"]


def _namespace(value: object) -> dict[str, Any] | types.MappingProxyType | None:
    kind = type(value)
    # Why this: exact builtin containers have no native instance dictionary.
    # Identity checks preserve subclasses and never invoke metaclass equality.
    if kind is dict or kind is list or kind is tuple or kind is set or kind is frozenset:
        return None
    if issubclass(kind, type):
        return _TYPE_DICT.__get__(value, kind)
    if kind is types.ModuleType:
        return _MODULE_DICT.__get__(value, kind)
    # Why this: object.__getattribute__ would still invoke a custom __dict__
    # property. Only a native instance-dict slot may be read here.
    for parent in kind.__mro__:
        descriptor = _TYPE_DICT.__get__(parent, type(parent)).get("__dict__")
        if descriptor is not None:
            if type(descriptor) is types.GetSetDescriptorType:
                result = descriptor.__get__(value, kind)
                return result if type(result) is dict else None
            return None
    return None


def _code_identity(code: types.CodeType) -> dict[str, Any]:
    raw = marshal.dumps(code)
    return {
        "byte_count": len(raw),
        "sha256": sha256(raw).hexdigest(),
        "filename": code.co_filename,
        "qualname": code.co_qualname,
        "first_line": code.co_firstlineno,
    }


def _executable_edges(value: Any, seen: set[int] | None = None) -> Any:
    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return None
    seen.add(identity)
    kind = type(value)
    if kind is types.CodeType:
        return [kind.__name__, identity, _code_identity(value)]
    if issubclass(kind, type) or kind in (
        types.FunctionType,
        types.BuiltinFunctionType,
        types.MethodType,
        types.CodeType,
        types.WrapperDescriptorType,
        types.MethodDescriptorType,
        types.ClassMethodDescriptorType,
        types.MethodWrapperType,
    ):
        return [kind.__name__, identity]
    if kind in (property, staticmethod, classmethod):
        values = (value.fget, value.fset, value.fdel) if kind is property else (value.__func__,)
        return [kind.__name__, identity, [_executable_edges(item, seen) for item in values]]
    if kind in (tuple, list, dict, set, frozenset):
        entries = value.items() if kind is dict else enumerate(value)
        found = []
        for key, item in entries:
            edge = _executable_edges(item, seen)
            if edge is not None:
                label = (
                    [type(key).__name__, key]
                    if type(key) in (str, int, bool)
                    else [type(key).__name__, id(key)]
                )
                found.append([label, edge])
        return [kind.__name__, identity, found] if found else None
    # Callable instances are bound by identity/type. Their native instance-dict
    # slots and class methods are also inspected in the complete GC pass.
    if "__call__" in _TYPE_DICT.__get__(kind, type(kind)):
        return [kind.__name__, identity, id(kind)]
    return None


def _bound_value(value: Any) -> Any:
    if type(value) in (str, int, bool, float, bytes, type(None)):
        if type(value) is bytes:
            return ["bytes", len(value), sha256(value).hexdigest()]
        return [type(value).__name__, repr(value)]
    if type(value) in (tuple, list):
        return [type(value).__name__, id(value), [_bound_value(item) for item in value]]
    if type(value) is dict:
        return [
            "dict",
            id(value),
            sorted(
                [[_bound_value(key), _bound_value(item)] for key, item in value.items()], key=str
            ),
        ]
    return [type(value).__name__, id(value), _executable_edges(value)]


def _function_row(function: types.FunctionType) -> dict[str, Any]:
    closures = []
    for cell in function.__closure__ or ():
        try:
            contents = _bound_value(cell.cell_contents)
        except ValueError:
            contents = ["empty_cell"]
        closures.append([id(cell), contents])
    return {
        "object_id": id(function),
        "module": function.__module__,
        "qualname": function.__qualname__,
        "code": _code_identity(function.__code__),
        "globals_id": id(function.__globals__),
        "defaults": _bound_value(function.__defaults__),
        "keyword_defaults": _bound_value(function.__kwdefaults__),
        "closure": closures,
        "attributes": _namespace_edges(function.__dict__),
    }


def _namespace_edges(namespace: Any) -> list[Any]:
    rows = []
    for key, value in tuple(namespace.items()):
        edge = _executable_edges(value)
        if edge is not None:
            label = (
                [type(key).__name__, key]
                if type(key) in (str, int, bool)
                else [type(key).__name__, id(key)]
            )
            rows.append([label, edge])
    return sorted(rows, key=lambda row: str(row[0]))


def retain_python_executable_objects() -> tuple[object, ...]:
    # Keep identities and native code values alive until the lease's final check.
    # Raw marshal bytes stay authoritative; no observation is normalized/cached.
    return retain_python_runtime_values(_namespace)


def capture_python_runtime() -> dict[str, Any]:
    membership = tuple(sorted(sys.modules.copy().items()))
    files: dict[str, dict[str, Any]] = {}
    modules = []
    for name, module in membership:
        if module is None:
            modules.append(
                {
                    "name": name,
                    "object_id": 0,
                    "namespace_id": 0,
                    "file": None,
                    "origin": "unloaded_sentinel",
                    "loader_id": 0,
                    "members": [],
                }
            )
            continue
        if type(module) is not types.ModuleType:
            raise ValueError(f"runtime sys.modules contains opaque module {name}")
        namespace = _MODULE_DICT.__get__(module, types.ModuleType)
        filename = namespace.get("__file__")
        file = None
        if filename is not None:
            if type(filename) is not str:
                raise ValueError("runtime module file path is foreign")
            if filename not in files:
                files[filename] = whole_runtime_file(Path(filename))
            file = files[filename]
        spec = namespace.get("__spec__")
        spec_values = _namespace(spec) if spec is not None else None
        origin = spec_values.get("origin") if spec_values else None
        if origin is not None and type(origin) is not str:
            raise ValueError("runtime module origin is opaque")
        modules.append(
            {
                "name": name,
                "object_id": id(module),
                "namespace_id": id(namespace),
                "file": file,
                "origin": origin,
                "loader_id": id(namespace.get("__loader__")),
                "members": _namespace_edges(namespace),
            }
        )
    objects = gc.get_objects()
    functions, namespaces = [], []
    namespace_ids = set()
    for value in sorted(objects, key=id):
        if type(value) is types.FunctionType:
            functions.append(_function_row(value))
            namespace = value.__globals__
        else:
            namespace = _namespace(value)
        namespace_id = id(value) if issubclass(type(value), type) else id(namespace)
        if namespace is not None and namespace_id not in namespace_ids:
            namespace_ids.add(namespace_id)
            edges = _namespace_edges(namespace)
            if edges:
                namespaces.append(
                    {
                        "owner_id": id(value),
                        "type_id": id(type(value)),
                        "namespace_id": namespace_id,
                        "members": edges,
                    }
                )
    if tuple(sorted(sys.modules.copy().items())) != membership:
        raise ValueError("runtime module membership changed during observation")
    return {
        "modules": modules,
        "functions": sorted(functions, key=lambda row: row["object_id"]),
        "namespaces": sorted(namespaces, key=lambda row: (row["owner_id"], row["namespace_id"])),
        "import_state": {
            "path": tuple(sys.path),
            "meta_path": tuple(id(value) for value in sys.meta_path),
            "path_hooks": tuple(id(value) for value in sys.path_hooks),
            "modules_id": id(sys.modules),
            "executable": sys.executable,
            "version": sys.version,
            "implementation": sys.implementation.name,
            "trace_id": id(sys.gettrace()),
            "profile_id": id(sys.getprofile()),
        },
    }


def bind_runtime_entrypoints(names: tuple[str, ...]) -> list[Any]:
    rows = []
    for name in names:
        module_name, attributes = name.split(":", 1)
        module = sys.modules.get(module_name)
        if type(module) is not types.ModuleType:
            raise ValueError("runtime entrypoint module is not actually loaded")
        value: Any = module
        for attribute in attributes.split("."):
            namespace = _namespace(value)
            if namespace is None or attribute not in namespace:
                raise ValueError("runtime entrypoint is detached from its actual namespace")
            value = namespace[attribute]
        if type(value) is not types.FunctionType:
            raise ValueError("runtime entrypoint must be an actual Python function")
        rows.append([name, id(value), _code_identity(value.__code__), id(value.__globals__)])
    return rows
