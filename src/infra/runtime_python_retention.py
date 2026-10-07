"""Retain actual code values before a whole Python runtime baseline is observed.

Inputs are the complete GC roots, loaded modules, and a native namespace reader.
Output is a strong-reference tuple held for the lease lifetime. Keep original
function/class identities and every reached code object's public native fields
and recursive constants alive. References stabilize marshal's sharing flags;
the observer still hashes full raw bytes and detects actual content changes.

This module does not hash, normalize, filter observations, cache source authority,
attest private interpreter fields/builds, or close unobserved/transient coverage.
"""

from __future__ import annotations

from collections.abc import Callable
import gc
import sys
import types
from typing import Any
import warnings

NamespaceReader = Callable[[object], dict[str, Any] | types.MappingProxyType | None]
CODE_FIELDS = tuple(
    name
    for name, descriptor in vars(types.CodeType).items()
    if name.startswith("co_")
    and type(descriptor) in (types.GetSetDescriptorType, types.MemberDescriptorType)
)


def _children(value: Any) -> list[Any]:
    kind = type(value)
    # Identity dispatch never invokes a foreign metaclass's equality or hashing.
    if kind is dict:
        return [item for pair in value.items() for item in pair]
    if kind is tuple or kind is list or kind is set or kind is frozenset:
        return list(value)
    if kind is types.FunctionType:
        result = [
            value.__code__,
            value.__globals__,
            value.__defaults__,
            value.__kwdefaults__,
            value.__dict__,
        ]
        for cell in value.__closure__ or ():
            try:
                result.append(cell.cell_contents)
            except ValueError:
                # Empty native cells contain no code value to retain.
                pass
        return result
    if kind is property:
        return [value.fget, value.fset, value.fdel]
    if kind is staticmethod or kind is classmethod or kind is types.MethodType:
        return [value.__func__]
    return []


def _collect_codes(objects: list[object], read_namespace: NamespaceReader) -> list[types.CodeType]:
    pending = list(objects)
    # Some code-only dictionaries/containers are untracked. Their loaded module
    # or native class/instance namespaces remain authoritative traversal roots.
    for value in objects:
        namespace = read_namespace(value)
        if namespace is not None:
            pending.extend(namespace.values())
    codes: dict[int, types.CodeType] = {}
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if type(value) is types.CodeType:
            codes[id(value)] = value
        else:
            pending.extend(_children(value))
    return list(codes.values())


def _retain_code_values(codes: list[types.CodeType], retained: dict[int, object]) -> None:
    pending: list[Any] = list(codes)
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        retained[id(value)] = value
        if type(value) is types.CodeType:
            # Why this: constants alone miss shared metadata such as line tables.
            # Native public fields also include recursively nested code constants.
            # This projection intentionally includes deprecated native fields.
            # Silence only their new getter warnings before freeze activation;
            # catch_warnings restores the caller's filters on exit.
            with warnings.catch_warnings(action="ignore", category=DeprecationWarning):
                pending.extend(getattr(value, name) for name in CODE_FIELDS)
        else:
            pending.extend(_children(value))


def retain_python_runtime_values(read_namespace: NamespaceReader) -> tuple[object, ...]:
    objects = gc.get_objects() + list(sys.modules.copy().values())
    retained = {
        id(value): value
        for value in objects
        if type(value) is types.FunctionType or issubclass(type(value), type)
    }
    _retain_code_values(_collect_codes(objects, read_namespace), retained)
    return tuple(retained.values())
