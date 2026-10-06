"""Pure native code/constant fixtures shared by unit and actual runtime controls.

Inputs are local diagnostic code objects. Outputs copy every public native field
and recursively retain code values. This module imports no pytest/algorithm and
does not grant loaded-source/runtime authority.
"""

from __future__ import annotations

import types
from typing import Any

CONSTANT_KINDS = ("str", "bytes", "float", "tuple", "frozenset", "code")
CODE_FIELDS = tuple(
    name
    for name, descriptor in vars(types.CodeType).items()
    if name.startswith("co_")
    and type(descriptor) in (types.GetSetDescriptorType, types.MemberDescriptorType)
)


def describe_native_value(value: Any) -> Any:
    """Copy complete values without retaining references to original constants."""
    kind = type(value)
    if kind is types.CodeType:
        return {"type": "code", "fields": describe_native_code(value)}
    if kind is tuple:
        return {"type": "tuple", "items": [describe_native_value(item) for item in value]}
    if kind is frozenset:
        rows = [describe_native_value(item) for item in value]
        return {"type": "frozenset", "items": sorted(rows, key=repr)}
    if kind is bytes:
        return {"type": "bytes", "hex": value.hex()}
    if kind is float:
        return {"type": "float", "hex": value.hex()}
    if kind in (type(None), bool, int, str):
        return {"type": kind.__name__, "repr": repr(value)}
    raise AssertionError(f"unhandled native control value: {kind}")


def describe_native_code(code: types.CodeType) -> dict[str, Any]:
    return {name: describe_native_value(getattr(code, name)) for name in CODE_FIELDS}


def make_constant_code(kind: str) -> types.CodeType:
    # A real None assignment keeps both slots on Python 3.11 through 3.14.
    module = compile(
        "def controlled():\n    unused = None\n    return 'placeholder'\n",
        "<marshal-control>",
        "exec",
    )
    template = module.co_consts[0]
    assert type(template) is types.CodeType
    text = " ".join(("fresh", "constant", "for", "marshal", "controls"))
    if kind == "str":
        value: Any = text
    elif kind == "bytes":
        value = bytes(bytearray(text.encode()))
    elif kind == "float":
        value = float("3.141592653589793")
    elif kind == "tuple":
        value = (text, float("3.141592653589793"))
    elif kind == "frozenset":
        value = frozenset((text, float("3.141592653589793")))
    elif kind == "code":
        value = make_constant_code("str")
    else:
        raise AssertionError(f"unknown marshal control kind: {kind}")
    return template.replace(co_consts=(None, value))


def retain_code_values(code: types.CodeType) -> tuple[object, ...]:
    """Control only: retain all public native fields and recursive constants.

    Why this: code replacement can share metadata such as line-table bytes;
    retaining constants alone cannot stabilize every marshal reference flag.
    Private interpreter fields and trusted source correspondence remain unproved.
    """
    pending: list[object] = [code]
    retained: dict[int, object] = {}
    while pending:
        value = pending.pop()
        if id(value) in retained:
            continue
        retained[id(value)] = value
        if type(value) is types.CodeType:
            pending.extend(getattr(value, name) for name in CODE_FIELDS)
        elif type(value) in (tuple, frozenset):
            assert isinstance(value, (tuple, frozenset))
            pending.extend(value)
    return tuple(retained.values())
