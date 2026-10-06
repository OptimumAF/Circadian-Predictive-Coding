"""Pure complete V1 executable-edge, native-key and bound-value structural unions.

Names for opaque references are descriptive and may equal builtin names. This
validates every encoded child/identity without inferring actual type provenance.
Iterative walks avoid imposing a new arbitrary valid-payload nesting limit.
"""

from __future__ import annotations

from _codecs import raw_unicode_escape_encode, unicode_escape_decode
import re
from typing import Any

from src.core.runtime_record_schema import (
    RuntimeSchemaBindings,
    require_runtime,
    runtime_hash,
    runtime_integer,
    runtime_list,
    runtime_string,
)


# Why this: first regex compilation may intern the code constant. Freeze-ready
# validators prepare fixed text patterns at module loading, never while held.
INTEGER_PATTERN = re.compile(r"-?(0|[1-9][0-9]*)")
STRING_ESCAPE_PATTERN = re.compile(
    r"(?:[^\\]|\\(?:[\\'\"abfnrtv]|x[0-9a-f]{2}|u[0-9a-f]{4}|U[0-9a-f]{8}))*"
)


def validate_runtime_key(value: Any, context: str) -> None:
    row = runtime_list(value, context)
    require_runtime(len(row) == 2, context, "key label length differs")
    name = runtime_string(row[0], context + ".type_name")
    item = row[1]
    if type(item) is str:
        require_runtime(name == "str", context, "string key label differs")
    elif type(item) is bool:
        require_runtime(name == "bool", context, "boolean key label differs")
    else:
        # A class named str/int/bool can be an opaque key, identified by pointer.
        require_runtime(type(item) is int, context, "key value differs")
        if name != "int":
            runtime_integer(item, context + ".value", 1)


def validate_runtime_members(
    value: Any, bindings: RuntimeSchemaBindings, context: str, ordered: bool = True
) -> None:
    rows = runtime_list(value, context)
    labels = []
    for row in rows:
        runtime_list(row, context + ".member")
        require_runtime(len(row) == 2, context, "member length differs")
        validate_runtime_key(row[0], context + ".key")
        labels.append(str(row[0]))
        validate_runtime_edge(row[1], bindings, context + ".edge")
    require_runtime(len(set(labels)) == len(labels), context, "duplicate member key")
    if ordered:
        require_runtime(labels == sorted(labels), context, "member order differs")


def validate_runtime_edge(value: Any, bindings: RuntimeSchemaBindings, context: str) -> None:
    work = [value]
    while work:
        row = runtime_list(work.pop(), context)
        require_runtime(len(row) in (2, 3), context, "executable reference length differs")
        name = runtime_string(row[0], context + ".type_name")
        identity = runtime_integer(row[1], context + ".object_id", 1)
        bindings.validate_known_reference(name, identity, len(row), context)
        if len(row) == 2:
            continue  # Native callable/class/metaclass reference; name is opaque.
        item = row[2]
        if type(item) is int:
            runtime_integer(item, context + ".type_id", 1)
        elif type(item) is dict:
            require_runtime(name == "code", context, "code reference tag differs")
            bindings.validate_code(item, context + ".code")
        else:
            children = runtime_list(item, context + ".children")
            if name in {"property", "staticmethod", "classmethod"}:
                require_runtime(
                    len(children) == (3 if name == "property" else 1),
                    context,
                    "descriptor accessor count differs",
                )
                work.extend(child for child in children if child is not None)
            else:
                require_runtime(
                    name in {"tuple", "list", "dict", "set", "frozenset"} and bool(children),
                    context,
                    "container reference differs",
                )
                labels = []
                positions = []
                for child in children:
                    runtime_list(child, context + ".container_member")
                    require_runtime(len(child) == 2, context, "container member length differs")
                    validate_runtime_key(child[0], context + ".container_key")
                    labels.append(str(child[0]))
                    if name != "dict":
                        require_runtime(
                            child[0][0] == "int", context, "container index type differs"
                        )
                        positions.append(runtime_integer(child[0][1], context + ".index"))
                    work.append(child[1])
                require_runtime(len(set(labels)) == len(labels), context, "duplicate container key")
                require_runtime(
                    positions == sorted(set(positions)), context, "container index order differs"
                )


def _validate_string(text: str, context: str) -> None:
    require_runtime(
        len(text) >= 2 and text[0] in {"'", '"'} and text[-1] == text[0],
        context,
        "string repr differs",
    )
    inner = text[1:-1]
    require_runtime(
        STRING_ESCAPE_PATTERN.fullmatch(inner) is not None,
        context,
        "string escape differs",
    )
    # Why this: literal_eval compiles input and constructs transient functions,
    # violating the active executable freeze. Native escape codecs are passive;
    # raw Unicode encoding preserves literal non-ASCII and isolated surrogates.
    try:
        encoded = raw_unicode_escape_encode(inner)[0]
        parsed, consumed = unicode_escape_decode(encoded)
    except UnicodeError as error:
        raise ValueError(f"runtime payload {context} string repr differs") from error
    require_runtime(
        consumed == len(encoded) and repr(parsed) == text, context, "string repr differs"
    )


def _validate_primitive(name: str, value: Any, context: str) -> None:
    text = runtime_string(value, context)
    if name == "int":
        require_runtime(
            INTEGER_PATTERN.fullmatch(text) is not None and text != "-0",
            context,
            "integer repr differs",
        )
    elif name == "float":
        try:
            valid = repr(float(text)) == text
        except (ValueError, OverflowError):
            valid = False
        require_runtime(valid, context, "float repr differs")
    elif name == "bool":
        require_runtime(text in {"True", "False"}, context, "boolean repr differs")
    elif name == "NoneType":
        require_runtime(text == "None", context, "None repr differs")
    else:
        require_runtime(name == "str", context, "primitive tag differs")
        _validate_string(text, context)


def validate_runtime_bound(value: Any, bindings: RuntimeSchemaBindings, context: str) -> None:
    work = [value]
    while work:
        row = runtime_list(work.pop(), context)
        require_runtime(len(row) in (2, 3), context, "bound value length differs")
        name = runtime_string(row[0], context + ".type_name")
        if len(row) == 2:
            _validate_primitive(name, row[1], context)
            continue
        item = row[2]
        if name == "bytes" and type(item) is str:
            runtime_integer(row[1], context + ".byte_count")
            runtime_hash(item, context + ".sha256")
            continue
        runtime_integer(row[1], context + ".object_id", 1)
        if (
            type(item) is list
            and (not item or type(item[0]) is list)
            and name in {"tuple", "list", "dict"}
        ):
            if name == "dict":
                require_runtime(
                    item == sorted(item, key=str), context, "bound mapping order differs"
                )
                for entry in item:
                    runtime_list(entry, context + ".mapping_member")
                    require_runtime(len(entry) == 2, context, "bound mapping member differs")
                    work.extend(entry)
            else:
                work.extend(item)
        elif item is not None:
            validate_runtime_edge(item, bindings, context + ".opaque_reference")
