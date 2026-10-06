"""Pure full Python membership, function state, namespaces and endpoint joins.

Inputs are every provided V1 Python record and ordered entrypoints. Cross-check
all encoded aliases/identities; detached globals and opaque paths do not become
source correspondence or proof that a caller omitted no unreferenced object.
"""

from __future__ import annotations

import re
from typing import Any

from src.core.runtime_record_schema import (
    RuntimeSchemaBindings,
    require_runtime,
    runtime_integer,
    runtime_list,
    runtime_object,
    runtime_string,
)
from src.core.runtime_value_schema import validate_runtime_bound, validate_runtime_members

FUNCTION_FIELDS = {
    "object_id",
    "module",
    "qualname",
    "code",
    "globals_id",
    "defaults",
    "keyword_defaults",
    "closure",
    "attributes",
}
MODULE_FIELDS = {"name", "object_id", "namespace_id", "file", "origin", "loader_id", "members"}


def _validate_defaults(
    value: Any, kind: str, bindings: RuntimeSchemaBindings, context: str
) -> None:
    validate_runtime_bound(value, bindings, context)
    if len(value) == 2:
        require_runtime(
            value == ["NoneType", "None"], context, "native defaults require a container or None"
        )
    else:
        item = value[2]
        require_runtime(
            not (value[0] == "bytes" and type(item) is str),
            context,
            "native defaults cannot be bytes",
        )
        if type(item) is list and (not item or type(item[0]) is list):
            require_runtime(value[0] == kind, context, "native defaults container differs")
    # V1 opaque tuple/dict subclasses have only descriptive names/IDs; preserve
    # those structural references without asserting their original native type.


def _validate_functions(value: Any, bindings: RuntimeSchemaBindings) -> None:
    rows = runtime_list(value, "python.functions", True)
    for index, value in enumerate(rows):
        row = runtime_object(value, FUNCTION_FIELDS, f"python.functions[{index}]")
        identity = runtime_integer(row["object_id"], "function.object_id", 1)
        require_runtime(
            identity not in bindings.functions, "python.functions", "duplicate object ID"
        )
        bindings.functions[identity] = row
    require_runtime(
        list(bindings.functions) == sorted(bindings.functions),
        "python.functions",
        "function order differs",
    )
    for index, row in enumerate(rows):
        context = f"python.functions[{index}]"
        require_runtime(
            row["module"] is None or type(row["module"]) is str, context, "module name differs"
        )
        runtime_string(row["qualname"], context + ".qualname")
        bindings.validate_code(row["code"], context + ".code")
        runtime_integer(row["globals_id"], context + ".globals_id", 1)
        _validate_defaults(row["defaults"], "tuple", bindings, context + ".defaults")
        _validate_defaults(row["keyword_defaults"], "dict", bindings, context + ".keyword_defaults")
        for cell in runtime_list(row["closure"], context + ".closure"):
            runtime_list(cell, context + ".cell")
            require_runtime(len(cell) == 2, context, "closure cell length differs")
            runtime_integer(cell[0], context + ".cell_id", 1)
            if cell[1] != ["empty_cell"]:
                validate_runtime_bound(cell[1], bindings, context + ".cell_value")
        validate_runtime_members(row["attributes"], bindings, context + ".attributes")


def _validate_modules(value: Any, bindings: RuntimeSchemaBindings) -> dict[str, dict[str, Any]]:
    rows = runtime_list(value, "python.modules", True)
    modules: dict[str, dict[str, Any]] = {}
    objects: dict[int, dict[str, Any]] = {}
    for index, value in enumerate(rows):
        context = f"python.modules[{index}]"
        row = runtime_object(value, MODULE_FIELDS, context)
        name = runtime_string(row["name"], context + ".name")
        require_runtime(name not in modules, context, "duplicate module name")
        modules[name] = row
        identity = runtime_integer(row["object_id"], context + ".object_id")
        runtime_integer(row["namespace_id"], context + ".namespace_id")
        runtime_integer(row["loader_id"], context + ".loader_id")
        if identity == 0:
            require_runtime(
                row["namespace_id"] == row["loader_id"] == 0
                and row["file"] is None
                and row["origin"] == "unloaded_sentinel"
                and row["members"] == [],
                context,
                "unloaded sentinel differs",
            )
            continue
        require_runtime(
            identity not in bindings.functions, context, "module/function identities conflict"
        )
        require_runtime(
            row["namespace_id"] > 0 and row["loader_id"] > 0, context, "live module IDs differ"
        )
        require_runtime(
            row["origin"] is None or type(row["origin"]) is str, context, "origin differs"
        )
        if row["file"] is not None:
            bindings.validate_file(row["file"], context + ".file")
        validate_runtime_members(row["members"], bindings, context + ".members")
        previous = objects.setdefault(identity, row)
        require_runtime(
            all(previous[key] == row[key] for key in MODULE_FIELDS - {"name"}),
            context,
            "module alias bindings conflict",
        )
    require_runtime(list(modules) == sorted(modules), "python.modules", "module order differs")
    return modules


def _validate_namespaces(
    value: Any, bindings: RuntimeSchemaBindings, modules: dict[str, dict[str, Any]]
) -> None:
    rows = runtime_list(value, "python.namespaces")
    owners, ids, order = set(), set(), []
    module_objects = {row["object_id"]: row for row in modules.values() if row["object_id"]}
    for index, value in enumerate(rows):
        context = f"python.namespaces[{index}]"
        row = runtime_object(value, {"owner_id", "type_id", "namespace_id", "members"}, context)
        owner = runtime_integer(row["owner_id"], context + ".owner_id", 1)
        namespace = runtime_integer(row["namespace_id"], context + ".namespace_id", 1)
        runtime_integer(row["type_id"], context + ".type_id", 1)
        require_runtime(
            owner not in owners and namespace not in ids, context, "duplicate namespace identity"
        )
        require_runtime(
            row["type_id"] not in bindings.functions, context, "type/function identities conflict"
        )
        owners.add(owner)
        ids.add(namespace)
        order.append((owner, namespace))
        runtime_list(row["members"], context + ".members", True)
        validate_runtime_members(row["members"], bindings, context + ".members")
        if owner in bindings.functions:
            require_runtime(
                namespace == bindings.functions[owner]["globals_id"],
                context,
                "function globals join differs",
            )
        if owner in module_objects:
            require_runtime(
                namespace == module_objects[owner]["namespace_id"]
                and row["members"] == module_objects[owner]["members"],
                context,
                "module namespace join differs",
            )
    require_runtime(order == sorted(order), "python.namespaces", "namespace order differs")


def _validate_import_state(value: Any) -> None:
    row = runtime_object(
        value,
        {
            "path",
            "meta_path",
            "path_hooks",
            "modules_id",
            "executable",
            "version",
            "implementation",
            "trace_id",
            "profile_id",
        },
        "python.import_state",
    )
    for value in runtime_list(row["path"], "python.import_state.path"):
        runtime_string(value, "python.import_state.path")
    for name in ("meta_path", "path_hooks"):
        for value in runtime_list(row[name], "python.import_state." + name):
            runtime_integer(value, "python.import_state." + name, 1)
    for name in ("modules_id", "trace_id", "profile_id"):
        runtime_integer(row[name], "python.import_state." + name, 1)
    for name in ("executable", "version", "implementation"):
        runtime_string(row[name], "python.import_state." + name, True)


def validate_runtime_python(
    value: Any, bindings: RuntimeSchemaBindings
) -> dict[str, dict[str, Any]]:
    row = runtime_object(value, {"modules", "functions", "namespaces", "import_state"}, "python")
    _validate_functions(row["functions"], bindings)
    modules = _validate_modules(row["modules"], bindings)
    _validate_namespaces(row["namespaces"], bindings, modules)
    _validate_import_state(row["import_state"])
    return modules


def validate_runtime_endpoints(
    value: Any,
    names: tuple[str, ...],
    bindings: RuntimeSchemaBindings,
    modules: dict[str, dict[str, Any]],
) -> None:
    pattern = r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
    require_runtime(
        type(names) is tuple
        and bool(names)
        and all(
            type(name) is str and re.fullmatch(pattern, name, flags=re.ASCII) for name in names
        ),
        "entrypoints",
        "ordered names differ",
    )
    require_runtime(len(set(names)) == len(names), "entrypoints", "duplicate names")
    rows = runtime_list(value, "entrypoints")
    require_runtime(len(rows) == len(names), "entrypoints", "membership differs")
    for row, name in zip(rows, names, strict=True):
        runtime_list(row, "entrypoint")
        require_runtime(
            len(row) == 4 and type(row[0]) is str and row[0] == name,
            "entrypoint",
            "fields/name differ",
        )
        identity = runtime_integer(row[1], "entrypoint.object_id", 1)
        code = bindings.validate_code(row[2], "entrypoint.code")
        globals_id = runtime_integer(row[3], "entrypoint.globals_id", 1)
        require_runtime(identity in bindings.functions, "entrypoint", "actual function is absent")
        function = bindings.functions[identity]
        require_runtime(
            code == function["code"] and globals_id == function["globals_id"],
            "entrypoint",
            "function code/globals join differs",
        )
        module, attributes = name.split(":", 1)
        require_runtime(
            module in modules and modules[module]["object_id"] > 0,
            "entrypoint",
            "loaded module is absent",
        )
        if "." not in attributes:
            found = [edge for key, edge in modules[module]["members"] if key == ["str", attributes]]
            require_runtime(
                found == [["function", identity]],
                "entrypoint",
                "direct module function binding differs",
            )
        # V1 omits non-callable intermediate module references. A nested path's
        # final function is joined above; its full route needs future attestation.
