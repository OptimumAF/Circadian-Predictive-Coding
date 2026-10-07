"""Pure full Windows image/file/executable-memory record structure and joins.

All provided images/regions are validated. Private/mapped memory remains visible;
no disk-to-loaded-memory correspondence or source-version authority is inferred.
"""

from __future__ import annotations

from typing import Any

from src.core.runtime_record_schema import (
    RuntimeSchemaBindings,
    require_runtime,
    runtime_identity,
    runtime_integer,
    runtime_list,
    runtime_object,
)


def validate_runtime_native(value: Any, bindings: RuntimeSchemaBindings) -> None:
    row = runtime_object(value, {"platform", "images", "executable_regions"}, "native")
    require_runtime(
        type(row["platform"]) is str and row["platform"] == "windows", "native", "platform differs"
    )
    previous = 0
    for index, value in enumerate(runtime_list(row["images"], "native.images", True)):
        context = f"native.images[{index}]"
        image = runtime_object(value, {"base", "file"}, context)
        base = runtime_integer(image["base"], context + ".base", 1)
        require_runtime(previous < base < 2**64, context, "image order/base differs")
        previous = base
        bindings.validate_file(image["file"], context + ".file")
    previous_end = 0
    for index, value in enumerate(
        runtime_list(row["executable_regions"], "native.executable_regions", True)
    ):
        context = f"native.executable_regions[{index}]"
        region = runtime_object(
            value, {"base", "allocation_base", "size", "protect", "kind", "identity"}, context
        )
        base = runtime_integer(region["base"], context + ".base", 1)
        allocation = runtime_integer(region["allocation_base"], context + ".allocation_base", 1)
        size = runtime_integer(region["size"], context + ".size", 1)
        require_runtime(
            allocation <= base and previous_end <= base and base + size <= 2**64,
            context,
            "memory order/allocation/overlap differs",
        )
        previous_end = base + size
        protect = runtime_integer(region["protect"], context + ".protect")
        require_runtime(
            protect & 0xFF in {0x10, 0x20, 0x40, 0x80} and not protect & 0x100,
            context,
            "memory protection is unsupported or guarded",
        )
        require_runtime(
            type(region["kind"]) is int and region["kind"] in {0x20000, 0x40000, 0x1000000},
            context,
            "memory kind differs",
        )
        identity = runtime_identity(region["identity"], context + ".identity")
        require_runtime(identity["byte_count"] == size, context, "memory whole byte count differs")
