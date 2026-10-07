"""Pure exact runtime record primitives and within-body identity consistency.

Inputs are decoded V1 JSON. Outputs are checked values or contextual ValueError.
No IO, object introspection, executable loading or source attestation occurs here.
The registry is local to one validation, never a process/file observation cache.
"""

from __future__ import annotations

from pathlib import PureWindowsPath
import re
from typing import Any


# Fixed regex compilation can intern its input text. Prepare it before any live
# freeze so complete marshalled code bytes remain stable during validation.
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
# A live exception retains detail references. Keep this text outside code
# constants so their whole marshal reference flags remain stable during errors.
FIELDS_DIFFER = "fields differ"


def require_runtime(condition: bool, context: str, detail: str) -> None:
    if not condition:
        raise ValueError(f"runtime payload {context} {detail}")


def runtime_object(value: Any, fields: set[str], context: str) -> dict[str, Any]:
    require_runtime(type(value) is dict and set(value) == fields, context, FIELDS_DIFFER)
    return value


def runtime_list(value: Any, context: str, nonempty: bool = False) -> list[Any]:
    require_runtime(type(value) is list and (not nonempty or bool(value)), context, "array differs")
    return value


def runtime_integer(value: Any, context: str, minimum: int = 0) -> int:
    require_runtime(type(value) is int and value >= minimum, context, "integer differs")
    return value


def runtime_string(value: Any, context: str, nonempty: bool = False) -> str:
    require_runtime(type(value) is str and (not nonempty or bool(value)), context, "string differs")
    return value


def runtime_hash(value: Any, context: str) -> str:
    require_runtime(
        type(value) is str and HASH_PATTERN.fullmatch(value) is not None,
        context,
        "hash differs",
    )
    return value


def runtime_identity(value: Any, context: str) -> dict[str, Any]:
    row = runtime_object(value, {"byte_count", "sha256"}, context)
    runtime_integer(row["byte_count"], context + ".byte_count")
    runtime_hash(row["sha256"], context + ".sha256")
    return row


class RuntimeSchemaBindings:
    """Consistent declared identities within a full body; no provenance authority."""

    def __init__(self) -> None:
        self.functions: dict[int, dict[str, Any]] = {}
        self.files: dict[str, tuple[Any, ...]] = {}
        self.physical_files: dict[tuple[int, int], tuple[Any, ...]] = {}
        self.codes: dict[str, dict[str, Any]] = {}

    def validate_code(self, value: Any, context: str) -> dict[str, Any]:
        row = runtime_object(
            value, {"byte_count", "sha256", "filename", "qualname", "first_line"}, context
        )
        runtime_integer(row["byte_count"], context + ".byte_count", 1)
        runtime_hash(row["sha256"], context + ".sha256")
        runtime_integer(row["first_line"], context + ".first_line")
        runtime_string(row["filename"], context + ".filename")
        runtime_string(row["qualname"], context + ".qualname")
        previous = self.codes.setdefault(row["sha256"], row)
        require_runtime(previous == row, context, "same whole code has conflicting metadata")
        return row

    def validate_file(self, value: Any, context: str) -> dict[str, Any]:
        row = runtime_object(
            value, {"path", "device", "inode", "link_count", "byte_count", "sha256"}, context
        )
        path = runtime_string(row["path"], context + ".path", True)
        parsed = PureWindowsPath(path)
        require_runtime(
            parsed.is_absolute() and parsed.as_posix() == path and ".." not in parsed.parts,
            context,
            "file path is not canonical absolute Windows metadata",
        )
        runtime_integer(row["device"], context + ".device")
        runtime_integer(row["inode"], context + ".inode")
        runtime_integer(row["link_count"], context + ".link_count", 1)
        runtime_integer(row["byte_count"], context + ".byte_count")
        runtime_hash(row["sha256"], context + ".sha256")
        fields = tuple(
            row[name] for name in ("device", "inode", "link_count", "byte_count", "sha256")
        )
        previous = self.files.setdefault(path.casefold(), fields)
        require_runtime(previous == fields, context, "file alias identities conflict")
        if row["inode"]:
            physical = self.physical_files.setdefault((row["device"], row["inode"]), fields)
            require_runtime(physical == fields, context, "physical file identities conflict")
        return row

    def validate_known_reference(self, name: str, identity: int, length: int, context: str) -> None:
        if identity in self.functions:
            require_runtime(
                name == "function" and length == 2,
                context,
                "known Python function reference conflicts",
            )
