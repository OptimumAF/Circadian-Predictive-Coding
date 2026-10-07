"""Preserve complete seed declarations with conservative schema context.

Inputs are decoded metadata. Outputs retain every b1 declaration/issue plus
structured seed-map base/stream declarations. Known quantities and nonrandom
policy nulls are distinct from unresolved history. No IO, source construction,
RNG, execution inference or untouched-role admission belongs to this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from src.core.seed_usage import (
    SeedDeclaration,
    UnresolvedSeedDeclaration,
    collect_seed_usage,
    strict_seed_metadata_json,
)


_QUANTITIES = {
    "bounded_mean_required_seeds",
    "distinct_source_seeds",
    "persistent_labeled_array_bytes_per_seed",
    "projected_feature_seconds_per_seed",
}


@dataclass(frozen=True)
class SeedMeaning:
    declaration: SeedDeclaration | UnresolvedSeedDeclaration
    meaning: str
    related_seed: int | None
    reason: str


@dataclass(frozen=True)
class SeedStreamDeclaration:
    pointer: str
    base_seed: int
    stream: str
    value: int


@dataclass(frozen=True)
class MetadataSeedEvidence:
    declarations: tuple[SeedMeaning, ...]
    unresolved: tuple[SeedMeaning, ...]
    mapped_streams: tuple[SeedStreamDeclaration, ...]
    map_issues: tuple[UnresolvedSeedDeclaration, ...]


def _canonical(value: Any) -> bool:
    return (
        type(value) is dict
        and set(value) == {"dict"}
        and type(value["dict"]) is list
        and all(
            type(row) is list and len(row) == 2 and type(row[0]) is str for row in value["dict"]
        )
    )


def _mapping(value: Any) -> tuple[tuple[str, Any], ...] | None:
    if type(value) is not dict:
        return None
    if _canonical(value):
        rows = value["dict"]
        if all(type(row) is list and len(row) == 2 and type(row[0]) is str for row in rows):
            return tuple((row[0], row[1]) for row in rows)
    return tuple(value.items())


def _unique_mapping(value: Any) -> dict[str, Any] | None:
    rows = _mapping(value)
    if rows is None or len({name for name, _ in rows}) != len(rows):
        return None
    return dict(rows)


class _LocationResolver:
    def __init__(self, metadata: Any) -> None:
        self.metadata = metadata
        self.embedded: dict[tuple[str, ...], Any] = {}

    def resolve(self, pointer: str) -> tuple[Any, Any]:
        parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer.split("/")[1:]]
        value, parent, position = self.metadata, None, 0
        while position < len(parts):
            part = parts[position]
            if _canonical(value) and part == "dict" and position + 2 < len(parts):
                row_index, member = parts[position + 1 : position + 3]
                rows = value["dict"]
                if type(rows) is list and member == "1":
                    parent, value = value, rows[int(row_index)][1]
                    position += 3
                    continue
            if type(value) is str and part == "embedded_json":
                key = tuple(parts[:position])
                if key not in self.embedded:
                    self.embedded[key] = strict_seed_metadata_json(value)
                value = self.embedded[key]
            elif type(value) is str:
                value = value.split(",")[int(part)].strip()
            elif type(value) is dict:
                parent, value = value, value[part]
            elif type(value) is list:
                value = value[int(part)]
            else:
                raise ValueError(
                    "seed declaration location cannot be resolved in its complete metadata"
                )
            position += 1
        return value, parent


def _meaning(
    row: SeedDeclaration | UnresolvedSeedDeclaration, value: Any, parent: Any
) -> SeedMeaning:
    field = row.field.lower()
    siblings = _unique_mapping(parent)
    if field in _QUANTITIES:
        return SeedMeaning(
            row, "quantity", None, "named count/byte/duration field; not an RNG identity"
        )
    if isinstance(row, SeedDeclaration):
        if siblings is not None and {"field", "pointer", "representation", "seed"}.issubset(
            siblings
        ):
            origin = siblings["field"]
            if type(origin) is str and origin.lower() in _QUANTITIES:
                return SeedMeaning(
                    row, "quantity", None, "preserved inventory value of a named quantity"
                )
            return SeedMeaning(
                row,
                "recorded_seed_inventory_value",
                row.seed,
                "inventory lineage; not a new execution",
            )
        return SeedMeaning(
            row,
            "numeric_seed_candidate",
            row.seed,
            "numeric declaration alone does not prove execution or source role",
        )
    if field == "python_hash_seed" and value is None:
        return SeedMeaning(
            row,
            "unrecorded_process_hash_seed",
            None,
            "process hash setting not recorded; do not infer non-use",
        )
    if (
        field == "seed"
        and value is None
        and siblings is not None
        and set(siblings) == {"name", "seed"}
        and siblings["name"] in ("content_hash", "recent_fifo")
    ):
        return SeedMeaning(
            row,
            "nonrandom_retention_policy",
            None,
            "declared retention policy requires null and chooses no RNG",
        )
    record = _unique_mapping(value)
    if (
        row.reason == "non_scalar_seed_list_member"
        and record is not None
        and type(record.get("seed")) is int
        and record["seed"] >= 0
    ):
        return SeedMeaning(
            row,
            "seed_record",
            record["seed"],
            "record container with explicit seed; child declaration retained separately",
        )
    return SeedMeaning(
        row, "unresolved", None, "insufficient schema/provenance; unknown usage remains unknown"
    )


def _pointer(parent: str, part: str | int) -> str:
    return parent + "/" + str(part).replace("~", "~0").replace("/", "~1")


def _fields(value: Any, pointer: str) -> Iterator[tuple[str, Any, str]]:
    rows = _mapping(value)
    if rows is None:
        return
    canonical = _canonical(value)
    for index, (name, item) in enumerate(rows):
        location = (
            _pointer(_pointer(_pointer(pointer, "dict"), index), 1)
            if canonical
            else _pointer(pointer, name)
        )
        yield name, item, location


def _seed_maps(value: Any, pointer: str = "", depth: int = 0) -> Iterator[tuple[Any, str]]:
    if depth > 256:
        raise ValueError("seed map metadata nesting exceeds the explicit 256-level limit")
    if type(value) is dict:
        for name, item, location in _fields(value, pointer):
            if name == "seed_map":
                yield item, location
            yield from _seed_maps(item, location, depth + 1)
    elif type(value) is list:
        for index, item in enumerate(value):
            yield from _seed_maps(item, _pointer(pointer, index), depth + 1)
    elif type(value) is str and value.lstrip().startswith(("{", "[")):
        try:
            nested = strict_seed_metadata_json(value)
        except ValueError:
            return
        yield from _seed_maps(nested, _pointer(pointer, "embedded_json"), depth + 1)


def _streams(
    metadata: Any,
) -> tuple[tuple[SeedStreamDeclaration, ...], tuple[UnresolvedSeedDeclaration, ...]]:
    declarations: list[SeedStreamDeclaration] = []
    issues: list[UnresolvedSeedDeclaration] = []
    for mapping, pointer in _seed_maps(metadata):
        rows = _mapping(mapping)
        if rows is None:
            issues.append(
                UnresolvedSeedDeclaration(
                    pointer, "seed_map", type(mapping).__name__, "invalid_seed_map"
                )
            )
            continue
        if not rows:
            issues.append(UnresolvedSeedDeclaration(pointer, "seed_map", "dict", "empty_seed_map"))
        if len({name for name, _ in rows}) != len(rows):
            issues.append(
                UnresolvedSeedDeclaration(
                    pointer, "seed_map", "dict", "duplicate_canonical_seed_map_keys"
                )
            )
        for name, entry, location in _fields(mapping, pointer):
            if not name.isascii() or not name.isdecimal() or _mapping(entry) is None:
                issues.append(
                    UnresolvedSeedDeclaration(
                        location,
                        "seed_map",
                        type(entry).__name__,
                        "unresolved_seed_map_base_or_streams",
                    )
                )
                continue
            base = int(name)
            declarations.append(SeedStreamDeclaration(location, base, "base_seed_key", base))
            stream_rows = _mapping(entry)
            assert stream_rows is not None
            if not stream_rows or len({stream for stream, _ in stream_rows}) != len(stream_rows):
                issues.append(
                    UnresolvedSeedDeclaration(
                        location, "seed_map", "dict", "empty_or_duplicate_mapped_streams"
                    )
                )
            for stream, item, stream_location in _fields(entry, location):
                if type(item) is int and item >= 0:
                    declarations.append(SeedStreamDeclaration(stream_location, base, stream, item))
                else:
                    issues.append(
                        UnresolvedSeedDeclaration(
                            stream_location,
                            stream,
                            type(item).__name__,
                            "unresolved_mapped_stream_value",
                        )
                    )
    return tuple(declarations), tuple(issues)


def audit_seed_metadata(metadata: Any) -> MetadataSeedEvidence:
    """Attach context without erasing any original declaration or uncertainty."""
    original = collect_seed_usage(metadata)
    resolver = _LocationResolver(metadata)
    declarations = tuple(
        _meaning(row, *resolver.resolve(row.pointer)) for row in original.declarations
    )
    unresolved = tuple(_meaning(row, *resolver.resolve(row.pointer)) for row in original.unresolved)
    streams, issues = _streams(metadata)
    return MetadataSeedEvidence(declarations, unresolved, streams, issues)
