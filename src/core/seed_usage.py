"""Extract declared seed identities from complete decoded metadata.

Inputs are JSON metadata, including canonical mappings, embedded JSON and CLI
arguments. Outputs preserve every seed location and every unresolved declaration.
This is a conservative declaration inventory, not proof of execution or fresh
role authority. It owns no IO, scientific source, outcome selection or RNG.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any


@dataclass(frozen=True)
class SeedDeclaration:
    pointer: str
    field: str
    seed: int
    representation: str


@dataclass(frozen=True)
class UnresolvedSeedDeclaration:
    pointer: str
    field: str
    value_type: str
    reason: str


@dataclass(frozen=True)
class SeedUsage:
    declarations: tuple[SeedDeclaration, ...]
    unresolved: tuple[UnresolvedSeedDeclaration, ...]


_DECIMAL = re.compile(r"[0-9]+\Z", re.ASCII)
_ARGUMENTS = {"--seed", "--seeds", "--data-seed", "--random-seed", "--rng-seed"}
_SEED_QUANTITIES = {"distinct_confirmation_seeds", "distinct_pilot_source_seeds"}


def _seed_field(name: str) -> bool:
    normalized = name.lower()
    if normalized in _SEED_QUANTITIES:
        return False
    return normalized in {"seed", "seeds", "random_state"} or normalized.endswith(
        ("_seed", "_seeds")
    )


def _pointer(parent: str, part: str | int) -> str:
    return parent + "/" + str(part).replace("~", "~0").replace("/", "~1")


def strict_seed_metadata_json(raw: str | bytes) -> Any:
    """Do not discard an earlier seed via duplicate keys or nonfinite JSON."""

    def unique(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise ValueError("nonfinite JSON literal")

    try:
        return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("invalid seed metadata JSON") from error


class _Collector:
    def __init__(self) -> None:
        self.declarations: list[SeedDeclaration] = []
        self.unresolved: list[UnresolvedSeedDeclaration] = []

    def issue(self, value: Any, pointer: str, field: str, reason: str) -> None:
        self.unresolved.append(
            UnresolvedSeedDeclaration(pointer, field, type(value).__name__, reason)
        )

    def seed(self, value: Any, pointer: str, field: str) -> None:
        if type(value) is int and value >= 0:
            self.declarations.append(SeedDeclaration(pointer, field, value, "integer"))
        elif type(value) is str:
            parts = [part.strip() for part in value.split(",")]
            if parts and all(_DECIMAL.fullmatch(part) for part in parts):
                for index, part in enumerate(parts):
                    self.declarations.append(
                        SeedDeclaration(_pointer(pointer, index), field, int(part), "decimal_text")
                    )
            else:
                self.issue(value, pointer, field, "unresolved_seed_text")
        elif type(value) is list:
            for index, item in enumerate(value):
                if type(item) not in (int, str):
                    self.issue(item, _pointer(pointer, index), field, "non_scalar_seed_list_member")
                else:
                    self.seed(item, _pointer(pointer, index), field)
        elif type(value) is dict and set(value) in ({"tuple"}, {"list"}):
            tag = next(iter(value))
            if type(value[tag]) is list:
                self.seed(value[tag], _pointer(pointer, tag), field)
            else:
                self.issue(value, pointer, field, "invalid_canonical_seed_sequence")
        else:
            self.issue(value, pointer, field, "seed_is_not_a_nonnegative_integer_declaration")

    def arguments(self, value: list[Any], pointer: str) -> None:
        index = 0
        while index < len(value):
            item = value[index]
            if type(item) is not str:
                index += 1
                continue
            option, equal, inline = item.partition("=")
            if option not in _ARGUMENTS:
                index += 1
                continue
            if equal:
                self.seed(inline, _pointer(pointer, index), option)
                index += 1
                continue
            limit = index + 1
            while (
                limit < len(value)
                and type(value[limit]) is str
                and not value[limit].startswith("--")
            ):
                if option != "--seeds" and limit > index + 1:
                    break
                limit += 1
            if limit == index + 1:
                self.issue(None, _pointer(pointer, index), option, "missing_seed_argument")
            for position in range(index + 1, limit):
                self.seed(value[position], _pointer(pointer, position), option)
            index = limit

    def field(self, name: str, value: Any, pointer: str, depth: int) -> None:
        if _seed_field(name):
            self.seed(value, pointer, name)
        self.walk(value, pointer, depth)

    def walk(self, value: Any, pointer: str, depth: int = 0) -> None:
        if depth > 256:
            raise ValueError("seed metadata nesting exceeds the explicit 256-level limit")
        if type(value) is dict:
            if set(value) == {"dict"} and type(value["dict"]) is list:
                rows = value["dict"]
                if all(type(r) is list and len(r) == 2 and type(r[0]) is str for r in rows):
                    for index, (name, item) in enumerate(rows):
                        self.field(
                            name,
                            item,
                            _pointer(_pointer(_pointer(pointer, "dict"), index), 1),
                            depth + 1,
                        )
                    return
            for name, item in value.items():
                if type(name) is not str:
                    raise ValueError("seed metadata keys must be strings")
                self.field(name, item, _pointer(pointer, name), depth + 1)
        elif type(value) is list:
            self.arguments(value, pointer)
            for index, item in enumerate(value):
                self.walk(item, _pointer(pointer, index), depth + 1)
        elif type(value) is str and value.lstrip().startswith(("{", "[")):
            # Whole embedded configuration strings can contain all declarations.
            try:
                nested = strict_seed_metadata_json(value)
            except ValueError:
                if "seed" in value.lower():
                    self.issue(value, pointer, "embedded_json", "unparsed_embedded_seed_metadata")
                return
            self.walk(nested, _pointer(pointer, "embedded_json"), depth + 1)


def collect_seed_usage(metadata: Any) -> SeedUsage:
    """Retain unresolved values; never silently certify their absence or execution."""
    collector = _Collector()
    collector.walk(metadata, "")
    return SeedUsage(tuple(collector.declarations), tuple(collector.unresolved))
