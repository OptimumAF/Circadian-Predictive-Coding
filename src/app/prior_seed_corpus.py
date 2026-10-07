"""Validate complete decoded saved seed membership and every original alias.

Inputs are the complete versioned saved report. Outputs are its original
ordered contents after full physical/Git identity and census reconciliation.
This owns no filesystem, semantic seed interpretation, source execution,
chronological inference, numeric selection or admission authority.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.core.seed_stream_screening import EvidenceIdentity, validate_evidence_identity


_REPORT_FIELDS = frozenset(
    {
        "schema_id",
        "files",
        "history",
        "contents",
        "physical_file_count",
        "physical_byte_count",
        "git_object_count",
        "git_object_byte_count",
        "distinct_complete_content_count",
        "all_aliases_retained",
        "exact_duplicate_parsing_only",
        "limitations",
        "complete_prior_usage_acceptance",
        "fresh_roles_authorized",
        "original_p67_acceptance_complete",
    }
)
_CONTENT_FIELDS = {"byte_count", "sha256", "aliases", "text", "json", "jsonl", "csv", "issues"}


def _shape(value: Any, fields: set[str] | frozenset[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError("saved seed corpus has missing or unknown fields")
    return value


def _ordered_strings(value: Any, *, hexadecimal: bool = False) -> list[str]:
    if type(value) is not list or any(type(item) is not str or not item for item in value):
        raise ValueError("saved seed corpus requires complete ordered strings")
    if value != sorted(set(value)):
        raise ValueError("saved seed corpus strings are duplicate or reordered")
    if hexadecimal and any(
        len(item) != 40 or any(c not in "0123456789abcdef" for c in item) for item in value
    ):
        raise ValueError("saved seed corpus Git identities differ")
    return value


def _relative_path(value: Any) -> str:
    if (
        type(value) is not str
        or not value
        or "\0" in value
        or "\\" in value
        or value.startswith("/")
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError("saved seed corpus relative path differs")
    return value


def _append(
    groups: dict[str, list[str]], sizes: dict[str, int], row: dict[str, Any], aliases: list[str]
) -> None:
    identity = EvidenceIdentity(row["byte_count"], row["sha256"])
    validate_evidence_identity(identity)
    if identity.sha256 in sizes and sizes[identity.sha256] != identity.byte_count:
        raise ValueError("saved seed corpus same-content byte counts differ")
    sizes[identity.sha256] = identity.byte_count
    groups[identity.sha256].extend(aliases)


def _physical_membership(
    report: dict[str, Any], groups: dict[str, list[str]], sizes: dict[str, int]
) -> None:
    files = report["files"]
    if type(files) is not list or not files:
        raise ValueError("saved seed corpus physical membership is empty or invalid")
    names = []
    for value in files:
        row = _shape(value, {"path", "byte_count", "sha256"})
        name = _relative_path(row["path"])
        names.append(name)
        _append(groups, sizes, row, [name])
    if names != sorted(set(names)):
        raise ValueError("saved seed corpus physical membership differs")


def _object_aliases(row: dict[str, Any], commits: list[str]) -> list[str]:
    pairs = row["commit_path_aliases"]
    if type(pairs) is not list or any(type(pair) is not list or len(pair) != 2 for pair in pairs):
        raise ValueError("saved seed corpus Git aliases differ")
    normalized = []
    for commit, path in pairs:
        if commit not in commits:
            raise ValueError("saved seed corpus alias names an unknown commit")
        normalized.append((commit, _relative_path(path)))
    if normalized != sorted(set(normalized)):
        raise ValueError("saved seed corpus Git aliases are duplicate or reordered")
    aliases = [f"git:{commit}:{path}" for commit, path in normalized]
    fallback = (
        f"git-unreachable:{row['object_id']}"
        if row["kind"] == "blob"
        else f"git-metadata:{row['kind']}:{row['object_id']}"
    )
    return aliases or [fallback]


def _git_membership(
    report: dict[str, Any], groups: dict[str, list[str]], sizes: dict[str, int]
) -> None:
    history = _shape(report["history"], {"commits", "objects"})
    commits = _ordered_strings(history["commits"], hexadecimal=True)
    objects = history["objects"]
    if type(objects) is not list:
        raise ValueError("saved seed corpus Git object membership differs")
    names = []
    for value in objects:
        row = _shape(value, {"object_id", "kind", "byte_count", "sha256", "commit_path_aliases"})
        _ordered_strings([row["object_id"]], hexadecimal=True)
        if row["kind"] not in {"blob", "commit", "tree", "tag"}:
            raise ValueError("saved seed corpus Git object kind differs")
        names.append(row["object_id"])
        _append(groups, sizes, row, _object_aliases(row, commits))
    if names != sorted(set(names)):
        raise ValueError("saved seed corpus Git object membership differs")


def _require_census(report: dict[str, Any], contents: list[dict[str, Any]]) -> None:
    expected = {
        "physical_file_count": len(report["files"]),
        "physical_byte_count": sum(row["byte_count"] for row in report["files"]),
        "git_object_count": len(report["history"]["objects"]),
        "git_object_byte_count": sum(row["byte_count"] for row in report["history"]["objects"]),
        "distinct_complete_content_count": len(contents),
    }
    if any(
        type(report[name]) is not int or report[name] != count for name, count in expected.items()
    ):
        raise ValueError("saved seed corpus complete census differs")


def validated_seed_contents(report: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Reject any omitted/extra content, alias, whole identity or census claim."""
    _shape(report, _REPORT_FIELDS)
    if (
        report["schema_id"] != "complete_retained_prior_seed_evidence_v1"
        or report["all_aliases_retained"] is not True
        or report["exact_duplicate_parsing_only"] is not True
        or type(report["limitations"]) is not str
    ):
        raise ValueError("saved seed corpus version/limitations differ")
    if any(
        report[name] is not False
        for name in (
            "complete_prior_usage_acceptance",
            "fresh_roles_authorized",
            "original_p67_acceptance_complete",
        )
    ):
        raise ValueError("saved seed declarations cannot grant execution or release authority")
    groups: dict[str, list[str]] = defaultdict(list)
    sizes: dict[str, int] = {}
    _physical_membership(report, groups, sizes)
    _git_membership(report, groups, sizes)
    contents = report["contents"]
    if type(contents) is not list or len(contents) != len(groups):
        raise ValueError("saved seed corpus complete content membership differs")
    for digest, value in zip(sorted(groups), contents, strict=True):
        row = _shape(value, _CONTENT_FIELDS)
        if (
            row["sha256"] != digest
            or type(row["byte_count"]) is not int
            or row["byte_count"] != sizes[digest]
            or row["aliases"] != sorted(groups[digest])
        ):
            raise ValueError("saved seed corpus whole content identity/aliases differ")
    _require_census(report, contents)
    return tuple(contents)
