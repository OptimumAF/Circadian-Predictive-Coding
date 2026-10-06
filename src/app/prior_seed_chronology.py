"""Project all saved witnesses into conservative chronology and stream screening.

Inputs are a complete decoded saved report and its externally bound whole
identity, with optional caller-declared ordered bases for numeric screening.
Outputs bind every content/alias/section, semantic count and unknown chronology.
No source is constructed, expression executed, seed chosen or role admitted.
The filesystem boundary must bind input bytes; this consumer proves no new
execution/release event and does not replace the original resource gates.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from typing import Any, Iterator

from src.app.prior_seed_corpus import validated_seed_contents
from src.core.seed_stream_screening import (
    EvidenceIdentity,
    screen_seed_roles,
    validate_evidence_identity,
)


_MEANINGS = {
    "numeric_seed_candidate",
    "recorded_seed_inventory_value",
    "quantity",
    "seed_record",
    "nonrandom_retention_policy",
    "unrecorded_process_hash_seed",
    "unresolved",
}
_EXPRESSION_KINDS = {
    "assignment_or_default",
    "call_or_rng_state",
    "mapping_seed_field",
    "parameter_default",
    "unresolved_assignment",
    "unresolved_call",
    "unresolved_parameter_default",
}
_ISSUE_KINDS = {
    "opaque_asset_or_checkpoint",
    "opaque_or_undecodable",
    "opaque_binary",
    "unparsed_json",
    "unparsed_csv",
    "unreachable_blob_without_path",
}


def _shape(value: Any, keys: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError("saved seed witness has missing or unknown fields")
    return value


def _rows(value: Any) -> list[Any]:
    if type(value) is not list:
        raise ValueError("saved seed witness requires every ordered record")
    return value


def _count(value: Any, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("saved seed witness requires an exact nonnegative count/position")
    return value


def _string(value: Any) -> str:
    if type(value) is not str or not value:
        raise ValueError("saved seed witness requires a nonempty string")
    return value


def _digest(value: Any) -> str:
    identity = EvidenceIdentity(0, value)
    validate_evidence_identity(identity)
    return identity.sha256


def _metadata_views(
    content: dict[str, Any], coverage: Counter[str], unknown: Counter[str]
) -> Iterator[dict[str, Any]]:
    if content["json"] is not None:
        yield content["json"]
    if content["csv"] is not None:
        csv = _shape(
            content["csv"],
            {
                "row_count",
                "columns",
                "duplicate_headers",
                "row_issues",
                "metadata",
                "execution_or_fresh_role_authority",
            },
        )
        _count(csv["row_count"])
        columns = _rows(csv["columns"])
        # Why this: csv.reader allows blank names, including duplicates. Keep
        # their original positional metadata instead of discarding valid cells.
        if any(type(name) is not str for name in columns):
            raise ValueError("saved seed CSV header type differs")
        if (
            not columns
            or csv["duplicate_headers"]
            != sorted(name for name, count in Counter(columns).items() if count > 1)
            or csv["execution_or_fresh_role_authority"] is not False
        ):
            raise ValueError("saved seed CSV header/authority differs")
        for value in _rows(csv["row_issues"]):
            issue = _shape(value, {"row", "ending_line", "reason"})
            if _count(issue["row"]) >= csv["row_count"]:
                raise ValueError("saved seed CSV row issue lies outside its full row count")
            _count(issue["ending_line"], 1)
            _string(issue["reason"])
            unknown["csv_row_width"] += 1
        yield csv["metadata"]
    if content["jsonl"] is not None:
        # Why this: the saved reader counts every physical line from one,
        # retaining blank and malformed rows. Gaps cannot erase witnesses.
        for index, value in enumerate(_rows(content["jsonl"]), 1):
            row = _shape(value, {"line", "parse_status", "metadata", "parse_error"})
            if _count(row["line"]) != index or row["parse_status"] not in {
                "parsed",
                "blank",
                "unparsed",
            }:
                raise ValueError("saved seed JSONL complete ordered lines differ")
            coverage["jsonl_lines"] += 1
            if row["parse_status"] == "parsed":
                if row["metadata"] is None or row["parse_error"] is not None:
                    raise ValueError("saved seed JSONL parsed metadata differs")
                yield row["metadata"]
            elif row["metadata"] is not None or (
                row["parse_status"] == "blank" and row["parse_error"] is not None
            ):
                raise ValueError("saved seed JSONL failure metadata differs")
            elif row["parse_status"] == "unparsed":
                _string(row["parse_error"])
                unknown["unparsed_jsonl"] += 1


def _meaning(value: Any, section: str) -> tuple[str, int | None]:
    row = _shape(value, {"declaration", "meaning", "related_seed", "reason"})
    meaning = row["meaning"]
    if type(meaning) is not str or meaning not in _MEANINGS:
        raise ValueError("saved seed declaration meaning is unknown")
    declaration = _shape(
        row["declaration"],
        {"pointer", "field", "seed", "representation"}
        if section == "declarations"
        else {"pointer", "field", "value_type", "reason"},
    )
    if type(declaration["pointer"]) is not str or not declaration["pointer"].startswith("/"):
        raise ValueError("saved seed declaration pointer differs")
    _string(declaration["field"])
    _string(row["reason"])
    if section == "declarations":
        _count(declaration["seed"])
        if declaration["representation"] not in {"integer", "decimal_text"} or meaning not in {
            "quantity",
            "numeric_seed_candidate",
            "recorded_seed_inventory_value",
        }:
            raise ValueError("saved seed declaration representation/meaning differs")
    else:
        _string(declaration["value_type"])
        _string(declaration["reason"])
        if meaning not in {
            "quantity",
            "seed_record",
            "nonrandom_retention_policy",
            "unrecorded_process_hash_seed",
            "unresolved",
        }:
            raise ValueError("saved seed unresolved meaning differs")
    if meaning in {"numeric_seed_candidate", "recorded_seed_inventory_value", "seed_record"}:
        related = _count(row["related_seed"])
        if section == "declarations" and related != declaration["seed"]:
            raise ValueError("saved seed related identity differs")
        return meaning, related
    if row["related_seed"] is not None:
        raise ValueError("saved seed quantity/null/unknown cannot invent an RNG identity")
    return meaning, None


def _metadata(
    metadata: Any,
    coverage: Counter[str],
    meanings: Counter[str],
    candidates: Counter[int],
    unknown: Counter[str],
) -> None:
    view = _shape(metadata, {"declarations", "unresolved", "mapped_streams", "map_issues"})
    coverage["metadata_views"] += 1
    for section in ("declarations", "unresolved"):
        for value in _rows(view[section]):
            meaning, related = _meaning(value, section)
            coverage[section] += 1
            meanings[meaning] += 1
            if related is not None:
                candidates[related] += 1
            if meaning in {"unrecorded_process_hash_seed", "unresolved"}:
                unknown[meaning] += 1
    for value in _rows(view["mapped_streams"]):
        row = _shape(value, {"pointer", "base_seed", "stream", "value"})
        _string(row["pointer"])
        _string(row["stream"])
        candidates[_count(row["base_seed"])] += 1
        candidates[_count(row["value"])] += 1
        coverage["mapped_streams"] += 1
    for value in _rows(view["map_issues"]):
        row = _shape(value, {"pointer", "field", "value_type", "reason"})
        for item in row.values():
            _string(item)
        unknown["unresolved_seed_map"] += 1
        coverage["map_issues"] += 1


def _text_lines(lines: Any, coverage: Counter[str]) -> None:
    previous_line = 0
    for value in _rows(lines):
        row = _shape(value, {"line", "line_sha256", "references"})
        number = _count(row["line"], 1)
        if number <= previous_line:
            raise ValueError("saved seed text line order differs")
        previous_line = number
        _digest(row["line_sha256"])
        references = _rows(row["references"])
        if not references:
            raise ValueError("saved seed bearing line has no reference")
        previous_column = -1
        for value in references:
            reference = _shape(value, {"column", "token"})
            column = _count(reference["column"])
            if column <= previous_column:
                raise ValueError("saved seed text character column order differs")
            previous_column = column
            _string(reference["token"])
            coverage["text_references"] += 1
        coverage["seed_lines"] += 1


def _expressions(rows: Any, coverage: Counter[str], unknown: Counter[str]) -> None:
    previous: tuple[int, int, str, str] | None = None
    for value in _rows(rows):
        row = _shape(
            value,
            {"line", "column", "kind", "expression", "integer_literals_are_not_seed_role_proof"},
        )
        if row["kind"] not in _EXPRESSION_KINDS:
            raise ValueError("saved seed source expression kind differs")
        position = (
            _count(row["line"]),
            _count(row["column"]),
            row["kind"],
            _string(row["expression"]),
        )
        if previous is not None and position < previous:
            raise ValueError("saved seed source expression order differs")
        previous = position
        if any(
            type(item) is not int for item in _rows(row["integer_literals_are_not_seed_role_proof"])
        ):
            raise ValueError("saved seed AST literal type differs")
        coverage["python_expressions"] += 1
        unknown["source_expression_execution_unverified"] += 1


def _text(text: Any, coverage: Counter[str], unknown: Counter[str]) -> None:
    if text is None:
        return
    row = _shape(
        text,
        {
            "seed_lines",
            "python_expressions",
            "python_parse_status",
            "python_parse_error",
            "execution_or_fresh_role_authority",
        },
    )
    if row["execution_or_fresh_role_authority"] is not False or row["python_parse_status"] not in {
        "not_python",
        "parsed",
        "unparsed",
    }:
        raise ValueError("saved seed text parse/authority differs")
    _text_lines(row["seed_lines"], coverage)
    if row["python_parse_status"] == "unparsed":
        _string(row["python_parse_error"])
        unknown["unparsed_python"] += 1
    elif row["python_parse_error"] is not None:
        raise ValueError("saved seed text unexpected parse error")
    if row["python_parse_status"] != "parsed" and row["python_expressions"] != []:
        raise ValueError("saved seed unparsed text cannot claim AST records")
    _expressions(row["python_expressions"], coverage, unknown)


def _content(index: int, content: dict[str, Any]) -> dict[str, Any]:
    coverage: Counter[str] = Counter()
    meanings: Counter[str] = Counter()
    candidates: Counter[int] = Counter()
    unknown: Counter[str] = Counter()
    for metadata in _metadata_views(content, coverage, unknown):
        _metadata(metadata, coverage, meanings, candidates, unknown)
    _text(content["text"], coverage, unknown)
    for value in _rows(content["issues"]):
        issue = _shape(value, {"kind", "reason"})
        if issue["kind"] not in _ISSUE_KINDS:
            raise ValueError("saved seed content issue kind differs")
        _string(issue["reason"])
        unknown[issue["kind"]] += 1
    # Why this: retain every original witness via a full-content digest and
    # exact input pointer, avoiding another copy of millions of expressions.
    witness_digest = sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {
        "input_pointer": f"/contents/{index}",
        "byte_count": content["byte_count"],
        "sha256": content["sha256"],
        "aliases": list(content["aliases"]),
        "whole_saved_witness_record_sha256": witness_digest,
        "execution_and_release_status": "unverified",
        "coverage": dict(sorted(coverage.items())),
        "meaning_counts": dict(sorted(meanings.items())),
        "uncertainty_counts": dict(sorted(unknown.items())),
        "numeric_candidate_occurrences": [
            {"value": seed, "occurrences": count} for seed, count in sorted(candidates.items())
        ],
    }


def build_seed_chronology(
    report: dict[str, Any],
    evidence_identity: EvidenceIdentity,
    *,
    proposed_base_seeds: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Preserve the entire saved corpus and leave unproved role history open."""
    validate_evidence_identity(evidence_identity)
    contents = validated_seed_contents(report)
    ledger = [_content(index, content) for index, content in enumerate(contents)]
    coverage: Counter[str] = Counter()
    meanings: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    candidates: Counter[int] = Counter()
    for row in ledger:
        coverage.update(row["coverage"])
        meanings.update(row["meaning_counts"])
        unknown.update(row["uncertainty_counts"])
        for item in row["numeric_candidate_occurrences"]:
            candidates[item["value"]] += item["occurrences"]
    coverage.update(
        {
            "contents": len(ledger),
            "alias_occurrences": sum(len(row["aliases"]) for row in ledger),
            "physical_files": len(report["files"]),
            "git_objects": len(report["history"]["objects"]),
            "git_commits": len(report["history"]["commits"]),
        }
    )
    unknown["actual_content_execution_and_release_unverified"] = len(ledger)
    screen = screen_seed_roles(proposed_base_seeds, candidates, unknown)
    return {
        "schema_id": "conservative_saved_seed_chronology_v1",
        "input_identity": asdict(evidence_identity),
        "coverage": dict(sorted(coverage.items())),
        "meaning_counts": dict(sorted(meanings.items())),
        "uncertainty_counts": dict(sorted(unknown.items())),
        "content_ledger": ledger,
        "numeric_candidate_occurrences": [
            {"value": seed, "occurrences": count} for seed, count in sorted(candidates.items())
        ],
        "role_screen": asdict(screen),
        "verification_scope": "this_saved_projection_verifies_no_new_execution_or_actual_release_events",
        "verified_execution_events": 0,
        "verified_role_release_events": 0,
        "independent_source_replication_count": None,
        "all_original_witnesses_bound_by_input_pointer_and_whole_record_digest": True,
        "commit_aliases_and_copy_order_are_not_actual_release_chronology": True,
        "prospective_real_role_seeds_selected": False,
        "complete_prior_usage_acceptance": False,
        "fresh_roles_authorized": False,
        "original_p67_acceptance_complete": False,
    }
