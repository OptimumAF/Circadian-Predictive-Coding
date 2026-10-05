"""Compose unchanged whole readers and rebuild every complete fixed finding.

Inputs are a just-validated current request and three explicit complete ports.
Outputs are the full unchanged pure body/Markdown and stable part identities.
No cached reader, model/data/scoring/final source, selection or new statistic.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Any

from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_confirmation_scoring_execution import elapsed_seconds
from src.app.continual_findings_readers import FindingsReaders
from src.app.continual_findings_synthesis import build_complete_findings
from src.app.continual_findings_synthesis_rendering import render_complete_findings
from src.infra.continual_confirmation_training_references import stream_file_identity
from src.infra.continual_findings_current_bindings import (
    BUDGETS,
    checked_findings_request,
    publication_declarations,
    read_bound_json,
)


def _bundle_reader(
    kind: str, reader: Any, reference: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = monotonic()
    parts = reader()
    elapsed_seconds(
        monotonic() - started,
        "findings complete " + kind + " reader",
        BUDGETS["outcome_cost_reader" if kind == "outcome_costs" else "matrix_reader"],
    )
    require(
        type(parts) is tuple and len(parts) == 3 and all(type(part) is dict for part in parts),
        "findings requires whole current " + kind + " request/result/audit",
    )
    identities = {}
    for name, body in zip(("request", "result", "audit"), parts, strict=True):
        identity = canonical_body_identity(body)
        same_json(
            identity,
            reference["parts"][name]["identity"],
            "findings whole returned " + kind + " " + name,
        )
        identities[name] = identity
    return parts[1], {"returned_complete": True, "whole_part_identities": identities}


def _current_reader_inputs(
    readers: FindingsReaders, catalog: dict[str, Any], synthesis_catalog: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    bodies, evidence = {}, {}
    for kind, reader in (("outcome_costs", readers.outcome_costs), ("matrix", readers.matrix)):
        reference = next(row for row in catalog["current_bundles"] if row["kind"] == kind)
        bodies[kind], evidence[kind] = _bundle_reader(kind, reader, reference)
    started = monotonic()
    bodies["development"] = readers.development()
    elapsed_seconds(
        monotonic() - started, "findings complete development reader", BUDGETS["development_reader"]
    )
    require(
        type(bodies["development"]) is dict,
        "findings requires the whole current development ledger",
    )
    identity = canonical_body_identity(bodies["development"])
    same_json(
        identity,
        synthesis_catalog["bodies"]["development"]["identity"],
        "findings whole returned current development",
    )
    evidence["development"] = {"returned_complete": True, "whole_ledger_identity": identity}
    return bodies, evidence


def _raw_record(root: Path, path: str, expected: dict[str, Any]) -> dict[str, Any]:
    location = root / path
    same_json(
        stream_file_identity(location), expected, "findings complete preserved record " + path
    )
    raw = location.read_bytes()
    same_json(
        {"byte_count": len(raw), "sha256": sha256(raw).hexdigest()},
        expected,
        "findings preserved record read bytes " + path,
    )
    same_json(
        stream_file_identity(location),
        expected,
        "findings preserved record changed during read " + path,
    )
    return {"identity": expected.copy(), "content": raw.decode("utf-8")}


def _synthesis_inputs(
    root: Path, catalog: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    bodies = current.copy()
    for name, reference in catalog["bodies"].items():
        if name not in bodies:
            bodies[name] = read_bound_json(root / reference["path"], reference["identity"])
    records = {
        path: _raw_record(root, path, expected) for path, expected in catalog["records"].items()
    }
    return {"catalog": catalog, "bodies": bodies, "preserved_records": records}


def rebuild_current_findings(
    root: Path,
    paths: dict[str, Path],
    scope_file: Path,
    readers: FindingsReaders,
    request: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Use only the caller's just-validated snapshot; no reuse across operations.

    Why this: repeating an immediate full precheck caused the earlier P6.10
    timeout. Complete current bindings still run before readers, after readers,
    and at final publication/readback. Every operation dispatches fresh ports.
    """
    same_json(
        canonical_body_identity(request),
        stream_file_identity(paths["request"]),
        "findings just-validated owned request snapshot",
    )
    catalog, _ = publication_declarations(root)
    reference = catalog["synthesis_catalog"]
    synthesis_catalog = read_bound_json(root / reference["path"], reference["identity"])
    current, evidence = _current_reader_inputs(readers, catalog, synthesis_catalog)
    same_json(
        checked_findings_request(root, paths, scope_file),
        request,
        "findings complete bindings changed after fresh readers",
    )
    # Load large raw contexts only after both original report readers return;
    # their unchanged time/resource gates stay independent of this derivative.
    started = monotonic()
    inputs = _synthesis_inputs(root, synthesis_catalog, current)
    body = build_complete_findings(inputs)
    same_json(
        canonical_body_identity(body),
        catalog["accepted_whole_result_identity"],
        "findings unchanged whole accepted pure result",
    )
    text = render_complete_findings(body)
    raw = text.encode("utf-8")
    same_json(
        {"byte_count": len(raw), "sha256": sha256(raw).hexdigest()},
        catalog["accepted_whole_markdown_identity"],
        "findings unchanged exhaustive accepted Markdown",
    )
    pure_elapsed = elapsed_seconds(
        monotonic() - started,
        "findings complete pure rebuild and presentation",
        BUDGETS["pure_synthesis_and_presentation"],
    )
    return (
        body,
        text,
        {
            "current_complete_reader_ports": evidence,
            "pure_derivation_elapsed_seconds": pure_elapsed,
        },
    )
