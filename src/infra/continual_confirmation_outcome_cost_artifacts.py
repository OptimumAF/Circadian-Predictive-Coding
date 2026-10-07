"""Publish exclusive presentation parts and independently rebuild all of them.

Inputs are fixed current bindings and the whole unchanged official reader port.
Output is request/result/Markdown/audit or an explicit owned failure marker.
No models, data, scoring, final view, profiling, shared allocation or selection.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Any, Iterator
from uuid import uuid4

from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_outcome_cost_rendering import render_outcome_cost_presentation
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_confirmation_scoring_execution import elapsed_seconds
from src.infra.continual_confirmation_io import read_json, write_exclusive
from src.infra.continual_confirmation_outcome_cost_bindings import (
    VALIDATION_SECONDS,
    CompletedReportReader,
    checked_outcome_cost_request,
    outcome_cost_request,
    _read_bound_outcome_cost_inputs,
)
from src.infra.continual_confirmation_training_references import stream_file_identity


def artifact_paths(directory: Path) -> dict[str, Path]:
    paths = {
        name: directory / f"outcome-costs.{name}.json"
        for name in ("request", "result", "audit", "failure")
    }
    paths["markdown"], paths["claim"] = (
        directory / "outcome-costs.md",
        directory / "outcome-costs.claim",
    )
    return paths


def _complete(paths: dict[str, Path]) -> None:
    require(
        not paths["claim"].exists()
        and not paths["failure"].exists()
        and all(paths[name].is_file() for name in ("request", "result", "markdown", "audit")),
        "outcome costs require a complete successful bundle",
    )


@contextmanager
def _claim(paths: dict[str, Path]) -> Iterator[dict[str, Any]]:
    # Why this: a distinct owner token permits safe cleanup after another writer
    # replaces a claim. Foreign markers must survive failures and races.
    body = {"schema_id": "p610_outcome_costs_claim_v1", "owner": str(uuid4())}
    identity = canonical_body_identity(body)
    write_exclusive(paths["claim"], body)
    try:
        same_json(stream_file_identity(paths["claim"]), identity, "outcome costs claim ownership")
        occupied = [str(path) for name, path in paths.items() if name != "claim" and path.exists()]
        if occupied:
            raise FileExistsError(f"outcome costs occupied after claim: {occupied}")
        yield identity
    finally:
        if paths["claim"].is_file() and stream_file_identity(paths["claim"]) == identity:
            paths["claim"].unlink()


def _json(path: Path, body: dict[str, Any]) -> dict[str, Any]:
    identity = canonical_body_identity(body)
    write_exclusive(path, body)
    same_json(stream_file_identity(path), identity, "outcome costs published canonical JSON bytes")
    return identity


def _markdown_identity(text: str) -> dict[str, Any]:
    encoded = text.encode("utf-8")
    return {"byte_count": len(encoded), "sha256": sha256(encoded).hexdigest()}


def _body_files(paths: dict[str, Path], body: dict[str, Any]) -> dict[str, Any]:
    result = _json(paths["result"], body)
    text = render_outcome_cost_presentation(body)
    with paths["markdown"].open("xb") as stream:
        stream.write(text.encode("utf-8"))
    markdown = _markdown_identity(text)
    same_json(stream_file_identity(paths["markdown"]), markdown, "outcome costs published Markdown")
    return {"result": result, "markdown": markdown}


def _audit(
    request: dict[str, Any],
    request_id: dict[str, Any],
    files: dict[str, Any],
    body: dict[str, Any],
    elapsed: Any,
) -> dict[str, Any]:
    return {
        "schema_id": "p610_outcome_costs_audit_v1",
        "status": "completed",
        "protocol_id": request["protocol_id"],
        "request_identity": request_id,
        "source_map_sha256": request["source_map_sha256"],
        "inputs": request["inputs"],
        "files": files,
        "coverage": body["coverage"],
        "stage_storage_totals": body["stage_storage_totals"],
        "complete_original_report_parts": body["provenance"]["complete_original_report_parts"],
        "complete_original_report_readers": 1,
        "derivative_validation_elapsed_seconds": elapsed_seconds(
            elapsed, "outcome costs validation", VALIDATION_SECONDS
        ),
        "validation_scope": "one_current_source_bound_unchanged_whole_official_report_reader_and_all_original_outcomes_against_scoped_costs",
        "new_measurement_training_or_final_access": False,
        "original_P6_10_acceptance_complete": False,
    }


def _bytes(paths: dict[str, Path], identities: dict[str, Any]) -> None:
    for name, identity in identities.items():
        same_json(stream_file_identity(paths[name]), identity, "outcome costs late " + name)


def _failure(paths: dict[str, Path], error: BaseException, started: float) -> None:
    write_exclusive(
        paths["failure"],
        {
            "schema_id": "p610_outcome_costs_failure_v1",
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "request_identity": stream_file_identity(paths["request"])
            if paths["request"].is_file()
            else None,
            "derivative_elapsed_seconds": monotonic() - started,
        },
    )


def _finish_publication(
    root: Path,
    paths: dict[str, Path],
    scope_file: Path,
    request: dict[str, Any],
    identities: dict[str, Any],
    claim_identity: dict[str, Any],
    started: float,
) -> None:
    same_json(
        checked_outcome_cost_request(root, paths, scope_file),
        request,
        "outcome costs final bindings",
    )
    _bytes(paths, identities)
    require(not paths["failure"].exists(), "outcome costs late failure marker")
    same_json(
        stream_file_identity(paths["claim"]), claim_identity, "outcome costs final claim ownership"
    )
    elapsed_seconds(monotonic() - started, "outcome costs final publication", VALIDATION_SECONDS)


def _publish_claimed(
    root: Path,
    paths: dict[str, Path],
    scope_file: Path,
    reader: CompletedReportReader,
    claim_identity: dict[str, Any],
) -> dict[str, Any]:
    started, owned, audit_identity = monotonic(), False, None
    try:
        request = outcome_cost_request(
            root, paths["request"].parent, scope_file, datetime.now(timezone.utc).isoformat()
        )
        request_id = _json(paths["request"], request)
        owned = True
        body = _read_bound_outcome_cost_inputs(root, paths, scope_file, reader, request)
        elapsed_seconds(monotonic() - started, "outcome costs prepublication", VALIDATION_SECONDS)
        files = _body_files(paths, body)
        audit = _audit(request, request_id, files, body, monotonic() - started)
        audit_identity = _json(paths["audit"], audit)
        _finish_publication(
            root,
            paths,
            scope_file,
            request,
            {"request": request_id, **files, "audit": audit_identity},
            claim_identity,
            started,
        )
        return audit
    except BaseException as error:
        if not owned and isinstance(error, FileExistsError):
            raise
        try:
            _failure(paths, error, started)
        except BaseException:
            if (
                audit_identity is not None
                and paths["audit"].is_file()
                and stream_file_identity(paths["audit"]) == audit_identity
            ):
                paths["audit"].unlink()
            raise
        raise


def publish_outcome_costs(
    root: Path,
    output_dir: Path,
    scope_file: Path,
    reader: CompletedReportReader,
) -> dict[str, Any]:
    paths = artifact_paths(output_dir.resolve())
    occupied = [str(path) for path in paths.values() if path.exists()]
    if occupied:
        raise FileExistsError(f"outcome costs output already exists: {occupied}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with _claim(paths) as claim_identity:
        return _publish_claimed(root, paths, scope_file, reader, claim_identity)


def read_completed_outcome_costs(
    root: Path,
    output_dir: Path,
    scope_file: Path,
    reader: CompletedReportReader,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    started = monotonic()
    paths = artifact_paths(output_dir.resolve())
    _complete(paths)
    identities = {
        name: stream_file_identity(paths[name])
        for name in ("request", "result", "markdown", "audit")
    }
    request = checked_outcome_cost_request(root, paths, scope_file)
    expected = _read_bound_outcome_cost_inputs(root, paths, scope_file, reader, request)
    body, audit = read_json(paths["result"]), read_json(paths["audit"])
    same_json(body, expected, "outcome costs whole independently rebuilt presentation")
    same_json(
        canonical_body_identity(body), identities["result"], "outcome costs canonical result bytes"
    )
    same_json(
        _markdown_identity(render_outcome_cost_presentation(expected)),
        identities["markdown"],
        "outcome costs whole independently rebuilt Markdown",
    )
    same_json(
        canonical_body_identity(audit), identities["audit"], "outcome costs canonical audit bytes"
    )
    same_json(
        audit,
        _audit(
            request,
            identities["request"],
            {name: identities[name] for name in ("result", "markdown")},
            expected,
            audit.get("derivative_validation_elapsed_seconds"),
        ),
        "outcome costs whole independently rebuilt audit",
    )
    same_json(
        checked_outcome_cost_request(root, paths, scope_file),
        request,
        "outcome costs final readback bindings",
    )
    _bytes(paths, identities)
    _complete(paths)
    elapsed_seconds(monotonic() - started, "outcome costs independent readback", VALIDATION_SECONDS)
    return request, body, audit
