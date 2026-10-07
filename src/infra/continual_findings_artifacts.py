"""Publish exclusive complete findings and freshly reconstruct every readback.

Inputs are fixed current bindings and complete reader ports. Outputs are whole
request/result/Markdown/audit or an owned failure marker. Original bodies and
gates remain unchanged; no model/data/score/final view or selection occurs.
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
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_confirmation_scoring_execution import elapsed_seconds
from src.app.continual_findings_readers import FindingsReaders
from src.infra.continual_confirmation_io import read_json, write_exclusive
from src.infra.continual_confirmation_training_references import stream_file_identity
from src.infra.continual_findings_current_bindings import (
    BUDGETS,
    checked_findings_request,
    findings_request,
)
from src.infra.continual_findings_current_inputs import rebuild_current_findings


def artifact_paths(directory: Path) -> dict[str, Path]:
    paths = {
        name: directory / f"findings.{name}.json"
        for name in ("request", "result", "audit", "failure")
    }
    paths["markdown"], paths["claim"] = directory / "findings.md", directory / "findings.claim"
    return paths


def _complete(paths: dict[str, Path]) -> None:
    require(
        not paths["claim"].exists()
        and not paths["failure"].exists()
        and all(paths[name].is_file() for name in ("request", "result", "markdown", "audit")),
        "findings require a complete successful bundle",
    )


@contextmanager
def _claim(paths: dict[str, Path]) -> Iterator[dict[str, Any]]:
    # A unique owner prevents cleanup from removing a competing writer's claim.
    body = {"schema_id": "p612_complete_findings_claim_v1", "owner": str(uuid4())}
    identity = canonical_body_identity(body)
    write_exclusive(paths["claim"], body)
    try:
        same_json(stream_file_identity(paths["claim"]), identity, "findings claim ownership")
        occupied = [str(path) for name, path in paths.items() if name != "claim" and path.exists()]
        if occupied:
            raise FileExistsError(f"findings occupied after claim: {occupied}")
        yield identity
    finally:
        if paths["claim"].is_file() and stream_file_identity(paths["claim"]) == identity:
            paths["claim"].unlink()


def _json(path: Path, body: dict[str, Any]) -> dict[str, Any]:
    identity = canonical_body_identity(body)
    write_exclusive(path, body)
    same_json(stream_file_identity(path), identity, "findings published canonical JSON bytes")
    return identity


def _markdown_identity(text: str) -> dict[str, Any]:
    raw = text.encode("utf-8")
    return {"byte_count": len(raw), "sha256": sha256(raw).hexdigest()}


def _body_files(paths: dict[str, Path], body: dict[str, Any], text: str) -> dict[str, Any]:
    result = _json(paths["result"], body)
    with paths["markdown"].open("xb") as stream:
        stream.write(text.encode("utf-8"))
    markdown = _markdown_identity(text)
    same_json(
        stream_file_identity(paths["markdown"]), markdown, "findings published Markdown bytes"
    )
    return {"result": result, "markdown": markdown}


def _audit(
    request: dict[str, Any],
    request_id: dict[str, Any],
    files: dict[str, Any],
    body: dict[str, Any],
    ports: dict[str, Any],
    elapsed: Any,
    pure_elapsed: Any,
) -> dict[str, Any]:
    return {
        "schema_id": "p612_complete_findings_audit_v1",
        "status": "completed",
        "protocol_id": request["protocol_id"],
        "request_identity": request_id,
        "source_map_sha256": request["source_map_sha256"],
        "complete_bindings_identity": canonical_body_identity(request["bindings"]),
        "files": files,
        "coverage": body["coverage"],
        "current_complete_reader_ports": ports,
        "derivative_validation_elapsed_seconds": elapsed_seconds(
            elapsed, "findings full derivative validation", BUDGETS["outer_operation"]
        ),
        "pure_synthesis_and_artifacts_elapsed_seconds": elapsed_seconds(
            pure_elapsed,
            "findings pure synthesis and artifacts",
            BUDGETS["pure_synthesis_and_presentation"],
        ),
        "inner_budget_seconds": BUDGETS.copy(),
        "validation_scope": "complete_current_bindings_fresh_complete_reader_ports_and_complete_unchanged_pure_findings",
        "new_scientific_operation_or_final_source_access": False,
        "original_P6_12_acceptance_complete": False,
    }


def _bytes(paths: dict[str, Path], identities: dict[str, Any]) -> None:
    for name, identity in identities.items():
        same_json(stream_file_identity(paths[name]), identity, "findings late " + name + " bytes")


def _failure(paths: dict[str, Path], error: BaseException, started: float) -> None:
    write_exclusive(
        paths["failure"],
        {
            "schema_id": "p612_complete_findings_failure_v1",
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "request_identity": stream_file_identity(paths["request"])
            if paths["request"].is_file()
            else None,
            "derivative_elapsed_seconds": monotonic() - started,
        },
    )


def _finish(
    root: Path,
    paths: dict[str, Path],
    scope_file: Path,
    request: dict[str, Any],
    identities: dict[str, Any],
    started: float,
    claim: dict[str, Any] | None = None,
) -> None:
    same_json(
        checked_findings_request(root, paths, scope_file),
        request,
        "findings complete final current bindings",
    )
    _bytes(paths, identities)
    if claim is None:
        _complete(paths)
    else:
        require(not paths["failure"].exists(), "findings late failure marker")
        same_json(stream_file_identity(paths["claim"]), claim, "findings final claim ownership")
    elapsed_seconds(
        monotonic() - started, "findings final complete operation", BUDGETS["outer_operation"]
    )


def _publish_claimed(
    root: Path,
    paths: dict[str, Path],
    scope_file: Path,
    readers: FindingsReaders,
    claim: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    owned, audit_id = False, None
    try:
        request = findings_request(
            root, paths["request"].parent, scope_file, datetime.now(timezone.utc).isoformat()
        )
        request_id = _json(paths["request"], request)
        owned = True
        body, text, evidence = rebuild_current_findings(root, paths, scope_file, readers, request)
        files_started = monotonic()
        files = _body_files(paths, body, text)
        pure_elapsed = evidence["pure_derivation_elapsed_seconds"] + monotonic() - files_started
        audit = _audit(
            request,
            request_id,
            files,
            body,
            evidence["current_complete_reader_ports"],
            monotonic() - started,
            pure_elapsed,
        )
        audit_id = _json(paths["audit"], audit)
        _finish(
            root,
            paths,
            scope_file,
            request,
            {"request": request_id, **files, "audit": audit_id},
            started,
            claim,
        )
        return audit
    except BaseException as error:
        if not owned and isinstance(error, FileExistsError):
            raise
        try:
            _failure(paths, error, started)
        except BaseException:
            if (
                audit_id is not None
                and paths["audit"].is_file()
                and stream_file_identity(paths["audit"]) == audit_id
            ):
                paths["audit"].unlink()
            raise
        raise


def publish_complete_findings(
    root: Path, output_dir: Path, scope_file: Path, readers: FindingsReaders
) -> dict[str, Any]:
    started = monotonic()
    paths = artifact_paths(output_dir.resolve())
    occupied = [str(path) for path in paths.values() if path.exists()]
    if occupied:
        raise FileExistsError(f"findings output already exists: {occupied}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with _claim(paths) as claim:
        return _publish_claimed(root, paths, scope_file, readers, claim, started)


def read_completed_findings(
    root: Path, output_dir: Path, scope_file: Path, readers: FindingsReaders
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    started = monotonic()
    paths = artifact_paths(output_dir.resolve())
    _complete(paths)
    identities = {
        name: stream_file_identity(paths[name])
        for name in ("request", "result", "markdown", "audit")
    }
    request = checked_findings_request(root, paths, scope_file)
    expected, text, evidence = rebuild_current_findings(root, paths, scope_file, readers, request)
    files_started = monotonic()
    body, audit = read_json(paths["result"]), read_json(paths["audit"])
    same_json(body, expected, "findings whole independent reconstructed result")
    same_json(
        canonical_body_identity(body),
        identities["result"],
        "findings canonical stored result bytes",
    )
    same_json(
        _markdown_identity(text),
        identities["markdown"],
        "findings whole independent reconstructed Markdown",
    )
    same_json(
        canonical_body_identity(audit), identities["audit"], "findings canonical stored audit bytes"
    )
    same_json(
        audit,
        _audit(
            request,
            identities["request"],
            {name: identities[name] for name in ("result", "markdown")},
            expected,
            evidence["current_complete_reader_ports"],
            audit.get("derivative_validation_elapsed_seconds"),
            audit.get("pure_synthesis_and_artifacts_elapsed_seconds"),
        ),
        "findings whole independent reconstructed audit",
    )
    elapsed_seconds(
        evidence["pure_derivation_elapsed_seconds"] + monotonic() - files_started,
        "findings pure independent artifact verification",
        BUDGETS["pure_synthesis_and_presentation"],
    )
    _finish(root, paths, scope_file, request, identities, started)
    return request, body, audit
