"""Publish exclusive matrix artifacts and reconstruct them from a complete reader.

Inputs are current request/input/source bindings and one complete report port.
Outputs are request/result/Markdown/audit or explicit failure. No model/data,
scientific evaluation, selection or new resource measurement belongs here.
Elapsed time is derivative validation; original scientific caps stay fixed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Any

from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_matrix_rendering import render_confirmation_matrix
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_confirmation_scoring_execution import elapsed_seconds
from src.infra.continual_confirmation_io import claim_artifacts, read_json, write_exclusive
from src.infra.continual_confirmation_matrix_bindings import (
    ReportReader,
    VALIDATION_SECONDS,
    checked_matrix_request,
    matrix_request,
    verified_matrix_body,
)
from src.infra.continual_confirmation_training_references import stream_file_identity


def artifact_paths(directory: Path) -> dict[str, Path]:
    paths = {
        name: directory / f"confirmation-matrix.{name}.json"
        for name in ("request", "result", "audit", "failure")
    }
    paths["markdown"] = directory / "confirmation-matrix.md"
    paths["claim"] = directory / "confirmation-matrix.claim"
    return paths


def _complete(paths: dict[str, Path]) -> None:
    require(
        not paths["claim"].exists()
        and not paths["failure"].exists()
        and all(paths[n].is_file() for n in ("request", "result", "markdown", "audit")),
        "matrix requires a complete successful bundle",
    )


def _json(path: Path, body: dict[str, Any]) -> dict[str, Any]:
    identity = canonical_body_identity(body)
    write_exclusive(path, body)
    same_json(stream_file_identity(path), identity, f"matrix published {path.name} bytes")
    return identity


def _markdown_identity(text: str) -> dict[str, Any]:
    encoded = text.encode("utf-8")
    return {"sha256": sha256(encoded).hexdigest(), "byte_count": len(encoded)}


def _body_files(paths: dict[str, Path], body: dict[str, Any]) -> dict[str, Any]:
    identity = _json(paths["result"], body)
    text = render_confirmation_matrix(body)
    markdown = _markdown_identity(text)
    with paths["markdown"].open("xb") as output:
        output.write(text.encode("utf-8"))
    same_json(stream_file_identity(paths["markdown"]), markdown, "matrix published Markdown bytes")
    return {"result": identity, "markdown": markdown}


def _audit(
    request: dict[str, Any],
    request_id: dict[str, Any],
    files: dict[str, Any],
    body: dict[str, Any],
    elapsed: Any,
) -> dict[str, Any]:
    observed = elapsed_seconds(elapsed, "matrix derivative validation", VALIDATION_SECONDS)
    return {
        "schema_id": "p69_confirmation_matrix_audit_v1",
        "status": "completed",
        "protocol_id": request["protocol_id"],
        "request_identity": request_id,
        "source_map_sha256": request["source_map_sha256"],
        "inputs": request["inputs"],
        "files": files,
        "coverage": body["coverage"],
        "report_identity": body["report_identity"],
        "derivative_validation_elapsed_seconds": observed,
        "validation_scope": "bound_complete_stored_matrix_and_artifact_readback",
        "new_training_or_final_source_access": False,
        "original_fully_measured_matrix_acceptance_complete": False,
    }


def _bytes(paths: dict[str, Path], identities: dict[str, Any]) -> None:
    for name, identity in identities.items():
        same_json(stream_file_identity(paths[name]), identity, f"matrix late {name} bytes")


def _failure(paths: dict[str, Path], error: BaseException, started: float) -> None:
    write_exclusive(
        paths["failure"],
        {
            "schema_id": "p69_confirmation_matrix_failure_v1",
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "request_identity": stream_file_identity(paths["request"])
            if paths["request"].is_file()
            else None,
            "derivative_elapsed_seconds": monotonic() - started,
        },
    )


def _publish_claimed(
    root: Path, paths: dict[str, Path], scope_file: Path, reader: ReportReader
) -> dict[str, Any]:
    started = monotonic()
    owned = False
    audit_identity = None
    try:
        request = matrix_request(
            root, paths["request"].parent, scope_file, datetime.now(timezone.utc).isoformat()
        )
        request_id = _json(paths["request"], request)
        owned = True
        body = verified_matrix_body(root, paths, scope_file, reader)
        same_json(
            checked_matrix_request(root, paths, scope_file),
            request,
            "matrix prepublication bindings",
        )
        elapsed_seconds(monotonic() - started, "matrix prepublication", VALIDATION_SECONDS)
        files = _body_files(paths, body)
        audit = _audit(request, request_id, files, body, monotonic() - started)
        audit_identity = _json(paths["audit"], audit)
        same_json(checked_matrix_request(root, paths, scope_file), request, "matrix final bindings")
        _bytes(paths, {"request": request_id, **files, "audit": audit_identity})
        return audit
    except BaseException as error:
        if not owned and isinstance(error, FileExistsError):
            raise
        try:
            _failure(paths, error, started)
        except BaseException:
            # A completion must not survive failed failure-marker IO. Revoke
            # only the still-identical audit owned by this publication.
            if (
                audit_identity is not None
                and paths["audit"].is_file()
                and stream_file_identity(paths["audit"]) == audit_identity
            ):
                paths["audit"].unlink()
            raise
        raise


def publish_confirmation_matrix(
    root: Path, output_dir: Path, scope_file: Path, reader: ReportReader
) -> dict[str, Any]:
    paths = artifact_paths(output_dir.resolve())
    occupied = [str(path) for path in paths.values() if path.exists()]
    if occupied:
        raise FileExistsError(f"confirmation matrix output already exists: {occupied}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with claim_artifacts(paths):
        return _publish_claimed(root, paths, scope_file, reader)


def read_completed_confirmation_matrix(
    root: Path, output_dir: Path, scope_file: Path, reader: ReportReader
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    started = monotonic()
    paths = artifact_paths(output_dir.resolve())
    _complete(paths)
    identities = {
        name: stream_file_identity(paths[name])
        for name in ("request", "result", "markdown", "audit")
    }
    request = checked_matrix_request(root, paths, scope_file)
    expected = verified_matrix_body(root, paths, scope_file, reader)
    body, audit = read_json(paths["result"]), read_json(paths["audit"])
    same_json(body, expected, "matrix complete independently rebuilt body")
    same_json(canonical_body_identity(body), identities["result"], "matrix canonical result bytes")
    same_json(
        _markdown_identity(render_confirmation_matrix(expected)),
        identities["markdown"],
        "matrix complete rebuilt Markdown",
    )
    same_json(canonical_body_identity(audit), identities["audit"], "matrix canonical audit bytes")
    files = {name: identities[name] for name in ("result", "markdown")}
    same_json(
        audit,
        _audit(
            request,
            identities["request"],
            files,
            expected,
            audit.get("derivative_validation_elapsed_seconds"),
        ),
        "matrix complete rebuilt audit",
    )
    same_json(
        checked_matrix_request(root, paths, scope_file), request, "matrix final readback bindings"
    )
    _bytes(paths, identities)
    _complete(paths)
    elapsed_seconds(monotonic() - started, "matrix independent readback", VALIDATION_SECONDS)
    return request, body, audit
