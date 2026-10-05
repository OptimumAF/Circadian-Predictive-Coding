"""Publish exclusive retention parts and independently reconstruct every field.

Inputs are fixed current bindings and the unchanged complete training reader.
Output is request/result/Markdown/audit or an explicit owned failure marker.
No data/model/training, scoring, final view or resource profiling belongs here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Any

from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_confirmation_retention_rendering import render_retention_costs
from src.app.continual_confirmation_scoring_execution import elapsed_seconds
from src.infra.continual_confirmation_io import claim_artifacts, read_json, write_exclusive
from src.infra.continual_confirmation_retention_bindings import (
    VALIDATION_SECONDS,
    checked_retention_request,
    read_retention_inputs,
    retention_request,
)
from src.infra.continual_confirmation_training_references import (
    CompletedTrainingReader,
    stream_file_identity,
)


def artifact_paths(directory: Path) -> dict[str, Path]:
    paths = {
        name: directory / f"retention-costs.{name}.json"
        for name in ("request", "result", "audit", "failure")
    }
    paths["markdown"], paths["claim"] = (
        directory / "retention-costs.md",
        directory / "retention-costs.claim",
    )
    return paths


def _complete(paths: dict[str, Path]) -> None:
    require(
        not paths["claim"].exists()
        and not paths["failure"].exists()
        and all(paths[name].is_file() for name in ("request", "result", "markdown", "audit")),
        "retention costs require a complete successful bundle",
    )


def _json(path: Path, body: dict[str, Any]) -> dict[str, Any]:
    identity = canonical_body_identity(body)
    write_exclusive(path, body)
    same_json(stream_file_identity(path), identity, "retention published canonical JSON bytes")
    return identity


def _markdown_identity(text: str) -> dict[str, Any]:
    encoded = text.encode("utf-8")
    return {"byte_count": len(encoded), "sha256": sha256(encoded).hexdigest()}


def _body_files(paths: dict[str, Path], body: dict[str, Any]) -> dict[str, Any]:
    result = _json(paths["result"], body)
    text = render_retention_costs(body)
    with paths["markdown"].open("xb") as stream:
        stream.write(text.encode("utf-8"))
    markdown = _markdown_identity(text)
    same_json(
        stream_file_identity(paths["markdown"]), markdown, "retention published Markdown bytes"
    )
    return {"result": result, "markdown": markdown}


def _audit(
    request: dict[str, Any],
    request_id: dict[str, Any],
    files: dict[str, Any],
    body: dict[str, Any],
    elapsed: Any,
) -> dict[str, Any]:
    return {
        "schema_id": "p610_retention_costs_audit_v1",
        "status": "completed",
        "protocol_id": request["protocol_id"],
        "request_identity": request_id,
        "source_map_sha256": request["source_map_sha256"],
        "inputs": request["inputs"],
        "files": files,
        "coverage": body["coverage"],
        "stage_totals": body["stage_totals"],
        "complete_training_references_identity": canonical_body_identity(
            body["provenance"]["training_references"]
        ),
        "derivative_validation_elapsed_seconds": elapsed_seconds(
            elapsed, "retention costs validation", VALIDATION_SECONDS
        ),
        "validation_scope": "two_current_source_bound_unchanged_complete_original_training_readers_and_full_retention_projection",
        "new_measurement_training_or_final_access": False,
        "original_P6_10_acceptance_complete": False,
    }


def _bytes(paths: dict[str, Path], identities: dict[str, Any]) -> None:
    for name, identity in identities.items():
        same_json(stream_file_identity(paths[name]), identity, f"retention late {name} bytes")


def _failure(paths: dict[str, Path], error: BaseException, started: float) -> None:
    write_exclusive(
        paths["failure"],
        {
            "schema_id": "p610_retention_costs_failure_v1",
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
    root: Path, paths: dict[str, Path], scope_file: Path, reader: CompletedTrainingReader
) -> dict[str, Any]:
    started, owned, audit_identity = monotonic(), False, None
    try:
        request = retention_request(
            root, paths["request"].parent, scope_file, datetime.now(timezone.utc).isoformat()
        )
        request_id = _json(paths["request"], request)
        owned = True
        body = read_retention_inputs(root, paths, scope_file, reader)
        same_json(
            checked_retention_request(root, paths, scope_file),
            request,
            "retention prepublication bindings",
        )
        elapsed_seconds(monotonic() - started, "retention prepublication", VALIDATION_SECONDS)
        files = _body_files(paths, body)
        audit = _audit(request, request_id, files, body, monotonic() - started)
        audit_identity = _json(paths["audit"], audit)
        same_json(
            checked_retention_request(root, paths, scope_file), request, "retention final bindings"
        )
        _bytes(paths, {"request": request_id, **files, "audit": audit_identity})
        require(
            paths["claim"].is_file() and not paths["failure"].exists(),
            "retention late publication marker differs",
        )
        elapsed_seconds(monotonic() - started, "retention final publication", VALIDATION_SECONDS)
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


def publish_retention_costs(
    root: Path, output_dir: Path, scope_file: Path, reader: CompletedTrainingReader
) -> dict[str, Any]:
    paths = artifact_paths(output_dir.resolve())
    occupied = [str(path) for path in paths.values() if path.exists()]
    if occupied:
        raise FileExistsError(f"retention costs output already exists: {occupied}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with claim_artifacts(paths):
        return _publish_claimed(root, paths, scope_file, reader)


def read_completed_retention_costs(
    root: Path, output_dir: Path, scope_file: Path, reader: CompletedTrainingReader
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    started = monotonic()
    paths = artifact_paths(output_dir.resolve())
    _complete(paths)
    identities = {
        name: stream_file_identity(paths[name])
        for name in ("request", "result", "markdown", "audit")
    }
    request = checked_retention_request(root, paths, scope_file)
    expected = read_retention_inputs(root, paths, scope_file, reader)
    body, audit = read_json(paths["result"]), read_json(paths["audit"])
    same_json(body, expected, "retention complete rebuilt ledger")
    same_json(
        canonical_body_identity(body), identities["result"], "retention canonical result bytes"
    )
    same_json(
        _markdown_identity(render_retention_costs(expected)),
        identities["markdown"],
        "retention complete rebuilt Markdown",
    )
    same_json(
        canonical_body_identity(audit), identities["audit"], "retention canonical audit bytes"
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
        "retention complete rebuilt audit",
    )
    same_json(
        checked_retention_request(root, paths, scope_file),
        request,
        "retention final readback bindings",
    )
    _bytes(paths, identities)
    _complete(paths)
    elapsed_seconds(monotonic() - started, "retention independent readback", VALIDATION_SECONDS)
    return request, body, audit
