"""Publish exclusive resource inventory files and fully reconstruct readback.

Inputs are fixed current saved-evidence bindings; output is request/result/
Markdown/audit or explicit failure. No model/data, scientific execution,
new resource measurement or original P6.10 completion claim belongs here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Any

from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_confirmation_resource_rendering import render_resource_inventory
from src.app.continual_confirmation_scoring_execution import elapsed_seconds
from src.infra.continual_confirmation_io import claim_artifacts, read_json, write_exclusive
from src.infra.continual_confirmation_resource_bindings import (
    VALIDATION_SECONDS,
    checked_resource_request,
    read_resource_inventory_inputs,
    resource_request,
)
from src.infra.continual_confirmation_training_references import stream_file_identity


def artifact_paths(directory: Path) -> dict[str, Path]:
    paths = {
        name: directory / f"resource-inventory.{name}.json"
        for name in ("request", "result", "audit", "failure")
    }
    paths["markdown"], paths["claim"] = (
        directory / "resource-inventory.md",
        directory / "resource-inventory.claim",
    )
    return paths


def _complete(paths: dict[str, Path]) -> None:
    require(
        not paths["claim"].exists()
        and not paths["failure"].exists()
        and all(paths[name].is_file() for name in ("request", "result", "markdown", "audit")),
        "resource inventory requires a complete successful bundle",
    )


def _json(path: Path, body: dict[str, Any]) -> dict[str, Any]:
    identity = canonical_body_identity(body)
    write_exclusive(path, body)
    same_json(stream_file_identity(path), identity, "resource published canonical JSON bytes")
    return identity


def _markdown_identity(value: str) -> dict[str, Any]:
    encoded = value.encode("utf-8")
    return {"byte_count": len(encoded), "sha256": sha256(encoded).hexdigest()}


def _body_files(paths: dict[str, Path], body: dict[str, Any]) -> dict[str, Any]:
    result = _json(paths["result"], body)
    text = render_resource_inventory(body)
    markdown = _markdown_identity(text)
    with paths["markdown"].open("xb") as stream:
        stream.write(text.encode("utf-8"))
    same_json(
        stream_file_identity(paths["markdown"]), markdown, "resource published Markdown bytes"
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
        "schema_id": "p610_resource_inventory_audit_v1",
        "status": "completed",
        "protocol_id": request["protocol_id"],
        "request_identity": request_id,
        "source_map_sha256": request["source_map_sha256"],
        "inputs": request["inputs"],
        "files": files,
        "coverage": body["coverage"],
        "derivative_validation_elapsed_seconds": elapsed_seconds(
            elapsed, "resource inventory validation", VALIDATION_SECONDS
        ),
        "validation_scope": "complete_current_saved_evidence_and_inventory_reconstruction_not_new_scientific_readback",
        "new_measurement_training_or_final_access": False,
        "original_P6_10_acceptance_complete": False,
    }


def _bytes(paths: dict[str, Path], identities: dict[str, Any]) -> None:
    for name, identity in identities.items():
        same_json(stream_file_identity(paths[name]), identity, f"resource late {name} bytes")


def _failure(paths: dict[str, Path], error: BaseException, started: float) -> None:
    write_exclusive(
        paths["failure"],
        {
            "schema_id": "p610_resource_inventory_failure_v1",
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "request_identity": stream_file_identity(paths["request"])
            if paths["request"].is_file()
            else None,
            "derivative_elapsed_seconds": monotonic() - started,
        },
    )


def _publish_claimed(root: Path, paths: dict[str, Path], scope_file: Path) -> dict[str, Any]:
    started, owned, audit_identity = monotonic(), False, None
    try:
        request = resource_request(
            root, paths["request"].parent, scope_file, datetime.now(timezone.utc).isoformat()
        )
        request_id = _json(paths["request"], request)
        owned = True
        body = read_resource_inventory_inputs(root, paths, scope_file)
        same_json(
            checked_resource_request(root, paths, scope_file),
            request,
            "resource prepublication bindings",
        )
        elapsed_seconds(monotonic() - started, "resource prepublication", VALIDATION_SECONDS)
        files = _body_files(paths, body)
        audit = _audit(request, request_id, files, body, monotonic() - started)
        audit_identity = _json(paths["audit"], audit)
        same_json(
            checked_resource_request(root, paths, scope_file), request, "resource final bindings"
        )
        _bytes(paths, {"request": request_id, **files, "audit": audit_identity})
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


def publish_resource_inventory(root: Path, output_dir: Path, scope_file: Path) -> dict[str, Any]:
    paths = artifact_paths(output_dir.resolve())
    occupied = [str(path) for path in paths.values() if path.exists()]
    if occupied:
        raise FileExistsError(f"resource inventory output already exists: {occupied}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with claim_artifacts(paths):
        return _publish_claimed(root, paths, scope_file)


def read_completed_resource_inventory(
    root: Path, output_dir: Path, scope_file: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    started = monotonic()
    paths = artifact_paths(output_dir.resolve())
    _complete(paths)
    identities = {
        name: stream_file_identity(paths[name])
        for name in ("request", "result", "markdown", "audit")
    }
    request = checked_resource_request(root, paths, scope_file)
    expected = read_resource_inventory_inputs(root, paths, scope_file)
    body, audit = read_json(paths["result"]), read_json(paths["audit"])
    same_json(body, expected, "resource complete rebuilt inventory")
    same_json(
        canonical_body_identity(body), identities["result"], "resource canonical result bytes"
    )
    same_json(
        _markdown_identity(render_resource_inventory(expected)),
        identities["markdown"],
        "resource complete rebuilt Markdown",
    )
    same_json(canonical_body_identity(audit), identities["audit"], "resource canonical audit bytes")
    same_json(
        audit,
        _audit(
            request,
            identities["request"],
            {name: identities[name] for name in ("result", "markdown")},
            expected,
            audit.get("derivative_validation_elapsed_seconds"),
        ),
        "resource complete rebuilt audit",
    )
    same_json(
        checked_resource_request(root, paths, scope_file),
        request,
        "resource final readback bindings",
    )
    _bytes(paths, identities)
    _complete(paths)
    elapsed_seconds(monotonic() - started, "resource independent readback", VALIDATION_SECONDS)
    return request, body, audit
