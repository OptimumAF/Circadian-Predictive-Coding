"""Own exclusive complete report publication and independently rebuilt readback.

Inputs are current bound requests and unchanged complete input-reader ports.
Output is request/result/Markdown/audit or an explicit failure. No scientific
data/model/scoring, cost/statistic selection or resource measurement here.
The recorded time is local derivative validation, separate from original caps.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Any

from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_confirmation_report_rendering import render_confirmation_report
from src.app.continual_confirmation_scoring_execution import elapsed_seconds
from src.infra.continual_confirmation_io import claim_artifacts, read_json, write_exclusive
from src.infra.continual_confirmation_report_bindings import (
    CostReader,
    ScoredReader,
    VALIDATION_SECONDS,
    checked_report_request,
    report_request,
    verified_report_body,
)
from src.infra.continual_confirmation_training_references import stream_file_identity


def artifact_paths(output_dir: Path) -> dict[str, Path]:
    paths = {
        name: output_dir / f"confirmation-report.{name}.json"
        for name in ("request", "result", "audit", "failure")
    }
    paths["markdown"] = output_dir / "confirmation-report.md"
    paths["claim"] = output_dir / "confirmation-report.claim"
    return paths


def _require_complete(paths: dict[str, Path]) -> None:
    require(
        not paths["failure"].exists()
        and not paths["claim"].exists()
        and all(paths[name].is_file() for name in ("request", "result", "markdown", "audit")),
        "confirmation report requires a complete successful bundle",
    )


def _publish_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    identity = canonical_body_identity(value)
    write_exclusive(path, value)
    same_json(stream_file_identity(path), identity, f"report published {path.name} bytes")
    return identity


def _markdown_identity(text: str) -> dict[str, Any]:
    encoded = text.encode("utf-8")
    return {"sha256": sha256(encoded).hexdigest(), "byte_count": len(encoded)}


def _publish_body(paths: dict[str, Path], body: dict[str, Any]) -> dict[str, Any]:
    result_identity = _publish_json(paths["result"], body)
    text = render_confirmation_report(body)
    markdown_identity = _markdown_identity(text)
    with paths["markdown"].open("xb") as output:
        output.write(text.encode("utf-8"))
    same_json(
        stream_file_identity(paths["markdown"]),
        markdown_identity,
        "report published Markdown bytes",
    )
    return {"result": result_identity, "markdown": markdown_identity}


def _audit(
    request: dict[str, Any],
    request_identity: dict[str, Any],
    files: dict[str, Any],
    body: dict[str, Any],
    elapsed: Any,
) -> dict[str, Any]:
    observed = elapsed_seconds(elapsed, "report derivative validation", VALIDATION_SECONDS)
    return {
        "schema_id": "p611_confirmation_report_audit_v1",
        "status": "completed",
        "protocol_id": request["protocol_id"],
        "request_identity": request_identity,
        "source_map_sha256": request["source_map_sha256"],
        "inputs": request["inputs"],
        "files": files,
        "coverage": body["coverage"],
        "analysis_repetition": body["analysis_repetition"],
        "cost_vector_identity": body["cost_vector_identity"],
        "derivative_validation_elapsed_seconds": observed,
        "validation_scope": "bound_complete_report_and_artifact_readback",
        "new_training_or_final_source_access": False,
    }


def _check_owned_bytes(paths: dict[str, Path], identities: dict[str, Any]) -> None:
    for name, identity in identities.items():
        same_json(stream_file_identity(paths[name]), identity, f"report late {name} bytes")


def _failure(paths: dict[str, Path], error: BaseException, started: float) -> None:
    write_exclusive(
        paths["failure"],
        {
            "schema_id": "p611_confirmation_report_failure_v1",
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
    root: Path,
    paths: dict[str, Path],
    scope_file: Path,
    scored_reader: ScoredReader,
    cost_reader: CostReader,
) -> dict[str, Any]:
    started = monotonic()
    request_owned = False
    audit_identity = None
    try:
        request = report_request(
            root, paths["request"].parent, scope_file, datetime.now(timezone.utc).isoformat()
        )
        request_identity = _publish_json(paths["request"], request)
        request_owned = True
        body = verified_report_body(root, paths, scope_file, scored_reader, cost_reader)
        same_json(
            checked_report_request(root, paths, scope_file),
            request,
            "report prepublication bindings",
        )
        elapsed_seconds(monotonic() - started, "report prepublication", VALIDATION_SECONDS)
        files = _publish_body(paths, body)
        audit = _audit(request, request_identity, files, body, monotonic() - started)
        audit_identity = _publish_json(paths["audit"], audit)
        same_json(checked_report_request(root, paths, scope_file), request, "report final bindings")
        _check_owned_bytes(paths, {"request": request_identity, **files, "audit": audit_identity})
        return audit
    except BaseException as error:
        if not request_owned and isinstance(error, FileExistsError):
            raise
        try:
            _failure(paths, error, started)
        except BaseException:
            # A failure must not leave our completion audit readable. Preserve
            # every result/request and any changed or foreign audit instead.
            if (
                audit_identity is not None
                and paths["audit"].is_file()
                and stream_file_identity(paths["audit"]) == audit_identity
            ):
                paths["audit"].unlink()
            raise
        raise


def publish_confirmation_report(
    root: Path,
    output_dir: Path,
    scope_file: Path,
    scored_reader: ScoredReader,
    cost_reader: CostReader,
) -> dict[str, Any]:
    paths = artifact_paths(output_dir.resolve())
    occupied = [str(path) for path in paths.values() if path.exists()]
    if occupied:
        raise FileExistsError(f"confirmation report output already exists: {occupied}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # The shared claim helper enforces exclusive ownership; its historical
    # marker text grants no training or experiment authority in this directory.
    with claim_artifacts(paths):
        return _publish_claimed(root, paths, scope_file, scored_reader, cost_reader)


def read_completed_confirmation_report(
    root: Path,
    output_dir: Path,
    scope_file: Path,
    scored_reader: ScoredReader,
    cost_reader: CostReader,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    started = monotonic()
    paths = artifact_paths(output_dir.resolve())
    _require_complete(paths)
    identities = {
        name: stream_file_identity(paths[name])
        for name in ("request", "result", "markdown", "audit")
    }
    request = checked_report_request(root, paths, scope_file)
    expected_body = verified_report_body(root, paths, scope_file, scored_reader, cost_reader)
    body, audit = read_json(paths["result"]), read_json(paths["audit"])
    same_json(body, expected_body, "report complete independently rebuilt body")
    same_json(canonical_body_identity(body), identities["result"], "report canonical result bytes")
    same_json(
        _markdown_identity(render_confirmation_report(expected_body)),
        identities["markdown"],
        "report complete rebuilt Markdown",
    )
    same_json(canonical_body_identity(audit), identities["audit"], "report canonical audit bytes")
    files = {name: identities[name] for name in ("result", "markdown")}
    same_json(
        audit,
        _audit(
            request,
            identities["request"],
            files,
            expected_body,
            audit.get("derivative_validation_elapsed_seconds"),
        ),
        "report complete rebuilt audit",
    )
    same_json(
        checked_report_request(root, paths, scope_file), request, "report final readback bindings"
    )
    _check_owned_bytes(paths, identities)
    _require_complete(paths)
    elapsed_seconds(monotonic() - started, "report independent readback", VALIDATION_SECONDS)
    return request, body, audit
