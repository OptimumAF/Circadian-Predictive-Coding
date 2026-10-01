"""Own exclusive scored artifacts and independent complete readback.

Inputs are local paths and a source-bound complete-reference reader port.
Outputs are verified request/result/audit bundles or explicit failures.
No algorithm, metric, source construction or resource measurement belongs here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
from time import monotonic
from typing import Any, Callable

from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_scoring_execution import (
    FAILURE_SCHEMA,
    elapsed_seconds,
    encoded_identity,
    scoring_audit,
    scoring_execution_request,
    verify_scored_worker,
)
from src.infra.continual_confirmation_io import (
    claim_artifacts,
    parse_json,
    read_json,
    write_exclusive,
)
from src.infra.continual_confirmation_scoring_bindings import (
    checked_scoring_request,
    scoring_bindings,
)
from src.infra.continual_confirmation_training_references import stream_file_identity


ReferenceReader = Callable[[], dict[str, Any]]


def artifact_paths(output_dir: Path) -> dict[str, Path]:
    paths = {
        name: output_dir / f"confirmation-scored.{name}.json"
        for name in ("request", "result", "audit", "failure")
    }
    paths["claim"] = output_dir / "confirmation-scored.claim"
    return paths


def _require_unoccupied(paths: dict[str, Path]) -> None:
    occupied = [str(path) for path in paths.values() if path.exists()]
    if occupied:
        raise FileExistsError(f"scored confirmation output already exists: {occupied}")


def _publish(path: Path, value: Any) -> dict[str, Any]:
    identity = encoded_identity(value)
    write_exclusive(path, value)
    same_json(stream_file_identity(path), identity, f"scored published {path.name} bytes")
    return identity


def _failure(path: Path, request_file: Path, exc: BaseException, started: float) -> None:
    write_exclusive(
        path,
        {
            "schema_id": FAILURE_SCHEMA,
            "status": "failed",
            "reason": "wall_limit"
            if isinstance(exc, subprocess.TimeoutExpired)
            else "canceled"
            if isinstance(exc, (KeyboardInterrupt, SystemExit))
            else "worker_or_audit",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "request_identity": stream_file_identity(request_file)
            if request_file.is_file()
            else None,
            "elapsed_seconds": monotonic() - started,
        },
    )


def run_bounded_scoring(
    root: Path,
    output_dir: Path,
    scope_file: Path,
    read_references: ReferenceReader,
) -> dict[str, Any]:
    """Read both complete train references before launching the fixed child."""
    paths = artifact_paths(output_dir.resolve())
    _require_unoccupied(paths)
    report = read_references()
    bindings = scoring_bindings(root, paths["request"], scope_file, report)
    request = scoring_execution_request(
        started_utc=datetime.now(timezone.utc).isoformat(), **bindings
    )
    paths["request"].parent.mkdir(parents=True, exist_ok=True)
    with claim_artifacts(paths):
        return _run_claimed(root, paths, scope_file, request)


def _run_claimed(
    root: Path,
    paths: dict[str, Path],
    scope_file: Path,
    request: dict[str, Any],
) -> dict[str, Any]:
    started = monotonic()
    published = False
    audit_identity: dict[str, Any] | None = None
    try:
        request_identity = encoded_identity(request)
        write_exclusive(paths["request"], request)
        published = True
        same_json(
            stream_file_identity(paths["request"]),
            request_identity,
            "scored published request bytes",
        )
        same_json(
            checked_scoring_request(root, paths["request"], scope_file, request_identity),
            request,
            "scored pre-worker request",
        )
        process = subprocess.run(
            request["command"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=request["limits"]["wall_limit_seconds"],
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"scored confirmation worker exited {process.returncode}: {process.stderr}"
            )
        payload = parse_json(process.stdout)
        verify_scored_worker(payload, request, request_identity["sha256"])
        same_json(
            checked_scoring_request(root, paths["request"], scope_file, request_identity),
            request,
            "scored post-worker request",
        )
        result_identity = _publish(paths["result"], payload["result"])
        audit = scoring_audit(
            request,
            request_identity["sha256"],
            result_identity["sha256"],
            payload,
            monotonic() - started,
        )
        audit_identity = _publish(paths["audit"], audit)
        same_json(
            checked_scoring_request(root, paths["request"], scope_file, request_identity),
            request,
            "scored post-publication request",
        )
        for name, identity in (
            ("request", request_identity),
            ("result", result_identity),
            ("audit", audit_identity),
        ):
            same_json(stream_file_identity(paths[name]), identity, f"scored final {name} bytes")
        same_json(encoded_identity(request), request_identity, "scored original request mutated")
    except BaseException as exc:
        if not published and isinstance(exc, FileExistsError):
            # A writer outside the cooperative claim still owns that request.
            raise
        try:
            _failure(paths["failure"], paths["request"], exc, started)
        except BaseException:
            # A completion audit must not survive a failure that could not be
            # recorded. Revoke only our verified, still-identical audit; retain
            # request/result evidence and any foreign or changed audit bytes.
            if (
                audit_identity is not None
                and paths["audit"].is_file()
                and stream_file_identity(paths["audit"]) == audit_identity
            ):
                paths["audit"].unlink()
            raise
        raise
    return audit


def _require_complete(paths: dict[str, Path]) -> None:
    require(
        not paths["failure"].exists()
        and not paths["claim"].exists()
        and all(paths[name].is_file() for name in ("request", "result", "audit")),
        "scored confirmation requires a complete successful bundle",
    )


def read_completed_scored_bundle(
    root: Path,
    output_dir: Path,
    scope_file: Path,
    read_references: ReferenceReader,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Require current complete references and every saved scientific/audit link."""
    paths = artifact_paths(output_dir.resolve())
    _require_complete(paths)
    identities = {
        name: stream_file_identity(paths[name]) for name in ("request", "result", "audit")
    }
    request = checked_scoring_request(root, paths["request"], scope_file, identities["request"])
    report = read_references()
    same_json(report, request["reference_report"], "scored complete training readback")
    result, audit = read_json(paths["result"]), read_json(paths["audit"])
    for name, body in (("request", request), ("result", result), ("audit", audit)):
        same_json(encoded_identity(body), identities[name], f"scored decoded {name} bytes")
    payload = {
        "result": result,
        "request_sha256": identities["request"]["sha256"],
        "reference_report_sha256": audit.get("reference_report_sha256"),
        "source_map_sha256": audit.get("source_map_sha256"),
        "observed_updates": audit.get("observed_updates"),
        "final_observation": audit.get("final_observation"),
        "process_rss": audit.get("process_rss"),
        "worker_elapsed_seconds": audit.get("worker_elapsed_seconds"),
    }
    expected = scoring_audit(
        request,
        identities["request"]["sha256"],
        identities["result"]["sha256"],
        payload,
        elapsed_seconds(audit.get("elapsed_seconds"), "parent"),
    )
    same_json(audit, expected, "complete scored audit")
    same_json(
        checked_scoring_request(root, paths["request"], scope_file, identities["request"]),
        request,
        "scored readback request",
    )
    for name, identity in identities.items():
        same_json(
            stream_file_identity(paths[name]), identity, f"scored {name} changed during readback"
        )
    _require_complete(paths)
    return request, result, audit
