"""Bind the whole scored request, observed execution and complete audit links.

Inputs are the fixed scoring manifest and boundary-supplied bytes/metadata.
Outputs are closed request/worker/audit declarations. No IO, source/model
construction, live state, process measurement or scientific selection belongs here.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite
from typing import Any

from src.app.continual_confirmation_execution import (
    digest_json,
    json_value,
    verify_process_memory,
)
from src.app.continual_confirmation_final_observation import verify_final_observation
from src.app.continual_confirmation_json import hash_value, object_fields, require, same_json
from src.app.continual_confirmation_scoring_manifest import (
    ConfirmationScoringManifest,
    scoring_manifest_digest,
    scoring_summary,
    validate_scoring_manifest,
)


REQUEST_SCHEMA = "p67_confirmation_scored_request_v1"
AUDIT_SCHEMA = "p67_confirmation_scored_audit_v1"
FAILURE_SCHEMA = "p67_confirmation_scored_failure_v1"
SOURCE_FILE_COUNT = 97
# Why this: this exact metadata was independently produced by both complete
# original readers before any reserved final value. Reconstructing it through
# those readers remains mandatory in the source-bound parent, not the child.
REFERENCE_REPORT_SHA256 = "cc1c1deb4c721af5d8250f17783c9daada2501c108be91f825fa37324626b001"
REFERENCE_REPORT_BYTES = 72050


def encoded_identity(value: Any) -> dict[str, Any]:
    """Match the existing exclusive pretty-JSON writer, including its LF."""
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    return {"sha256": sha256(encoded).hexdigest(), "byte_count": len(encoded)}


def verify_reference_report(report: Any) -> None:
    require(type(report) is dict, "scoring complete reference report must be an object")
    same_json(report, json_value(report), "scoring reference JSON types")
    same_json(
        encoded_identity(report),
        {"sha256": REFERENCE_REPORT_SHA256, "byte_count": REFERENCE_REPORT_BYTES},
        "scoring complete reference readback identity",
    )


def elapsed_seconds(value: Any, name: str, maximum: float | None = None) -> float:
    require(
        type(value) in (int, float) and isfinite(value) and value >= 0,
        f"scoring {name} elapsed differs",
    )
    if maximum is not None:
        require(value < maximum, f"scoring {name} wall limit exceeded")
    return float(value)


def _require_time(started_utc: str) -> None:
    require(type(started_utc) is str, "scoring request time type differs")
    try:
        timestamp = datetime.fromisoformat(started_utc)
    except ValueError as error:
        raise ValueError("scoring request time is malformed") from error
    require(
        timestamp.tzinfo is not None and timestamp.utcoffset() == timezone.utc.utcoffset(timestamp),
        "scoring request time must be UTC",
    )


def _require_sources(source_sha256: Any) -> None:
    require(
        type(source_sha256) is dict and len(source_sha256) == SOURCE_FILE_COUNT,
        "scoring complete source inventory differs",
    )
    for name, digest in source_sha256.items():
        require(
            type(name) is str
            and name.startswith(("src/", "scripts/"))
            and name.endswith(".py")
            and "\\" not in name
            and all(part not in {"", ".", ".."} for part in name.split("/")),
            "scoring source name differs",
        )
        hash_value(digest, f"scoring source {name}")


def scoring_execution_request(
    manifest: ConfirmationScoringManifest,
    reference_report: dict[str, Any],
    source_sha256: dict[str, str],
    command: list[str],
    environment: dict[str, str],
    started_utc: str,
) -> dict[str, Any]:
    """Resolve only the full fixed scientific declaration; IO proof is external."""
    validate_scoring_manifest(manifest)
    verify_reference_report(reference_report)
    _require_sources(source_sha256)
    require(type(command) is list and bool(command), "scoring worker command empty")
    require(all(type(part) is str and bool(part) for part in command), "scoring command differs")
    object_fields(
        environment,
        {"python_version", "numpy_version", "platform", "processor"},
        "scoring environment",
    )
    require(
        all(type(value) is str for value in environment.values()),
        "scoring environment types differ",
    )
    _require_time(started_utc)
    train = manifest.train_manifest
    return {
        "schema_id": REQUEST_SCHEMA,
        "protocol_id": manifest.protocol_id,
        "manifest": json_value(asdict(manifest)),
        "manifest_sha256": scoring_manifest_digest(manifest),
        "analysis_contract_sha256": manifest.analysis_contract_sha256,
        "reference_report": json_value(reference_report),
        "reference_report_sha256": REFERENCE_REPORT_SHA256,
        "source_sha256": dict(source_sha256),
        "source_map_sha256": digest_json(source_sha256),
        "command": list(command),
        "environment": dict(environment),
        "started_utc": started_utc,
        "summary": scoring_summary(manifest),
        "limits": {
            "max_optimizer_updates": train.max_optimizer_updates,
            "wall_limit_seconds": train.wall_limit_seconds,
            "max_process_rss_bytes": train.max_process_rss_bytes,
            "rss_interval_seconds": train.rss_interval_seconds,
        },
        "outer_selection_scored": False,
        "complete_independent_final_required": True,
    }


def verify_scoring_execution_request(payload: Any, **bindings: Any) -> None:
    require(type(payload) is dict, "scoring request must be an object")
    expected = scoring_execution_request(started_utc=payload.get("started_utc"), **bindings)
    same_json(payload, expected, "complete scoring execution request")


def verify_scored_worker(
    payload: Any, request: dict[str, Any], request_sha256: str
) -> dict[str, Any]:
    """Validate every scientific JSON/observation/resource link independently."""
    manifest = _request_manifest(request)
    object_fields(
        payload,
        {
            "result",
            "request_sha256",
            "reference_report_sha256",
            "source_map_sha256",
            "observed_updates",
            "final_observation",
            "process_rss",
            "worker_elapsed_seconds",
        },
        "scored worker envelope",
    )
    hash_value(request_sha256, "scoring request identity")
    same_json(payload["request_sha256"], request_sha256, "scored worker request identity")
    for field in ("reference_report_sha256", "source_map_sha256"):
        same_json(payload[field], request[field], f"scored worker {field}")
    verify_final_observation(payload["final_observation"], payload["result"], manifest)
    verify_scored_updates(payload["observed_updates"], request)
    verify_process_memory(payload["process_rss"], manifest.train_manifest)
    elapsed_seconds(
        payload["worker_elapsed_seconds"], "worker", manifest.train_manifest.wall_limit_seconds
    )
    return json_value(request["reference_report"]["bundles"][0]["work"])


def _request_manifest(request: dict[str, Any]) -> ConfirmationScoringManifest:
    from src.app.continual_confirmation_scoring_manifest import fixed_scoring_manifest

    manifest = fixed_scoring_manifest()
    verify_scoring_execution_request(
        request,
        manifest=manifest,
        reference_report=request.get("reference_report"),
        source_sha256=request.get("source_sha256"),
        command=request.get("command"),
        environment=request.get("environment"),
    )
    return manifest


def verify_scored_updates(observed: Any, request: dict[str, Any]) -> None:
    verify_reference_report(request["reference_report"])
    same_json(
        observed,
        request["reference_report"]["bundles"][0]["observed_updates"],
        "scored actual optimizer/complete reference work",
    )


def scoring_audit(
    request: dict[str, Any],
    request_sha256: str,
    result_sha256: str,
    payload: dict[str, Any],
    parent_elapsed: float,
) -> dict[str, Any]:
    """Construct the closed expected audit only after complete worker validation."""
    work = verify_scored_worker(payload, request, request_sha256)
    hash_value(result_sha256, "scoring result bytes")
    elapsed_seconds(parent_elapsed, "parent")
    return json_value(
        {
            "schema_id": AUDIT_SCHEMA,
            "status": "completed",
            "protocol_id": request["protocol_id"],
            "manifest_sha256": request["manifest_sha256"],
            "analysis_contract_sha256": request["analysis_contract_sha256"],
            "reference_report_sha256": request["reference_report_sha256"],
            "source_sha256": request["source_sha256"],
            "source_map_sha256": request["source_map_sha256"],
            "request_sha256": request_sha256,
            "result_sha256": result_sha256,
            "work": work,
            "observed_updates": payload["observed_updates"],
            "final_observation": payload["final_observation"],
            "process_rss": payload["process_rss"],
            "worker_elapsed_seconds": payload["worker_elapsed_seconds"],
            "elapsed_seconds": parent_elapsed,
            "validation_scope": "bound_child_execution_and_complete_artifact_links",
            "outer_selection_scored": False,
        }
    )
