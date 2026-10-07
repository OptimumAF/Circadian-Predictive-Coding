"""Publish and re-verify static dashboard bytes from a checked v14 report.

Inputs are a local completed run with a verified P5.6a summary. Outputs are
exclusive HTML/PNG files or verified metadata. This boundary does not train,
score, rank cells, edit source files, or alter the historical dashboard.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from src.app.v14_dashboard_projection import (
    DASHBOARD_ID,
    V14DashboardProjection,
    build_v14_dashboard,
)
from src.infra.atomic_artifact_directory import publish_artifact_directory
from src.infra.v14_artifact_report_files import (
    REPORT_DIRECTORY,
    REPORT_MANIFEST,
    verify_v14_artifact_report,
)


DASHBOARD_DIRECTORY = "dashboard-v1"
DASHBOARD_MANIFEST = "dashboard-manifest.json"


def _reject_nonfinite(token: str) -> object:
    raise ValueError(f"nonfinite dashboard source token is invalid: {token}")


def _read_verified_report(run_path: Path) -> tuple[dict[str, Any], bytes, V14DashboardProjection]:
    metadata = verify_v14_artifact_report(run_path)
    directory = run_path / REPORT_DIRECTORY
    try:
        manifest_bytes = (directory / REPORT_MANIFEST).read_bytes()
        summary_bytes = (directory / "summary.json").read_bytes()
        csv_bytes = (directory / "summary.csv").read_bytes()
        if json.loads(manifest_bytes) != metadata:
            raise ValueError("v14 dashboard report manifest changed after verification")
        for name, payload in (("summary.json", summary_bytes), ("summary.csv", csv_bytes)):
            if sha256(payload).hexdigest() != metadata["files"][name]["sha256"]:
                raise ValueError(f"v14 dashboard source {name} SHA-256 changed after verification")
        summary = json.loads(summary_bytes, parse_constant=_reject_nonfinite)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("v14 dashboard report is missing or malformed") from exc
    return metadata, manifest_bytes, build_v14_dashboard(summary)


def _expected_metadata(
    report_metadata: dict[str, Any],
    report_manifest_bytes: bytes,
    projection: V14DashboardProjection,
) -> dict[str, Any]:
    return {
        "dashboard_id": DASHBOARD_ID,
        "source_run_id": report_metadata["source_run_id"],
        "source_report_id": report_metadata["report_id"],
        "source_report_manifest_sha256": sha256(report_manifest_bytes).hexdigest(),
        "source_files": {
            name: report_metadata["files"][name]["sha256"]
            for name in ("summary.json", "summary.csv")
        },
        "files": {
            name: {"sha256": sha256(payload).hexdigest()}
            for name, payload in projection.files.items()
        },
    }


def write_v14_dashboard(run_path: str | Path) -> Path:
    """Create one exclusive derived directory after full report verification."""
    run_directory = Path(run_path)
    destination = run_directory / DASHBOARD_DIRECTORY
    if destination.exists():
        raise FileExistsError(f"v14 dashboard already exists: {destination}")
    metadata, manifest_bytes, projection = _read_verified_report(run_directory)
    dashboard_metadata = _expected_metadata(metadata, manifest_bytes, projection)
    return publish_artifact_directory(
        destination,
        {
            **projection.files,
            DASHBOARD_MANIFEST: (
                json.dumps(dashboard_metadata, indent=2, sort_keys=True) + "\n"
            ).encode(),
        },
    )


def verify_v14_dashboard(run_path: str | Path) -> dict[str, Any]:
    """Reject changed report, chart, HTML, or dashboard manifest bytes."""
    run_directory = Path(run_path)
    metadata, manifest_bytes, projection = _read_verified_report(run_directory)
    expected = _expected_metadata(metadata, manifest_bytes, projection)
    directory = run_directory / DASHBOARD_DIRECTORY
    try:
        saved = json.loads((directory / DASHBOARD_MANIFEST).read_bytes())
        if saved != expected:
            raise ValueError("v14 dashboard manifest differs from verified report")
        if {entry.name for entry in directory.iterdir()} != set(projection.files) | {
            DASHBOARD_MANIFEST
        }:
            raise ValueError("v14 dashboard file set differs")
        for name, payload in projection.files.items():
            if (directory / name).read_bytes() != payload:
                raise ValueError(f"v14 dashboard content differs: {name}")
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("v14 dashboard is missing or malformed") from exc
    return expected
