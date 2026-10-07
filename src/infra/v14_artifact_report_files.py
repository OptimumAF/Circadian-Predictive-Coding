"""Publish and verify a table derived only from a completed v14 bundle.

Inputs are a local P5.1 run directory. Outputs are exclusive report files
or verified metadata. This boundary does not train, score, select an arm,
edit the source bundle, or infer unpublished failures.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from src.app.v14_artifact_report import (
    REPORT_SCHEMA_ID,
    V14ArtifactReport,
    build_v14_artifact_report,
)
from src.infra.atomic_artifact_directory import publish_artifact_directory
from src.infra.versioned_run_files import verify_run_bundle


REPORT_DIRECTORY = "summary-report-v1"
REPORT_MANIFEST = "report-manifest.json"


def _reject_nonfinite(token: str) -> object:
    raise ValueError(f"nonfinite outcome token is invalid: {token}")


def _read_verified_source(run_path: Path) -> tuple[dict[str, Any], bytes, V14ArtifactReport]:
    manifest = verify_run_bundle(run_path)
    try:
        manifest_bytes = (run_path / "manifest.json").read_bytes()
        outcomes_bytes = (run_path / manifest["files"]["outcomes"]["path"]).read_bytes()
        if json.loads(manifest_bytes) != manifest:
            raise ValueError("v14 report source manifest changed after verification")
        if sha256(outcomes_bytes).hexdigest() != manifest["files"]["outcomes"]["sha256"]:
            raise ValueError("v14 report outcomes SHA-256 changed after verification")
        outcomes = json.loads(outcomes_bytes, parse_constant=_reject_nonfinite)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("v14 report source is missing or malformed") from exc
    return manifest, manifest_bytes, build_v14_artifact_report(manifest, outcomes)


def _expected_metadata(
    manifest: dict[str, Any], manifest_bytes: bytes, report: V14ArtifactReport
) -> dict[str, Any]:
    files = {"summary.json": report.json_bytes, "summary.csv": report.csv_bytes}
    return {
        "report_id": REPORT_SCHEMA_ID,
        "source_run_id": manifest["run_id"],
        "source_manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "source_files": {
            name: manifest["files"][name]["sha256"] for name in ("training", "outcomes")
        },
        "files": {name: {"sha256": sha256(payload).hexdigest()} for name, payload in files.items()},
    }


def write_v14_artifact_report(run_path: str | Path) -> Path:
    """Write one exclusive report only after the source bundle verifies."""
    run_directory = Path(run_path)
    destination = run_directory / REPORT_DIRECTORY
    if destination.exists():
        raise FileExistsError(f"v14 report already exists: {destination}")
    manifest, manifest_bytes, report = _read_verified_source(run_directory)
    metadata = _expected_metadata(manifest, manifest_bytes, report)
    return publish_artifact_directory(
        destination,
        {
            "summary.json": report.json_bytes,
            "summary.csv": report.csv_bytes,
            REPORT_MANIFEST: (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(),
        },
    )


def verify_v14_artifact_report(run_path: str | Path) -> dict[str, Any]:
    """Re-derive exact bytes; reject changed source or hand-edited tables."""
    run_directory = Path(run_path)
    manifest, manifest_bytes, report = _read_verified_source(run_directory)
    expected = _expected_metadata(manifest, manifest_bytes, report)
    directory = run_directory / REPORT_DIRECTORY
    try:
        saved = json.loads((directory / REPORT_MANIFEST).read_bytes())
        if saved != expected:
            raise ValueError("v14 report manifest differs from verified source")
        if {entry.name for entry in directory.iterdir()} != {
            "summary.json",
            "summary.csv",
            REPORT_MANIFEST,
        }:
            raise ValueError("v14 report file set differs")
        if (directory / "summary.json").read_bytes() != report.json_bytes:
            raise ValueError("v14 report JSON differs from verified source")
        if (directory / "summary.csv").read_bytes() != report.csv_bytes:
            raise ValueError("v14 report CSV differs from verified source")
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("v14 report is missing or malformed") from exc
    return expected
