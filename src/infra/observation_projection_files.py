"""Persist and verify a derived observation directory beside a v14 bundle.

The P5.1 completed-run verifier remains the source gate. This adapter reads
its raw files, writes one exclusive projection, and checks exact derivation;
it does not train, evaluate, select results, or repair an interrupted write.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from src.app.v14_observation_projection import (
    OBSERVATION_PROJECTION_ID,
    ObservationProjection,
    build_v14_observation_projection,
)
from src.infra.versioned_run_files import verify_run_bundle
from src.infra.atomic_artifact_directory import publish_artifact_directory


PROJECTION_DIRECTORY = "observations-v1"
PROJECTION_MANIFEST = "projection-manifest.json"


def _read_source(run_path: Path) -> tuple[dict[str, Any], ObservationProjection]:
    manifest = verify_run_bundle(run_path)
    training = json.loads((run_path / manifest["files"]["training"]["path"]).read_bytes())
    outcomes = json.loads((run_path / manifest["files"]["outcomes"]["path"]).read_bytes())
    return manifest, build_v14_observation_projection(training, outcomes)


def _expected_manifest(
    run_path: Path, source: dict[str, Any], projection: ObservationProjection
) -> dict[str, Any]:
    return {
        "projection_id": OBSERVATION_PROJECTION_ID,
        "source_run_id": source["run_id"],
        "source_manifest_sha256": sha256((run_path / "manifest.json").read_bytes()).hexdigest(),
        "source_files": {
            name: source["files"][name]["sha256"] for name in ("training", "outcomes")
        },
        "files": {
            name: {"sha256": sha256(payload).hexdigest(), "record_count": projection.counts[name]}
            for name, payload in projection.files.items()
        },
    }


def write_observation_projection(run_path: str | Path) -> Path:
    """Project one completed source run into an exclusive local directory."""
    run_directory = Path(run_path)
    destination = run_directory / PROJECTION_DIRECTORY
    if destination.exists():
        raise FileExistsError(f"observation projection already exists: {destination}")
    source, projection = _read_source(run_directory)
    metadata = _expected_manifest(run_directory, source, projection)
    return publish_artifact_directory(
        destination,
        {
            **projection.files,
            PROJECTION_MANIFEST: (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        },
    )


def verify_observation_projection(run_path: str | Path) -> dict[str, Any]:
    """Reject a changed source, missing stream, or rehashed projection forgery."""
    run_directory = Path(run_path)
    source, projection = _read_source(run_directory)
    expected = _expected_manifest(run_directory, source, projection)
    directory = run_directory / PROJECTION_DIRECTORY
    try:
        saved = json.loads((directory / PROJECTION_MANIFEST).read_bytes())
        if saved != expected:
            raise ValueError("observation projection manifest differs from verified source")
        if {entry.name for entry in directory.iterdir()} != set(projection.files) | {
            PROJECTION_MANIFEST
        }:
            raise ValueError("observation projection file set differs")
        for name, payload in projection.files.items():
            if (directory / name).read_bytes() != payload:
                raise ValueError(f"observation projection content differs: {name}")
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("observation projection is missing or malformed") from exc
    return expected
