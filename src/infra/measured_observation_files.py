"""Bind opt-in wake measurements and their projection to a completed run.

Inputs are one P5.1 bundle and an in-memory train-only diagnostic stream.
Outputs are exclusive local sidecar/projection directories with manifests
written last. This boundary does not train, score, or recover partial IO.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from src.app.v14_measured_observations import (
    MEASURED_PROJECTION_ID,
    WAKE_DIAGNOSTIC_ID,
    build_measured_observation_projection,
    validate_wake_diagnostic_bytes,
)
from src.app.v14_observation_projection import (
    ObservationProjection,
    build_v14_observation_projection,
)
from src.infra.atomic_artifact_directory import publish_artifact_directory
from src.infra.versioned_run_files import verify_run_bundle


MEASUREMENT_DIRECTORY = "measurements-v1"
DIAGNOSTIC_FILE = "wake-diagnostics.jsonl"
MEASUREMENT_MANIFEST = "measurement-manifest.json"
MEASURED_PROJECTION_DIRECTORY = "observations-measured-v1"
PROJECTION_MANIFEST = "projection-manifest.json"


def _read_completed_source(
    run_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = verify_run_bundle(run_path)
    training = json.loads((run_path / manifest["files"]["training"]["path"]).read_bytes())
    outcomes = json.loads((run_path / manifest["files"]["outcomes"]["path"]).read_bytes())
    build_v14_observation_projection(training, outcomes)
    return manifest, training, outcomes


def _source_identity(run_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_run_id": manifest["run_id"],
        "source_manifest_sha256": sha256((run_path / "manifest.json").read_bytes()).hexdigest(),
        "source_files": {
            name: manifest["files"][name]["sha256"] for name in ("training", "outcomes")
        },
    }


def _manifest_bytes(record: dict[str, Any]) -> bytes:
    return (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _measurement_metadata(
    run_path: Path, manifest: dict[str, Any], payload: bytes, count: int
) -> dict[str, Any]:
    return {
        "observation_id": WAKE_DIAGNOSTIC_ID,
        **_source_identity(run_path, manifest),
        "file": {
            "path": DIAGNOSTIC_FILE,
            "sha256": sha256(payload).hexdigest(),
            "record_count": count,
        },
    }


def write_wake_diagnostic_sidecar(run_path: str | Path, payload: bytes) -> Path:
    """Publish diagnostics only beside a completed and verified scored run."""
    directory = Path(run_path)
    destination = directory / MEASUREMENT_DIRECTORY
    if destination.exists():
        raise FileExistsError(f"wake diagnostic sidecar already exists: {destination}")
    source, training, _ = _read_completed_source(directory)
    records = validate_wake_diagnostic_bytes(training, payload)
    metadata = _measurement_metadata(directory, source, payload, len(records))
    return publish_artifact_directory(
        destination,
        {DIAGNOSTIC_FILE: payload, MEASUREMENT_MANIFEST: _manifest_bytes(metadata)},
    )


def verify_wake_diagnostic_sidecar(run_path: str | Path) -> dict[str, Any]:
    """Verify completed source identity and all saved diagnostic rows."""
    directory = Path(run_path)
    source, training, _ = _read_completed_source(directory)
    sidecar = directory / MEASUREMENT_DIRECTORY
    try:
        payload = (sidecar / DIAGNOSTIC_FILE).read_bytes()
        saved = json.loads((sidecar / MEASUREMENT_MANIFEST).read_bytes())
        records = validate_wake_diagnostic_bytes(training, payload)
        expected = _measurement_metadata(directory, source, payload, len(records))
        if saved != expected or {entry.name for entry in sidecar.iterdir()} != {
            DIAGNOSTIC_FILE,
            MEASUREMENT_MANIFEST,
        }:
            raise ValueError("wake diagnostic sidecar manifest or file set differs")
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("wake diagnostic sidecar is missing or malformed") from exc
    return expected


def _measured_projection(
    run_path: Path,
) -> tuple[dict[str, Any], ObservationProjection]:
    source, training, outcomes = _read_completed_source(run_path)
    verify_wake_diagnostic_sidecar(run_path)
    payload = (run_path / MEASUREMENT_DIRECTORY / DIAGNOSTIC_FILE).read_bytes()
    return source, build_measured_observation_projection(training, outcomes, payload)


def _projection_metadata(
    run_path: Path, source: dict[str, Any], projection: ObservationProjection
) -> dict[str, Any]:
    return {
        "projection_id": MEASURED_PROJECTION_ID,
        **_source_identity(run_path, source),
        "measurement_manifest_sha256": sha256(
            (run_path / MEASUREMENT_DIRECTORY / MEASUREMENT_MANIFEST).read_bytes()
        ).hexdigest(),
        "files": {
            name: {"sha256": sha256(payload).hexdigest(), "record_count": projection.counts[name]}
            for name, payload in projection.files.items()
        },
    }


def write_measured_observation_projection(run_path: str | Path) -> Path:
    """Derive measured streams after source and sidecar verification."""
    directory = Path(run_path)
    destination = directory / MEASURED_PROJECTION_DIRECTORY
    if destination.exists():
        raise FileExistsError(f"measured observation projection already exists: {destination}")
    source, projection = _measured_projection(directory)
    metadata = _projection_metadata(directory, source, projection)
    return publish_artifact_directory(
        destination, {**projection.files, PROJECTION_MANIFEST: _manifest_bytes(metadata)}
    )


def verify_measured_observation_projection(run_path: str | Path) -> dict[str, Any]:
    """Re-derive every measured and original stream from the verified source."""
    directory = Path(run_path)
    source, projection = _measured_projection(directory)
    expected = _projection_metadata(directory, source, projection)
    destination = directory / MEASURED_PROJECTION_DIRECTORY
    try:
        saved = json.loads((destination / PROJECTION_MANIFEST).read_bytes())
        if saved != expected or {entry.name for entry in destination.iterdir()} != (
            set(projection.files) | {PROJECTION_MANIFEST}
        ):
            raise ValueError("measured observation projection manifest or file set differs")
        for name, payload in projection.files.items():
            if (destination / name).read_bytes() != payload:
                raise ValueError(f"measured observation projection content differs: {name}")
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("measured observation projection is missing or malformed") from exc
    return expected
