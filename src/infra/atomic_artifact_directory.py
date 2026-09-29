"""Publish complete local artifact directories from same-volume staging.

Inputs are already validated filename/byte pairs and one new destination.
Output is an atomically visible directory or a hidden staged directory
with an explicit incomplete/failed/canceled state. This does not validate
experiment facts, resume training, or remove failed user-visible work.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import tempfile


PUBLICATION_STATE = "publication-state.json"


def _write_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _write_stage_state(
    stage: Path, target: str, status: str, completed_files: list[str], error_type: str | None
) -> None:
    record = {
        "schema_id": "artifact_publication_v1",
        "target": target,
        "status": status,
        "completed_files": completed_files,
        "error_type": error_type,
    }
    payload = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with (stage / PUBLICATION_STATE).open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _validate_file_set(files: Mapping[str, bytes]) -> None:
    if not files:
        raise ValueError("artifact publication requires at least one file")
    for name, payload in files.items():
        if (
            type(name) is not str
            or not name
            or name.startswith(".")
            or Path(name).name != name
            or name == PUBLICATION_STATE
            or type(payload) is not bytes
        ):
            raise ValueError("artifact publication requires local byte-file entries")


def publish_artifact_directory(destination: str | Path, files: Mapping[str, bytes]) -> Path:
    """Write, check, then rename a complete directory into public view."""
    _validate_file_set(files)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"artifact directory already exists: {target}")
    lock = target.parent / f".{target.name}.publish.lock"
    acquired = False
    try:
        with lock.open("xb") as claim:
            acquired = True
            claim.write(b"artifact_publication_v1\n")
            claim.flush()
            os.fsync(claim.fileno())
    except BaseException:
        if acquired:
            lock.unlink(missing_ok=True)
        raise
    try:
        if target.exists():
            raise FileExistsError(f"artifact directory already exists: {target}")
        stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.pending.", dir=target.parent))
        completed: list[str] = []
        try:
            _write_stage_state(stage, target.name, "incomplete", completed, None)
            for name, payload in files.items():
                _write_file(stage / name, payload)
                completed.append(name)
                _write_stage_state(stage, target.name, "incomplete", completed, None)
            if any((stage / name).read_bytes() != payload for name, payload in files.items()):
                raise ValueError("staged artifact bytes differ before publication")
            (stage / PUBLICATION_STATE).unlink()
            if target.exists():
                raise FileExistsError(f"artifact directory already exists: {target}")
            # Why this: staging and target share a parent, so readers see the
            # complete directory only after the same-volume rename succeeds.
            os.rename(stage, target)
        except BaseException as exc:
            if stage.exists():
                status = (
                    "canceled" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
                )
                try:
                    _write_stage_state(stage, target.name, status, completed, type(exc).__name__)
                except OSError:
                    # The hidden pending path remains visible if the disk cannot
                    # record the more precise terminal state.
                    pass
            raise
    finally:
        lock.unlink(missing_ok=True)
    return target
