"""Read all five declared whole saved bodies before producing pending metadata.

Inputs are root-relative paths and whole identities for every fixed input kind.
Outputs carry the complete pending contract after strict JSON and before/after
whole byte checks. Actual current source/history proof, historical execution,
scientific construction and fresh-role admission are outside this boundary.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from src.app.prospective_confirmation_evidence import build_pending_confirmation_contract
from src.core.seed_stream_screening import EvidenceIdentity, validate_evidence_identity
from src.core.seed_usage import strict_seed_metadata_json


INPUT_NAMES = frozenset(
    ("chronology", "precision", "precision_repeat", "canonical_witness", "repeat_witness")
)


def _path(root: Path, name: str) -> Path:
    if (
        type(name) is not str
        or not name
        or "\\" in name
        or ":" in name
        or "\0" in name
        or name.startswith("/")
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        raise ValueError("prospective input path must be canonical and root-relative")
    path = root / name
    if not path.resolve(strict=True).is_relative_to(root) or any(
        parent.is_symlink() for parent in (path, *path.parents) if parent.is_relative_to(root)
    ):
        raise ValueError("prospective input path must stay inside its root without symlinks")
    if not path.is_file():
        raise ValueError("prospective input must be a complete regular file")
    return path


def _read(path: Path, expected: EvidenceIdentity) -> bytes:
    raw = path.read_bytes()
    if EvidenceIdentity(len(raw), sha256(raw).hexdigest()) != expected:
        raise ValueError(f"prospective whole saved input drift: {path.name}")
    return raw


def load_pending_confirmation_contract(
    root: Path, paths: dict[str, str], identities: dict[str, EvidenceIdentity]
) -> dict[str, Any]:
    """Bind every complete file; callers must separately prove source provenance."""
    if (
        type(paths) is not dict
        or set(paths) != INPUT_NAMES
        or type(identities) is not dict
        or set(identities) != INPUT_NAMES
    ):
        raise ValueError("prospective boundary requires every one of the five input kinds")
    for identity in identities.values():
        validate_evidence_identity(identity)
    root = root.resolve(strict=True)
    files = {name: _path(root, path) for name, path in paths.items()}
    if len(set(files.values())) != len(files):
        raise ValueError("prospective inputs require five distinct physical paths")
    for path in files.values():
        if any(
            marker.exists()
            for marker in (path.with_suffix(".claim"), path.with_suffix(".failure.json"))
        ):
            raise ValueError("prospective input has a pending or failed producer marker")
    inputs = {
        name: strict_seed_metadata_json(_read(path, identities[name]))
        for name, path in files.items()
    }
    if any(type(body) is not dict for body in inputs.values()):
        raise ValueError("prospective inputs must be entire JSON objects")
    result = build_pending_confirmation_contract(
        inputs, {name: asdict(value) for name, value in identities.items()}
    )
    for name, path in files.items():
        _read(path, identities[name])
        if _path(root, paths[name]) != path:
            raise ValueError("prospective input path changed after complete projection")
        if any(
            marker.exists()
            for marker in (path.with_suffix(".claim"), path.with_suffix(".failure.json"))
        ):
            raise ValueError("prospective input producer became pending or failed")
    return result
