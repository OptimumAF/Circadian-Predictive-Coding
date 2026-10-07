"""Read whole saved evidence/source bytes without rerunning the semantic reader.

Inputs are explicit root-relative file names and exact whole identities.
Outputs carry the complete decoded report after before/after identity and
full frozen physical/Git membership checks. This owns filesystem/JSON IO;
it never constructs a source, unpickles, executes expressions or grants roles.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from src.app.prior_seed_corpus import validated_seed_contents
from src.core.seed_stream_screening import EvidenceIdentity, validate_evidence_identity
from src.core.seed_usage import strict_seed_metadata_json


@dataclass(frozen=True)
class SavedSeedEvidence:
    report: dict[str, Any]
    evidence_identity: EvidenceIdentity
    source_identity: EvidenceIdentity


def _path(root: Path, name: str) -> Path:
    if (
        type(name) is not str
        or not name
        or "\\" in name
        or "\0" in name
        or Path(name).is_absolute()
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        raise ValueError("saved seed input path must be explicitly root-relative")
    path = root / name
    if not path.resolve(strict=True).is_relative_to(root) or any(
        parent.is_symlink() for parent in (path, *path.parents) if parent.is_relative_to(root)
    ):
        raise ValueError("saved seed input path must stay inside its root without symlinks")
    return path


def _identity(path: Path) -> EvidenceIdentity:
    digest, count = sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            count += len(chunk)
            digest.update(chunk)
    return EvidenceIdentity(count, digest.hexdigest())


def _read(path: Path, expected: EvidenceIdentity) -> dict[str, Any]:
    raw = path.read_bytes()
    if EvidenceIdentity(len(raw), sha256(raw).hexdigest()) != expected:
        raise ValueError(f"saved seed whole bytes changed before decoding: {path.name}")
    try:
        value = strict_seed_metadata_json(raw)
    except ValueError as error:
        raise ValueError(f"saved seed JSON cannot be decoded: {path.name}") from error
    if type(value) is not dict:
        raise ValueError("saved seed evidence/source must be complete objects")
    return value


def load_saved_seed_evidence(
    root: Path,
    *,
    evidence_path: str,
    evidence_identity: EvidenceIdentity,
    source_path: str,
    source_identity: EvidenceIdentity,
) -> SavedSeedEvidence:
    """Validate both complete saved bodies and all original membership aliases."""
    validate_evidence_identity(evidence_identity)
    validate_evidence_identity(source_identity)
    root = root.resolve(strict=True)
    evidence_file, source_file = _path(root, evidence_path), _path(root, source_path)
    if evidence_file == source_file:
        raise ValueError("saved seed evidence and frozen source must be distinct files")
    source = _read(source_file, source_identity)
    report = _read(evidence_file, evidence_identity)
    if source.get("schema_id") != "p67_complete_prior_seed_evidence_source_v1":
        raise ValueError("saved seed frozen source schema differs")
    if any(name not in source or report.get(name) != source[name] for name in ("files", "history")):
        raise ValueError("saved seed evidence differs from the whole frozen source membership")
    validated_seed_contents(report)
    if _identity(evidence_file) != evidence_identity or _identity(source_file) != source_identity:
        raise ValueError("saved seed whole bytes changed after complete decoding")
    return SavedSeedEvidence(report, evidence_identity, source_identity)
