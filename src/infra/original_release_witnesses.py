"""Read fixed original release evidence through unchanged complete readers.

Inputs are the actual repository root and canonical/repeat bundle name. Outputs
bind all original scoring parts, full reader validation and supported causal
records. Both original training references are reconstructed by their unchanged
complete readers. No publisher, child worker, source/model/RNG construction,
training, final release/prediction, seed selection or fresh admission belongs here.
Private ports support boundary fixtures and establish no original-reader proof.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from src.app.continual_confirmation_json import same_json
from src.app.continual_confirmation_scoring_manifest import fixed_scoring_manifest
from src.app.scored_release_witnesses import build_scored_release_witnesses
from src.core.seed_stream_screening import EvidenceIdentity, validate_evidence_identity
from src.infra.continual_confirmation_report_bindings import SCORED_FILES
from src.infra.continual_confirmation_scoring_artifacts import read_completed_scored_bundle
from src.infra.continual_confirmation_scoring_bindings import current_sources
from src.infra.continual_confirmation_training_references import (
    read_training_references,
    stream_file_identity,
)


CompleteScoredReader = Callable[[Path], tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]
SourceChecker = Callable[[], dict[str, str]]


@dataclass(frozen=True)
class ScoredReleaseReadback:
    chronology: dict[str, Any]
    file_identities: dict[str, dict[str, Any]]
    complete_original_reader_verified: bool = False


def _directory(root: Path, name: str) -> Path:
    if (
        type(name) is not str
        or "\\" in name
        or ":" in name
        or not name
        or name.startswith("/")
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ValueError("release witness directory must be a canonical root-relative path")
    base = root.resolve(strict=True)
    current = base
    for part in PurePosixPath(name).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("release witness directory cannot traverse symbolic links")
    try:
        directory = current.resolve(strict=True)
        directory.relative_to(base)
    except (OSError, ValueError) as error:
        raise ValueError("release witness directory is missing or outside its root") from error
    if not directory.is_dir():
        raise ValueError("release witness directory must exist")
    return directory


def _files(directory: Path, expected: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if type(expected) is not dict or set(expected) != {"request", "result", "audit"}:
        raise ValueError("release witness requires every complete artifact identity")
    if any(
        (directory / ("confirmation-scored." + name)).exists() for name in ("claim", "failure.json")
    ):
        raise ValueError("release witness requires a complete successful original bundle")
    files = {}
    for name, wanted in expected.items():
        if type(wanted) is not dict or set(wanted) != {"byte_count", "sha256"}:
            raise ValueError("release witness whole identity fields differ")
        validate_evidence_identity(EvidenceIdentity(**wanted))
        path = directory / f"confirmation-scored.{name}.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("release witness complete artifact is missing or symbolic")
        actual = stream_file_identity(path)
        same_json(actual, wanted, "release witness original whole file " + name)
        files[name] = actual
    return files


def _read_scored_witness(
    root: Path,
    directory_name: str,
    expected: dict[str, dict[str, Any]],
    read_complete: CompleteScoredReader,
    check_sources: SourceChecker,
) -> ScoredReleaseReadback:
    """Private whole-IO seam; reader spies cannot certify historical execution."""
    directory = _directory(root, directory_name)
    before = _files(directory, expected)
    sources = check_sources()
    parts = read_complete(directory)
    if type(parts) is not tuple or len(parts) != 3 or any(type(part) is not dict for part in parts):
        raise ValueError("release witness unchanged complete reader return differs")
    request, result, audit = parts
    chronology = build_scored_release_witnesses(request, result, audit)
    same_json(
        chronology["declared_decoded_file_identities"],
        before,
        "release witness complete decoded whole bytes",
    )
    same_json(sources, request["source_sha256"], "release witness source-bound original request")
    same_json(
        _files(directory, expected), before, "release witness files changed during full readback"
    )
    same_json(check_sources(), sources, "release witness source changed during full readback")
    return ScoredReleaseReadback(chronology, deepcopy(before))


def read_original_scoring_release(root: Path, bundle_name: str) -> ScoredReleaseReadback:
    """Read one fixed scoring bundle and both unchanged full training references."""
    if type(bundle_name) is not str or bundle_name not in {"canonical", "repeat"}:
        raise ValueError("release witness original bundle name must be canonical/repeat")
    from scripts import run_p67_confirmation_training as training_adapter

    actual_root = root.resolve(strict=True)
    if actual_root != training_adapter.REPO_ROOT.resolve(strict=True):
        raise ValueError("release witness requires the actual original reader root")
    scope_file = actual_root / "artifacts/runs/p67-confirmation-scope.json"
    directory, expected = SCORED_FILES[0 if bundle_name == "canonical" else 1]

    def original_reader(output: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        def references() -> dict[str, Any]:
            return read_training_references(
                actual_root,
                fixed_scoring_manifest(),
                lambda training_dir: training_adapter.read_completed_bundle(
                    training_dir, scope_file
                ),
            )

        return read_completed_scored_bundle(actual_root, output, scope_file, references)

    result = _read_scored_witness(
        actual_root, directory, expected, original_reader, lambda: current_sources(actual_root)
    )
    # Why this: this flag can only follow the fixed public adapter; the private
    # IO/reader-spy seam and pure decoded consumer leave it false. Fresh roles
    # and complete prior-usage/resource acceptance remain separate and false.
    return replace(result, complete_original_reader_verified=True)
