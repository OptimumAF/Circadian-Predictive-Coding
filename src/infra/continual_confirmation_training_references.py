"""Read both pinned complete training bundles without retaining their bodies.

Inputs are a repo root, the fixed scoring manifest and a complete-reader port.
Outputs retain only byte identities, checked declarations, costs and historical
resource metadata. The adapter supplies the unchanged full training reader.
No source/model/train/final access or new scoring/resource authority belongs here.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable

from src.app.continual_confirmation_analysis_contract import analysis_contract_digest
from src.app.continual_confirmation_execution import digest_json, json_value
from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_scoring_manifest import (
    ConfirmationScoringManifest,
    TrainingBundleReference,
    scoring_manifest_digest,
    validate_scoring_manifest,
)


CompletedTrainingReader = Callable[[Path], tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]
_ARTIFACTS = ("request", "result", "audit")


def stream_file_identity(path: Path) -> dict[str, Any]:
    """Hash actual bytes in bounded chunks, including the observed byte count."""
    digest = sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "byte_count": size}


def _decoded_identity(value: dict[str, Any]) -> dict[str, Any]:
    encoder = json.JSONEncoder(indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
    digest = sha256()
    size = 0
    try:
        for fragment in encoder.iterencode(value):
            encoded = fragment.encode("utf-8")
            digest.update(encoded)
            size += len(encoded)
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError(
            "scoring training reference decoded body is malformed/nonfinite"
        ) from error
    digest.update(b"\n")
    return {"sha256": digest.hexdigest(), "byte_count": size + 1}


def _bundle_bytes(root: Path, reference: TrainingBundleReference) -> dict[str, Any]:
    directory = root / reference.directory
    require(
        not (directory / "confirmation-train.failure.json").exists()
        and not (directory / "confirmation-train.claim").exists(),
        f"scoring training reference {reference.name} has failure/claim",
    )
    result = {}
    for name in _ARTIFACTS:
        path = directory / f"confirmation-train.{name}.json"
        require(path.is_file(), f"scoring training reference missing {path}")
        identity = stream_file_identity(path)
        require(
            identity["sha256"] == getattr(reference, f"{name}_sha256"),
            f"scoring training reference {reference.name}/{name} bytes differ",
        )
        if name == "result":
            same_json(
                identity["byte_count"], reference.result_bytes, "scoring training result length"
            )
        result[name] = identity
    return result


def _verify_reference_bytes(root: Path, manifest: ConfirmationScoringManifest) -> dict[str, Any]:
    """Private IO-fixture seam; public gates require the unchanged full manifest."""
    return {
        reference.name: _bundle_bytes(root, reference) for reference in manifest.training_bundles
    }


def verify_training_reference_bytes(
    root: Path, manifest: ConfirmationScoringManifest
) -> dict[str, Any]:
    """Recheck all six bound files/markers without decoding another large body.

    Complete independent readback remains mandatory before the future child.
    Reuse this byte check before data/release and after scoring/serialization.
    """
    validate_scoring_manifest(manifest)
    return _verify_reference_bytes(root, manifest)


def _require_training_bindings(
    request: dict[str, Any],
    audit: dict[str, Any],
    manifest: ConfirmationScoringManifest,
    reference: TrainingBundleReference,
) -> None:
    expected = {
        "manifest_sha256": manifest.analysis_contract.train_manifest_sha256,
        "scope_record_sha256": manifest.scope_record_sha256,
        "adapter_sha256": manifest.training_adapter_sha256,
    }
    try:
        for body in (request, audit):
            same_json(
                {key: body[key] for key in expected},
                expected,
                "scoring training declaration binding",
            )
            same_json(
                digest_json(body["source_sha256"]),
                manifest.training_source_map_sha256,
                "scoring training source binding",
            )
        same_json(
            audit["request_sha256"], reference.request_sha256, "scoring training request binding"
        )
        same_json(
            audit["result_sha256"], reference.result_sha256, "scoring training result binding"
        )
    except (KeyError, TypeError) as error:
        raise ValueError("scoring training reference declaration binding is malformed") from error


def _read_bundle(
    root: Path,
    manifest: ConfirmationScoringManifest,
    reference: TrainingBundleReference,
    files: dict[str, Any],
    reader: CompletedTrainingReader,
) -> dict[str, Any]:
    # Why separate scope: the large decoded result is discarded on return,
    # before the next complete reader or future held-model reproduction.
    parts = reader((root / reference.directory).resolve())
    require(
        type(parts) is tuple and len(parts) == 3 and all(type(part) is dict for part in parts),
        "scoring training complete reader contract differs",
    )
    request, result, audit = parts
    for name, body in zip(_ARTIFACTS, parts, strict=True):
        same_json(
            _decoded_identity(body),
            files[name],
            f"scoring training decoded {reference.name}/{name} bytes",
        )
    _require_training_bindings(request, audit, manifest, reference)
    same_json(
        _bundle_bytes(root, reference), files, "scoring training reference changed during readback"
    )
    try:
        metadata = {
            "reference": asdict(reference),
            "files": files,
            "complete_readback": True,
            "work": json_value(audit["work"]),
            "observed_updates": json_value(audit["observed_updates"]),
            "process_rss": json_value(audit["process_rss"]),
            "worker_elapsed_seconds": audit["worker_elapsed_seconds"],
            "elapsed_seconds": audit["elapsed_seconds"],
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("scoring training complete reader metadata is malformed") from error
    return metadata


def _read_reference_bundles(
    root: Path, manifest: ConfirmationScoringManifest, reader: CompletedTrainingReader
) -> dict[str, Any]:
    files = _verify_reference_bytes(root, manifest)
    bundles = [
        _read_bundle(root, manifest, reference, files[reference.name], reader)
        for reference in manifest.training_bundles
    ]
    same_json(
        _verify_reference_bytes(root, manifest),
        files,
        "scoring training references changed after last readback",
    )
    for bundle in bundles[1:]:
        same_json(bundle["work"], bundles[0]["work"], "scoring training repeated cost facts")
        same_json(
            bundle["observed_updates"],
            bundles[0]["observed_updates"],
            "scoring training repeated observed updates",
        )
    return {"bundles": bundles}


def read_training_references(
    root: Path, manifest: ConfirmationScoringManifest, reader: CompletedTrainingReader
) -> dict[str, Any]:
    """Compose both unchanged complete readers before any new source/model.

    The source-bound adapter must supply the actual complete training reader;
    private IO spies establish no scientific validation or final authority.
    """
    validate_scoring_manifest(manifest)
    report = _read_reference_bundles(root, manifest, reader)
    return {
        "schema_id": "p67_confirmation_scoring_train_references_v1",
        "scoring_manifest_sha256": scoring_manifest_digest(manifest),
        "analysis_contract": json_value(asdict(manifest.analysis_contract)),
        "analysis_contract_sha256": analysis_contract_digest(manifest.analysis_contract),
        "training_cost_facts_reference_sha256": manifest.analysis_contract.train_result_sha256,
        **report,
        "validation_scope": "complete_train_bundle_readback_and_bound_reference_bytes_only",
        "final_release_authorized": False,
        "confirmation_source_constructed": False,
        "outer_selection_scored": False,
        "final_released": False,
    }
