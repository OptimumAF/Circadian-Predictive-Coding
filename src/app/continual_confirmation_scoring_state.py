"""Verify complete reproduced training before and after future final scoring.

Inputs are live unscored training and the fixed scientific scoring manifest.
Outputs prove complete fact-byte and held-state equality only. File/source
provenance, final release, evaluation, resource enforcement and IO belong to
separate boundaries; a state proof does not authorize final access.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from hashlib import sha256
import json
from typing import Any

from src.app.continual_confirmation_manifest import ConfirmationManifest
from src.app.continual_confirmation_scoring_manifest import (
    ConfirmationScoringManifest,
    scoring_manifest_digest,
    validate_scoring_manifest,
)
from src.app.continual_confirmation_state import FamilySeedFacts, HeldSeed, require_held_seed
from src.app.continual_confirmation_training import ConfirmationTrainingFacts, TrainedConfirmation


@dataclass(frozen=True)
class TrainingArtifactIdentity:
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class TrainingStateProof:
    training_artifact: TrainingArtifactIdentity
    family_seed_rows: int
    cells: int
    held_checkpoints: int
    scoring_manifest_sha256: str | None = None
    validation_scope: str = "training_facts_and_live_held_state_only"
    source_provenance_verified: bool = False
    final_release_authorized: bool = False


def _shallow_dataclass(value: Any) -> dict[str, Any]:
    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError(f"unsupported training artifact value: {type(value).__name__}")
    return {field.name: getattr(value, field.name) for field in fields(value)}


def training_artifact_identity(facts: ConfirmationTrainingFacts) -> TrainingArtifactIdentity:
    """Match the original complete writer bytes without a deepcopy/string join."""
    if type(facts) is not ConfirmationTrainingFacts:
        raise ValueError("confirmation training artifact encoding requires training facts")
    encoder = json.JSONEncoder(
        indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False, default=_shallow_dataclass
    )
    digest = sha256()
    byte_count = 0
    try:
        for fragment in encoder.iterencode(facts):
            encoded = fragment.encode("utf-8")
            digest.update(encoded)
            byte_count += len(encoded)
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError(
            "confirmation training artifact encoding is malformed or nonfinite"
        ) from error
    digest.update(b"\n")
    return TrainingArtifactIdentity(digest.hexdigest(), byte_count + 1)


def _require_inventory(trained: TrainedConfirmation, manifest: ConfirmationManifest) -> None:
    if (
        type(trained) is not TrainedConfirmation
        or type(trained.facts) is not ConfirmationTrainingFacts
        or type(trained.facts.seed_results) is not tuple
        or type(trained.held) is not tuple
        or trained.facts.manifest != manifest
    ):
        raise ValueError("confirmation reproduced training inventory differs")
    expected = tuple((family.name, seed) for family in manifest.families for seed in family.seeds)
    rows = trained.facts.seed_results
    if (
        len(rows) != len(expected)
        or len(trained.held) != len(expected)
        or any(type(row) is not FamilySeedFacts or type(row.seed) is not int for row in rows)
        or tuple((row.family, row.seed) for row in rows) != expected
        or any(type(item) is not HeldSeed for item in trained.held)
    ):
        raise ValueError("confirmation reproduced training inventory differs")
    for row, held in zip(rows, trained.held, strict=True):
        # The unchanged producer shares this exact row with its held object.
        # Detached equal facts could later diverge from the global fingerprint.
        if held.facts is not row:
            raise ValueError("confirmation reproduced held inventory/fact attachment differs")


def _require_artifact(facts: ConfirmationTrainingFacts, expected: TrainingArtifactIdentity) -> None:
    if training_artifact_identity(facts) != expected:
        raise ValueError(
            "confirmation reproduced training artifact differs from bound complete result"
        )


def _verify_reproduced_state(
    trained: TrainedConfirmation,
    manifest: ConfirmationManifest,
    expected: TrainingArtifactIdentity,
) -> TrainingStateProof:
    """Private development seam; public verification never takes fixture digests."""
    _require_inventory(trained, manifest)
    _require_artifact(trained.facts, expected)
    for item in trained.held:
        require_held_seed(item)
    # No verifier should mutate facts. Detect drift even in the last live check.
    _require_inventory(trained, manifest)
    _require_artifact(trained.facts, expected)
    cells = sum(len(family.seeds) * len(family.arms) for family in manifest.families)
    return TrainingStateProof(expected, len(trained.held), cells, 2 * cells)


def verify_scoring_training_state(
    trained: TrainedConfirmation, manifest: ConfirmationScoringManifest
) -> TrainingStateProof:
    """Require the complete bound production facts and every live held state.

    The outer boundary must independently verify both referenced train bundles
    and all current sources/requests/resources. Keep original roles sealed so
    this same global check remains valid after separately released final views.
    """
    validate_scoring_manifest(manifest)
    reference = manifest.training_bundles[0]
    expected = TrainingArtifactIdentity(reference.result_sha256, reference.result_bytes)
    proof = _verify_reproduced_state(trained, manifest.train_manifest, expected)
    return replace(proof, scoring_manifest_sha256=scoring_manifest_digest(manifest))
