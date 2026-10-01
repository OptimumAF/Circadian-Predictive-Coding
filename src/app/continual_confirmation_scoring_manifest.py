"""Bind independent scoring to complete training and analysis declarations.

Inputs are unchanged frozen factories and recorded artifact identities.
Outputs describe the full scientific scope; no file/source/model, final
release, resource measurement or score is accessed or authorized here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from hashlib import sha256
import json
from typing import Any

from src.app.continual_confirmation_analysis_contract import (
    ConfirmationAnalysisContract,
    TRAIN_RESULT_SHA256,
    analysis_contract_digest,
    fixed_analysis_contract,
)
from src.app.continual_confirmation_manifest import (
    ConfirmationManifest,
    fixed_confirmation_manifest,
)


PROTOCOL_ID = "continual_mechanism_confirmation_final_scoring_v1"
ANALYSIS_CONTRACT_SHA256 = "5e33ef28862bcdf9d92fe14dd6cf6b71672a2336ffd760a1214ef04666b594b1"


@dataclass(frozen=True)
class TrainingBundleReference:
    name: str
    directory: str
    request_sha256: str
    audit_sha256: str
    result_sha256: str = TRAIN_RESULT_SHA256
    result_bytes: int = 134554378


@dataclass(frozen=True)
class ConfirmationScoringManifest:
    train_manifest: ConfirmationManifest
    analysis_contract: ConfirmationAnalysisContract
    training_bundles: tuple[TrainingBundleReference, ...]
    protocol_id: str = PROTOCOL_ID
    scope_record_sha256: str = "622feead54f155521928341151c8496c23c02a54b355b1b3b1e0f68b76c5772f"
    training_source_map_sha256: str = (
        "b8e6ea624228c7422b61bf48fc736cd187893eedd6ce1fd49558f6b11bbd3735"
    )
    training_adapter_sha256: str = (
        "e6bafe8bc6c7e2199a578b26ff991a45fbbf4d45085e760fd61d0f9417250445"
    )
    analysis_contract_sha256: str = ANALYSIS_CONTRACT_SHA256
    evaluation_endpoints: tuple[str, ...] = ("a_after_a", "a_after_b", "b_after_b")
    global_facts_match_required: bool = True
    global_live_state_before_final_required: bool = True
    global_live_state_after_final_required: bool = True
    outer_selection_scored: bool = False


def fixed_scoring_manifest() -> ConfirmationScoringManifest:
    """Retain both successful complete runs and every original planned cell."""
    analysis = fixed_analysis_contract()
    if analysis_contract_digest(analysis) != ANALYSIS_CONTRACT_SHA256:
        raise ValueError("confirmation frozen analysis declaration changed")
    references = (
        TrainingBundleReference(
            "canonical",
            "artifacts/runs/p67-confirmation-train",
            "672ec760c83ce7c423058fecd0bf535ff72734bb4bf9cf1fc4a43891c3edbdc3",
            "ff308e7a4b72e7e5ffa4c182223825d4b6e5f3f5768a7c7abf81c26cade9a0e0",
        ),
        TrainingBundleReference(
            "repeat",
            "artifacts/runs/p67-confirmation-train-repeat",
            "2023443417597607c9a16f4cba4f5ad5fb2777ccad54d5af0783059066944456",
            "d1f461e1c21e6d0721102ba13349f6156d625cfa761399e09a0409e83047d01a",
        ),
    )
    return ConfirmationScoringManifest(fixed_confirmation_manifest(), analysis, references)


def _require_same_shape(actual: Any, expected: Any) -> None:
    # JSON alone would silently identify mutable lists with declared tuples.
    if type(actual) is not type(expected):
        raise ValueError("requires its frozen complete scoring manifest")
    if is_dataclass(expected) and not isinstance(expected, type):
        for field in fields(expected):
            _require_same_shape(getattr(actual, field.name), getattr(expected, field.name))
    elif type(expected) is tuple:
        if len(actual) != len(expected):
            raise ValueError("requires its frozen complete scoring manifest")
        for left, right in zip(actual, expected, strict=True):
            _require_same_shape(left, right)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_scoring_manifest(manifest: ConfirmationScoringManifest) -> None:
    """Reject changes, partial scope and ambiguous nested type identities."""
    expected = fixed_scoring_manifest()
    _require_same_shape(manifest, expected)
    try:
        equal = _canonical_json(asdict(manifest)) == _canonical_json(asdict(expected))
    except (ValueError, TypeError) as error:
        raise ValueError("requires its frozen complete scoring manifest") from error
    if not equal:
        raise ValueError("requires its frozen complete scoring manifest")


def scoring_manifest_digest(manifest: ConfirmationScoringManifest) -> str:
    validate_scoring_manifest(manifest)
    return sha256(_canonical_json(asdict(manifest)).encode("utf-8")).hexdigest()


def scoring_summary(manifest: ConfirmationScoringManifest) -> dict[str, int | float]:
    validate_scoring_manifest(manifest)
    families = manifest.train_manifest.families
    cells = sum(len(f.seeds) * len(f.arms) for f in families)
    return {
        "family_seed_rows": sum(len(f.seeds) for f in families),
        "cells": cells,
        "pairs": sum(len(f.seeds) * len(f.contrasts) for f in families),
        "final_role_views": 2 * sum(len(f.seeds) for f in families),
        "final_calls": len(manifest.evaluation_endpoints) * cells,
        "final_examples": sum(
            len(f.seeds)
            * len(f.arms)
            * (2 * f.expected_final_counts[0] + f.expected_final_counts[1])
            for f in families
        ),
        "max_optimizer_updates": manifest.train_manifest.max_optimizer_updates,
        "wall_limit_seconds": manifest.train_manifest.wall_limit_seconds,
        "max_process_rss_bytes": manifest.train_manifest.max_process_rss_bytes,
        "rss_interval_seconds": manifest.train_manifest.rss_interval_seconds,
    }
