"""Declare the complete scored scope without any source, final value or IO."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app.continual_confirmation_analysis_contract import (
    analysis_contract_digest,
    fixed_analysis_contract,
)
from src.app.continual_confirmation_manifest import fixed_confirmation_manifest
from src.app.continual_confirmation_scoring_manifest import (
    fixed_scoring_manifest,
    scoring_manifest_digest,
    scoring_summary,
    validate_scoring_manifest,
)
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from src.infra import continual_roles


def _forbid(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("scored declaration opened a source/model/score/final/RNG/file")


def test_should_bind_both_complete_train_bundles_and_full_analysis_contract() -> None:
    manifest = fixed_scoring_manifest()
    validate_scoring_manifest(manifest)
    assert manifest.train_manifest == fixed_confirmation_manifest()
    assert manifest.analysis_contract == fixed_analysis_contract()
    assert analysis_contract_digest(manifest.analysis_contract) == manifest.analysis_contract_sha256
    refs = manifest.training_bundles
    assert tuple(item.directory for item in refs) == (
        "artifacts/runs/p67-confirmation-train",
        "artifacts/runs/p67-confirmation-train-repeat",
    )
    assert tuple(item.request_sha256 for item in refs) == (
        "672ec760c83ce7c423058fecd0bf535ff72734bb4bf9cf1fc4a43891c3edbdc3",
        "2023443417597607c9a16f4cba4f5ad5fb2777ccad54d5af0783059066944456",
    )
    assert tuple(item.audit_sha256 for item in refs) == (
        "ff308e7a4b72e7e5ffa4c182223825d4b6e5f3f5768a7c7abf81c26cade9a0e0",
        "d1f461e1c21e6d0721102ba13349f6156d625cfa761399e09a0409e83047d01a",
    )
    assert all(
        item.result_sha256 == manifest.analysis_contract.train_result_sha256 for item in refs
    )
    assert all(item.result_bytes == 134554378 for item in refs)
    assert manifest.training_source_map_sha256 == (
        "b8e6ea624228c7422b61bf48fc736cd187893eedd6ce1fd49558f6b11bbd3735"
    )
    assert manifest.training_adapter_sha256 == (
        "e6bafe8bc6c7e2199a578b26ff991a45fbbf4d45085e760fd61d0f9417250445"
    )
    assert scoring_manifest_digest(manifest) == scoring_manifest_digest(fixed_scoring_manifest())


def test_should_rederive_all_original_cells_pairs_endpoints_and_unchanged_caps() -> None:
    assert scoring_summary(fixed_scoring_manifest()) == {
        "family_seed_rows": 60,
        "cells": 560,
        "pairs": 580,
        "final_role_views": 120,
        "final_calls": 1680,
        "final_examples": 67200,
        "max_optimizer_updates": 16000,
        "wall_limit_seconds": 600,
        "max_process_rss_bytes": 536870912,
        "rss_interval_seconds": 0.005,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol_id", "partial"),
        ("scope_record_sha256", "0" * 64),
        ("training_source_map_sha256", "0" * 64),
        ("training_adapter_sha256", "0" * 64),
        ("analysis_contract_sha256", "0" * 64),
        ("evaluation_endpoints", ("a_after_a", "b_after_b")),
        ("evaluation_endpoints", ["a_after_a", "a_after_b", "b_after_b"]),
        ("global_facts_match_required", False),
        ("global_live_state_before_final_required", 1),
        ("global_live_state_after_final_required", False),
        ("outer_selection_scored", True),
    ],
)
def test_should_refuse_changed_scored_policy_or_reference(field: str, value: Any) -> None:
    manifest = replace(fixed_scoring_manifest(), **{field: value})
    with pytest.raises(ValueError, match="complete scoring manifest"):
        validate_scoring_manifest(manifest)


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "order", "list", "float_bytes", "digest", "path"]
)
def test_should_refuse_incomplete_ambiguous_or_changed_training_references(mutation: str) -> None:
    manifest = fixed_scoring_manifest()
    refs: Any = manifest.training_bundles
    if mutation == "missing":
        refs = refs[:1]
    elif mutation == "duplicate":
        refs = (refs[0], refs[0])
    elif mutation == "order":
        refs = refs[::-1]
    elif mutation == "list":
        refs = list(refs)
    elif mutation == "float_bytes":
        refs = (replace(refs[0], result_bytes=float(refs[0].result_bytes)), refs[1])
    elif mutation == "digest":
        refs = (refs[0], replace(refs[1], result_sha256="0" * 64))
    else:
        refs = (refs[0], replace(refs[1], directory="other"))
    with pytest.raises(ValueError, match="complete scoring manifest"):
        validate_scoring_manifest(replace(manifest, training_bundles=refs))


@pytest.mark.parametrize(
    "mutation",
    [
        "seeds",
        "arms",
        "pairs",
        "final_count",
        "wall_float",
        "families_list",
        "seed_list",
        "retuned_alpha",
    ],
)
def test_should_refuse_partial_or_retyped_training_and_analysis_scope(mutation: str) -> None:
    manifest = fixed_scoring_manifest()
    family = manifest.train_manifest.families[-1]
    if mutation == "retuned_alpha":
        manifest = replace(
            manifest, analysis_contract=replace(manifest.analysis_contract, family_alpha=0.1)
        )
    elif mutation == "wall_float":
        bad_wall: Any = 600.0
        manifest = replace(
            manifest, train_manifest=replace(manifest.train_manifest, wall_limit_seconds=bad_wall)
        )
    elif mutation == "families_list":
        bad_families: Any = list(manifest.train_manifest.families)
        manifest = replace(
            manifest,
            train_manifest=replace(manifest.train_manifest, families=bad_families),
        )
    else:
        updates: Any = {
            "seeds": {"seeds": family.seeds[:-1]},
            "arms": {"arms": family.arms[:-1]},
            "pairs": {"contrasts": family.contrasts[:-1]},
            "final_count": {"expected_final_counts": (39, 40)},
            "seed_list": {"seeds": list(family.seeds)},
        }[mutation]
        family = replace(family, **updates)
        train = replace(
            manifest.train_manifest, families=(*manifest.train_manifest.families[:-1], family)
        )
        manifest = replace(manifest, train_manifest=train)
    with pytest.raises(ValueError, match="complete scoring manifest"):
        validate_scoring_manifest(manifest)


def test_should_declare_scope_with_source_model_score_final_rng_and_file_access_forbidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for model in (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork):
        monkeypatch.setattr(model, "__init__", _forbid)
        monkeypatch.setattr(model, "train_epoch", _forbid)
        monkeypatch.setattr(model, "predict_proba", _forbid)
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "compute_accuracy", _forbid)
    monkeypatch.setattr(arrived, "_build_phase_a_roles", _forbid)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", _forbid)
    monkeypatch.setattr(continual_roles, "release_final_test", _forbid)
    monkeypatch.setattr(np.random, "default_rng", _forbid)
    monkeypatch.setattr(Path, "read_bytes", _forbid)
    monkeypatch.setattr(Path, "read_text", _forbid)
    manifest = fixed_scoring_manifest()
    assert scoring_summary(manifest)["cells"] == 560
    assert len(scoring_manifest_digest(manifest)) == 64
    assert "a_after_b" in asdict(manifest)["evaluation_endpoints"]
