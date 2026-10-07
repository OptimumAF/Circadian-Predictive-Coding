"""Inspect every reserved family and additive budget without any training data."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any

import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_confirmation_manifest as scope
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork


def test_should_freeze_all_factors_reservations_and_exact_work_without_sources_or_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def prohibited(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("scope inspection constructed sources/models or scored")

    monkeypatch.setattr(arrived, "_build_phase_a_roles", prohibited)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", prohibited)
    for model_type in (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork):
        monkeypatch.setattr(model_type, "__init__", prohibited)
        monkeypatch.setattr(model_type, "compute_accuracy", prohibited)
    manifest = scope.fixed_confirmation_manifest()
    assert scope.validate_confirmation_manifest(manifest) == {
        "family_count": 6,
        "family_seed_instances": 60,
        "distinct_confirmation_seeds": 50,
        "cell_count": 560,
        "wake_updates": 13440,
        "maximum_optimizer_updates": 15620,
        "maximum_guarded_attempts": 920,
        "final_evaluations": 1680,
        "final_examples": 67200,
        "contrast_count": 580,
    }
    assert manifest.families[0].seeds == manifest.families[1].seeds
    assert all(len(family.seeds) == 10 for family in manifest.families)
    assert [len(family.contrasts) for family in manifest.families] == [1, 3, 3, 9, 22, 20]
    assert [family.maximum_optimizer_updates for family in manifest.families] == [
        720,
        2280,
        2160,
        3380,
        5160,
        1920,
    ]
    for family in manifest.families:
        original = json.loads(family.development_manifest_json)
        assert original["confirmation_seeds"] == list(family.seeds)
        assert original["seeds"] == list(family.development_seeds)
        assert family.expected_final_counts == (40, 40)
    assert manifest.max_process_rss_bytes == 512 * 1024 * 1024
    assert manifest.wall_limit_seconds == 600
    assert (
        manifest.global_train_gate_required
        and manifest.uncertainty_contract_required_before_scoring
    )


@pytest.mark.parametrize(
    "change", ["seed", "cell", "contrast", "updates", "final_count", "reference"]
)
def test_should_reject_changed_or_partial_family_scope(change: str) -> None:
    manifest = scope.fixed_confirmation_manifest()
    family = manifest.families[-1]
    if change == "seed":
        family = replace(family, seeds=family.seeds[:-1])
    elif change == "cell":
        family = replace(family, arms=family.arms[:-1])
    elif change == "contrast":
        family = replace(family, contrasts=family.contrasts[:-1])
    elif change == "updates":
        family = replace(family, maximum_optimizer_updates=0)
    elif change == "final_count":
        family = replace(family, expected_final_counts=(24, 12))
    else:
        family = replace(family, development_result_sha256="0" * 64)
    with pytest.raises(ValueError, match="frozen complete scope"):
        scope.validate_confirmation_manifest(
            replace(manifest, families=manifest.families[:-1] + (family,))
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_optimizer_updates", 20000),
        ("wall_limit_seconds", 1200),
        ("max_process_rss_bytes", 1024 * 1024 * 1024),
        ("global_train_gate_required", False),
        ("score_role", "outer_selection"),
        ("uncertainty_contract_required_before_scoring", False),
    ],
)
def test_should_reject_changed_budget_or_evaluation_boundary(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match="frozen complete scope"):
        scope.validate_confirmation_manifest(
            replace(scope.fixed_confirmation_manifest(), **{field: value})
        )


def test_should_reject_source_settings_drift_even_when_its_factory_validates_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = scope.gating.fixed_gating_pilot_manifest
    monkeypatch.setattr(
        scope.gating,
        "fixed_gating_pilot_manifest",
        lambda: replace(original(), max_planned_wake_updates=300),
    )
    with pytest.raises(ValueError, match="source manifest"):
        scope.fixed_confirmation_manifest()
