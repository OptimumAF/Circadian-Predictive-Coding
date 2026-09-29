"""Sealed-selection and manifest checks for repeated matched-head confirmation."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.app import matched_head_tuning as tuning  # noqa: E402
from src.app import repeated_head_confirmation as repeated  # noqa: E402
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig  # noqa: E402


def _base_config() -> ResNet50BenchmarkConfig:
    return ResNet50BenchmarkConfig(
        train_samples=4,
        guard_samples=4,
        validation_samples=4,
        test_samples=4,
        num_classes=3,
        image_size=32,
        batch_size=4,
        epochs=1,
        seed=47,
        device="cpu",
        target_accuracy=None,
        backprop_freeze_backbone=True,
        backbone_weights="none",
        predictive_head_hidden_dim=8,
        circadian_head_hidden_dim=8,
        circadian_min_hidden_dim=8,
        circadian_max_hidden_dim=8,
        predictive_inference_steps=1,
        circadian_inference_steps=1,
        circadian_sleep_interval=1,
        circadian_force_sleep=True,
        circadian_sleep_warmup_steps=0,
        circadian_use_adaptive_sleep_trigger=False,
    )


def _selection(monkeypatch: pytest.MonkeyPatch) -> tuning.MatchedHeadTuningResult:
    features = torch.tensor(
        [[0.2, 0.1, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)
    role_loader = [(features, labels)]

    class Loaders:
        train_loader = role_loader
        guard_loader = list(role_loader)
        validation_loader = list(role_loader)
        num_classes = 3
        split_hashes = {role: role for role in ("train", "guard", "validation", "test")}

        @property
        def test_loader(self) -> Any:
            pytest.fail("Validation selection opened final test")

    monkeypatch.setattr(tuning.matched, "_build_benchmark_loaders", lambda config: Loaders())
    monkeypatch.setattr(
        tuning.matched,
        "_build_resnet50_backbone",
        lambda **kwargs: (torch.nn.Identity(), 5),
    )
    base = _base_config()
    candidates: dict[str, tuple[tuning.HeadTuningCandidate, ...]] = {
        head: (
            tuning.HeadTuningCandidate("a", base),
            tuning.HeadTuningCandidate(
                "b",
                replace(base, **{field: getattr(base, field) * 0.8}),
            ),
        )
        for head, field in (
            ("backprop_mlp", "backprop_learning_rate"),
            ("predictive_coding", "predictive_learning_rate"),
            ("circadian_predictive_coding", "circadian_learning_rate"),
        )
    }
    return tuning.run_matched_head_tuning(
        base, candidates, seeds=(47,), candidates_per_head=2, confirm_test=False
    )


def test_manifest_freezes_validation_selection_without_final_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection(monkeypatch)
    manifest = repeated.create_confirmation_manifest(
        selection,
        confirmation_seeds=(53, 59, 61),
        wall_time_budget_seconds=0.05,
        wall_time_epoch_cap=1000,
    )

    assert selection.confirmations == ()
    assert manifest.selection_seeds == (47,)
    assert manifest.confirmation_seeds == (53, 59, 61)
    assert manifest.scopes == repeated.CONFIRMATION_SCOPES
    assert manifest.metric_names == repeated.METRIC_NAMES
    assert {head.head_name for head in manifest.selected_heads} == set(tuning.HEAD_NAMES)
    repeated._validate_manifest(manifest)
    with pytest.raises(ValueError, match="changed after freezing"):
        repeated._validate_manifest(replace(manifest, confirmation_seeds=(53, 59, 67)))


def test_saved_manifest_restores_and_rejects_changed_seed_before_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection(monkeypatch)
    manifest = repeated.create_confirmation_manifest(
        selection,
        confirmation_seeds=(53, 59, 61),
        wall_time_budget_seconds=0.05,
        wall_time_epoch_cap=1000,
    )
    saved = json.loads(json.dumps(asdict(manifest)))
    assert repeated.restore_confirmation_manifest(saved) == manifest
    changed = {**saved, "confirmation_seeds": [53, 59, 67]}
    with pytest.raises(ValueError, match="changed after freezing"):
        repeated.restore_confirmation_manifest(changed)
    incomplete = {**saved, "base_config": {"seed": 47}}
    with pytest.raises(ValueError, match="Malformed"):
        repeated.restore_confirmation_manifest(incomplete)


def test_manifest_rejects_test_informed_or_overlap_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection(monkeypatch)
    with pytest.raises(ValueError, match="validation-only"):
        repeated.create_confirmation_manifest(
            replace(selection, protocol_id=tuning.MATCHED_HEAD_TUNING_PROTOCOL),
            confirmation_seeds=(53, 59, 61),
            wall_time_budget_seconds=0.05,
            wall_time_epoch_cap=1000,
        )
    with pytest.raises(ValueError, match="disjoint"):
        repeated.create_confirmation_manifest(
            selection,
            confirmation_seeds=(47, 59, 61),
            wall_time_budget_seconds=0.05,
            wall_time_epoch_cap=1000,
        )


def test_summary_requires_every_declared_seed() -> None:
    scores = {(head, seed): 0.5 for head in tuning.HEAD_NAMES for seed in (53, 59, 61)}
    summaries = repeated._accuracy_summaries((53, 59, 61), scores)
    assert all(summary.seeds == (53, 59, 61) for summary in summaries.values())
    scores.pop(("circadian_predictive_coding", 59))
    with pytest.raises(AssertionError, match="every declared head and seed"):
        repeated._accuracy_summaries((53, 59, 61), scores)


def test_fixed_data_opens_test_after_all_declared_training_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = repeated.create_confirmation_manifest(
        _selection(monkeypatch),
        confirmation_seeds=(53, 59, 61),
        wall_time_budget_seconds=0.05,
        wall_time_epoch_cap=1000,
    )
    role_loader = [
        (
            torch.tensor([[0.2, 0.1, 0.3, 0.4, 0.5]], dtype=torch.float32),
            torch.tensor([1], dtype=torch.long),
        )
    ]
    state = {"trained": 0, "test_reads": 0}

    class Loaders:
        train_loader = role_loader
        guard_loader = list(role_loader)
        validation_loader = list(role_loader)
        num_classes = 3
        split_hashes = {role: role for role in ("train", "guard", "validation", "test")}

        @property
        def test_loader(self) -> Any:
            assert state["trained"] == 9
            state["test_reads"] += 1
            return role_loader

    train = tuning._run_candidate_trial

    def counted_train(*args: Any) -> Any:
        outcome = train(*args)
        state["trained"] += 1
        return outcome

    monkeypatch.setattr(tuning.matched, "_build_benchmark_loaders", lambda config: Loaders())
    monkeypatch.setattr(tuning, "_run_candidate_trial", counted_train)
    result = tuning.run_matched_head_tuning(
        manifest.base_config,
        repeated._candidate_map(manifest.selected_heads),
        seeds=manifest.confirmation_seeds,
        candidates_per_head=1,
    )

    repeated._verify_fixed_data(manifest, result)
    assert state == {"trained": 9, "test_reads": 3}
    assert len(result.attempts) == len(result.trials) == len(result.confirmations) == 9
    with pytest.raises(AssertionError, match="missing or duplicate final tests"):
        repeated._verify_fixed_data(
            manifest, replace(result, confirmations=result.confirmations[:-1])
        )
